#from dataclasses import dataclass
import numpy as np
from numba import njit, prange, get_thread_id, get_num_threads
from levy_flights import estimate_mean_abs_levy, levy_step, levy_bit_flip,levy_jump_Q
from Swarm_IQ import decode_prod_plan, compute_objective


# --------------------------------------------------------------------------------------------------------------------------------------- #
class BatAlgorithm:
    
    def __init__(self, n_particles,config, f_min=0.0, f_max=2.0, alpha=0.9,
        gamma=0.005, r0=0.2, Vzmax=4.0, Vqmax=0.1, bit_flip_share = 0.1, 
        levy_alpha = 1.5, levy_samples = 100000):


        
        self.cfg = config()
        self.N, self.M, self.T = n_particles, self.cfg.M, self.cfg.T

        self.Vzmax, self.Vqmax = Vzmax, Vqmax

        # initialize Qs
        self.Qs = np.random.rand(self.N, self.T, self.M)
        
        
        # initialize Xs 
        self.Xs = np.random.randint(0, 2, size=(self.N, self.T, self.M)).astype(np.float64)


        # Velocities (N × T × M)
        self.VZs = (np.random.rand(self.N, self.T, self.M) * 2 - 1) * Vzmax
        self.VQs = (np.random.rand(self.N, self.T, self.M) * 2 - 1) * Vqmax

        # Personal bests (N × T × M and N × 1 × 2)
        self.pbest_X = self.Xs.copy()
        self.pbest_Q = self.Qs.copy()
        self.pbest_Val = np.full((self.N, 1, 2), np.inf)

        # Global best (T × M and 1 × 2)
        self.gbest_X = np.zeros((self.T, self.M), dtype=np.float64)
        self.gbest_Q = np.zeros((self.T, self.M), dtype=np.float64)
        self.gbest_Val = np.full((1, 2), np.inf)

        # random walk parameters
        mean_abs_L = estimate_mean_abs_levy(levy_alpha, levy_samples)
        self.bit_flip_share = (bit_flip_share ) / mean_abs_L
        self.levy_alpha = levy_alpha

        # Bat-specific hyperparameters
        self.f_min, self.f_max = f_min, f_max
        self.alpha, self.gamma = alpha, gamma

        # Pulse-rate schedule: rᵢ starts at 0, will linearly ramp up to r0 by max_iter
        self.r0 = r0
        self.rs = np.zeros(self.N, dtype=np.float64)  # overrides decay schedule

        # Loudness per bat (N,)
        #self.As = np.random.rand(self.N)
        self.As = np.random.uniform(1.0, 2.0, self.N)
        # Iteration counters
        self.t = 0
        self.stagnation = 0

        # Cold-start: run one “iteration” with zeroed randoms so that gbest fill 
        # this is necessary before running an actual iteration (gbest is needed for updating)
        self._reduce_global(
            update_swarm_bat(
                self.Xs, self.Qs, self.VZs, self.VQs,
                self.pbest_X, self.pbest_Q, self.pbest_Val,
                self.gbest_X, self.gbest_Q,
                self.As, self.rs,
                self.f_min, self.f_max, self.alpha, self.gamma,
                self.r0, self.t, 0,      # t_global=0, max_iter=0 for cold-start
                self.Vzmax, self.Vqmax,
                self.cfg.demand, self.cfg.production_times,
                self.cfg.setup_times, self.cfg.capacities,
                self.cfg.production_costs, self.cfg.setup_costs,
                self.cfg.inventory_costs,
                np.zeros(self.N), np.zeros(self.N), np.zeros((self.N, self.T, self.M)),
                np.zeros((self.N, self.T, self.M)) , 0.0, 0.0
            )
        )



    def _reduce_global(self, thread_results):
        # reduces all the thread-local minima into a single true global minimum
        values, indices = thread_results
        for t in range(values.shape[0]):
            i = indices[t]
            v0 = values[t, 0, 0]
            v1 = values[t, 0, 1]
            b0 = self.gbest_Val[0, 0]
            b1 = self.gbest_Val[0, 1]
            if (v0 < b0) or (v0 == b0 and v1 < b1):
                self.gbest_Val[0, 0] = v0
                self.gbest_Val[0, 1] = v1
                self.gbest_X[:] = self.pbest_X[i]
                self.gbest_Q[:] = self.pbest_Q[i]
                
    def optimize(self, n_iter):
        N, T, M = self.Xs.shape
        self.max_iter = n_iter   

        for i in range(n_iter):
            
            # Increment global iteration counter
            self.t += 1

            # Pre‐generate all random arrays
            rand_f      = np.random.rand(N)         # frequency ∈ [0,1]
            rand_local  = np.random.rand(N)         # local‐walk decision ∈ [0,1]
            rand_eps   = np.random.rand(N, T, M)    # ε for local walk ∈ [0,1]
            rand_vals   = np.random.rand(N, T, M)   # for X‐threshold ∈ [0,1]

            # Call the updated swarm‐update with linear‐ramp signature
            thread_results = update_swarm_bat(
                self.Xs, self.Qs, self.VZs, self.VQs,
                self.pbest_X, self.pbest_Q, self.pbest_Val,
                self.gbest_X, self.gbest_Q,
                self.As, self.rs,
                self.f_min, self.f_max, self.alpha, self.gamma,
                self.r0, self.t, self.max_iter,       
                self.Vzmax, self.Vqmax,
                self.cfg.demand, self.cfg.production_times,
                self.cfg.setup_times, self.cfg.capacities,
                self.cfg.production_costs, self.cfg.setup_costs,
                self.cfg.inventory_costs,
                rand_f, rand_local, rand_eps, rand_vals, self.bit_flip_share, self.levy_alpha
            )

            # Reduce thread‐local bests into a single gbest
            self._reduce_global(thread_results)
         # At the end, return the best plan and its value
        return decode_prod_plan(self.gbest_X.T, self.gbest_Q.T, self.cfg.demand), self.gbest_Val



        
# --------------------------------------------------------------------------------------------------------------------------------------- #

@njit
def iterate_bat(
    X, Q, VZPK, VQPK, pbest_X, pbest_Q, pbest_value,
    gbest_X, gbest_Q, A_i, r_i, f_min, f_max, alpha, gamma,
    r0, t_global, max_iter, Vzmax, Vqmax,demand, production_times, 
    setup_times, capacities, production_costs, setup_costs, inventory_costs,
    rand_f, rand_local, rand_eps, rand_vals, particle_idx, A_avg, bit_flip_share, levy_alpha
):

    M, T = X.shape

    # 1) Frequency update
    f = f_min + (f_max - f_min) * rand_f

    # 2) Velocity updates
    for i in range(M):
        for j in range(T):
            # binary component velocity (pre-sigmoid)
            VZPK[i, j] += f * (gbest_X[i, j] - X[i, j])
            if VZPK[i, j] > Vzmax:
                VZPK[i, j] = Vzmax
            elif VZPK[i, j] < -Vzmax:
                VZPK[i, j] = -Vzmax

            # continuous component velocity
            VQPK[i, j] += f * (gbest_Q[i, j] - Q[i, j])
            if VQPK[i, j] > Vqmax:
                VQPK[i, j] = Vqmax
            elif VQPK[i, j] < -Vqmax:
                VQPK[i, j] = -Vqmax

    # 3) Position updates
    for i in range(M):
        for j in range(T):
            # X‐bit (sigmoid threshold)
            if 1.0 / (1.0 + np.exp(-VZPK[i, j])) > rand_vals[i, j]:
                X[i, j] = 1.0
            else:
                X[i, j] = 0.0

            # Q component
            Q[i, j] = Q[i, j] + VQPK[i, j]
            if Q[i, j] > 1.0:
                Q[i, j] = 1.0
            elif Q[i, j] < 0.0:
                Q[i, j] = 0.0


    
    # 5) Local random walk if rand_local < r_i
    if rand_local < r_i:

    
        for i in range(M):
            for j in range(T):
                eps = 2.0 * rand_eps[i,j] - 1.0
                Q[i, j] = Q[i, j] + eps * A_avg
                
                if Q[i, j] < 0.0:
                    Q[i, j] = 0.0
                elif Q[i, j] > 1.0:
                    Q[i, j] = 1.0

        X[:] = levy_bit_flip(X, bit_flip_share, levy_alpha)

    # 6) Decode & evaluate fitness
    prod_quant = decode_prod_plan(X.T, Q.T, demand)  # T×M
    X_float = X.astype(np.float64)
    fitness = compute_objective(
        prod_quant, X_float,
        setup_costs, production_costs,
        production_times, setup_times,
        capacities, inventory_costs, demand
    )

    # 7) Update personal best
    fv0 = fitness[0]
    fv1 = fitness[1]
    bv0 = pbest_value[0, 0]
    bv1 = pbest_value[0, 1]

    if (fv0 < bv0) or (fv0 == bv0 and fv1 < bv1):
        
        # update pulse rate and loudness if bat improved
        A_i = A_i * alpha
        r_i = r0 * (1 - np.exp(-gamma * t_global))
        
        # new personal best
        for ii in range(M):
            for jj in range(T):
                pbest_X[ii, jj] = X[ii, jj]
                pbest_Q[ii, jj] = Q[ii, jj]
        pbest_value[0, 0] = fv0
        pbest_value[0, 1] = fv1


    return (
        X, Q, VZPK, VQPK,
        pbest_X, pbest_Q, pbest_value,
        A_i, r_i
    )

# --------------------------------------------------------------------------------------------------------------------------------------- #

@njit(parallel=True)
def update_swarm_bat(
    Xs, Qs, VZs, VQs, pbest_X, pbest_Q, pbest_Val,
    gbest_X, gbest_Q, As, rs, f_min, f_max, alpha,
    gamma, r0, t_global, max_iter, Vzmax, Vqmax,
    demand, production_times, setup_times, capacities,
    production_costs, setup_costs, inventory_costs,
    rand_f, rand_local, rand_eps, rand_vals, bit_flip_share, levy_alpha
):
    """
    Parallel update of all N bats; returns per-thread bests for gbest reduction.
    """
    K = As.shape[0]

    # 1) Compute A_avg manually (fastest under Numba)
    A_sum = 0.0
    for k in range(K):
        A_sum += As[k]
    A_avg = A_sum / K

    nthreads = get_num_threads()
    local_best = np.full((nthreads, 1, 2), np.inf)
    local_idx  = np.zeros(nthreads, np.int64)
    N = Xs.shape[0]

    for i in prange(N):
        # Directly call iterate_bat on the indexed arrays
        Xs[i], Qs[i], VZs[i], VQs[i], \
        pbest_X[i], pbest_Q[i], pbest_Val[i], \
        As[i], rs[i] = iterate_bat(
            Xs[i], Qs[i], VZs[i], VQs[i],
            pbest_X[i], pbest_Q[i], pbest_Val[i],
            gbest_X, gbest_Q,
            As[i], rs[i],
            f_min, f_max, alpha, gamma,
            r0, t_global, max_iter,    # ← pass them here
            Vzmax, Vqmax,
            demand, production_times, setup_times, capacities,
            production_costs, setup_costs, inventory_costs,
            rand_f[i], rand_local[i], rand_eps[i], rand_vals[i], i,
            A_avg, bit_flip_share, levy_alpha
        )

        # Thread-local best tracking
        t_id = get_thread_id()
        pv0 = pbest_Val[i][0, 0]
        pv1 = pbest_Val[i][0, 1]
        lb0 = local_best[t_id, 0, 0]
        lb1 = local_best[t_id, 0, 1]
        if (pv0 < lb0) or (pv0 == lb0 and pv1 < lb1):
            local_best[t_id, 0, 0] = pv0
            local_best[t_id, 0, 1] = pv1
            local_idx[t_id]       = i

    return local_best, local_idx

# --------------------------------------------------------------------------------------------------------------------------------------- #