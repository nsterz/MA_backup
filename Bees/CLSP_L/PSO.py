from numba import njit,prange, get_thread_id, get_num_threads
import numpy as np
from Swarm_IQ_L import decode_prod_plan, decode_and_evaluate
from helper_functions import rounded

# ------------------------------------------------------------------------------------------------------------------------------------- #
@njit
def iterate(X, Q, O, VZPK, VQPK, VOPK, 
            pbest_X, pbest_Q, pbest_O, pbest_value,
            gbest_X, gbest_Q, gbest_O,
            inertia, C1, C2, Vzmax, Vqmax, Vomax,
            demand, production_times, setup_times, capacities,
            production_costs, setup_costs, inventory_costs, 
            r1, r2, r3, r4, r5, r6, rand_vals1, rand_vals2):
    M, T = X.shape
    # update VZPK
    for i in range(M):
        for j in range(T):
            VZPK[i, j] = inertia * VZPK[i, j] \
                         + C1 * r1 * (pbest_X[i, j] - X[i, j]) \
                         + C2 * r2 * (gbest_X[i, j] - X[i, j])
            if VZPK[i, j] > Vzmax:
                VZPK[i, j] = Vzmax
            elif VZPK[i, j] < -Vzmax:
                VZPK[i, j] = -Vzmax

    # update VQPK
    for i in range(M):
        for j in range(T):
            VQPK[i, j] = inertia * VQPK[i, j] \
                         + C1 * r3 * (pbest_Q[i, j] - Q[i, j]) \
                         + C2 * r4 * (gbest_Q[i, j] - Q[i, j])
            if VQPK[i, j] > Vqmax:
                VQPK[i, j] = Vqmax
            elif VQPK[i, j] < -Vqmax:
                VQPK[i, j] = -Vqmax

    
    # update VOPK
    for i in range(M):
        for j in range(T):
            VOPK[i, j] = inertia * VOPK[i, j] \
                         + C1 * r5 * (pbest_O[i, j] - O[i, j]) \
                         + C2 * r6 * (gbest_O[i, j] - O[i, j])
            if VOPK[i, j] > Vomax:
                VOPK[i, j] = Vomax
            elif VOPK[i, j] < -Vomax:
                VOPK[i, j] = -Vomax

    # update X
    for i in range(M):
        for j in range(T):
            if 1.0 / (1.0 + np.exp(-VZPK[i, j])) > rand_vals1[i, j]:
                X[i, j] = 1.0
            else:
                X[i, j] = 0.0
    # update O
    for i in range(M):
        for j in range(T):
            if 1.0 / (1.0 + np.exp(-VOPK[i, j])) > rand_vals2[i, j]:
                O[i, j] = 1.0
            else:
                O[i, j] = 0.0

    # update Q
    for i in range(M):
        for j in range(T):
            Q[i, j] = Q[i, j] + VQPK[i, j]
            if Q[i, j] > 1.0:
                Q[i, j] = 1.0
            elif Q[i, j] < 0.0:
                Q[i, j] = 0.0
 

    fitness = decode_and_evaluate(
                X, Q, O, demand,
                setup_costs, production_costs,
                production_times, setup_times,
                capacities, inventory_costs)

    # unpack scalar fitness and pbest
    fv0 = fitness[0]
    fv1 = fitness[1]
    bv0 = pbest_value[0, 0]
    bv1 = pbest_value[0, 1]

    # update personal best
    if (fv0 < bv0) or (fv0 == bv0 and fv1 < bv1):
        for ii in range(M):
            for jj in range(T):
                pbest_X[ii, jj] = X[ii, jj]
                pbest_Q[ii, jj] = Q[ii, jj]
                pbest_O[ii, jj] = O[ii, jj]
        pbest_value[0, 0] = fv0
        pbest_value[0, 1] = fv1


    return X, Q, O, VZPK, VQPK, VOPK, pbest_X, pbest_Q, pbest_O, pbest_value

# ------------------------------------------------------------------------------------------------------------------------------------- #
@njit(parallel=True)
def update_swarm(Xs, Qs, Os, VXs, VQs, VOs, pbest_X, pbest_Q, pbest_O, pbest_Val,
                 gbest_X, gbest_Q, gbest_O, inertia, C1, C2, Vzmax, Vqmax, Vomax,
                 demand, production_times, setup_times, capacities,
                 production_costs, setup_costs, inventory_costs,
                 r1_arr, r2_arr, r3_arr, r4_arr, r5_arr, r6_arr, rand_vals1,rand_vals2):

    nthreads = get_num_threads()
    local_best = np.full((nthreads, 1, 2), np.inf)
    local_idx  = np.zeros(nthreads, np.int64)
    N = Xs.shape[0]

    for i in prange(N):
        Xi, Qi, Oi, VXi, VQi, VOi, pbest_Xi, pbest_Qi, pbest_Oi, pbest_Values = iterate(
            Xs[i], Qs[i], Os[i], VXs[i], VQs[i], VOs[i],
            pbest_X[i], pbest_Q[i], pbest_O[i], pbest_Val[i],
            gbest_X, gbest_Q, gbest_O,
            inertia, C1, C2, Vzmax, Vqmax, Vomax,
            demand, production_times, setup_times, capacities,
            production_costs, setup_costs, inventory_costs, 
            r1_arr[i], r2_arr[i], r3_arr[i], r4_arr[i],
            r5_arr[i],r6_arr[i], rand_vals1[i], rand_vals2[i]
        )
        # update swarm in place 
        Xs[i], Qs[i], Os[i],  VXs[i], VQs[i], VOs[i] = Xi, Qi, Oi, VXi, VQi, VOi
        pbest_X[i], pbest_Q[i], pbest_O[i], pbest_Val[i] = pbest_Xi, pbest_Qi, pbest_Oi, pbest_Values


        t = get_thread_id()
        pv0 = pbest_Values[0, 0]
        pv1 = pbest_Values[0, 1]
        lb0 = local_best[t, 0, 0]
        lb1 = local_best[t, 0, 1]
        if (pv0 < lb0) or (pv0 == lb0 and pv1 < lb1):
            local_best[t, 0, 0] = pv0
            local_best[t, 0, 1] = pv1
            local_idx[t]       = i

    return local_best, local_idx
# ------------------------------------------------------------------------------------------------------------------------------------- #
class Swarm:
    def __init__(self, n_particles, config, C1=1.5, C2=1.5, Vzmax=4.0, Vqmax=0.1, Vomax=4.0,inertia = 1.0):

        self.cfg = config()
        N, M, T       = n_particles, self.cfg.M, self.cfg.T


        self.inertia = inertia
        self.C1, self.C2, self.Vzmax, self.Vqmax, self.Vomax = C1, C2, Vzmax, Vqmax, Vomax

        # initialize all particles using 3D arrays
        self.Xs   = np.random.randint(0, 2, size=(N, M, T)).astype(np.int8)
        self.Os   = np.random.randint(0, 2, size=(N, M, T)).astype(np.int8)
        self.Qs = np.random.rand(N, M, T)
        self.VXs  = (np.random.rand(N, M, T)*2 - 1) * Vzmax
        self.VOs  = (np.random.rand(N, M, T)*2 - 1) * Vomax
        self.VQs  = (np.random.rand(N, M, T)*2 - 1) * Vqmax

        self.pbest_X   = self.Xs.copy()
        self.pbest_Q   = self.Qs.copy()
        self.pbest_O   = self.Os.copy()
        self.pbest_Val = np.full((N,  1, 2), np.inf)

        self.gbest_X   = np.zeros((M, T), dtype=np.int8)
        self.gbest_O   = np.zeros((M, T), dtype=np.int8)
        self.gbest_Q   = np.zeros((M, T), dtype=np.float64)
        self.gbest_Val = np.full((1,2), np.inf)



        # one cold-start update to fill pbest and gbest
        self._reduce_global(update_swarm(
            self.Xs, self.Qs, self.Os, self.VXs, self.VQs, self.VOs,
            self.pbest_X, self.pbest_Q, self.pbest_O, self.pbest_Val,
            self.gbest_X, self.gbest_Q, self.gbest_O,
            self.inertia, self.C1, self.C2, self.Vzmax, self.Vqmax, self.Vomax,
            self.cfg.demand,
            self.cfg.production_times,
            self.cfg.setup_times,
            self.cfg.capacities,
            self.cfg.production_costs,
            self.cfg.setup_costs,
            self.cfg.inventory_costs,
            np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N),
            np.zeros((N, T, M)), np.zeros((N, T, M))
        ))

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
                self.gbest_O[:] = self.pbest_O[i]
                self.gbest_Q[:] = self.pbest_Q[i]

    def optimize(self, n_iter):
        
        N, T, M = self.Xs.shape
    
        for t in range(n_iter):
           
 
            # pre-generate random numbers
            r1, r2, r3, r4, r5, r6 = (np.random.rand(N) for _ in range(6))
            rand_vals1     = np.random.rand(N, T, M)
            rand_vals2     = np.random.rand(N, T, M)

            # perform iteration of entire swarm
            thread_results = update_swarm(
                self.Xs, self.Qs, self.Os, self.VXs, self.VQs, self.VOs,
                self.pbest_X, self.pbest_Q, self.pbest_O, self.pbest_Val,
                self.gbest_X, self.gbest_Q, self.gbest_O,
                self.inertia, self.C1, self.C2, self.Vzmax, self.Vqmax, self.Vomax,
                self.cfg.demand, self.cfg.production_times,
                self.cfg.setup_times, self.cfg.capacities,
                self.cfg.production_costs, self.cfg.setup_costs, self.cfg.inventory_costs,
                r1, r2, r3, r4, r5, r6, rand_vals1, rand_vals2
            )
            # update global best solution
            self._reduce_global(thread_results)
    
        return self.gbest_Val

        