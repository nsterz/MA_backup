import numpy as np
from numba import njit, prange, get_thread_id, get_num_threads
from levy_flights import estimate_mean_abs_levy, levy_step, levy_bit_flip,levy_jump_Q
from Swarm_IQ import decode_prod_plan, compute_objective

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
                Q[i, j] = pbest_Q[i, j] + eps * A_avg
                
                if Q[i, j] < 0.0:
                    Q[i, j] = 0.0
                elif Q[i, j] > 1.0:
                    Q[i, j] = 1.0

        X[:] = levy_bit_flip(pbest_X, bit_flip_share, levy_alpha)

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