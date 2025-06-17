from numba import njit, prange, get_thread_id, get_num_threads
from levy_flights import estimate_mean_abs_levy, levy_step, levy_bit_flip,levy_jump_Q
from Swarm_IQ import decode_prod_plan, compute_objective
import numpy as np
# ------------------------------------------------------------------------------------------------------------------------------------- #
@njit
def iterate(X, Q, VZPK, VQPK,
            pbest_X, pbest_Q, pbest_value,
            gbest_X, gbest_Q,
            inertia, C1, C2, Vzmax, Vqmax,
            demand, production_times, setup_times, capacities,
            production_costs, setup_costs, inventory_costs, 
            r1, r2, r3, r4, rand_vals, stagn_count, levy_Sx, levy_Sq, levy_alpha,levy_q_step,stagn_thres,  particle_idx):
    M, T = X.shape
    stagnation = stagn_count[particle_idx]
    
    # perform a levy flight after particle stagnated for a certain time
    if stagnation > stagn_thres:
        
        X = levy_bit_flip(X, levy_Sx, levy_alpha)
        Q = levy_jump_Q(Q, levy_Sq, levy_q_step, levy_alpha)
        

        stagnation = 0.0
        
    # perform regular update of particle
    else:
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
    
        # update X
        for i in range(M):
            for j in range(T):
                if 1.0 / (1.0 + np.exp(-VZPK[i, j])) > rand_vals[i, j]:
                    X[i, j] = 1.0
                else:
                    X[i, j] = 0.0
    
        for i in range(M):
            for j in range(T):
                Q[i, j] = Q[i, j] + VQPK[i, j]
                if Q[i, j] > 1.0:
                    Q[i, j] = 1.0
                elif Q[i, j] < 0.0:
                    Q[i, j] = 0.0


        

    # decode and evaluate
    prod_quant = decode_prod_plan(X.T, Q.T, demand)
    X_float = X.astype(np.float64)
    fitness = compute_objective(
        prod_quant, X_float,
        setup_costs, production_costs,
        production_times, setup_times,
        capacities, inventory_costs, demand
    )

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
        pbest_value[0, 0] = fv0
        pbest_value[0, 1] = fv1
    else:
        stagnation += 1.0

    return X, Q, VZPK, VQPK, pbest_X, pbest_Q, pbest_value, stagnation

# ------------------------------------------------------------------------------------------------------------------------------------- #
@njit(parallel=True)
def update_swarm(Xs, Qs, VXs, VQs, pbest_X, pbest_Q, pbest_Val,
                 gbest_X, gbest_Q, inertia, C1, C2, Vzmax, Vqmax,
                 demand, production_times, setup_times, capacities,
                 production_costs, setup_costs, inventory_costs,
                 r1_arr, r2_arr, r3_arr, r4_arr, rand_vals, stagn_count, levy_Sx, levy_Sq, levy_alpha, levy_q_step,stagn_thres):

    nthreads = get_num_threads()
    local_best = np.full((nthreads, 1, 2), np.inf)
    local_idx  = np.zeros(nthreads, np.int64)
    N = Xs.shape[0]

    for i in prange(N):
        Xi, Qi, VXi, VQi, pbest_Xi, pbest_Qi, pbest_Values, stagn_val_i = iterate(
            Xs[i], Qs[i], VXs[i], VQs[i],
            pbest_X[i], pbest_Q[i], pbest_Val[i],
            gbest_X, gbest_Q,
            inertia, C1, C2, Vzmax, Vqmax,
            demand, production_times, setup_times, capacities,
            production_costs, setup_costs, inventory_costs, 
            r1_arr[i], r2_arr[i], r3_arr[i], r4_arr[i],
            rand_vals[i], stagn_count, levy_Sx, levy_Sq, levy_alpha, levy_q_step, stagn_thres,  particle_idx = i
        )
        # update swarm in place 
        Xs[i], Qs[i], VXs[i], VQs[i] = Xi, Qi, VXi, VQi
        pbest_X[i], pbest_Q[i], pbest_Val[i] = pbest_Xi, pbest_Qi, pbest_Values

        # write back the updated stagnation count
        stagn_count[i] = stagn_val_i


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