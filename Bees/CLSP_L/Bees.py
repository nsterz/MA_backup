from numba import njit,prange, get_thread_id, get_num_threads
import numpy as np
from Swarm_IQ_L import decode_prod_plan, decode_and_evaluate
from helper_functions import rounded

# ------------------------------------------------------------------------------------------------------------------------------------------ #

@njit
def iterate_abc(
    X, Q, O, trial, fitness,
    gbest_X, gbest_Q, gbest_O, gbest_val,
    demand, production_times, setup_times, capacities,
    production_costs, setup_costs, inventory_costs,
    limit, K_onlookers,
    rand_partner_e, rand_phi_e, rand_Xflip_e, rand_Oflip_e,
    rand_partner_o, rand_phi_o, rand_Xflip_o, rand_Oflip_o, best_idx
):
    SN, M, T = X.shape
    #best_viols = gbest_val[0, 0]
    #best_obj   = gbest_val[0, 1]

    # 1. Employed bees phase
    for i in range(SN):
        k = rand_partner_e[i]
        if k == i:
            continue
            
        phi   = rand_phi_e[i]
        share = abs(phi)

        # generate candidate
        Xc = X[i].copy()
        Qc = Q[i].copy()
        Oc = O[i].copy()

        #position update
        for m in range(M):
            for t in range(T):
                if X[i, m, t] != X[k, m, t] and rand_Xflip_e[i, m, t] < (0.2*share):
                    Xc[m, t] = 1.0 - Xc[m, t]
                    
                if O[i, m, t] != O[k, m, t] and rand_Oflip_e[i, m, t] < (0.2* share):
                    Oc[m, t] = 1.0 - Oc[m, t]
                    

                val = Q[i, m, t] + phi * (Q[i, m, t] - Q[k, m, t])
                if val < 0.0:
                    val = 0.0
                elif val > 1.0:
                    val = 1.0
                Qc[m, t] = val

        
        fit_c = decode_and_evaluate(
            Xc, Qc, Oc, demand,
            setup_costs, production_costs,
            production_times, setup_times,
            capacities, inventory_costs
        )

        # compare & update
        if (fit_c[0] < fitness[i, 0]) or (fit_c[0] == fitness[i, 0] and fit_c[1] < fitness[i, 1]):
            X[i]          = Xc
            Q[i]          = Qc
            O[i]          = Oc
            fitness[i, 0] = fit_c[0]
            fitness[i, 1] = fit_c[1]
            trial[i]      = 0
        else:
            trial[i] += 1

    # 2. Onlooker bees phase
    w_sum = 0.0
    w     = np.empty(SN, dtype=np.float64)
    for i in range(SN):
        w_sum += 1.0 / (1.0 + fitness[i, 0] )

    for o in range(K_onlookers):
        # roulette‐wheel select source idx
        threshold = rand_phi_o[o] * w_sum
        accum     = 0.0
        idx       = 0
        for j in range(SN):
            accum += 1.0 / (1.0 +fitness[j, 0])
            if accum >= threshold:
                idx = j
                break

        k = rand_partner_o[o]
        if k == idx:
                continue

        phi   = rand_phi_o[o] * 2.0 - 1.0
        share = abs(phi)

        # generate candidate
        Xc = X[idx].copy()
        Qc = Q[idx].copy()
        Oc = O[idx].copy()

        # update position
        for m in range(M):
            for t in range(T):
                if X[idx, m, t] != X[k, m, t] and rand_Xflip_o[o, m, t] <(share):
                    Xc[m, t] = 1.0 - Xc[m, t]

                if O[idx, m, t] != O[k, m, t] and rand_Oflip_o[o, m, t] <(share):
                    Oc[m, t] = 1.0 - Oc[m, t]

                
                val = Q[idx, m, t] +  phi * (Q[idx, m, t] - Q[k, m, t])
                if val < 0.0:
                    val = 0.0
                elif val > 1.0:
                    val = 1.0
                Qc[m, t] = val

        fit_c = decode_and_evaluate(
                Xc, Qc, Oc, demand,
                setup_costs, production_costs,
                production_times, setup_times,
                capacities, inventory_costs)

        #  compare & update
        if (fit_c[0] < fitness[idx, 0]) or (fit_c[0] == fitness[idx, 0] and fit_c[1] < fitness[idx, 1]):

            X[idx]          = Xc
            Q[idx]          = Qc
            O[idx]          = Oc
            fitness[idx, 0] = fit_c[0]
            fitness[idx, 1] = fit_c[1]
            trial[idx]      = 0
        else:
            trial[idx] += 1

    # 3. Scout bee phase (on‐demand)
    for i in range(SN):
        if i == best_idx:          # protect only the true best slot
                continue
        if trial[i] > limit:
            for m in range(M):
                for t in range(T):
                    O[i, m, t] = 1.0 if np.random.rand() < 0.5 else 0.0
                    X[i, m, t] = 1.0 if np.random.rand() < 0.5 else 0.0
                    Q[i, m, t] = np.random.rand()

            trial[i] = 0


            fit_i = decode_and_evaluate(
                    X[i], Q[i], O[i], demand,
                    setup_costs, production_costs,
                    production_times, setup_times,
                    capacities, inventory_costs)
            
            fitness[i, 0] = fit_i[0]
            fitness[i, 1] = fit_i[1]

    # 4. Update global bests

    best_local_idx = best_idx
    best_val_0 = gbest_val[0, 0]
    best_val_1 = gbest_val[0, 1]
    
    for i in range(SN):
        fi0, fi1 = fitness[i, 0], fitness[i, 1]
        if (fi0 < best_val_0) or (fi0 == best_val_0 and fi1 < best_val_1):
            best_local_idx = i
            best_val_0 = fi0
            best_val_1 = fi1
    
    if best_local_idx != best_idx:
        gbest_val[0] = fitness[best_local_idx].copy()
        gbest_X[:] = X[best_local_idx].copy()
        gbest_Q[:] = Q[best_local_idx].copy()
        gbest_O[:] = O[best_local_idx].copy()
        best_idx = best_local_idx

    return (
        X, Q, O, trial, fitness,
        gbest_X, gbest_Q, gbest_O, gbest_val, best_idx)


# ------------------------------------------------------------------------------------------------------------------------------------------ #


class ABC:
    def __init__(
        self, n_bees, config, limit, K_onlookers):
        
        # Problem config
        self.cfg = config()
        self.SN = n_bees
        self.M = self.cfg.M
        self.T = self.cfg.T

        # Scout threshold
        self.limit = limit
        # number of onlooker 
        self.K = K_onlookers 

        # Initialize solution arrays
        self.X = np.random.randint(0, 2, size=(self.SN, self.M, self.T)).astype(np.float64)
        self.O = np.random.randint(0, 2, size=(self.SN, self.M, self.T)).astype(np.float64)
        #self.O = np.zeros((self.SN, self.M, self.T)).astype(np.float64)
        self.Q = np.random.rand(self.SN, self.M, self.T)

        # Trial counters
        self.trial = np.zeros(self.SN, dtype=np.int32)

        # Evaluate initial fitness
        self.fitness = np.full((self.SN, 2), np.inf)
        for i in range(self.SN):
            
            fit = decode_and_evaluate(
                    self.X[i], self.Q[i], self.O[i], self.cfg.demand,
                    self.cfg.setup_costs, self.cfg.production_costs,
                    self.cfg.production_times, self.cfg.setup_times,
                    self.cfg.capacities, self.cfg.inventory_costs)
            self.fitness[i, 0] = fit[0]
            self.fitness[i, 1] = fit[1]


        # Global best iniitalization
        # find minimal constraint violations
        min_viols = np.min(self.fitness[:, 0])
        # indices with minimal violations
        candidates = np.where(self.fitness[:, 0] == min_viols)[0]
        # among them, pick the one with minimal objective cost
        obj_vals = self.fitness[candidates, 1]
        self.best_idx = candidates[np.argmin(obj_vals)]
        # set global best value, X, and Q
        self.gbest_val = self.fitness[self.best_idx:self.best_idx+1].copy()
        self.gbest_X = self.X[self.best_idx].copy()
        self.gbest_Q = self.Q[self.best_idx].copy()   
        self.gbest_O = self.O[self.best_idx].copy()   
        
        
    def step(self):
        # pre-generate random values
        self.rand_partner_empl = np.random.randint(0,self.SN,self.SN)
        self.rand_phi_empl    = np.random.rand(self.SN)*2-1
        self.rand_Xflip_empl    = np.random.rand(self.SN,self.M,self.T)
        self.rand_Oflip_empl    = np.random.rand(self.SN,self.M,self.T)
        self.rand_partner_onl = np.random.randint(0,self.SN,self.K)
        self.rand_phi_onl    = np.random.rand(self.K)
        self.rand_Xflip_onl    = np.random.rand(self.K,self.M,self.T)
        self.rand_Oflip_onl    = np.random.rand(self.K,self.M,self.T)
    
        (self.X, self.Q, self.O, self.trial, self.fitness,
         self.gbest_X, self.gbest_Q, self.gbest_O, self.gbest_val, 
         self.best_idx) = iterate_abc(
            self.X, self.Q, self.O, self.trial, self.fitness,
            self.gbest_X, self.gbest_Q, self.gbest_O, self.gbest_val,
            self.cfg.demand, self.cfg.production_times, self.cfg.setup_times,
            self.cfg.capacities, self.cfg.production_costs, self.cfg.setup_costs,
            self.cfg.inventory_costs, self.limit, self.K,
            self.rand_partner_empl, self.rand_phi_empl, self.rand_Xflip_empl, self.rand_Oflip_empl,
            self.rand_partner_onl, self.rand_phi_onl, self.rand_Xflip_onl, self.rand_Oflip_onl, self.best_idx
        )
    
    def optimize(self, n_iter):
        for _ in range(n_iter):
            self.step()
        return self.gbest_val

    def _print_setup(self):
        
        X = self.gbest_X
        O = self.gbest_O
        M, T = len(X), len(X[0])
        w_k = max(len("Product k"), len(str(M))) + 2
        w_c = max(len('-->'), 2) + 2
        total_width = w_k + T * (w_c + 1) + 2
        print("Setup sequence".center(total_width))
        def border():
            parts = ['+' + '-' * w_k] + ['+' + '-' * w_c for _ in range(T)]
            print(''.join(parts) + '+')
    
        border()
        merged_w = T * w_c + (T - 1)
        print(
            '|' + ' ' * w_k +
            '|' + 'Period t'.center(merged_w) +
            '|'
        )
        border()
        header = f"| {'Product k'.center(w_k-1)}"
        for t in range(1, T+1):
            header += f"| {str(t).center(w_c-1)}"
        print(header + "|")
        border()
    
        for m in range(M):
            row = f"| {str(m+1).ljust(w_k-1)}"
            for t in range(T):
                # arrow condition
                if 0 < t < T-1 \
                 and O[m][t-1] == 1 and X[m][t-1] == 1 \
                 and O[m][t+1] == 0 and X[m][t+1] == 1 \
                and  np.sum(O[:,t])==0 \
                and  np.sum(X[:,t])==0:
                    cell = '-->'
                else:
                    cell = ('x□' if O[m][t] and X[m][t]
                           else 'x'   if O[m][t]
                           else '□'   if X[m][t]
                           else '')
                row += f"| {cell.center(w_c-1)}"
            print(row + "|")
            border()
    def _print_prod_plan(self):

        M, T = self.gbest_X.shape[0], self.gbest_X.shape[1]
        w_k = max(len("Product k"), len(str(M))) + 2
        w_c = max(len('-->'), 2) + 2
        total_width = w_k + T * (w_c + 1) + 2
        plan = decode_prod_plan(self.gbest_X, self.gbest_Q, self.cfg.demand).T
                # Production quantities
        print("Production quantities".center(total_width))
        def border():
            parts = ['+' + '-' * w_k] + ['+' + '-' * w_c for _ in range(T)]
            print(''.join(parts) + '+')
        border()
        print('|' + ' '*w_k + '|' + 'Period t'.center(T*w_c + T-1) + '|')
        border()
        header = "| " + "Product k".center(w_k-1)
        for t in range(1, T+1):
            header += f"| {str(t).center(w_c-1)}"
        print(header + "|")
        border()
        for m in range(M):
            row = "| " + str(m+1).ljust(w_k-1)
            for t in range(T):
                num = f"{plan[m][t]:.0f}"
                row += f"| {num.center(w_c-1)}"
            print(row + "|")
            border()

    def print(self):
        self._print_prod_plan()
        self._print_setup()

