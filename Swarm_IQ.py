from numba import njit
import numpy as np
from helper_functions import round

@njit
def encode_prod_plan(plan, demand):
    T = plan.shape[1]
    M = plan.shape[0]
    X = np.zeros((T, M))
    Q = np.zeros((T, M)).astype(np.float64)
    # iterate products
    for i in range(M):
        for t in range(T):
            # skip non production periods
            if plan[i, t] == 0:
                X[t,i] = 0
            else:
                X[t,i] = 1
                # Find next production period t2
                t2 = T - 1
                for j in range(t+1, T):
                    
                    if plan[i, j] > 0:
                        t2 = j
                        break
                
                # ---------------------------------------------------------------------------------------------------------- #
                # t = 1
                if t == 0:
                    
                    # Case 1: If no production is planned in the future, meet all remaining demand now
                    if (t2 == T - 1) and (plan[i, t2] == 0):
                        
                        continue
                    
                   # Case 2: If production is planned in the future, meet all the demand until the next production period
                    else:
                        
                        Q [t,i] = 0.0
                        s = 0.0
                        
                        for j in range(t, t2):
                            s += demand[i, j]

                        if  (plan[i,t] - s < 0) and (demand[i,t] > 0):
                            Q[t,i] = (plan[i,t] - s)/ demand[i,t]
                            
                        elif (plan[i,t] - s > 0) and (demand[i,t2] > 0):
                            
                            Q[t2,i] = (plan[i,t] - s)/demand[i,t2]
                


                    # ---------------------------------------------------------------------------------------------------------- #
                # t = T
                elif t == T - 1:
                    
                    # Look backwards for a prior production period t1
                    t1 = 0
                    for j in range(t-1, -1, -1):
                        if plan[i, j] > 0:
                            t1 = j
                            break
                            
                    # Case 3: no prior production before the last period --> meet all demand now
                    if (t1==0) and (plan[i, t1]==0):
                        continue

                            
                    # Case 4: prior production --> meet  backordered demand from t1 and current demand
                    else:
                        s = plan[i,t] + round(min(Q[t1, i] * demand[i, t1],0))
                        # avoid zerodivisionError
                        if demand[i,t] > 0:
                            Q[t,i] = 1 - (s / demand[i,t])
                        
                        
                # ----------------------------------------------------------------------------------------------------------#
                # t in [2,T-1]
                else:
                    # Look backwards for a prior production period t1
                    t1 = 0
                    for j in range(t-1, -1, -1):
                        if plan[i, j] > 0:
                            t1 = j
                            break

                    
                    # Case 5: no production before AND after t --> meet all demand now
                    if (plan[i,t1]==0) and (plan[i,t2]==0):
                        continue

                    # Case 6: production before but not after t --> produce backorders and then produce everything for t to T               
                    elif (plan[i,t1]>0) and (plan[i,t2]==0):
                        continue

                    # Case 7: production after but not before t -->  produce everything for t1 to t and preproduce t+1 to t2
                    elif (plan[i,t1]==0) and (plan[i,t2]>0):
                        
                        
                        s1 = 0.0
                        for j in range(t1, t):
                            s1 += demand[i, j]
                        
                        s2 = 0.0
                        for j in range(t+1, t2):
                            s2 += demand[i, j]

                        q = plan[i,t] - s1 -s2 
                        if q ==  demand[i,t]:
                            Q[t,i] = 0.0
                        elif q > demand[i,t] and (demand[i,t2] > 0):
                            Q[t,i] = 0.0
                            Q[t2,i] = (q-demand[i,t])/demand[i,t2]
                            print(i,t,(q-demand[i,t])/demand[i,t2])
                            
                        elif (demand[i,t] > 0):
                            Q[t,i] = (q-demand[i,t])/demand[i,t]
                        
                        
                    # Case 8: production after and before t --> produce in advance t to t2 and cover backorders of t1 
                    else:
                        
                        s = 0.0
                        for j in range(t+1, t2):
                            s += demand[i, j]
                            
                        q = plan[i,t] -s + round(min(Q[t1, i] *demand[i, t1],0))
                        # if Q is already positive, a partial lot was already produced
                        if Q[t,i] > 0:
                            q1 = plan[i,t] -s + round(min(Q[t1, i] * demand[i, t1],0) - (1-Q[t,i]) * demand[i, t])
                            if q1 > 0 and (demand[i,t2]>0):
                                Q[t2,i] = q1/ demand[i,t2]
                                
                        elif q ==  demand[i,t]:
                            Q[t,i] = 0.0

                        elif q > demand[i,t] and (demand[i,t2] > 0):
                            
                            Q[t,i] = 0.0
                            Q[t2,i] = (q-demand[i,t])/demand[i,t2]
                            
                        elif (demand[i,t] > 0):
                            Q[t,i] = (q-demand[i,t])/demand[i,t]
                                
                      
                        
    return X, Q


@njit
def decode_prod_plan(ZPK, QPK, demand):
    T = ZPK.shape[1]
    M = ZPK.shape[0]
    XPK = np.zeros((M, T))
    # iterate products
    for i in range(M):
        for t in range(T):
            # skip non production periods
            if ZPK[i, t] == 0:
                continue
            else:
                
                # Find next production period t2
                t2 = T - 1
                for j in range(t+1, T):
                    if ZPK[i, j] == 1:
                        t2 = j
                        break
                # ---------------------------------------------------------------------------------------------------------- #
                # t = 1
                if t == 0:
                    # Case 1: If no production is planned in the future, meet all remaining demand now
                    if (t2 == T - 1) and (ZPK[i, t2] == 0):
                        
                        s = 0.0
                        for j in range(t, T):
                            s += demand[i, j]
                        XPK[i, t] = s
                        
                    # Case 2: If production is planned in the future, meet all the demand until the next production period
                    else:
                        XPK[i, t] = (1 + min(QPK[i, t],0)) * demand[i, t] + max(QPK[i, t2] * demand[i, t2],0)
        
                        s = 0.0
                        
                        for j in range(t+1, t2):
                            s += demand[i, j]
                            
                        XPK[i, t] = round(XPK[i, t] + s)
                        
                # ---------------------------------------------------------------------------------------------------------- #
                # t = T
                elif t == T - 1:
                    
                    # Look backwards for a prior production period t1
                    t1 = 0
                    for j in range(t-1, -1, -1):
                        if ZPK[i, j] == 1:
                            t1 = j
                            break
                            
                    # Case 3: no prior production before the last period --> meet all demand now
                    if (t1==0) and (ZPK[i,t1]==0):
                        s = 0.0
                        for j in range(t1, t+1):
                            s += demand[i, j]
                        XPK[i, t] = s
                        
                        
                    # Case 4: prior production --> meet  backordered demand from t1 and current demand
                    else:
                        XPK[i, t] = round((1-QPK[i, t]) * demand[i, t] - min(QPK[i, t1] * demand[i, t1],0))
                        #XPK[i, t] = round((1-max(0,QPK[i, t])) * demand[i, t] - min(QPK[i, t1] * demand[i, t1],0))

                       
                # ----------------------------------------------------------------------------------------------------------#
                # t in [2,T-1]
                else:
                    # Look backwards for a prior production period t1
                    t1 = 0
                    for j in range(t-1, -1, -1):
                        if ZPK[i, j] == 1:
                            t1 = j
                            break

                    
                    # Case 5: no production before AND after t --> meet all demand now
                    if (ZPK[i,t1]==0) and (ZPK[i,t2]==0):
                        s = 0.0
                        for j in range(t1, t2+1):
                            s += demand[i, j]
                        XPK[i, t] = s
                        
                        
                    # Case 6: production before but not after t --> produce backorders and then produce everything for t to T               
                    elif (ZPK[i,t1]==1) and (ZPK[i,t2]==0):
                        
                        
                        XPK[i, t] = (1 - QPK[i, t]) * demand[i, t] - min(QPK[i, t1] * demand[i, t1],0)

                        s = 0.0
                        for j in range(t+1, t2+1):
                            s += demand[i, j]
                        
                        XPK[i, t] = round(XPK[i, t] + s)

                    # Case 7: production after but not before t -->  produce everything for t1 to t and preproduce t+1 to t2
                    elif (ZPK[i,t1]==0) and (ZPK[i,t2]==1):
                        
                        XPK[i, t] = (1 + QPK[i, t]) * demand[i, t] + max(QPK[i, t2] * demand[i, t2],0)
                        #XPK[i, t] = (1 + min(0,QPK[i, t])) * demand[i, t] + max(QPK[i, t2] * demand[i, t2],0)
                        
                        s1 = 0.0
                        for j in range(t1, t):
                            s1 += demand[i, j]
                        
                        s2 = 0.0
                        for j in range(t+1, t2):
                            s2 += demand[i, j]
                         
                        XPK[i, t] = round(XPK[i, t] + s1 + s2)
                        
                    # Case 8: production after and before t --> produce in advance t to t2 and cover backorders of t1 
                    else:
                        
                            
                        XPK[i, t] = (ZPK[i, t]-abs(QPK[i, t])) * demand[i, t] + max(QPK[i, t2] * demand[i, t2],0) - min(QPK[i, t1] *demand[i, t1],0)

                            
                        s = 0.0
                        for j in range(t+1, t2):
                            s += demand[i, j]

                        XPK[i, t] = round(XPK[i, t] + s)
                    

    # Build the output array and apply np.maximum with 0, then transpose.
    out = np.empty((M, T))
    for i in range(M):
        for t in range(T):
            if XPK[i, t] < 0:
                out[i, t] = 0.0
            else:
                out[i, t] = XPK[i, t]
    outT = np.empty((T, M))
    for t in range(T):
        for i in range(M):
            outT[t, i] = out[i, t]
    return outT




@njit
def compute_objective(prod_quant, X, setup_costs, production_costs,
                            production_times, setup_times, capacities,
                            inventory_costs,  demand):
  
    # prod_quant : (T, M)
    # X          : (T, M)
    # production_times : shape (M,)
    # setup_times      : shape (M,)
    # capacities       : shape (T,)
    # demand           : shape (M, T) 

    # init counter for number of constraint violations
    num_violations = 0.0
    
    T = prod_quant.shape[0]  # periods
    M = prod_quant.shape[1]  # products

    # Compute production times and setup times for each period using loops.
    prod_time = np.zeros(T)
    setup_time = np.zeros(T)
    for t in range(T):
        for i in range(M):
            prod_time[t] += prod_quant[t, i] * production_times[i]
            setup_time[t] += X[t, i] * setup_times[i]
            
    # Compute overtime per period:
    overtime = np.zeros(T)
    for t in range(T):
        ot = prod_time[t] + setup_time[t] - capacities[t]
        num_violations += ot if ot > 0 else 0.0


    # Compute net inventory for each product per period.
    # Note: swarm.demand is shape (M, T); we want to compute per product over T
    net_inv = np.zeros((T, M))
    for i in range(M):
        for t in range(T):
            if t == 0:
                net_inv[t, i] = prod_quant[t, i] - demand[i, t]
            else:
                net_inv[t, i] = net_inv[t-1, i] + prod_quant[t, i] - demand[i, t]
    inv = np.zeros((T, M))
    # calculate backorders for each period
    backorders = np.zeros((T, M))
    for t in range(T):
        for i in range(M):
            if net_inv[t, i] >= 0:
                inv[t, i] = net_inv[t, i]
                backorders[t, i] = 0.0
            else:
                inv[t, i] = 0.0
                num_violations += -net_inv[t, i]

                
    # sum all the costs
    total = 0.0
    # Sum production and setup costs (per period and product)
    for t in range(T):
        for i in range(M):
            total += production_costs[i] * prod_quant[t, i]
            total += setup_costs[i] * X[t, i]
            total += inventory_costs[i] * inv[t, i]


    return  np.array([num_violations, total], dtype=np.float64)