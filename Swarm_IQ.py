from numba import njit
import numpy as np
from helper_functions import round



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
@njit
def decode_prod_plan(ZPK, QPK, demand):
    """
    Decodes a production plan from swarm-algorithm position
    -----
    Considers eight cases for each period t and product i:
      1 t = first period
      1.1. No future production after period 1--> meet current demand only.
      1.2 Future production planned after period 1 --> meet until next, preproduce if needed for t2 to t3
      2 t = final period
      2.1 At last period, no prior production --> meet current demand.
      2.1. At last period with prior production --> meet rest of current demand
      3 t in [2,T-1]
      3.1 No production before or after t  --> meet current demand.
      3.2 Production before but not after t --> produce rest of demand from t to T
      3.3 Production after but not before t --> produce until next, preproduce ahead for t2 to t3
      3.4 Production both before and after t --> produce rest of demand from t to t2, preproduce ahead for t2 to t3
    
    Parameters
    ----------
    ZPK : 2D ndarray of int
        Binary production decision matrix, shape (M, T).
    QPK : 2D ndarray of float
        Preproduction fractions, shape (M, T).
    demand : 2D ndarray of float
        Demand per product per period, shape (M, T).

    Returns
    -------
    outT : 2D ndarray of float
        Production quantities per period per product, shape (T, M).
    """
    T = ZPK.shape[1]
    M = ZPK.shape[0]
    XPK = np.zeros((M, T))
    
    
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
                    # Case 1: no production is planned in the future
                    if (t2 == T - 1) and (ZPK[i, t2] == 0):
                        s = 0.0
                        for j in range(t, T):
                            s += demand[i, j]
                        
                        XPK[i, t] = s
                        
                    # Case 2:  production is planned in the future
                    else:
                        # find production period after the next production period
                        t3 = T - 1
                        for j in range(t2+1, T):
                            if ZPK[i, j] == 1:
                                t3 = j
                                break

                        end = t3 + 1 if (ZPK[i, t3] == 0 or t3 == t2) else t3
   
                        s1 = 0.0
                        for j in range(t2, end):
                            s1 += demand[i, j]
                            
                            
                        s2 = 0.0                       
                        for j in range(t, t2):
                            s2 += demand[i, j]
                            
                        XPK[i, t] = round(s2 + QPK[i, t2] * s1)
                        
                # ---------------------------------------------------------------------------------------------------------- #
                # t = T
                elif t == T - 1:
                    
                    # Look backwards for a prior production period t1
                    t1 = 0
                    for j in range(t-1, -1, -1):
                        if ZPK[i, j] == 1:
                            t1 = j
                            break
                            
                    # Case 3: no prior production before the last period 
                    if (t1==0) and (ZPK[i,t1]==0):
                        
                        XPK[i, t] = demand[i,t]
                        
                    # Case 4: prior production 
                    else:
                        XPK[i, t] = round((1-QPK[i, t]) * demand[i, t])
                          
                # ----------------------------------------------------------------------------------------------------------#
                # t in [2,T-1]
                else:
                    # Look backwards for a prior production period t1
                    t1 = 0
                    for j in range(t-1, -1, -1):
                        if ZPK[i, j] == 1:
                            t1 = j
                            break

                    # Case 5: no production before AND after t
                    if (ZPK[i,t1]==0) and (ZPK[i,t2]==0):
                        s = 0.0
                        for j in range(t, T):
                            s += demand[i, j]
                        XPK[i, t] = s
                        
                        
                    # Case 6: production before but not after t              
                    elif (ZPK[i,t1]==1) and (ZPK[i,t2]==0):
                        s = 0.0
                        for j in range(t, t2+1):
                            s += demand[i, j]
                        
                        XPK[i, t] = round((1 - QPK[i, t]) * s)

                    # Case 7: production after but not before t 
                    elif (ZPK[i,t1]==0) and (ZPK[i,t2]==1):
                        t3 = T - 1
                        for j in range(t2+1, T):
                            if ZPK[i, j] == 1:
                                t3 = j
                                break
                                
                        end = t3 + 1 if (ZPK[i, t3] == 0 or t3 == t2) else t3
                        s1 = 0.0
                        for j in range(t2, end):
                            s1 += demand[i, j]
                            
                        s2 = 0.0
                        for j in range(t, t2):
                            s2 += demand[i, j]

                        XPK[i, t] = round(s2 + QPK[i, t2] * s1)
                        
                        
                    # Case 8: production after and before t  
                    else:
                        t3 = T - 1
                        for j in range(t2+1, T):
                            if ZPK[i, j] == 1:
                                t3 = j
                                break
                        end = t3 + 1 if (ZPK[i, t3] == 0 or t3 == t2) else t3
   
                        s1 = 0.0
                        for j in range(t2, end):
                            s1 += demand[i, j]
   
                        s2 = 0.0
                        for j in range(t, t2):
                            s2 += demand[i, j]

                        XPK[i, t] = round((1-QPK[i, t]) * s2 + QPK[i, t2] * s1)
                    

    # Build the output array and remove negative values, then transpose.
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

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

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
                        
                        t3 = T - 1
                        for j in range(t2+1, T):
                            if plan[i, j] > 0:
                                t3 = j
                                break
                        
                        s1 = 0.0
                        for j in range(t, t2):
                            s1 += demand[i, j]

                        end = t3 + 1 if (plan[i, t3] == 0 or t3 == t2) else t3
   
                        s2 = 0.0
                        for j in range(t2, end):
                            s2 += demand[i, j]

                        q = plan[i,t] - s1
                        if (q > 0) and (s2>0):
                            
                            Q[t2,i] = q / s2

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

                            
                    # Case 4: prior production --> 
                    else:
                        
                        if demand[i,t] > 0:
                            Q[t,i] = 1 - (plan[i,t] / demand[i,t])
                        
                        
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
                        
                        t3 = T - 1
                        for j in range(t2+1, T):
                            if plan[i, j] > 0:
                                t3 = j
                                break
                                
                        end = t3 + 1 if (plan[i, t3] == 0 or t3 == t2) else t3
                        
                        s1 = 0.0
                        for j in range(t2, end):
                            s1 += demand[i, j]
                            
                        s2 = 0.0
                        for j in range(t, t2):
                            s2 += demand[i, j]
                        
                        
                        # units that were preproduced for t2 to t3
                        q = plan[i,t] -s2 

                        if (q > 0) and (s1 > 0):
                            Q[t2,i] = q / s1
                            
                        
                        
                    # Case 8: production after and before t --> produce in advance t to t2 and cover backorders of t1 
                    else:
                        t3 = T - 1
                        for j in range(t2+1, T):
                            if plan[i, j] > 0:
                                t3 = j
                                break
                        end = t3 + 1 if (plan[i, t3] == 0 or t3 == t2) else t3
                        
                        s1 = 0.0
                        for j in range(t2, end):
                            s1 += demand[i, j]
   
                        s2 = 0.0
                        for j in range(t, t2):
                            s2 += demand[i, j]


                        # units that were preproduced
                        q = plan[i,t] - s2*(1-Q[t,i])
                        # if q is positive, a partial lot for t2 to t3 was already produced
                        if (q>0) and (s1>0):
                            Q[t2,i] = q / s1                      
    return X, Q
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


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