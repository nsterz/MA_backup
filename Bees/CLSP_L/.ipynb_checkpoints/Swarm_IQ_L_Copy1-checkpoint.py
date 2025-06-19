
from numba import njit
import numpy as np
from helper_functions import rounded



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
                            
                        XPK[i, t] = rounded(s2 + QPK[i, t2] * s1)
                        
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
                        XPK[i, t] = rounded((1-QPK[i, t]) * demand[i, t])
                          
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
                        
                        XPK[i, t] = rounded((1 - QPK[i, t]) * s)

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

                        XPK[i, t] = rounded(s2 + QPK[i, t2] * s1)
                        
                        
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

                        XPK[i, t] = rounded((1-QPK[i, t]) * s2 + QPK[i, t2] * s1)
                    

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

# ------------------------------------------------------------------------------------------------------------------------------- #
@njit
def encode_prod_plan(plan, demand):
    T = plan.shape[1]
    M = plan.shape[0]
    X = np.zeros((M, T))
    Q = np.zeros((M, T)).astype(np.float64)
    # iterate products
    for i in range(M):
        for t in range(T):
            # skip non production periods
            if plan[i, t] == 0:
                X[i,t] = 0
            else:
                X[i,t] = 1
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
                            
                            Q[i, t2] = q / s2

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
                            Q[i, t] = 1 - (plan[i,t] / demand[i,t])
                        
                        
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
                            Q[i, t2] = q / s1
                            
                        
                    # Case 8: production after and before t --> produce in advance t to t2 and cover backorders of t1 
                    else:
                        t3 = T - 1
                        for j in range(t2+1, T):
                            if plan[i, j] > 0:
                                t3 = j
                                break
                        end = t3 + 1 if (plan[i, t3] == 0 or t3 == t2) else t3

                        s1 = 0.0
                        for j in range(t, t2):
                            s1 += demand[i, j]
                            
                        s2 = 0.0
                        for j in range(t2, end):
                            s2 += demand[i, j]
  

                        # units that were preproduced
                        q = plan[i,t] - s1*(1-Q[i,t])
                        # if q is positive, a partial lot for t2 to t3 was already produced
                        if (q>0) and (s2>0):
                            Q[i, t2] = q / s2                      
    return X, Q
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



@njit
def decode_and_evaluate(X, Q, O, demand, setup_costs, production_costs,
                        production_times, setup_times, capacities, inventory_costs, idle_carry=2):
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
    X : 2D ndarray of int
        Binary production decision matrix, shape (M, T).
    Q : 2D ndarray of float
        Preproduction fractions, shape (M, T).
    demand : 2D ndarray of float
        Demand per product per period, shape (M, T).

    Returns
    -------
    result : 1D ndarray of float
        [violations, total_cost]
    """
    T = X.shape[1]
    M = X.shape[0]

    prod_quant = np.zeros((M, T))

    # ------------------------------------------------------------------------------------------------------------------ # 
    # 1. Decode the production plan
    
    for i in range(M):
        for t in range(T):
            # skip non production periods
            if X[i, t] == 0:
                continue
            else:
                
                # Find next production period t2
                t2 = T - 1
                for j in range(t+1, T):
                    if X[i, j] == 1:
                        t2 = j
                        break
                # ---------------------------------------------------------------------------------------------------------- #
                # t = 1
                if t == 0:
                    # Case 1: no production is planned in the future
                    if (t2 == T - 1) and (X[i, t2] == 0):
                        s = 0.0
                        for j in range(t, T):
                            s += demand[i, j]
                        
                        prod_quant[i, t] = s
                        
                    # Case 2:  production is planned in the future
                    else:
                        # find production period after the next production period
                        t3 = T - 1
                        for j in range(t2+1, T):
                            if X[i, j] == 1:
                                t3 = j
                                break

                        end = t3 + 1 if (X[i, t3] == 0 or t3 == t2) else t3
   
                        s1 = 0.0
                        for j in range(t2, end):
                            s1 += demand[i, j]
                            
                            
                        s2 = 0.0                       
                        for j in range(t, t2):
                            s2 += demand[i, j]
                            
                        prod_quant[i, t] = s2 + Q[i, t2] * s1
                        
                # ---------------------------------------------------------------------------------------------------------- #
                # t = T
                elif t == T - 1:
                    
                    # Look backwards for a prior production period t1
                    t1 = 0
                    for j in range(t-1, -1, -1):
                        if X[i, j] == 1:
                            t1 = j
                            break
                            
                    # Case 3: no prior production before the last period 
                    if (t1==0) and (X[i,t1]==0):
                        
                        prod_quant[i, t] = demand[i,t]
                        
                    # Case 4: prior production 
                    else:
                        prod_quant[i, t] = (1-Q[i, t]) * demand[i, t]
                          
                # ----------------------------------------------------------------------------------------------------------#
                # t in [2,T-1]
                else:
                    # Look backwards for a prior production period t1
                    t1 = 0
                    for j in range(t-1, -1, -1):
                        if X[i, j] == 1:
                            t1 = j
                            break

                    # Case 5: no production before AND after t
                    if (X[i,t1]==0) and (X[i,t2]==0):
                        s = 0.0
                        for j in range(t, T):
                            s += demand[i, j]
                        prod_quant[i, t] = s
                        
                        
                    # Case 6: production before but not after t              
                    elif (X[i,t1]==1) and (X[i,t2]==0):
                        s = 0.0
                        for j in range(t, t2+1):
                            s += demand[i, j]
                        
                        prod_quant[i, t] = (1 - Q[i, t]) * s

                    # Case 7: production after but not before t 
                    elif (X[i,t1]==0) and (X[i,t2]==1):
                        t3 = T - 1
                        for j in range(t2+1, T):
                            if X[i, j] == 1:
                                t3 = j
                                break
                                
                        end = t3 + 1 if (X[i, t3] == 0 or t3 == t2) else t3
                        s1 = 0.0
                        for j in range(t2, end):
                            s1 += demand[i, j]
                            
                        s2 = 0.0
                        for j in range(t, t2):
                            s2 += demand[i, j]

                        prod_quant[i, t] = s2 + Q[i, t2] * s1
                        
                        
                    # Case 8: production after and before t  
                    else:
                        t3 = T - 1
                        for j in range(t2+1, T):
                            if X[i, j] == 1:
                                t3 = j
                                break
                        end = t3 + 1 if (X[i, t3] == 0 or t3 == t2) else t3
   
                        s1 = 0.0
                        for j in range(t2, end):
                            s1 += demand[i, j]
   
                        s2 = 0.0
                        for j in range(t, t2):
                            s2 += demand[i, j]

                        prod_quant[i, t] = (1-Q[i, t]) * s2 + Q[i, t2] * s1

            # Clamp negatives
            if prod_quant[i ,t] < 0.0:
                prod_quant[i, t] = 0.0

    # ------------------------------------------------------------------------------------------------------------------ # 
    # 2. Check the setup sequence for feasibility

    
    violations = 0.0
    # no product is setup initially, use negative values
    last_setup = np.full(M, -10_000_000, np.int64)
    

    for t in range(T):
        #print('last setup', last_setup)

        # 1) production without explicit setup
        carried = -1
        max_last = -10_000_000
        # pick carried candidate: most recent within idle window
        for m in range(M):
            if X[m, t] == 1 and O[m, t] == 0:
                
                if t - last_setup[m] <= idle_carry:
                    if last_setup[m] > max_last:
                        max_last = last_setup[m]
                        carried = m

        # 2) apply carry
        if carried >= 0:
            #print(f"Carry-over setup for product {carried} at period {t}")
            last_setup[carried] = t
            # invalidate all other prior setup states when we carry over one state
            for m in range(M):
                if m != carried:
                    last_setup[m] = -10_000_000

        # 3) explicit setups
        did_setup = False
        for m in range(M):
            if O[m, t] == 1:
 
                #print(f"Setup cost for product {k} at period {t}: +{setup_costs[k]}")
                last_setup[m] = t
                did_setup = True

        # 4) reconfiguration: if any explicit setup, lose carry for others
        if did_setup:
            for m in range(M):
                if O[m, t] == 0:
                    last_setup[m] = -10_000_000
                    

        


        # 5) penalties for others
        penalty_occurred = False
        for m in range(M):
            if X[m, t] == 1 and O[m, t] == 0 and m != carried:
                violations +=1
                penalty_occurred = True

        # 6) after any penalty, invalidate all prior setup states
        if penalty_occurred:
            for m in range(M):
                last_setup[m] = -10_000_000
            carried = -1
        #print(last_setup)
   
    
    # ------------------------------------------------------------------------------------------------------------------ # 
    # 3. Determine total costs, backorders and overtime

                
    net_inv = np.zeros((M, T))
    inv = np.zeros((M, T))
    prod_time = np.zeros(T)
    setup_time = np.zeros(T)
    total_cost = 0.0

    # Inventory and violations
    for i in range(M):
        for t in range(T):
            if t == 0:
                net_inv[i, t] = prod_quant[i, t] - demand[i, t]
            else:
                net_inv[i, t] = net_inv[i, t - 1] + prod_quant[i, t] - demand[i, t]

            if net_inv[i, t] >= 0:
                inv[i, t] = net_inv[i, t]
            # count number of backorders
            else:
                violations += -net_inv[i, t]  
                
            # penalize when 0 units are produced (swarm otherwise has no incentive to do this)
            if X[i,t]== 1 and prod_quant[i,t]==0:
                violations +=1
                

    # Cost and time
    for t in range(T):
        for i in range(M):
            total_cost += production_costs[i] * prod_quant[i, t]
            total_cost += setup_costs[i] * O[i, t]
            total_cost += inventory_costs[i] * inv[i, t]
            prod_time[t] += np.ceil(prod_quant[i, t] * production_times[i])
            setup_time[t] += O[i, t] * setup_times[i]

        ot = prod_time[t] + setup_time[t] - capacities[t]
        # count number of overtime units
        if ot > 0:
            violations += ot

    return np.array([violations, total_cost], dtype=np.float64)