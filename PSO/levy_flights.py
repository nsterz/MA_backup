from numba import njit
import math
import numpy as np

@njit
def estimate_mean_abs_levy(alpha, samples):
    beta = alpha
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))) ** (1 / beta)
    sum_abs_L = 0.0
    for _ in range(samples):
        # Note: numba supports np.random.rand, but for normal you may need to write
        # a Box–Muller transform manually if true jitting is required.
        u = np.random.normal(0.0, sigma_u)
        v = np.random.normal(0.0, 1.0)
        L = u / (abs(v) ** (1 / beta))
        sum_abs_L += abs(L)
    return sum_abs_L / samples





@njit
def levy_step(alpha):
    beta = alpha
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0.0, sigma_u)
    v = np.random.normal(0.0, 1.0)
    return u / (abs(v) ** (1 / beta))

@njit
def levy_bit_flip(X, share, alpha):
    """
    Flip a Levy‐distributed number of bits, scaled so that on average 
    'share' fraction of the bits are flipped.

    Parameters:
    -----------
    X     : 2D array of shape (m, t), containing 0/1 entries.
    share : float in (0, 1], the target fraction of bits to flip on average.
    alpha : float > 0, the Levy exponent.

    Returns:
    --------
    X (modified in place) with bits flipped.
    """
    m, t = X.shape
    n = m * t

    # 1. Draw a Levy step L ≥ 0
    L = abs(levy_step(alpha))

    # 2. Compute number of bits to flip:
    #    share*n*L  = desired flip count before rounding
    #    ceil(...)  = integer count; clamp to n
    k = math.ceil(share * n * L)
    if k > n:
        k = n

    # 3. Flatten X, flip k random bits, then reshape
    flat = X.ravel().copy()
    for _ in range(k):
        idx = np.random.randint(0, n)
        flat[idx] = 1.0 - flat[idx]

    # 4. Write back into X
    for i in range(m):
        for j in range(t):
            X[i, j] = flat[i * t + j]

    return X




@njit
def levy_jump_Q(Q, Sq, Lqmax, alpha):
    T, M = Q.shape
    out = Q.copy()
    beta = alpha
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))) ** (1 / beta)

    for i in range(T):
        for j in range(M):
            # draw Lévy step
            u = math.sqrt(-2.0 * math.log(np.random.rand())) * math.cos(2 * math.pi * np.random.rand())
            v = math.sqrt(-2.0 * math.log(np.random.rand())) * math.cos(2 * math.pi * np.random.rand())
            L = u * sigma_u / (abs(v) ** (1 / beta))
            # clip step
            if L > Lqmax:
                L = Lqmax
            elif L < -Lqmax:
                L = -Lqmax

            # jump around current (or pbest) Q
            val = out[i, j] + Sq * L
            # clamp to [0,1]
            if val < -1.0:
                val = -1.0
            elif val > 1.0:
                val = 1.0

            out[i, j] = val

    return out
