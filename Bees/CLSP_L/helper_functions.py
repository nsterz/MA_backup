from numba import njit
import math

@njit
def rounded(x):
    if x >= 0:
        return math.floor(x + 0.5)
    else:
        return math.ceil(x - 0.5)

