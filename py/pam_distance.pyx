import cython
from libc.math cimport exp
import math
import numpy as np
cimport numpy as np

cdef A = 10

@cython.cdivision(True)
@cython.boundscheck(False)
def calc_t_change(np.uint8_t[:] seq_x, np.uint8_t[:] seq_y, int L, double t, double tau, double[:,:] v):
    cdef double r, num, denom, num_term, denom_term, exp_v, term1, term2, Z, p_i
    cdef int i
    cdef np.uint8_t x_i, y_i
    r = exp(-t)
    num_term = 0
    denom_term = 0
    
    for i in range(L):
        x_i = seq_x[i]
        y_i = seq_y[i]
        if x_i == y_i:
            exp_v = exp(v[i, y_i])
            Z = 0
            for j in range(A):
                Z += exp(v[i, j])
            p_i = exp_v / Z
            term1 = p_i + r * (1 - p_i)
            term2 = p_i + r*r * (1 - p_i)
            
            num_term += 1 / term1
            denom_term += term2 / (term1*term1)

    num = - (1 - r) * (num_term - L) - (1 - r)*(1 - r) / r / tau
    denom = denom_term - L
    
    return num / denom


class NonConvergenceException(RuntimeError):
    pass


def optimize_t(seq1, seq2, v, prec=1e-6, tau=1, alpha=0.9, max_iter=100):
    L = len(seq1)
    t_old = 0.1

    n_iter = 0
    while n_iter < max_iter:
        t_new = t_old - alpha * calc_t_change(seq1, seq2, L, t_old, tau, v)
        if abs(t_new - t_old) < prec:
            break
        t_old = t_new
        n_iter += 1
    else:
        raise NonConvergenceException(f'Optimization did not converge after {max_iter} iterations')

    return t_new
