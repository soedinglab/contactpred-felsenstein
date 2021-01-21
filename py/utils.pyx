from libc.math cimport exp
import numpy as np
cimport numpy as np
from scipy.special import logsumexp
import cython


@cython.boundscheck(False)
def seq2ind(np.uint8_t[:] seq, int A=20):
    cdef int L, i
    cdef long ind
    L = len(seq)
    ind = 0
    for i in range(L):
        ind += seq[i] * A**i
    return ind

@cython.boundscheck(False)
@cython.cdivision(True)
def ind2seqarr(long ind, int L, int A=20):
    seq = np.zeros(L, dtype=np.uint8)
    cdef int i
    cdef np.uint8_t[:] seq_view = seq
    for i in range(L):
        seq_view[i] = ind % A
        ind //= A
    return seq


@cython.boundscheck(False)
@cython.cdivision(True)
def calculate_seq_probs(v_arr, w_arr):
    
    cdef long L, A
    cdef long n, i, k, l, seq_idx
    cdef long[:] seq
    cdef double prob
    
    cdef double[:, :] v
    cdef double[:,:,:,:] w
    cdef double[:,:,:,:] pair_probs
    cdef double[:] seq_probs
    
    L, A = v_arr.shape
    pair_probs_arr = np.zeros((L, L, A, A))
    seq_probs_arr = np.zeros(A**L)

    pair_probs = pair_probs_arr
    seq_probs = seq_probs_arr
    
    v = v_arr
    w = w_arr
    
    seq = np.zeros(L, dtype=int)
    

    for seq_idx in range(A**L):
        n = seq_idx
        for i in range(L):
            seq[i] = n % A
            n //= A

        prob = 0
        for k in range(L):
            prob += v[k, seq[k]]

        for k in range(L):
            for l in range(k+1, L):
                prob += w[k, l, seq[k], seq[l]]
        
        seq_probs[seq_idx] = prob  # store probs in logspace
        prob = exp(prob)
       
        for k in range(L):
            for l in range(k+1, L):
                pair_probs[k, l, seq[k], seq[l]] += prob

    pair_probs_arr = pair_probs_arr / (pair_probs_arr.sum(axis=(2,3))[:, :, None, None])
    seq_probs_arr -= logsumexp(seq_probs_arr)
    return seq_probs_arr, pair_probs_arr

