cimport numpy as np
import numpy as np
from math import exp
from libc.stdlib cimport malloc, free
from scipy.optimize import fmin_l_bfgs_b
ctypedef float c_float_t


A = 20  # the alphabet is hard-coded to have 20 letters for now
log0 = -1000  # for numerical reasons we assume that the logarithm of 0 is -1000

cdef extern from "felsenstein_logspace_simd.c":
    pass

cdef extern from "felsenstein_simd.h":

    ctypedef struct Node:
        Node* left
        Node* right
        int seq_id
        c_float_t phi_left
        c_float_t phi_right
    
    ctypedef struct Buffer:
        pass
    
    ctypedef struct Constants:
        int L
        Node* phylo_tree
        np.uint8_t* msa
        int i
        int j
        int A_i
        int A_j
        int A_i_p_A_j
        int AA_ij
    
    void initialize_constants(Constants* consts)
    c_float_t calculate_fx_grad(c_float_t* x, c_float_t* grad, Constants* consts, Buffer* buf)
    void initialize_buffer(Buffer* buffer, Constants* consts)
    void deinitialize_buffer(Buffer*)
    void deinitialize_constants(Constants*)


cdef map_tree_py_to_c(node, Node* c_array):
    queue = [node]
    arr_idx = 0
    cdef Node* c_node
    while len(queue) > 0:
        cur_node = queue.pop(0)
        c_node = &c_array[arr_idx]
        c_node.left = NULL
        c_node.right = NULL

        if cur_node.has_left_child:
            queue.append(cur_node.left_child)
            c_node.left = &c_array[arr_idx + len(queue)]
            c_node.phi_left = exp(-cur_node.left_branchlength)
        if cur_node.has_right_child:
            queue.append(cur_node.right_child)
            c_node.right = &c_array[arr_idx + len(queue)]
            c_node.phi_right = exp(-cur_node.right_branchlength)
        if cur_node.is_leaf:
            c_node.seq_id = cur_node.seq_id
        
        arr_idx += 1


def count_nodes(node):
    n_nodes = 0
    queue = [node]
    while len(queue) > 0:
        cur_node = queue.pop(0)
        n_nodes += 1
        if cur_node.has_left_child:
            queue.append(cur_node.left_child)
        if cur_node.has_right_child:
            queue.append(cur_node.right_child)
    return n_nodes


cdef class ExtraArguments:
    
    cdef Constants consts
    cdef Buffer buffer
    cdef c_float_t lam
    cdef int n_nodes
    cdef Node* tree_nodes
    cdef object backmapping_i
    cdef object backmapping_j
    cdef object msa
    
    def __cinit__(self, msa, i, j, lam, tree):
        N, L = msa.shape
        self.lam = lam
        
        # we build a new msa with a reduced alphabet containing only the letters that are present
        # in the individual columns i and j.
        # e.g. a column containing [0, 3, 18] is mapped to an alphabet [0, 1, 2] with alphabet size 3.
        counts_i = np.bincount(msa[:,i], minlength=A)
        counts_j = np.bincount(msa[:,j], minlength=A)
        backmapping_i, = np.where(counts_i != 0)
        backmapping_j, = np.where(counts_j != 0)
        
        self.backmapping_i = backmapping_i
        self.backmapping_j = backmapping_j

        col_0 = 0
        col_1 = 1

        msa_new = np.empty((N, 2), dtype='uint8')
        msa_new[:, col_0] = msa[:, i]
        msa_new[:, col_1] = msa[:, j]
        self.msa = msa_new
        for new_aa, old_aa in enumerate(backmapping_i):
            msa_new[msa_new[:, col_0] == old_aa, col_0] = new_aa 

        for new_aa, old_aa in enumerate(backmapping_j):
            msa_new[msa_new[:, col_1] == old_aa, col_1] = new_aa 
        
        N, L = msa_new.shape
        A_i = len(backmapping_i)
        A_j = len(backmapping_j)
        
        # translate the tree into the native structure of linked Node objects
        n_nodes = count_nodes(tree)
        cdef Node* nodes = <Node*> malloc(sizeof(Node)*n_nodes)
        map_tree_py_to_c(tree, nodes)

        # create constant object to be passed to the objective function
        cdef Constants consts = Constants()
        consts.L = 2
        consts.phylo_tree = &nodes[0]
        cdef np.uint8_t[:] my_msa = msa_new.ravel()
        consts.msa = &my_msa[0]
        consts.i = 0
        consts.j = 1
        consts.A_i = A_i
        consts.A_j = A_j
        
        initialize_constants(&consts)
        self.consts = consts
        
        # preallocate buffer for storing temporary results
        cdef Buffer buffer = Buffer()
        initialize_buffer(&buffer, &consts)
        self.buffer = buffer
        
    def __dealloc__(self):
        free(self.tree_nodes)
        deinitialize_constants(&self.consts)
        deinitialize_buffer(&self.buffer)


def felsenstein_fx_grad(double[:] x, ExtraArguments extra_args):
    cdef Constants* consts = &extra_args.consts
    cdef int AA_ij = extra_args.consts.AA_ij
    cdef int A_i_p_A_j = extra_args.consts.A_i_p_A_j

    x_np = np.asarray(x, dtype=np.float64)
    cdef c_float_t[:] x_float = x_np.astype(np.float32)

    cdef Buffer* buffer = &extra_args.buffer
    cdef c_float_t lam = extra_args.lam
    grad = np.empty(AA_ij + A_i_p_A_j, dtype=np.float32)
    cdef c_float_t[:] grad_c = grad
    fx = -calculate_fx_grad(&x_float[0], &grad_c[0], consts, buffer)
    grad = -grad
    cdef c_float_t penalty = 0
    cdef c_float_t w

    cdef int i
    for i in range(AA_ij):
        w = x_float[A_i_p_A_j + i]
        penalty += 0.5 * lam * (w*w)
        grad[A_i_p_A_j + i] += lam * w
    
    return fx + penalty, grad.astype(np.float, casting='safe')


class OptimizationFailure(Exception):

    def __init__(self, info_object, last_x):
        msg = 'Unexpected optimization problem.'
        super().__init__(msg)
        self._info = info_object
        self._last_x = last_x

    @property
    def info(self):
        return self._info

    @property
    def last_x(self):
        return self._last_x


def optimize_felsenstein(msa, i, j, tree, lam_w=0, factr=1e7, pgtol=1e-5, x0=None):

    extra_args = ExtraArguments(msa, i, j, lam_w, tree)
    AA_ij = extra_args.consts.AA_ij
    A_i_p_A_j = extra_args.consts.A_i_p_A_j
    
    N, L = msa.shape
    if x0 is None:
        x0 = np.zeros(A_i_p_A_j + AA_ij)
    x_opt, fx_opt, info = fmin_l_bfgs_b(felsenstein_fx_grad, x0, args=(extra_args,), 
                                        factr=factr, pgtol=pgtol)
    info['fx_opt'] = fx_opt

    if info['warnflag'] != 0:
        raise OptimizationFailure(info, x_opt)

    x_opt_full = reduced2long_params(x_opt, msa, i, j)
    v = x_opt_full[:2*A].reshape(2, A)
    w = x_opt_full[2*A:].reshape(A, A)

    return v, w, info


def reduced2long_params(x, msa, i, j):
    counts_i = np.bincount(msa[:,i], minlength=A)
    counts_j = np.bincount(msa[:,j], minlength=A)
    backmapping_i, = np.where(counts_i != 0)
    backmapping_j, = np.where(counts_j != 0)

    A_i = len(backmapping_i)
    A_j = len(backmapping_j)
    
    v = np.ones((2, A)) * log0
    w = np.zeros((A, A))
    
    x_v = x[:A_i + A_j]
    x_w = x[A_i + A_j:]
    
    for mapped_i, orig_i in enumerate(backmapping_i):
        v[0, orig_i] = x_v[mapped_i]
    for mapped_j, orig_j in enumerate(backmapping_j):
        v[1, orig_j] = x_v[A_i + mapped_j]
    
    for mapped_i in range(A_i):
        for mapped_j in range(A_j):
            orig_i = backmapping_i[mapped_i]
            orig_j = backmapping_j[mapped_j]
            w[orig_i, orig_j] = x_w[mapped_i*A_j + mapped_j]
            
    return np.concatenate((v.ravel(), w.ravel()))