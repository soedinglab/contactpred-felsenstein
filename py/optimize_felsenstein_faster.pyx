
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from scipy.optimize import fmin_l_bfgs_b
ctypedef np.float64_t c_float_t

cdef extern from "felsenstein_logspace_faster.c":
    pass

A = 20  # the alphabet is hard-coded to have 20 letters for now
log0 = -1000  # for numerical reasons we assume that the logarithm of 0 is -1000

cdef extern from "felsenstein.h":
    
    ctypedef struct NodePrecomputation:
        pass
    
    ctypedef struct Node:
        Node* left
        Node* right

        int seq_id
        c_float_t phi_left
        c_float_t phi_right
        
        NodePrecomputation* data
        
    ctypedef struct NodeBuffer:
        pass
    
    ctypedef struct Buffer:
        NodeBuffer* left
        NodeBuffer* right
    
    ctypedef struct Constants:
        int L
        Node* phylo_tree
        np.uint8_t* msa
        int i
        int j
        int A_a
        int A_b
        int A_a_p_A_b
        int AA_ab
    
    cdef void initialize_constants(Constants* consts)
    cdef c_float_t calculate_fx_grad(c_float_t* x, c_float_t* grad, Constants* consts, Buffer* buf)
    cdef void initialize_buffer(NodeBuffer* buffer, Constants* consts)


cdef class ExtraArguments:
    
    cdef Constants consts
    cdef Buffer buffer
    cdef c_float_t lam
    cdef int n_nodes
    cdef Node* tree_nodes
    cdef object backmapping_i
    cdef object backmapping_j
    cdef object msa
    
    def __cinit__(self, msa, i, j, lam, node_info):
        
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
        n_nodes = len(node_info)
        cdef Node* nodes = <Node*> malloc(sizeof(Node)*n_nodes)
        
        leaf_no = 0
        inner_no = -1
        for node_idx, connectivity in enumerate(node_info):
            if connectivity is None:
                # this is a leaf
                nodes[node_idx].left = NULL
                nodes[node_idx].right = NULL
                nodes[node_idx].seq_id = leaf_no
                leaf_no += 1
            else:
                (left_node, left_time), (right_node, right_time) = connectivity
                if left_node is not None:
                    nodes[node_idx].left = &nodes[left_node]
                    nodes[node_idx].phi_left = np.exp(-left_time)
                else:
                    nodes[node_idx].left = NULL
                
                if right_node is not None:
                    nodes[node_idx].right = &nodes[right_node]
                    nodes[node_idx].phi_right = np.exp(-right_time)
                else:
                    nodes[node_idx].right = NULL
                nodes[node_idx].seq_id = inner_no
                inner_no -= 1
        self.tree_nodes = nodes


        # create constant object to be passed to the objective function
        cdef Constants consts = Constants()
        consts.L = 2
        consts.phylo_tree = &nodes[0]
        cdef np.uint8_t[:] my_msa = msa_new.ravel()
        consts.msa = &my_msa[0]
        consts.i = 0
        consts.j = 1
        consts.A_a = A_i
        consts.A_b = A_j
        consts.A_a_p_A_b = A_i + A_j
        consts.AA_ab = A_i * A_j
        
        initialize_constants(&consts)
        self.consts = consts
        
        # preallocate buffer for storing temporary results
        cdef Buffer buffer = Buffer()
        cdef NodeBuffer* buffer_left = <NodeBuffer*> malloc(sizeof(NodeBuffer))
        initialize_buffer(buffer_left, &consts)
        buffer.left = buffer_left
        cdef NodeBuffer* buffer_right = <NodeBuffer*> malloc(sizeof(NodeBuffer))
        initialize_buffer(buffer_right, &consts)
        buffer.right = buffer_right
        self.buffer = buffer
        
    def __dealloc__(self):
        
        for i in range(self.n_nodes):
            free(&self.tree_nodes[i])
        free(self.tree_nodes)
        free(self.buffer.left)
        free(self.buffer.right)


def felsenstein_fx_grad(double[:] x, ExtraArguments extra_args):
    cdef Constants* consts = &extra_args.consts
    AA_ab = extra_args.consts.AA_ab
    A_a_p_A_b = extra_args.consts.A_a_p_A_b
    
    cdef Buffer* buffer = &extra_args.buffer
    cdef c_float_t lam = extra_args.lam
    grad = np.empty(AA_ab + A_a_p_A_b)
    cdef c_float_t[:] grad_c = grad
    fx = -calculate_fx_grad(&x[0], &grad_c[0], consts, buffer)
    grad = -grad
    cdef c_float_t penalty = 0
    cdef c_float_t w
    
    cdef int i
    
    for i in range(AA_ab):
        w = x[A_a_p_A_b + i]
        penalty += 0.5 * lam * (w*w)
        grad[A_a_p_A_b + i] += lam * w
    
    return fx + penalty, grad


class OptimizationFailure(Exception):

    def __init__(self, info_object):
        msg = 'Unexpected optimization problem.'
        super().__init__(msg)
        self._info = info_object

    @property
    def info(self):
        return self._info


def optimize_felsenstein(msa, i, j, tree, lam_w=0, factr=1e7, pgtol=1e-5):
    
    extra_args = ExtraArguments(msa, i, j, lam_w, tree)
    AA_ab = extra_args.consts.AA_ab
    A_a_p_A_b = extra_args.consts.A_a_p_A_b
    
    N, L = msa.shape
    x0 = np.zeros(A_a_p_A_b + AA_ab)
    x_opt, fx_opt, info = fmin_l_bfgs_b(felsenstein_fx_grad, x0, args=(extra_args,), 
                                        factr=factr, pgtol=pgtol)
    info['fx_opt'] = fx_opt

    if info['warnflag'] != 0:
        raise OptimizationFailure(info)

    x_opt_full = reduced2long_params(x_opt, msa, i, j)
    v = x_opt_full[:2*A].reshape(2, A)
    w = x_opt_full[2*A:].reshape(A, A)

    return v, w


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
            w[orig_i, orig_j] = x_w[mapped_i*A_i + mapped_j]
            
    return np.concatenate((v.ravel(), w.ravel()))