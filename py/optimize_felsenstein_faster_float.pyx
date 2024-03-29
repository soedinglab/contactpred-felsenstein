
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from scipy.optimize import fmin_l_bfgs_b
ctypedef np.float32_t c_float_t
from math import exp

cdef extern from "felsenstein_logspace_faster_float.c":
    pass

A = 20  # the alphabet is hard-coded to have 20 letters for now
log0 = -1000  # for numerical reasons we assume that the logarithm of 0 is -1000

cdef extern from "felsenstein_faster_float.h":
    
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
        int A_i
        int A_j
        int A_i_p_A_j
        int AA_ij
    
    cdef void initialize_constants(Constants* consts)
    cdef c_float_t calculate_fx_grad(c_float_t* x, c_float_t* grad, Constants* consts, Buffer* buf)
    cdef void initialize_buffer(NodeBuffer* buffer, Constants* consts)


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
        consts.A_i_p_A_j = A_i + A_j
        consts.AA_ij = A_i * A_j
        
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


def felsenstein_fx_grad(c_float_t[:] x, ExtraArguments extra_args):
    cdef Constants* consts = &extra_args.consts
    AA_ij = extra_args.consts.AA_ij
    A_i_p_A_j = extra_args.consts.A_i_p_A_j
    
    cdef Buffer* buffer = &extra_args.buffer
    cdef c_float_t lam = extra_args.lam
    grad = np.empty(AA_ij + A_i_p_A_j, dtype=np.float32)
    cdef c_float_t[:] grad_c = grad
    fx = -calculate_fx_grad(&x[0], &grad_c[0], consts, buffer)
    grad = -grad
    cdef c_float_t penalty = 0
    cdef c_float_t w
    
    cdef int i
    
    for i in range(AA_ij):
        w = x[A_i_p_A_j + i]
        penalty += 0.5 * lam * (w*w)
        grad[A_i_p_A_j + i] += lam * w
    
    return fx + penalty, grad



def get_parameter_length(msa, i, j):
    A_i = len(np.unique(msa[:,i]))
    A_j = len(np.unique(msa[:,j]))
    return A_i+A_j + A_i*A_j


def evaluate_felsenstein(x0, msa, i, j, tree, lam_w):
    extra_args = ExtraArguments(msa, i, j, lam_w, tree)
    return felsenstein_fx_grad(x0, extra_args)