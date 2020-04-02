cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from scipy.optimize import fmin_l_bfgs_b
from math import exp

ctypedef np.float32_t c_float_t
cdef extern from "felsenstein_logspace_float_log2.c":
    pass
    
A = 20
cdef extern from "felsenstein_float_log2.h":
    
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
    
    void initialize_constants(Constants* consts)
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
    cdef object msa
    
    def __cinit__(self, msa, i, j, lam, tree):
        N, L = msa.shape
        self.lam = lam
        
        n_nodes = count_nodes(tree)
        cdef Node* nodes = <Node*> malloc(sizeof(Node)*n_nodes)
        map_tree_py_to_c(tree, nodes)
        self.tree_nodes = nodes

        cdef Constants consts = Constants()
        consts.L = L
        consts.phylo_tree = &nodes[0]
        cdef np.uint8_t[:] my_msa = msa.ravel()
        consts.msa = &my_msa[0]
        consts.i = i
        consts.j = j
 
        initialize_constants(&consts)
        self.consts = consts
        
        cdef Buffer buffer = Buffer()
        cdef NodeBuffer* buffer_left = <NodeBuffer*> malloc(sizeof(NodeBuffer))
        initialize_buffer(buffer_left, &consts)
        buffer.left = buffer_left
        cdef NodeBuffer* buffer_right = <NodeBuffer*> malloc(sizeof(NodeBuffer))
        initialize_buffer(buffer_right, &consts)
        buffer.right = buffer_right
        self.buffer = buffer
        
    def __dealloc__(self):
        free(self.tree_nodes)
        free(self.buffer.left)
        free(self.buffer.right)
    
    
def optimize_felsenstein(msa, i, j, tree, lambda_w=10):
    A = 20
    N, L = msa.shape
    np.random.seed(42)
    x0 = np.random.rand(2*A + A*A)
    
    extra_args = ExtraArguments(msa, i, j, lambda_w, tree)
    x_opt, fx_opt, info = fmin_l_bfgs_b(felsenstein_fx_grad, x0, args=(extra_args,), factr=100, pgtol=1e-5)
    #x_opt, fx_opt, info = fmin_l_bfgs_b(felsenstein_fx_grad, x0, args=(extra_args,))
    info['fx_opt'] = fx_opt
    return x_opt[:2*A], x_opt[2*A:], info


def felsenstein_fx_grad(c_float_t[:] x, ExtraArguments extra_args):
    cdef Constants* consts = &extra_args.consts
    cdef Buffer* buffer = &extra_args.buffer
    cdef c_float_t lam = extra_args.lam
    grad = np.empty(2*A + A*A, dtype=np.float32)
    cdef c_float_t[:] grad_c = grad
    fx = -calculate_fx_grad(&x[0], &grad_c[0], consts, buffer)
    grad = -grad
    cdef c_float_t penalty = 0
    cdef c_float_t w
    
    cdef int i
    
    for i in range(A*A):
        w = x[2*A + i]
        penalty += 0.5 * lam * (w*w)
        grad[2*A + i] += lam * w
    
    return fx + penalty, grad


def evaluate_felsenstein(x0, msa, i, j, tree, lam_w):
    extra_args = ExtraArguments(msa, i, j, lam_w, tree)
    return felsenstein_fx_grad(x0, extra_args)
