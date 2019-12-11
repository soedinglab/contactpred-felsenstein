cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from scipy.optimize import fmin_l_bfgs_b
ctypedef np.float64_t c_float_t
cdef extern from "felsenstein_logspace.c":
    pass
    
A = 20
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
        int L;
        c_float_t* single_aa_frequencies
        Node* phylo_tree;
        np.uint8_t* msa;
        int i;
        int j;
    
    void initialize_constants(Constants* consts);
    cdef c_float_t calculate_fx_grad(c_float_t* x, c_float_t* grad, Constants* consts, Buffer* buf)
    cdef void initialize_buffer(NodeBuffer* buffer)


cdef class ExtraArguments:
    
    cdef single_aa_frequencies
    cdef Constants consts
    cdef Buffer buffer
    cdef c_float_t lam
    cdef int n_nodes
    cdef Node* tree_nodes
    
    def __cinit__(self, msa, i, j, lam, node_info):
        N, L = msa.shape
        self.lam = lam
        
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


        cdef Constants consts = Constants()
        consts.L = L
        consts.phylo_tree = &nodes[0]
        cdef np.uint8_t[:] my_msa = msa.ravel()
        consts.msa = &my_msa[0]
        consts.i = i
        consts.j = j
        aa_counts = np.bincount(msa.ravel(), minlength=20) + 1e-9
        aa_freqs = np.log(aa_counts / aa_counts.sum())
        self.single_aa_frequencies = aa_freqs
        cdef c_float_t* aa_freqs_c = <c_float_t*> malloc(sizeof(c_float_t)*A)
        for a in range(A):
            aa_freqs_c[a] = aa_freqs[a]
        consts.single_aa_frequencies = aa_freqs_c
        initialize_constants(&consts)
        self.consts = consts
        
        cdef Buffer buffer = Buffer()
        cdef NodeBuffer* buffer_left = <NodeBuffer*> malloc(sizeof(NodeBuffer))
        initialize_buffer(buffer_left)
        buffer.left = buffer_left
        cdef NodeBuffer* buffer_right = <NodeBuffer*> malloc(sizeof(NodeBuffer))
        initialize_buffer(buffer_right)
        buffer.right = buffer_right
        self.buffer = buffer
        
    def __dealloc__(self):
        
        for i in range(self.n_nodes):
            free(&self.tree_nodes[i])
        free(self.tree_nodes)
        free(self.buffer.left)
        free(self.buffer.right)
        free(self.consts.single_aa_frequencies)
    
    
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


def felsenstein_fx_grad(double[:] x, ExtraArguments extra_args):
    cdef Constants* consts = &extra_args.consts
    cdef Buffer* buffer = &extra_args.buffer
    cdef c_float_t lam = extra_args.lam
    grad = np.empty(2*A + A*A)
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
