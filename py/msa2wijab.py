import argparse
import copy
import math
from math import exp, sqrt
from multiprocessing import Pool

import numpy as np
import ccmpred
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
import pam_distance


try:
    from optimize_felsenstein_simd import optimize_felsenstein, OptimizationFailure
    print('--- using simd optimization')
except ImportError:
    from optimize_felsenstein_faster import optimize_felsenstein, OptimizationFailure
    print('--- running without simd optimization')

# hard coded alphabet size
A = 20

def create_parser():
    parser = argparse.ArgumentParser('msa2wijab')
    parser.add_argument('msa_psicov')
    parser.add_argument('w_ijab_out')
    parser.add_argument('--estimate-nj-tree', action='store_true')
    parser.add_argument('--lambda_w', type=float, default=10)
    parser.add_argument('--branch-length', type=float, default=0.1)
    parser.add_argument('--lbfgs-pgtol', type=float, default=1e-3)
    parser.add_argument('--lbfgs-factr', type=float, default=1e7)
    parser.add_argument('--n-threads', type=int, default=1)
    parser.add_argument('--n-tries', type=int, default=3)
    parser.add_argument('--debug', action='store_true')
    return parser


def w_ijab_job(msa, i, j, tree, lambda_w, factr, pgtol, max_tries):

    n_tries = max_tries
    x0 = None
    while n_tries > 0:
        n_tries -= 1
        try:
            v, w, info = optimize_felsenstein(msa, i, j, tree, lambda_w, factr=factr, pgtol=pgtol, x0=x0)
            break
        except OptimizationFailure as ex:
            x0 = ex.last_x + np.random.normal(0, pgtol, len(ex.last_x))
    else:
        raise Exception(f'Failed optimization with lambda_w={lambda_w} after {max_tries} tries.')

    return v, w, info


def main():
    parser = create_parser()
    args = parser.parse_args()

    msa = ccmpred.io.read_msa_psicov(args.msa_psicov)
    N, L = msa.shape

    lambda_w = args.lambda_w
    pgtol = args.lbfgs_pgtol
    factr = args.lbfgs_factr
    n_tries = args.n_tries

    if args.estimate_nj_tree:
        tree = estimate_nj_tree(msa)
    else:
        tree = create_binary_tree(N, args.branch_length) 

    w_full = np.zeros((L, L, A, A))

    jobs = []
    with Pool(args.n_threads) as pool:
        for i in range(0, L):
            for j in range(i+1, L):
                job = pool.apply_async(w_ijab_job, args=(msa, i, j, tree, lambda_w, factr, pgtol, n_tries))
                jobs.append(((i, j), job))

        for num, ((i, j), job) in enumerate(jobs):
            v, w, info = job.get()
            if args.debug:
                for key, value in info.items():
                    print(f'{key:20s}|', value)
            w_full[i, j] = w
            print(f'finished {num+1}/{len(jobs)}')

    """ unparallelized version
    for i in range(0, L):
        for j in range(i+1, L):
            v, w = w_ijab_job(msa, i, j, tree, lambda_w, factr, pgtol, n_tries)
            w_full[i, j] = w
    """

    np.save(args.w_ijab_out, w_full)


class Node:
    def __init__(self):
        self._parent = None
        self._left_child = None
        self._right_child = None
        self._deleted = False
        self._left_branchlength = None
        self._right_branchlength = None

    @property
    def parent(self):
        return self._parent
    
    @parent.setter
    def parent(self, parent):
        self._parent = parent
    

    @property
    def has_left_child(self):
        left_child = self._left_child
        if left_child is None:
            return False
        else:
            return False if left_child._deleted else True

    @property
    def left_child(self):
        left_child = self._left_child
        if left_child is None:
            return None
        else:
            return None if left_child._deleted else left_child
    
    @left_child.setter
    def left_child(self, left_child):
        self._left_child = left_child

    @property
    def has_right_child(self):
        right_child = self.right_child
        if right_child is None:
            return False
        else:
            return False if right_child._deleted else True

    @property
    def right_child(self):
        right_child = self._right_child
        if right_child is None:
            return None
        else:
            return None if right_child._deleted else right_child
    
    @right_child.setter
    def right_child(self, right_child):
        self._right_child = right_child
        
    @property
    def deleted(self):
        return self._deleted
    
    @deleted.setter
    def deleted(self, deleted):
        self._deleted = deleted
        
    @property
    def is_leaf(self):
        return self._left_child is None and self._right_child is None
    
    @property
    def left_branchlength(self):
        return self._left_branchlength
    
    @left_branchlength.setter
    def left_branchlength(self, left_branchlength):
        self._left_branchlength = left_branchlength
    
    @property
    def right_branchlength(self):
        return self._right_branchlength
    
    @right_branchlength.setter
    def right_branchlength(self, right_branchlength):
        self._right_branchlength = right_branchlength


def prune_tree_helper(node):
    
    has_left_child = node.left_child is not None
    has_right_child = node.right_child is not None
        
    if has_left_child:
        prune_tree_helper(node.left_child)
    if has_right_child:
        prune_tree_helper(node.right_child)
        
    has_left_child = node.left_child is not None
    has_right_child = node.right_child is not None
    
    parent = node.parent
        
    if has_left_child ^ has_right_child:
        if has_left_child:
            child = node.left_child
            branch_length = node.left_branch_length
        else:
            child = node.right_child
            branch_length = node.right_branch_length

        if parent.left_child == node:
            parent.left_child = child
            parent.left_branch_length = parent.left_branch_length + branch_length
        else:
            parent.right_child = child
            parent.right_branch_length = parent.right_branch_length + branch_length
        child.parent = node.parent
        
    if not node.is_leaf and not (has_left_child or has_right_child):
        node.deleted = True


def prune_tree(node):
    node = copy.deepcopy(node)
    prune_tree_helper(node)
    
    if node.left_child is None:
        root = node.right_child
    elif node.right_child is None:
        root = node.left_child
    else:
        root = node
    return root


def create_binary_tree(n_leaves, branchlength):
    last_layer = []
    depth = math.ceil(math.log2(n_leaves)) + 1
    for i in range(2**(depth - 1)):
        leaf = Node()
        leaf.seq_id = i
        if i >= n_leaves:
            leaf.deleted = True
        last_layer.append(leaf)

    for layer in range(depth - 2, -1, -1):
        new_layer = []
        for i in range(2**layer):
            node = Node()

            left_child = last_layer[2*i]
            node.left_child = left_child
            left_child.parent = node
            node.left_branchlength = branchlength
            
            right_child = last_layer[2*i + 1]
            right_child.parent = node
            node.right_child = right_child
            node.right_branchlength = branchlength
            
            new_layer.append(node)
        last_layer = new_layer
    
    root, = last_layer
    root.parent = root
    return prune_tree(root)


def estimate_nj_tree(msa):

    single_counts = ccmpred.counts.single_counts(msa)[:,:A] + 1e-9
    single_counts /= single_counts.sum(axis=1)[:, None]
    v = np.log(single_counts)

    N, L = msa.shape
    dist_list = []
    for i in range(N):
        row_distances = []
        for j in range(0, i):
            row_distances.append(pam_distance.optimize_t(msa[i], msa[j], v))
        row_distances.append(0) # distance of i to itself
        dist_list.append(row_distances)
    
    leaf_names = [str(i) for i in range(N)]
    
    tc = DistanceTreeConstructor()
    dm = DistanceMatrix(leaf_names, dist_list)
    tree = tc.nj(dm)
    tree.root_at_midpoint()
    
    return biopython_phylo_to_tree(tree)


def biopython_phylo_to_tree(phylo_tree):
    root = Node()
    root.parent = root
    tree_queue = [root]
    phylo_queue = [phylo_tree.clade]

    while len(phylo_queue) > 0:
        phylo_node = phylo_queue.pop(0)
        tree_node = tree_queue.pop(0)

        if len(phylo_node.clades) == 0:
            # this is a leaf
            tree_node.seq_id = int(phylo_node.name)
            continue

        left_clade, right_clade = phylo_node.clades

        left_child = Node()
        left_child.parent = tree_node
        tree_node.left_child = left_child
        tree_node.left_branchlength = max(left_clade.branch_length, 0)

        right_child = Node()
        right_child.parent = tree_node
        tree_node.right_child = right_child
        tree_node.right_branchlength = max(right_clade.branch_length, 0)

        tree_queue.append(left_child)
        tree_queue.append(right_child)

        phylo_queue.append(left_clade)
        phylo_queue.append(right_clade)

    return root


if __name__ == '__main__':
    main()
