import argparse
import copy
import math
import sys
from multiprocessing import Pool
from datetime import datetime

import numpy as np
import ccmpred
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
from Bio import Phylo
import pam_distance

np.random.seed(42)


# hard coded alphabet size
A = 20


def create_parser():
    parser = argparse.ArgumentParser('msa2nijab')
    parser.add_argument('msa_psicov')
    parser.add_argument('n_ijab_out')
    parser.add_argument('--mode', choices=['NJ_TREE', 'BIN_TREE', 'NEWICK_TREE'],
                        default='BIN_TREE')
    parser.add_argument('--newick-file')
    parser.add_argument('--lambda_w', type=float, default=10)
    parser.add_argument('--branch-length', type=float, default=0.1)
    parser.add_argument('--lbfgs-pgtol', type=float, default=1e-5)
    parser.add_argument('--lbfgs-factr', type=float, default=1000)
    parser.add_argument('--lbfgs-maxls', type=int, default=20)
    parser.add_argument('--n-threads', type=int, default=1)
    parser.add_argument('--n-tries', type=int, default=3)
    parser.add_argument('--debug', action='store_true', help='deprecated and removed')
    parser.add_argument('--skip-n-calc', action='store_true')
    parser.add_argument('--w_ijab_out')
    parser.add_argument('--w_ijab_prime_out')
    parser.add_argument('--v_ijab_out')
    parser.add_argument('--v_ijab_prime_out')
    parser.add_argument('--fs-impl', choices=['SIMD', 'RED_ALPH'], default='SIMD')
    parser.add_argument('--x-init', choices=['INIT_V', 'INIT_ZERO'], default='INIT_V')
    return parser


def pool_initializer(fs_impl):
    global optimize_felsenstein
    global OptimizationFailure
    if fs_impl == 'SIMD':
        import optimize_felsenstein_simd as fs
    elif fs_impl == 'RED_ALPH':
        import optimize_felsenstein_faster as fs
    optimize_felsenstein = fs.optimize_felsenstein
    OptimizationFailure = fs.OptimizationFailure


def optimize_vw(msa, i, j, tree, lambda_w, factr, pgtol, max_ls_steps, max_tries, x0):
    n_fun_eval = 0
    grad_norm = np.inf
    best_v = None
    best_w = None
    info = {}
    n_tries = max_tries
    while n_tries > 0:
        n_tries -= 1
        try:
            v, w, opt_info = optimize_felsenstein(
                msa, i, j, tree, lambda_w, factr=factr, pgtol=pgtol, max_ls_steps=max_ls_steps, x0=x0
            )
            best_v, best_w = v, w
            grad_norm = np.linalg.norm(opt_info['grad'])
            n_fun_eval += opt_info['funcalls']
            break
        except OptimizationFailure as ex:
            opt_info = ex.info
            n_fun_eval += opt_info['funcalls']
            grad_norm_try = np.linalg.norm(opt_info['grad'])
            if grad_norm_try < grad_norm:
                best_v, best_w = ex.v_opt, ex.w_opt
            x0 = ex.last_x + np.random.normal(0, 1e-1, len(ex.last_x))
    info['total_fun_calls'] = n_fun_eval
    info['grad_norm'] = grad_norm
    info['n_tries'] = max_tries - n_tries
    info['n_params'] = len(opt_info['grad'])
    return best_v, best_w, info


def nan_like(arr):
    nan_arr = np.empty(arr.shape)
    nan_arr.fill(np.nan)
    return nan_arr


def n_ijab_job(msa, i, j, tree, lambda_w, factr, pgtol, max_ls_steps, max_tries, skip_n_calc, x0):

    v, w, info_vw = optimize_vw(msa, i, j, tree, lambda_w, factr, pgtol, max_ls_steps, max_tries, x0)

    if not skip_n_calc:
        lambda_w_half = lambda_w/2
        v_p, w_p, info_vw_p = optimize_vw(msa, i, j, tree, lambda_w_half, factr, pgtol, max_ls_steps, max_tries, x0)

        try:
            N_ij = calculate_nij(v, v_p, w, w_p, lambda_w, lambda_w_half)
            n_ijab = calculate_nijab(v, w, lambda_w, N_ij)
        except OverflowError:
            n_ijab = nan_like(w)
    else:
        v_p = nan_like(v)
        w_p = nan_like(w)
        n_ijab = nan_like(w)
        info_vw_p = None

    return n_ijab, v, w, v_p, w_p, info_vw, info_vw_p


def calculate_p_ijab(v, w):
    p_ijab = np.empty((A, A))
    norm = 0
    for a in range(A):
        for b in range(A):
            p_ijab[a, b] = math.exp(v[0, a] + v[1, b] + w[a, b])
            norm += p_ijab[a, b]
    for a in range(A):
        for b in range(A):
            p_ijab[a, b] /= norm
    return p_ijab


def calculate_nij(v1, v2, w1, w2, lambda_w1, lambda_w2):
    num = 0
    for a in range(A):
        for b in range(A):
            num += (lambda_w1 * w1[a, b] - lambda_w2 * w2[a, b])**2
    denom = 0
    p1 = calculate_p_ijab(v1, w1)
    p2 = calculate_p_ijab(v2, w2)
    for a in range(A):
        for b in range(A):
            denom += (p1[a, b] - p2[a, b])**2
    return math.sqrt(num / denom)


def calculate_nijab(v_ij, w_ij, lambda_w, n_ij):
    p_ijab = calculate_p_ijab(v_ij, w_ij)
    n_ijab = n_ij * p_ijab + lambda_w * w_ij
    return n_ijab


def main():
    parser = create_parser()
    args = parser.parse_args()

    msa = ccmpred.io.read_msa_psicov(args.msa_psicov)
    N, L = msa.shape
    A = 20

    lambda_w = args.lambda_w
    pgtol = args.lbfgs_pgtol
    factr = args.lbfgs_factr
    n_tries = args.n_tries

    if args.mode == 'NJ_TREE':
        print('mode "NJ_TREE" not available anymore. '
              'Pass tree via "NEWICK_TREE" mode instead.', file=sys.stderr)
        sys.exit(1)
    elif args.mode == 'BIN_TREE':
        tree = create_binary_tree(N, args.branch_length)
    elif args.mode == 'NEWICK_TREE':
        if not args.newick_file:
            print('Expected argument --newick-file, but no tree specified.', file=sys.stderr)
            sys.exit(1)
        tree = read_newick_tree(args.newick_file)

    v_full = np.zeros((L, L, 2, A))
    v_prime = np.zeros((L, L, 2, A))
    w_full = np.zeros((L, L, A, A))
    w_prime = np.zeros((L, L, A, A))
    n_full = np.zeros((L, L, A, A))

    jobs = []
    with Pool(args.n_threads, initializer=pool_initializer(args.fs_impl)) as pool:
        for i in range(0, L):
            for j in range(i+1, L):
                job = pool.apply_async(
                    n_ijab_job,
                    args=(
                        msa, i, j, tree, lambda_w, factr, pgtol,
                        args.lbfgs_maxls, n_tries, args.skip_n_calc, args.x_init
                        )
                    )
                jobs.append(((i, j), job))

        for num, ((i, j), job) in enumerate(jobs):
            n, v, w, v_p, w_p, info1, info2 = job.get()
            v_full[i, j] = v
            v_prime[i, j] = v_p
            w_full[i, j] = w
            w_prime[i, j] = w_p
            n_full[i, j] = n
            finish_time = datetime.today().strftime("%Y/%m/%d|%H:%M:%S")

            n_params = info1['n_params']
            n_eval1 = info1['total_fun_calls']
            grad_norm1 = info1['grad_norm']
            n_tries1 = info1['n_tries']

            if not args.skip_n_calc:
                n_eval2 = info2['total_fun_calls']
                grad_norm2 = info2['grad_norm']
                n_tries2 = info2['n_tries']

                print(
                    f'{finish_time} finished {num+1}/{len(jobs)} ',
                    f'[fun_evals: {n_eval1}|{n_eval2},',
                    f' grad_norms: {grad_norm1:.2e}|{grad_norm2:.2e},',
                    f' n_tries: {n_tries1}|{n_tries2}, n_params: {n_params}]',
                    sep='', flush=True
                )
            else:
                 print(
                    f'{finish_time} finished {num+1}/{len(jobs)} ',
                    f'[fun_evals: {n_eval1},',
                    f' grad_norms: {grad_norm1:.2e},',
                    f' n_tries: {n_tries1}, n_params: {n_params}]',
                    sep='', flush=True
                 )

    L, L, A, A = n_full.shape
    for i in range(L):
        for j in range(0, i):
            for a in range(A):
                for b in range(A):
                    n_full[i, j, a, b] = n_full[j, i, b, a]

    np.save(args.n_ijab_out, n_full)
    if args.v_ijab_out:
        np.save(args.v_ijab_out, v_full)
    if args.v_ijab_prime_out:
        np.save(args.v_ijab_prime_out, v_prime)
    if args.w_ijab_out:
        np.save(args.w_ijab_out, w_full)
    if args.w_ijab_prime_out:
        np.save(args.w_ijab_prime_out, w_prime)


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


def read_newick_tree(newick_tree_file):
    with open(newick_tree_file) as newick_handle:
        tree = next(Phylo.NewickIO.parse(newick_handle))
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
