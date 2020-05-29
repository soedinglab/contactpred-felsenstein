import argparse
import math
import sys
from multiprocessing import Pool
from datetime import datetime

import numpy as np
import ccmpred

from tree_utils import create_binary_tree, read_newick_tree, create_seq_node_map, prune_tree

np.random.seed(42)


# hard coded alphabet size
A = 20
GAP_STATE = A


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


def n_ijab_job(msa, i, j, tree, n_seqs, lambda_w, factr, pgtol, max_ls_steps, max_tries, skip_n_calc, x0):

    if n_seqs < 3:
        dummy_info = {}
        dummy_info['n_params'] = 0
        dummy_info['total_fun_calls'] = 0
        dummy_info['grad_norm'] = np.nan
        dummy_info['n_tries'] = 0
        dummy_info['n_seqs'] = n_seqs

        return None, None, None, None, None, dummy_info, dummy_info

    v, w, info_vw = optimize_vw(msa, i, j, tree, lambda_w, factr, pgtol, max_ls_steps, max_tries, x0)
    info_vw['n_seqs'] = n_seqs
    if not skip_n_calc:
        lambda_w_half = lambda_w/2
        v_p, w_p, info_vw_p = optimize_vw(msa, i, j, tree, lambda_w_half, factr, pgtol, max_ls_steps, max_tries, x0)
        info_vw_p['n_seqs'] = n_seqs
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

    seq_node_map = create_seq_node_map(tree)

    jobs = []
    with Pool(args.n_threads, initializer=pool_initializer(args.fs_impl)) as pool:
        for i in range(0, L):
            for j in range(i+1, L):

                gap_mask = (msa[:, i] == GAP_STATE) | (msa[:, j] == GAP_STATE)
                for k in range(N):
                    seq_node_map[k].deleted = gap_mask[k]
                pair_tree = prune_tree(tree)

                n_seqs = np.sum(~gap_mask)

                job = pool.apply_async(
                    n_ijab_job,
                    args=(
                        msa, i, j, pair_tree, n_seqs, lambda_w, factr, pgtol,
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
            n_seqs = info1['n_seqs']
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
                    f' n_tries: {n_tries1}|{n_tries2}, n_params: {n_params}, n_seqs: {n_seqs}]',
                    sep='', flush=True
                )
            else:
                 print(
                    f'{finish_time} finished {num+1}/{len(jobs)} ',
                    f'[fun_evals: {n_eval1},',
                    f' grad_norms: {grad_norm1:.2e},',
                    f' n_tries: {n_tries1}, n_params: {n_params}, n_seqs: {n_seqs}]',
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


if __name__ == '__main__':
    main()
