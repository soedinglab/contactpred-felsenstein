import argparse
import numpy as np
import ccmpred
from multiprocessing import Pool

from math import exp, sqrt

try:
    from optimize_felsenstein_simd import optimize_felsenstein, OptimizationFailure
    print('--- using simd optimization')
except ImportError:
    from optimize_felsenstein_faster import optimize_felsenstein, OptimizationFailure
    print('--- running without simd optimization')

# hard coded alphabet size
A = 20

def create_parser():
    parser = argparse.ArgumentParser('msa2indep_counts')
    parser.add_argument('msa_psicov')
    parser.add_argument('n_ijab_out')
    parser.add_argument('--lambda_w', type=float, default=10)
    parser.add_argument('--branch-length', type=float, default=0.1)
    parser.add_argument('--lbfgs-pgtol', type=float, default=1e-3)
    parser.add_argument('--lbfgs-factr', type=float, default=1e7)
    parser.add_argument('--n-threads', type=int, default=1)
    parser.add_argument('--n-tries', type=int, default=3)
    parser.add_argument('--w_ijab_out')
    return parser


def n_ijab_job(msa, i, j, tree, lambda_w, factr, pgtol, max_tries):
    lambda_w_half = lambda_w / 2

    n_tries = max_tries
    x0 = None
    while n_tries > 0:
        n_tries -= 1
        try:
            v, w = optimize_felsenstein(msa, i, j, tree, lambda_w, factr=factr, pgtol=pgtol, x0=x0)
            break
        except OptimizationFailure as ex:
            x0 = ex.last_x + np.random.normal(0, pgtol, len(ex.last_x))
    else:
        raise Exception(f'Failed optimization with lambda_w={lambda_w} after {max_tries} tries.')

    n_tries = max_tries
    x0 = None
    while n_tries > 0:
        n_tries -= 1
        try:
            v_p, w_p = optimize_felsenstein(msa, i, j, tree, lambda_w_half, factr=factr, pgtol=pgtol, x0=x0)
            break
        except OptimizationFailure as ex:
            x0 = ex.last_x + np.random.normal(0, pgtol, len(ex.last_x))
    else:
        raise Exception(f'Failed optimization with lambda_w={lambda_w_half} after {max_tries} tries.')

    
    N_ij = calculate_nij(v, v_p, w, w_p, lambda_w, lambda_w_half)
    n_ijab = calculate_nijab(v, w, lambda_w, N_ij)
    return n_ijab, v, w


def main():
    parser = create_parser()
    args = parser.parse_args()

    msa = ccmpred.io.read_msa_psicov(args.msa_psicov)
    N, L = msa.shape

    lambda_w = args.lambda_w
    pgtol = args.lbfgs_pgtol
    factr = args.lbfgs_factr
    n_tries = args.n_tries

    n_mut = args.branch_length
    # building a binary tree with branches of equal, fixed length.
    tree = [((2*i+1, n_mut), (2*i+2, n_mut)) for i in range(N-1)] + [None] * N

    jobs = []
    n_pair = np.zeros((L, L, A, A))
    w_full = np.zeros((L, L, A, A))

    with Pool(args.n_threads) as pool:
        for i in range(0, L):
            for j in range(i+1, L):
                job = pool.apply_async(n_ijab_job, args=(msa, i, j, tree, lambda_w, factr, pgtol, n_tries))
                jobs.append(((i, j), job))

        for num, ((i, j), job) in enumerate(jobs):
            n_ij, v, w = job.get()
            n_pair[i, j] = n_ij
            w_full[i, j] = w
            print(f'finished {num+1}/{len(jobs)}')

    np.save(args.n_ijab_out, n_pair)
    if args.w_ijab_out:
        np.save(args.w_ijab_out, w_full)


def calculate_p_ijab(v, w):
    p_ijab = np.empty((A, A))
    norm = 0
    for a in range(A):
        for b in range(A):
            p_ijab[a, b] = exp(v[0, a] + v[1, b] + w[a, b])
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
    return sqrt(num / denom)


def calculate_nijab(v_ij, w_ij, lambda_w, n_ij):
    p_ijab = calculate_p_ijab(v_ij, w_ij)
    n_ijab = n_ij * p_ijab + lambda_w * w_ij
    return n_ijab


if __name__ == '__main__':
    main()
