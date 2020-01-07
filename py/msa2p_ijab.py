import argparse
import numpy as np
import ccmpred
from multiprocessing import Pool

from math import exp, sqrt

from optimize_felsenstein_faster import optimize_felsenstein, OptimizationFailure

# hard coded alphabet size
A = 20

def create_parser():
    parser = argparse.ArgumentParser('msa2p_ijab')
    parser.add_argument('msa_psicov')
    parser.add_argument('p_ijab_out')
    parser.add_argument('--lambda_w', type=float, default=10)
    parser.add_argument('--branch-length', type=float, default=0.1)
    parser.add_argument('--lbfgs-pgtol', type=float, default=1e-5)
    parser.add_argument('--lbfgs-factr', type=float, default=1e7)
    parser.add_argument('--n-threads', type=int, default=1)
    return parser


def p_ijab_job(msa, i, j, tree, lambda_w, factr, pgtol):
    lambda_w_half = lambda_w / 2
    try:
        v, w = optimize_felsenstein(msa, i, j, tree, lambda_w, factr=factr, pgtol=pgtol)
        p_ijab = calculate_p_ijab(v, w)
    except OptimizationFailure as ex:
        for key, value in ex.info.items():
            print(f'{key}:', value)
        raise
    return p_ijab


def main():
    parser = create_parser()
    args = parser.parse_args()

    msa = ccmpred.io.read_msa_psicov(args.msa_psicov)
    N, L = msa.shape

    lambda_w = args.lambda_w
    pgtol = args.lbfgs_pgtol
    factr = args.lbfgs_factr

    n_mut = args.branch_length
    # building a binary tree with branches of equal, fixed length.
    tree = [((2*i+1, n_mut), (2*i+2, n_mut)) for i in range(N-1)] + [None] * N

    jobs = []
    p_matrix = np.zeros((L, L, A, A))
    with Pool(args.n_threads) as pool:
        for i in range(0, L):
            for j in range(i+1, L):
                job = pool.apply_async(p_ijab_job, args=(msa, i, j, tree, lambda_w, factr, pgtol))
                jobs.append(((i, j), job))
 
        for num, ((i, j), job) in enumerate(jobs):
            p_matrix[i, j] = job.get()
            print(f'finished {num+1}/{len(jobs)}')

    np.save(args.p_ijab_out, p_matrix)


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


if __name__ == '__main__':
    main()