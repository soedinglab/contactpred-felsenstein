import argparse
import math
import numpy as np
import ccmpred


A = 20


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('v_npy')
    parser.add_argument('w_npy')
    parser.add_argument('--aln-file')
    parser.add_argument('N', type=float)
    parser.add_argument('lambda_w', type=float)
    parser.add_argument('n_ijab_out')
    return parser


def calculate_nijab(v_ij, w_ij, lambda_w, n_ij):
    p_ijab = calculate_p_ijab(v_ij, w_ij)
    n_ijab = n_ij * p_ijab + lambda_w * w_ij
    return n_ijab


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


def main():
    parser = create_parser()
    args = parser.parse_args()

    v = np.load(args.v_npy)
    w = np.load(args.w_npy)

    L, L, A, A = w.shape

    if not args.aln_file:
        no_gap_frac = np.ones((L, L))
    else:
        msa = ccmpred.io.read_msa_psicov(args.aln_file)
        N, L = msa.shape
        pair_counts = ccmpred.counts.pair_counts(msa)[:, :, :A, :A]
        no_gap_frac = pair_counts.sum(axis=(2, 3)) / N
    n_ij = args.N * no_gap_frac
   
    n = np.zeros(w.shape)
    for i in range(L):
        for j in range(i+1, L):
            n_ab = calculate_nijab(v[i, j], w[i, j], args.lambda_w, n_ij[i, j])
            n[i, j] = n_ab
            n[j, i] = n_ab.T

    np.save(args.n_ijab_out, n)


if __name__ == '__main__':
    main()
