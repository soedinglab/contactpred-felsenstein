import argparse
import numpy as np
import ccmpred
import pam_distance

A = 20


def create_parser():
    parser = argparse.ArgumentParser('msa2distmatrix')
    parser.add_argument('msa_psicov')
    parser.add_argument('dist_matrix')
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    msa = ccmpred.io.read_msa_psicov(args.msa_psicov)
    single_counts = ccmpred.counts.single_counts(msa)[:, :A] + 1e-9
    single_counts /= single_counts.sum(axis=1)[:, None]
    v = np.log(single_counts)

    N, L = msa.shape
    dist_matrix = np.empty((N, N))
    dist_matrix.fill(np.nan)

    for i in range(N):
        for j in range(0, i):
            dist_matrix[i, j] = pam_distance.optimize_t(msa[i], msa[j], v)

    np.save(args.dist_matrix, dist_matrix)


if __name__ == '__main__':
    main()