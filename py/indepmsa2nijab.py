import argparse
import numpy as np
from ccmpred.io import read_msa_psicov
from ccmpred.counts import pair_counts


def create_parser():
    parser = argparse.ArgumentParser('indepmsa2nijab')
    parser.add_argument('msa')
    parser.add_argument('out_n_ijab')
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    msa = read_msa_psicov(args.msa)
    N, L = msa.shape
    n = pair_counts(msa)
    diag_ind = np.diag_indices(L)
    n[diag_ind] = np.nan

    np.save(args.out_n_ijab, n)


if __name__ == '__main__':
    main()
