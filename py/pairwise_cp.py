import argparse
import numpy as np
import ccmpred
from multiprocessing import Pool

from math import exp, sqrt

from optimize_felsenstein_faster import optimize_felsenstein, OptimizationFailure

# hard coded alphabet size
A = 20

def create_parser():
    parser = argparse.ArgumentParser('pairwise_cp')
    parser.add_argument('w_ijab_file')
    parser.add_argument('n_ijab_file')
    parser.add_argument('msa_psicov')
    parser.add_argument('lambda_w', type=float)
    parser.add_argument('--correction', choices=['None', 'APC', 'EVC'], default='None')
    parser.add_argument('prediction_matrix')
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    msa = ccmpred.io.read_msa_psicov(args.msa_psicov)
    N, L = msa.shape

    lambda_w = args.lambda_w
    w = np.load(args.w_ijab_file)
    n_ijab = np.load(args.n_ijab_file)

    for i in range(L):
        for j in range(i+1, L):
            for a in range(A):
                for b in range(A):
                    w[j, i, b, a] = w[i, j, a, b]
                    n_ijab[j, i, b, a] = n_ijab[i, j, a, b]

    c_ij = np.sqrt((w**2).sum(axis=(2,3)))
    if args.correction == 'EVC':
        e_ijab = calculate_e_ijab(msa, n_ijab, lambda_w)
        predictions = (w**2 - e_ijab).sum(axis=(2,3))
    elif args.correction == 'APC':
        row_mean = np.nanmean(c_ij, axis=1)
        total_mean = np.nanmean(c_ij)
        predictions = c_ij - np.outer(row_mean, row_mean) / total_mean
    elif args.correction == 'None':
        predictions = c_ij    
    
    predictions[np.triu_indices_from(predictions)] = np.nan
    np.save(args.prediction_matrix, predictions)


def calculate_e_ijab(msa, n_ijab, lambda_w):
    n_ia, _ = ccmpred.counts.both_counts(msa)
    q_ia = n_ia[:, :A]
    q_ia = q_ia / q_ia.sum(axis=1)[:, None]

    n_ij = n_ijab[:, :, :A, :A].sum(axis=(2, 3))

    scaling = n_ij * n_ij / (n_ij - 1)
    num_factor = q_ia * (1 - q_ia)
    num = np.transpose(np.multiply.outer(num_factor, num_factor), axes=(0,2,1,3))
    denom = n_ij[:, :, np.newaxis, np.newaxis] * np.transpose(np.multiply.outer(q_ia, q_ia), axes=(0,2,1,3))
    denom += lambda_w
    denom = denom**2
    
    e_ijab = scaling[:, :, None, None] * num / denom
    return e_ijab


if __name__ == '__main__':
    main()
