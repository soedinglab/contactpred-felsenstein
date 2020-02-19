import argparse
import numpy as np
import ccmpred

from contextlib import redirect_stdout

# hard coded alphabet size
A = 20

def create_parser():
    parser = argparse.ArgumentParser('global_cp')
    parser.add_argument('braw_file')
    parser.add_argument('msa_psicov')
    parser.add_argument('lambda_w', type=float)
    parser.add_argument('--correction', choices=['None', 'APC', 'EVC'], default='None')
    parser.add_argument('prediction_matrix')
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    aln_file = args.msa_psicov
    braw_file = args.braw_file
    msa = ccmpred.io.read_msa_psicov(aln_file)
    N, L = msa.shape

    lambda_w = args.lambda_w
   
    ccm = ccmpred.CCMpred()
    ccm.set_alignment_file(aln_file)
    ccm.set_initraw_file(braw_file)
    with open('/dev/null', 'w') as sink:
        with redirect_stdout(sink):
            ccm.read_alignment()
            ccm.intialise_potentials()
    
    w = ccm.x_pair[:,:,:A,:A]

    c_ij = np.sqrt((w**2).sum(axis=(2,3)))
    if args.correction == 'EVC':
        e_ijab = calculate_e_ijab(msa, lambda_w)
        predictions = (w**2 - e_ijab).sum(axis=(2,3))
    elif args.correction == 'APC':
        row_mean = np.nanmean(c_ij, axis=1)
        total_mean = np.nanmean(c_ij)
        predictions = c_ij - np.outer(row_mean, row_mean) / total_mean
    elif args.correction == 'None':
        predictions = c_ij    
    
    np.save(args.prediction_matrix, predictions)


def calculate_e_ijab(msa, lambda_w):
    n_ia, n_ijab = ccmpred.counts.both_counts(msa)
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
