import argparse
import sys
import numpy as np
from ccmpred import CCMpred
from utils import calculate_seq_probs

A = 20


def create_parser():
    parser = argparse.ArgumentParser('braw2seqprobs')
    parser.add_argument('braw_file')
    parser.add_argument('--seq_probs')
    parser.add_argument('--pair_probs')
    return parser


def load_mrf_params(braw_file):
    ccm = CCMpred()
    ccm.set_initraw_file(braw_file)
    ccm.intialise_potentials()
    v = ccm.x_single
    w = ccm.x_pair[:, :, :A, :A]
    return v, w


def main():
    parser = create_parser()
    args = parser.parse_args()
    v, w = load_mrf_params(args.braw_file)

    L, A = v.shape
    if L > 7:
        print('For your own savety, I\'m not iterating through {L} sequences.', file=sys.err)
        sys.exit(1)

    seq_probs, pair_probs = calculate_seq_probs(v, w)
    if args.seq_probs:
        np.save(args.seq_probs, seq_probs, allow_pickle=False)
    if args.pair_probs:
        np.save(args.pair_probs, pair_probs, allow_pickle=False)


if __name__ == '__main__':
    main()
