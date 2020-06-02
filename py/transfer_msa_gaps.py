import argparse
import numpy as np
import ccmpred

GAP_STATE = 20


def create_parser():
    parser = argparse.ArgumentParser('transfer_msa_gaps')
    parser.add_argument('msa_in')
    parser.add_argument('gap_template_msa')
    parser.add_argument('msa_out')
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    msa = ccmpred.io.read_msa_psicov(args.msa_in)
    gap_template_msa = ccmpred.io.read_msa_psicov(args.gap_template_msa)
    gap_pos = np.where(gap_template_msa == GAP_STATE)
    msa[gap_pos] = 20
    with open(args.msa_out, 'w') as handle:
        ccmpred.io.write_msa_psicov(handle, msa)


if __name__ == '__main__':
    main()
