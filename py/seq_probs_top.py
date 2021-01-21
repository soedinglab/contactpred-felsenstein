import argparse
import numpy as np


def create_parser():
    parser = argparse.ArgumentParser('braw2seqprobs')
    parser.add_argument('reference_seqprobs')
    parser.add_argument('seqprobs2')
    parser.add_argument('top_ref_seqprobs')
    parser.add_argument('top_seqprobs2')
    parser.add_argument('--top_n', type=int, default=1000)
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    seq_probs = np.load(args.reference_seqprobs)
    seq_probs2 = np.load(args.seqprobs2)

    sort_order = np.argsort(seq_probs)

    top_indices = sort_order[-args.top_n:]

    out_seqprobs1 = seq_probs[top_indices]
    put_seqprobs2 = seq_probs2[top_indices]

    np.save(args.top_ref_seqprobs, out_seqprobs1)
    np.save(args.top_seqprobs2, put_seqprobs2)


if __name__ == '__main__':
    main()
