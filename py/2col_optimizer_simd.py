import argparse
import numpy as np
import ccmpred
from optimize_felsenstein_simd import optimize_felsenstein, OptimizationFailure


def create_parser():
    parser = argparse.ArgumentParser('optimizer_2col')
    parser.add_argument('msa')
    parser.add_argument('--col1', type=int, default=0)
    parser.add_argument('--col2', type=int, default=1)
    parser.add_argument('v_out')
    parser.add_argument('w_out')
    parser.add_argument('--branch-length', type=float, default=0.5)
    parser.add_argument('--lambda-w', type=float, default=10)
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    msa = ccmpred.io.read_msa_psicov(args.msa)
    N_leaves, L = msa.shape
    n_mut = args.branch_length
    condensed_tree = [((2*i+1, n_mut), (2*i+2, n_mut)) for i in range(N_leaves-1)] + [None] * N_leaves

    x0 = None
    n_tries = 20
    while n_tries > 0:
        n_tries -= 1
        try:
            v_opt, w_opt, info = optimize_felsenstein(msa, args.col1, args.col2, condensed_tree, args.lambda_w, x0=x0)
            break
        except OptimizationFailure as ex:
            x0 = ex.last_x + np.random.normal(0, 1e-5, len(ex.last_x))
    else:
        raise Exception(f'Failed optimization with lambda_w={lambda_w} after {max_tries} tries.')

    np.save(args.v_out, v_opt)
    np.save(args.w_out, w_opt)


if __name__ == '__main__':
    main()
