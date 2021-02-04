import argparse
import numpy as np
import math

A = 20


def create_parser():
    parser = argparse.ArgumentParser()
    for lambda_w in (1, 2):
        parser.add_argument(f'v{lambda_w}_npy')
        parser.add_argument(f'w{lambda_w}_npy')
        parser.add_argument(f'l{lambda_w}', type=float)
    parser.add_argument('n_npy')
    return parser


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


def calculate_nij(v1, v2, w1, w2, lambda_w1, lambda_w2):
    num = 0
    for a in range(A):
        for b in range(A):
            num += (lambda_w1 * w1[a, b] - lambda_w2 * w2[a, b])**2
    denom = 0
    p1 = calculate_p_ijab(v1, w1)
    p2 = calculate_p_ijab(v2, w2)
    for a in range(A):
        for b in range(A):
            denom += (p1[a, b] - p2[a, b])**2
    return math.sqrt(num / denom)


def calculate_nijab(v_ij, w_ij, lambda_w, n_ij):
    p_ijab = calculate_p_ijab(v_ij, w_ij)
    n_ijab = n_ij * p_ijab + lambda_w * w_ij
    return n_ijab


def nan_like(arr):
    nan_arr = np.empty(arr.shape)
    nan_arr.fill(np.nan)
    return nan_arr


def main():
    parser = create_parser()
    args = parser.parse_args()

    v1 = np.load(args.v1_npy)
    w1 = np.load(args.w1_npy)
    l1 = args.l1

    v2 = np.load(args.v2_npy)
    w2 = np.load(args.w2_npy)
    l2 = args.l2

    assert v1.shape == v2.shape
    assert w1.shape == w2.shape

    n_full = np.empty(w1.shape)
    n_full.fill(np.nan)

    L, L, _, _ = v1.shape

    for i in range(L):
        for j in range(i+1, L):
            try:
                N_ij = calculate_nij(v1[i, j], v2[i, j], w1[i, j], w2[i, j], l1, l2)
                n_ijab = calculate_nijab(v1[i, j], w1[i, j], l1, N_ij)
            except OverflowError:
                n_ijab = nan_like(w1[i, j])
            n_full[i, j] = n_ijab

    for i in range(L):
        for j in range(0, i):
            for a in range(A):
                for b in range(A):
                    n_full[i, j, a, b] = n_full[j, i, b, a]

    np.save(args.n_npy, n_full)


if __name__ == '__main__':
    main()
