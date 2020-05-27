import argparse
import numpy as np
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
from Bio import Phylo
import ccmpred

import pam_distance

A = 20


def create_parser():
    parser = argparse.ArgumentParser('msa2tree')
    parser.add_argument('msa_file')
    parser.add_argument('out_newick_tree')
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    msa = ccmpred.io.read_msa_psicov(args.msa_file)
    nw_tree = estimate_nj_tree(msa)

    with open(args.out_newick_tree, 'w') as out:
        Phylo.NewickIO.write([nw_tree], out)


def estimate_nj_tree(msa):

    single_counts = ccmpred.counts.single_counts(msa)[:, :A]
    p_ia = single_counts / single_counts.sum(axis=1)[:, None]

    N, _ = msa.shape
    dist_list = []
    for i in range(N):
        row_distances = []
        for j in range(0, i):
            row_distances.append(pam_distance.optimize_t(msa[i], msa[j], p_ia))
        row_distances.append(0)  # distance of i to itself
        dist_list.append(row_distances)

    leaf_names = [str(i) for i in range(N)]

    tc = DistanceTreeConstructor()
    dm = DistanceMatrix(leaf_names, dist_list)
    tree = tc.nj(dm)
    tree.root_at_midpoint()
    tree.root.name = 'root'

    for clade in tree.find_clades():
        clade.branch_length = max(clade.branch_length, 1e-5)

    return tree


if __name__ == '__main__':
    main()