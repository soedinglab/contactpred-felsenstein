import argparse
import numpy as np
from ccmpred.io import read_msa_psicov, distance_map


def create_parser():
    parser = argparse.ArgumentParser('pdb2contacts')
    parser.add_argument('pdb_file')
    parser.add_argument('aln_file')
    parser.add_argument('contact_file')
    parser.add_argument('--distance-threshold', default=8, type=float,
                        help='Cb-Cb distance threshold in Angstrom')
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    msa = read_msa_psicov(args.aln_file)
    N, L = msa.shape
    pdb_file = args.pdb_file

    dist_map = distance_map(pdb_file, L=L)
    contacts = dist_map < args.distance_threshold
    np.save(args.contact_file, contacts)


if __name__ == '__main__':
    main()
