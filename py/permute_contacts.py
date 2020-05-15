import argparse
import numpy as np


def create_parser():
    parser = argparse.ArgumentParser('permute_contacts')
    parser.add_argument('contact_matrix')
    parser.add_argument('permuted_contact_matrix')
    parser.add_argument('--seed', type=int, default=42)
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)

    mat = np.load(args.contact_matrix)
    triu_ind = np.triu_indices_from(mat)
    contacts = mat[triu_ind]
    permutated_contacts = np.random.permutation(contacts)

    new_mat = np.zeros(mat.shape)
    new_mat[triu_ind] = permutated_contacts
    new_mat = new_mat + new_mat.T
    new_mat[new_mat > 1] = 1

    np.save(args.permuted_contact_matrix, new_mat)


if __name__ == '__main__':
    main()