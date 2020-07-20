import argparse
from Bio import Phylo


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('newick_in')
    parser.add_argument('newick_out')
    parser.add_argument('scaling_factor', type=float)
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    with open(args.newick_in) as handle:
        tree = next(Phylo.NewickIO.parse(handle))

    for node in tree.root.find_elements():
        if node.branch_length is not None:
            node.branch_length = max(node.branch_length*args.scaling_factor, 1e-5)

    with open(args.newick_out, 'w') as out:
        Phylo.NewickIO.write([tree], out)


if __name__ == '__main__':
    main()
