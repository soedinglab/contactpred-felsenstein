import argparse

import matplotlib.pyplot as plt
import matplotlib
from Bio import Phylo

matplotlib.use('Agg')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('newick_tree')
    parser.add_argument('output_file')
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    with open(args.newick_tree) as handle:
        tree = next(Phylo.NewickIO.parse(handle))

    for node in tree.find_elements():
        node.name = ''

    n_leaves = len(tree.get_terminals())

    fig, ax = plt.subplots(figsize=(8, n_leaves/50))
    Phylo.draw(tree, axes=ax)
    fig.savefig(args.output_file)
    plt.close(fig)


if __name__ == '__main__':
    main()
