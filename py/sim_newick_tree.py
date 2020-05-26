import argparse
import math
import copy
import random
import uuid

from scipy.stats import norm
import numpy as np
from Bio import Phylo
from Bio.Phylo.Newick import Clade, Tree
np.random.seed(42)
random.seed(42)


def create_parser():
    parser = argparse.ArgumentParser('sim_newick_tree')
    parser.add_argument('N_leaves', type=int)
    parser.add_argument('--path-mu', type=float, default=1)
    parser.add_argument('--path-sd', type=float, default=0.2)
    parser.add_argument('newick_tree')
    return parser


class Node:
    def __init__(self):
        self._parent = None
        self._left_child = None
        self._right_child = None
        self._deleted = False
        self._left_branchlength = None
        self._right_branchlength = None

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent

    @property
    def has_left_child(self):
        left_child = self._left_child
        if left_child is None:
            return False
        else:
            return False if left_child._deleted else True

    @property
    def left_child(self):
        left_child = self._left_child
        if left_child is None:
            return None
        else:
            return None if left_child._deleted else left_child

    @left_child.setter
    def left_child(self, left_child):
        self._left_child = left_child

    @property
    def has_right_child(self):
        right_child = self.right_child
        if right_child is None:
            return False
        else:
            return False if right_child._deleted else True

    @property
    def right_child(self):
        right_child = self._right_child
        if right_child is None:
            return None
        else:
            return None if right_child._deleted else right_child

    @right_child.setter
    def right_child(self, right_child):
        self._right_child = right_child

    @property
    def deleted(self):
        return self._deleted

    @deleted.setter
    def deleted(self, deleted):
        self._deleted = deleted

    @property
    def is_leaf(self):
        return self._left_child is None and self._right_child is None

    @property
    def left_branchlength(self):
        return self._left_branchlength

    @left_branchlength.setter
    def left_branchlength(self, left_branchlength):
        self._left_branchlength = left_branchlength

    @property
    def right_branchlength(self):
        return self._right_branchlength

    @right_branchlength.setter
    def right_branchlength(self, right_branchlength):
        self._right_branchlength = right_branchlength


def prune_tree_helper(node):

    has_left_child = node.left_child is not None
    has_right_child = node.right_child is not None

    if has_left_child:
        prune_tree_helper(node.left_child)
    if has_right_child:
        prune_tree_helper(node.right_child)

    has_left_child = node.left_child is not None
    has_right_child = node.right_child is not None

    parent = node.parent

    if has_left_child ^ has_right_child:
        if has_left_child:
            child = node.left_child
            branch_length = node.left_branch_length
        else:
            child = node.right_child
            branch_length = node.right_branch_length

        if parent.left_child == node:
            parent.left_child = child
            parent.left_branch_length = parent.left_branch_length + branch_length
        else:
            parent.right_child = child
            parent.right_branch_length = parent.right_branch_length + branch_length
        child.parent = node.parent

    if not node.is_leaf and not (has_left_child or has_right_child):
        node.deleted = True


def prune_tree(node):
    node = copy.deepcopy(node)
    prune_tree_helper(node)

    if node.left_child is None:
        root = node.right_child
    elif node.right_child is None:
        root = node.left_child
    else:
        root = node
    return root


def create_binary_tree(n_leaves, bl_generator):
    last_layer = []
    depth = math.ceil(math.log2(n_leaves)) + 1
    for i in range(2**(depth - 1)):
        leaf = Node()
        leaf.seq_id = i
        if i >= n_leaves:
            leaf.deleted = True
        last_layer.append(leaf)

    for layer in range(depth - 2, -1, -1):
        new_layer = []
        for i in range(2**layer):
            node = Node()

            left_child = last_layer[2*i]
            node.left_child = left_child
            left_child.parent = node
            node.left_branchlength = next(bl_generator)

            right_child = last_layer[2*i + 1]
            right_child.parent = node
            node.right_child = right_child
            node.right_branchlength = next(bl_generator)

            new_layer.append(node)
        last_layer = new_layer

    root, = last_layer
    root.parent = root
    return prune_tree(root)


def cladify(node, branch_length):
    clades = []
    if node.has_left_child:
        clades.append(cladify(node.left_child, node.left_branchlength))
    if node.has_right_child:
        clades.append(cladify(node.right_child, node.right_branchlength))
    clade = Clade(clades=clades, branch_length=branch_length, name=str(uuid.uuid4()))
    return clade


def tree2biopython(tree):
    root = cladify(tree, 0)
    root.name = 'root'
    newick_tree = Tree(root, rooted=True)

    # rename leaf nodes with seq_ids
    cur_seq_id = 0
    for clade in newick_tree.find_clades(order='level'):
        if not clade.clades:
            clade.name = str(cur_seq_id)
            cur_seq_id += 1
    return newick_tree


def main():
    parser = create_parser()
    args = parser.parse_args()

    depth = np.ceil(np.log2(args.N_leaves))

    mu = args.path_mu / depth
    sd = args.path_sd / np.sqrt(depth)

    edge_length_distr = norm(mu, sd)
    edge_lengths = edge_length_distr.rvs(min(2*args.N_leaves, int(1e6)))
    edge_lengths = np.maximum(edge_lengths, 1e-9)

    edge_gen = (el for el in edge_lengths)
    bin_tree = create_binary_tree(args.N_leaves, edge_gen)
    nw_tree = tree2biopython(bin_tree)

    with open(args.newick_tree, 'w') as out:
        Phylo.NewickIO.write([nw_tree], out)


if __name__  == '__main__':
    main()
