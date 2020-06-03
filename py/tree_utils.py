from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree
from ccmpred.io.alignment import AMINO_ACIDS

import math
import copy


def read_newick_tree(newick_tree_file):
    with open(newick_tree_file) as newick_handle:
        tree = next(Phylo.NewickIO.parse(newick_handle))
    return biopython_phylo_to_tree(tree)


def create_binary_tree(n_leaves, branchlength):
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
            node.left_branchlength = branchlength

            right_child = last_layer[2*i + 1]
            right_child.parent = node
            node.right_child = right_child
            node.right_branchlength = branchlength

            new_layer.append(node)
        last_layer = new_layer

    root, = last_layer
    root.parent = root
    return prune_tree(root)


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

    def mark_deleted(self):
        self._deleted = True

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

    @property
    def left_branch_length(self):
        return self._left_branchlength

    @left_branch_length.setter
    def left_branch_length(self, left_branch_length):
        self._left_branchlength = left_branch_length

    @property
    def right_branch_length(self):
        return self._right_branchlength

    @right_branch_length.setter
    def right_branch_length(self, right_branch_length):
        self._right_branchlength = right_branch_length


def biopython_phylo_to_tree(phylo_tree):
    root = Node()
    root.parent = root
    tree_queue = [root]
    phylo_queue = [phylo_tree.clade]

    while len(phylo_queue) > 0:
        phylo_node = phylo_queue.pop(0)
        tree_node = tree_queue.pop(0)

        if len(phylo_node.clades) == 0:
            # this is a leaf
            tree_node.seq_id = int(phylo_node.name)
            continue

        left_clade, right_clade = phylo_node.clades

        left_child = Node()
        left_child.parent = tree_node
        tree_node.left_child = left_child
        tree_node.left_branchlength = max(left_clade.branch_length, 0)

        right_child = Node()
        right_child.parent = tree_node
        tree_node.right_child = right_child
        tree_node.right_branchlength = max(right_clade.branch_length, 0)

        tree_queue.append(left_child)
        tree_queue.append(right_child)

        phylo_queue.append(left_clade)
        phylo_queue.append(right_clade)

    return root


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


def print_tree_sequences(tree, msa, i, j):
    queue = [tree]
    while len(queue) > 0:
        node = queue.pop()
        if node.is_leaf:
            print(msa[node.seq_id, i], msa[node.seq_id, j])
            continue
        if node.has_left_child:
            queue.append(node.left_child)
        if node.has_right_child:
            queue.append(node.right_child)



def create_seq_node_map(tree):
    seq_node_map = {}
    queue = [tree]

    while len(queue) > 0:
        node = queue.pop()
        if node.is_leaf:
            seq_node_map[node.seq_id] = node
            continue
        if node.has_left_child:
            queue.append(node.left_child)
        if node.has_right_child:
            queue.append(node.right_child)
    return seq_node_map


def tree2biopython_helper(node, msa, i, j):
    if node.parent.left_child == node:
        branch_length = node.parent.left_branch_length
    else:
        branch_length = node.parent.right_branch_length

    if node.is_leaf:
        aa_i = AMINO_ACIDS[msa[node.seq_id, i]]
        aa_j = AMINO_ACIDS[msa[node.seq_id, j]]
        label = f'{aa_i}|{aa_j}'
    else:
        label = ''

    clades = []
    if node.has_left_child:
        clades.append(tree2biopython_helper(node.left_child, msa, i, j))
    if node.has_right_child:
        clades.append(tree2biopython_helper(node.right_child, msa, i, j))

    clade = Clade(branch_length=branch_length, name=label, clades=clades)
    return clade


def tree2biopython(tree, msa, i, j):
    clades = []
    if tree.has_left_child:
        clades.append(tree2biopython_helper(tree.left_child, msa, i, j))
    if tree.has_right_child:
        clades.append(tree2biopython_helper(tree.right_child, msa, i, j))
    clade = Clade(branch_length=0, name='root', clades=clades)
    return Tree(clade)
