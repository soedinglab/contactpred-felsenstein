import argparse
import yaml
import os
from os import path
import glob
import sys
from collections import Counter, defaultdict

from ccmpred.io import read_msa_psicov
from ccmpred import counts
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('plot_prefix')
    return parser


def calculate_prec_rec(evaluation, contact_dir, aln_dir, diag_diff=1, max_gap_pct=0):
    precision_curves = {}
    for data_dir, label, *_ in evaluation:
        precisions = []
        mat_glob = glob.glob(f'{data_dir}/*.mat.npy')
        if len(mat_glob) == 0:
            print('no matrices found in:', data_dir, file=sys.stderr)
            continue
        for matfile in sorted(mat_glob):
            bname = path.basename(matfile)
            stem, *_ = bname.split(os.extsep)

            contact_file = f'{contact_dir}/{stem}.npy'
            aln_file = f'{aln_dir}/{stem}.aln'

            msa = read_msa_psicov(aln_file)
            N, L = msa.shape
            gap_counts = N - counts.pair_counts(msa)[:, :, :20, :20].sum(axis=(2, 3))
            gap_pct = gap_counts / N

            contacts = np.load(contact_file)
            pred_mat = np.load(matfile)

            pred_mat_space = np.zeros(pred_mat.shape)
            pred_mat_space[np.triu_indices_from(pred_mat, k=diag_diff)] = 1

            pred_mat_space[gap_pct > max_gap_pct] = 0
            pred_ind = np.where(pred_mat_space)

            pred_contacts = contacts.copy().astype(bool)
            pred_mask = np.zeros(contacts.shape, dtype=bool)
            pred_mask[pred_ind] = 1
            pred_contacts &= pred_mask

            predictions = pred_mat[pred_ind]
            predictions[~np.isfinite(predictions)] = -np.inf
            ground_truth = contacts[pred_ind]

            sorter = np.argsort(predictions)[::-1]
            precision = ground_truth[sorter].cumsum() / (np.arange(len(ground_truth)) + 1)
            precisions.append(precision)

        mean_precisions = np.vstack(precisions).mean(axis=0)
        precision_curves[label] = mean_precisions
    return precision_curves


def get_color_mapping(evaluation):
    color_counter = Counter([i for _, _, i in evaluation])
    colors_ind = defaultdict(int)
    color_mapping = {}
    for _, label, palette_name in evaluation:
        linmap = np.linspace(0.3, 0.9, color_counter[palette_name])[colors_ind[palette_name]]
        color = mpl.cm.get_cmap(palette_name)(linmap)
        colors_ind[palette_name] += 1
        color_mapping[label] = color
    return color_mapping


def plot(precision_curves, color_mapping, x_max, x_max_auc):
    fig, ax = plt.subplots(figsize=(14, 8))
    for label, curve in precision_curves.items():
        auc = np.trapz(curve[:x_max_auc], dx=1)
        ls = '--'
        lw = 2
        alpha=0.75
        ax.step(
            np.arange(len(curve)) + 1, curve,
            label=label + f', AUC{x_max_auc}={auc:.1f}',
            linestyle=ls, lw=lw, alpha=alpha,
            color=color_mapping[label]
        )
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=18)
    ax.set_xlim(1, x_max)
    ax.set_xlabel('#predictions', fontsize=18)
    ax.set_ylabel('precision', fontsize=18)
    return fig, ax


def main():
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_file) as yml_file:
        cfg = yaml.load(yml_file, Loader=yaml.FullLoader)

    evaluation = []
    for curve_dict in cfg['curves'].values():
        evaluation.append((
            curve_dict['matrix_dir'],
            curve_dict['label'],
            curve_dict['color']
        ))

    precisions = calculate_prec_rec(
        evaluation, cfg['contact_dir'], cfg['aln_dir'], cfg['diag_min_diff'], cfg['max_gap_pct']
    )
    color_mapping = get_color_mapping(evaluation)
    fig, ax = plot(precisions, color_mapping, cfg['x_max'], cfg['x_max_auc'])

    plt.tight_layout()
    plt.savefig(f'{args.plot_prefix}.png', dpi=600)
    plt.savefig(f'{args.plot_prefix}.pdf')
    plt.close()


if __name__ == '__main__':
    main()
