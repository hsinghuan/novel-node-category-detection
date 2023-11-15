import os
import numpy as np
import argparse
from dataset import StochasticBlockModelBlobDataset

import sys
sys.path.append("..")
from utils import set_model_seed


def create(args):
    set_model_seed(args.seed)

    os.makedirs(args.root_dir, exist_ok=True)

    num_src_base = 300
    num_tgt_base = 300
    src_base_dist = np.array([0.5, 0.5]) # can control subpopulation shift here
    tgt_base_dist = np.array([0.5, 0.5])
    alpha = 0.05 # can control novel ratio here

    block_sizes = np.zeros(3)
    block_sizes[:2] += num_src_base * src_base_dist + num_tgt_base * tgt_base_dist
    block_sizes[2] = num_tgt_base * alpha // (1 - alpha)
    block_sizes = block_sizes.astype(int)

    edge_probs = np.array([[0.02, 0.005, 0.005],
                           [0.005, 0.02, 0.005],
                           [0.005, 0.005, 0.02]])
    feat_dim = 2
    centers = np.stack([[np.cos(0), np.sin(0)],
                        [np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)],
                        [np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3)]])
    cluster_std = 1 / np.sqrt(2)
    src_ratio = np.divide(src_base_dist, src_base_dist + tgt_base_dist)

    print(block_sizes)
    print(src_ratio)
    dataset = StochasticBlockModelBlobDataset(root=args.root_dir, block_sizes=block_sizes, edge_probs=edge_probs,
                                              num_channels=feat_dim, centers=centers, cluster_std=cluster_std,
                                              random_state=args.seed, src_ratio=src_ratio)

    data = dataset[0]

    block_sz_cum_sum = np.cumsum(block_sizes)
    print(f"Number of src data in cls 0: {data.src_mask[:block_sz_cum_sum[0]].sum()}")
    print(f"Number of src data in cls 1: {data.src_mask[block_sz_cum_sum[0]:block_sz_cum_sum[1]].sum()}")
    print(f"Number of src data in cls 2: {data.src_mask[block_sz_cum_sum[1]:block_sz_cum_sum[2]].sum()}")

    print(f"Number of tgt data in cls 0: {data.tgt_mask[:block_sz_cum_sum[0]].sum()}")
    print(f"Number of tgt data in cls 1: {data.tgt_mask[block_sz_cum_sum[0]:block_sz_cum_sum[1]].sum()}")
    print(f"Number of tgt data in cls 2: {data.tgt_mask[block_sz_cum_sum[1]:block_sz_cum_sum[2]].sum()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="produce synthetic dataset generated from a Stochastic Block Model")
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create(args)