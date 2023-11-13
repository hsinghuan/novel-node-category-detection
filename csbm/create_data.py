import os
import numpy as np
from dataset import StochasticBlockModelBlobDataset

import sys
sys.path.append("..")
from utils import set_model_seed

SEED = 42

set_model_seed(42)
root_dir = "/home/hhchung/data/sbm_ncd"
os.makedirs(root_dir, exist_ok=True)

block_sizes = np.array([470, 470, 60]) # last class is the novel category, we can control the positive rate here
edge_probs = np.array([[0.02, 0.009, 0.003],
                       [0.009, 0.02, 0.003],
                       [0.003, 0.003, 0.02]])
feat_dim = 2
centers = np.stack([[np.cos(0), np.sin(0)],
                    [np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)],
                    [np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3)]])
cluster_std = 1 / np.sqrt(2)
src_ratio = 0.5 # ratio of source data among the seen classes (class 0 and 1)

dataset = StochasticBlockModelBlobDataset(root=root_dir, block_sizes=block_sizes, edge_probs=edge_probs,
                                          num_channels=feat_dim, centers=centers, cluster_std=cluster_std,
                                          random_state=SEED, src_ratio=src_ratio)

# print(dataset[0])