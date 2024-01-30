import random
import numpy as np
import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes

def subgraph_negative_sampling(edge_index, subgraph_mask, num_nodes=None, num_neg_samples=None):
    r""" Given an edge_index and subgraph mask, return a negative sampled edge index that only contains node in the subgraph mask
    Args:
        edge_index (LongTensor): The edge indices.
        subgraph_mask (BoolTensor): Mask indicating nodes to be contained in the negative samples
        num_nodes (int, Optional): The number of nodes
        num_neg_samples (int, optional): The number of negative samples to return
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    num_subgraph_nodes = subgraph_mask.sum().item() # number of subgraph nodes available for negative sampling
    num_neg_samples = num_neg_samples or edge_index.size(1)

    subgraph_node_idx = subgraph_mask.nonzero().view(-1)
    subgraph_edge_index = torch.isin(edge_index, subgraph_node_idx)
    subgraph_edge_index = edge_index[:, torch.logical_and(subgraph_edge_index[0], subgraph_edge_index[1])]

    num_neg_samples = min(num_neg_samples, num_subgraph_nodes * num_subgraph_nodes - subgraph_edge_index.size(1)) # second term is the maximum possible number of positive samples

    idx = (edge_index[0] * num_nodes + edge_index[1]).to("cpu")

    # rng = range(num_nodes**2) # change to sample from subgraph node <-> subgraph node
    full_subgraph_edge_index = torch.zeros((2, num_subgraph_nodes ** 2), dtype=torch.int)
    for i in range(num_subgraph_nodes):
        for j in range(num_subgraph_nodes):
            full_subgraph_edge_index[0][i * num_subgraph_nodes + j] = subgraph_node_idx[i]
            full_subgraph_edge_index[1][i * num_subgraph_nodes + j] = subgraph_node_idx[j]
    rng = (full_subgraph_edge_index[0] * num_nodes + full_subgraph_edge_index[1]).tolist()

    perm = torch.tensor(random.sample(rng, num_neg_samples))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8)) # whether random sampled edge is in positive
    rest = mask.nonzero().view(-1) # randomly sampled elements that are in positive
    while rest.numel() > 0: # resample
        tmp = torch.tensor(random.sample(rng, rest.size(0)))
        mask = torch.from_numpy(np.isin(tmp, idx).astype(np.uint8))
        perm[rest] = tmp # refill the positive edges
        rest = rest[mask.nonzero().view(-1)]

    row, col = perm / num_nodes, perm % num_nodes
    return torch.stack([row, col], dim=0).long().to(edge_index.device)
