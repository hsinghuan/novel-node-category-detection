import numpy as np
import argparse
import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, subgraph
from torch_geometric.utils.map import map_index
from ogb.nodeproppred import PygNodePropPredDataset
import lightning as L

def take_second(element):
    return element[1]


def preprocess(data, domain_bound=[0, 2007, 2011], test_year_bound=2009, proportion=1.0):
    """
    [-1, 2010, 2011]: load graph up till 2011. Source: 0~2010. Target: 2011.
    """
    assert len(domain_bound) == 3
    assert test_year_bound > domain_bound[1] and test_year_bound < domain_bound[2]
    node_years = data['node_year']
    n = node_years.shape[0]
    node_years = node_years.reshape(n)

    d = np.zeros(len(node_years))  # frequency of interaction of each node before year upper bound
    edges = data['edge_index']
    for i in range(edges.shape[1]):
        if node_years[edges[0][i]] < domain_bound[2] and node_years[edges[1][i]] < domain_bound[
            2]:  # if the edge happens before year upper bound
            d[edges[0][i]] += 1  # out node += 1
            d[edges[1][i]] += 1  # in node += 1

    nodes = []  # node id and frequency of interaction before year upper bound
    for i, year in enumerate(node_years):
        if year < domain_bound[2]:
            nodes.append([i, d[i]])

    nodes.sort(key=take_second, reverse=True)

    nodes_new = nodes[:int(proportion * len(nodes))]  # take top popular nodes that happens before year upper bound

    # reproduce id
    result_edges = []
    result_features = []
    result_labels = []
    for node in nodes_new:
        result_features.append(data.x[node[0]])
    result_features = torch.stack(result_features)

    ids = {}
    for i, node in enumerate(nodes_new):
        ids[node[0]] = i  # previous node id to new node id

    for i in range(edges.shape[1]):
        if edges[0][i].item() in ids and edges[1][
            i].item() in ids:  # if in node and out node of an edge are both in result nodes, add the edge
            result_edges.append([ids[edges[0][i].item()], ids[edges[1][i].item()]])
    result_edges = torch.LongTensor(result_edges).transpose(1, 0)
    result_labels = data.y[[node[0] for node in nodes_new]]
    result_labels = result_labels.squeeze(dim=-1)  # to accomodate to GBT repo

    data_new = Data(x=result_features, edge_index=result_edges, y=result_labels)
    node_years_new = torch.tensor([node_years[node[0]] for node in nodes_new])
    # data_new.node_year = node_years_new
    data_new.src_mask = torch.logical_and(node_years_new >= domain_bound[0], node_years_new < domain_bound[1])
    data_new.tgt_mask = torch.logical_and(node_years_new >= domain_bound[1], node_years_new < domain_bound[2])
    data_new.test_mask = node_years_new >= test_year_bound

    # drop theory categories
    # theory_categories = torch.tensor([9, 36, 34, 39, 33, 28, 2, 0], dtype=torch.int64)
    to_keep_categories = torch.tensor([10, 11, 16, 19, 24, 27], dtype=torch.int64)
    edge_index = data_new.edge_index
    # wo_theory_mask = torch.logical_not(torch.isin(data_new.y, theory_categories))
    to_keep_mask = torch.isin(data_new.y, to_keep_categories)
    # wo_theory_edge_index, _ = subgraph(wo_theory_mask, edge_index)
    to_keep_edge_index, _ = subgraph(to_keep_mask, edge_index)
    # wo_theory_num_nodes = wo_theory_mask.size(0)
    to_keep_num_nodes = to_keep_mask.size(0)
    # wo_theory_subset = wo_theory_mask.nonzero().view(-1)
    to_keep_subset = to_keep_mask.nonzero().view(-1)
    # relabeled_wo_theory_edge_index, _ = map_index(wo_theory_edge_index, wo_theory_subset, max_index=wo_theory_num_nodes, inclusive=True)
    relabeled_to_keep_edge_index, _ = map_index(to_keep_edge_index, to_keep_subset, max_index=to_keep_num_nodes, inclusive=True)
    # relabeled_wo_theory_edge_index = relabeled_wo_theory_edge_index.view(2, -1)
    relabeled_to_keep_edge_index = relabeled_to_keep_edge_index.view(2, -1)
    # wo_theory_y = data_new.y[wo_theory_mask]
    to_keep_y = data_new.y[to_keep_mask]
    full_labels = torch.arange(40)
    # wo_theory_labels = full_labels[torch.logical_not(torch.isin(full_labels, theory_categories))]
    to_keep_labels = to_keep_categories
    # print(f"wo theory labels: {wo_theory_labels}")
    print(f"to keep labels: {to_keep_labels}")
    # wo_theory_y_max = torch.max(wo_theory_y)
    to_keep_y_max = torch.max(to_keep_y)
    # print(f"number of wo theory y in theory categories: {torch.isin(wo_theory_y, theory_categories).sum().item()}")
    # relabeled_wo_theory_y, _ = map_index(wo_theory_y, wo_theory_labels, max_index=wo_theory_y_max, inclusive=True)
    relabeled_to_keep_y, _ = map_index(to_keep_y, to_keep_labels, max_index=to_keep_y_max, inclusive=True)
    # print(wo_theory_y[:50])
    # print(relabeled_wo_theory_y[:50])
    print(to_keep_y[:50])
    print(relabeled_to_keep_y[:50])
    # data_new = Data(x=data_new.x[wo_theory_mask],
    #                 y=relabeled_wo_theory_y,
    #                 edge_index=relabeled_wo_theory_edge_index,
    #                 src_mask=data_new.src_mask[wo_theory_mask],
    #                 tgt_mask=data_new.tgt_mask[wo_theory_mask],
    #                 test_mask=data_new.test_mask[wo_theory_mask])
    data_new = Data(x=data_new.x[to_keep_mask],
                    y=relabeled_to_keep_y,
                    edge_index=relabeled_to_keep_edge_index,
                    src_mask=data_new.src_mask[to_keep_mask],
                    tgt_mask=data_new.tgt_mask[to_keep_mask],
                    test_mask=data_new.test_mask[to_keep_mask])

    # train/val split
    src_train_val_ratio = np.array([0.8, 0.2])
    tgt_train_val_ratio = np.array([0.8, 0.2])

    src_indices = torch.nonzero(data_new.src_mask)
    src_num = len(src_indices)
    src_perm_indices = torch.randperm(src_num)
    src_train_num = int(src_num * src_train_val_ratio[0])
    src_train_indices = src_indices[src_perm_indices[:src_train_num]]
    src_val_indices = src_indices[src_perm_indices[src_train_num:]]

    tgt_nontest_indices = torch.nonzero(torch.logical_and(data_new.tgt_mask, ~data_new.test_mask))
    tgt_nontest_num = len(tgt_nontest_indices)
    tgt_nontest_perm_indices = torch.randperm(tgt_nontest_num)
    tgt_train_num = int(tgt_nontest_num * tgt_train_val_ratio[0])
    tgt_val_num = int(tgt_nontest_num * tgt_train_val_ratio[1])
    tgt_train_indices = tgt_nontest_indices[tgt_nontest_perm_indices[:tgt_train_num]]
    tgt_val_indices = tgt_nontest_indices[tgt_nontest_perm_indices[tgt_train_num:]]

    train_mask = torch.zeros(len(data_new.y), dtype=torch.bool)
    train_mask[src_train_indices] = 1
    train_mask[tgt_train_indices] = 1
    data_new.train_mask = train_mask

    val_mask = torch.zeros(len(data_new.y), dtype=torch.bool)
    val_mask[src_val_indices] = 1
    val_mask[tgt_val_indices] = 1
    data_new.val_mask = val_mask


    print("src and tgt mask sum:", torch.logical_and(data_new.src_mask, data_new.tgt_mask).sum())
    print("train and val mask sum:", torch.logical_and(data_new.train_mask, data_new.val_mask).sum())
    print("train and test mask sum:", torch.logical_and(data_new.train_mask, data_new.test_mask).sum())
    print("test and val mask sum:", torch.logical_and(data_new.test_mask, data_new.val_mask).sum())
    print("train val split within source:", torch.logical_and(data_new.src_mask, data_new.train_mask).sum(),
          torch.logical_and(data_new.src_mask, data_new.val_mask).sum())
    print("train val test split within target:", torch.logical_and(data_new.tgt_mask, data_new.train_mask).sum(),
          torch.logical_and(data_new.tgt_mask, data_new.val_mask).sum(), torch.logical_and(data_new.tgt_mask, data_new.test_mask).sum())
    src_subgroup_distribution = torch.bincount(data_new.y[data_new.src_mask])
    tgt_subgroup_distribution = torch.bincount(data_new.y[data_new.tgt_mask])
    print("src distribution", src_subgroup_distribution)
    print("tgt distribution", tgt_subgroup_distribution)
    novel_cls = torch.argwhere(src_subgroup_distribution == 0).view(-1)
    print("novel cls:", novel_cls)
    print(data_new.src_mask.shape)
    print(data_new.tgt_mask.shape)
    print(data_new.train_mask.shape)
    print(data_new.val_mask.shape)
    print(data_new.test_mask.shape)
    tgt_val_mask = torch.logical_and(data_new.tgt_mask, data_new.val_mask)
    tgt_test_mask = torch.logical_and(data_new.tgt_mask, data_new.test_mask)
    print("Novel nodes in tgt val:", torch.logical_and(torch.isin(data_new.y, novel_cls), tgt_val_mask).sum())
    print("Novel nodes in tgt test:", torch.logical_and(torch.isin(data_new.y, novel_cls), tgt_test_mask).sum())
    
    return data_new



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ogbn-arxiv for novel node category detection")
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    L.seed_everything(args.seed)
    domain_bound = [0, 2007, 2013]
    test_year_bound = 2012
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=args.root_dir)
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    data = preprocess(data, domain_bound, test_year_bound)
    partition_name = str(domain_bound[0]) + "_" + str(domain_bound[1]) + "_" + str(test_year_bound) + "_" + str(domain_bound[2]) + "_robotics_related"
    dir = os.path.join(args.root_dir, "ogbn_arxiv", partition_name)
    os.makedirs(dir)
    torch.save(data, os.path.join(dir, "data.pt"))