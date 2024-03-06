import numpy as np
import argparse
import os
import torch
from torch_geometric import datasets
import lightning as L

def preprocess(dataset_name, preprocess_name, cls_num, novel_cls, src_ratio_per_cls, args):
    root_dir = args.root_dir
    dataset = datasets.Planetoid(root=root_dir, name=dataset_name)
    data = dataset[0]

    num_nodes = len(data.y)

    # src/tgt split
    src_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for cls in range(cls_num):
        if cls == novel_cls:
            continue
        cls_indices = torch.argwhere(data.y == cls).reshape(-1)
        cls_num_nodes = len(cls_indices)
        perm_indices = torch.randperm(cls_num_nodes)
        cls_num_src_nodes = int(cls_num_nodes * src_ratio_per_cls[cls])
        cls_src_indices = cls_indices[perm_indices[:cls_num_src_nodes]]
        src_mask[cls_src_indices] = True

    tgt_mask = torch.logical_not(src_mask)

    # train/val/(test) split
    src_train_val_ratio = np.array([0.8, 0.2])
    tgt_train_val_ratio = np.array([0.6, 0.2])

    src_indices = torch.nonzero(src_mask)
    src_num = len(src_indices)
    src_perm_indices = torch.randperm(src_num)
    src_train_num = int(src_num * src_train_val_ratio[0])
    src_train_indices = src_indices[src_perm_indices[:src_train_num]]
    src_val_indices = src_indices[src_perm_indices[src_train_num:]]

    tgt_indices = torch.nonzero(tgt_mask)
    tgt_num = len(tgt_indices)
    tgt_perm_indices = torch.randperm(tgt_num)
    tgt_train_num = int(tgt_num * tgt_train_val_ratio[0])
    tgt_val_num = int(tgt_num * tgt_train_val_ratio[1])
    tgt_train_indices = tgt_indices[tgt_perm_indices[:tgt_train_num]]
    tgt_val_indices = tgt_indices[tgt_perm_indices[tgt_train_num:tgt_train_num + tgt_val_num]]
    tgt_test_indices = tgt_indices[tgt_perm_indices[tgt_train_num + tgt_val_num:]]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[src_train_indices] = 1
    train_mask[tgt_train_indices] = 1
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[src_val_indices] = 1
    val_mask[tgt_val_indices] = 1
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[tgt_test_indices] = 1

    data.src_mask = src_mask
    data.tgt_mask = tgt_mask
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    dir = os.path.join(root_dir, dataset_name, preprocess_name)
    os.makedirs(dir)
    torch.save(data, os.path.join(dir, "data.pt"))

    print("src and tgt mask sum:", torch.logical_and(data.src_mask, data.tgt_mask).sum())
    print("train and val mask sum:", torch.logical_and(data.train_mask, data.val_mask).sum())
    print("train and test mask sum:", torch.logical_and(data.train_mask, data.test_mask).sum())
    print("test and val mask sum:", torch.logical_and(data.test_mask, data.val_mask).sum())
    print("train val split within source:", torch.logical_and(data.src_mask, data.train_mask).sum(),
          torch.logical_and(data.src_mask, data.val_mask).sum())
    print("train val test split within target:", torch.logical_and(data.tgt_mask, data.train_mask).sum(),
          torch.logical_and(data.tgt_mask, data.val_mask).sum(), torch.logical_and(data.tgt_mask, data.test_mask).sum())
    print("src distribution", torch.bincount(data.y[data.src_mask]))
    print("tgt distribution", torch.bincount(data.y[data.tgt_mask]))
    print(data.src_mask.shape)
    print(data.tgt_mask.shape)
    print(data.train_mask.shape)
    print(data.val_mask.shape)
    print(data.test_mask.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Cora and Citeseer for novel node category detection")
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--preprocess_name", type=str, default="shift")
    args = parser.parse_args()

    if args.preprocess_name == "no_shift":
        seed = 10
    else:
        seed = 42

    L.seed_everything(seed)
    if args.dataset == "Cora":
        cls_num = 7
        novel_cls = 6

        if args.preprocess_name == "shift":
            src_ratio_per_cls = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.])
        elif args.preprocess_name == "minor_shift":
            src_ratio_per_cls = np.array([0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.])
        elif args.preprocess_name == "no_shift":
            src_ratio_per_cls = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.])

        preprocess(args.dataset, args.preprocess_name, cls_num, novel_cls, src_ratio_per_cls, args)

    elif args.dataset == "CiteSeer":
        cls_num = 6
        novel_cls = 5

        if args.preprocess_name == "shift":
            src_ratio_per_cls = np.array([0.9, 0.1, 0.9, 0.1, 0.5, 0.])
        elif args.preprocess_name == "minor_shift":
            src_ratio_per_cls = np.array([0.7, 0.3, 0.7, 0.3, 0.5, 0.])
        elif args.preprocess_name == "no_shift":
            src_ratio_per_cls = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.])

        preprocess(args.dataset, args.preprocess_name, cls_num, novel_cls, src_ratio_per_cls, args)