import os
import numpy as np
import argparse
from typing import Callable, List, Optional, Union
import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import stochastic_blockmodel_graph
import lightning as L


class CSBM(InMemoryDataset):
    def __init__(
            self,
            root: str,
            block_sizes: Union[List[int], Tensor],
            edge_probs: Union[List[List[float]], Tensor],
            src_ratio: List[float],
            num_channels: Optional[int] = None,
            centers=None,
            cluster_std=1.0,
            is_undirected: bool = True,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            flip_y: float = 0.,
            random_state: int = 42,
            src_train_val_ratio: List = [0.8, 0.2],
            tgt_train_val_ratio: List = [0.6, 0.2],
            **kwargs,
    ):
        if not isinstance(block_sizes, torch.Tensor):
            block_sizes = torch.tensor(block_sizes, dtype=torch.long)
        if not isinstance(edge_probs, torch.Tensor):
            edge_probs = torch.tensor(edge_probs, dtype=torch.float)

        self.block_sizes = block_sizes
        self.edge_probs = edge_probs
        self.src_ratio = src_ratio
        self.num_channels = num_channels
        self.is_undirected = is_undirected
        self.flip_y = flip_y
        self.random_state = random_state
        self.src_train_val_ratio = src_train_val_ratio
        self.tgt_train_val_ratio = tgt_train_val_ratio
        self.kwargs = {
            'centers': centers,
            'cluster_std': cluster_std,
            'shuffle': False,
        }
        self.kwargs.update(kwargs)

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        from sklearn.datasets import make_blobs

        edge_index = stochastic_blockmodel_graph(
            self.block_sizes, self.edge_probs, directed=not self.is_undirected)

        num_samples = int(self.block_sizes.sum())
        num_classes = self.block_sizes.size(0)

        x = None
        if self.num_channels is not None:

            x, y_not_sorted, c = make_blobs(n_samples=self.block_sizes, n_features=self.num_channels, return_centers=True,
                                            random_state=self.random_state, **self.kwargs)
            x = x[np.argsort(y_not_sorted)]
            x = torch.from_numpy(x).to(torch.float)

        y = torch.arange(num_classes).repeat_interleave(self.block_sizes)
        if self.flip_y > 0.0:
            flip_mask = torch.bernoulli(self.flip_y * torch.ones_like(y)).type(torch.bool)
            y[flip_mask] = torch.randint(num_classes, size=(int(flip_mask.sum().item()),))


        # src/tgt split with configurable class ratio
        src_per_class = torch.round(self.block_sizes[:-1] * self.src_ratio).type(torch.int64)
        # shuffle indices within each class
        # input: [4, 4, 2]
        # output: [0, 3, 1, 2, 4, 6, 7, 5, 9, 8]
        src_mask = torch.zeros(num_samples, dtype=torch.bool)
        for cls in range(num_classes - 1):
            cls_indices = np.arange(sum(self.block_sizes[:cls]), sum(self.block_sizes[:cls+1]))
            np.random.shuffle(cls_indices)
            cls_indices = torch.from_numpy(cls_indices)
            src_mask[cls_indices[:src_per_class[cls]]] = True
        tgt_mask = torch.logical_not(src_mask)

        # train/val split among src and train/val/test split among tgt
        src_indices = torch.nonzero(src_mask)
        src_num = len(src_indices)
        src_subindices = torch.randperm(src_num)
        src_train_num = int(src_num*self.src_train_val_ratio[0])
        src_train_indices = src_indices[src_subindices[:src_train_num]]
        src_val_indices = src_indices[src_subindices[src_train_num:]]
        tgt_indices = torch.nonzero(tgt_mask)
        tgt_num = len(tgt_indices)
        tgt_subindices = torch.randperm(tgt_num)
        tgt_train_num = int(tgt_num*self.tgt_train_val_ratio[0])
        tgt_val_num = int(tgt_num*self.tgt_train_val_ratio[1])
        tgt_train_indices = tgt_indices[tgt_subindices[:tgt_train_num]]
        tgt_val_indices = tgt_indices[tgt_subindices[tgt_train_num:tgt_train_num+tgt_val_num]]
        tgt_test_indices = tgt_indices[tgt_subindices[tgt_train_num+tgt_val_num:]]

        train_mask = torch.zeros(num_samples, dtype=torch.bool)
        train_mask[src_train_indices] = 1
        train_mask[tgt_train_indices] = 1
        val_mask = torch.zeros(num_samples, dtype=torch.bool)
        val_mask[src_val_indices] = 1
        val_mask[tgt_val_indices] = 1
        test_mask = torch.zeros(num_samples, dtype=torch.bool)
        test_mask[tgt_test_indices] = 1

        data = Data(x=x, edge_index=edge_index, y=y, src_mask=src_mask, tgt_mask=tgt_mask, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

def create(args):
    root_dir = os.path.join(args.root_dir, "csbm3", str(args.shift) + "_" + str(args.alpha))

    os.makedirs(root_dir)

    num_src_base = 2000
    num_tgt_base = 2000
    src_base_dist = np.array([0.5, 0.5]) # can control subpopulation shift here
    tgt_base_dist = np.array([0.5 - args.shift, 0.5 + args.shift])
    # num_novel = 1000

    block_sizes = np.zeros(3)
    block_sizes[:2] += num_src_base * src_base_dist + num_tgt_base * tgt_base_dist
    block_sizes[2] = num_tgt_base * args.alpha // (1 - args.alpha)
    block_sizes = block_sizes.astype(int)

    edge_probs = np.array([[0.012, 0.006, 0.006], # 0.012, 0.06
                           [0.006, 0.012, 0.006],
                           [0.006, 0.006, 0.012]])
    feat_dim = 2
    centers = np.stack([[np.cos(0), np.sin(0)],
                        [np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)],
                        [np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3)]])
    cluster_std = 1 # / np.sqrt(2)
    src_ratio_per_cls = np.divide(src_base_dist, src_base_dist + tgt_base_dist)

    src_train_val_ratio = [0.8, 0.2]
    tgt_train_val_ratio = [0.6, 0.2]

    print(block_sizes)
    print(src_ratio_per_cls)
    dataset = CSBM(root=root_dir, block_sizes=block_sizes, edge_probs=edge_probs,
                   num_channels=feat_dim, centers=centers, cluster_std=cluster_std,
                   random_state=args.seed, src_ratio=src_ratio_per_cls, src_train_val_ratio=src_train_val_ratio, tgt_train_val_ratio=tgt_train_val_ratio)

    data = dataset[0]

    block_sz_cum_sum = np.cumsum(block_sizes)
    print(f"Number of src data in cls 0: {data.src_mask[:block_sz_cum_sum[0]].sum()}")
    print(f"Number of src data in cls 1: {data.src_mask[block_sz_cum_sum[0]:block_sz_cum_sum[1]].sum()}")
    print(f"Number of src data in cls 2: {data.src_mask[block_sz_cum_sum[1]:block_sz_cum_sum[2]].sum()}")

    print(f"Number of tgt data in cls 0: {data.tgt_mask[:block_sz_cum_sum[0]].sum()}")
    print(f"Number of tgt data in cls 1: {data.tgt_mask[block_sz_cum_sum[0]:block_sz_cum_sum[1]].sum()}")
    print(f"Number of tgt data in cls 2: {data.tgt_mask[block_sz_cum_sum[1]:block_sz_cum_sum[2]].sum()}")

    print("src and tgt mask sum:", torch.logical_and(data.src_mask, data.tgt_mask).sum())
    print("train and val mask sum:", torch.logical_and(data.train_mask, data.val_mask).sum())
    print("train and test mask sum:", torch.logical_and(data.train_mask, data.test_mask).sum())
    print("test and val mask sum:", torch.logical_and(data.test_mask, data.val_mask).sum())
    print("train val split within source:", torch.logical_and(data.src_mask, data.train_mask).sum(), torch.logical_and(data.src_mask, data.val_mask).sum())
    print("train val test split within target:", torch.logical_and(data.tgt_mask, data.train_mask).sum(), torch.logical_and(data.tgt_mask, data.val_mask).sum(), torch.logical_and(data.tgt_mask, data.test_mask).sum())
    print("src distribution", torch.bincount(data.y[data.src_mask]))
    print("tgt distribution", torch.bincount(data.y[data.tgt_mask]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="produce synthetic dataset generated from a Stochastic Block Model")
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--shift", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    L.seed_everything(args.seed)
    create(args)