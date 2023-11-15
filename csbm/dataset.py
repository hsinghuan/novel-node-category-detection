import os
import numpy as np
from typing import Callable, List, Optional, Union
import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import stochastic_blockmodel_graph

class StochasticBlockModelBlobDataset(InMemoryDataset):
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
            train_val_ratio: List = [0.6, 0.2],
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
        self.train_val_ratio = train_val_ratio
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



        src_per_class = (self.block_sizes[:-1] * self.src_ratio).type(torch.int64)
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

        data = Data(x=x, edge_index=edge_index, y=y, src_mask=src_mask, tgt_mask=tgt_mask) # , train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


def load_sbm_dataset(data_dir:str):
    dataset_name = "StochasticBlockModelBlobDataset"
    leaf_dir = os.path.join(os.path.join(data_dir, dataset_name, "processed"))
    dir_list = os.listdir(leaf_dir)
    path = os.path.join(leaf_dir, list(filter(lambda fname: "data" in fname, dir_list))[0])
    data, _ = torch.load(path)
    return data