import os
from typing import List, Union, Tuple
import lightning as L
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph

class DataModule(L.LightningDataModule):
    """
    Currently for single graph and single novel class, can be extended further in the future.
    """
    def __init__(self,
                 data_dir: str,
                 dataset_name: str,
                 novel_cls: int,
                 alpha: float = None,
                 dataset_subdir: str = None,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.novel_cls = novel_cls
        self.alpha = alpha
        self.dataset_subdir = dataset_subdir

    def setup(self, stage):
        if self.dataset_subdir:
            leaf_dir = os.path.join(self.data_dir, self.dataset_name, self.dataset_subdir)
        else:
            leaf_dir = os.path.join(self.data_dir, self.dataset_name, "processed")
        dir_list = os.listdir(leaf_dir)
        path = os.path.join(leaf_dir, list(filter(lambda fname: "data" in fname, dir_list))[0])
        loaded = torch.load(path)
        if isinstance(loaded, Tuple):
            self.data = loaded[0]
        else:
            self.data = loaded

    def train_dataloader(self):
        return DataLoader([self.data], batch_size=1)

    def val_dataloader(self):
        return DataLoader([self.data], batch_size=1)

    def test_dataloader(self):
        return DataLoader([self.data], batch_size=1)

    def predict_dataloader(self):
        return DataLoader([self.data], batch_size=1)