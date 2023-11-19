import os
import lightning as L
import torch
from torch_geometric.loader import DataLoader

class DataModule(L.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 dataset_name: str
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name

    def setup(self, stage):
        leaf_dir = os.path.join(os.path.join(self.data_dir, self.dataset_name, "processed"))
        dir_list = os.listdir(leaf_dir)
        path = os.path.join(leaf_dir, list(filter(lambda fname: "data" in fname, dir_list))[0])
        self.dataset, _ = torch.load(path)

    def train_dataloader(self):
        return DataLoader([self.dataset], batch_size=1)

    def val_dataloader(self):
        return DataLoader([self.dataset], batch_size=1)