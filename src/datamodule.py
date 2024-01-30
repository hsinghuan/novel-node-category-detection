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
                 # link_predict: str = None,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.novel_cls = novel_cls
        self.alpha = alpha
        self.dataset_subdir = dataset_subdir
        # self.link_predict = link_predict

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

        # if self.link_predict:
        #     from torch_geometric.transforms import RandomLinkSplit
        #     transform = RandomLinkSplit(num_val=0.2, num_test=0.05, split_labels=True, add_negative_train_samples=False, is_undirected=True)
        #     self.lp_train_data, self.lp_val_data, self.lp_test_data = transform(self.data)

    def train_dataloader(self):
        # if self.link_predict:
        #     return DataLoader([(self.data, self.lp_train_data)], batch_size=1)
        # else:
        return DataLoader([self.data], batch_size=1)

    def val_dataloader(self):
        # if self.link_predict:
        #     return DataLoader([(self.data, self.lp_val_data)], batch_size=1)
        # else:
        return DataLoader([self.data], batch_size=1)

    def test_dataloader(self):
        # if self.link_predict:
        #     return DataLoader([(self.data, self.lp_test_data)], batch_size=1)
        # else:
        return DataLoader([self.data], batch_size=1)

    def predict_dataloader(self):
        return DataLoader([self.data], batch_size=1)

    def _dropout_novel_nodes(self):
        """
        Drop novel class nodes to maintain th
        :return:
        """
        data = self.data
        num_nodes = len(data.y)
        tgt_label_hist = torch.bincount(data.y[data.tgt_mask])
        novel_cls_num = tgt_label_hist[self.novel_cls].item()
        tgt_num_nodes = tgt_label_hist.sum().item()
        num_to_drop_nodes = int((novel_cls_num - self.alpha * tgt_num_nodes) / (1 - self.alpha))
        novel_cls_indices = torch.argwhere(data.y == self.novel_cls)
        to_drop_indices = novel_cls_indices[torch.randperm(len(novel_cls_indices))[:num_to_drop_nodes]]
        node_mask = torch.ones_like(data.y, dtype=torch.bool)
        node_mask[to_drop_indices] = 0

        new_edge_index, _, _ = subgraph(node_mask, data.edge_index, num_nodes=num_nodes, return_edge_mask=True, relabel_nodes=True)
        # subgraph implementation relabel nodes in original node index order, i.e. smaller original node index will result in smaller new node index

        new_x = data.x[node_mask]
        new_y = data.y[node_mask]
        new_src_mask = data.src_mask[node_mask]
        new_tgt_mask = data.tgt_mask[node_mask]
        new_train_mask = data.train_mask[node_mask]
        new_val_mask = data.val_mask[node_mask]
        new_test_mask = data.test_mask[node_mask]
        new_data = Data(x=new_x, edge_index=new_edge_index, y=new_y,
                        src_mask=new_src_mask, tgt_mask=new_tgt_mask,
                        train_mask=new_train_mask,val_mask=new_val_mask, test_mask=new_test_mask)

        self.data = new_data