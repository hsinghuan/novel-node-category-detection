import numpy as np
from tqdm import tqdm
import networkx as nx
from sklearn.metrics import roc_auc_score
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import LabelPropagation

class LabelPropPU(L.LightningModule):
    def __init__(self, dataset_name, novel_cls, novel_ratio, num_layers, alpha, seed):
        super().__init__()
        self.dataset_name = dataset_name
        self.novel_cls = novel_cls
        self.novel_ratio = novel_ratio
        self.num_layers = num_layers
        self.alpha = alpha
        self.seed = seed
        self.lp_model = LabelPropagation(num_layers=num_layers, alpha=alpha)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()

    def label_prop(self, label, edge_index):
        self.lp_model(label, edge_index)

    def training_step(self, batch):
        # identify nodes that are most likely to be novel via node distances
        G = to_networkx(batch, to_undirected=True)
        print("Start calculating path length")
        length = dict(nx.all_pairs_shortest_path_length(G)) # key: node idx, value: dict(key: node idx, value: path length)
        print("End calculating path length")
        novel_score = torch.zeros(batch.src_mask.size(0)) # tensor saving the score for each node belonging in novel category
        src_node_idx = batch.src_mask.nonzero().view(-1)
        tgt_node_idx = batch.tgt_mask.nonzero().view(-1)
        print("Start calculating novel scores")
        for node in tqdm(tgt_node_idx):
            node_len_dict = length[node.item()]
            avg_len = 0
            connected_src_num = 0
            for node2 in node_len_dict.keys():
                if node2 in src_node_idx:
                    avg_len += node_len_dict[node2]
                    connected_src_num += 1
            if connected_src_num > 0:
                avg_len = avg_len / connected_src_num
            else:
                avg_len = torch.tensor(np.inf)
            novel_score[node.item()] = avg_len
        print("End calculating novel scores")
        # identify the novel nodes for future label propagation
        novel_score_rank = torch.argsort(novel_score, descending=True)
        num_novel = int(batch.tgt_mask.sum().item() * self.novel_ratio)
        # num_novel = min(batch.src_mask.sum().item(), batch.tgt_mask.sum().item())
        novel_node_idx = novel_score_rank[:num_novel]

        self.novel_score = novel_score
        self.novel_node_idx = novel_node_idx

    def on_save_checkpoint(self, checkpoint):
        checkpoint["novel_node_idx"] = self.novel_node_idx

    def on_load_checkpoint(self, checkpoint):
        self.novel_node_idx = checkpoint["novel_node_idx"]

    def validation_step(self, batch):
        # do label propagation
        label = batch.tgt_mask.type(torch.int64)
        mask = batch.src_mask
        mask[self.novel_node_idx] = True
        out = self.lp_model(label, batch.edge_index, mask=mask)
        scores = out[:,1]
        # print(scores[batch.src_mask].mean())
        # print(scores[batch.tgt_mask].mean())
        # print(scores[self.novel_node_idx].mean())
        y_oracle = torch.zeros_like(batch.tgt_mask, dtype=torch.int64)
        y_oracle[batch.y == self.novel_cls] = 1
        outputs = {"scores": scores,
                   "y_oracle": y_oracle,
                   "tgt_mask": batch.tgt_mask,
                   "val_mask": batch.val_mask}
        self.validation_step_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        scores = torch.cat([o["scores"] for o in outputs], dim=0).detach().cpu().numpy()
        y_oracle = torch.cat([o["y_oracle"] for o in outputs], dim=0).detach().cpu().numpy()
        tgt_mask = torch.cat([o["tgt_mask"] for o in outputs], dim=0).detach().cpu().numpy()
        val_mask = torch.cat([o["val_mask"] for o in outputs], dim=0).detach().cpu().numpy()
        tgt_val_mask = np.logical_and(tgt_mask, val_mask)
        roc_auc = roc_auc_score(y_oracle[tgt_val_mask], scores[tgt_val_mask])
        self.log("val/performance.AU-ROC", roc_auc, on_step=False, on_epoch=True)
        self.validation_step_outputs = []


    def test_step(self, batch):
        # do label propagation
        label = batch.tgt_mask.type(torch.int64)
        mask = batch.src_mask
        mask[self.novel_node_idx] = True
        out = self.lp_model(label, batch.edge_index, mask=mask)
        scores = out[:, 1]
        # print(scores[batch.src_mask].mean())
        # print(scores[batch.tgt_mask].mean())
        # print(scores[self.novel_node_idx].mean())
        y_oracle = torch.zeros_like(batch.tgt_mask, dtype=torch.int64)
        y_oracle[batch.y == self.novel_cls] = 1
        outputs = {"scores": scores,
                   "y_oracle": y_oracle,
                   "tgt_mask": batch.tgt_mask,
                   "test_mask": batch.test_mask}
        self.test_step_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        scores = torch.cat([o["scores"] for o in outputs], dim=0).detach().cpu().numpy()
        y_oracle = torch.cat([o["y_oracle"] for o in outputs], dim=0).detach().cpu().numpy()
        tgt_mask = torch.cat([o["tgt_mask"] for o in outputs], dim=0).detach().cpu().numpy()
        test_mask = torch.cat([o["test_mask"] for o in outputs], dim=0).detach().cpu().numpy()
        tgt_test_mask = np.logical_and(tgt_mask, test_mask)
        roc_auc = roc_auc_score(y_oracle[tgt_test_mask], scores[tgt_test_mask])
        self.log("test/performance.AU-ROC", roc_auc, on_step=False, on_epoch=True)

        tgt_roc_auc = roc_auc_score(y_oracle[tgt_mask], scores[tgt_mask])
        self.log("tgt/performance.AU-ROC", tgt_roc_auc, on_step=False, on_epoch=True)
        self.test_step_outputs = []

    def configure_optimizers(self):
        pass