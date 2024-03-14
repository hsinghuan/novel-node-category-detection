import numpy as np
from copy import deepcopy
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import lightning as L
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph, structured_negative_sampling
from torch_geometric.nn.models import InnerProductDecoder
from src.utils.model_utils import get_model_optimizer
from src.utils.mpe_utils import p_probs, u_probs, BBE_estimator

# Adapted from https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective/blob/main/losses/distributionLoss.py
class LabelDistributionLoss(torch.nn.Module):
    def __init__(self, prior_high, prior_low, device, num_bins=1, proxy='polar', dist='L1'):
        super(LabelDistributionLoss, self).__init__()
        self.prior_high = prior_high
        self.prior_low = prior_low
        self.frac_prior = 1.0 / (2 * (prior_high + prior_low))

        self.step = 1 / num_bins  # bin width. predicted scores in [0, 1].
        self.device = device
        self.t = torch.arange(0, 1 + self.step, self.step).view(1, -1).requires_grad_(False)  # [0, 1+bin width)
        self.t_size = num_bins + 1

        self.dist = None
        if dist == 'L1':
            self.dist = F.l1_loss
        else:
            raise NotImplementedError("The distance: {} is not defined!".format(dist))

        # proxy
        proxy_p, proxy_n = None, None
        if proxy == 'polar':
            proxy_p = np.zeros(self.t_size, dtype=float) # [0, 0]
            proxy_n = np.zeros_like(proxy_p) # [0, 0]
            proxy_p[-1] = 1 # [0, 1]
            proxy_n[0] = 1 # [1, 0]
        else:
            raise NotImplementedError("The proxy: {} is not defined!".format(proxy))

        proxy_mix_high = prior_high * proxy_p + (1 - prior_high) * proxy_n # [1 - prior_high, prior_high]
        proxy_mix_low = prior_low * proxy_p + (1 - prior_low) * proxy_n # [1 - prior_low, prior_low]
        print('#### Label Distribution Loss ####')
        print('ground truth P:')
        print(proxy_p)
        print('pseudo ground truth U high:')
        print(proxy_mix_high)
        print('pseudo ground truth U low:')
        print(proxy_mix_low)

        # to torch tensor
        self.proxy_p = torch.from_numpy(proxy_p).requires_grad_(False).float() # [0, 1]
        self.proxy_mix_high = torch.from_numpy(proxy_mix_high).requires_grad_(False).float() # [1 - prior_high, prior_high]
        self.proxy_mix_low = torch.from_numpy(proxy_mix_low).requires_grad_(False).float() # [1 - prior_low, prior_low]

        # to device
        self.t = self.t.to(self.device)
        self.proxy_p = self.proxy_p.to(self.device)
        self.proxy_mix_high = self.proxy_mix_high.to(self.device)
        self.proxy_mix_low = self.proxy_mix_low.to(self.device)

    def histogram(self, scores):
        scores_rep = scores.repeat(1, self.t_size)

        hist = torch.abs(scores_rep - self.t)

        inds = (hist > self.step)
        hist = self.step - hist  # switch values
        hist[inds] = 0

        return hist.sum(dim=0) / (len(scores) * self.step)

    def forward(self, outputs, labels, unlabeled_high_mask: torch.BoolTensor):
        scores = torch.sigmoid(outputs)
        labels = labels.view(-1, 1)
        unlabeled_high_mask = unlabeled_high_mask.view(-1, 1)
        scores = scores.view_as(labels)
        s_p = scores[labels == 1].view(-1, 1) # scores for (labeled) positive data
        s_u_high = scores[torch.logical_and(labels == 0, unlabeled_high_mask)].view(-1, 1) # scores for unlabeled high data (unlabeled nodes closer to source nodes)
        s_u_low = scores[torch.logical_and(labels == 0, ~unlabeled_high_mask)].view(-1, 1) # scores for unlabeled low data (unlabeled nodes far from source nodes)
        # print(f"s p: {s_p}")
        # print(f"s u high: {s_u_high}")
        # print(f"s u low: {s_u_low}")
        l_p = 0
        l_u_high = 0
        l_u_low = 0
        if s_p.numel() > 0:
            hist_p = self.histogram(s_p)
            # print(f"hist p: {hist_p}, proxy p: {self.proxy_p}")
            l_p = self.dist(hist_p, self.proxy_p, reduction='mean') # match (labeled) positive score distribution with [0, 1]
        if s_u_high.numel() > 0:
            hist_u_high = self.histogram(s_u_high)
            # print(f"hist u high: {hist_u_high}, proxy mix high: {self.proxy_mix_high}")
            l_u_high = self.dist(hist_u_high, self.proxy_mix_high, reduction='mean') # match unlabeled high score distribution with [1 - prior_high, prior_high]
        if s_u_low.numel() > 0:
            hist_u_low = self.histogram(s_u_low)
            # print(f"hist u low: {hist_u_low}, proxy mix low: {self.proxy_mix_low}")
            l_u_low = self.dist(hist_u_low, self.proxy_mix_low, reduction='mean') # match unlabeled high score distribution with [1 - prior_high, prior_high]

        return l_p + self.frac_prior * l_u_high + self.frac_prior * l_u_low


class PUGNN(L.LightningModule):
    def __init__(self,
                 model_type,
                 arch_param,
                 dataset_name,
                 novel_cls,
                 learning_rate,
                 max_epochs,
                 warmup_epochs,
                 seed,
                 weight_decay=0.,
                 num_hops=3,
                 reg_loss_weight=1e-2,
                 reg_K=50
                 ):
        super().__init__()

        self.model_type = model_type
        self.dataset_name = dataset_name
        self.novel_cls = novel_cls
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.reg_loss_weight = reg_loss_weight
        self.reg_decoder = InnerProductDecoder()
        self.reg_K = reg_K
        self.num_hops = num_hops

        self.model, self.optimizer = get_model_optimizer(model_type, arch_param, learning_rate, weight_decay)

        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.best_warmup_loss = np.inf
        self.warm_start = True
        self.seed = seed

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.automatic_optimization = False
        self.save_hyperparameters()


    def forward(self, data):
        if self.model_type == "mlp":
            return self.model(data.x)
        else:
            return self.model(data.x, data.edge_index)

    def reg_loss(self, data):
        z = self.model.encoder(data.x, data.edge_index)
        pos_edge_index = data.edge_index
        neg_edge_index = None
        for _ in range(self.reg_K):
            i, j, k = structured_negative_sampling(pos_edge_index, contains_neg_self_loops=False)
            tmp_edge_index = torch.stack((i, k), dim=0)
            if neg_edge_index is None:
                neg_edge_index = tmp_edge_index
            else:
                neg_edge_index = torch.cat((neg_edge_index, tmp_edge_index), dim=1)
        pos_loss = (self.reg_decoder(z, pos_edge_index, sigmoid=True) - 1) ** 2
        neg_loss = (self.reg_decoder(z, neg_edge_index, sigmoid=True)) ** 2
        return (pos_loss.mean() + neg_loss.mean()) * self.reg_loss_weight

    def process_batch(self, batch, stage): # , batch_linkpred=None):
        y = batch.tgt_mask.type(torch.int64)

        if stage == "train":
            mask = batch.train_mask
        elif stage == "val":
            mask = batch.val_mask
        elif stage == "test":
            mask = batch.test_mask

        logits = self.forward(batch)
        # TODO: make src_neighbor_tgt and src_distant_tgt masks
        src_node_idx = batch.src_mask.nonzero().view(-1)
        # print(f"src node idx: {src_node_idx}")
        # print(f"edge index: {batch.edge_index}")
        # print(src_node_idx.max(), batch.edge_index.max())
        src_node_idx = src_node_idx[torch.isin(src_node_idx, batch.edge_index)]
        src_k_hop_subset, src_k_hop_edge_index, _, _ = k_hop_subgraph(src_node_idx, num_hops=self.num_hops, edge_index=batch.edge_index)
        src_k_hop_subset_mask = torch.zeros_like(batch.src_mask, dtype=torch.bool)
        src_k_hop_subset_mask[src_k_hop_subset] = 1
        src_neighbor_tgt_mask = torch.zeros_like(batch.src_mask, dtype=torch.bool)
        src_neighbor_tgt_mask[torch.logical_and(src_k_hop_subset_mask, batch.tgt_mask)] = True
        # check
        # print(torch.logical_and(src_neighbor_tgt_mask, batch.src_mask).sum().item()) # should be 0
        # print(torch.logical_and(src_neighbor_tgt_mask, batch.tgt_mask).sum().item()) # number of source neighbor target nodes
        # print(torch.logical_and(~src_neighbor_tgt_mask, batch.tgt_mask).sum().item()) # number of source distant target nodes

        if self.warm_start:
            loss = F.cross_entropy(logits[mask], y[mask])
            pu_loss, reg_loss = np.inf, np.inf
        else:
            pu_outputs = logits[mask][:,0]
            pu_targets = 1 - y[mask] # 0 -> 1, 1 -> 0
            pu_loss = self.pu_loss(pu_outputs, pu_targets, src_neighbor_tgt_mask[mask])
            reg_loss = self.reg_loss(batch)
            loss = pu_loss + reg_loss


        if stage == "train":
            optimizer = self.optimizers()
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            return loss, pu_loss, reg_loss
        elif stage == "val":
            probs = F.softmax(logits, dim=1)
            if self.warm_start:
                pos_probs = p_probs(self.model, self.device, batch, model_type=self.model_type, val=True)
                unlabeled_probs, unlabeled_targets = u_probs(self.model, self.device, batch, model_type=self.model_type,
                                                             val=True, novel_cls=self.novel_cls)
                src_neighbor_tgt_mask = src_neighbor_tgt_mask[torch.logical_and(batch.tgt_mask, batch.val_mask)] # extract tgt & val nodes, should have same length as unlabeled probs
            else:
                pos_probs, unlabeled_probs, unlabeled_targets = None, None, None
            return loss, pu_loss, reg_loss, probs, pos_probs, unlabeled_probs, unlabeled_targets, src_neighbor_tgt_mask
        elif stage == "test":
            probs = F.softmax(logits, dim=1)
            return loss, pu_loss, reg_loss, probs
        else:
            raise ValueError(f"Invalid stage {stage}")


    def training_step(self, batch, batch_idx):
        loss, pu_loss, reg_loss = self.process_batch(batch, "train")
        batch_size = batch.train_mask.sum().item()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("train/pu_loss", pu_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("train/reg_loss", reg_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)

        return {"loss": loss.detach()}

    def validation_step(self, batch, batch_idx):
        loss, pu_loss, reg_loss, probs, pos_probs, unlabeled_probs, unlabeled_targets, src_neighbor_tgt_mask = self.process_batch(batch, "val")
        batch_size = batch.val_mask.sum().item()

        if self.warm_start:
            self.log("val/warm_loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log("val/loss", 1e3 - self.current_epoch, on_step=True, on_epoch=True, prog_bar=False,
                     batch_size=batch_size)  # dummy val loss just to avoid earlystopping
        else:
            self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("val/pu_loss", pu_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("val/reg_loss", reg_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        y_oracle = torch.zeros_like(batch.y, dtype=torch.int64)
        y_oracle[batch.y == self.novel_cls] = 1
        outputs = {"loss": loss.detach(),
                   "probs": probs,
                   "pos_probs": pos_probs,
                   "unlabeled_probs": unlabeled_probs,
                   "unlabeled_targets": unlabeled_targets,
                   "y_oracle": y_oracle,
                   "tgt_mask": batch.tgt_mask,
                   "val_mask": batch.val_mask,
                   "src_neighbor_tgt_mask": src_neighbor_tgt_mask}
        self.validation_step_outputs.append(outputs)
        return outputs

    def test_step(self, batch, batch_idx):
        loss, pu_loss, reg_loss, probs = self.process_batch(batch, "test")
        batch_size = batch.test_mask.sum().item()
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("test/pu_loss", pu_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("test/reg_loss", reg_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        y_oracle = torch.zeros_like(batch.y, dtype=torch.int64)
        y_oracle[batch.y == self.novel_cls] = 1
        outputs = {"loss": loss.detach(),
                   "probs": probs,
                   "y_oracle": y_oracle,
                   "tgt_mask": batch.tgt_mask,
                   "test_mask": batch.test_mask}
        self.test_step_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        probs = torch.cat([o["probs"] for o in outputs], dim=0).detach().cpu().numpy()
        y_oracle = torch.cat([o["y_oracle"] for o in outputs], dim=0).detach().cpu().numpy()
        tgt_mask = torch.cat([o["tgt_mask"] for o in outputs], dim=0).detach().cpu().numpy()
        val_mask = torch.cat([o["val_mask"] for o in outputs], dim=0).detach().cpu().numpy()
        src_neighbor_tgt_mask = torch.cat([o["src_neighbor_tgt_mask"] for o in outputs], dim=0).detach().cpu().numpy()

        if self.warm_start:
            pos_probs = np.concatenate([o["pos_probs"] for o in outputs], axis=0)
            unlabeled_probs = np.concatenate([o["unlabeled_probs"] for o in outputs], axis=0)
            unlabeled_targets = np.concatenate([o["unlabeled_targets"] for o in outputs], axis=0)
            mpe_estimate_high, _, _ = BBE_estimator(pos_probs, unlabeled_probs[src_neighbor_tgt_mask], unlabeled_targets[src_neighbor_tgt_mask])  # unlabeled_targets isn't used for calculating the mpe estimate
            mpe_estimate_low, _, _ = BBE_estimator(pos_probs, unlabeled_probs[~src_neighbor_tgt_mask], unlabeled_targets[~src_neighbor_tgt_mask])  # unlabeled_targets isn't used for calculating the mpe estimate
            self.prior_high = mpe_estimate_high
            self.prior_low = mpe_estimate_low

        tgt_val_mask = np.logical_and(tgt_mask, val_mask)
        true_prior = 1 - y_oracle.sum().item() / len(y_oracle)  # true source proportion
        batch_size = tgt_val_mask.sum()

        roc_auc = roc_auc_score(y_oracle[tgt_val_mask], probs[:, 1][tgt_val_mask])
        ap = average_precision_score(y_oracle[tgt_val_mask], probs[:, 1][tgt_val_mask])
        f1 = f1_score(y_oracle[tgt_val_mask], np.argmax(probs, axis=1)[tgt_val_mask])
        self.log("val/performance.AU-ROC", roc_auc, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val/performance.AP", ap, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val/performance.F1", f1, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val/estimated_prior_high", self.prior_high, on_step=False, on_epoch=True)
        self.log("val/estimated_prior_low", self.prior_low, on_step=False, on_epoch=True)
        self.log("val/true_prior", true_prior, on_step=False, on_epoch=True)

        if self.warm_start:  # checkpoint best warmup model
            assert len(outputs) == 1  # only for single graph dataset
            loss = outputs[0]["loss"]
            if loss < self.best_warmup_loss:
                self.best_warmup_loss = loss
                self.best_prior_high = self.prior_high
                self.best_prior_low = self.prior_low
                self.best_warmup_model = deepcopy(self.model)

            if self.current_epoch < self.warmup_epochs:
                self.warm_start = True  # keep it true
            else:
                print(f"End warm up at epoch: {self.current_epoch}")
                self.warm_start = False
                self.model = deepcopy(self.best_warmup_model)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay) # reinitialize optimizer
                del self.best_warmup_model
                self.prior_high = self.best_prior_high
                self.prior_low = self.best_prior_low
                self.pu_loss = LabelDistributionLoss(prior_high=self.prior_high, prior_low=self.prior_low, device=self.device)

        self.validation_step_outputs = []

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        probs = torch.cat([o["probs"] for o in outputs], dim=0).detach().cpu().numpy()
        y_oracle = torch.cat([o["y_oracle"] for o in outputs], dim=0).detach().cpu().numpy()
        tgt_mask = torch.cat([o["tgt_mask"] for o in outputs], dim=0).detach().cpu().numpy()
        test_mask = torch.cat([o["test_mask"] for o in outputs], dim=0).detach().cpu().numpy()

        tgt_test_mask = np.logical_and(tgt_mask, test_mask)
        roc_auc = roc_auc_score(y_oracle[tgt_test_mask], probs[:, 1][tgt_test_mask])
        ap = average_precision_score(y_oracle[tgt_test_mask], probs[:, 1][tgt_test_mask])
        f1 = f1_score(y_oracle[tgt_test_mask], np.argmax(probs, axis=1)[tgt_test_mask])
        self.log("test/performance.AU-ROC", roc_auc, on_step=False, on_epoch=True, batch_size=tgt_test_mask.sum())
        self.log("test/performance.AP", ap, on_step=False, on_epoch=True, batch_size=tgt_test_mask.sum())
        self.log("test/performance.F1", f1, on_step=False, on_epoch=True, batch_size=tgt_test_mask.sum())

        tgt_roc_auc = roc_auc_score(y_oracle[tgt_mask], probs[:, 1][tgt_mask])
        tgt_ap = average_precision_score(y_oracle[tgt_mask], probs[:, 1][tgt_mask])
        tgt_f1 = f1_score(y_oracle[tgt_mask], np.argmax(probs, axis=1)[tgt_mask])
        self.log("tgt/performance.AU-ROC", tgt_roc_auc, on_step=False, on_epoch=True, batch_size=tgt_mask.sum())
        self.log("tgt/performance.AP", tgt_ap, on_step=False, on_epoch=True, batch_size=tgt_mask.sum())
        self.log("tgt/performance.F1", tgt_f1, on_step=False, on_epoch=True, batch_size=tgt_mask.sum())

        self.test_step_outputs = []

    def configure_optimizers(self):
        return self.optimizer
