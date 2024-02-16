"""
Copied from https://github.com/cimeist=Falseter/pu-learning/blob/master/loss.py
"""
import numpy as np
from copy import deepcopy
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from src.utils.model_utils import get_model_optimizer
from src.utils.mpe_utils import p_probs, u_probs, BBE_estimator

class PULoss(nn.Module):
    def __init__(self, prior, loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnPU=False):
        super(PULoss, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss
        self.nnPU = nnPU
        print(f"nnPU?: {self.nnPU}")
        self.positive = 1 # source
        self.unlabeled = -1 # target
        self.min_count = torch.tensor(1.)

    def forward(self, inp, y):
        assert (inp.shape == y.shape)
        positive, unlabeled = y == self.positive, y == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)
        self.min_count.type_as(inp)
        n_positive, n_unlabeled = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count,
                                                                                            torch.sum(unlabeled))

        y_positive = self.loss_func(positive * inp) * positive
        y_positive_inv = self.loss_func(-positive * inp) * positive
        y_unlabeled = self.loss_func(-unlabeled * inp) * unlabeled

        positive_risk = self.prior * torch.sum(y_positive) / n_positive
        negative_risk = - self.prior * torch.sum(y_positive_inv) / n_positive + torch.sum(y_unlabeled) / n_unlabeled

        if negative_risk < -self.beta and self.nnPU:
            return -self.gamma * negative_risk
        else:
            return positive_risk + negative_risk


class PULearning(L.LightningModule):
    def __init__(
            self,
            model_type: str,
            arch_param,
            dataset_name,
            novel_cls,
            learning_rate,
            weight_decay,
            max_epochs,
            warmup_epochs,
            nnPU,
            seed):
        super().__init__()
        self.model_type = model_type
        self.novel_cls = novel_cls
        self.dataset_name = dataset_name
        self.seed = seed
        self.nnPU = nnPU
        self.pu_loss = None # assigned after mixture prior estimation (warm start epochs)

        self.model, self.optimizer = get_model_optimizer(model_type, arch_param, learning_rate, weight_decay)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs # MPE estimation in the first half
        self.best_warmup_loss = np.inf
        self.warm_start = True

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, data):
        if self.model_type == "mlp":
            return self.model(data.x)
        else:
            return self.model(data.x, data.edge_index)

    def process_batch(self, batch, stage):
        y_oracle = torch.zeros_like(batch.tgt_mask, dtype=torch.int64)
        y_oracle[batch.y == self.novel_cls] = 1
        y = batch.tgt_mask.type(torch.int64)
        logits = self.forward(batch)

        if stage == "train":
            mask = batch.train_mask
        elif stage == "val":
            mask = batch.val_mask
        elif stage == "test":
            mask = batch.test_mask

        if self.warm_start: # train plain domain discriminator during warm start for MPE estimate
            loss = F.cross_entropy(logits[mask], y[mask])
        else: # then use MPE estimate for uPU or nnPU risk estimates
            # transform logits and targets into PULoss format (src label: 1, tgt label: -1)
            pu_outputs = logits[mask][:,0]
            pu_targets = y[mask] # 0 -> 1, 1 -> -1
            pu_targets[pu_targets == 0] = 1
            pu_targets[pu_targets == 1] = -1
            loss = self.pu_loss(pu_outputs, pu_targets)
        if stage == "train":
            optimizer = self.optimizers()
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            return loss
        elif stage == "val":
            probs = F.softmax(logits, dim=1)
            if self.warm_start:
                pos_probs = p_probs(self.model, self.device, batch, model_type=self.model_type, val=True)
                unlabeled_probs, unlabeled_targets = u_probs(self.model, self.device, batch, model_type=self.model_type, val=True, novel_cls=self.novel_cls)
            else:
                pos_probs, unlabeled_probs, unlabeled_targets = None, None, None
            return loss, probs, pos_probs, unlabeled_probs, unlabeled_targets, y, y_oracle, batch.tgt_mask, mask
        elif stage == "test":
            probs = F.softmax(logits, dim=1)
            return loss, probs, y, y_oracle, batch.tgt_mask, mask
        else:
            raise ValueError(f"Invalid stage {stage}")

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch, "train")
        batch_size = batch.train_mask.sum().item()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        return {"loss": loss.detach()}

    def validation_step(self, batch, batch_idx):
        loss, probs, pos_probs, unlabeled_probs, unlabeled_targets, y, y_oracle, tgt_mask, val_mask = self.process_batch(batch, "val")
        batch_size = batch.val_mask.sum().item()
        if self.warm_start:
            self.log("val/warm_loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log("val/loss", 1e3 - self.current_epoch, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size) # dummy val loss just to avoid earlystopping
        else:
            self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        outputs =  {"loss": loss.detach(),
                    "probs": probs,
                    "pos_probs": pos_probs,
                    "unlabeled_probs": unlabeled_probs,
                    "unlabeled_targets": unlabeled_targets,
                    "y": y,
                    "y_oracle": y_oracle,
                    "tgt_mask": tgt_mask,
                    "val_mask": val_mask}
        self.validation_step_outputs.append(outputs)
        return outputs

    def test_step(self, batch, batch_idx):
        loss, probs, y, y_oracle, tgt_mask, test_mask = self.process_batch(batch, "test")
        batch_size = batch.test_mask.sum().item()
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        outputs = {"loss": loss.detach(),
                   "probs": probs,
                   "y": y,
                   "y_oracle": y_oracle,
                   "tgt_mask": tgt_mask,
                   "test_mask": test_mask}
        self.test_step_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        probs = torch.cat([o["probs"] for o in outputs], dim=0).detach().cpu().numpy()
        y_oracle = torch.cat([o["y_oracle"] for o in outputs], dim=0).detach().cpu().numpy()
        tgt_mask = torch.cat([o["tgt_mask"] for o in outputs], dim=0).detach().cpu().numpy()
        val_mask = torch.cat([o["val_mask"] for o in outputs], dim=0).detach().cpu().numpy()

        if self.warm_start:
            pos_probs = np.concatenate([o["pos_probs"] for o in outputs], axis=0)
            unlabeled_probs = np.concatenate([o["unlabeled_probs"] for o in outputs], axis=0)
            unlabeled_targets = np.concatenate([o["unlabeled_targets"] for o in outputs], axis=0)
            mpe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets) # unlabeled_targets isn't used for calculating the mpe estimate
            self.prior = mpe_estimate
        true_prior = 1 - y_oracle.sum().item() / len(y_oracle) # true source proportion

        # compute roc_auc_score, average precision
        tgt_val_mask = np.logical_and(tgt_mask, val_mask)
        roc_auc = roc_auc_score(y_oracle[tgt_val_mask], probs[:, 1][tgt_val_mask])
        ap = average_precision_score(y_oracle[tgt_val_mask], probs[:, 1][tgt_val_mask])
        f1 = f1_score(y_oracle[tgt_val_mask], np.argmax(probs, axis=1)[tgt_val_mask])

        self.log("val/performance.AU-ROC", roc_auc, on_step=False, on_epoch=True)
        self.log("val/performance.AP", ap, on_step=False, on_epoch=True)
        self.log("val/performance.F1", f1, on_step=False, on_epoch=True)
        self.log("val/estimated_prior", self.prior, on_step=False, on_epoch=True)
        self.log("val/true_prior", true_prior, on_step=False, on_epoch=True)


        if self.warm_start: # checkpoint best warmup model
            assert len(outputs) == 1 # only for single graph dataset
            loss = outputs[0]["loss"]
            if loss < self.best_warmup_loss:
                self.best_warmup_loss = loss
                self.best_warmup_model = deepcopy(self.model)

            if self.current_epoch < self.warmup_epochs:
                self.warm_start = True # keep it true
            else:
                print(f"End warm up at epoch: {self.current_epoch}")
                self.warm_start = False
                self.pu_loss = PULoss(prior=self.prior, nnPU=self.nnPU)
                self.model = deepcopy(self.best_warmup_model)
                del self.best_warmup_model

        self.validation_step_outputs = []

    def on_test_epoch_end(self) -> None:
        outputs = self.test_step_outputs
        probs = torch.cat([o["probs"] for o in outputs], dim=0).detach().cpu().numpy()
        y_oracle = torch.cat([o["y_oracle"] for o in outputs], dim=0).detach().cpu().numpy()
        tgt_mask = torch.cat([o["tgt_mask"] for o in outputs], dim=0).detach().cpu().numpy()
        test_mask = torch.cat([o["test_mask"] for o in outputs], dim=0).detach().cpu().numpy()

        # compute roc_auc_score, average precision
        tgt_test_mask = np.logical_and(tgt_mask, test_mask)
        roc_auc = roc_auc_score(y_oracle[tgt_test_mask], probs[:, 1][tgt_test_mask])
        ap = average_precision_score(y_oracle[tgt_test_mask], probs[:, 1][tgt_test_mask])
        f1 = f1_score(y_oracle[tgt_test_mask], np.argmax(probs, axis=1)[tgt_test_mask])
        self.log("test/performance.AU-ROC", roc_auc, on_step=False, on_epoch=True)
        self.log("test/performance.AP", ap, on_step=False, on_epoch=True)
        self.log("test/performance.F1", f1, on_step=False, on_epoch=True)

        tgt_roc_auc = roc_auc_score(y_oracle[tgt_mask], probs[:, 1][tgt_mask])
        tgt_ap = average_precision_score(y_oracle[tgt_mask], probs[:, 1][tgt_mask])
        tgt_f1 = f1_score(y_oracle[tgt_mask], np.argmax(probs, axis=1)[tgt_mask])
        self.log("tgt/performance.AU-ROC", tgt_roc_auc, on_step=False, on_epoch=True)
        self.log("tgt/performance.AP", tgt_ap, on_step=False, on_epoch=True)
        self.log("tgt/performance.F1", tgt_f1, on_step=False, on_epoch=True)

        self.test_step_outputs = []

    def configure_optimizers(self):
        return self.optimizer

