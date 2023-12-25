import numpy as np
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

from src.utils.model_utils import get_model_optimizer

class VanillaPU(L.LightningModule):
    def __init__(
            self,
            model_type: str,
            arch_param,
            dataset_name,
            novel_cls,
            constrained_penalty,
            learning_rate,
            weight_decay,
            max_epochs,
            seed,
            oracle=False):
        super().__init__()
        self.model_type = model_type
        self.novel_cls = novel_cls
        self.constrained_penalty = constrained_penalty
        self.seed = seed

        self.dataset_name = dataset_name
        self.criterion = nn.CrossEntropyLoss()

        self.model, self.optimizer = get_model_optimizer(model_type, arch_param, learning_rate, weight_decay)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.oracle = oracle
        self.warmup_epochs = max_epochs # l2 regularize all the time
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

    def get_penalty(self, model, penalty_type="l2", wd=0.01):
        penalty_lambda = wd
        if penalty_type == "l2":
            penalty_term = sum(p.pow(2.0).sum() for p in model.parameters())
        else:
            penalty_term = sum(torch.abs(p).sum() for p in model.parameters())
        return penalty_lambda * penalty_term

    def process_batch(self, batch, stage):
        y_oracle = torch.zeros_like(batch.tgt_mask, dtype=torch.int64)
        y_oracle[batch.y == self.novel_cls] = 1
        if self.oracle:
            y = y_oracle
        else:
            y = batch.tgt_mask.type(torch.int64)
        logits = self.forward(batch)

        if stage == "train":
            mask = batch.train_mask
        elif stage == "val":
            mask = batch.val_mask
        elif stage == "test":
            mask = batch.test_mask

        loss = F.cross_entropy(logits[mask], y[mask])

        if self.warm_start:
            loss += self.get_penalty(self.model, wd=self.constrained_penalty)
        if stage == "train":
            optimizer = self.optimizers()
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            return loss

        elif stage == "val" or stage == "test":
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
        loss, probs, y, y_oracle, tgt_mask, val_mask = self.process_batch(batch, "val")
        batch_size = batch.val_mask.sum().item()
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        outputs =  {"loss": loss.detach(),
                    "probs": probs,
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

    def on_train_epoch_end(self):
        if self.current_epoch < self.warmup_epochs:
            self.warm_start = True
        else:
            self.warm_start = False

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        probs = torch.cat([o["probs"] for o in outputs], dim=0).detach().cpu().numpy()
        y_oracle = torch.cat([o["y_oracle"] for o in outputs], dim=0).detach().cpu().numpy()
        tgt_mask = torch.cat([o["tgt_mask"] for o in outputs], dim=0).detach().cpu().numpy()
        val_mask = torch.cat([o["val_mask"] for o in outputs], dim=0).detach().cpu().numpy()

        # compute roc_auc_score, average precision
        tgt_val_mask = np.logical_and(tgt_mask, val_mask)
        # print(tgt_val_mask.sum().item())
        # print(f"y oracle: {y_oracle[tgt_val_mask]}")
        # print(f"probs: {probs[:, 1][tgt_val_mask]}")
        roc_auc = roc_auc_score(y_oracle[tgt_val_mask], probs[:, 1][tgt_val_mask])

        self.log("val/performance.AU-ROC", roc_auc, on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        outputs = self.test_step_outputs
        probs = torch.cat([o["probs"] for o in outputs], dim=0).detach().cpu().numpy()
        y_oracle = torch.cat([o["y_oracle"] for o in outputs], dim=0).detach().cpu().numpy()
        tgt_mask = torch.cat([o["tgt_mask"] for o in outputs], dim=0).detach().cpu().numpy()
        test_mask = torch.cat([o["test_mask"] for o in outputs], dim=0).detach().cpu().numpy()

        # compute roc_auc_score, average precision
        tgt_test_mask = np.logical_and(tgt_mask, test_mask)
        roc_auc = roc_auc_score(y_oracle[tgt_test_mask], probs[:, 1][tgt_test_mask])
        self.log("test/performance.AU-ROC", roc_auc, on_step=False, on_epoch=True)

        tgt_roc_auc = roc_auc_score(y_oracle[tgt_mask], probs[:, 1][tgt_mask])
        self.log("tgt/performance.AU-ROC", tgt_roc_auc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return self.optimizer

