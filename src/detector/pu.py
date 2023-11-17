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
            model_name,
            dataset,
            num_src_cls,
            fraction_ood_cls,
            ood_cls_ratio,
            constrained_penalty,
            learning_rate,
            weight_decay,
            target_precision,
            precision_confidence,
            max_epochs,
            seed,):
        super().__init__()

        self.model_name = model_name
        self.num_cls = num_src_cls
        self.fraction_ood_cls = fraction_ood_cls
        self.ood_cls_ratio = ood_cls_ratio
        self.constrained_penalty = constrained_penalty
        self.seed = seed

        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss()

        self.num_outputs = self.num_cls

        arch_param = None
        self.oracle_model, self.oracle_optimizer = get_model_optimizer(model_name, arch_param, learning_rate, weight_decay)

        self.target_precision = target_precision
        self.precision_confidence = precision_confidence

        self.discriminator, self.optimizer = get_model_optimizer(model_name, arch_param, learning_rate, weight_decay)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_epochs = 0
        self.warm_start = False

        self.validation_step_outputs = []

    def forward_oracle(self, data):
        if self.model_name == "mlp":
            return self.oracle_model(data.x)
        else:
            return self.oracle_model(data.x, data.edge_index)

    def forward_discriminator(self, data):
        if self.model_name == "mlp":
            return self.discriminator(data.x)
        else:
            return self.discriminator(data.x, data.edge_index)

    def get_penalty(self, model, penalty_type="l2", wd=0.01):
        penalty_lambda = wd
        if penalty_type == "l2":
            penalty_term = sum(p.pow(2.0).sum() for p in model.parameters())
        else:
            penalty_term = sum(torch.abs(p).sum() for p in model.parameters())
        return penalty_lambda * penalty_term

    def process_batch(self, batch, stage):
        y_oracle = batch.y[batch.y == self.num_cls].type(torch.int64)
        y_disc = batch.tgt_mask.type(torch.int64)
        logits_oracle = self.forward_oracle(batch)
        logits_disc = self.forward_discriminator(batch)
        if self.warm_start:
            pass
        else:
            loss_oracle = F.cross_entropy(logits_oracle[batch.train_mask], y_oracle[batch.train_mask])
            loss_disc = F.cross_entropy(logits_disc[batch.train_mask], y_disc[batch.train_mask])

        if stage == "train":
            oracle_optimizer, optimizer = self.optimizers()
            oracle_optimizer.zero_grad()
            self.manual_backward(loss_oracle)
            oracle_optimizer.step()

            optimizer.zero_grad()
            self.manual_backward(loss_disc)
            optimizer.step()
            return loss_oracle, loss_disc

        elif stage == "val":
            mask = batch.val_mask
            loss_oracle = F.cross_entropy(logits_oracle[mask], y_oracle[mask])
            loss_disc = F.cross_entropy(logits_disc[mask], y_disc[mask])

            probs_oracle = F.softmax(logits_oracle[mask], dim=1)
            probs_disc = F.softmax(logits_disc[mask], dim=1)
            return loss_oracle, loss_disc, probs_oracle, probs_disc, y_oracle, batch.tgt_mask, mask # batch.y[mask]

        else:
            raise ValueError(f"Invalid stage {stage}")

    def training_step(self, batch, batch_idx):
        loss_oracle, loss_disc = self.process_batch(batch, "train")
        self.log("train/loss.oracle", loss_oracle, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/loss.discriminator", loss_disc, on_step=True, on_epoch=True, prog_bar=False)
        return {"oracle_loss": loss_oracle.detach(),
                "loss": loss_disc.detach()}

    def validation_step(self, batch, batch_idx):
        loss_oracle, loss_disc, probs_oracle, probs_disc, y_oracle, tgt_mask, val_mask = self.process_batch(batch, "val")
        self.log("val/loss.oracle", loss_oracle, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/loss.discriminator", loss_disc, on_step=True, on_epoch=True, prog_bar=False)
        outputs =  {"loss_oracle": loss_oracle.detach(),
                    "probs_oracle": probs_oracle,
                    "loss": loss_disc.detach(),
                    "probs_disc": probs_disc,
                    "y_oracle": y_oracle,
                    "tgt_mask": tgt_mask,
                    "val_mask": val_mask}
        self.validation_step_outputs.append(outputs)
        return outputs

    def on_train_epoch_end(self):
        if self.current_epoch < self.warmup_epochs:
            self.warm_start = True
        else:
            self.warm_start = False

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        probs_oracle = torch.cat([o["probs_oracle"] for o in outputs], dim=0).detach().cpu().numpy()
        probs_disc = torch.cat([o["probs_disc"] for o in outputs], dim=0).detach().cpu().numpy()
        y_oracle = torch.cat([o["y_oracle"] for o in outputs], dim=0).detach().cpu().numpy()
        tgt_mask = torch.cat([o["tgt_mask"] for o in outputs], dim=0).detach().cpu().numpy()
        val_mask = torch.cat([o["val_mask"] for o in outputs], dim=0).detach().cpu().numpy()

        # compute roc_auc_score, average precision
        tgt_val_mask = np.logical_and(tgt_mask, val_mask)
        roc_auc_disc = roc_auc_score(y_oracle[tgt_val_mask], probs_disc[:, 1][tgt_val_mask])
        ap_disc = average_precision_score(y_oracle[tgt_val_mask], probs_disc[:, 1][tgt_val_mask])
        roc_auc_oracle = roc_auc_score(y_oracle[tgt_val_mask], probs_oracle[:, 1][tgt_val_mask])
        ap_oracle = average_precision_score(y_oracle[tgt_val_mask], probs_oracle[:, 1][tgt_val_mask])
        precision_disc, recall_disc, _ = precision_recall_curve(y_oracle[tgt_val_mask], probs_disc[:, 1][tgt_val_mask])
        precision_oracle, recall_oracle, _ = precision_recall_curve(y_oracle[tgt_val_mask], probs_oracle[:, 1][tgt_val_mask])
        auc_pr_disc = auc(recall_disc, precision_disc)
        auc_pr_oracle = auc(recall_oracle, precision_oracle)

        self.log("pred/performance.AU-ROC", roc_auc_disc, on_step=False, on_epoch=True)
        self.log("pred/performance.oracle AU-ROC", roc_auc_oracle, on_step=False, on_epoch=True)
        self.log("pred/performance.AP", ap_disc, on_step=False, on_epoch=True)
        self.log("pred/performance.oracle AP", ap_oracle, on_step=False, on_epoch=True)
        self.log("pred/performance.AU-PR", auc_pr_disc, on_step=False, on_epoch=True)
        self.log("pred/performance.oracle AU-PR", auc_pr_oracle, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return self.oracle_optimizer, self.optimizer

