import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import lightning as L
import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph, negative_sampling, structured_negative_sampling
from src.utils.model_utils import get_model_optimizer
from src.utils.mpe_utils import p_probs, u_probs, BBE_estimator

class PUGNN(L.LightningModule):
    def __init__(self,
                 mode,
                 model_type,
                 arch_param,
                 dataset_name,
                 novel_cls,
                 learning_rate,
                 max_epochs,
                 warmup_epochs,
                 seed,
                 weight_decay=0.,
                 reg_loss_weight=1e-2
                 ):
        super().__init__()

        self.model_type = model_type
        self.dataset_name = dataset_name
        self.novel_cls = novel_cls
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.mode = mode
        self.reg_loss_weight = reg_loss_weight

        self.model, self.optimizer = get_model_optimizer(model_type, arch_param, learning_rate, weight_decay)

        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
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

    def pu_loss(self, data, logits, y):
        # pass
        return 0.

    def reg_loss(self, data):
        # pass
        return 0.

    def process_batch(self, batch, stage): # , batch_linkpred=None):
        y = batch.tgt_mask.type(torch.int64)

        if stage == "train":
            mask = batch.train_mask
        elif stage == "val":
            mask = batch.val_mask
        elif stage == "test":
            mask = batch.test_mask

        logits = self.forward(batch)

        if self.warm_start:
            loss = F.cross_entropy(logits[mask], y[mask])
            pu_loss, reg_loss = np.inf, np.inf
        else:
            pu_loss = self.loss(batch, logits, y, mask)
            reg_loss = self.reg_loss(batch, mask)
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
            else:
                pos_probs, unlabeled_probs, unlabeled_targets = None, None, None
            return loss, pu_loss, reg_loss, probs, pos_probs, unlabeled_probs, unlabeled_targets
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
        loss, pu_loss, reg_loss, probs, pos_probs, unlabeled_probs, unlabeled_targets = self.process_batch(batch, "val") # , batch_linkpred)
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
                   "y_oracle": y_oracle,
                   "tgt_mask": batch.tgt_mask,
                   "val_mask": batch.val_mask}
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

        tgt_val_mask = np.logical_and(tgt_mask, val_mask)
        batch_size = tgt_val_mask.sum()

        roc_auc = roc_auc_score(y_oracle[tgt_val_mask], probs[:, 1][tgt_val_mask])
        self.log("val/performance.AU-ROC", roc_auc, on_step=False, on_epoch=True, batch_size=batch_size)

        self.validation_step_outputs = []

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        probs = torch.cat([o["probs"] for o in outputs], dim=0).detach().cpu().numpy()
        y_oracle = torch.cat([o["y_oracle"] for o in outputs], dim=0).detach().cpu().numpy()
        tgt_mask = torch.cat([o["tgt_mask"] for o in outputs], dim=0).detach().cpu().numpy()
        test_mask = torch.cat([o["test_mask"] for o in outputs], dim=0).detach().cpu().numpy()

        tgt_test_mask = np.logical_and(tgt_mask, test_mask)
        roc_auc = roc_auc_score(y_oracle[tgt_test_mask], probs[:, 1][tgt_test_mask])
        self.log("test/performance.AU-ROC", roc_auc, on_step=False, on_epoch=True, batch_size=tgt_test_mask.sum())

        tgt_roc_auc = roc_auc_score(y_oracle[tgt_mask], probs[:, 1][tgt_mask])
        self.log("tgt/performance.AU-ROC", tgt_roc_auc, on_step=False, on_epoch=True, batch_size=tgt_mask.sum())

        self.test_step_outputs = []

    def configure_optimizers(self):
        return self.optimizer
