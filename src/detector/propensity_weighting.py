import numpy as np
import lightning as L
import torch
from sklearn.metrics import roc_auc_score
from src.utils.model_utils import get_model_optimizer


class PropensityWeighting(L.LightningModule):
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
            seed):
        super().__init__()
        self.model_type = model_type
        self.novel_cls = novel_cls
        self.constrained_penalty = constrained_penalty
        self.seed = seed

        self.dataset_name = dataset_name

        self.novelty_detector, self.detector_optimizer = get_model_optimizer(model_type, arch_param, learning_rate, weight_decay)
        self.ratio_estimator, self.ratio_optimizer = get_model_optimizer(model_type, arch_param, learning_rate, weight_decay)

        self.max_epochs = max_epochs
        self.density_estimation_epochs = self.max_epochs // 2

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warm_start = True

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.automatic_optimization = False

    def forward(self, data):
        if self.model_type == "mlp":
            return self.model(data.x)
        else:
            return self.model(data.x, data.edge_index)

    def process_batch(self, batch, stage):
        pass

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch, "train")
        batch_size = batch.train_mask.sum().item()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        return {"loss": loss.detach()}

    def on_train_epoch_end(self):
        if self.current_epoch < self.density_estimation_epochs:
            self.warm_start = True
        else:
            self.warm_start = False

    def validation_step(self, batch, batch_idx):
        loss, probs, y, y_oracle, tgt_mask, val_mask = self.process_batch(batch, "val")
        batch_size = val_mask.sum().item()
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        outputs = {"loss": loss.detach(),
                   "probs": probs,
                   "y": y,
                   "y_oracle": y_oracle,
                   "tgt_mask": tgt_mask,
                   "val_mask": val_mask}
        self.validation_step_outputs.append(outputs)
        return outputs


    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        probs = torch.cat([o["probs"] for o in outputs], dim=0).detach().cpu().numpy()
        y_oracle = torch.cat([o["y_oracle"] for o in outputs], dim=0).detach().cpu().numpy()
        tgt_mask = torch.cat([o["tgt_mask"] for o in outputs], dim=0).detach().cpu().numpy()
        val_mask = torch.cat([o["val_mask"] for o in outputs], dim=0).detach().cpu().numpy()

        # MP estimate
        MP_estimate_BBE = 1. - BBE_estimate_binary()
        MP_estimate_dedpul = 1. - dedpul()
        pure_bin_estimate = pure_MPE_estimator()
        true_label_dist = None

        self.log("val/MPE.BBE", MP_estimate_BBE)
        self.log("val/MPE.dedpul", MP_estimate_dedpul)
        self.log("val/MPE.pure_bin", pure_bin_estimate)
        self.log("val/MPE.true", true_label_dist)

        tgt_val_mask = np.logical_and(tgt_mask, val_mask)
        roc_auc = roc_auc_score(y_oracle[tgt_val_mask], probs[:, 1][tgt_val_mask])
        self.log("val/performance.AU-ROC", roc_auc, on_step=False, on_epoch=True)

        if self.warm_start:
            pass
        else:



    def test_step(self, batch, batch_idx):
        loss, probs, y, y_oracle, tgt_mask, test_mask = self.process_batch(batch, "test")
        batch_size = test_mask.sum().item()
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        outputs = {"loss": loss.detach(),
                   "probs": probs,
                   "y": y,
                   "y_oracle": y_oracle,
                   "tgt_mask": tgt_mask,
                   "test_mask": test_mask}
        self.test_step_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        pass