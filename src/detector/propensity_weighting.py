import numpy as np
from copy import deepcopy
import lightning as L
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from abstention.calibration import VectorScaling
from src.utils.model_utils import get_model_optimizer

class PropensityWeighting(L.LightningModule):
    def __init__(
            self,
            model_type: str,
            arch_param,
            dataset_name,
            novel_cls,
            learning_rate,
            weight_decay,
            max_epochs,
            pe_patience,
            seed):
        super().__init__()
        self.model_type = model_type
        self.novel_cls = novel_cls
        self.seed = seed

        self.dataset_name = dataset_name

        self.novelty_detector, self.detector_optimizer = get_model_optimizer(model_type, arch_param, learning_rate, weight_decay)
        self.propensity_estimator, self.propensity_optimizer = get_model_optimizer(model_type, arch_param, learning_rate, weight_decay)

        self.max_epochs = max_epochs
        self.pe_epochs = self.max_epochs // 2
        self.pe_staleness = 0
        self.pe_patience = pe_patience


        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warm_start = True
        self.best_estimator_selection_score = 0.

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.automatic_optimization = False

    def forward(self, model, data):
        if self.model_type == "mlp":
            return model(data.x)
        else:
            return model(data.x, data.edge_index)

    def process_batch(self, batch, stage):
        y = batch.tgt_mask.type(torch.int64)
        y_oracle = torch.zeros_like(batch.tgt_mask, dtype=torch.int64)
        y_oracle[batch.y == self.novel_cls] = 1
        # specify train/val/test masks
        if stage == "train":
            mask = batch.train_mask
        elif stage == "val":
            mask = batch.val_mask
        elif stage == "test":
            mask = batch.test_mask

        # forward and calculate logits
        if self.warm_start and stage != "test":
            logits_propensity = self.forward(self.propensity_estimator, batch)
            loss = F.cross_entropy(logits_propensity[mask], y[mask])
        else:
            logits_detector = self.forward(self.novelty_detector, batch)
            # # make a copy of source weights (s = 1 term in Bekker et al.)
            # # y_s_pseudo = 1 indicates synthetic target data points with 1 - 1 / e weight
            # y_s_pseudo = torch.randint(2, batch.src_mask[mask].sum().item())
            # sample_weights = 1. / self.propensity_src[mask].detach()
            # sample_weights[y_s_pseudo == 1] = 1 - sample_weights[y_s_pseudo == 1]
            # sample_weights = torch.cat([sample_weights, torch.ones(batch.tgt_mask[mask].sum().item())], dim=0)
            # # reorder to src train then tgt train
            # logits_detector_reordered = torch.cat([logits_detector[torch.logical_and(batch.src_mask, mask)],
            #                                        logits_detector[torch.logical_and(batch.tgt_mask, mask)]], dim=0)
            # loss = F.cross_entropy(logits_detector_reordered,
            #                        torch.cat([y_s_pseudo, torch.ones(batch.tgt_mask[mask].sum().item())], dim=0),
            #                        reduction="none")
            # loss = torch.mean(loss * sample_weights)

            weights_src_train = ((1. - y) / self.propensity)[mask] # shape = (train/val/test num,)
            weights_tgt_train = (y + (1. - y) * (1. - 1. / self.propensity))[mask] # shape = (train/val/test num,)
            pseudo_logits_detector = torch.cat([logits_detector[mask], logits_detector[mask]], dim=0) # shape = (2 * train/val/test num, 2)
            pseudo_y = torch.cat([torch.ones(mask.sum().item()), torch.zeros(mask.sum().item())], dim=0).type_as(y) # shape = (2 * train/val/test num,)
            weights = torch.cat([weights_src_train, weights_tgt_train], dim=0)
            loss = F.cross_entropy(pseudo_logits_detector, pseudo_y, reduction="none")
            loss = torch.mean(loss * weights)
        if stage == "train":
            propensity_optimizer, detector_optimizer = self.optimizers()
            if self.warm_start:
                propensity_optimizer.zero_grad()
                self.manual_backward(loss)
                propensity_optimizer.step()
            else:
                detector_optimizer.zero_grad()
                self.manual_backward(loss)
                detector_optimizer.step()

            return loss

        elif stage == "val":
            if self.warm_start:
                logits = logits_propensity
            else:
                logits = logits_detector
            probs = F.softmax(logits, dim=1)

            return loss, logits, probs, y, y_oracle, batch.tgt_mask, mask

        elif stage == "test":
            probs = F.softmax(logits_detector, dim=1)
            return loss, logits_detector, probs, y, y_oracle, batch.tgt_mask, mask

        else:
            raise ValueError(f"Invalid stage: {stage}")


    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch, "train")
        batch_size = batch.train_mask.sum().item()
        if self.warm_start:
            self.log("train/loss", np.inf, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log("train/pe_loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        else:
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        return {"loss": loss.detach()}


    def validation_step(self, batch, batch_idx):
        loss, logits, probs, y, y_oracle, tgt_mask, val_mask = self.process_batch(batch, "val")
        batch_size = val_mask.sum().item()
        if self.warm_start:
            self.log("val/loss", np.inf, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log("val/pe_loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        else:
            self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        outputs = {"loss": loss.detach(),
                   "logits": logits,
                   "probs": probs,
                   "y": y,
                   "y_oracle": y_oracle,
                   "tgt_mask": tgt_mask,
                   "val_mask": val_mask}
        self.validation_step_outputs.append(outputs)
        return outputs


    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        logits = torch.cat([o["logits"] for o in outputs], dim=0).detach().cpu().numpy()
        probs = torch.cat([o["probs"] for o in outputs], dim=0).detach().cpu().numpy()
        y = torch.cat([o["y"] for o in outputs], dim=0).detach().cpu().numpy()
        y_oracle = torch.cat([o["y_oracle"] for o in outputs], dim=0).detach().cpu().numpy()
        # y_original = torch.cat([o["y_original"] for o in outputs], dim=0).detach().cpu().numpy()
        tgt_mask = torch.cat([o["tgt_mask"] for o in outputs], dim=0).detach().cpu().numpy()
        val_mask = torch.cat([o["val_mask"] for o in outputs], dim=0).detach().cpu().numpy()


        if self.warm_start:
            y_pred = np.argmax(probs, axis=1)
            estimator_selection_score = accuracy_score(y[val_mask], y_pred[val_mask])
            self.log("val/estimator_selection_score", estimator_selection_score, on_step=False, on_epoch=True, prog_bar=False, batch_size=val_mask.sum().item())
            encoder = OneHotEncoder(sparse_output=False, categories=[range(2)])
            calibrator = VectorScaling()(logits[val_mask], encoder.fit_transform(y[val_mask].reshape(-1, 1)))

            if estimator_selection_score > self.best_estimator_selection_score:
                self.best_estimator_selection_score = estimator_selection_score
                self.pe_staleness = 0
                self.propensity = torch.from_numpy(calibrator(logits)[:,0]).type_as(outputs[0]["logits"])
                self.best_propensity_estimator = deepcopy(self.propensity_estimator)
            else:
                self.pe_staleness += 1
        else:
            tgt_val_mask = np.logical_and(tgt_mask, val_mask)
            roc_auc = roc_auc_score(y_oracle[tgt_val_mask], probs[:, 1][tgt_val_mask])
            self.log("val/performance.AU-ROC", roc_auc, on_step=False, on_epoch=True)

        if self.current_epoch >= self.pe_epochs or self.pe_staleness >= self.pe_patience:
            self.warm_start = False
            self.propensity_estimator = deepcopy(self.best_propensity_estimator)
        else:
            self.warm_start = True

        self.validation_step_outputs = []

    def test_step(self, batch, batch_idx):
        loss, logits, probs, y, y_oracle, tgt_mask, test_mask = self.process_batch(batch, "test")
        batch_size = test_mask.sum().item()
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        outputs = {"loss": loss.detach(),
                   "logits": logits,
                   "probs": probs,
                   "y": y,
                   "y_oracle": y_oracle,
                   "tgt_mask": tgt_mask,
                   "test_mask": test_mask}
        self.test_step_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
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

        self.test_step_outputs = []

    def configure_optimizers(self):
        return [self.propensity_optimizer, self.detector_optimizer]