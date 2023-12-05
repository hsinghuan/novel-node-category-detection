import numpy as np
from sklearn.metrics import roc_auc_score
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import cooper

from src.utils.core_utils import recall_from_logits
from src.utils.model_utils import get_model_optimizer


class RecallConstrainedNodeClassification(cooper.ConstrainedMinimizationProblem):
    def __init__(self, target_recall, wd, penalty_type, logit_multiplier, device, mode="domain_disc"):
        self.criterion = nn.CrossEntropyLoss()
        self.target_recall = target_recall
        self.wd = wd
        self.penalty_type = penalty_type
        self.logit_multiplier = logit_multiplier
        self.device = device
        if mode == "constrained_opt":
            super().__init__(is_constrained=True)
        else:
            super().__init__(is_constrained=False)

    def get_penalty(self, model):
        penalty_lambda = self.wd
        if self.penalty_type == "l2":
            penalty_term = sum(p.pow(2.0).sum() for p in model.parameters())
        else:
            penalty_term = sum(torch.abs(p).sum() for p in model.parameters())
        return penalty_lambda * penalty_term

    def closure(self, model, data, targets, mask):
        pred_logits = model.forward(data.x, data.edge_index) # should specify eval or train?
        pred_logits = pred_logits.reshape(pred_logits.shape[0], -1, 2) # num nodes x num recall constraints (heads) x binary labels
        # with torch.no_grad():
        #     preds = torch.argmax(pred_logits, dim=-1)
        pred_logits = pred_logits[mask] # mask by train/val etc
        targets = targets[mask]

        penalty = self.get_penalty(model)
        cross_ent_ls = []
        recall_ls = []
        recall_proxy_ls = []
        recall_loss_ls = []
        # pred_temp_ls = []
        cross_ent_target_ls = [] # torch.tensor([], requires_grad=True, device=self.device)

        for i in range(pred_logits.shape[1]): # iterate through recall constraints
            cross_ent = self.criterion(pred_logits[targets==0][:,i,:], targets[targets==0]) # why mask with target == 0
            cross_ent_target = self.criterion(pred_logits[targets==1][:,i,:], targets[targets==1]) # why mask with target == 1
            recall, recall_proxy, recall_loss = recall_from_logits(self.logit_multiplier * pred_logits[:,i,:], targets)

            cross_ent_ls.append(cross_ent.item())
            cross_ent_target_ls.append(cross_ent_target.item())
            recall_ls.append(recall.item())
            recall_proxy_ls.append(recall_proxy.item())
            recall_loss_ls.append(recall_loss.item())

        cross_ent_ls = torch.tensor(cross_ent_ls, device=self.device)
        recall_ls = torch.tensor(recall_ls, device=self.device)
        recall_proxy_ls = torch.tensor(recall_proxy_ls, device=self.device)
        recall_loss_ls = torch.tensor(recall_loss_ls, device=self.device)
        cross_ent_target_ls = torch.tensor(cross_ent_target_ls, device=self.device)

        loss = cross_ent_ls + penalty
        loss_target = cross_ent_target + penalty

        ineq_defect = torch.tensor(self.target_recall, device=self.device) - recall_ls
        proxy_ineq_defect = torch.tensor(self.target_recall, device=self.device) - recall_proxy_ls

        print(f"loss: {loss}")
        print(f"loss target: {loss_target}")
        print(f"cross ent list: {cross_ent_ls}")
        print(f"cross ent target list: {cross_ent_target_ls}")
        print(f"recall list: {recall_ls}")
        print(f"inequality defect: {ineq_defect}")
        print(f"proxy ineq defect: {proxy_ineq_defect}")

        return cooper.CMPState(loss=loss,
                               ineq_defect=ineq_defect,
                               proxy_ineq_defect=proxy_ineq_defect,
                               eq_defect=None,
                               misc={"cross_ent": cross_ent_ls,
                                    "cross_ent_target": cross_ent_target_ls,
                                    "recall_proxy": recall_proxy_ls,
                                    "recall_loss": recall_loss_ls})


class CoNoC(L.LightningModule):
    def __init__(self,
                 mode,
                 model_type: str,
                 arch_param,
                 dataset_name,
                 novel_cls,
                 target_precision,
                 precision_confidence,
                 logit_multiplier,
                 constrained_penalty,
                 learning_rate,
                 dual_learning_rate,
                 weight_decay,
                 max_epochs,
                 warmup_epochs,
                 seed,
                 device,
                 penalty_type="l2"):
        super().__init__()

        self.target_recalls = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45]
        self.num_outputs = 2 * len(self.target_recalls)
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.novel_cls = novel_cls
        self.learning_rate = learning_rate
        self.dual_learning_rate = dual_learning_rate
        self.weight_decay = weight_decay
        self.constrained_penalty = constrained_penalty
        self.penalty_type = penalty_type
        self._device = device
        self.mode = mode

        self.model, self.primal_optimizer = get_model_optimizer(model_type, arch_param, learning_rate, weight_decay)
        self.target_precision = target_precision
        self.precision_confidence = precision_confidence

        if self.mode == "constrained_opt":
            self.dual_optimizer = cooper.optim.partial_optimizer(torch.optim.Adam, lr=dual_learning_rate, weight_decay=weight_decay)
            self.cmp = RecallConstrainedNodeClassification(target_recall=self.target_recalls,
                                                           wd=constrained_penalty,
                                                           penalty_type=penalty_type,
                                                           logit_multiplier=logit_multiplier,
                                                           device=device,
                                                           mode=self.mode)
            self.formulation = cooper.LagrangianFormulation(self.cmp, ineq_init=torch.tensor([1. for _ in range(len(self.target_recalls))]))
            self.coop = cooper.ConstrainedOptimizer(self.formulation, self.primal_optimizer, self.dual_optimizer)
        else:
            self.dual_optimizer = None
            self.dual_lr_scheduler = None

        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.seed = seed
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
        if stage == "train":
            mask = batch.train_mask
            y = batch.tgt_mask.type(torch.int64)
            if self.warm_start:
                logits = self.forward(batch)
                logits = logits.reshape(logits.shape[0], -1, 2)
                loss_ls = []
                for i in range(logits.shape[1]):
                    loss = nn.CrossEntropyLoss(logits[:,i,:], y)
                    loss_ls.append(loss.item())
                loss_ls = torch.tensor(loss_ls, device=self._device)
                self.primal_optimizer.zero_grad()
                self.manual_backward(loss_ls.sum())
                self.primal_optimizer.step()
                return loss_ls, torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
            else:
                lagrangian = self.formulation.composite_objective
                self.formulation.custom_backward(lagrangian)
                self.coop.step(self.cmp.closure, self.model, batch, y, mask)
                return self.cmp.state.loss, self.cmp.get_penalty(self.model), self.cmp.state.ineq_defect, lagrangian
        elif stage == "val" or stage == "test":
            if stage == "val":
                mask = batch.val_mask
            elif stage == "test":
                mask = batch.test_mask
            logits = self.forward(batch)
            logits = logits.reshape(logits.shape[0], -1, 2)
            probs = F.softmax(logits, dim=-1)
            return probs, batch.tgt_mask, mask

    def training_step(self, batch, batch_idx):
        loss, penalty, ineq_defect, lagrangian_value = self.process_batch(batch, "train")
        self.log("train/loss.constraint_penalty", penalty, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/loss.lagrangian", lagrangian_value, on_step=True, on_epoch=True, prog_bar=False)
        for i in range(len(self.target_recalls)):
            self.log("train/loss.cross_ent_" + str(self.target_recalls[i]), loss[i], on_step=True, on_epoch=True, prog_bar=False)

        if not self.warm_start and self.mode.startwith("constrained"):
            for i in range(len(self.target_recalls)):
                self.log("train/constraints.inequality_violation_" + str(self.target_recalls[i]), ineq_defect[i], on_step=True, on_epoch=True, prog_bar=False)
                self.log("train/constraints.multiplier_value_" + str(self.target_recalls[i]), self.formulation.ineq_multipliers.weight.detach().cpu()[i], on_step=True, on_epoch=True, prog_bar=False)

        return {"lagrangian_loss": lagrangian_value}

    def on_training_epoch_end(self):
        if self.current_epoch >= self.warmup_epochs:
            self.warm_start = False

    def validation_step(self, batch, batch_idx):
        probs, tgt_mask, val_mask = self.process_batch(batch, "val")
        y_oracle = torch.zeros_like(batch.y, dtype=torch.int64)
        y_oracle[batch.y == self.novel_cls] = 1
        outputs = {"probs": probs,
                   "y_oracle": y_oracle,
                   "tgt_mask": tgt_mask,
                   "val_mask": val_mask}
        self.validation_step_outputs.append(outputs)
        return outputs

    def test_step(self, batch, batch_idx):
        probs, tgt_mask, test_mask = self.process_batch(batch, "test")
        y_oracle = torch.zeros_like(batch.y, dtype=torch.int64)
        y_oracle[batch.y == self.novel_cls] = 1
        outputs = {"probs": probs,
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

        tgt_val_mask = np.logical_and(tgt_mask, val_mask)

        for i in range(len(self.target_recalls)):
            roc_auc = roc_auc_score(y_oracle[tgt_val_mask], probs[:, i, 1][tgt_val_mask])
            self.log("val/performance.AU-ROC_" + str(self.target_recalls[i]), roc_auc, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        probs = torch.cat([o["probs"] for o in outputs], dim=0).detach().cpu().numpy()
        y_oracle = torch.cat([o["y_oracle"] for o in outputs], dim=0).detach().cpu().numpy()
        tgt_mask = torch.cat([o["tgt_mask"] for o in outputs], dim=0).detach().cpu().numpy()
        test_mask = torch.cat([o["test_mask"] for o in outputs], dim=0).detach().cpu().numpy()

        tgt_test_mask = np.logical_and(tgt_mask, test_mask)

        for i in range(len(self.target_recalls)):
            roc_auc = roc_auc_score(y_oracle[tgt_test_mask], probs[:, i, 1][tgt_test_mask])
            self.log("val/performance.AU-ROC_" + str(self.target_recalls[i]), roc_auc, on_step=False, on_epoch=True)

        for i in range(len(self.target_recalls)):
            tgt_roc_auc = roc_auc_score(y_oracle[tgt_mask], probs[:, i, 1][tgt_mask])
            self.log("tgt/performance.AU-ROC_" + str(self.target_recalls[i]), tgt_roc_auc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return self.primal_optimizer