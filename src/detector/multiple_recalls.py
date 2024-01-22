from typing import Optional, Callable
import numpy as np
from sklearn.metrics import roc_auc_score
import lightning as L
from lightning.pytorch.core.optimizer import LightningOptimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import cooper

from src.utils.core_utils import recall_from_logits, fpr_from_logits
from src.utils.model_utils import get_model_optimizer
from torchviz import make_dot

class RecallConstrainedNodeClassification(cooper.ConstrainedMinimizationProblem):
    def __init__(self, target_recall, wd, penalty_type, logit_multiplier, mode="domain_disc"):
        self.target_recall = target_recall
        self.wd = wd
        self.penalty_type = penalty_type
        self.logit_multiplier = logit_multiplier
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

    def closure(self, model, pred_logits, y):
        pred_logits = pred_logits.reshape(pred_logits.shape[0], -1, 2) # num nodes x num recall constraints (heads) x binary labels

        penalty = self.get_penalty(model)
        cross_ent_ls = torch.tensor([], requires_grad=True).to(pred_logits)
        fpr_ls = torch.tensor([], requires_grad=True).to(pred_logits)
        fpr_proxy_ls = torch.tensor([], requires_grad=True).to(pred_logits)
        recall_ls = torch.tensor([], requires_grad=True).to(pred_logits)
        recall_proxy_ls = torch.tensor([], requires_grad=True).to(pred_logits)
        recall_loss_ls = torch.tensor([], requires_grad=True).to(pred_logits)
        cross_ent_target_ls = torch.tensor([], requires_grad=True).to(pred_logits)
        loss = 0.

        for i in range(pred_logits.shape[1]): # iterate through recall constraints
            cross_ent = F.cross_entropy(pred_logits[y==0][:,i,:], y[y==0])
            cross_ent_target = F.cross_entropy(pred_logits[y==1][:,i,:], y[y==1])
            fpr, fpr_proxy = fpr_from_logits(self.logit_multiplier * pred_logits[:,i,:], y)
            recall, recall_proxy, recall_loss = recall_from_logits(self.logit_multiplier * pred_logits[:,i,:], y)
            loss += cross_ent # fpr_proxy

            cross_ent_ls = torch.cat((cross_ent_ls, torch.unsqueeze(cross_ent, 0)))
            cross_ent_target_ls = torch.cat((cross_ent_target_ls, torch.unsqueeze(cross_ent_target,0)))
            fpr_ls = torch.cat((fpr_ls, torch.unsqueeze(fpr,0)))
            fpr_proxy_ls = torch.cat((fpr_proxy_ls, torch.unsqueeze(fpr_proxy,0)))
            recall_ls = torch.cat((recall_ls, torch.unsqueeze(recall,0)))
            recall_proxy_ls = torch.cat((recall_proxy_ls, torch.unsqueeze(recall_proxy,0)))
            recall_loss_ls = torch.cat((recall_loss_ls, torch.unsqueeze(recall_loss,0)))


        ineq_defect = torch.tensor(self.target_recall).to(pred_logits) - recall_ls
        proxy_ineq_defect = torch.tensor(self.target_recall).to(pred_logits) - recall_proxy_ls

        return cooper.CMPState(loss=loss + penalty, # .sum(),
                               ineq_defect=ineq_defect,
                               proxy_ineq_defect=proxy_ineq_defect,
                               eq_defect=None,
                               misc={"cross_ent": cross_ent_ls,
                                     "cross_ent_target": cross_ent_target_ls,
                                     "fpr": fpr_ls,
                                     "fpr_proxy": fpr_proxy_ls,
                                     "recall": recall_ls,
                                     "recall_proxy": recall_proxy_ls,
                                     "recall_loss": recall_loss_ls})



class CoNoC(L.LightningModule):
    def __init__(self,
                 mode,
                 model_type,
                 arch_param,
                 dataset_name,
                 novel_cls,
                 target_recalls,
                 learning_rate,
                 dual_learning_rate,
                 max_epochs,
                 seed,
                 warmup_epochs=0,
                 weight_decay=0.,
                 penalty_type="l2",
                 constrained_penalty=0.,
                 logit_multiplier=1.,
                 lagrange_multiplier_init=0.1
                 ):
        super().__init__()

        self.target_recalls = target_recalls
        self.num_outputs = 2 * len(self.target_recalls)
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.novel_cls = novel_cls
        self.learning_rate = learning_rate
        self.dual_learning_rate = dual_learning_rate
        self.weight_decay = weight_decay
        self.constrained_penalty = constrained_penalty
        self.penalty_type = penalty_type
        self.mode = mode

        self.model, self.primal_optimizer = get_model_optimizer(model_type, arch_param, learning_rate, weight_decay)

        if self.mode == "constrained_opt":
            self.cmp = RecallConstrainedNodeClassification(target_recall=self.target_recalls,
                                                           wd=constrained_penalty,
                                                           penalty_type=penalty_type,
                                                           logit_multiplier=logit_multiplier,
                                                           mode=self.mode)
            self.formulation = cooper.LagrangianFormulation(self.cmp, ineq_init=torch.tensor([lagrange_multiplier_init for _ in range(len(self.target_recalls))]))
            self.dual_optimizer = cooper.optim.partial_optimizer(torch.optim.Adam, lr=dual_learning_rate, weight_decay=weight_decay)
            self.coop = cooper.ConstrainedOptimizer(formulation=self.formulation, primal_optimizer=self.primal_optimizer, dual_optimizer=self.dual_optimizer)
            self.coop.zero_grad()
        else:
            self.cmp = RecallConstrainedNodeClassification(target_recall=self.target_recalls,
                                                           wd=constrained_penalty,
                                                           penalty_type=penalty_type,
                                                           logit_multiplier=logit_multiplier,
                                                           mode=self.mode)
            self.formulation = cooper.LagrangianFormulation(self.cmp, ineq_init=torch.tensor([lagrange_multiplier_init for _ in range(len(self.target_recalls))]))
            self.dual_optimizer = None
            self.coop = cooper.ConstrainedOptimizer(formulation=self.formulation,
                                                    primal_optimizer=self.primal_optimizer,
                                                    dual_optimizer=self.dual_optimizer)
            self.coop.zero_grad()
        self.coop.sanity_checks()
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.warm_start = False if self.warmup_epochs == 0 else True
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

    def process_batch(self, batch, stage):
        y = batch.tgt_mask.type(torch.int64)
        if stage == "train":
            mask = batch.train_mask

            if self.warm_start:
                logits = self.forward(batch)
                logits = logits.reshape(logits.shape[0], -1, 2)
                loss_ls = []
                for i in range(logits.shape[1]):
                    loss = F.cross_entropy(logits[:,i,:][mask], y[mask])
                    loss_ls.append(loss.item())
                loss_ls = torch.tensor(loss_ls, requires_grad=True)
                primal_optimizer = self.optimizers()
                primal_optimizer.zero_grad()
                self.manual_backward(loss_ls.sum())
                primal_optimizer.step()
                return loss_ls,\
                       torch.tensor(0.),\
                       None,\
                       None,\
                       torch.tensor(0.),\
                       torch.tensor([0.] * len(self.target_recalls)),\
                       torch.tensor([0.] * len(self.target_recalls)),\
                       torch.tensor([0.] * len(self.target_recalls)),\
                       torch.tensor([0.] * len(self.target_recalls))
            else:
                logits = self.forward(batch)
                lagrangian = self.formulation.composite_objective(self.cmp.closure, self.model, logits[mask], y[mask])
                primal_optimizer = self.optimizers()
                primal_optimizer.zero_grad(); primal_optimizer.step() # dummy call to make lightning module do checkpointing, won't update the weights
                self.formulation.custom_backward(lagrangian)
                self.coop.step()
                self.coop.zero_grad()
                return self.cmp.state.loss,\
                       self.cmp.get_penalty(self.model),\
                       self.cmp.state.ineq_defect,\
                       self.cmp.state.proxy_ineq_defect,\
                       lagrangian,\
                       self.cmp.state.misc["fpr_proxy"],\
                       self.cmp.state.misc["fpr"],\
                       self.cmp.state.misc["recall_proxy"],\
                       self.cmp.state.misc["recall"]

        elif stage == "val" or stage == "test":
            if stage == "val":
                mask = batch.val_mask
            elif stage == "test":
                mask = batch.test_mask
            logits = self.forward(batch)
            probs = F.softmax(logits, dim=-1)
            probs = probs.reshape(probs.shape[0], -1, 2)

            lagrangian = self.formulation.composite_objective(self.cmp.closure, self.model, logits[mask], y[mask])

            return self.cmp.state.loss, \
                   self.cmp.get_penalty(self.model), \
                   self.cmp.state.ineq_defect, \
                   self.cmp.state.proxy_ineq_defect,\
                   lagrangian, \
                   self.cmp.state.misc["fpr_proxy"], \
                   self.cmp.state.misc["fpr"], \
                   self.cmp.state.misc["recall_proxy"], \
                   self.cmp.state.misc["recall"], \
                   probs

    def training_step(self, batch, batch_idx):
        objective, penalty, ineq_defect, proxy_ineq_defect, lagrangian_value, fpr_proxy, fpr, recall_proxy, recall = self.process_batch(batch, "train")
        batch_size = batch.train_mask.sum().item()

        self.log("train/constraint_penalty", penalty, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("train/loss", lagrangian_value, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        for i in range(len(self.target_recalls)):
            self.log("train/fpr_proxy_" + str(self.target_recalls[i]), fpr_proxy[i], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log("train/performance.fpr_" + str(self.target_recalls[i]), fpr[i], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log("train/recall_proxy_" + str(self.target_recalls[i]), recall_proxy[i], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log("train/performance.recall_" + str(self.target_recalls[i]), recall[i], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)

        if not self.warm_start and self.mode == "constrained_opt":
            for i in range(len(self.target_recalls)):
                self.log("train/constraints.inequality_violation_" + str(self.target_recalls[i]), ineq_defect[i], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
                self.log("train/constraints.proxy_inequality_violation_" + str(self.target_recalls[i]), proxy_ineq_defect[i], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
                self.log("train/constraints.multiplier_value_" + str(self.target_recalls[i]), self.formulation.ineq_multipliers.weight.detach().cpu()[i], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)

        return {"loss": lagrangian_value.detach()}

    def on_training_epoch_end(self):
        if self.current_epoch >= self.warmup_epochs:
            self.warm_start = False


    def validation_step(self, batch, batch_idx):
        objective, penalty, ineq_defect, proxy_ineq_defect, lagrangian_value, fpr_proxy, fpr, recall_proxy, recall, probs = self.process_batch(batch, "val")
        batch_size = batch.val_mask.sum().item()
        self.log("val/constraint_penalty", penalty, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("val/loss", lagrangian_value, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        for i in range(len(self.target_recalls)):
            self.log("val/fpr_proxy_" + str(self.target_recalls[i]), fpr_proxy[i], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log("val/performance.fpr_" + str(self.target_recalls[i]), fpr[i], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log("val/recall_proxy_" + str(self.target_recalls[i]), recall_proxy[i], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log("val/performance.recall_" + str(self.target_recalls[i]), recall[i], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
            if not self.warm_start and self.mode == "constrained_opt":
                self.log("val/constraints.inequality_violation_" + str(self.target_recalls[i]), ineq_defect[i],
                         on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
                self.log("val/constraints.proxy_inequality_violation_" + str(self.target_recalls[i]),
                         proxy_ineq_defect[i], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
                self.log("val/constraints.multiplier_value_" + str(self.target_recalls[i]),
                         self.formulation.ineq_multipliers.weight.detach().cpu()[i], on_step=True, on_epoch=True,
                         prog_bar=False, batch_size=batch_size)
        y_oracle = torch.zeros_like(batch.y, dtype=torch.int64)
        y_oracle[batch.y == self.novel_cls] = 1
        outputs = {"loss": lagrangian_value.detach(),
                   "probs": probs,
                   "y_oracle": y_oracle,
                   "tgt_mask": batch.tgt_mask,
                   "val_mask": batch.val_mask}
        self.validation_step_outputs.append(outputs)
        return outputs

    def test_step(self, batch, batch_idx):
        objective, penalty, ineq_defect, proxy_ineq_defect, lagrangian_value, fpr_proxy, fpr, recall_proxy, recall, probs = self.process_batch(batch, "test")
        batch_size = batch.test_mask.sum().item()
        self.log("test/constraint_penalty", penalty, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("test/loss", lagrangian_value, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        for i in range(len(self.target_recalls)):
            self.log("test/fpr_proxy_" + str(self.target_recalls[i]), fpr_proxy[i], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log("test/performance.fpr_" + str(self.target_recalls[i]), fpr[i], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log("test/recall_proxy_" + str(self.target_recalls[i]), recall_proxy[i], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log("test/performance.recall_" + str(self.target_recalls[i]), recall[i], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
            if not self.warm_start and self.mode == "constrained_opt":
                self.log("test/constraints.inequality_violation_" + str(self.target_recalls[i]), ineq_defect[i],
                         on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
                self.log("test/constraints.proxy_inequality_violation_" + str(self.target_recalls[i]),
                         proxy_ineq_defect[i], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
                self.log("test/constraints.multiplier_value_" + str(self.target_recalls[i]),
                         self.formulation.ineq_multipliers.weight.detach().cpu()[i], on_step=True, on_epoch=True,
                         prog_bar=False, batch_size=batch_size)
        y_oracle = torch.zeros_like(batch.y, dtype=torch.int64)
        y_oracle[batch.y == self.novel_cls] = 1
        outputs = {"loss": lagrangian_value.detach(),
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
        for i in range(len(self.target_recalls)):
            roc_auc = roc_auc_score(y_oracle[tgt_val_mask], probs[:, i, 1][tgt_val_mask])
            self.log("val/performance.AU-ROC_" + str(self.target_recalls[i]), roc_auc, on_step=False, on_epoch=True, batch_size=batch_size)

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        probs = torch.cat([o["probs"] for o in outputs], dim=0).detach().cpu().numpy()
        y_oracle = torch.cat([o["y_oracle"] for o in outputs], dim=0).detach().cpu().numpy()
        tgt_mask = torch.cat([o["tgt_mask"] for o in outputs], dim=0).detach().cpu().numpy()
        test_mask = torch.cat([o["test_mask"] for o in outputs], dim=0).detach().cpu().numpy()

        tgt_test_mask = np.logical_and(tgt_mask, test_mask)
        for i in range(len(self.target_recalls)):
            roc_auc = roc_auc_score(y_oracle[tgt_test_mask], probs[:, i, 1][tgt_test_mask])
            self.log("test/performance.AU-ROC_" + str(self.target_recalls[i]), roc_auc, on_step=False, on_epoch=True, batch_size=tgt_test_mask.sum())

        for i in range(len(self.target_recalls)):
            tgt_roc_auc = roc_auc_score(y_oracle[tgt_mask], probs[:, i, 1][tgt_mask])
            self.log("tgt/performance.AU-ROC_" + str(self.target_recalls[i]), tgt_roc_auc, on_step=False, on_epoch=True, batch_size=tgt_mask.sum())

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        return self.primal_optimizer