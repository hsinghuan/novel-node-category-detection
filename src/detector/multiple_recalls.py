import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import lightning as L
import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph, negative_sampling, structured_negative_sampling
from torch_geometric.utils.map import map_index
import cooper

from src.utils.core_utils import recall_from_logits, fpr_from_logits
from src.utils.model_utils import get_model_optimizer
from src.utils.graph_utils import subgraph_negative_sampling

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

    def closure(self, model, pred_logits, y, aux_loss=0.):
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
            loss += fpr_proxy

            cross_ent_ls = torch.cat((cross_ent_ls, torch.unsqueeze(cross_ent, 0)))
            cross_ent_target_ls = torch.cat((cross_ent_target_ls, torch.unsqueeze(cross_ent_target,0)))
            fpr_ls = torch.cat((fpr_ls, torch.unsqueeze(fpr,0)))
            fpr_proxy_ls = torch.cat((fpr_proxy_ls, torch.unsqueeze(fpr_proxy,0)))
            recall_ls = torch.cat((recall_ls, torch.unsqueeze(recall,0)))
            recall_proxy_ls = torch.cat((recall_proxy_ls, torch.unsqueeze(recall_proxy,0)))
            recall_loss_ls = torch.cat((recall_loss_ls, torch.unsqueeze(recall_loss,0)))


        ineq_defect = torch.tensor(self.target_recall).to(pred_logits) - recall_ls
        proxy_ineq_defect = torch.tensor(self.target_recall).to(pred_logits) - recall_proxy_ls

        return cooper.CMPState(loss=loss + penalty + aux_loss, # .sum(),
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
                 lagrange_multiplier_init=0.1,
                 link_predict=None,
                 aux_loss_weight=1e-2
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
        self.link_predict = link_predict
        self.aux_loss_weight = aux_loss_weight

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

    def aux_loss(self, data):
        if self.link_predict == "gae":
            assert "gae" in self.model_type
            z = self.model.encoder.encode(data.x, data.edge_index)
            aux_loss = self.model.encoder.recon_loss(z, data.edge_index)
        elif self.link_predict == "gae_tgt" or self.link_predict == "gae_tgt_struct":
            assert "gae" in self.model_type
            z = self.model.encoder.encode(data.x, data.edge_index)
            tgt_edge_index, _ = subgraph(data.tgt_mask, data.edge_index)
            # relabel nodes
            num_nodes = data.tgt_mask.size(0)
            node_mask = data.tgt_mask
            subset = node_mask.nonzero().view(-1)
            relabeled_tgt_edge_index, _ = map_index(tgt_edge_index, subset, max_index=num_nodes, inclusive=True)
            relabeled_tgt_edge_index = relabeled_tgt_edge_index.view(2, -1)
            if self.link_predict == "gae_tgt_struct":
                relabeled_tgt_neg_edge_index_0, _, relabeled_tgt_neg_edge_index_1 = structured_negative_sampling(relabeled_tgt_edge_index)
                relabeled_tgt_neg_edge_index = torch.stack([relabeled_tgt_neg_edge_index_0, relabeled_tgt_neg_edge_index_1])
            else:
                relabeled_tgt_neg_edge_index = negative_sampling(relabeled_tgt_edge_index)
            aux_loss = self.model.encoder.recon_loss(z[data.tgt_mask], relabeled_tgt_edge_index, relabeled_tgt_neg_edge_index)
        elif self.link_predict == "gae_hard_tgt":
            assert "gae" in self.model_type
            z = self.model.encoder.encode(data.x, data.edge_index)
            # get the mask indicating the "hard" target nodes
            novel_scores = self.model.classifier(z)[:, 1].detach()
            num_nodes = data.tgt_mask.size(0)
            num_tgt_nodes = data.tgt_mask.sum().item()
            tgt_set = data.tgt_mask.nonzero().view(-1)
            tgt_novel_scores = novel_scores[data.tgt_mask]
            sorted_score_idx = torch.argsort(tgt_novel_scores)
            hard_tgt_set = tgt_set[sorted_score_idx[:int(num_tgt_nodes * (1 - self.target_recalls[0]))]]
            hard_tgt_set, _ = torch.sort(hard_tgt_set)
            hard_tgt_mask = torch.zeros_like(data.tgt_mask, dtype=torch.bool)
            hard_tgt_mask[hard_tgt_set] = 1
            hard_tgt_edge_index, _ = subgraph(hard_tgt_mask, data.edge_index)
            # print(novel_scores[hard_tgt_mask].mean() < novel_scores[data.tgt_mask].mean())
            relabeled_hard_tgt_edge_index, _ = map_index(hard_tgt_edge_index, hard_tgt_set, max_index=num_nodes, inclusive=True)
            relabeled_hard_tgt_edge_index = relabeled_hard_tgt_edge_index.view(2, -1)
            relabeled_hard_tgt_neg_edge_index = negative_sampling(relabeled_hard_tgt_edge_index)
            aux_loss = self.model.encoder.recon_loss(z[hard_tgt_mask], relabeled_hard_tgt_edge_index, relabeled_hard_tgt_neg_edge_index)
        elif self.link_predict == "gae_hard_tgt_2": # negative samples contain (confident tgt, non-confident tgt) pair
            assert "gae" in self.model_type
            z = self.model.encoder.encode(data.x, data.edge_index) # calculate node representations
            novel_scores = self.model.classifier(z)[:, 1].detach() # obtain novel scores for each node
            num_nodes = data.tgt_mask.size(0)
            num_tgt_nodes = data.tgt_mask.sum().item()
            tgt_set = data.tgt_mask.nonzero().view(-1)
            tgt_novel_scores = novel_scores[data.tgt_mask] # the novel scores of target nodes
            sorted_score_idx = torch.argsort(tgt_novel_scores) # the ranked index of target novel scores
            hard_tgt_set = tgt_set[sorted_score_idx[:int(num_tgt_nodes * (1 - self.target_recalls[0]))]] # keep the least novel target nodes
            hard_tgt_set, _ = torch.sort(hard_tgt_set)
            hard_tgt_mask = torch.zeros_like(data.tgt_mask, dtype=torch.bool)
            hard_tgt_mask[hard_tgt_set] = 1 # mask for harder target nodes
            tgt_edge_index, _ = subgraph(data.tgt_mask, data.edge_index) # subgraph edge index only including target nodes
            relabeled_tgt_edge_index, _ = map_index(tgt_edge_index, tgt_set, max_index=num_nodes, inclusive=True) # relabel node index
            relabeled_tgt_edge_index = relabeled_tgt_edge_index.view(2, -1)
            relabeled_tgt_neg_edge_index = negative_sampling(relabeled_tgt_edge_index)
            # get a bool tensor with the same shape as the neg edge index, each element specifiy if a node is hard or not
            relabeled_tgt_neg_edge_index_mask = torch.isin(relabeled_tgt_neg_edge_index, hard_tgt_set)
            # make a mask to filter out easy-easy pair
            has_hard_edge_index_mask = torch.any(relabeled_tgt_neg_edge_index_mask, dim=0)
            neg_edge_index = relabeled_tgt_neg_edge_index[:,has_hard_edge_index_mask]
            aux_loss = self.model.encoder.recon_loss(z[data.tgt_mask], relabeled_tgt_edge_index, neg_edge_index)
        else:
            aux_loss = 0.

        return aux_loss * self.aux_loss_weight

    def process_batch(self, batch, stage): # , batch_linkpred=None):
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
                aux_loss = self.aux_loss(batch) # _linkpred)
                lagrangian = self.formulation.composite_objective(self.cmp.closure, self.model, logits[mask], y[mask], aux_loss=aux_loss)
                primal_optimizer = self.optimizers()
                primal_optimizer.zero_grad(); primal_optimizer.step() # dummy call to make lightning module do checkpointing, won't update the weights
                self.formulation.custom_backward(lagrangian)
                self.coop.step()
                self.coop.zero_grad()
                return self.cmp.state.loss,\
                       self.cmp.get_penalty(self.model),\
                       aux_loss,\
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
            aux_loss = self.aux_loss(batch) # _linkpred)
            probs = F.softmax(logits, dim=-1)
            probs = probs.reshape(probs.shape[0], -1, 2)

            lagrangian = self.formulation.composite_objective(self.cmp.closure, self.model, logits[mask], y[mask], aux_loss=aux_loss)

            return self.cmp.state.loss, \
                   self.cmp.get_penalty(self.model),\
                   aux_loss,\
                   self.cmp.state.ineq_defect, \
                   self.cmp.state.proxy_ineq_defect,\
                   lagrangian, \
                   self.cmp.state.misc["fpr_proxy"], \
                   self.cmp.state.misc["fpr"], \
                   self.cmp.state.misc["recall_proxy"], \
                   self.cmp.state.misc["recall"], \
                   probs

    def training_step(self, batch, batch_idx):
        # if self.link_predict:
        #     batch_linkpred = batch[1]
        #     batch = batch[0]
        # else:
        #     batch_linkpred = None
        objective, penalty, aux_loss, ineq_defect, proxy_ineq_defect, lagrangian_value, fpr_proxy, fpr, recall_proxy, recall = self.process_batch(batch, "train") # , batch_linkpred)
        batch_size = batch.train_mask.sum().item()

        self.log("train/constraint_penalty", penalty, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("train/aux_loss", aux_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
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
        # if self.link_predict:
        #     batch_linkpred = batch[1]
        #     batch = batch[0]
        # else:
        #     batch_linkpred = None
        objective, penalty, aux_loss, ineq_defect, proxy_ineq_defect, lagrangian_value, fpr_proxy, fpr, recall_proxy, recall, probs = self.process_batch(batch, "val") # , batch_linkpred)
        batch_size = batch.val_mask.sum().item()
        self.log("val/constraint_penalty", penalty, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("val/aux_loss", aux_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
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
        # if self.link_predict:
        #     batch_linkpred = batch[1]
        #     batch = batch[0]
        # else:
        #     batch_linkpred = None
        objective, penalty, aux_loss, ineq_defect, proxy_ineq_defect, lagrangian_value, fpr_proxy, fpr, recall_proxy, recall, probs = self.process_batch(batch, "test") # , batch_linkpred)
        batch_size = batch.test_mask.sum().item()
        self.log("test/constraint_penalty", penalty, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("test/aux_loss", aux_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
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
            ap = average_precision_score(y_oracle[tgt_val_mask], probs[:, i, 1][tgt_val_mask])
            cm = confusion_matrix(y_oracle[tgt_val_mask], np.argmax(probs[:, i][tgt_val_mask], axis=1))
            self.log("val/performance.AU-ROC_" + str(self.target_recalls[i]), roc_auc, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("val/performance.AP_" + str(self.target_recalls[i]), ap, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("val/performance.TP_" + str(self.target_recalls[i]), cm[1,1], on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("val/performance.FP_" + str(self.target_recalls[i]), cm[0,1], on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("val/performance.TN_" + str(self.target_recalls[i]), cm[0,0], on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("val/performance.FN_" + str(self.target_recalls[i]), cm[1,0], on_step=False, on_epoch=True, batch_size=batch_size)

        self.validation_step_outputs = []

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        probs = torch.cat([o["probs"] for o in outputs], dim=0).detach().cpu().numpy()
        y_oracle = torch.cat([o["y_oracle"] for o in outputs], dim=0).detach().cpu().numpy()
        tgt_mask = torch.cat([o["tgt_mask"] for o in outputs], dim=0).detach().cpu().numpy()
        test_mask = torch.cat([o["test_mask"] for o in outputs], dim=0).detach().cpu().numpy()

        tgt_test_mask = np.logical_and(tgt_mask, test_mask)
        for i in range(len(self.target_recalls)):
            roc_auc = roc_auc_score(y_oracle[tgt_test_mask], probs[:, i, 1][tgt_test_mask])
            ap = average_precision_score(y_oracle[tgt_test_mask], probs[:, i, 1][tgt_test_mask])
            self.log("test/performance.AU-ROC_" + str(self.target_recalls[i]), roc_auc, on_step=False, on_epoch=True, batch_size=tgt_test_mask.sum())
            self.log("test/performance.AP_" + str(self.target_recalls[i]), ap, on_step=False, on_epoch=True, batch_size=tgt_test_mask.sum())

        for i in range(len(self.target_recalls)):
            tgt_roc_auc = roc_auc_score(y_oracle[tgt_mask], probs[:, i, 1][tgt_mask])
            tgt_ap = average_precision_score(y_oracle[tgt_mask], probs[:, i, 1][tgt_mask])
            self.log("tgt/performance.AU-ROC_" + str(self.target_recalls[i]), tgt_roc_auc, on_step=False, on_epoch=True, batch_size=tgt_mask.sum())
            self.log("tgt/performance.AP_" + str(self.target_recalls[i]), tgt_ap, on_step=False, on_epoch=True, batch_size=tgt_mask.sum())

        self.test_step_outputs = []

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        return self.primal_optimizer