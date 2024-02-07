from copy import deepcopy
import numpy as np
from sklearn.metrics import roc_auc_score
import lightning as L
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from src.utils.model_utils import get_model_optimizer

class SAREM(L.LightningModule):
    def __init__(
            self,
            model_type: str,
            arch_param,
            dataset_name,
            novel_cls,
            learning_rate,
            weight_decay,
            max_epochs,
            inner_epochs,
            refit,
            seed):
        super().__init__()
        self.model_type = model_type
        self.arch_param = arch_param
        self.novel_cls = novel_cls
        self.seed = seed

        self.dataset_name = dataset_name
        self.max_epochs = max_epochs
        self.inner_epochs = inner_epochs

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.refit = refit

        # self.novelty_detector = None # either initialized during on_train_start or load from checkpoint during testing
        # self.propensity_estimator = None # either initialized during on_train_start or load from checkpoint during testing
        self.novelty_detector, self.detector_optimizer = get_model_optimizer(model_type,
                                                                             arch_param,
                                                                             learning_rate,
                                                                             weight_decay)
        self.novelty_detector.to(self.device)
        self.propensity_estimator, self.propensity_optimizer = get_model_optimizer(model_type,
                                                                                   arch_param,
                                                                                   learning_rate,
                                                                                   weight_decay)
        self.propensity_estimator.to(self.device)
        _, self.dummy_optimizer = get_model_optimizer(model_type, # dummy optimizer only for lightning module checkpointing
                                                      arch_param,
                                                      learning_rate,
                                                      weight_decay)

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, model, data):
        if self.model_type == "mlp":
            return model(data.x)
        else:
            return model(data.x, data.edge_index)

    def expectation_nonnovel(self, expectation_nonnovel, expectation_propensity, s):
        # probability of data points being in non-novel class
        # if s = 1 (src data), must be in non-novel class
        result = s + (1 - s) * (expectation_nonnovel * (1 - expectation_propensity)) / (1 - expectation_nonnovel * expectation_propensity)
        return result

    def loglikelihood_probs(self, nonnovel_probs, propensity_scores, s):
        prob_src = nonnovel_probs * propensity_scores
        prob_tgt_nonnovel = nonnovel_probs * (1 - propensity_scores)
        prob_tgt_novel = 1 - nonnovel_probs
        prob_nonnovel_given_tgt = prob_tgt_nonnovel / (prob_tgt_nonnovel + prob_tgt_novel)
        prob_novel_given_tgt = 1 - prob_nonnovel_given_tgt
        return (s * torch.log(prob_src) + (1 - s) * (prob_nonnovel_given_tgt * torch.log(prob_tgt_nonnovel) + prob_novel_given_tgt * torch.log(prob_tgt_novel))).mean()

    def on_train_start(self):
        # initialize with unlabeled=negative, but reweighting the examples so that the expected class prior is 0.5
        datamodule = self.trainer.datamodule
        loader = datamodule.train_dataloader()
        data = next(iter(loader)).to(self.device)
        s = data.src_mask.type(torch.int64)
        proportion_src = s.sum() / s.size(0)
        detector_class_weights = torch.tensor([1 - proportion_src, proportion_src]).to(self.device)
        # novelty_detector, detector_optimizer = get_model_optimizer(self.model_type,
        #                                                            self.arch_param,
        #                                                            self.learning_rate,
        #                                                            self.weight_decay)
        # novelty_detector.to(self.device)
        # for novelty_detector/propensity_estimator, output = 0 is non-novel/propensity=1, output = 1 is novel/propensity=0
        self.novelty_detector = self._inner_fit(self.novelty_detector, data, 1 - s, self.detector_optimizer, self.inner_epochs, class_weight=detector_class_weights)
        detector_expectation = F.softmax(self.forward(self.novelty_detector, data), dim=1)[:,0].detach() # prob of being non-novel (positive in PU terms)

        # propensity_estimator, propensity_optimizer = get_model_optimizer(self.model_type,
        #                                                                  self.arch_param,
        #                                                                  self.learning_rate,
        #                                                                  self.weight_decay)
        # propensity_estimator.to(self.device)
        propensity_sample_weights = s + (1 - s) * detector_expectation
        self.propensity_estimator = self._inner_fit(self.propensity_estimator, data, 1 - s, self.propensity_optimizer, self.inner_epochs, sample_weight=propensity_sample_weights)

        self.expected_prior_nonnovel = detector_expectation
        self.expected_propensity = F.softmax(self.forward(self.propensity_estimator, data), dim=1)[:,0].detach()
        self.expected_posterior_nonnovel = self.expectation_nonnovel(self.expected_prior_nonnovel, self.expected_propensity, s)

    def on_train_end(self):
        if self.refit:
            datamodule = self.trainer.datamodule
            loader = datamodule.train_dataloader()
            data = next(iter(loader)).to(self.device)
            print("Average propensity in original class 0:", self.expected_propensity[data.y == 0].mean())
            print("Average propensity in original class 1:", self.expected_propensity[data.y == 1].mean())
            print("Average propensity in original class 2:", self.expected_propensity[data.y == 2].mean())
            print("Average propensity in original class 3:", self.expected_propensity[data.y == 3].mean())
            print("Average propensity in original class 4:", self.expected_propensity[data.y == 4].mean())
            print("Average propensity in original class 5:", self.expected_propensity[data.y == 5].mean())
            print("Average propensity in novel class:", self.expected_propensity[data.y == self.novel_cls].mean())
            s = data.src_mask.type(torch.int64)
            novelty_detector, detector_optimizer = get_model_optimizer(self.model_type,
                                                                       self.arch_param,
                                                                       self.learning_rate,
                                                                       self.weight_decay)
            novelty_detector.to(self.device)
            weights_nonnovel = s / (self.expected_propensity + 1e-12)
            weights_novel = (1 - s) + s * (1 - 1 / (self.expected_propensity + 1e-12))

            detector_data = Data().to(self.device)
            detector_data.x = torch.cat([data.x, data.x], dim=0)
            detector_data.edge_index = torch.cat([data.edge_index, data.edge_index + data.x.size(0)], dim=1)
            detector_data.train_mask = torch.cat([data.train_mask, data.train_mask], dim=0)
            detector_data.val_mask = torch.cat([data.val_mask, data.val_mask], dim=0)
            y = torch.cat([torch.zeros_like(s), torch.ones_like(s)], dim=0)
            sample_weights = torch.cat([weights_nonnovel, weights_novel], dim=0)

            self.novelty_detector = self._inner_fit(novelty_detector, detector_data, y, detector_optimizer, self.inner_epochs, sample_weight=sample_weights)



    def on_save_checkpoint(self, checkpoint):
        checkpoint["expected_prior_nonnovel"] = self.expected_prior_nonnovel
        checkpoint["expected_propensity"] = self.expected_propensity
        checkpoint["expected_posterior_nonnovel"] = self.expected_posterior_nonnovel

    def on_load_checkpoint(self, checkpoint):
        self.expected_prior_nonnovel = checkpoint["expected_prior_nonnovel"]
        self.expected_propensity = checkpoint["expected_propensity"]
        self.expected_posterior_nonnovel = checkpoint["expected_posterior_nonnovel"]

    def process_batch(self, batch, stage):
        s = batch.src_mask.type(torch.int64)
        y = 1 - s
        y_oracle = torch.zeros_like(batch.tgt_mask, dtype=torch.int64)
        y_oracle[batch.y == self.novel_cls] = 1
        if stage == "train":
            mask = batch.train_mask
        elif stage == "val":
            mask = batch.val_mask
        elif stage == "test":
            mask = batch.test_mask

        if stage == "train":
            # maximization
            propensity_estimator, propensity_optimizer = get_model_optimizer(self.model_type,
                                                                             self.arch_param,
                                                                             self.learning_rate,
                                                                             self.weight_decay)
            propensity_estimator.to(self.device)
            self.propensity_estimator = self._inner_fit(propensity_estimator, batch, 1 - s, propensity_optimizer, self.inner_epochs, sample_weight=self.expected_posterior_nonnovel)
            detector_s = torch.cat([torch.ones_like(self.expected_posterior_nonnovel, dtype=torch.int64), torch.zeros_like(self.expected_posterior_nonnovel, dtype=torch.int64)], dim=0)
            detector_weights = torch.cat([self.expected_posterior_nonnovel, 1 - self.expected_posterior_nonnovel], dim=0)
            novelty_detector, detector_optimizer = get_model_optimizer(self.model_type,
                                                                       self.arch_param,
                                                                       self.learning_rate,
                                                                       self.weight_decay)
            novelty_detector.to(self.device)
            # duplicate graph data (two disconnected, identical graphs)
            detector_data = Data().to(self.device)
            detector_data.x = torch.cat([batch.x, batch.x], dim=0)
            detector_data.edge_index = torch.cat([batch.edge_index, batch.edge_index + batch.x.size(0)], dim=1)
            detector_data.train_mask = torch.cat([batch.train_mask, batch.train_mask], dim=0)
            detector_data.val_mask = torch.cat([batch.val_mask, batch.val_mask], dim=0)
            # target of 1st half of the data is 0 (non-novel)
            self.novelty_detector = self._inner_fit(novelty_detector, detector_data, 1 - detector_s, detector_optimizer, self.inner_epochs, sample_weight=detector_weights)

            # expectation
            self.expected_prior_nonnovel = F.softmax(self.forward(self.novelty_detector, batch), dim=1)[:,0].detach()
            self.expected_propensity = F.softmax(self.forward(self.propensity_estimator, batch), dim=1)[:,0].detach()
            self.expected_posterior_nonnovel = self.expectation_nonnovel(self.expected_prior_nonnovel, self.expected_propensity, s)

            ll = self.loglikelihood_probs(self.expected_prior_nonnovel[mask], self.expected_propensity[mask], s[mask])
            dummy_optimizer = self.optimizers()
            dummy_optimizer.zero_grad(); dummy_optimizer.step() # a dummy call for checkpointing
            return -ll

        elif stage == "val" or stage == "test":
            ll = self.loglikelihood_probs(self.expected_prior_nonnovel[mask], self.expected_propensity[mask], s[mask])
            novelty_logits = self.forward(self.novelty_detector, batch)
            novelty_probs = F.softmax(novelty_logits, dim=1)
            return -ll, novelty_probs, y, y_oracle, batch.tgt_mask, mask

        else:
            raise ValueError(f"Invalid stage: {stage}")


    def training_step(self, batch, batch_idx):
        # Does 1 pass of expectation/maximization
        nll_loss = self.process_batch(batch, "train")
        batch_size = batch.train_mask.sum().item()
        self.log("train/loss", nll_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        return {"loss": nll_loss.detach()}


    def validation_step(self, batch, batch_idx):
        nll_loss, probs, y, y_oracle, tgt_mask, val_mask = self.process_batch(batch, "val")
        batch_size = val_mask.sum().item()
        self.log("val/loss", nll_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        outputs = {"loss": nll_loss.detach(),
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
        tgt_val_mask = np.logical_and(tgt_mask, val_mask)
        roc_auc = roc_auc_score(y_oracle[tgt_val_mask], probs[:, 1][tgt_val_mask])
        self.log("val/performance.AU-ROC", roc_auc, on_step=False, on_epoch=True)
        self.validation_step_outputs = []

    def test_step(self, batch, batch_idx):
        nll_loss, probs, y, y_oracle, tgt_mask, test_mask = self.process_batch(batch, "test")
        batch_size = test_mask.sum().item()
        self.log("test/loss", nll_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        outputs = {"loss": nll_loss.detach(),
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
        return self.dummy_optimizer

    def _inner_fit(self, model, data, y, optimizer, inner_epochs, patience=20, class_weight=None, sample_weight=None):
        best_val_loss = np.inf
        staleness = 0
        for e in range(1, inner_epochs + 1):
            train_out = self.forward(model, data)
            if sample_weight is None:
                train_loss = F.nll_loss(F.log_softmax(train_out[data.train_mask], dim=1), y[data.train_mask], weight=class_weight)
            else:
                train_loss = F.nll_loss(F.log_softmax(train_out[data.train_mask], dim=1), y[data.train_mask], weight=class_weight, reduction="none")
                train_loss = torch.mean(train_loss * sample_weight[data.train_mask])
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            val_out = self.forward(model, data)
            if sample_weight is None:
                val_loss = F.nll_loss(F.log_softmax(val_out[data.val_mask], dim=1), y[data.val_mask], weight=class_weight)
            else:
                val_loss = F.nll_loss(F.log_softmax(val_out[data.val_mask], dim=1), y[data.val_mask], weight=class_weight, reduction="none")
                val_loss = torch.mean(val_loss * sample_weight[data.val_mask])
            if val_loss < best_val_loss:
                best_model = deepcopy(model)
                best_val_loss = val_loss
                staleness = 0
            else:
                staleness += 1

            if staleness > patience:
                break
        return best_model

