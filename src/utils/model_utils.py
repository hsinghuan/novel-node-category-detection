import numpy as np
import warnings
from pathlib import Path
from typing import Optional, Union
from datetime import timedelta
import torch
from torch_geometric.nn import GAE
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from src.model import Model, MLP, LinearGCN, GCN, GAT



def get_model_optimizer(model_type, arch_param, learning_rate, weight_decay):
    if model_type == "mlp":
        model = MLP(arch_param["mlp_dim_list"], dropout_list=arch_param["mlp_dr_list"])
    elif model_type == "lingcn":
        model = LinearGCN(arch_param["gnn_dim_list"])
    elif model_type == "gcn":
        encoder = GCN(arch_param["gnn_dim_list"], dropout_list=arch_param["gnn_dr_list"])
        mlp = MLP(arch_param["mlp_dim_list"], dropout_list=arch_param["mlp_dr_list"])
        model = Model(encoder, mlp)
    elif model_type == "gat":
        encoder = GAT(arch_param["gnn_dim_list"], dropout_list=arch_param["gnn_dr_list"])
        mlp = MLP(arch_param["mlp_dim_list"], dropout_list=arch_param["mlp_dr_list"])
        model = Model(encoder, mlp)
    elif model_type == "gcn_gae":
        encoder = GCN(arch_param["gnn_dim_list"], dropout_list=arch_param["gnn_dr_list"])
        mlp = MLP(arch_param["mlp_dim_list"], dropout_list=arch_param["mlp_dr_list"])
        model = Model(GAE(encoder), mlp)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    return model, optimizer

def model_selection_from_val_outputs(val_outputs_dict: dict, model_selection_args: dict):
    criteria = model_selection_args["criteria"]
    if criteria == "val_loss":
        pass
    elif criteria == "reco_slip":
        beta = model_selection_args["beta"]
        highest_recall = -np.inf
        best_ckpt_dirpath = None
        lowest_fpr = np.inf
        for ckpt_dirpath, val_outputs in val_outputs_dict.items():
            val_outputs = val_outputs[0]
            for k in val_outputs.keys():
                if k.startswith("val/performance.fpr"):
                    fpr = val_outputs[k]
                if k.startswith("val/performance.recall"):
                    recall = val_outputs[k]
            print(f"checkpoint dirpath: {ckpt_dirpath} fpr: {fpr} recall: {recall}")
            if fpr < lowest_fpr:
                lowest_fpr = fpr
                lowest_fpr_ckpt_dir_path = ckpt_dirpath

            if fpr > beta:
                continue
            elif recall > highest_recall:
                highest_recall = recall
                best_ckpt_dirpath = ckpt_dirpath

        if best_ckpt_dirpath:
            return best_ckpt_dirpath
        else:
            warnings.warn(f"No models fulfill the beta = {beta} requirement, choose the model with the lowest fpr: {lowest_fpr}")
            return lowest_fpr_ckpt_dir_path
            # raise TypeError("Best checkpoint directory path is None. The false positive rates may not be low enough for the specified threshold.")


class MonitorAfterModelCheckpoint(ModelCheckpoint):
    def __init__(self,
                 monitor_after: int,
                 dirpath: Optional[Union[str, Path]] = None,
                 filename: Optional[str] = None,
                 monitor: Optional[str] = None,
                 verbose: bool = False,
                 save_last: Optional[bool] = None,
                 save_top_k: int = 1,
                 save_weights_only: bool = False,
                 mode: str = "min",
                 auto_insert_metric_name: bool = True,
                 every_n_train_steps: Optional[int] = None,
                 train_time_interval: Optional[timedelta] = None,
                 every_n_epochs: Optional[int] = None,
                 save_on_train_epoch_end: Optional[bool] = None,
                 enable_version_counter: bool = True):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter)
        self.is_monitoring_on = False
        self.monitor_after = monitor_after

    def on_train_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule", unused: Optional = None) -> None:
        """Save a checkpoint at the end of the training epoch."""
        # as we advance one step at end of training, we use `global_step - 1` to avoid saving duplicates
        # trainer.fit_loop.global_step -= 1
        # if (
        #     not self._should_skip_saving_checkpoint(trainer)
        #     and self._save_on_train_epoch_end
        #     and self._every_n_epochs > 0
        #     and (trainer.current_epoch + 1) % self._every_n_epochs == 0
        #     and (self.is_monitoring_on or self.monitor_can_start(trainer, pl_module))
        # ):
        #     self.save_checkpoint(trainer)
        # trainer.fit_loop.global_step += 1
        if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer) and (self.is_monitoring_on or self.monitor_can_start(trainer)):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)

    def monitor_can_start(self, trainer: L.Trainer) -> bool: #, pl_module: L.LightningModule) -> bool:
        """Let start monitoring only after the loss curve start increasing"""
        # monitor_candidates = self._monitor_candidates(trainer, trainer.current_epoch, trainer.global_step - 1)
        # current = monitor_candidates.get(self.monitor)

        # Check if the critic loss is increasing (the network is starting to
        # train)
        # if trainer.current_epoch > 0 and pl_module.previous_metric < current:
        if trainer.current_epoch > self.monitor_after:
            self.is_monitoring_on = True

        # pl_module.previous_metric = current.detach().clone()

        return self.is_monitoring_on