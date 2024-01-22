import numpy as np
import torch
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

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    return model, optimizer

def model_selection_from_val_outputs(val_outputs_dict: dict, model_selection_args: dict):
    criteria = model_selection_args["criteria"]
    if criteria == "val_loss":
        pass
    elif criteria == "conoc":
        beta = model_selection_args["beta"]
        highest_recall = -np.inf
        best_ckpt_dirpath = None
        for ckpt_dirpath, val_outputs in val_outputs_dict.items():
            val_outputs = val_outputs[0]
            for k in val_outputs.keys():
                if k.startswith("val/performance.fpr"):
                    fpr = val_outputs[k]
                if k.startswith("val/performance.recall"):
                    recall = val_outputs[k]
            print(f"checkpoint dirpath: {ckpt_dirpath} fpr: {fpr} recall: {recall}")
            if fpr > beta:
                continue
            elif recall > highest_recall:
                highest_recall = recall
                best_ckpt_dirpath = ckpt_dirpath

        if best_ckpt_dirpath:
            return best_ckpt_dirpath
        else:
            raise TypeError("Best checkpoint directory path is None. The false positive rates may not be low enough for the specified threshold.")

