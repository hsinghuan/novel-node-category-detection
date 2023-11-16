import torch
from src.model import Model, MLP, LinearGCN, GCN, GAT



def get_model_optimizer(model_name, arch_param, learning_rate, weight_decay):
    if model_name == "mlp":
        model = MLP(arch_param["mlp_dim_list"], dropout_list=arch_param["mlp_dr_list"])
    elif model_name == "lingcn":
        model = LinearGCN(arch_param["gnn_dim_list"])
    elif model_name == "gcn":
        encoder = GCN(arch_param["gnn_dim_list"], dropout_list=arch_param["gnn_dr_list"])
        mlp = MLP(arch_param["mlp_dim_list"], dropout_list=arch_param["mlp_dr_list"])
        model = Model(encoder, mlp)
    elif model_name == "gat":
        encoder = GAT(arch_param["gnn_dim_list"], dropout_list=arch_param["gnn_dr_list"])
        mlp = MLP(arch_param["mlp_dim_list"], dropout_list=arch_param["mlp_dr_list"])
        model = Model(encoder, mlp)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    return model, optimizer