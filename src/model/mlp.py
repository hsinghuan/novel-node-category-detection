import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_list, dropout_list=None):
        super().__init__()
        if dropout_list:
            assert len(dropout_list) == len(dim_list) - 2 # # layer = len(dim_list) - 1 and last layer doesn't need dropout
            self.dropout_list = dropout_list
        else:
            self.dropout_list = [0. for _ in range(len(dim_list) - 2)]
        self.linears = nn.ModuleList([nn.Linear(dim_list[i], dim_list[i+1]) for i in range(len(dim_list) - 1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(dim) for dim in dim_list[1:-1]])

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = l(x)
            if i != len(self.linears) - 1:
                x = self.bns[i](x)
                x = x.relu()
                x = F.dropout(x, p=self.dropout_list[i], training=self.training)

        return x

    def reset_parameters(self):
        for linear in self.linears:
            linear.reset_parameters()
        for bn in self.bns:
            bn.reset_running_stats()