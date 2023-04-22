"""
@Project   : IMvGCN
@Time      : 2021/10/4
@Author    : Zhihao Wu
@File      : model.py
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def ortho_norm(weight):
    wtw = torch.mm(weight.t(), weight) + 1e-4 * torch.eye(weight.shape[1]).to(weight.device)
    L = torch.linalg.cholesky(wtw)
    weight_ortho = torch.mm(weight, L.inverse().t())
    return weight_ortho


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, num_views, activation=F.tanh):
        super(GraphConvSparse, self).__init__()
        self.weight = glorot_init(input_dim, output_dim)
        self.ortho_weight = torch.zeros_like(self.weight)
        self.activation = activation
        self.num_views = num_views

    def forward(self, inputs, flt, fea_sp=False):
        x = inputs
        self.ortho_weight = ortho_norm(self.weight)
        # self.ortho_weight = self.weight
        if fea_sp:
            x = torch.spmm(x, self.ortho_weight)
        else:
            x = torch.mm(x, self.ortho_weight)
        x = torch.spmm(flt, x)
        if self.activation is None:
            return x
        else:
            return self.activation(x)


class FGCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_views, activation=F.tanh):
        super(FGCN, self).__init__()
        self.weight = nn.ParameterList()
        for i in range(num_views):
            self.weight.append(glorot_init(input_dim[i], output_dim))
        self.activation = activation
        self.num_views = num_views

    def forward(self, hidden_list, flt_f):
        ortho_weight = []
        ortho_weight.append(ortho_norm(self.weight[0]))
        hidden_list[0] = hidden_list[0] - hidden_list[0].mean(dim=0)
        hidden = torch.mm(hidden_list[0], ortho_weight[0])
        for i in range(1, self.num_views):
            ortho_weight.append(ortho_norm(self.weight[i]))
            hidden_list[i] = hidden_list[i] - hidden_list[i].mean(dim=0)
            hidden += torch.mm(hidden_list[i], ortho_weight[i])
        output = torch.spmm(flt_f, hidden)
        return self.activation(output), ortho_weight


class MvGCN(nn.Module):
    def __init__(self, hidden_dims, num_views, dropout):
        super(MvGCN, self).__init__()
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.num_views = num_views
        self.gc1 = GraphConvSparse(self.hidden_dims[0], self.hidden_dims[1], self.num_views)
        self.gc2 = GraphConvSparse(self.hidden_dims[1], self.hidden_dims[2], self.num_views)

    def forward(self, input, flt):
        hidden = self.gc1(input, flt)
        output = self.gc2(hidden, flt)
        output = F.dropout(output, self.dropout, training=self.training)
        return output


class IMvGCN(nn.Module):
    def __init__(self, input_dims, num_classes, dropout, layers, device):
        super(IMvGCN, self).__init__()
        self.device = device
        self.input_dims = input_dims
        self.num_views = len(input_dims)
        self.mv_module = nn.ModuleList()
        hidden_dim = []
        for i in range(self.num_views):
            temp_dims = []
            temp_dims.append(input_dims[i])
            temp_dims.append(input_dims[i] // layers[0] if (input_dims[i] // layers[0]) >= num_classes else num_classes)
            temp_dims.append(input_dims[i] // layers[1] if (input_dims[i] // layers[1]) >= num_classes else num_classes)
            hidden_dim.append(temp_dims[-1])
            print(temp_dims)
            self.mv_module.append(MvGCN(hidden_dims=temp_dims, num_views=self.num_views, dropout=dropout))
        self.fusion_module = FGCN(hidden_dim, num_classes, self.num_views)

    def forward(self, feature_list, flt_list, flt_f):
        hidden_list = []
        w_list = []
        for i in range(self.num_views):
            hidden = self.mv_module[i](feature_list[i], flt_list[i])
            hidden_list.append(hidden)
            w_list.append(self.mv_module[i].gc1.ortho_weight)
            w_list.append(self.mv_module[i].gc2.ortho_weight)
        common_feature, ortho_weight = self.fusion_module(hidden_list, flt_f)
        w_list += ortho_weight

        return common_feature, hidden_list, w_list