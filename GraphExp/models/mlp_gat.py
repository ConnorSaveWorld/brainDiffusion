#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File Name:     mlp_gat.py
# Author:        wangchenze
# Created Time:  2024-09-06  13:48
# Last Modified: <none>-<none>
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
# from dgl.nn import GINConv
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GraphConv
# from dgl.nn import GCNConv
# from dgl.nn import EGATConv
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from .utils import create_activation, create_norm


def exists(x):
    return x is not None


class Denoising_Unet(nn.Module):
    def __init__(self,
                 in_dim_fmri,
                 in_dim_dti,
                 num_hidden,
                 out_dim,
                 num_layers,
                 nhead,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 norm,
                 ):
        super(Denoising_Unet, self).__init__()
        self.num_layers = num_layers

        # 输入的 MLP 层，用于混淆矩阵的初始处理
        self.mlp_in_t = MlpBlock(in_dim=in_dim_fmri, hidden_dim=num_hidden*2, out_dim=num_hidden,
                                 norm=norm, activation=activation)
        # 修改 GCN，使输入和输出的维度相同
        self.gcn_fmri = GraphConv(in_dim_fmri, in_dim_fmri)  # 输入和输出维度相同
        self.gcn_dti = GraphConv(in_dim_dti, in_dim_dti)  # 输入和输出维度相同
        # 混淆矩阵处理的U-Net结构
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.mlp_middle = MlpBlock(num_hidden, num_hidden, num_hidden, norm=norm, activation=activation)
        self.mlp_out_fmri = MlpBlock(num_hidden, out_dim, out_dim, norm=norm, activation=activation)
        self.mlp_out_dti = MlpBlock(num_hidden, out_dim, out_dim, norm=norm, activation=activation)
        self.mlp_in_S = MlpBlock(in_dim=90, hidden_dim=num_hidden * 2, out_dim=num_hidden,
                                 norm=norm, activation=activation)
        for _ in range(num_layers):
            self.down_layers.append(GATConv(num_hidden, num_hidden // nhead, nhead, feat_drop,
                                            attn_drop, negative_slope))
            self.up_layers.append(GATConv(num_hidden, num_hidden // nhead, nhead, feat_drop,
                                          attn_drop, negative_slope))
        self.up_layers = self.up_layers[::-1]

    def create_graph_from_adjacency(self, adj_matrix):
        """
        使用邻接矩阵创建图对象，并将邻接矩阵作为节点特征矩阵，同时将边权重添加到图中。
        :param adj_matrix: 形状为 (num_nodes, num_nodes) 的张量，其中非零元素为边的权重
        :return: DGL 图对象
        """
        # 确保邻接矩阵是稀疏矩阵
        adj_matrix_sparse = adj_matrix.to_sparse()

        # 从稀疏邻接矩阵中提取边和边权重
        edges = adj_matrix_sparse.indices()
        edge_weights = adj_matrix_sparse.values()

        # 创建图对象
        num_nodes = adj_matrix.size(0)
        g = dgl.graph((edges[0], edges[1]), num_nodes=num_nodes)

        # 将边权重添加到图中
        g.edata['weight'] = edge_weights

        # 将邻接矩阵作为节点特征添加到图中
        g.ndata['feat'] = adj_matrix

        return g

    def forward(self, g_fmri, g_dti, fmri_data, dti_data, time_embed):
        # # 打印输入特征的形状
        # print(f"fmri_data shape before MLP: {fmri_data.shape}")
        # print(f"dti_data shape before MLP: {dti_data.shape}")
        # 先通过GCN层处理
        fmri_data = self.gcn_fmri(g_fmri, fmri_data)
        dti_data = self.gcn_dti(g_dti, dti_data)
        if torch.isnan(fmri_data).any():
            print("fmri_data contains NaN values")
        if torch.isnan(dti_data).any():
            print("dti_data contains NaN values")

        # 使用 fMRI 和 DTI 特征的点积生成混淆矩阵
        S_hat1 = fmri_data @ dti_data.transpose(-1, -2)
        S_hat2 = dti_data @ fmri_data.transpose(-1, -2)
        S_hat0 = (S_hat1 + S_hat2) / 2  # 取两个点积矩阵的平均值
        # 检查 fmri_data 和 dti_data 是否包含 NaN
        if torch.isnan(fmri_data).any():
            print("fmri_data contains NaN values")
        if torch.isnan(dti_data).any():
            print("dti_data contains NaN values")

        # 检查 S_hat1, S_hat2 是否包含 NaN
        if torch.isnan(S_hat1).any():
            print("S_hat1 contains NaN values")
        if torch.isnan(S_hat2).any():
            print("S_hat2 contains NaN values")

        # 检查 S_hat0 是否包含 NaN
        if torch.isnan(S_hat0).any():
            print("S_hat0 contains NaN values after averaging")
        # 计算 S_hat_G 并创建图对象
        S_hat_G = (fmri_data + dti_data) / 2
        g_S_hat_G = self.create_graph_from_adjacency(S_hat_G)
        batch_g_eval = g_S_hat_G
        # 对 fMRI 和 DTI 数据分别进行 MLP 处理
        fmri_data = self.mlp_in_t(fmri_data)
        dti_data = self.mlp_in_t(dti_data)
        # 对初始混淆矩阵进行 MLP 处理
        # 检查 S_hat0 是否有 NaN
        if torch.isnan(S_hat0).any():
            print("S_hat0 contains NaN values before MLP")

        # 归一化/标准化操作，防止出现过大或过小的值
        S_hat0 = torch.clamp(S_hat0, min=-1e10, max=1e10)

        S_hat0 = self.mlp_in_t(S_hat0)

        # fMRI 和 DTI 通过第一层下采样并保留特征
        h_fmri = self.down_layers[0](g_fmri, fmri_data)
        h_dti = self.down_layers[0](g_dti, dti_data)
        expert_fmri = h_fmri.flatten(1)
        expert_dti = h_dti.flatten(1)

        # 混淆矩阵进入后续的下采样层
        h_t = S_hat0
        down_hidden = []

        for l in range(self.num_layers):
            if h_t.ndim > 2:
                h_t = h_t + time_embed.unsqueeze(1).repeat(1, h_t.shape[1], 1)
            h_t = self.down_layers[l](g_S_hat_G, h_t)  # 使用 S_hat0 作为输入
            h_t = h_t.flatten(1)
            down_hidden.append(h_t)

        h_middle = self.mlp_middle(h_t)
        h_t = h_middle
        out_hidden = []
        for l in range(self.num_layers):
            h_t = h_t + down_hidden[self.num_layers - l - 1]
            if h_t.ndim > 2:
                h_t = h_t + time_embed.unsqueeze(1).repeat(1, h_t.shape[1], 1)
            if l == self.num_layers - 1:
                # 在最后一层上采样时处理 fMRI 和 DTI 特征
                # h_t_fmri = h_t + expert_fmri
                # h_t_dti = h_t + expert_dti
                h_t_fmri = h_t
                h_t_dti = h_t
                h_t_fmri = self.up_layers[l](g_fmri, h_t_fmri)
                h_t_dti = self.up_layers[l](g_dti, h_t_dti)
                final_fmri_output = self.mlp_out_fmri(h_t_fmri.flatten(1))
                final_dti_output = self.mlp_out_dti(h_t_dti.flatten(1))
                # 合并 fMRI 和 DTI 上采样后的特征到 out_hidden
                out_hidden_fmri = final_fmri_output.flatten(1)
                out_hidden_dti = final_dti_output.flatten(1)
                out_hidden.append(h_t_fmri)
                out_hidden.append(h_t_dti)
                # out_hidden_fmri.append(out_hidden_fmri)
                # out_hidden_dti.append(out_hidden_dti)
            else:
                h_t = self.up_layers[l](g_S_hat_G, h_t)  # 使用 S_hat0 作为输入
                h_t = h_t.flatten(1)
        out_hidden = torch.cat(out_hidden, dim=-1)

        return final_fmri_output, final_dti_output, out_hidden,batch_g_eval


class Residual(nn.Module):
    def __init__(self, fnc):
        super().__init__()
        self.fnc = fnc

    def forward(self, x, *args, **kwargs):
        return self.fnc(x, *args, **kwargs) + x


class MlpBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 norm: str = 'layernorm', activation: str = 'prelu'):
        super(MlpBlock, self).__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.res_mlp = Residual(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                              create_norm(norm)(hidden_dim),
                                              create_activation(activation),
                                              nn.Linear(hidden_dim, hidden_dim)))
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.act = create_activation(activation)
    def forward(self, x):
        x = self.in_proj(x)
        x = self.res_mlp(x)
        x = self.out_proj(x)
        x = self.act(x)
        return x
