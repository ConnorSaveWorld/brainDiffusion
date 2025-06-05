#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File Name:     diffusion.py
# Author:        wangchenze
# Created Time:  202409  17:09
# Last Modified: <none>-<none>

import sys
from typing import Optional
import networkx as nx
import community as community_louvain  # 确保导入 community_louvain 库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import math
import dgl
import dgl.function as fn
# from utils.utils import make_edge_weights
from .mlp_gat import Denoising_Unet
import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class SDiff(nn.Module):
    def __init__(
            self,
            in_dim_fmri:int , 
            in_dim_dti:int , 
            # in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            norm: Optional[str],
            alpha_l: float = 2,
            beta_schedule: str = 'linear',
            beta_1: float = 0.0001,
            beta_T: float = 0.02,
            T: int = 1000,
            **kwargs

         ):
        super(SDiff, self).__init__()
        self.T = T
        beta = get_beta_schedule(beta_schedule, beta_1, beta_T, T)
        self.register_buffer(
                'betas', beta
                )
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
                'sqrt_alphas_bar', torch.sqrt(alphas_bar)
                )
        self.register_buffer(
                'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar)
                )

        self.alpha_l = alpha_l
        assert num_hidden % nhead == 0
        # 初始化Denoising_Unet时，提供两个输入维度：in_dim_fmri和in_dim_dti
        self.net = Denoising_Unet(
            in_dim_fmri=in_dim_fmri,
            in_dim_dti=in_dim_dti,
            num_hidden=num_hidden,
            out_dim=in_dim_fmri,  # 输出维度可以保持不变
            num_layers=num_layers,
            nhead=nhead,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=0.2,
            norm=norm
        )

        self.time_embedding = nn.Embedding(T, num_hidden)

    def forward(self, g_fmri, g_dti, x_fmri, x_dti, labels,community_fmri,community_dti):
        with torch.no_grad():
            # 对 fMRI 和 DTI 进行归一化
            x_fmri = F.layer_norm(x_fmri, (x_fmri.shape[-1],))
            x_dti = F.layer_norm(x_dti, (x_dti.shape[-1],))
        num_steps = 2
        loss_total = 0

        # for step  in range(num_steps):
        #     print(f"Step {step}/{num_steps}")
        t = torch.randint(self.T, size=(x_fmri.shape[0],), device=x_fmri.device)

         # 对 fMRI 和 DTI 进行加噪
        x_t_fmri, x_t_dti, time_embed, g_fmri, g_dti = self.sample_q(t, x_fmri, x_dti, g_fmri, g_dti)
        t_community = torch.randint(self.T, size=(1,), device=x_fmri.device).item()

        print(f"t_community1097742134: {t_community}")
            # 计算去噪损失，结合 fMRI 和 DTI 的耦合关系
        loss= self.node_denoising(x_fmri, x_t_fmri, x_dti, x_t_dti, time_embed, g_fmri,
                                  g_dti,community_fmri,community_dti)
        loss_total+=loss
        # 平均损失
        # loss_total /= num_steps

        # 返回损失值
        loss_item = {"total_loss": loss_total.item()}
        return loss_total, loss_item
    def sample_q_alone(self, t, x):
        miu, std = x.mean(dim=0), x.std(dim=0)
        noise = torch.randn_like(x, device=x.device)
        with torch.no_grad():
            noise = F.layer_norm(noise, (noise.shape[-1], ))
        noise = noise * std + miu
        noise = torch.sign(x) * torch.abs(noise)
        x_t = (
                extract(self.sqrt_alphas_bar, t, x.shape) * x +
                extract(self.sqrt_one_minus_alphas_bar, t, x.shape) * noise
                )
        time_embed = self.time_embedding(t)
        return x_t, time_embed
    def sample_q(self, t, x_fmri, x_dti, g_fmri, g_dti):
        # 对 fMRI 数据进行加噪
        miu_fmri, std_fmri = x_fmri.mean(dim=0), x_fmri.std(dim=0)
        noise_fmri = torch.randn_like(x_fmri, device=x_fmri.device)
        with torch.no_grad():
            noise_fmri = F.layer_norm(noise_fmri, (noise_fmri.shape[-1],))
        noise_fmri = noise_fmri * std_fmri + miu_fmri
        noise_fmri = torch.sign(x_fmri) * torch.abs(noise_fmri)
        x_t_fmri = (
                extract(self.sqrt_alphas_bar, t, x_fmri.shape) * x_fmri +
                extract(self.sqrt_one_minus_alphas_bar, t, x_fmri.shape) * noise_fmri
        )

        # 对 DTI 数据进行加噪
        miu_dti, std_dti = x_dti.mean(dim=0), x_dti.std(dim=0)
        noise_dti = torch.randn_like(x_dti, device=x_dti.device)
        with torch.no_grad():
            noise_dti = F.layer_norm(noise_dti, (noise_dti.shape[-1],))
        noise_dti = noise_dti * std_dti + miu_dti
        noise_dti = torch.sign(x_dti) * torch.abs(noise_dti)
        x_t_dti = (
                extract(self.sqrt_alphas_bar, t, x_dti.shape) * x_dti +
                extract(self.sqrt_one_minus_alphas_bar, t, x_dti.shape) * noise_dti
        )

        # 对应时间步的嵌入
        time_embed = self.time_embedding(t)

        return x_t_fmri, x_t_dti, time_embed, g_fmri, g_dti

    def detect_communities(self,node_features, T):
        G = nx.from_numpy_array(node_features)
        print(f"TTTTTTTTTTTTTTTTTTT: {T}")

        # # 根据 T 的值确定社区划分的标准
        # if 0 <= T < 50:
        #     resolution = 1
        # elif 50 <= T < 800:
        #     resolution = 0.001
        # else:
        #     raise ValueError("T 值超出预定义范围")
        #
        # for u, v, data in G.edges(data=True):
        #     if data.get('weight', 1) < 0:
        #         print(f"Edge ({u}, {v}) has negative weight: {data['weight']}")

        # 使用 Louvain 算法进行社区检测
        partition = community_louvain.best_partition(G)
        # print(f"resolution: {resolution}")
        # 输出社区数目，调试时可以使用
        community_count = len(set(partition.values()))
        print(f"Number of detected communities: {community_count}")

        return partition, G
    def node_denoising(self, x_fmri, x_t_fmri, x_dti, x_t_dti, time_embed, g_fmri, g_dti,community_fmri,community_dti):
        print(f"x_fmri: {x_fmri.shape}")
        print(f"x_dti: {x_dti.shape}")

        print(f"community_fmri: {community_fmri.shape}")
        print(f"community_dti: {community_dti.shape}")
        num_sub_matrices = 16  # 假设一共有16个子矩阵
        sub_matrix_size = 90
        total_loss = 0

        for i in range(num_sub_matrices):
            print(f"i12356: {i}")
            start_idx = i * sub_matrix_size
            end_idx = (i + 1) * sub_matrix_size
            x_fmri_sub = x_fmri[start_idx:end_idx]
            x_t_fmri_sub = x_t_fmri[start_idx:end_idx]
            x_dti_sub = x_dti[start_idx:end_idx]
            x_t_dti_sub = x_t_dti[start_idx:end_idx]
            g_fmri_sub = g_fmri.subgraph(range(start_idx, end_idx))
            g_dti_sub = g_dti.subgraph(range(start_idx, end_idx))
            # 社区
            community_fmri_sub = community_fmri[start_idx:end_idx]
            community_dti_sub = community_dti[start_idx:end_idx]
            if torch.isnan(community_dti_sub).any():
                print("noisy_x_t_dti_sub contains NaN values")
            # 对 fMRI 进行社区检测和处理
            noisy_x_t_fmri_sub = self.process_by_community(x_t_fmri_sub,community_fmri_sub)
            # 对 DTI 进行社区检测和处理
            noisy_x_t_dti_sub = self.process_by_community(x_t_dti_sub,community_dti_sub)
            if torch.isnan(noisy_x_t_dti_sub).any():
                print("noisy_x_t_dti_sub contains NaN values")

            # 传入去噪网络
            out_fmri, out_dti, _, _ = self.net(g_fmri_sub, g_dti_sub, noisy_x_t_fmri_sub, noisy_x_t_dti_sub, time_embed)
            # 计算损失
            loss_fmri = loss_fn(out_fmri, x_fmri_sub, self.alpha_l)
            loss_dti = loss_fn(out_dti, x_dti_sub, self.alpha_l)
            total_loss += loss_fmri + loss_dti
        return total_loss

    def process_by_community(self, x_t_fmri_sub, community_data):
        unique_communities = torch.unique(community_data)  # 获取所有唯一的社区编号
        final_noisy_features = torch.zeros_like(x_t_fmri_sub)  # 用于存储最终的加噪矩阵

        for community in unique_communities:
            # 获取该社区对应的节点索引
            community_indices = torch.where(community_data == community)[0]

            # 创建一个大小为 x_t_fmri_sub 的掩码矩阵
            mask = torch.zeros_like(x_t_fmri_sub)

            # 仅保留该社区内节点的非自连接
            for i in community_indices:
                for j in community_indices:
                    if i != j:  # 跳过自连接
                        mask[i, j] = 1  # 设置社区内非自连接节点对应位置为1

            # 将原图与掩码相乘，保留该社区的子图
            subgraph_features = x_t_fmri_sub * mask

            # 随机生成 t 值，注意 t 的大小应该与社区内节点数量匹配
            t = torch.randint(self.T, size=(subgraph_features.shape[0],), device=subgraph_features.device)

            # 对该社区的子图加噪
            noisy_subgraph_features, _ = self.sample_q_alone(t, subgraph_features)

            # 再次应用掩码，确保只有社区内的节点保留，其他部分置为0
            noisy_subgraph_features = noisy_subgraph_features * mask

            # 将加噪后的社区子图累加到最终矩阵
            final_noisy_features += noisy_subgraph_features

        # 获取原始矩阵的自连接部分
        diag_mask = torch.eye(x_t_fmri_sub.size(0), device=x_t_fmri_sub.device)
        diag_part = x_t_fmri_sub * diag_mask

        # 将自连接部分加回到最终的加噪结果中
        final_noisy_features += diag_part

        return final_noisy_features

    # 获取 fMRI 或 DTI 图的边和权重
    def get_adjacency_matrix(self, graph):
        # 获取边的起点和终点，并转到 CPU
        src, dst = graph.edges()
        src = src.cpu()
        dst = dst.cpu()

        # 获取边的权重，先转到 CPU 再转为 numpy 数组
        weights = graph.edata['weight'].cpu().numpy()

        # 移除自环边
        mask = src != dst  # 只保留起点和终点不同的边
        src = src[mask]
        dst = dst[mask]
        weights = weights[mask]

        # 获取图的节点数
        num_nodes = graph.number_of_nodes()

        # 初始化一个全零的邻接矩阵
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        # 将权重填入邻接矩阵
        adj_matrix[src.numpy(), dst.numpy()] = weights

        return adj_matrix

    def embed_node_denoising(self, x_fmri, x_t_fmri, x_dti, x_t_dti, time_embed, g_fmri, g_dti,community_fmri,community_dti):
        print(f"x_fmri: {x_fmri.shape}")
        print(f"x_dti: {x_dti.shape}")
        num_sub_matrices = 88
        sub_matrix_size = 90
        all_hidden = []
        all_batch_g_eval = []
        for i in range(num_sub_matrices):
            start_idx = i * sub_matrix_size
            end_idx = (i + 1) * sub_matrix_size

            x_fmri_sub = x_fmri[start_idx:end_idx]
            x_t_fmri_sub = x_t_fmri[start_idx:end_idx]
            x_dti_sub = x_dti[start_idx:end_idx]
            x_t_dti_sub = x_t_dti[start_idx:end_idx]

            g_fmri_sub = g_fmri.subgraph(range(start_idx, end_idx))
            g_dti_sub = g_dti.subgraph(range(start_idx, end_idx))

            # 社区
            community_fmri_sub = community_fmri[start_idx:end_idx]
            community_dti_sub = community_dti[start_idx:end_idx]
            # 对 fMRI 进行社区检测和处理
            noisy_x_t_fmri_sub = self.process_by_community(x_t_fmri_sub,community_fmri_sub)


            # 对 DTI 进行社区检测和处理
            noisy_x_t_dti_sub = self.process_by_community(x_t_dti_sub,community_dti_sub)

            _, _, hidden,batch_g_eval = self.net(g_fmri_sub, g_dti_sub, noisy_x_t_fmri_sub, noisy_x_t_dti_sub, time_embed)
            all_hidden.append(hidden)
            all_batch_g_eval.append(batch_g_eval)
        # 将所有子矩阵的隐藏特征拼接在一起
        hidden = torch.cat(all_hidden, dim=0)  # 根据需要调整 dim 参数
        finally_batch_g_eval = dgl.batch(all_batch_g_eval)  # 根据需要调整 dim 参数
        # 查看 finally_batch_g_eval 的维度大小
        print(f"Number of nodes in finally_batch_g_eval: {finally_batch_g_eval.number_of_nodes()}")
        print(f"Number of edges in finally_batch_g_eval: {finally_batch_g_eval.number_of_edges()}")
        return hidden,finally_batch_g_eval


    def embed(self, g_fmri, g_dti, x_fmri, x_dti, T,community_fmri,community_dti):
        # 生成时间步，批量大小为 x_fmri 的第一个维度大小
        t = torch.full((x_fmri.shape[0],), T, device=x_fmri.device)

        # 对 fMRI 和 DTI 数据进行归一化
        with torch.no_grad():
            x_fmri = F.layer_norm(x_fmri, (x_fmri.shape[-1],))
            x_dti = F.layer_norm(x_dti, (x_dti.shape[-1],))

        # 生成加噪后的数据和时间嵌入
        x_t_fmri, x_t_dti, time_embed, g_fmri, g_dti = self.sample_q(t, x_fmri, x_dti, g_fmri, g_dti)
        # t_community = torch.randint(self.T, size=(1,), device=x_fmri.device).item()

        # print(f"t_communityTest: {t_community}")
        # 调用网络进行前向传播，获得隐藏特征
        hidden, finally_batch_g_eval= self.embed_node_denoising(x_fmri, x_t_fmri, x_dti, x_t_dti, time_embed, g_fmri, g_dti,community_fmri,community_dti)

        return hidden ,finally_batch_g_eval

def loss_fn(x, y, alpha=2):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas)
