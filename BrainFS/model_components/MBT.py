
from model_components.layers.encoders import DiffTransformerEncoder, FusionTransformerEncoder
import torch
import csv
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU, Sequential, Dropout, Sigmoid
from torch_geometric.nn import (GraphConv, TopKPooling, global_add_pool,
                                JumpingKnowledge)
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import dgl
from einops import rearrange, repeat
from .layers.layers import *


class DynamicCouplingLayer(nn.Module):
    def __init__(self, hidden_dim, dim_coupling, dropout_rate):
        super().__init__()
        print(
            f"DEBUG DynamicCouplingLayer init: hidden_dim={hidden_dim} (type: {type(hidden_dim)}), dim_coupling={dim_coupling} (type: {type(dim_coupling)})")
        self.hidden_dim = hidden_dim
        self.dim_coupling = dim_coupling  # Dimension of the coupling representation C
        self.dropout_rate = dropout_rate

        # MLP to generate coupling representation C_i from [H_sc_i || H_fc_i]
        print(
            f"DEBUG DynamicCouplingLayer: MLP couple input_dim={hidden_dim * 2 if hidden_dim is not None else 'None*2'}, output_dim={dim_coupling}")
        self.mlp_couple = Sequential(
            Linear(hidden_dim * 2, dim_coupling),
            ReLU(True),
            Dropout(dropout_rate)
        )

        # MLPs to generate gates from C_i
        self.gate_mlp_sc = Linear(dim_coupling, hidden_dim)
        self.gate_mlp_fc = Linear(dim_coupling, hidden_dim)

        # MLP for fusing gated representations
        self.fusion_mlp = Sequential(
            # Input is [gated_H_sc || gated_H_fc]
            Linear(hidden_dim * 2, hidden_dim),
            ReLU(True),
            Dropout(dropout_rate)
        )

        # MLPs to update modality representations using the fused info (can be simple Linear or Identity)
        self.sc_update_mlp = Linear(hidden_dim, hidden_dim)
        self.fc_update_mlp = Linear(hidden_dim, hidden_dim)

        self.norm_sc = LayerNorm(hidden_dim)
        self.norm_fc = LayerNorm(hidden_dim)
        self.act = ReLU(True)
        self.dropout = Dropout(dropout_rate)

    def forward(self, h_sc, h_fc):
        # h_sc, h_fc: (total_nodes_in_batch, hidden_dim)

        # 1. Generate Coupling Representations C
        combined_hidden = torch.cat((h_sc, h_fc), dim=-1)
        C = self.mlp_couple(combined_hidden)  # (total_nodes, dim_coupling)

        # 2. Generate Gates
        gate_sc_params = self.gate_mlp_sc(C)    # (total_nodes, hidden_dim)
        gate_fc_params = self.gate_mlp_fc(C)    # (total_nodes, hidden_dim)
        gate_sc = torch.sigmoid(gate_sc_params)
        gate_fc = torch.sigmoid(gate_fc_params)

        # 3. Apply Gates
        h_sc_gated = gate_sc * h_sc
        h_fc_gated = gate_fc * h_fc

        # 4. Fuse and Update
        h_fused_input = torch.cat((h_sc_gated, h_fc_gated), dim=-1)
        # (total_nodes, hidden_dim * 2)
        h_fused = self.fusion_mlp(h_fused_input)  # (total_nodes, hidden_dim)

        # Update modality representations with residual connection
        h_sc_updated_projection = self.dropout(
            self.act(self.sc_update_mlp(h_fused)))
        h_sc_new = self.norm_sc(h_sc + h_sc_updated_projection)

        h_fc_updated_projection = self.dropout(
            self.act(self.fc_update_mlp(h_fused)))
        h_fc_new = self.norm_fc(h_fc + h_fc_updated_projection)

        return h_sc_new, h_fc_new


class DAC_GNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Default to False if not present
        self.use_cuda = getattr(args, 'use_cuda', False)

        # Common args from other models
        self.sc_features = args.sc_features
        self.fc_features = args.fc_features
        self.hidden_dim = args.hidden_dim
        self.num_classes = args.num_classes
        self.dropout_rate = args.dropout  # Renamed from self.dropout for clarity

        # DAC_GNet specific args (ensure these are in your args Namespace)
        self.num_initial_gnn_layers = getattr(
            args, 'num_initial_gnn_layers', 1)
        self.num_coupling_layers = getattr(args, 'num_coupling_layers', 2)
        # Temporarily change to None to see effect
        self.dim_coupling = getattr(args, 'dim_coupling', None)
        if self.dim_coupling is None:
            if self.hidden_dim is not None:
                 self.dim_coupling = self.hidden_dim // 2
             else:
                 # This case should ideally not happen if hidden_dim is required or inferred
                 raise ValueError(
                     "hidden_dim is None, cannot determine default dim_coupling")
        print(
            f"DEBUG DAC_GNet init: self.hidden_dim = {self.hidden_dim}, type: {type(self.hidden_dim)}")
        print(
            f"DEBUG DAC_GNet init: self.dim_coupling = {self.dim_coupling}, type: {type(self.dim_coupling)}")

        # Initial node embeddings
        self.node_embedding_sc = Linear(self.sc_features, self.hidden_dim)
        self.node_embedding_fc = Linear(self.fc_features, self.hidden_dim)

        # --- Intra-Modal Graph Representation Learners (GINConv) ---
        # Helper MLP for GINConv (similar to other models)
        def create_gin_mlp(in_dim, hidden_dim, out_dim):
            return Sequential(
                Linear(in_dim, hidden_dim),
                ReLU(True),
                # LayerNorm(hidden_dim), # Optional: add layernorm
                Linear(hidden_dim, out_dim),
                ReLU(True),
                # LayerNorm(out_dim) # Optional: add layernorm
            )

        self.sc_initial_gnns = nn.ModuleList()
        self.fc_initial_gnns = nn.ModuleList()
        for _ in range(self.num_initial_gnn_layers):
            self.sc_initial_gnns.append(
                GINConv(create_gin_mlp(self.hidden_dim,
                        self.hidden_dim, self.hidden_dim), train_eps=True)
            )
            self.fc_initial_gnns.append(
                GINConv(create_gin_mlp(self.hidden_dim,
                        self.hidden_dim, self.hidden_dim), train_eps=True)
            )

        self.initial_gnn_norms_sc = nn.ModuleList(
            [LayerNorm(self.hidden_dim) for _ in range(self.num_initial_gnn_layers)])
        self.initial_gnn_norms_fc = nn.ModuleList(
            [LayerNorm(self.hidden_dim) for _ in range(self.num_initial_gnn_layers)])

        # --- Dynamically Coupled Interaction Layers ---
        self.coupling_layers = nn.ModuleList()
        for _ in range(self.num_coupling_layers):
            self.coupling_layers.append(
                DynamicCouplingLayer(
                    self.hidden_dim, self.dim_coupling, self.dropout_rate)
            )

        # --- Final Readout MLP ---
        # Input to classifier is concatenation of final H_sc and H_fc graph embeddings
        self.classifier_mlp = Sequential(
            Linear(self.hidden_dim * 2, self.hidden_dim),
            ReLU(True),
            Dropout(self.dropout_rate),
            Linear(self.hidden_dim, self.num_classes)
        )

        self.dropout_embed = Dropout(self.dropout_rate)

    def process(self, data, modality='sc'):
        # This process method is adapted from the provided GNN_Transformer example
        # It extracts original features and edge indices.
        # DAC_GNet assumes nodes are aligned between SC and FC for concatenation.
        # It does not use subgraph features directly in coupling layers but operates on full graph node embeddings.

        # Use original graph features and edges
        edge_index = data.__getattr__(f'{modality}_original_edge_index')
        x = data.__getattr__(f'{modality}_original_x')

        # Embed initial features
        if modality == 'sc':
            x = self.node_embedding_sc(x)
        else:  # modality == 'fc'
            x = self.node_embedding_fc(x)

        # Handle potential extra dimension if features are e.g. time series per node
        if len(x.shape) == 3:  # (batch_nodes, sequence_len, features)
            x = torch.sum(x, dim=-2)  # Sum over sequence_len, adapt if different aggregation needed

        x = self.dropout_embed(x)

        return x, edge_index

    def forward(self, data, return_fused_nodes=False):
        # 1. Initial Processing & Embedding
        # The 'batch' attribute from PyG Data object is used for pooling later
        # It maps each node to its respective graph in the batch
        sc_x_initial, sc_edge_index = self.process(data, modality='sc')
        fc_x_initial, fc_edge_index = self.process(data, modality='fc')

        h_sc = sc_x_initial
        h_fc = fc_x_initial

        # 2. Intra-Modal GNN Layers
        for i in range(self.num_initial_gnn_layers):
            h_sc_res = h_sc
            h_sc = self.sc_initial_gnns[i](h_sc, sc_edge_index)
            h_sc = self.initial_gnn_norms_sc[i](h_sc + h_sc_res)  # Residual connection + Norm
            h_sc = F.relu(h_sc)
            h_sc = F.dropout(h_sc, p=self.dropout_rate, training=self.training)

            h_fc_res = h_fc
            h_fc = self.fc_initial_gnns[i](h_fc, fc_edge_index)
            h_fc = self.initial_gnn_norms_fc[i](h_fc + h_fc_res)  # Residual connection + Norm
            h_fc = F.relu(h_fc)
            h_fc = F.dropout(h_fc, p=self.dropout_rate, training=self.training)

        # 3. Dynamically Coupled Interaction Layers
        for coupling_layer in self.coupling_layers:
            h_sc, h_fc = coupling_layer(h_sc, h_fc)

        # 4. Final Readout
        # Concatenate node embeddings from both modalities
        # h_sc and h_fc are (total_nodes_in_batch, hidden_dim)
        fused_node_embeddings_per_node = torch.cat((h_sc, h_fc), dim=-1)
        # Shape: (total_nodes_in_batch, hidden_dim * 2)

        if return_fused_nodes:
            # The generator script expects a list of tensors, one per graph in the batch.
            # Or, if it can handle the batched tensor and unbatch itself using data.ptr,
            # then just returning fused_node_embeddings_per_node is fine.
            # Let's assume the generator script will handle unbatching using data.ptr.
            # The existing generator script does:
            # batch_fused_node_features[i] which suggests it expects a list or a (batch_size, N, D) tensor.
            #
            # To return a list of tensors, one for each graph:
            output_fused_nodes_list = []
            for i in range(data.num_graphs):
                start_node_idx = data.ptr[i]
                end_node_idx = data.ptr[i+1]
                graph_fused_nodes = fused_node_embeddings_per_node[start_node_idx:end_node_idx]
                output_fused_nodes_list.append(graph_fused_nodes)
            return output_fused_nodes_list  # List of [N_nodes_graph_i, Hidden_Dim * 2]

        # --- Standard Classification Path ---
        # Graph pooling (e.g., global mean pooling) using the fused node embeddings
        graph_embedding = global_mean_pool(
            fused_node_embeddings_per_node, data.batch)
        # (batch_size, hidden_dim * 2)

        # 5. Classification
        logit = self.classifier_mlp(graph_embedding)
        # (batch_size, num_classes)

        # Return logits and optionally final node embeddings (h_sc, h_fc before concatenation) if needed for other tasks
        return logit, h_sc, h_fc
