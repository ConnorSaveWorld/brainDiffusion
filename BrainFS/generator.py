import argparse
import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader # Use PyG DataLoader
from model_components.MBT import GNN_Transformer, DAC_GNet, Alternately_Attention_Bottlenecks # Import your modified model
from NewdataLoader import MyOwnDataset # Import your dataset class
# Assuming these transforms are used during dataset creation/processing
# If they are only needed for the model forward pass, they might not be needed here
# If they are part of the dataset's pre_transform, they ARE needed here.
from transform import HHopSubgraphs, LapEncoding
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm # For progress bar
import time # For timing

def generate_fused_graphs(args):
    start_time = time.time()
    # Determine CUDA usage and device
    args.use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.use_cuda else "cpu")
    print(f"Using device: {device}")

    # --- Load checkpoint and infer embedding dimensions ---
    print(f"Loading checkpoint from: {args.model_ckpt_path}")
    raw_ckpt    = torch.load(args.model_ckpt_path, map_location=device)
    # Handle potential nested state_dict (e.g., from Lightning)
    state_dict = raw_ckpt.get('state_dict', raw_ckpt)
    # Remove potential prefix like 'model.' if it exists (common in Lightning)
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

    # Infer dimensions ONLY IF they are not provided or you want to double-check
    # It's generally safer to use the args provided if they match the saved model's training args
    if 'node_embedding_sc.weight' in state_dict and 'node_embedding_fc.weight' in state_dict:
        emb_sc_w    = state_dict['node_embedding_sc.weight']
        emb_fc_w    = state_dict['node_embedding_fc.weight']
        inferred_hidden_dim  = emb_sc_w.size(0)
        inferred_sc_features = emb_sc_w.size(1)
        inferred_fc_features = emb_fc_w.size(1)
        print(f"Inferred from checkpoint: hidden_dim={inferred_hidden_dim}, sc_features={inferred_sc_features}, fc_features={inferred_fc_features}")
        # Optionally override args if needed, but be cautious
        args.hidden_dim = inferred_hidden_dim
        args.sc_features = inferred_sc_features
        args.fc_features = inferred_fc_features
    else:
        print("Could not infer dimensions from checkpoint keys. Using provided args.")
        # Ensure required args are provided if inference fails
        assert args.hidden_dim is not None, "hidden_dim must be provided if not inferrable"
        assert args.sc_features is not None, "sc_features must be provided if not inferrable"
        assert args.fc_features is not None, "fc_features must be provided if not inferrable"
        print(f"Using provided args: hidden_dim={args.hidden_dim}, sc_features={args.sc_features}, fc_features={args.fc_features}")


    # --- Load Full Dataset with Transforms ---
    # These transforms should match those used when *training* the model
    # The pre_transform is applied once when the dataset is processed
    print("Initializing dataset transforms...")
    subgraph_pre = LapEncoding(dim=args.lap_dim)
    transform    = HHopSubgraphs(
        h=args.h,
        max_nodes_per_hop=args.max_nodes_per_hop,
        node_label='hop',
        use_rd=False,
        subgraph_pretransform=subgraph_pre
    )
    print(f"Loading dataset: {args.dataset}...")
    # Assuming MyOwnDataset uses args for configuration internally if needed
    full_dataset = MyOwnDataset(args.dataset, args=args, pre_transform=transform)
    print(f"Total graphs loaded: {len(full_dataset)}")
    if len(full_dataset) == 0:
        print("Error: Dataset is empty. Check dataset path and processing.")
        return

    # --- Define Splits ---
    print("Splitting data into train/eval...")
    labels = [int(data.y.item()) for data in full_dataset]
    indices = list(range(len(full_dataset)))

    if len(set(labels)) < 2:
        print("Warning: Only one class found in labels. Stratified split may behave like random split.")
        train_idx, eval_idx = train_test_split(
            indices,
            test_size=args.eval_ratio,
            random_state=42 # Use a fixed random state for reproducibility
        )
    elif len(full_dataset) * args.eval_ratio < 2 : # Ensure enough samples for stratification
         print(f"Warning: Very small dataset or eval_ratio. Splitting without stratification.")
         train_idx, eval_idx = train_test_split(
            indices,
            test_size=args.eval_ratio,
            random_state=42
        )
    else:
        try:
             train_idx, eval_idx = train_test_split(
                indices,
                test_size=args.eval_ratio,
                stratify=labels,
                random_state=42 # Use a fixed random state for reproducibility
            )
        except ValueError as e:
            print(f"Warning: Stratified split failed ({e}). Falling back to non-stratified split.")
            train_idx, eval_idx = train_test_split(
                indices,
                test_size=args.eval_ratio,
                random_state=42
            )


    # Use sets for efficient lookup later
    train_idx_set = set(train_idx)
    eval_idx_set = set(eval_idx)
    print(f"Defined splits: Train={len(train_idx)}, Eval={len(eval_idx)}")

    # --- Create DataLoader for the *entire* dataset ---
    # IMPORTANT: shuffle=False to maintain order for split assignment
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Initialize and load model ---
    print("Initializing model...")
    model = GNN_Transformer(args)
    print("Loading model state dict...")
    model.load_state_dict(state_dict)
    model.to(device).eval() # Set to evaluation mode

    # --- Generation Function ---
    @torch.no_grad() # Disable gradient calculations for inference
    def generate_all_fused(loader, train_indices, eval_indices):
        all_fused_graphs = []
        current_original_idx = 0 # Keep track of the original index
        for data in tqdm(loader, desc="Generating fused graphs"):
            data = data.to(device)

            # Step 1: Ensure model returns node-level embeddings.
            # Assuming GNN_Transformer's forward method supports a 'return_fused_nodes' flag.
            # If your model returns node embeddings by default from model(data),
            # you might not need the flag, but it's safer given the comment.
            # Check the actual signature of your GNN_Transformer.forward method.
            try:
                # Try calling with the flag mentioned in the original comments
                all_nodes_fused_features_in_batch = model(data, return_fused_nodes=True)
            except TypeError as e:
                # If the flag causes a TypeError, maybe it's not needed or named differently.
                # This is a fallback; ideally, you know the model's API.
                print(f"Warning: model call with 'return_fused_nodes=True' failed ('{e}'). "
                      "Falling back to model(data). Ensure model(data) returns node-level features.")
                all_nodes_fused_features_in_batch = model(data)

            # Ensure the output is a tensor (it should be for batched node features)
            if not isinstance(all_nodes_fused_features_in_batch, torch.Tensor):
                raise TypeError(f"Model output is not a tensor. Got {type(all_nodes_fused_features_in_batch)}. "
                                "Expected batched node features [total_nodes_in_batch, hidden_dim].")

            # Ensure output dimension makes sense for node features
            if all_nodes_fused_features_in_batch.dim() != 2:
                raise ValueError(f"Model output tensor has incorrect dimensions: {all_nodes_fused_features_in_batch.shape}. "
                                 "Expected 2D tensor [total_nodes_in_batch, hidden_dim].")


            # Process each graph in the batch
            for i in range(data.num_graphs):
                split_type = 'train' if current_original_idx in train_indices else 'eval'
                assert current_original_idx in train_indices or current_original_idx in eval_indices, \
                    f"Error: Original index {current_original_idx} not found in any split."

                # Get the range of nodes for the current graph within the batch
                node_start_in_batch = int(data.ptr[i])
                node_end_in_batch = int(data.ptr[i+1])

                # Step 2: Correctly slice features for the current graph from the batched tensor
                fused_node_feat = all_nodes_fused_features_in_batch[node_start_in_batch:node_end_in_batch].cpu()
                
                label = data.y[i].unsqueeze(0).cpu()
                num_nodes_this_graph = fused_node_feat.shape[0]

                # --- Reconstruct original combined edge_index ---
                # Filter sc edges belonging to this graph and shift indices relative to the graph's start
                # These indices (node_start_in_batch, node_end_in_batch) are batch-wise absolute indices.
                sc_mask = (data.sc_original_edge_index[0] >= node_start_in_batch) & \
                          (data.sc_original_edge_index[0] < node_end_in_batch) & \
                          (data.sc_original_edge_index[1] >= node_start_in_batch) & \
                          (data.sc_original_edge_index[1] < node_end_in_batch)
                sc_ei_graph = data.sc_original_edge_index[:, sc_mask] - node_start_in_batch

                # Filter fc edges belonging to this graph and shift indices relative to the graph's start
                fc_mask = (data.fc_original_edge_index[0] >= node_start_in_batch) & \
                          (data.fc_original_edge_index[0] < node_end_in_batch) & \
                          (data.fc_original_edge_index[1] >= node_start_in_batch) & \
                          (data.fc_original_edge_index[1] < node_end_in_batch)
                fc_ei_graph = data.fc_original_edge_index[:, fc_mask] - node_start_in_batch
                
                # Validate edge indices are within bounds for the current graph's nodes
                if sc_ei_graph.numel() > 0:
                    if not (sc_ei_graph.max() < num_nodes_this_graph):
                        # Add more debug info if assertion fails
                        print(f"DEBUG: Graph {current_original_idx}, Split {split_type}")
                        print(f"DEBUG: num_nodes_this_graph (from fused_node_feat.shape[0]): {num_nodes_this_graph}")
                        print(f"DEBUG: sc_ei_graph.shape: {sc_ei_graph.shape}, sc_ei_graph.max(): {sc_ei_graph.max().item()}")
                        print(f"DEBUG: node_start_in_batch: {node_start_in_batch}, node_end_in_batch: {node_end_in_batch}")
                        print(f"DEBUG: data.sc_original_edge_index shape: {data.sc_original_edge_index.shape}")
                        # You might want to print a snippet of sc_original_edge_index around the problematic area
                    assert sc_ei_graph.max() < num_nodes_this_graph, \
                        f"SC edge index {sc_ei_graph.max().item()} out of bounds for {num_nodes_this_graph} nodes (graph {current_original_idx})."
                
                if fc_ei_graph.numel() > 0:
                    if not (fc_ei_graph.max() < num_nodes_this_graph):
                         print(f"DEBUG: Graph {current_original_idx}, Split {split_type}")
                         print(f"DEBUG: num_nodes_this_graph (from fused_node_feat.shape[0]): {num_nodes_this_graph}")
                         print(f"DEBUG: fc_ei_graph.shape: {fc_ei_graph.shape}, fc_ei_graph.max(): {fc_ei_graph.max().item()}")
                    assert fc_ei_graph.max() < num_nodes_this_graph, \
                        f"FC edge index {fc_ei_graph.max().item()} out of bounds for {num_nodes_this_graph} nodes (graph {current_original_idx})."

                combined_edge_index = torch.cat([sc_ei_graph, fc_ei_graph], dim=1).cpu()
                # Optional: Remove duplicate edges if SC and FC can have overlapping edges
                # combined_edge_index = torch.unique(combined_edge_index, dim=1)

                fused_graph_data = {
                    'node_feat': fused_node_feat,
                    'edge_index': combined_edge_index,
                    'edge_attr': None,
                    'y': label,
                    'split': split_type
                }
                all_fused_graphs.append(fused_graph_data)
                current_original_idx += 1
        return all_fused_graphs

    # --- Generate and Save ---
    print("Starting fused graph generation...")
    all_fused_data = generate_all_fused(full_loader, train_idx_set, eval_idx_set)

    output_filename = os.path.join(args.output_dir, f"{args.dataset}_fused_full.pkl")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving {len(all_fused_data)} fused graphs to {output_filename}...")
    torch.save(all_fused_data, output_filename)

    end_time = time.time()
    print(f"Successfully saved dataset.")
    print(f"Total time: {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fused Graph Dataset Generation using GNN_Transformer')

    # --- Arguments ---
    # Essential args for loading model/data
    parser.add_argument('--dataset', type=str, required=True, help="Name of the original dataset (e.g., ZDXX, HCP_MBT)")
    parser.add_argument('--model_ckpt_path', type=str, required=True, help='Path to the BEST trained GNN_Transformer checkpoint (.pth file)')
    parser.add_argument('--output_dir', type=str, default='./fused_dataset_output', help='Directory to save the fused graph list')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--eval_ratio', type=float, default=0.2, help='Fraction of data to designate as eval split')

    # --- Model architecture parameters ---
    # These should match the parameters of the loaded checkpoint's model
    # Provide defaults, but ideally load from a config or ensure they match the training run
    parser.add_argument('--hidden_dim', type=int, default=None, help='Model hidden dimension (can often be inferred)')
    parser.add_argument('--sc_features', type=int, default=None, help='Number of input features for SC nodes (can often be inferred)')
    parser.add_argument('--fc_features', type=int, default=None, help='Number of input features for FC nodes (can often be inferred)')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GNN/Transformer layers in the model')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_bottlenecks', type=int, default=4, help='Number of bottlenecks') # If used by your model
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Dimension of feedforward networks')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (used for model init)')
    parser.add_argument('--readout', type=str, default='mean', help="Readout function (mean/sum/max)")
    parser.add_argument('--layer_norm', type=bool, default=True, help="Use layer normalization")
    parser.add_argument('--batch_norm', type=bool, default=False, help="Use batch normalization")
    parser.add_argument('--residual', type=bool, default=True, help="Use residual connections")
    parser.add_argument('--lappe', action='store_true', default=True, help='Use laplacian PE') # Set default based on common usage
    parser.add_argument('--lap_dim', type=int, default=8, help='Dimension for LapEncoding pre-transform') # Adjusted default
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for the final prediction head (used for model init)')


    # --- Dataset processing parameters (should match original dataset creation) ---
    parser.add_argument('--h', type=int, default=1, help='Value for HHopSubgraphs h parameter')
    parser.add_argument('--max_nodes_per_hop', type=int, default=None, help='Value for HHopSubgraphs max_nodes_per_hop (None means no limit)') # Adjusted default

    # --- Positional Encoding parameters (if used by the model directly, not just pre_transform) ---
    # These might only be needed if your GNN_Transformer uses them internally beyond the dataset transform stage
    parser.add_argument('--pos_enc', choices=[None, 'diffusion', 'pstep', 'adj'], default=None, help='Type of positional encoding used *within* the model')
    parser.add_argument('--pos_enc_dim', type=int, default=32, help='Dimension for *internal* positional encoding')
    parser.add_argument('--normalization', choices=[None, 'sym', 'rw'], default='sym', help='Normalization for Laplacian PE (if used internally)')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta for diffusion kernel (if used internally)')
    parser.add_argument('--p', type=int, default=2, help='P for p-step kernel (if used internally)')
    parser.add_argument('--zero_diag', action='store_true', help='Zero diagonal for PE matrix (if used internally)')

    # --- Parse arguments ---
    args = parser.parse_args()

    # --- Run Generation ---
    generate_fused_graphs(args)