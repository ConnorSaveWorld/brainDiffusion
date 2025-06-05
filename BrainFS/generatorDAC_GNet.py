import argparse
import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader # Use PyG DataLoader
from model_components.MBT import DAC_GNet # Import your modified model
from NewdataLoader import MyOwnDataset # Import your dataset class
# Assuming these transforms are used during dataset creation/processing
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
    try:
        raw_ckpt = torch.load(args.model_ckpt_path, map_location=device)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.model_ckpt_path}")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    state_dict = raw_ckpt.get('state_dict', raw_ckpt)
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

    if 'node_embedding_sc.weight' in state_dict and 'node_embedding_fc.weight' in state_dict:
        emb_sc_w = state_dict['node_embedding_sc.weight']
        emb_fc_w = state_dict['node_embedding_fc.weight']
        inferred_hidden_dim = emb_sc_w.size(0)
        inferred_sc_features = emb_sc_w.size(1)
        inferred_fc_features = emb_fc_w.size(1)
        print(f"Inferred from checkpoint: hidden_dim={inferred_hidden_dim}, sc_features={inferred_sc_features}, fc_features={inferred_fc_features}")
        args.hidden_dim = inferred_hidden_dim if args.hidden_dim is None else args.hidden_dim
        args.sc_features = inferred_sc_features if args.sc_features is None else args.sc_features
        args.fc_features = inferred_fc_features if args.fc_features is None else args.fc_features
    else:
        print("Could not infer all dimensions from checkpoint keys. Using provided args or defaults.")
        assert args.hidden_dim is not None, "hidden_dim must be provided if not inferrable"
        assert args.sc_features is not None, "sc_features must be provided if not inferrable"
        assert args.fc_features is not None, "fc_features must be provided if not inferrable"
    print(f"Using effective model params: hidden_dim={args.hidden_dim}, sc_features={args.sc_features}, fc_features={args.fc_features}")


    # --- Load Full Dataset with Transforms ---
    print("Initializing dataset transforms...")
    subgraph_pre = LapEncoding(dim=args.lap_dim)
    transform = HHopSubgraphs(
        h=args.h,
        max_nodes_per_hop=args.max_nodes_per_hop,
        node_label='hop',
        use_rd=False,
        subgraph_pretransform=subgraph_pre
    )
    print(f"Loading dataset: {args.dataset}...")
    try:
        full_dataset = MyOwnDataset(args.dataset, args=args, pre_transform=transform)
    except Exception as e:
        print(f"Error initializing dataset {args.dataset}: {e}")
        print("Please ensure your MyOwnDataset class and data paths are correct.")
        return

    print(f"Total graphs loaded: {len(full_dataset)}")
    if len(full_dataset) == 0:
        print("Error: Dataset is empty. Check dataset path and processing.")
        return

    # --- Define Splits ---
    print("Splitting data into train/eval...")
    labels = []
    for i in range(len(full_dataset)):
        try:
            labels.append(int(full_dataset[i].y.item()))
        except AttributeError:
            print(f"Error: Graph {i} in dataset does not have 'y' attribute or it's not a scalar.")
            return
        except ValueError:
            print(f"Error: Graph {i} 'y' attribute cannot be converted to int.")
            return

    indices = list(range(len(full_dataset)))

    if len(set(labels)) < 2:
        print("Warning: Only one class found in labels. Stratified split may behave like random split.")
        train_idx, eval_idx = train_test_split(
            indices,
            test_size=args.eval_ratio,
            random_state=42
        )
    elif len(full_dataset) * args.eval_ratio < len(set(labels)) or (1-args.eval_ratio) * len(full_dataset) < len(set(labels)): # Ensure enough samples for stratification in both splits for each class
         print(f"Warning: Very small dataset or eval_ratio for stratified split given number of classes. Splitting without stratification.")
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
                random_state=42
            )
        except ValueError as e:
            print(f"Warning: Stratified split failed ({e}). Falling back to non-stratified split.")
            train_idx, eval_idx = train_test_split(
                indices,
                test_size=args.eval_ratio,
                random_state=42
            )

    train_idx_set = set(train_idx)
    eval_idx_set = set(eval_idx)
    print(f"Defined splits: Train={len(train_idx)}, Eval={len(eval_idx)}")

    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Initialize and load model ---
    print("Initializing model...")
    try:
        model = DAC_GNet(args)
    except Exception as e:
        print(f"Error initializing DAC_GNet model: {e}")
        print("Ensure all required model arguments are correctly provided and match the model definition.")
        return

    print("Loading model state dict...")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Error loading state_dict into model: {e}")
        print("This often means the model architecture defined by args does not match the checkpoint's architecture.")
        return
    model.to(device).eval()

    # --- Generation and Saving Function (Chunked) ---
    @torch.no_grad()
    def generate_and_save_fused_chunked(loader, train_indices_set, eval_indices_set,
                                        output_dir_path, dataset_name_str, chunk_size_val):
        processed_graphs_total = 0
        current_chunk_data = []
        chunk_counter = 0
        original_graph_idx_counter = 0 # Tracks original index across all batches

        for batch_idx, data_batch in enumerate(tqdm(loader, desc="Processing batches")):
            data_batch = data_batch.to(device)
            
            # Get fused node features from the model
            # CRITICAL: Verify how your model returns features.
            # Option A: Returns a list/tuple of tensors, one per graph in the batch.
            #   batch_fused_node_features_list = model(data_batch, return_fused_nodes=True)
            # Option B: Returns a single tensor for all nodes in the batch (more common for PyG).
            #   all_nodes_fused_features_batched_tensor = model(data_batch, return_fused_nodes=True)
            
            # Assuming Option A for now, based on original code structure. If it's B, adjust feature extraction.
            try:
                batch_fused_node_features_list = model(data_batch, return_fused_nodes=True)
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                # Potentially stop or skip batch
                return 0 # Indicate failure

            if not isinstance(batch_fused_node_features_list, (list, tuple)) or \
               (batch_fused_node_features_list and not isinstance(batch_fused_node_features_list[0], torch.Tensor)):
                print(f"Warning: model output for 'return_fused_nodes=True' is not a list/tuple of tensors as expected.")
                print(f"Type: {type(batch_fused_node_features_list)}")
                if isinstance(batch_fused_node_features_list, torch.Tensor):
                     print(f"Shape: {batch_fused_node_features_list.shape}")
                     print("If model returns a single batched tensor, feature extraction logic needs change (see Option B in comments).")
                # return 0 # Stop if format is unexpected

            num_graphs_in_batch = data_batch.num_graphs
            if isinstance(batch_fused_node_features_list, (list, tuple)) and len(batch_fused_node_features_list) != num_graphs_in_batch:
                print(f"Warning: Number of fused feature tensors ({len(batch_fused_node_features_list)}) "
                      f"does not match number of graphs in batch ({num_graphs_in_batch}). Skipping batch.")
                original_graph_idx_counter += num_graphs_in_batch # Still advance counter
                continue


            for i in range(num_graphs_in_batch):
                split_type = 'train' if original_graph_idx_counter in train_indices_set else 'eval'
                assert original_graph_idx_counter in train_indices_set or original_graph_idx_counter in eval_indices_set, \
                    f"Error: Original index {original_graph_idx_counter} not found in any split."

                # --- Extract features for graph i ---
                if isinstance(batch_fused_node_features_list, (list, tuple)): # Expected path
                    fused_node_feat_tensor = batch_fused_node_features_list[i]
                elif isinstance(batch_fused_node_features_list, torch.Tensor) and batch_idx == 0 and i == 0: # If it was a single tensor, inform once
                    print("Model returned a single tensor; assuming it's batched. Adapting per-graph feature extraction.")
                    print("This warning will only appear once.")
                    # This path would require slicing using data_batch.ptr if it's a single tensor for all batch nodes
                    # Example: fused_node_feat_tensor = batch_fused_node_features_list[data_batch.ptr[i]:data_batch.ptr[i+1]]
                    # For now, if it's a tensor, this will likely fail or be incorrect unless it's [num_graphs, num_nodes_max, feat_dim]
                    # Sticking to list assumption as per original
                    fused_node_feat_tensor = batch_fused_node_features_list[i] # This would be an error if it's a batched node tensor
                else: # Fallback or error if unexpected type persists
                    fused_node_feat_tensor = batch_fused_node_features_list[i]


                label = data_batch.y[i].unsqueeze(0).cpu()

                start_node_idx_in_batch = int(data_batch.ptr[i])
                end_node_idx_in_batch = int(data_batch.ptr[i+1])
                num_nodes_in_original_graph = end_node_idx_in_batch - start_node_idx_in_batch

                assert num_nodes_in_original_graph == fused_node_feat_tensor.shape[0], \
                    f"Graph {original_graph_idx_counter}: Mismatch in node count. Original: {num_nodes_in_original_graph}, Fused: {fused_node_feat_tensor.shape[0]}"

                # --- Reconstruct original combined edge_index ---
                # Ensure sc_original_edge_index and fc_original_edge_index exist on data_batch
                if not (hasattr(data_batch, 'sc_original_edge_index') and hasattr(data_batch, 'fc_original_edge_index')):
                    print(f"Error: Batch data missing 'sc_original_edge_index' or 'fc_original_edge_index'. Cannot reconstruct edges.")
                    return processed_graphs_total # Stop processing

                sc_ei_batch = data_batch.sc_original_edge_index
                sc_mask = (sc_ei_batch[0] >= start_node_idx_in_batch) & (sc_ei_batch[0] < end_node_idx_in_batch) & \
                          (sc_ei_batch[1] >= start_node_idx_in_batch) & (sc_ei_batch[1] < end_node_idx_in_batch)
                sc_ei_graph = sc_ei_batch[:, sc_mask] - start_node_idx_in_batch

                fc_ei_batch = data_batch.fc_original_edge_index
                fc_mask = (fc_ei_batch[0] >= start_node_idx_in_batch) & (fc_ei_batch[0] < end_node_idx_in_batch) & \
                          (fc_ei_batch[1] >= start_node_idx_in_batch) & (fc_ei_batch[1] < end_node_idx_in_batch)
                fc_ei_graph = fc_ei_batch[:, fc_mask] - start_node_idx_in_batch
                
                if sc_ei_graph.numel() > 0:
                    assert sc_ei_graph.max() < num_nodes_in_original_graph, f"SC edge index {sc_ei_graph.max()} out of bounds for {num_nodes_in_original_graph} nodes in graph {original_graph_idx_counter}"
                if fc_ei_graph.numel() > 0:
                    assert fc_ei_graph.max() < num_nodes_in_original_graph, f"FC edge index {fc_ei_graph.max()} out of bounds for {num_nodes_in_original_graph} nodes in graph {original_graph_idx_counter}"

                combined_edge_index = torch.cat([sc_ei_graph, fc_ei_graph], dim=1).cpu()
                # combined_edge_index = torch.unique(combined_edge_index, dim=1) # Optional

                fused_graph_data = {
                    'node_feat': fused_node_feat_tensor.cpu(),
                    'edge_index': combined_edge_index,
                    'edge_attr': None,
                    'y': label,
                    'split': split_type
                }
                current_chunk_data.append(fused_graph_data)
                processed_graphs_total += 1
                original_graph_idx_counter += 1

                if len(current_chunk_data) >= chunk_size_val or original_graph_idx_counter == len(loader.dataset):
                    if current_chunk_data:
                        chunk_filename = os.path.join(output_dir_path, f"{dataset_name_str}_fused_chunk_{chunk_counter}.pkl")
                        print(f"Saving chunk {chunk_counter} ({len(current_chunk_data)} graphs) to {chunk_filename}...")
                        torch.save(current_chunk_data, chunk_filename)
                        current_chunk_data = []
                        chunk_counter += 1
        
        # Save any remaining data in the last chunk (should be covered by loop condition)
        if current_chunk_data: # Just in case, though previous logic should handle it
            chunk_filename = os.path.join(output_dir_path, f"{dataset_name_str}_fused_chunk_{chunk_counter}.pkl")
            print(f"Saving final chunk {chunk_counter} ({len(current_chunk_data)} graphs) to {chunk_filename}...")
            torch.save(current_chunk_data, chunk_filename)
            chunk_counter +=1
            
        print(f"Finished processing. Total fused graphs: {processed_graphs_total}. Saved in {chunk_counter} chunks.")
        return processed_graphs_total

    # --- Generate and Save ---
    print("Starting fused graph generation...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    num_generated = generate_and_save_fused_chunked(
        full_loader,
        train_idx_set,
        eval_idx_set,
        args.output_dir,
        args.dataset,
        chunk_size_val=args.save_chunk_size
    )

    end_time = time.time()
    if num_generated > 0:
        print(f"Successfully generated and saved {num_generated} fused graphs in chunks.")
    else:
        print("No graphs were generated or saved, or an error occurred during generation.")
    print(f"Total time: {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fused Graph Dataset Generation using GNN_Transformer/DAC_GNet')

    # Essential args
    parser.add_argument('--dataset', type=str, required=True, help="Name of the original dataset")
    parser.add_argument('--model_ckpt_path', type=str, required=True, help='Path to the GNN_Transformer/DAC_GNet checkpoint')
    parser.add_argument('--output_dir', type=str, default='./fused_dataset_output', help='Directory to save fused graph chunks')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--eval_ratio', type=float, default=0.2, help='Fraction for eval split')
    parser.add_argument('--save_chunk_size', type=int, default=1000, help='Number of graphs per saved chunk file')

    # Model architecture parameters (provide defaults, ensure they match training or can be inferred)
    parser.add_argument('--hidden_dim', type=int, default=None, help='Model hidden dimension (infer if None)')
    parser.add_argument('--sc_features', type=int, default=None, help='Input SC features (infer if None)')
    parser.add_argument('--fc_features', type=int, default=None, help='Input FC features (infer if None)')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GNN/Transformer layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_bottlenecks', type=int, default=4, help='Number of bottlenecks (if used)')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Feedforward network dimension')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--readout', type=str, default='mean', choices=['mean', 'sum', 'max'], help="Readout function")
    parser.add_argument('--layer_norm', type=bool, default=True, help="Use layer normalization") # Note: bool args can be tricky with store_true/false
    parser.add_argument('--batch_norm', type=bool, default=False, help="Use batch normalization")
    parser.add_argument('--residual', type=bool, default=True, help="Use residual connections")
    parser.add_argument('--lappe', action='store_true', default=True, help='Use Laplacian PE in pre-transform')
    parser.add_argument('--lap_dim', type=int, default=8, help='Dimension for LapEncoding pre-transform')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for model init')

    # Dataset processing parameters
    parser.add_argument('--h', type=int, default=1, help='h for HHopSubgraphs')
    parser.add_argument('--max_nodes_per_hop', type=int, default=None, help='max_nodes_per_hop for HHopSubgraphs (None=no limit)')

    # Model-internal Positional Encoding parameters (if applicable)
    parser.add_argument('--pos_enc', choices=[None, 'diffusion', 'pstep', 'adj'], default=None, help='Internal PE type')
    parser.add_argument('--pos_enc_dim', type=int, default=32, help='Internal PE dimension')
    # ... other PE args like normalization, beta, p, zero_diag

    # DAC_GNet specific parameters
    parser.add_argument('--num_initial_gnn_layers', type=int, default=2, help='Initial GNN layers in DAC_GNet')
    parser.add_argument('--num_coupling_layers', type=int, default=2, help='Dynamic coupling layers in DAC_GNet')
    parser.add_argument('--dim_coupling', type=int, default=None, help='Coupling representation dim (default: hidden_dim // 2)')

    args = parser.parse_args()
    
    # For boolean args that take True/False values rather than flags:
    # A common way to handle them if you want to pass "True" or "False" as string:
    def str_to_bool(value):
        if isinstance(value, bool):
            return value
        if value.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif value.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Example if you were to parse them this way:
    # parser.add_argument('--layer_norm', type=str_to_bool, default=True, help="Use layer normalization")
    # parser.add_argument('--batch_norm', type=str_to_bool, default=False, help="Use batch normalization")
    # parser.add_argument('--residual', type=str_to_bool, default=True, help="Use residual connections")
    # The current `type=bool` for these might not behave as expected from CLI (e.g. '--layer_norm False' might be True).
    # `action='store_true'` or `action='store_false'` is typical for flag-based booleans.
    # For simplicity, current `type=bool` will convert 'False' string to True. Using defaults is safer if not using action flags.
    # Or, ensure they are correctly set if passed from a config file. The current setup mainly relies on defaults for these.

    generate_fused_graphs(args)