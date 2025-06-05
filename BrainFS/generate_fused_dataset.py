import argparse
import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader # Use PyG DataLoader
from model_components.MBT import Alternately_Attention_Bottlenecks # Import your modified model
from NewdataLoader import MyOwnDataset # Import your dataset class
from transform import HHopSubgraphs, LapEncoding # Import necessary transforms (if used during dataset creation)
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm # For progress bar

def generate_fused_graphs(args):
    # Determine CUDA usage and device
    args.use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.use_cuda else "cpu")

    raw_ckpt    = torch.load(args.model_ckpt_path, map_location=device)
    state_dict  = raw_ckpt.get('state_dict', raw_ckpt)
    emb_sc_w    = state_dict['node_embedding_sc.weight']
    emb_fc_w    = state_dict['node_embedding_fc.weight']
    args.hidden_dim  = emb_sc_w.size(0)
    args.sc_features = emb_sc_w.size(1)
    args.fc_features = emb_fc_w.size(1)
    print(f"Inferred hidden_dim={args.hidden_dim}, sc_features={args.sc_features}, fc_features={args.fc_features}")

    subgraph_pre = LapEncoding(dim=args.lap_dim)
    transform    = HHopSubgraphs(
        h=args.h,
        max_nodes_per_hop=args.max_nodes_per_hop,
        node_label='hop',
        use_rd=False,
        subgraph_pretransform=subgraph_pre
    )
    full_dataset = MyOwnDataset(args.dataset, args=args, pre_transform=transform)
    print(f"Total graphs loaded: {len(full_dataset)}")

    # --- Split into train/eval ---
    labels = [int(data.y.item()) for data in full_dataset]
    indices = list(range(len(full_dataset)))
    train_idx, eval_idx = train_test_split(
        indices,
        test_size=args.eval_ratio,
        stratify=labels,
        random_state=42
    )
    # train_keys = {full_dataset[i]['subjectkey'] for i in train_idx}
    # eval_keys  = {full_dataset[i]['subjectkey'] for i in eval_idx}
    # assert train_keys.isdisjoint(eval_keys), "SPLIT LEAKAGE DETECTED!"

    train_set = torch.utils.data.Subset(full_dataset, train_idx)
    eval_set  = torch.utils.data.Subset(full_dataset, eval_idx)
    print(f"Train: {len(train_set)}, Eval: {len(eval_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    eval_loader  = DataLoader(eval_set,  batch_size=args.batch_size, shuffle=False)

    # # --- Load checkpoint and infer embedding dimensions ---
    # raw_ckpt = torch.load(args.model_ckpt_path, map_location=device)
    # state_dict = raw_ckpt.get('state_dict', raw_ckpt)
    # emb_sc_w = state_dict['node_embedding_sc.weight']
    # emb_fc_w = state_dict['node_embedding_fc.weight']
    # args.hidden_dim = emb_sc_w.size(0)
    # args.sc_features = emb_sc_w.size(1)
    # args.fc_features = emb_fc_w.size(1)
    # print(f"Inferred hidden_dim={args.hidden_dim}, sc_features={args.sc_features}, fc_features={args.fc_features}")

    # --- Initialize and load model ---
    model = Alternately_Attention_Bottlenecks(args)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # --- Generation helper ---
    def run_loader(loader):
        fused = []
        for data in tqdm(loader, desc="Fusing graphs"):
            data = data.to(device)
            batch_feats = model(data, return_fused_nodes=True)
            for i in range(data.num_graphs):
                nf = batch_feats[i].cpu()
                y = data.y[i].unsqueeze(0).cpu()
                start, end = int(data.ptr[i]), int(data.ptr[i+1])
                sc_mask = (data.sc_original_edge_index[0] >= start) & (data.sc_original_edge_index[0] < end)
                fc_mask = (data.fc_original_edge_index[0] >= start) & (data.fc_original_edge_index[0] < end)
                sc_ei = data.sc_original_edge_index[:, sc_mask] - start
                fc_ei = data.fc_original_edge_index[:, fc_mask] - start + args.sc_features
                edge_index = torch.cat([sc_ei, fc_ei], dim=1).cpu()
                fused.append({'node_feat': nf, 'edge_index': edge_index, 'edge_attr': None, 'y': y})
        return fused

    # --- Generate and save ---
    train_fused = run_loader(train_loader)
    eval_fused  = run_loader(eval_loader)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(train_fused, os.path.join(args.output_dir, f"{args.dataset}_fused_train.pkl"))
    torch.save(eval_fused,  os.path.join(args.output_dir, f"{args.dataset}_fused_eval.pkl"))

    print(f"Saved {len(train_fused)} train & {len(eval_fused)} eval fused graphs.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fused Graph Dataset Generation using GNN_Transformer')

    # --- Arguments ---
    # Essential args for loading model/data
    parser.add_argument('--dataset', type=str, required=False, help="Name of the original dataset (e.g., ZDXX, HCP_MBT)")
    parser.add_argument('--model_ckpt_path', type=str, required=True, help='Path to the BEST trained GNN_Transformer checkpoint (.pth file)')
    parser.add_argument('--output_dir', type=str, default='./fused_dataset_output', help='Directory to save the fused graph list')
    parser.add_argument('--batch_size', type=int, default=16)

    # Add model architecture parameters matching the *trained* model
    # Copy these from your main.py's parser or load from json
    parser.add_argument('--num_heads', type=int, default=4, help='value for num_heads')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='maximum number of epochs')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio (used for model init, but eval mode disables it)')
    # Add any other args required by GNN_Transformer.__init__ or MyOwnDataset.__init__
    parser.add_argument('--readout', type=str, default='mean', help="Readout function (mean/sum/max) - needed for model init")
    # Add args needed by the 'process' method if they aren't hardcoded
    # Add args needed by MyOwnDataset if not just the name
    parser.add_argument('--h', type=int, default=1, help='Value for HHopSubgraphs h parameter')
    parser.add_argument('--max_nodes_per_hop', type=int, default=5, help='Value for HHopSubgraphs max_nodes_per_hop')
    parser.add_argument('--lap_dim', type=int, default=4, help='Dimension for LapEncoding pre-transform within HHopSubgraphs')
    parser.add_argument('--pos_enc', choices=[None, 'diffusion', 'pstep', 'adj'], default='pstep')
    parser.add_argument('--pos_enc_dim', type=int, default=32, help='hidden size')
    parser.add_argument('--normalization', choices=[None, 'sym', 'rw'], default='sym',
                        help='normalization for Laplacian')
    parser.add_argument('--beta', type=float, default=1.0, help='bandwidth for the diffusion kernel')
    parser.add_argument('--p', type=int, default=2, help='p step random walk kernel')
    parser.add_argument('--zero_diag', action='store_true', help='zero diagonal for PE matrix')
    parser.add_argument('--lappe', action='store_true', help='use laplacian PE',default=True)
    # parser.add_argument('--lap_dim', type=int, default=32, help='dimension for laplacian PE')
    parser.add_argument('--layer_norm', type=bool, default=True, help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', type=bool, default=False, help="Please give a value for batch_norm")
    parser.add_argument('--residual', type=bool, default=True, help="Please give a value for residual")
    parser.add_argument('--num_classes', type=int, default=2, help='the number of classes (HC/MDD)')
    parser.add_argument('--eval_ratio', type=float, default=0.2,
                        help='Fraction of data to hold out for eval')
    parser.add_argument('--num_layers', type=int, default=4, help='the numbers of convolution layers')
    parser.add_argument('--num_bottlenecks', type=int, default=4, help='the numbers of bottlenecks')
    


    

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    generate_fused_graphs(args)