#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File Name:     main_graph.py
# Author:        Yang Run
# Created Time:  2022-10-28  22:32
# Last Modified: <none>-<none>
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
import numpy as np

import argparse

import shutil
import time
import os.path as osp

import dgl
import warnings
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl.dataloading import GraphDataLoader
from dgl import RandomWalkPE

import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from dgl.nn.functional import edge_softmax

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from utils.utils import (create_optimizer, create_pooler, set_random_seed, compute_ppr)

from datasets.data_util import load_graph_classification_dataset, load_sc_pkl_to_dgl_dataset

from models import DDM

import multiprocessing
from multiprocessing import Pool


from utils import comm
from utils.collect_env import collect_env_info
from utils.logger import setup_logger
from utils.misc import mkdir

from evaluator import graph_classification_evaluation
import yaml
from easydict import EasyDict as edict
from sklearn.model_selection import train_test_split
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)


parser = argparse.ArgumentParser(description='Graph DGL Training')
parser.add_argument('--resume', '-r', action='store_true', default=False,
                    help='resume from checkpoint')
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--yaml_dir", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
args = parser.parse_args()


def pretrain(model, train_loader, optimizer, device, epoch, logger=None):
    model.train()
    loss_list = []
    for batch in train_loader:
        batch_g, _ = batch
        batch_g = batch_g.to(device)
        feat = batch_g.ndata["attr"]
        loss, loss_dict = model(batch_g, feat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    lr = optimizer.param_groups[0]['lr']
    logger.info(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f} | lr: {lr:.6f}")


def collate_fn(batch):
    graphs, labels = map(list, zip(*batch))

    
    batch_g = dgl.batch(graphs)

    labels = torch.stack(labels, dim=0)

    return batch_g, labels


def save_checkpoint(state, is_best, filename):
    ckp = osp.join(filename, 'checkpoint.pth.tar')
    # ckp = filename + "checkpoint.pth.tar"
    torch.save(state, ckp)
    if is_best:
        shutil.copyfile(ckp, filename+'/model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, alpha, decay, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 80 epochs"""
    lr = lr * (alpha ** (epoch // decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def balance_dataset_by_downsampling(graph_tuples):
    """
    Balance a dataset by downsampling the majority class to match the minority class size.

    Args:
        graph_tuples: List of tuples (graph, label)

    Returns:
        Balanced list of tuples
    """
    import torch
    import numpy as np
    from collections import Counter

    # Extract labels
    labels = [label.item() if isinstance(label, torch.Tensor) else label for _, label in graph_tuples]

    # Count class distribution
    class_counts = Counter(labels)
    print(f"Original class distribution: {class_counts}")

    # Find minority class size
    if not class_counts: # Handle cases where labels might be missing or input is strange
        print("Warning: No class counts found. Returning empty list.")
        return []
    min_class_size = min(class_counts.values())
    print(f"Minority class size: {min_class_size}")

    # Group samples by class using indices
    class_indices = {}
    for idx, (_, label) in enumerate(graph_tuples):
        label_val = label.item() if isinstance(label, torch.Tensor) else label
        if label_val not in class_indices:
            class_indices[label_val] = []
        class_indices[label_val].append(idx)

    # Downsample majority classes
    balanced_indices = []
    for class_label, indices in class_indices.items():
        count = len(indices)
        # Ensure we don't try to sample more than available, though choice handles this if replace=False
        sample_size = min(min_class_size, count)
        if count >= sample_size:
             # Randomly select sample_size indices from this class without replacement
            selected_indices = np.random.choice(indices, sample_size, replace=False)
            balanced_indices.extend(selected_indices)
        else:
            print(f"Warning: Class {class_label} has {count} samples, less than min size {min_class_size}.")

    balanced_graphs_only = [graph_tuples[i][0] for i in balanced_indices]

    # Verify new distribution
    balanced_labels_check = [graph_tuples[i][1].item() if isinstance(graph_tuples[i][1], torch.Tensor) else graph_tuples[i][1]
                       for i in balanced_indices]
    print(f"Balanced class distribution: {Counter(balanced_labels_check)}")
    print(f"Total balanced samples: {len(balanced_graphs_only)}")

    return balanced_graphs_only


def main(cfg):
    best_auc = float('-inf')
    best_f1_epoch = float('inf')

    if cfg.output_dir:
        mkdir(cfg.output_dir)
        mkdir(cfg.checkpoint_dir)
        
    logger = setup_logger("graph", cfg.output_dir, comm.get_rank(), filename='train_log.txt')
    logger.info("Rank of current process: {}. World size: {}".format(comm.get_rank(), comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Command line arguments: " + str(args))

    # shutil.copyfile('./params.yaml', cfg.output_dir + '/params.yaml')
    shutil.copyfile('./main_graph.py', cfg.output_dir + '/graph.py')
    shutil.copyfile('./models/DDM.py', cfg.output_dir + '/DDM.py')
    shutil.copyfile('./models/mlp_gat.py', cfg.output_dir + '/mlp_gat.py')

    # graphs, (num_features, num_classes) = load_graph_classification_dataset(cfg.DATA.data_name,
    #                                                                         deg4feat=cfg.DATA.deg4feat,
    #                                                                         PE=False)


    graphs, (num_features, num_classes) = load_sc_pkl_to_dgl_dataset(cfg.DATA.sc_file_path,
                                                                            )
    # train_graphs, (num_features, num_classes) = load_sc_pkl_to_dgl_dataset("/home/jinghu/diffusion/testDiff/RH-BrainFS/fused_dataset_output/ZDXX_one_disease_fused_train.pkl")
    # eval_graphs, (num_features, num_classes) = load_sc_pkl_to_dgl_dataset("/home/jinghu/diffusion/testDiff/RH-BrainFS/fused_dataset_output/ZDXX_one_disease_fused_eval.pkl")
    # train_tuples = [(g, g.graph_data['label']) for g in train_graphs]
    # val_tuples = [(g, g.graph_data['label']) for g in eval_graphs]


    cfg.num_features = num_features

    graph_tuples = [(g, g.graph_data['label']) for g in graphs]
    labels = [label.cpu() if isinstance(label, torch.Tensor) else label for g, label in graph_tuples]

    # balanced_graphs = balance_dataset_by_downsampling(graph_tuples)//never need anymore
    # train_idx = torch.arange(len(graphs))
    # train_sampler = SubsetRandomSampler(train_idx)

    train_idx, val_idx = train_test_split(
        range(len(graph_tuples)),
        test_size=0.2, # Or your desired split ratio
        random_state=42, # For reproducibility
        stratify=labels)
    train_tuples = [graph_tuples[i] for i in train_idx]
    val_tuples = [graph_tuples[i] for i in val_idx]

    # balanced_train_graphs = balance_dataset_by_downsampling(train_tuples)
    # train_graphs = [t[0] for t in train_tuples] # Get just the graphs for the loader
    # val_graphs = [t[0] for t in val_tuples] # Get just the graphs for the loader

    train_loader = GraphDataLoader(train_tuples, collate_fn=collate_fn,
                               batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True)
    
    eval_loader = GraphDataLoader(val_tuples, collate_fn=collate_fn,
                              batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        all_feats = torch.cat([g.ndata['attr'] for g, _ in train_tuples])
        uniques   = torch.unique(all_feats, dim=0)
    print("unique rows:", uniques.shape[0], "out of", all_feats.shape[0])

    # train_loader = GraphDataLoader(graphs, sampler=train_sampler, collate_fn=collate_fn,
    #                                batch_size=cfg.DATALOADER.BATCH_SIZE, pin_memory=False)
    # eval_loader = GraphDataLoader(graphs, collate_fn=collate_fn, batch_size=len(graphs), shuffle=False)

    pooler = create_pooler(cfg.MODEL.pooler)

    acc_list = []
    for i, seed in enumerate(cfg.seeds):
        logger.info(f'Run {i}th for seed {seed}')
        set_random_seed(seed)

        ml_cfg = cfg.MODEL
        ml_cfg.update({'in_dim': num_features})
        model = DDM(**ml_cfg)
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('Total trainable params num : {}'.format(total_trainable_params))
        model.to(cfg.DEVICE)

        optimizer = create_optimizer(cfg.SOLVER.optim_type, model, cfg.SOLVER.LR, cfg.SOLVER.weight_decay)

        start_epoch = 0
        if args.resume:
            if osp.isfile(cfg.pretrain_checkpoint_dir):
                logger.info("=> loading checkpoint '{}'".format(cfg.checkpoint_dir))
                checkpoint = torch.load(cfg.checkpoint_dir, map_location=torch.device('cpu'))
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                            .format(cfg.checkpoint_dir, checkpoint['epoch']))

        logger.info("----------Start Training----------")
        
        for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
            adjust_learning_rate(optimizer, epoch=epoch, alpha=cfg.SOLVER.alpha, decay=cfg.SOLVER.decay, lr=cfg.SOLVER.LR)
            pretrain(model, train_loader, optimizer, cfg.DEVICE, epoch, logger)
            if ((epoch + 1) % 1 == 0) & (epoch > 1):
                model.eval()
                test_f1, test_auc = graph_classification_evaluation(model, cfg.eval_T, pooler, eval_loader,
                                                          cfg.DEVICE, logger)
                is_best = test_auc > best_auc
                if is_best:
                    best_f1_epoch = epoch
                best_auc = max(test_auc, best_auc)
                logger.info(f"Epoch {epoch}: get test f1 score: {test_f1: .3f}")
                # logger.info(f"best_f1 {best_f1:.3f} at epoch {best_f1_epoch}")
                logger.info(f"best_auc {best_auc:.3f} at epoch {best_f1_epoch}")
                save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': model.state_dict(),
                                 'best_auc': best_auc,
                                    'best_f1': test_f1,
                                 'optimizer': optimizer.state_dict()},
                                is_best, filename=cfg.checkpoint_dir)
        acc_list.append(best_auc)
    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    logger.info((f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}"))
    return final_acc


if __name__ == "__main__":
    root_dir = osp.abspath(osp.dirname(__file__))
    yaml_dir = osp.join(root_dir, 'MUTAG.yaml')
    output_dir = osp.join(root_dir, 'log')
    checkpoint_dir = osp.join(output_dir, "checkpoint")

    yaml_dir = args.yaml_dir if args.yaml_dir else yaml_dir
    output_dir = args.output_dir if args.output_dir else output_dir
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else checkpoint_dir

    with open(yaml_dir, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    cfg = edict(config)

    cfg.output_dir, cfg.checkpoint_dir = output_dir, checkpoint_dir
    print(cfg)
    f1 = main(cfg)













