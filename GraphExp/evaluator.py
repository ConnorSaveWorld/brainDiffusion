import numpy as np

import argparse

import shutil
import time
import os.path as osp
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl.dataloading import GraphDataLoader
from dgl import RandomWalkPE

import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from dgl.nn.functional import edge_softmax

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score

from utils.utils import (create_optimizer, create_pooler, set_random_seed, compute_ppr)

from datasets.data_util import load_graph_classification_dataset

from models import DDM


import multiprocessing
from multiprocessing import Pool


from utils import comm
from utils.collect_env import collect_env_info
from utils.logger import setup_logger
from utils.misc import mkdir


def graph_classification_evaluation(model, T, pooler, dataloader, device, logger):
    model.eval()
    embed_list = []
    head_list = []
    optim_list = []
    y_list_full = []
    with torch.no_grad():
        # for i, (batch_g, labels) in enumerate(dataloader):
        #      y_list_full.append(labels.cpu()) # Store labels on CPU
        # y_list_full = torch.cat(y_list_full, dim=0).numpy()

        for t in T:
            x_list = []
            y_list = []
            for i, (batch_g, labels) in enumerate(dataloader):
                batch_g = batch_g.to(device)
                feat = batch_g.ndata["attr"]
                out = model.embed(batch_g, feat, t)
                out = pooler(batch_g, out)
                y_list.append(labels)
                x_list.append(out)
            head_list.append(1)
            embed_list.append(torch.cat(x_list, dim=0).cpu().numpy())
        y_list = torch.cat(y_list, dim=0)
    embed_list = np.array(embed_list)
    y_list = y_list.cpu().numpy()
    test_f1, test_f1_std, test_auc, test_auc_std = evaluate_graph_embeddings_using_svm(T, embed_list, y_list)
    logger.info(f"#Test_f1: {test_f1:.4f}±{test_f1_std:.4f}")
    logger.info(f"#Test_AUC: {test_auc:.4f}±{test_auc_std:.4f}")
    return test_f1, test_auc


def inner_func(args):
    T = args[0]
    train_index = args[1]
    test_index = args[2]
    embed_list = args[3]
    y_list = args[4]
    pred_list = []

    score_list = [] # To store decision scores

    # Determine the number of classes
    num_classes = len(np.unique(y_list))
    is_binary = num_classes == 2


    for idx in range(len(T)):
        embeddings = embed_list[idx]
        labels = y_list
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)

        out = clf.predict(x_test)
        pred_list.append(out)

        # Get decision scores
        scores_t = clf.decision_function(x_test)
        score_list.append(scores_t)

    preds = np.stack(pred_list, axis=0)
    preds = torch.from_numpy(preds)
    preds = torch.mode(preds, dim=0)[0].long().numpy()
    f1 = f1_score(y_test, preds, average="micro")

    scores_stack = np.stack(score_list, axis=0)
    avg_scores = np.mean(scores_stack, axis=0)
    if avg_scores.ndim != 1:
        print("Warning: Unexpected score dimension for binary case.")
    if avg_scores.ndim == 1:
        auc = roc_auc_score(y_test, avg_scores)
    return f1, auc


def evaluate_graph_embeddings_using_svm(T, embed_list, y_list):
    ctx = mp.get_context("spawn")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    process_args = [(T, train_index, test_index, embed_list, y_list)
                    for train_index, test_index in kf.split(embed_list[0], y_list)]
    with ctx.Pool(processes=5) as p:
        results_list = p.map(inner_func, process_args)
    f1_results = [res[0] for res in results_list]
    auc_results = [res[1] for res in results_list]

    test_f1_mean = np.mean(f1_results)
    test_f1_std = np.std(f1_results)

    test_auc_mean = np.mean(auc_results)
    test_auc_std = np.std(auc_results)

    return test_f1_mean, test_f1_std, test_auc_mean, test_auc_std
