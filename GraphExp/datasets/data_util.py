
from collections import namedtuple, Counter
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F

import dgl
from dgl.data import (
    load_data, 
    TUDataset, 
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader

from sklearn.preprocessing import StandardScaler

import scipy.io as sio
GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset
}


def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def load_dataset(dataset_name):
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    else:
        dataset = GRAPH_DICT[dataset_name]()

    if dataset_name == "ogbn-arxiv":
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    else:
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)


def load_inductive_dataset(dataset_name):
    if dataset_name == "ppi":
        batch_size = 2
        # define loss function
        # create the dataset
        train_dataset = PPIDataset(mode='train')
        valid_dataset = PPIDataset(mode='valid')
        test_dataset = PPIDataset(mode='test')
        train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size)
        valid_dataloader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        eval_train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        g = train_dataset[0]
        num_classes = train_dataset.num_labels
        num_features = g.ndata['feat'].shape[1]
    else:
        _args = namedtuple("dt", "dataset")
        dt = _args(dataset_name)
        batch_size = 1
        dataset = load_data(dt)
        num_classes = dataset.num_classes

        g = dataset[0]
        num_features = g.ndata["feat"].shape[1]

        train_mask = g.ndata['train_mask']
        feat = g.ndata["feat"]
        feat = scale_feats(feat)
        g.ndata["feat"] = feat

        g = g.remove_self_loop()
        g = g.add_self_loop()

        train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
        train_g = dgl.node_subgraph(g, train_nid)
        train_dataloader = [train_g]
        valid_dataloader = [g]
        test_dataloader = valid_dataloader
        eval_train_dataloader = [train_g]
        
    return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes



# def load_graph_classification_dataset(dataset_name, deg4feat=False, PE=True):
#     dataset_name = dataset_name.upper()
#     dataset = TUDataset(dataset_name)
#     graph, _ = dataset[0]
#     if "attr" not in graph.ndata:
#         if "node_labels" in graph.ndata and not deg4feat:
#             print("Use node label as node features")
#             feature_dim = 0
#             for g, _ in dataset:
#                 feature_dim = max(feature_dim, g.ndata["node_labels"].max().item())
#
#             feature_dim += 1
#             x_attr = []
#             for g, l in dataset:
#                 node_label = g.ndata["node_labels"].view(-1)
#                 feat = F.one_hot(node_label, num_classes=feature_dim).float()
#                 g.ndata["attr"] = feat
#                 x_attr.append(feat)
#             x_attr = torch.cat(x_attr, dim=0).numpy()
#
#             scaler = StandardScaler()
#             scaler.fit(x_attr)
#             for g, l in dataset:
#                 g.ndata['attr'] = torch.from_numpy(scaler.transform(g.ndata['attr'])).float()
#
#
#
#         else:
#             print("Using degree as node features")
#             feature_dim = 0
#             degrees = []
#             for g, _ in dataset:
#                 feature_dim = max(feature_dim, g.in_degrees().max().item())
#                 degrees.extend(g.in_degrees().tolist())
#             MAX_DEGREES = 400
#
#             oversize = 0
#             for d, n in Counter(degrees).items():
#                 if d > MAX_DEGREES:
#                     oversize += n
#             # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
#             feature_dim = min(feature_dim, MAX_DEGREES)
#
#             feature_dim += 1
#             x_attr = []
#             for g, l in dataset:
#                 degrees = g.in_degrees()
#                 degrees[degrees > MAX_DEGREES] = MAX_DEGREES
#
#                 feat = F.one_hot(degrees, num_classes=feature_dim).float()
#                 g.ndata["attr"] = feat
#                 x_attr.append(feat)
#             x_attr = torch.cat(x_attr, dim=0).numpy()
#             scaler = StandardScaler()
#             scaler.fit(x_attr)
#             for g, l in dataset:
#                 g.ndata['attr'] = torch.from_numpy(scaler.transform(g.ndata['attr'])).float()
#     else:
#         print("******** Use `attr` as node features ********")
#         feature_dim = graph.ndata["attr"].shape[1]
#
#     labels = torch.tensor([x[1] for x in dataset])
#
#     num_classes = torch.max(labels).item() + 1
#     dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]
#
#     print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")
#
#     return dataset, (feature_dim, num_classes)


def load_graph_classification_dataset(fmri_filepath, labels_filepath, deg4feat=False):
    graphs = []
    labels = []

    # 读取标签数据
    label_data = sio.loadmat(labels_filepath)
    label_list = label_data['labels'].flatten()  # 假设标签数据是一个列向量，转换为一维数组

    # 读取图数据
    data = sio.loadmat(fmri_filepath)
    all_matrices = data['all_matrices']  # 460x1 的单元格数组

    # 确保标签数量与图数据数量匹配
    assert len(label_list) == all_matrices.shape[0], "Number of labels must match the number of graphs."

    for i in range(all_matrices.shape[0]):
        # 提取每个图的数据
        matrix = all_matrices[i][0]  # 每个单元格中的 90x90 矩阵

        # 创建 DGL 图
        g = dgl.graph(([], []), num_nodes=matrix.shape[0])

        # 添加边
        src, dst = np.nonzero(matrix)
        g.add_edges(src, dst)

        # 添加节点特征
        g.ndata['attr'] = torch.tensor(matrix, dtype=torch.float32)

        # 这里我们假设没有边特征矩阵，你可以根据需要添加边特征

        graphs.append(g)
        labels.append(torch.tensor(label_list[i], dtype=torch.long))  # 确保标签是长整型张量

    dataset = list(zip(graphs, labels))

    # 处理节点特征
    if not deg4feat:
        print("Processing node features")
        # 直接获取特征维度
        feature_dim = graphs[0].ndata['attr'].shape[1]
        scaler = StandardScaler()
        x_attr = [g.ndata['attr'].numpy() for g, _ in dataset]
        x_attr = np.concatenate(x_attr, axis=0)
        scaler.fit(x_attr)
        for g, _ in dataset:
            g.ndata['attr'] = torch.from_numpy(scaler.transform(g.ndata['attr'].numpy())).float()
    else:
        print("Using degree as node features")
        feature_dim = 0
        degrees = []
        for g, _ in dataset:
            feature_dim = max(feature_dim, g.in_degrees().max().item())
            degrees.extend(g.in_degrees().tolist())
        MAX_DEGREES = 400
        feature_dim = min(feature_dim, MAX_DEGREES) + 1

        x_attr = []
        for g, _ in dataset:
            degrees = g.in_degrees()
            degrees[degrees > MAX_DEGREES] = MAX_DEGREES
            feat = torch.nn.functional.one_hot(degrees, num_classes=feature_dim).float()
            g.ndata["attr"] = feat
            x_attr.append(feat)

        x_attr = torch.cat(x_attr, dim=0).numpy()
        scaler = StandardScaler()
        scaler.fit(x_attr)

        for g, _ in dataset:
            g.ndata['attr'] = torch.from_numpy(scaler.transform(g.ndata['attr'].numpy())).float()

    # 提取标签
    labels = torch.tensor(labels)
    num_classes = torch.max(labels).item() + 1

    # 处理数据集中的每个图，添加自环
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (feature_dim, num_classes)

# 定义填充函数
def pad_to_shape(matrix, target_shape):
    """
    将矩阵填充到指定形状，使用 0 进行填充。
    :param matrix: 需要填充的矩阵
    :param target_shape: 目标形状，例如 (90, 90)
    :return: 填充后的矩阵
    """
    pad_rows = target_shape[0] - matrix.shape[0]
    pad_cols = target_shape[1] - matrix.shape[1]
    padded_matrix = np.pad(matrix, ((0, pad_rows), (0, pad_cols)), mode='constant')
    return padded_matrix
# 加载社区划分数据
# 主要函数
def load_multimodal_graph_classification_dataset(fmri_filepath, dti_filepath, labels_filepath, deg4feat=False):
    graphs_fmri = []
    graphs_dti = []
    labels = []

    # 读取标签数据
    label_data = sio.loadmat(labels_filepath)
    label_list = label_data['labels'].flatten()

    # 读取 fMRI 图数据
    fmri_data = sio.loadmat(fmri_filepath)
    fmri_matrices = fmri_data['fcn_corr']  # 假设是 250x1 的单元格数组
    # 读取 DTI 图数据
    dti_data = sio.loadmat(dti_filepath)
    dti_matrices = dti_data['scn_corr']  # 假设也是 250x1 的单元格数组

    # 目标填充维度
    target_dim = (90, 90)
# 社区边矩阵   processed_matrices
    fmri_community = './datasets/ppmi/fcn_corrHcPd/processed_first_partition.mat'
    dti_community = './datasets/ppmi/scn_corrHcPd/processed_first_partition.mat'
    # 社区划分
    # 读取 fMRI 和 DTI 的社区划分信息
    fmri_partition_filepath = './datasets/ppmi/fcn_corrHcPd/Fcn_community_first_partition.mat'
    dti_partition_filepath = './datasets/ppmi/scn_corrHcPd/Scn_community_first_partition.mat'
    fmri_partitions = sio.loadmat(fmri_partition_filepath)['community_partitions']
    dti_partitions = sio.loadmat(dti_partition_filepath)['community_partitions']
    # 读取 fMRI 图数据
    fmri_data_community = sio.loadmat(fmri_community)
    fmri_matrices_community = fmri_data_community['processed_matrices']  # 假设是 460x1 的单元格数组
    dti_data_community = sio.loadmat(dti_community)
    dti_matrices_community = dti_data_community['processed_matrices']  # 假设也是 460x1 的单元格数组
    for i in range(len(dti_matrices)):
        # 获取 fMRI  dti 的attr矩阵
        fmri_matrix = fmri_matrices[i][0]
        dti_matrix = dti_matrices[i][0]
        # 获取 社区划分后的 边矩阵
        fmri_matrix_community = fmri_matrices_community[i][0]
        dti_matrix_community = dti_matrices_community[i][0]
        # 获取 社区划分信息矩阵
        fmri_matrix_partitions = fmri_partitions[i][0]
        dti_matrix_partitions = dti_partitions[i][0]
        if dti_matrix.shape != target_dim:
            print(f"Patient {i}: DTI matrix shape {dti_matrix.shape} differs from target {target_dim}, padding required.")
            dti_matrix = pad_to_shape(dti_matrix, target_dim)
        # 创建 fMRI 图
        g_fmri = dgl.graph(([], []), num_nodes=fmri_matrix.shape[0])
        src, dst = np.nonzero(fmri_matrix_community)
        print(f"Source indices: {src}, Destination indices: {dst}")
        weights_fmri = fmri_matrix_community[src, dst]  # 获取边的权重
        # 确保权重非空并添加边
        if len(src) > 0 and len(dst) > 0:
            weights_fmri = fmri_matrix_community[src, dst]
            g_fmri.add_edges(src, dst, data={'weight': torch.tensor(weights_fmri, dtype=torch.float32)})
        else:
            g_fmri.add_edges([], [], data={'weight': torch.zeros(0, dtype=torch.float32)})

        g_fmri.add_edges(src, dst, data={'weight': torch.tensor(weights_fmri, dtype=torch.float32)})
        g_fmri.ndata['attr'] = torch.tensor(fmri_matrix, dtype=torch.float32)
        g_fmri.ndata['community'] = torch.tensor(fmri_matrix_partitions, dtype=torch.long)  # 添加社区划分信息

        # 创建 DTI 图
        g_dti = dgl.graph(([], []), num_nodes=dti_matrix.shape[0])
        src, dst = np.nonzero(dti_matrix_community)
        weights_dti = dti_matrix_community[src, dst]  # 获取边的权重
        # 检查边权重是否为空
        if weights_dti.size == 0:
            print(f"Warning: No edges for DTI graph {i}.")
            # 你可以选择添加自环或使用默认权重
            g_dti.add_edges(src, dst)  # 仅添加边不带权重
        else:
            g_dti.add_edges(src, dst, data={'weight': torch.tensor(weights_dti, dtype=torch.float32)})

        g_dti.add_edges(src, dst, data={'weight': torch.tensor(weights_dti, dtype=torch.float32)})
        g_dti.ndata['attr'] = torch.tensor(dti_matrix, dtype=torch.float32)
        g_dti.ndata['community'] = torch.tensor(dti_matrix_partitions, dtype=torch.long)  # 添加社区划分信息
        # **调试信息：打印 fMRI 和 DTI 图的节点和边数量**
        print(f"fMRI Graph {i}: {g_fmri.num_nodes()} nodes, {g_fmri.num_edges()} edges")
        print(f"DTI Graph {i}: {g_dti.num_nodes()} nodes, {g_dti.num_edges()} edges")
        # print(f"Graph {i}: edata_schemes={g_fmri.edata_schemes}")
        # 保存图和标签
        graphs_fmri.append(g_fmri)
        graphs_dti.append(g_dti)
        labels.append(label_list[i])

    # 创建数据集
    dataset = list(zip(graphs_fmri, graphs_dti, labels))
    if not deg4feat:
        print("Processing node features for fMRI and DTI")
        # 获取特征维度
        feature_dim_fmri = graphs_fmri[0].ndata['attr'].shape[1]
        feature_dim_dti = graphs_dti[0].ndata['attr'].shape[1]
    # 转换标签为 tensor
    labels = torch.tensor(labels, dtype=torch.long)

    # 获取类别数量
    num_classes = torch.max(labels).item() + 1

    # 对图进行自环处理
    dataset = [(g_fmri.remove_self_loop().add_self_loop(),
                g_dti.remove_self_loop().add_self_loop(),
                label) for g_fmri, g_dti, label in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Classes: {num_classes} ********")

    # 返回数据集以及 fMRI 和 DTI 特征维度，和类别数量
    return dataset, (feature_dim_fmri, feature_dim_dti, num_classes)