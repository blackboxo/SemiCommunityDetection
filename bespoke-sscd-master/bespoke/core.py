import collections
from typing import List, Set, Dict, Union

import numpy as np
import tqdm
from scipy import sparse as sp
from sklearn.cluster import KMeans

import torch
import itertools
from torch import nn
import dgl
from dgl.nn.pytorch import GINConv

def prepare_graph(nodes,neighbors,graph):
    nodes = list(nodes)
    boundary = set()
    dgl_graph=dgl.DGLGraph(graph.adj_mat)
    for u in nodes:
        boundary |= neighbors[u]
    boundary.difference_update(nodes)
    if len(boundary) > 200:
        boundary = np.random.choice(list(boundary), size=200, replace=False)
    state = torch.zeros(len(nodes) + len(boundary), dtype=torch.long, device=torch.device('cpu'))
    state[:len(nodes)] = 1
    nodes = nodes + list(boundary)
    subg = dgl_graph.subgraph(nodes)
    subg.copy_from_parent()
    subg.ndata['state'] = state
    return subg

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def swish(x):
    return x * torch.sigmoid(x)

class LinearBlock(nn.Module):

    def __init__(self, in_size, out_size, act_cls=None, norm_type=None, bias=True, residual=True, dropout=0.):
        super().__init__()
        self.residual = residual and (in_size == out_size)
        layers = []
        if norm_type == 'batch_norm':
            layers.append(nn.BatchNorm1d(in_size))
        elif norm_type == 'layer_norm':
            layers.append(nn.LayerNorm(in_size))
        elif norm_type is not None:
            raise NotImplementedError
        if act_cls is not None:
            layers.append(act_cls())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_size, out_size, bias))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        z = self.f(x)
        if self.residual:
            z += x
        return z

def label_nodes(neighbors: Dict[int, Set[int]], n_roles: int, graph, train_comms) -> (np.ndarray, np.ndarray):
    """
    Compute the Jaccard similarities,
    then extract percentiles as features,
    and finally label nodes by k-means.
    """
    # Compute Jaccards
    n_nodes = len(neighbors)
    subgs = []
    for nodes in itertools.chain(train_comms):
        subg = prepare_graph(nodes,neighbors,graph)
        subgs.append(subg)
    batch_g = dgl.batch(subgs, 'state', None)
    state_embedding = nn.Embedding(2, 64)
    h = state_embedding(batch_g.ndata['state'])
    hs = []
    gconv_layers = nn.ModuleList([GINConv(None, 'sum') for i in range(3)])
    fc_layers = nn.ModuleList([LinearBlock(64, 64, act_cls=Swish,
                                           norm_type=None, dropout=0.0)
                               for i in range(3)])
    if batch_g.number_of_edges() == 0:
        for gn, fn in zip(gconv_layers, fc_layers):
            h = fn(h) * (1 + gn.eps)
            hs.append(h)
    else:
        for gn, fn in zip(gconv_layers, fc_layers):
            h = gn(batch_g, fn(h))
            hs.append(h)
    batch_g.ndata['h'] = torch.cat(hs, dim=1)
    scoring_layer = LinearBlock(192, 2, norm_type=None, dropout=0.0, act_cls=Swish)
    # Extract percentiles
    n_feats = 5
    features = np.zeros([n_nodes, n_feats])
    ps = np.linspace(0, 100, n_feats)
    scores=scoring_layer(batch_g.ndata['h']).detach().numpy()
    for u in tqdm.tqdm(range(len(scores)), desc='ExtractPercentiles'):
        # 生成百分位的数字，ps 是 [0,25,50,75,100]，用这几个表示为节点的特征，聚成四类作为 label
        np.percentile(scores[u], ps, out=features[u])
    # Kmeans
    kmeans = KMeans(n_roles)
    labels = kmeans.fit_predict(features)
    return labels, features


def get_comm_feature(nodes: List[int], neighbors: Dict[int, Set[int]],
                     labels: np.ndarray, n_labels: int) -> np.ndarray:
    """
    Compute the distribution of edge types as the community feature."
    """
    # 4类组成的矩阵，社区内节点对的类别数量分布情况
    count_mat = np.zeros([n_labels, n_labels])
    nodes = set(nodes)
    for u in nodes:
        for v in (neighbors[u] & nodes):
            if v > u:
                continue
            i, j = labels[u], labels[v]
            i, j = (i, j) if i < j else (j, i)
            count_mat[i, j] += 1
    # triu_indices_from返回上三角矩阵的index(row, col), 将index返回矩阵，矩阵返回对应index的值，返回的是array
    arr = count_mat[np.triu_indices_from(count_mat)]
    n = arr.sum()
    # 返回数组的副本，其中第k个对角线上方的元素为零。对角线上方为零元素。 k = 0（默认值）是主对角线，k<0在其下方，k>0在其上方。
    assert np.tril(count_mat, k=-1).sum() == 0
    arr /= n + 1e-9
    return arr


def get_patterns(comms: List[List[int]], neighbors: Dict[int, Set[int]],
                 node_labels: np.ndarray, n_patterns: int) -> (np.ndarray, List[List[int]], np.ndarray):
    """
    Compute community features and use k-means to get patterns' features, size distributions, and supports.
    """
    n_labels = node_labels.max() + 1
    comm_features = [get_comm_feature(nodes, neighbors, node_labels, n_labels) for nodes in comms]
    k_means = KMeans(n_patterns)
    comm_labels = k_means.fit_predict(comm_features)
    pattern_features = k_means.cluster_centers_
    pattern_sizes = [[] for _ in range(n_patterns)]
    pattern_support = np.zeros(n_patterns)
    for i, label in enumerate(comm_labels):
        pattern_support[label] += 1
        pattern_sizes[label].append(len(comms[i]))
    return pattern_features, pattern_sizes, pattern_support


def compute_node_pattern_score(pattern_features: np.ndarray,
                               adj_mat: sp.spmatrix,
                               neighbors: Dict[int, Set[int]],
                               node_labels: np.ndarray) -> np.ndarray:
    """
    Scoring nodes based on local structures.
    """

    
    n_labels = node_labels.max() + 1
    n_nodes = adj_mat.shape[0]
    count_mat = np.zeros([n_labels, n_labels])
    # Local structure
    # // 取整除 - 返回商的整数部分（向下取整）
    node_local_features = np.zeros((n_nodes, n_labels * (n_labels + 1) // 2))
    for u in tqdm.tqdm(range(n_nodes), desc='NodeLocalFeature'):
        count_mat.fill(0)
        for v in neighbors[u]:
            i, j = node_labels[u], node_labels[v]
            i, j = (i, j) if i < j else (j, i)
            count_mat[i, j] += 1
        arr = count_mat[np.triu_indices_from(count_mat)]
        arr /= arr.sum()
        node_local_features[u] = arr
    # First Order: Pass 1 in the paper
    # @ 矩阵-向量乘法
    node_first_order_scores = node_local_features @ pattern_features.T
    # Second Order: Pass 2 in the paper
    deg_vec = np.array(adj_mat.sum(1)).squeeze()
    # diags 提取对角线或构造对角线数组。
    node_second_order_scores = sp.diags((adj_mat @ deg_vec) ** -1) @ adj_mat @ (
            deg_vec[:, None] * node_first_order_scores)
    node_pattern_scores = node_first_order_scores + node_second_order_scores
    return node_pattern_scores


def get_seed(target_size: int, degree_seeds: Dict[int, List[int]],
             used_seeds: Set[int], eps: int = 5) -> Union[int, None]:
    """
    Find the best seed that has never be picked before.
    """
    for deg in range(target_size - 1, target_size + eps):
        sorted_seeds = degree_seeds.get(deg, [])
        if len(sorted_seeds) == 0:
            continue
        while len(sorted_seeds):
            seed = sorted_seeds.pop()
            if seed not in used_seeds:
                used_seeds.add(seed)
                return seed
    else:
        return None
