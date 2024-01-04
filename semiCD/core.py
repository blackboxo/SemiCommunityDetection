import collections
from typing import List, Set, Dict, Union

import numpy as np
import tqdm
from scipy import sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

def get_comm_feature(nodes: List[int],scores: np.ndarray) -> np.ndarray:
    """
    Compute the distribution of edge types as the community feature."
    """
    # 4类组成的矩阵，社区内节点对的类别数量分布情况
    nodes=set(nodes)
    arr=np.sum([scores[u] for u in nodes], axis=0)
    return arr


def get_patterns(comms: List[List[int]],
                 node_labels: np.ndarray, n_patterns: int) -> (np.ndarray, List[List[int]], np.ndarray):
    """
    Compute community features and use k-means to get patterns' features, size distributions, and supports.
    """
    comm_features = [get_comm_feature(nodes, node_labels) for nodes in comms]
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
                               scores:np.ndarray) -> np.ndarray:
    """
    Scoring nodes based on local structures.
    """


    n_nodes = adj_mat.shape[0]
    # score of center node
    node_local_features = np.array(scores, dtype=np.float64)
    for u in tqdm.tqdm(range(n_nodes), desc='NodeLocalFeature'):
        neighbor = set(list(neighbors[u]) + [u])
        node_local_features[u] += np.sum([scores[v] for v in neighbor], axis = 0)
    # First Order: Pass 1 in the paper
    # @ 矩阵-向量乘法
    node_first_order_scores = euclidean_distances(node_local_features, pattern_features)
    # Second Order: Pass 2 in the paper
    deg_vec = np.array(adj_mat.sum(1)).squeeze()
    # diags 提取对角线或构造对角线数组。
    node_second_order_scores = sp.diags((adj_mat @ deg_vec) ** -1) @ adj_mat @ (
            deg_vec[:, None] * node_first_order_scores)
    node_pattern_scores = node_first_order_scores + node_second_order_scores

    return node_pattern_scores


def get_seed(target_size: int, degree_seeds: List[int],
             used_seeds: Set[int], eps: int = 5) -> Union[int, None]:
    """
    Find the best seed that has never be picked before.
    """
    
    # while len(degree_seeds):
    #     seed = degree_seeds.pop()
    #     if seed not in used_seeds:
    #         used_seeds.add(seed)
    #         return seed
    # else:
    #     return None
    
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
