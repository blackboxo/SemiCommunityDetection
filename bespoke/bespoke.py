from .core import *
import numpy
import tqdm
from sklearn.metrics.pairwise import euclidean_distances


class Bespoke:

    def __init__(self,  n_roles=4, n_patterns=5, eps=5, unique=True):
        self.n_roles = n_roles
        self.n_patterns = n_patterns
        self.eps = eps
        self.unique = unique
        self.adj_mat = None
        self.n_nodes = None
        self.node_neighbors = None
        self.pattern_sizes = None
        self.pattern_p = None
        self.used_seeds = None
        self.node_pattern_scores = None
        self.pattern_degree_seeds = None
        self.pattern_features = None
        self.scores = None

    def fit(self, adj_mat, train_comms, scores):
        self.adj_mat = adj_mat
        self.n_nodes = adj_mat.shape[0]
        # indices 稀疏矩阵非0元素对应的列索引值所组成数组
        self.node_neighbors = {u: set(adj_mat[u].indices) for u in range(self.n_nodes)}
        # Extract Patterns
        pattern_features, self.pattern_sizes, pattern_support = get_patterns(
            train_comms, scores, self.n_patterns)
        self.pattern_features = pattern_features
        self.scores = scores
        self.pattern_p = pattern_support / pattern_support.sum()
        self.node_pattern_scores = compute_node_pattern_score(pattern_features, self.adj_mat, self.node_neighbors, scores)
        self.reset_seeds()
        return pattern_features

    def reset_seeds(self):
        self.used_seeds = set()
        # numpy.squeeze（）函数可以删除数组形状中的单维度条目，即把shape中为1的维度去掉，但是对非单维的维度不起作用。
        node_degrees = numpy.array(self.adj_mat.sum(1)).squeeze().astype(int)
        degree_node_dict = collections.defaultdict(list)
        for i, d in enumerate(node_degrees):
            degree_node_dict[d].append(i)

        self.pattern_degree_seeds = [{d: sorted(nodes, key=lambda i: -x[i])
                                      for d, nodes in degree_node_dict.items()}
                                     for x in self.node_pattern_scores.T]
        # self.pattern_degree_seeds = [sorted(range(self.n_nodes), key=lambda i: -x[i])
        #                              for x in self.node_pattern_scores.T]

    def sample(self):
        n_try = 0
        while (n_try < 20) and (len(self.used_seeds) < self.n_nodes):
            n_try += 1
            pattern_id = np.random.choice(len(self.pattern_p), p=self.pattern_p)
            target_size = np.random.choice(self.pattern_sizes[pattern_id])
            seed = get_seed(target_size, self.pattern_degree_seeds[pattern_id],
                            self.used_seeds if self.unique else set())
            if seed is None:
                continue
            return [seed] + list(self.node_neighbors[seed]), pattern_id, target_size
        else:
            raise ValueError('(Almost) Run out of seeds!')

    def beam_sample(self, k = 10):
        emb = self.scores
        pattern_features = self.pattern_features
        n_try = 0
        while (n_try < 20):
            n_try += 1
            pattern_id = numpy.random.choice(len(self.pattern_p), p=self.pattern_p)
            target_size = np.random.choice(self.pattern_sizes[pattern_id])
            seed = get_seed(target_size, self.pattern_degree_seeds[pattern_id], self.used_seeds)
            if seed is None:
                continue
            
            community_nodes = set([seed] + list(self.node_neighbors[seed]))
            community_features = np.sum((emb[v] for v in community_nodes), axis=0, dtype=np.float64)
            community_features_sum = np.sum((emb[v] for v in community_nodes), axis=0, dtype=np.float64)
            community_scores = euclidean_distances(community_features.reshape(1,-1), pattern_features[pattern_id].reshape(1,-1))[0][0]

            # start from one-hop neighbor
            sequences = [[community_nodes, community_features, community_features_sum, community_scores]]
            all_sequences = [[community_nodes, community_features, community_features_sum, community_scores]]
            # print(sequences)
            for _ in range(target_size + 3 - len(community_nodes)):
                all_candidates = list()

                for i in range(len(sequences)):
                    seq, feature, feature_sum, score = sequences[i]
                    new_neighbor_nodes = list()

                    for u in seq:
                        new_neighbor_nodes += list(self.node_neighbors[u]) 

                    new_neighbor_nodes = set(new_neighbor_nodes) - set(seq)
                    if len(new_neighbor_nodes) > 100:
                        new_neighbor_nodes = np.random.choice(list(new_neighbor_nodes), size=100, replace=False)
                    for j in new_neighbor_nodes:
                        new_feature_sum = feature_sum + emb[j]
                        new_feature = new_feature_sum # / (len(seq) + 1)
                        new_score = euclidean_distances(new_feature.reshape(1,-1),pattern_features[pattern_id].reshape(1,-1))[0][0]
                        candidate = [seq | set([j]), new_feature, new_feature_sum, new_score]
                        all_candidates.append(candidate)
                ordered = sorted(all_candidates, key=lambda tup: tup[-1])  # 按score排序
                sequences = ordered[:k]  # 选择前k个最好的
                all_sequences += sequences

            ordered_all_sequences = sorted(all_sequences, key=lambda tup: tup[-1])  # 按score排序
            # print(ordered_all_sequences)
            # print("search:", list(ordered_all_sequences[0][0]))
            return list(ordered_all_sequences[0][0])
        else:
            raise ValueError('(Almost) Run out of seeds!')

    def sample_batch(self, n, pattern_features,scores, reset=False):
        if reset:
            self.reset_seeds()
        pred_comms = []
        try:
            for i in tqdm.tqdm(range(n), desc='PredComms'):
                pred_comms.append(self.sample()[0])
                #pred_comms.append(self.beam_sample())
        except ValueError as e:
            print('Warning!!!', e)
        return pred_comms
