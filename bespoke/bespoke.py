from .core import *
import numpy
import tqdm

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

    def fit(self, adj_mat, train_comms, scores):
        self.adj_mat = adj_mat
        self.n_nodes = adj_mat.shape[0]
        # indices 稀疏矩阵非0元素对应的列索引值所组成数组
        self.node_neighbors = {u: set(adj_mat[u].indices) for u in range(self.n_nodes)}
        # Extract Patterns
        pattern_features, self.pattern_sizes, pattern_support = get_patterns(
            train_comms, scores, self.n_patterns)
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

    def sample(self, pattern_features, scores):
        n_try = 0
        while (n_try < 20) and (len(self.used_seeds) < self.n_nodes):
            n_try += 1
            pattern_id = numpy.random.choice(len(self.pattern_p), p=self.pattern_p)
            target_size = numpy.random.choice(self.pattern_sizes[pattern_id])
            seed = get_seed(target_size, self.pattern_degree_seeds[pattern_id],
                            self.used_seeds if self.unique else set())
            if seed is None:
                continue
            
            # return [seed] + list(self.node_neighbors[seed])
            
            community_nodes = set([seed] + list(self.node_neighbors[seed]))
            community_features = np.sum((scores[v] for v in community_nodes), dtype=np.float64)
            community_scores = community_features @ pattern_features[pattern_id].T

            # start from one-hop neighbor
            sequences = [[community_nodes, community_features, community_scores]]
            all_sequences = [[community_nodes, community_features, community_scores]]
            max_community_size = 10
            k = 6
            for _ in range(0, 5):
                all_candidates = list()

                for i in range(len(sequences)):
                    seq, feature, score = sequences[i]
                    new_neighbor_nodes = list()

                    for u in seq:
                        new_neighbor_nodes += list(self.node_neighbors[u]) 

                    new_neighbor_nodes = set(new_neighbor_nodes) - set(seq)
    
                    for j in new_neighbor_nodes:
                        new_feature = feature + scores[j]
                        candidate = [seq | set([j]), new_feature, new_feature @ pattern_features[pattern_id].T ]
                        all_candidates.append(candidate)
                ordered = sorted(all_candidates, key=lambda tup: tup[2], reverse=True)  # 按score排序
                sequences = ordered[:k]  # 选择前k个最好的
                all_sequences += sequences

            ordered_all_sequences = sorted(all_sequences, key=lambda tup: tup[2], reverse=True)  # 按score排序
            return list(ordered_all_sequences[0][0])
        else:
            raise ValueError('(Almost) Run out of seeds!')

    def sample_batch(self, n, pattern_features,scores, reset=False):
        if reset:
            self.reset_seeds()
        pred_comms = []
        try:
            for i in tqdm.tqdm(range(n), desc='PredComms'):
            # for i in range(n):
                pred_comms.append(self.sample(pattern_features,scores))
        except ValueError as e:
            print('Warning!!!', e)
        return pred_comms
