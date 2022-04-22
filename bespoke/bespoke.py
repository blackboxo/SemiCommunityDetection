from .core import *
import numpy


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

    def fit(self, adj_mat, train_comms,node_labels):
        self.adj_mat = adj_mat
        self.n_nodes = adj_mat.shape[0]
        # indices 稀疏矩阵非0元素对应的列索引值所组成数组
        self.node_neighbors = {u: set(adj_mat[u].indices) for u in range(self.n_nodes)}
        # Extract Patterns
        pattern_features, self.pattern_sizes, pattern_support = get_patterns(
            train_comms, self.node_neighbors, node_labels, self.n_patterns)
        self.pattern_p = pattern_support / pattern_support.sum()
        self.node_pattern_scores = compute_node_pattern_score(pattern_features, self.adj_mat, self.node_neighbors,
                                                              node_labels)
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

    def sample(self, node_labels, pattern_features):
        n_try = 0
        while (n_try < 20) and (len(self.used_seeds) < self.n_nodes):
            n_try += 1
            pattern_id = numpy.random.choice(len(self.pattern_p), p=self.pattern_p)
            target_size = numpy.random.choice(self.pattern_sizes[pattern_id])
            seed = get_seed(target_size, self.pattern_degree_seeds[pattern_id],
                            self.used_seeds if self.unique else set())
            if seed is None:
                continue
            a = numpy.zeros((200, 10000), dtype=numpy.int)   #存储一百多个二叉树的路径，最后选最佳路径
            b = numpy.zeros((200, 10000), dtype=numpy.int)
            n = numpy.zeros((200, 317080), dtype=numpy.int)
            m = numpy.zeros(200)
            d = numpy.zeros(200)
            a[:, 0] = seed
            n_labels = node_labels.max() + 1
            l = 0
            for i in self.node_neighbors[seed]:   #先把种子的所有一阶邻居选上
                l=l+1
                a[:, l] = i
                n[:, i] = 1   #已选节点标签为1
            for i in range(1,l):
                for j in self.node_neighbors[a[0,i]]:
                    if n[0, j] == 0:
                        b[:, int(m[0])] = j   #b为已选节点的邻居节点的集合
                        m[0] = m[0] + 1   #b中含有节点数
                        n[:, j] = 2   #已选节点的邻居节点标签为2
            i = 0
            p = 10 - l
            for l in range(11-p,11):   #每个团伙10个节点
                i=i+1
                print(l)
                for j in range(1, 2 ^ p):
                    if j - 2 ^ i > 0:   #比如第一轮只有2个选择，所有奇数路径的选择都是一样的，第二轮只有4个选择，1、5、9、……选择是一样的
                        m[j] = m[j - 2 ^ i]
                        n[j, :] = n[j - 2 ^ i, :]
                        continue
                    if j - 2 ^ (i - 1) > 0: continue   #比如第一轮1和3的选择一样，第二轮1和3的选择是一块排序选前二的，3的选择在1的选择的时候就选过了，这里就不用再选了
                    f1 = 10000   #f1表示和模式向量的距离最小值
                    f2 = 10000   #f2表示和模式向量的距离第二小值
                    g1 = 0   #g1表示f1对应的节点
                    g2 = 0   #g2表示f2对应的节点
                    for k in range(int(m[j])):
                        if n[j, b[j,k]] != 2: continue
                        a[j, l] = b[j,k]
                        e = get_comm_feature(list(set(a[j].tolist())), self.node_neighbors, node_labels, n_labels)
                        if numpy.sqrt(numpy.sum(numpy.square(e - pattern_features[pattern_id]))) < f2:
                            f2 = numpy.sqrt(numpy.sum(numpy.square(e - pattern_features[pattern_id])))
                            g2 = b[j,k]
                            if f1 > f2:
                                f1, f2 = f2, f1
                                g1, g2 = g2, g1
                    a[j, l] = g1
                    a[j + 2 ^ (i - 1), l] = g2
                    n[j, g1] = 1
                    n[j + 2 ^ (i - 1), g2] = 1
                    for k in self.node_neighbors[g1]:
                        if n[j, k] == 0:
                            b[j, int(m[j])] = k
                            m[j] = m[j] + 1
                            n[j, k] = 2
                    for k in self.node_neighbors[g2]:
                        if n[j + 2 ^ (i - 1), k] == 0:
                            b[j + 2 ^ (i - 1), int(m[j + 2 ^ (i - 1)])] = k
                            m[j + 2 ^ (i - 1)] = m[j + 2 ^ (i - 1)] + 1
                            n[j + 2 ^ (i - 1), k] = 2
                    if l == 10:   d[j] = f1
            o = numpy.argmin(d)
            return list(set(a[o].tolist()))
        else:
            raise ValueError('(Almost) Run out of seeds!')

    def sample_batch(self, n, node_labels, pattern_features, reset=False):
        if reset:
            self.reset_seeds()
        pred_comms = []
        try:
            for _ in range(n):
                pred_comms.append(self.sample(node_labels, pattern_features))
        except ValueError as e:
            print('Warning!!!', e)
        return pred_comms
