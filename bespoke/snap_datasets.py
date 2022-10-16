import collections
import pathlib
import dgl
import numpy as np
import torch as th
from scipy import sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split


def load_facebook_or_twitter_or_youtube_network(root, name):
    prefix = f'{name}-1.90'
    with open(root / f'{name}/{prefix}.ungraph.txt') as fh:
        edges = fh.read().strip().split('\n')
        edges = np.array([[int(i) for i in x.split()] for x in edges])
    with open(root / f'{name}/{prefix}.cmty.txt') as fh:
        comms = fh.read().strip().split('\n')
        comms = [[int(i) for i in x.split()] for x in comms]
    # if (root / f'{name}/{prefix}.features.npy').exists():
    #     x = np.load(root / f'{name}/{prefix}.features.npy')
    #     x = th.Tensor(x)
    # else:
    if (root / f'{name}/{prefix}.nodefeat.txt').exists():
        with open(root / f'{name}/{prefix}.nodefeat.txt') as fh:
            nodefeats = [x.split() for x in fh.read().strip().split('\n')]
            nodefeats = {int(k): [int(i) for i in v] for k, *v in nodefeats}
        ind = np.array([[i, j] for i, js in nodefeats.items() for j in js])
        sp_feats = sp.csr_matrix((np.ones(len(ind)), (ind[:, 0], ind[:, 1])))
        # convolved_feats = self.conv(sp_feats)
        # svd = TruncatedSVD(64, 'arpack')
        # x = svd.fit_transform(sp_feats)
        x = th.Tensor(sp_feats.toarray())
        # x = (x - x.mean(0, keepdims=True)) / x.std(0, keepdims=True)
        # np.save(root / f'{name}/{prefix}.features.npy', x)
    else:
        x = None
    nodes = {i for x in edges for i in x}
    mapping = {u: u for i, u in enumerate(range(max(nodes) + 1))}
    print(len(comms))
    return edges, comms, mapping, x

def load_dblp_or_amazon_network(root, name):
    edges = open(root / f'{name}/com-{name}.ungraph.txt').readlines()
    edges = [[int(i) for i in e.split()] for e in edges[4:]]
    edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]
    nodes = {i for x in edges for i in x}
    mapping = {u: i for i, u in enumerate(sorted(nodes))}
    edges = np.asarray([[mapping[u], mapping[v]] for u, v in edges])
    comms = open(root / f'{name}/com-{name}.top5000.cmty.txt').readlines()
    comms = [[mapping[int(i)] for i in x.split()] for x in comms]
    return edges, comms, mapping


def load_email_network(root):
    edges = open(root / 'email/email-Eu-core.txt').read().strip()
    edges = [[int(i) for i in e.split(' ')] for e in edges.split('\n')]
    edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]
    nodes = {i for x in edges for i in x}
    mapping = {u: i for i, u in enumerate(sorted(nodes))}
    edges = np.asarray([[mapping[u], mapping[v]] for u, v in edges])
    comm_membership = open(root / 'email/email-Eu-core-department-labels.txt').read().strip()
    comms = collections.defaultdict(list)
    for line in comm_membership.split('\n'):
        u, i = line.split(' ')
        mapped_u = mapping.get(int(u), None)
        if mapped_u is not None:
            comms[i].append(mapped_u)
    comms = list(comms.values())
    return edges, comms, mapping


def load_snap_dataset(name, root='datasets'):
    root = pathlib.Path(root)
    has_feats = True
    if name == 'dblp':
        edges, comms, mapping = load_dblp_or_amazon_network(root, name)
        has_feats = False
    elif name == 'amazon':
        edges, comms, mapping = load_dblp_or_amazon_network(root, name)
        has_feats = False
    elif name == 'amazonp':
        edges, comms, mapping, nodefeats = load_facebook_or_twitter_or_youtube_network(root, name)
        has_feats = False
    elif name == 'dblpp':
        edges, comms, mapping, nodefeats = load_facebook_or_twitter_or_youtube_network(root, name)
        has_feats = False
    elif name == 'twitter':
        edges, comms, mapping, nodefeats = load_facebook_or_twitter_or_youtube_network(root, name)
    elif name == 'facebook':
        edges, comms, mapping, nodefeats = load_facebook_or_twitter_or_youtube_network(root, name)
    elif name == 'youtube':
        edges, comms, mapping, nodefeats = load_facebook_or_twitter_or_youtube_network(root, name)
        has_feats = False
    elif name == 'email':
        edges, comms, mapping = load_email_network(root)
        has_feats = False
    else:
        raise NotImplementedError
    n_nodes = edges.max() + 1
    g_edges = [u for u, v in edges], [v for u, v in edges]
    g = dgl.graph(g_edges)
    g = dgl.to_bidirected(g)
    n_classes = 2
    labels = th.randint(0, n_classes, (n_nodes,))
    if has_feats == False:
        nodefeats = th.ones(n_nodes, 64)
    adj_mat = sp.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=[n_nodes, n_nodes])
    adj_mat += adj_mat.T

    train_nid, test_nid = train_test_split(np.arange(n_nodes), test_size=0.3, random_state=42)
    train_nid, val_nid = train_test_split(train_nid, test_size=0.5, random_state=42)

    train_nid = th.Tensor(train_nid).int()
    test_nid = th.Tensor(test_nid).int()
    val_nid = th.Tensor(val_nid).int()

    return adj_mat, comms, mapping, g, nodefeats, labels, train_nid, test_nid, val_nid, n_classes
