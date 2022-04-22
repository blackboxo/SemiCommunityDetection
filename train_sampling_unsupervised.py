import os
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import dgl.function as fn
import time
import argparse

from model import SAGE
from negative_sampler import NegativeSampler
import dgl.dataloading


from bespoke import *

from utils import *

import tqdm
from sklearn.cluster import KMeans

def evaluate2(pred_comms, test_comms):
    scores_1, scores_2 = eval_comms_bidirectional(pred_comms, test_comms)
    mean_score_1 = scores_1.mean(0)
    print_results(mean_score_1, 'AvgOverAxis0')
    mean_score_2 = scores_2.mean(0)
    print_results(mean_score_2, 'AvgOverAxis1')
    mean_score_all = (mean_score_1 + mean_score_2) / 2.
    print_results(mean_score_all, 'AvgGlobal')
    return mean_score_1, mean_score_2, mean_score_all

class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = th.cat([pos_score, neg_score])
        label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss

def evaluate(model, g, nfeat, device, args):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        # single gpu
        if isinstance(model, SAGE):
            pred = model.inference(g, nfeat, device, args.batch_size, args.num_workers)
        # multi gpu
        else:
            pred = model.module.inference(g, nfeat, device, args.batch_size, args.num_workers)
    model.train()
    return pred

#### Entry point
def run(proc_id, n_gpus, args, devices, data):
    # Unpack data
    device = th.device(devices[proc_id])
    train_nid, val_nid, test_nid, n_classes, g, nfeat, labels = data

    in_feats = nfeat.shape[1]

    # Create PyTorch DataLoader for constructing blocks
    n_edges = g.num_edges()
    train_seeds = th.arange(n_edges)

    # Create sampler
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, exclude='reverse_id',
        # For each edge with ID e in Reddit dataset, the reverse edge is e ± |E|/2.
        reverse_eids=th.cat([
            th.arange(n_edges // 2, n_edges),
            th.arange(0, n_edges // 2)]).to(train_seeds),
        negative_sampler=NegativeSampler(g, args.num_negs, args.neg_share,
                                         device if args.graph_device == 'uva' else None))
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_seeds, sampler,
        device=device,
        use_ddp=n_gpus > 1,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_uva=args.graph_device == 'uva')

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, args.num_hidden, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_pos = []
    iter_neg = []
    iter_d = []
    iter_t = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
            batch_inputs = nfeat[input_nodes].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            blocks = [block.int().to(device) for block in blocks]
            d_step = time.time()

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, pos_graph, neg_graph)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t = time.time()
            pos_edges = pos_graph.num_edges()
            neg_edges = neg_graph.num_edges()
            iter_pos.append(pos_edges / (t - tic_step))
            iter_neg.append(neg_edges / (t - tic_step))
            iter_d.append(d_step - tic_step)
            iter_t.append(t - d_step)
            if step % args.log_every == 0 and proc_id == 0:
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print(
                    '[{}]Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.4f}|{:.4f} | Load {:.4f}| train {:.4f} | GPU {:.1f} MB'.format(
                        proc_id, epoch, step, loss.item(), np.mean(iter_pos[3:]), np.mean(iter_neg[3:]),
                        np.mean(iter_d[3:]), np.mean(iter_t[3:]), gpu_mem_alloc))
            tic_step = time.time()

        toc = time.time()
        if proc_id == 0:
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
            if epoch >= 5:
                avg += toc - tic
            if (epoch + 1) % args.eval_every == 0:
                labels = evaluate(model, g, nfeat, device, args)
                return labels


def main(args):
    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)

    edges = open('com-dblp.ungraph.txt').readlines()
    edges = [[int(i) for i in e.split()] for e in edges[4:]]
    edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]
    nodes = {i for x in edges for i in x}
    mapping = {u: i for i, u in enumerate(sorted(nodes))}
    edges = np.asarray([[mapping[u], mapping[v]] for u, v in edges])
    edges = [u for u,v in edges],[v for u,v in edges]
    g = dgl.graph(edges)
    n_nodes = np.max(edges) + 1
    n_classes = 5
    g.ndata['labels'] = th.ones((n_nodes, 120))
    g.ndata['features'] = th.ones((n_nodes, 50))
    g.ndata['train_mask'] = th.cat((th.ones(int(n_nodes / 20)), th.zeros(n_nodes - int(n_nodes / 20))), 0)
    g.ndata['val_mask'] = th.cat(
        (th.zeros(int(n_nodes / 20)), th.ones(int(n_nodes / 20)), th.zeros(n_nodes - 2 * int(n_nodes / 20))), 0)
    train_nid = int(n_nodes / 20)
    val_nid = int(n_nodes / 20)
    test_nid = 2 * int(n_nodes / 20)

    nfeat = g.ndata.pop('features')
    labels = g.ndata.pop('labels')
    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves memory and CPU.
    g.create_formats_()

    # this to avoid competition overhead on machines with many cores.
    # Change it to a proper number on your machine, especially for multi-GPU training.
    os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count() // 2 // n_gpus)

    # Pack data
    data = train_nid, val_nid, test_nid, n_classes, g, nfeat, labels

    scores=run(0, 0, args, ['cpu'], data)
    n_feats = 5
    features = np.zeros([n_nodes, n_feats])
    ps = np.linspace(0, 100, n_feats)
    for u in tqdm.tqdm(range(n_nodes), desc='ExtractPercentiles'):
        # 生成百分位的数字，ps 是 [0,25,50,75,100]，用这几个表示为节点的特征，聚成四类作为 label
        np.percentile(scores[u], ps, out=features[u])
    # Kmeans
    kmeans = KMeans(4)
    labels = kmeans.fit_predict(features)

    adj_mat, comms, *_ = load_snap_dataset(args.dataset, args.root)
    # Split comms
    train_comms, test_comms = split_comms(comms, args.train_size, args.seed, args.max_size)
    print(f'[{args.dataset}] # Nodes: {adj_mat.shape[0]}'
          f' # TrainComms: {len(train_comms)} # TestComms: {len(test_comms)}',
          flush=True)
    # Fit
    model = Bespoke(args.n_roles, args.n_patterns, args.eps, unique=True)
    pattern_features=model.fit(adj_mat, train_comms,labels)
    pred_comms = model.sample_batch(args.pred_size,labels, pattern_features)
    # Evaluating
    print(f'-> (All)  # Comms: {len(pred_comms)}')
    evaluate2(pred_comms, test_comms)
    # Save
    if len(args.save_dst) > 0:
        write_comms_to_file(pred_comms, args.save_dst)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument("--gpu", type=str, default='-1',
                           help="GPU, can be a list of gpus for multi-gpu training,"
                                " e.g., 0,1,2,3; -1 for CPU")
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--num-negs', type=int, default=1)
    argparser.add_argument('--neg-share', default=False, action='store_true',
                           help="sharing neg nodes for positive nodes")
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=10000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--graph-device', choices=('cpu', 'gpu', 'uva'), default='cpu',
                           help="Device to perform the sampling. "
                                "Must have 0 workers for 'gpu' and 'uva'")
    argparser.add_argument('--data-device', choices=('cpu', 'gpu', 'uva'), default='cpu',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "Use 'cpu' to keep the features on host memory and "
                                "'uva' to enable UnifiedTensor (GPU zero-copy access on "
                                "pinned host memory).")
    argparser.add_argument('--dataset', type=str, help='dataset name', default='dblp')
    argparser.add_argument('--root', type=str, help='data directory', default='datasets')
    argparser.add_argument('--seed', type=int, help='random seed', default=0)
    argparser.add_argument('--train_size', type=int, help='the number of training communities', default=500)
    argparser.add_argument('--max_size', type=int,
                        help='Communities whose size is larger than this value will be discarded.',
                        default=0)
    argparser.add_argument('--n_roles', type=int, help='the number of node labels', default=4)
    argparser.add_argument('--n_patterns', type=int, help='the number of community patterns', default=5)
    argparser.add_argument('--eps', type=int, help='maximum tolerance for seed selection', default=5)
    argparser.add_argument('--pred_size', type=int, help='the number of communities to extract', default=50000)
    argparser.add_argument('--save_dst', type=str, help='where to save the searched communities',
                        default='bespoke_comms.txt')
    args = argparser.parse_args()
    seed_all(args.seed)

    main(args)
