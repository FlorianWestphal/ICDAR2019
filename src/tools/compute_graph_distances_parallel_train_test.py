import gmatch4py as gm
import os
import multiprocessing
import networkx as nx
import numpy as np


MAX = 100


def extract_img_path(path):
    f = os.path.basename(path)
    d = os.path.basename(os.path.dirname(path))
    p = os.path.join(d, f)
    return p.split('.')[0]


def load_graphs(base_path):
    folders = os.listdir(base_path)
    graphs = []
    labels = []
    paths = []
    for f in folders:
        label = int(f)
        path = os.path.join(base_path, f)
        gpickles = os.listdir(path)
        for pickle in gpickles:
            p = os.path.join(path, pickle)
            g = nx.read_gpickle(p)
            graphs.append(g)
            labels.append(label)
            paths.append(extract_img_path(p))
    return graphs, labels, paths


def compute_distances(data):
    train, test, tv, te, ed, alpha, beta = data
    ged = gm.HED(node_del=tv, node_ins=tv, edge_del=te, edge_ins=te,
                 alpha=alpha, beta=beta, edge_distance=ed)
    return ged.compare_test_train(test, train)


def distribute(ds):
    idxs = np.zeros(20).astype(int)
    for i in range(len(ds)):
        idx = i % 20
        idxs[idx] += 1
    sets = [[] for i in range(20)]
    last_idx = 0
    for i, idx in enumerate(idxs):
        sets[i] = ds[last_idx:last_idx+idx]
        last_idx += idx
    return sets


def create_work(train, test, tv, te, ed, alpha, beta):
    work = []
    params = [tv, te, ed, alpha, beta]
    distr_test = distribute(test)

    for t in distr_test:
        entry = []
        entry.append(train)
        entry.append(t)
        entry += params
        work.append(entry)

    return work


def select(graphs, paths, labels, selection):
    sel = [[], [], []]
    for s in selection:
        idx = paths.index(s)
        sel[0].append(graphs[idx])
        sel[1].append(labels[idx])
        sel[2].append(paths[idx])
    return sel


def split_graphs(graphs, paths, labels, split):
    split = np.load(split)
    train = select(graphs, paths, labels, split['train'])
    test = select(graphs, paths, labels, split['test'])
    return train, test


def main(train_graph_path, test_graph_path, out_file, tv=1, te=1, ed=0.5,
         alpha=0.5, beta=0.5):
    train_g, train_l, train_p = load_graphs(train_graph_path)
    test_g, test_l, test_p = load_graphs(test_graph_path)

    # compute distance matrix
    work = create_work(train_g, test_g, tv, te, ed, alpha, beta)
    pool = multiprocessing.Pool(processes=20)
    results = pool.map(compute_distances, work)
    result = np.concatenate(results)

    # write distances
    np.savez(out_file, distances=result, train_labels=train_l,
             train_paths=train_p, test_labels=test_l, test_paths=test_p)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                    description='Compute HED between all graphs in the train '
                    'set and all graphs in the test set.',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('train_graph_path', help='Path to folder structure '
                        'containing train graphs.')
    parser.add_argument('test_graph_path', help='Path to folder structure '
                        'containing test graphs.')
    parser.add_argument('out_file', help='Path to the output NPZ file.')
    parser.add_argument('--tv', help='Node cost.', type=int,
                        default=1)
    parser.add_argument('--te', help='Edge cost.', type=int,
                        default=1)
    parser.add_argument('--ed', help='Node distance to determine if an edge '
                        'substitution cost needs to be added.',
                        type=float, default=0.5)
    parser.add_argument('--alpha', help='Weight indicating importance of node '
                        'substitution cost over edge substitution cost when '
                        'computing overall substituion cost.',
                        type=float, default=0.5)
    parser.add_argument('--beta', help='Weight indicating importance of x '
                        'coordinate over y coordinate when computing node '
                        'substitution costs.', type=float,
                        default=0.5)

    args = vars(parser.parse_args())
    main(**args)
