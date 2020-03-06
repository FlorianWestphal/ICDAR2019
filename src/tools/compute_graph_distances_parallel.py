import multiprocessing
import os
import numpy as np
import networkx as nx
import gmatch4py as gm


MAX = 100


def extract_img_path(path):
    f = os.path.basename(path)
    d = os.path.basename(os.path.dirname(path))
    p = os.path.join(d, f)
    return p.split('.')[0]


# load graphs
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


def encode(x, y):
    return x*MAX + y


def fill_diagonal(s_x, e_x, s_y, e_y, result):
    for i in range(s_y, e_y):
        for j in range(i, e_x):
            result[j-s_y, i-s_y] = encode(j, i)


def fill(s_x, e_x, s_y, e_y, result):
    for i in range(s_x, e_x):
        for j in range(s_y, e_y):
            result[i-s_x, j-s_y] = encode(i, j)


def compute_distances(data):
    graphs, s_x, e_x, s_y, e_y, diagonal, tv, te, ed, alpha, beta = data
    ged = gm.HED(node_del=tv, node_ins=tv, edge_del=te, edge_ins=te,
                 alpha=alpha, beta=beta, edge_distance=ed)
    if diagonal:
        result = ged.compare_diagonal(graphs, s_x, e_x, s_y, e_y)
    else:
        result = ged.compare_block(graphs, s_x, e_x, s_y, e_y)
    return result


def create_work(graphs, number, tv, te, ed, alpha, beta):
    block_size = number // 6
    fill = number - block_size*6
    work = []
    params = [tv, te, ed, alpha, beta]
    for cols in range(6, 0, -1):
        first = True
        # offset is same as row number
        offset = 6-cols
        for col in range(cols):
            entry = []
            entry.append(graphs)
            entry.append(offset*block_size+col*block_size)  # s_x
            entry.append(offset*block_size+col*block_size+block_size)
            entry.append(offset*block_size)                 # s_y
            entry.append(offset*block_size+block_size)      # e_y
            if first:
                entry.append(True)
                first = False
            else:
                entry.append(False)
            entry += params
            work.append(entry)
        # add extra work for last worker
        work[-1][2] += fill
    work[-1][4] += fill
    return work


def assemble(results, number):
    result = np.full((number, number), -1.0)
    block_size = number // 6
    idx = 0
    for cols in range(6, 0, -1):
        offset = 6-cols
        for col in range(cols):
            x = offset*block_size+col*block_size
            y = offset*block_size
            r = results[idx]
            result[x:x+r.shape[0], y:y+r.shape[1]] = results[idx]
            idx += 1
    return result


def mirror(result):
    x, y = result.shape
    for i in range(y):
        for j in range(i, x):
            result[i, j] = result[j, i]


def main(graph_path, out_file, tv=1, te=1, ed=0.5, alpha=0.5, beta=0.5):
    graphs, labels, paths = load_graphs(graph_path)
    number = len(labels)

    # compute distance matrix
    work = create_work(graphs, number, tv, te, ed, alpha, beta)
    pool = multiprocessing.Pool(processes=20)
    results = pool.map(compute_distances, work)
    result = assemble(results, number)
    mirror(result)

    # write distances
    np.savez(out_file, distances=result, labels=labels, paths=paths)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                    description='Compute pairwise HED for all given graphs in '
                    'parallel using 20 processes.',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('graph_path', help='Path to folder structure '
                        'containing graphs.')
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
