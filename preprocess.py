from graph import KGraph
import numpy as np
import scipy.sparse
import sys

import graph2

print("loading graph")
path = sys.argv[1]

def get_arrays(path):
    if path.endswith('csv'):
        graph = KGraph.from_csv(path, int(sys.argv[2]), int(sys.argv[3]))
    else:
        graph = KGraph.load(path)
    return (graph.head.numpy(), graph.tail.numpy(), graph.relation.numpy())

def get_duplicates(head, tail):
    head_diff = np.diff(head)
    tail_diff = np.diff(tail)

    return np.concatenate(([False], (head_diff == 0) & (tail_diff == 0)))


# def runs_of_ones_array(bits):
#   # make sure all runs of ones are well-bounded
#   bounded = np.hstack(([0], bits, [0]))
#   # get 1 at run starts and -1 at run ends
#   difs = np.diff(bounded)
#   run_starts, = np.where(difs > 0)
#   run_ends, = np.where(difs < 0)
#   return run_ends - run_starts

def get_matrix(path):

    head, tail, relation = get_arrays(path)

    head -= 1
    tail -= 1
    relation -= 1

    n_entities = max(head.max(), tail.max()) + 1

    relation_shifted = np.left_shift(np.uint8(1), np.uint8(relation))

    duplicate = get_duplicates(head, tail)

    indices = np.where(duplicate)

    duplicate = np.roll(duplicate, -1)

    for i in range(8):
        relation_shifted |= np.roll(relation_shifted, -i) * duplicate

    head = np.delete(head, indices)
    tail = np.delete(tail, indices)
    relation_shifted = np.delete(relation_shifted, indices)

    # counts = np.array([bin(i).count('1') for i in range(256)])

    # n_rel = counts[relation]
    csr = scipy.sparse.csr_matrix((relation_shifted, (head, tail)), shape=(n_entities, n_entities), dtype=np.uint8)

    return csr

def get_graph(path):
    csr_mat = get_matrix(path)
    csc_mat = scipy.sparse.csc_matrix(csr_mat)

    active = np.where((np.diff(csr_mat.indptr) != 0) | (np.diff(csc_mat.indptr) != 0))[0]

    csr_mat = csr_mat[np.ix_(active, active)]
    # csc_mat = csc_mat[np.ix_(active, active)]

    return graph2.KGraph(csr_mat, 7)

graph = get_graph(path)

# graph.save(sys.argv[2])