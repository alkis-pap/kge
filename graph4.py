# import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List
import scipy.sparse
import math
import subprocess
from multiprocessing import Pool, shared_memory
from functools import partial

import code


@njit
def unique_indices(head, tail, relation):
    count = 0
    last_triple = (-1, -1, -1)
    for triple in zip(head,  tail, relation):
        if triple != last_triple:
            count += 1
        last_triple = triple

    result = np.empty(count, np.uint64)

    count = 0
    last_triple = (-1, -1, -1)
    for i, triple in enumerate(zip(head,  tail, relation)):
        if triple != last_triple:
            result[count] = i
            count += 1
        last_triple = triple
    return result


# provides fast access to children
class KGrahpIndex:
    def __init__(self, indptr, indices, relation):
        self.indices = indices
        self.relation = relation
        self.indptr = indptr


    @classmethod
    def from_graph(cls, head, tail, relation, n_entities):
        indptr = np.zeros(n_entities + 1)
        idx = np.where(np.diff(head, prepend=[-1]) != 0)[0]
        counts = np.diff(idx, append=len(head))
        indptr[head[idx]] = counts
        indptr.cumsum(out=indptr)
        return cls(indptr, tail, relation)


    # returns the list of children of entity e in relation r
    def __call__(self, e, r):
        idx = slice(self.indptr[e], self.indptr[e + 1])
        return self.indices[idx][self.relation[idx] == r]


    # def __call__(self, entities, relations):
    #     return [self.children(e, r) for e, r in zip(entities, relations)]


class PositiveSampler(object):
    def __init__(self, graphs):
        self.phases = ['train', 'valid', 'test']

        self.children_ = [graphs[phase].index() for phase in self.phases if phase in graphs]
        self.parents_ = [graphs[phase].inverse_index() for phase in self.phases if phase in graphs]


    def neighbors_(self, graph_indices, entities, relations, phase='train'):
        phase_id = self.phases.index(phase) + 1
        graph_indices = graph_indices[:phase_id]
        return [set(item for item in index.children(e, r) for index in graph_indices) for e, r in zip(entities, relations)]


    def children(self, *args, **kwargs):
        return self.neighbors_(self.children_, *args, **kwargs)


    def parents(self, *args, **kwargs):
        return self.neighbors_(self.parents_, *args, **kwargs)



# edge list dataset
class EdgeList(Dataset):

    def __init__(self, head, tail, relation, n_entities, n_relations):
        self.head = head
        self.tail = tail
        self.relation = relation
        self.n_entities = n_entities
        self.n_relations = n_relations


    def __repr__(self):
        return "n_entities: {}, n_relations: {}, n_edges: {}".format(self.n_entities, self.n_relations, len(self.head))


    @classmethod
    def from_csv(cls, path, columns=[0, 1, 2], sep='\t', dtypes=[np.uint32, np.uint32, np.uint32], n_entities=None, n_relations=None):
        print("reading csv...")
        df = pd.read_csv(
            path,
            sep=sep,
            names=['head', 'tail', 'relation'],
            usecols=columns,
            dtype={'head': dtypes[0], 'tail': dtypes[1], 'relation': dtypes[2]},
            header=None,
            error_bad_lines=False,
            warn_bad_lines=True,
            low_memory=True
        )
        head = df['head'].to_numpy(dtype=dtypes[0], copy=True)
        tail = df['tail'].to_numpy(dtype=dtypes[1], copy=True)
        relation = df['relation'].to_numpy(dtype=dtypes[2], copy=True)
        del df

        if n_entities is None:
            n_entities = max(head.max(), tail.max()) + 1
        n_relations = relation.max() + 1

        print('sorting edges...')
        sorted_indices = np.lexsort((relation, tail, head))
        head = np.take(head, sorted_indices)
        tail = np.take(tail, sorted_indices)
        relation = np.take(relation, sorted_indices)
        del sorted_indices

        print('removing duplicates...')
        unique = unique_indices(head, tail, relation)
        n_duplicates = head.shape[0] - unique.shape[0]
        head = head[unique]
        tail = tail[unique]
        relation = relation[unique]
        print(n_duplicates, 'edges removed.')
        del unique

        return cls(head, tail, relation, n_entities, n_relations)


    @classmethod
    def from_index(cls, graph_index, n_entities, n_relations=None):
        if n_relations is None:
            n_relations = int(math.ceil(math.log(graph_index.relation.max(), 2)))
        n_entities = len(graph_index.indptr) - 1
        counts = np.diff(graph_index.indptr)
        head = np.repeat(np.arange(n_entities), counts)
        return cls(head, graph_index.tail, graph_index.relation, n_entities, n_relations)


    @classmethod
    def load(cls, path):
        loaded = np.load(path)
        graph_index = KGrahpIndex(loaded['indptr'], loaded['tail'], loaded['relation'])
        return cls.from_index(graph_index, loaded.get('n_relations', None))


    def save(self, path):
        graph_index = self.index()
        np.savez_compressed(path, tail=self.tail, indptr=graph_index.indptr, relation=self.relation, n_entities=self.n_entities, n_relations=self.n_relations)


    def remove_unlinked(self):
        print('creating sparse matrices...')
        coo = scipy.sparse.coo_matrix((np.ones_like(self.relation), (self.head, self.tail)), shape=(self.n_entities, self.n_entities))
        csr = coo.tocsr()
        csc = coo.tocsc()
        del coo

        print('removing unlinked entities')
        linked_entities, = np.where((np.diff(csr.indptr) != 0) | (np.diff(csc.indptr) != 0))
        n_linked = len(linked_entities)

        entity_map = np.ndarray(self.n_entities, dtype=self.head.dtype)
        entity_map[linked_entities] = np.arange(n_linked)

        self.head = entity_map[self.head]
        self.tail = entity_map[self.tail]

        print('removed', self.n_entities - n_linked, 'unlinked entities.')
        self.n_entities = n_linked


    def random_split(self, sizes):
        sizes = np.array(sizes)
        sizes = np.round(len(self) * sizes / np.sum(sizes)).astype(np.uint32)
        sizes[0] += len(self) - np.sum(sizes)
        indices = np.random.permutation(len(self))
        return (
            KGraph(
                self.head[indices[i * s : (i + 1) * s]], 
                self.tail[indices[i * s : (i + 1) * s]],
                self.relation[indices[i * s : (i + 1) * s]],
                self.n_entities, 
                self.n_relations
            )
            for i, s in enumerate(sizes)
        )


    def index(self):
        return KGrahpIndex.from_graph(self.head, self.tail, self.relation, self.n_entities)


    def inverse_index(self):
        idx = np.lexsort((self.tail, self.relation, self.head))
        return KGrahpIndex.from_graph(self.head[idx], self.tail[idx], self.relation[idx], self.n_entities)


    def __len__(self):
        return len(self.head)


    def __getitem__(self, idx):
        return self.head[idx], self.tail[idx], self.relation[idx]