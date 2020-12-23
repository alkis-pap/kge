import math

import numpy as np
# import pandas as pd
import datatable as dt
from numba import njit
import scipy.sparse

from .utils import int_dtype_for


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
        indptr = np.zeros(n_entities + 1, dtype=int_dtype_for(len(head)))
        idx = np.where(np.diff(head, prepend=[-1]) != 0)[0]
        counts = np.diff(idx, append=len(head))
        indptr[head[idx] + 1] = counts
        indptr.cumsum(out=indptr)
        return cls(indptr, tail, relation)

    # returns the list of children of entity e in relation r
    def __call__(self, e, r=None):
        idx = slice(self.indptr[e], self.indptr[e + 1])
        if r is None:
            return (self.indices[idx], self.relation[idx])
        else:
            return self.indices[idx][self.relation[idx] == r]



class PositiveSampler:

    def __init__(self, graphs):
        self.graphs = graphs

    def get_graphs(self, phase):
        if phase == 'train' or 'valid' not in self.graphs:
            return [self.graphs['train']]
        else:
            return [self.graphs['train'], self.graphs['valid']]

    def children(self, e, r=None, phase='train'):
        result = np.concatenate([g.children(e, r) for g in self.get_graphs(phase)], axis=-1)
        if r is None:
            return result[:, np.lexsort(result)]
        else:
            return np.sort(result)
        # result.sort(kind='mergesort')
        # return np.concatenate([g.children(e, r) for g in self.get_graphs(phase)])
        # if r is None:
        #     code.interact(local=locals())
        # return np.take(result, np.lexsort(result), axis=-1)
        # return result[:, np.lexsort(result)]

    def parents(self, e, r=None, phase='train'):
        result = np.concatenate([g.parents(e, r) for g in self.get_graphs(phase)], axis=-1)
        if r is None:
            return result[:, np.lexsort(result)]
        else:
            return np.sort(result)
        # result.sort(kind='mergesort')
        # return result
        # code.interact(local=locals())
        # return np.take(result, np.lexsort(result), axis=-1)
        # return result[:, np.lexsort(result)]


# edge list dataset
class KGraph:

    def __init__(self, head, tail, relation, n_entities, n_relations, index, inverse_index, min_entity=0, min_rel=0):
        self.head = head
        self.tail = tail
        self.relation = relation
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.children = index
        self.parents = inverse_index
        self.min_entity = min_entity
        self.min_rel = min_rel

    def __repr__(self):
        return "n_entities: {}, n_relations: {}, n_edges: {}".format(self.n_entities, self.n_relations, len(self.head))

    @classmethod
    def from_csv(cls, path, columns=None, sep='\t', dtypes=None, n_entities=None, n_relations=None, min_entity=None, min_rel=None):
        if columns is None:
            columns = [0, 1, 2]
        if dtypes is None:
            dtypes = [np.int32, np.int32, np.int32]

        print("reading csv...")

        included_columns = np.zeros(np.max(columns) + 1, dtype=bool)
        included_columns[columns] = True

        df = dt.fread(
            path,
            sep=sep,
            header=False,
            columns=included_columns.tolist()
        )

        columns = np.argsort(columns).tolist()

        head = df[columns[0]].to_numpy().ravel().astype(dtypes[0], copy=False)
        tail = df[columns[1]].to_numpy().ravel().astype(dtypes[1], copy=False)
        relation = df[columns[2]].to_numpy().ravel().astype(dtypes[2], copy=False)

        del df

        if min_entity is None:
            min_entity = min(np.min(head), np.min(tail))
        head = head - min_entity
        tail = tail - min_entity

        if min_rel is None:
            min_rel = np.min(relation)
        relation = relation - min_rel

        if n_relations is None:
            n_relations = np.max(relation) + 1

        n_observed = max(np.max(head), np.max(tail)) + 1
        if n_entities is None:
            n_entities = n_observed

        print('sorting edges...')
        sorted_indices = np.lexsort((tail, relation, head))
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
        
        return cls.from_htr(head, tail, relation, n_entities, n_relations, min_entity, min_rel)

    @classmethod
    def from_htr(cls, head, tail, relation, n_entities, n_relations, min_entity=0, min_rel=0):
        index = cls.index(head, tail, relation, n_entities)
        inverse_index = cls.inverse_index(head, tail, relation, n_entities)
        return cls(head, tail, relation, n_entities, n_relations, index, inverse_index, min_entity, min_rel)

    @classmethod
    def from_index(cls, index, n_entities, n_relations=None):
        if n_relations is None:
            n_relations = int(math.ceil(math.log(index.relation.max(), 2)))
        n_entities = len(index.indptr) - 1
        counts = np.diff(index.indptr)
        head = np.repeat(np.arange(n_entities), counts)
        del counts
        tail = index.indices
        inverse_index = cls.inverse_index(head, tail, index.relation, n_entities)
        return cls(head, tail, index.relation, n_entities, n_relations, index, inverse_index)

    @classmethod
    def load(cls, path):
        loaded = np.load(path)
        graph_index = KGrahpIndex(loaded['indptr'], loaded['tail'], loaded['relation'])
        return cls.from_index(graph_index, loaded.get('n_entities', None), loaded.get('n_relations', None))

    def save(self, path):
        np.savez_compressed(path, tail=self.tail, indptr=self.children.indptr, relation=self.relation, n_entities=self.n_entities, n_relations=self.n_relations)

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
            self.from_htr(
                self.head[indices[i * s : (i + 1) * s]], 
                self.tail[indices[i * s : (i + 1) * s]],
                self.relation[indices[i * s : (i + 1) * s]],
                self.n_entities, 
                self.n_relations
            )
            for i, s in enumerate(sizes)
        )

    @staticmethod
    def index(head, tail, relation, n_entities):
        return KGrahpIndex.from_graph(head, tail, relation, n_entities)


    @staticmethod
    def inverse_index(head, tail, relation, n_entities):
        idx = np.lexsort((head, relation, tail))
        return KGrahpIndex.from_graph(tail[idx], head[idx], relation[idx], n_entities)

    def __len__(self):
        return len(self.head)

    def __getitem__(self, idx):
        return self.head[idx], self.tail[idx], self.relation[idx]