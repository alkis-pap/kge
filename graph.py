import math

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from numba import njit
import scipy.sparse

from utils import int_dtype_for

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

import code
# provides fast access to children
class KGrahpIndex(object):

    def __init__(self, indptr, indices, relation):
        self.indices = indices
        self.relation = relation
        self.indptr = indptr

    @classmethod
    def from_graph(cls, head, tail, relation, n_entities):
        indptr = np.zeros(n_entities + 1, dtype=int_dtype_for(len(head)))
        idx = np.where(np.diff(head, prepend=[-1]) != 0)[0]
        counts = np.diff(idx, append=len(head))
        # code.interact(local=locals())
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



class PositiveSampler(object):

    def __init__(self, graphs):
        self.graphs = graphs

    def get_graphs(self, phase):
        if phase == 'train':
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
class KGraph(Dataset):

    def __init__(self, head, tail, relation, n_entities, n_relations, index, inverse_index):
        self.head = head
        self.tail = tail
        self.relation = relation
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.children = index
        self.parents = inverse_index

    def __repr__(self):
        return "n_entities: {}, n_relations: {}, n_edges: {}".format(self.n_entities, self.n_relations, len(self.head))

    @classmethod
    def from_csv(cls, path, columns=[0, 1, 2], sep='\t', dtypes=[np.uint32, np.uint32, np.uint32], n_entities=None, n_relations=None):
        print("reading csv...")
        df = pd.read_csv(
            path,
            sep=sep,
            names=np.array(['head', 'tail', 'relation'])[np.argsort(columns)],
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

        # code.interact(local=locals())

        del df

        if n_entities is None:
            n_entities = max(head.max(), tail.max()) + 1
        n_relations = relation.max() + 1

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

        # code.interact(local=locals())

        return cls.from_htr(head, tail, relation, n_entities, n_relations)

    @classmethod
    def from_htr(cls, head, tail, relation, n_entities, n_relations):
        index = cls.index(head, tail, relation, n_entities)
        inverse_index = cls.inverse_index(head, tail, relation, n_entities)
        return cls(head, tail, relation, n_entities, n_relations, index, inverse_index)

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
        return cls.from_index(graph_index, loaded.get('n_relations', None))

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