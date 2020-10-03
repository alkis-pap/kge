# import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from numba import guvectorize, int32, uint8, jit
from numba.typed import List
import scipy.sparse
import math
import subprocess
from multiprocessing import Pool, shared_memory
from functools import partial

import code


# def int_dtype_for(max_value, module=np):
#     dtypes = [module.int8, module.int16, module.int32, module.int64]
#     for dtype in dtypes:
#         if module.iinfo(dtype).max >= max_value:
#             return dtype
#     raise ValueError('no int is enough')


def sum_bits(csr_mat):
    counts = np.array([bin(i).count('1') for i in range(1 << np.iinfo(csr_mat.dtype).bits)], dtype=np.uint8)
    return np.sum(counts[csr_mat.data])


def edge_list(csr_mat, n_relations):
    # map each possible value to a list of relations, eg. 0b10010 -> [2, 4]
    rel_map = List([List([i for i, b in enumerate(bin(d)[-1:1:-1]) if b == '1']) for d in range(1, 1 << n_relations)])
    return edge_list_(csr_mat.data, csr_mat.indices, csr_mat.indptr, sum_bits(csr_mat), rel_map)
    

@jit(nopython=True)
def edge_list_(data, indices, indptr, n_edges, rel_map):
    head = np.empty(n_edges, dtype=np.int32)
    tail = np.empty(n_edges, dtype=np.int32)
    relation = np.empty(n_edges, dtype=np.uint8)

    # map each possible value to a list of relations, eg. 0b10010 -> [2, 4]
    
    cnt = 0
    for row, (begin, end) in enumerate(zip(indptr[:-1], indptr[1:])):
        for i in range(begin, end):
            col = indices[i]
            d = data[i]
            for rel in rel_map[d - 1]:
                head[cnt] = row
                tail[cnt] = col
                relation[cnt] = rel
                cnt += 1
    
    return head, tail, relation


@jit(nopython=True)
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




class KGraph(Dataset):

    def __init__(self, head, tail, relation, n_entities, n_relations):
        self.head = head
        self.tail = tail
        self.relation = relation
        self.n_entities = n_entities
        self.n_relations = n_relations


    def __repr__(self):
        return "n_entities: {}, n_relations: {}, n_edges: {}".format(self.n_entities, self.n_relations, len(self.head))


    def random_split(self, sizes):
        sizes = np.round(len(self) * np.array(sizes) / sum(sizes)).astype(np.int32)
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


    def tocoo(self):
        # data = np.ndarray((len(self), -(-self.n_relations // 8))
        # for data in 
        data = np.left_shift(1, self.relation, dtype=np.uint8)
        return scipy.sparse.coo_matrix((data, (self.head, self.tail)), shape=(self.n_entities, self.n_entities))


    @classmethod
    def from_csv(cls, path, columns=[0, 1, 2], sep='\t', dtypes=[np.int32, np.int32, np.uint8], n_entities=None, n_relations=None):
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

    def remove_unlinked(self):
        print('creating sparse matrices...')
        coo = self.tocoo()
        csr = coo.tocsr()
        csc = coo.tocsc()
        del coo

        print('removing unlinked entities')
        linked_entities, = np.where((np.diff(csr.indptr) != 0) | (np.diff(csc.indptr) != 0))
        n_linked = len(linked_entities)

        entity_map = np.ndarray(n_entities, dtype=dtypes[0])
        entity_map[linked_entities] = np.arange(n_linked)

        self.head = entity_map[self.head]
        self.tail = entity_map[self.tail]

        print('removed', n_entities - n_linked, 'unlinked entities.')
        self.n_entities = n_linked

    @classmethod
    def from_csr(cls, csr_mat, n_relations=None):
        if n_relations is None:
            n_relations = int(math.ceil(math.log(csr_mat.max(), 2)))
        n_entities = csr_mat.shape[0]
        head, tail, relation = edge_list(csr_mat, n_relations)
        return cls(head, tail, relation, n_entities, n_relations)


    @classmethod
    def load(cls, path):
        loaded = np.load(path)
        csr = scipy.sparse.csr_matrix((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
        return cls.from_csr(csr, loaded.get('n_relations', None))


    def save(self, path):
        csr = self.tocoo().tocsr()
        np.savez_compressed(path, indices=csr.indices, indptr=csr.indptr, data=csr.data, shape=csr.shape, n_relations=self.n_relations)
        # scipy.sparse.save_npz(path, self.tocoo().tocsr())


    def __len__(self):
        return len(self.head)


    def __getitem__(self, idx):
        return self.head[idx], self.tail[idx], self.relation[idx]