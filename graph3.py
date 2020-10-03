# import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
# import dask.dataframe as dd
# import dask.array as da
import scipy.sparse
import subprocess
# import math
from numba import jit

import code


def int_dtype_for(max_value):
    dtypes = [np.int8, np.int16, np.int32, np.int64]
    for dtype in dtypes:
        if np.iinfo(dtype).max >= max_value:
            return dtype
    raise ValueError('no int is enough')

def line_count(filename):
    out = subprocess.Popen(['wc', '-l', filename],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT
                         ).communicate()[0]
    return int(out.partition(b' ')[0])


@jit
def unique_indices(head, tail, relation):
    count = 0
    last_triple = (-1, -1, -1)
    for triple in zip(head,  tail, relation):
        if triple != last_triple:
            count += 1
        last_triple = triple

    result = np.empty(count)

    count = 0
    last_triple = (-1, -1, -1)
    for i, triple in enumerate(zip(head,  tail, relation)):
        if triple != last_triple:
            result[count] = i
            count += 1
        last_triple = triple
    return result


class KGraph(Dataset):
    # def __init__(self, head, tail, relation):
    #     self.head = head
    #     self.tail = tail
    #     self.relation = relation

    def __init__(self, head, tail, relation, n_entities, n_relations):
        self.head = head
        self.tail = tail
        self.relation = relation
        self.n_entities = n_entities
        self.n_relations = n_relations
        print('graph loaded:', self.n_entities, self.n_relations)


    @classmethod
    def from_csv(cls, path, head_col=0, tail_col=1, rel_col=2, sep='\t', chunksize=1000000):
        # def get_arrays():
        
        n_edges = line_count(path)

        cols = ['h', 't', 'r']
        
        head = np.ndarray(n_edges)
        tail = np.ndarray(n_edges)
        relation = np.ndarray(n_edges)

        for i, chunk in enumerate(pd.read_csv(
            path,
            sep=sep,
            names=['head', 'tail', 'relation'],
            usecols=[head_col, tail_col, rel_col],
            dtype={'head': np.int32, 'tail': np.int32, 'relation': np.int16},
            header=None,
            chunksize=chunksize
        )):
            start = i * chunksize
            head[start : start + chunk.shape[0]] = chunk['head']
            tail[start : start + chunk.shape[0]] = chunk['tail']
            relation[start : start + chunk.shape[0]] = chunk['relation']

        unique = unique_indices(head, tail, relation)
        head = head[unique]
        tail = tail[unique]
        relation = relation[unique]

        return cls(head, tail, relation, max(head.max(), tail.max()) + 1, relation.max() + 1)
        
            
            # print('loading:', path)

            # cols = ['h', 't', 'r']

            # df = dd.read_csv(
            #     path,
            #     sep=sep,
            #     names=cols,
            #     usecols=[head_col, tail_col, rel_col],
            #     dtype={'h': np.int32, 't': np.int32, 'r': np.int8},
            #     header=None
            # )

            # print('data loaded:', df.columns)
            
            # df.drop_duplicates(inplace=True).compute()

            # print('duplicates dropped')

            # n_edges = df.shape[0].compute()

            # return (
            #     np.array(df['h']),
            #     np.array(df['t']),
            #     np.array(df['r'])
            # )

        # h, t, r = get_arrays() # keep arrays, discard dataframe
        
        # print("array lengths", len(h), len(t), len(r))

        # n_entities = max(h.max(), t.max()) + 1
        # n_relations = r.max() + 1

        # # r = r.astype(int_dtype_for(n_relations))
        # # h = h.astype(int_dtype_for(len(h)))
        # # t = t.astype(int_dtype_for(len(t)))
        
        # print('constructing matrix...')

        # return cls(h, t, r, n_entities, n_relations)
        # # return cls(scipy.sparse.coo_matrix((r, (h ,t)), shape=(n_entities, n_entities)), n_relations)


    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls(data['h'], data['t'], data['r'], data['n_entities'], data['n_relations'])
        # return cls(scipy.sparse.load(path))
        # return cls(data['head'], data['tail'], data['relation'])


    def save(self, path):
        np.savez_compressed(path, h=self.head, t=self.tail, r=self.relation, n_entities=self.n_entities, n_relations=self.n_relations)
        # scipy.sparse.save_npz(path, self.coo_mat)
        # np.savez_compressed(path, head=self.head, tail=self.tail, relation=self.relation)


    def remove_unlinked(self):
        csr = scipy.sparse.csr_matrix((self.relation, (self.head, self.tail)), shape=(self.n_entities, self.n_entities))
        csc = scipy.sparse.csc_matrix((self.relation, (self.head, self.tail)), shape=(self.n_entities, self.n_entities))

        linked_entities, = np.where((np.diff(csr.indptr) != 0) | (np.diff(csc.indptr) != 0))
        n_linked = len(linked_entities)

        entity_map = np.ndarray(self.n_entities)

        entity_map[linked_entities] = np.arange(n_linked)

        self.head = entity_map[self.head]
        self.tail = entity_map[self.tail]

        self.n_entities = n_linked
        

    def __len__(self):
        return self.coo_mat.nnz()

    def __getitem__(self, idx):
        return self.coo_mat.row[idx], self.coo_mat.col[idx], self.coo_mat.data[idx]