import math

import numpy as np
import pandas as pd
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



# edge list dataset
class KGraph:

    def __init__(self, head, tail, relation, n_entities, n_relations, index=None, inverse_index=None):
        self.head = head
        self.tail = tail
        self.relation = relation
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.children_ = index
        self.parents_ = inverse_index

    def __repr__(self):
        return "n_entities: {}, n_relations: {}, n_edges: {}".format(self.n_entities, self.n_relations, len(self.head))

    @classmethod
    def from_csv(cls, paths, columns=None, sep='\t', dtypes=None, n_entities=None, n_relations=None):
        if columns is None:
            columns = [0, 1, 2]
        if dtypes is None:
            dtypes = [np.int32, np.int32, np.int32]

        print("reading csv...")

        included_columns = np.zeros(np.max(columns) + 1, dtype=bool)
        included_columns[columns] = True

        return_list = True
        if not isinstance(paths, list):
            paths = [paths]
            return_list = False

        df = None
        sizes = []
        for path in paths:
            part = dt.fread(
                path,
                sep=sep,
                header=False,
                columns=included_columns.tolist()
            )
            
            sizes.append(part.nrows)

            if df is None:
                df = part
            else:
                df.rbind(part)

        columns = np.argsort(columns).tolist()

        heads = df[columns[0]].to_numpy().ravel()
        tails = df[columns[1]].to_numpy().ravel()

        headtail, _ = pd.factorize(np.concatenate((heads, tails)))

        heads = headtail[:df.nrows].astype(dtypes[0], copy=False)
        tails = headtail[df.nrows:].astype(dtypes[0], copy=False)
        
        relations = pd.factorize(df[columns[2]].to_numpy().ravel())[0].astype(dtypes[2], copy=False)

        del df

        if n_relations is None:
            n_relations = np.max(relations) + 1

        n_observed = max(np.max(heads), np.max(tails)) + 1
        if n_entities is None:
            n_entities = n_observed

        result = []

        offset = 0
        for size in sizes:
            head = heads[offset: offset + size]
            tail = tails[offset: offset + size]
            relation = relations[offset: offset + size]
            offset += size
            
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
            
            result.append(cls(head, tail, relation, n_entities, n_relations))
    
        if not return_list:
            return result[0]
        return result


    # @classmethod
    # def from_htr(cls, head, tail, relation, n_entities, n_relations):
        # index = cls.index(head, tail, relation, n_entities)
        # inverse_index = cls.inverse_index(head, tail, relation, n_entities)
        # return cls(head, tail, relation, n_entities, n_relations, index, inverse_index)


    @classmethod
    def from_index(cls, index, n_entities, n_relations=None):
        if n_relations is None:
            n_relations = int(math.ceil(math.log(index.relation.max(), 2)))
        n_entities = len(index.indptr) - 1
        counts = np.diff(index.indptr)
        head = np.repeat(np.arange(n_entities), counts)
        del counts
        tail = index.indices
        # inverse_index = cls.inverse_index(head, tail, index.relation, n_entities)
        # return cls(head, tail, index.relation, n_entities, n_relations, index, inverse_index)
        return cls(head, tail, index.relation, n_entities, n_relations)


    @classmethod
    def from_dgl(cls, dgl_graph):
        n_edges = dgl_graph.num_edges()
        relations = dgl_graph.etypes
        rel_ids = {r : i for i, r in enumerate(relations)}

        head = np.empty(n_edges, dtype=np.int32)
        tail = np.empty_like(head)
        relation = np.empty_like(head)

        offset = 0

        for edge_type in dgl_graph.canonical_etypes:
            _, r, _ = edge_type
            h, t = dgl_graph.edges(etype=edge_type)
            
            head[offset : offset + h.size(0)] = h
            tail[offset : offset + h.size(0)] = t
            relation[offset : offset + h.size(0)].fill(rel_ids[r])
            
            offset += h.size(0)

        idx = np.lexsort((tail, relation, head))
        return cls(head[idx], tail[idx], relation[idx], dgl_graph.num_nodes(), len(relations))


    @classmethod
    def load(cls, path):
        loaded = np.load(path)
        graph_index = KGrahpIndex(loaded['indptr'], loaded['tail'], loaded['relation'])
        return cls.from_index(graph_index, loaded.get('n_entities', None), loaded.get('n_relations', None))


    def save(self, path):
        children = self.children
        np.savez_compressed(path, tail=self.tail, indptr=self.children.indptr, relation=self.relation, n_entities=self.n_entities, n_relations=self.n_relations)


    def to_csv(self, path, columns=None):
        if columns is None:
            columns = [0, 1, 2]
        df = dt.Frame(np.stack([self.head, self.tail, self.relation], axis=1)[:, columns])
        df.to_csv(path, header=False)


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


    def random_split(self, *sizes):
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


    def combine(self, other):
        if (self.n_entities, self.n_relations) != (other.n_entities, other.n_relations):
            raise RuntimeError('Cannot combine graphs with different n_entities or n_relations.')

        head = np.concatenate((self.head, other.head))
        tail = np.concatenate((self.tail, self.head))
        relation = np.concatenate((self.relation, self.relation + self.n_relations))

        idx = np.lexsort((tail, relation, head))

        return KGraph(head[idx], tail[idx], relation[idx], self.n_entities, self.n_relations)


    def with_inverse_triples(self):        
        return KGraph(
            np.concatenate((self.head, self.tail)),
            np.concatenate((self.tail, self.head)),
            np.concatenate((self.relation, self.relation + self.n_relations)),
            self.n_entities,
            2 * self.n_relations
        )


    @property
    def children(self):
        if not self.children_:
            self.children_ = KGrahpIndex.from_graph(self.head, self.tail, self.relation, self.n_entities)
        return self.children_


    @property
    def parents(self):
        if not self.parents_:
            idx = np.lexsort((self.head, self.relation, self.tail))
            self.parents_ = KGrahpIndex.from_graph(self.head[idx], self.tail[idx], self.relation[idx], self.n_entities)
        return self.parents_


    def __len__(self):
        return len(self.head)


    def __getitem__(self, idx):
        return self.head[idx], self.tail[idx], self.relation[idx]