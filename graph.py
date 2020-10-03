import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


def int_dtype_for(max_value, module=np):
    dtypes = [module.int8, module.int16, module.int32, module.int64]
    for dtype in dtypes:
        if module.iinfo(dtype).max >= max_value:
            return dtype
    raise ValueError('no int is enough')

class KGraph(Dataset):
    def __init__(self, head, tail, relation, n_entities, n_relations):
        self.head = head
        self.tail = tail
        self.relation = relation
        self.n_entities = n_entities
        self.n_relations = n_relations

        # indices = torch.argsort(self.tail)
        # self.head_sorted_by_tail = torch.take(head, indices)
        # self.relation_sorted_by_tail = torch.take(relation, indices)
        # self.tail_cum_degree = torch.cumsum(torch.bincount(tail), 0)
        # self.tail_cum_degree = torch.cat(self.tail_cum_degree, self.tail_cum_degree[-1])

    # def get_children(self, entities, relations):
    #     begin_arr = torch.searchsorted(self.head, entities)
    #     end_arr = torch.searchsorted(self.head, entities, right=True)
    #     result = []
    #     for i, (begin, end) in enumerate(zip(begin_arr, end_arr)):
    #         indices = torch.where(self.relation[begin : end] == relations[i])
    #         result.append(self.tail[indices[0]])
    #     return result

    # def get_parents(self, entities, relations):
    #     begin_arr = self.tail_cum_degree[entities]
    #     end_arr = self.tail_cum_degree[entities + 1]
    #     result = []
    #     for i, (begin, end) in enumerate(zip(begin_arr, end_arr)):
    #         indices = torch.where(self.relation_sorted_by_tail[begin : end] == relations[i])
    #         result.append(self.head_sorted_by_tail[indices[0]])
    #     return result

    @classmethod
    def from_csv(cls, path, n_entities, n_relations):
        def get_tensors():
            dataframe = pd.read_csv(
                path,
                sep = '\t',
                names=['head', 'tail', 'relation'],
                dtype={
                    'head': int_dtype_for(n_entities), 
                    'tail': int_dtype_for(n_entities), 
                    'relation': int_dtype_for(n_relations)
                },
                header=None
            )
            return (torch.as_tensor(dataframe[c].values) for c in dataframe.columns)

        return cls(*[*get_tensors(), n_entities, n_relations])

    @classmethod
    def load(cls, path):
        result = torch.load(path)
        print(len(result))
        return result

    def save(self, path):
        torch.save(self, path)

    def __len__(self):
        return len(self.head)

    def __getitem__(self, idx):
        # print(idx)
        # print(self.head[idx], self.tail[idx], self.relation[idx])
        return torch.stack((self.head[idx], self.tail[idx], self.relation[idx]))
        