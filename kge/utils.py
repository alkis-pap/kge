# import torch
# from torch.utils.data import Sampler
import numpy as np

def int_dtype_for(max_value):
    types = [np.int8, np.int16, np.int32, np.int64]
    for t in types:
        if np.iinfo(t).max >= max_value:
            return t
    raise ValueError('no int is enough')

# class RandomSamplerLight(Sampler):

#     def __init__(self, n_examples, generator=None):
#         self.n_examples = n_examples
#         self.generator = generator

#     @property
#     def num_samples(self):
#         return self.n_examples

#     def __iter__(self):
#         return iter(torch.randperm(self.n_examples, generator=self.generator))

#     def __len__(self):
#         return self.num_samples
