import random

import numpy as np
from numba import njit

from .graph import PositiveSampler



# generates random negative triples without replacement and without checking if they exist in the graph
class UniformNegativeSamplerFast(object):

    def __init__(self, graphs, n_negatives):
        self.n_neg = int(n_negatives / 2)
        # self.chunk_size = chunk_size
        self.n_entities = graphs['train'].n_entities


    def __call__(self, batch, phase='train'):
        h, t, r = batch
        
        h_prime = np.random.randint(0, self.n_entities, (len(h), self.n_neg))
        t_prime = np.random.randint(0, self.n_entities, (len(h), self.n_neg))
        # h_prime = np.tile(np.random.randint(0, self.n_entities, (self.chunk_size, self.n_neg)), (-(-len(h) // self.chunk_size), 1))[:len(h)]
        # t_prime = np.tile(np.random.randint(0, self.n_entities, (self.chunk_size, self.n_neg)), (-(-len(h) // self.chunk_size), 1))[:len(h)]
        
        h = np.expand_dims(h, 1)
        t = np.expand_dims(t, 1)
        r = np.expand_dims(r, 1)

        h_stacked = np.repeat(h, self.n_neg, axis=1)
        t_stacked = np.repeat(t, self.n_neg, axis=1)
        r_stacked = np.repeat(r, 2 * self.n_neg + 1, axis=1)

        new_h = np.concatenate((h, h_stacked, h_prime), axis=1)
        new_t = np.concatenate((t, t_prime, t_stacked), axis=1)

        return new_h, new_t, r_stacked


@njit
def exclude(arr, excluded):
    j = 0
    for i in range(len(arr)):
        arr[i] += j
        while j < len(excluded) and excluded[j] <= arr[i]:
            j += 1
            arr[i] += 1

# generate a random sample of size k with replacement from the range [start, stop) excluding the elements of the 'exluded' sorted sequence 
def random_choice(start, stop, k, excluded=[]):
    result = np.sort(random.choices(range(start, stop - len(excluded)), k=k))
    if len(excluded) > 0:
        exclude(result, excluded)
    return result


class UniformNegativeSampler(object):

    def __init__(self, graphs, n_negatives):
        self.n_neg = int(n_negatives / 2)
        self.n_entities = graphs['train'].n_entities
        self.positive_sampler = PositiveSampler(graphs)

    def __call__(self, batch, phase='train'):
        head, tail, relation = batch

        children = [self.positive_sampler.children(e, r, phase) for e, r in zip(head, relation)]
        parents = [self.positive_sampler.parents(e, r, phase) for e, r in zip(tail, relation)]

        h_prime = np.array([random_choice(0, self.n_entities, self.n_neg, excluded) for excluded in parents])
        t_prime = np.array([random_choice(0, self.n_entities, self.n_neg, excluded) for excluded in children])

        head = np.expand_dims(head, 1)
        tail = np.expand_dims(tail, 1)
        relation = np.expand_dims(relation, 1)

        h_stacked = np.repeat(head, self.n_neg, axis=1)
        t_stacked = np.repeat(tail, self.n_neg, axis=1)
        r_stacked = np.repeat(relation, 2 * self.n_neg + 1, axis=1)
        
        new_h = np.concatenate((head, h_stacked, h_prime), axis=1)
        new_t = np.concatenate((tail, t_prime, t_stacked), axis=1)

        return new_h, new_t, r_stacked


