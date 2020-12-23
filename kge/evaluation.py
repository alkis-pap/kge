from numba import njit
import torch
import numpy as np

from .graph import PositiveSampler


@njit(debug=True)
def index_of(array, item):
    for idx, val in enumerate(array):
        if val == item:
            return idx
    return -1


@njit(debug=True)
def arange_excluding(n_entities, excluded):
    if len(excluded) == 0:
        return np.arange(n_entities, dtype=excluded.dtype)
    result = np.empty(n_entities - len(excluded), dtype=excluded.dtype)
    i = 0
    j = 0
    for k in range(n_entities):
        if j < len(excluded) and excluded[j] <= k:
            j += 1
            continue
        result[i] = k
        i += 1
    return result


class RankingStats():
    def __init__(self, n_hits=None):
        if n_hits is None:
            n_hits = [10]
        self.rank_sum = 0
        self.rrank_sum = 0
        self.n_hits = n_hits
        self.hits = np.zeros(len(n_hits))
        self.n = 0

    def add_sample(self, rank):
        self.rank_sum += rank
        self.rrank_sum += 1 / rank
        for i, n in enumerate(self.n_hits):
            if rank < n:
                self.hits[i] += 1
        self.n += 1

    def combine(self, other):
        result = RankingStats(self.n_hits)
        result.rank_sum = self.rank_sum + other.rank_sum
        result.rrank_sum = self.rrank_sum + other.rrank_sum
        result.n = self.n + other.n
        for i in range(len(self.hits)):
            result.hits[i] = self.hits[i] + other.hits[i]
        return result

    def get_stats(self):
        return {
            "mr": self.rank_sum / self.n,
            "mrr": self.rrank_sum / self.n,
            **{"hits@" + str(n): hits / self.n for hits, n in zip(self.hits, self.n_hits)}
        }


def entity_ranking(model, graphs, device, n_edges=2**31, batch_size=1000000):
    head_stats = RankingStats()
    tail_stats = RankingStats()

    positive_sampler = PositiveSampler(graphs)
    
    perm = np.random.permutation(len(graphs['test']))
    
    if n_edges is None:
        n_edges = len(perm)
    
    for i in range(n_edges):
        h, t, r = graphs['test'][i]
        print("%d / %d" % (i, n_edges), end='\r')

        h_rank = one_sided_rank(model, h, t, r, True, positive_sampler.parents, graphs['test'].n_entities, device, batch_size)
        head_stats.add_sample(h_rank)
        
        t_rank = one_sided_rank(model, t, h, r, False, positive_sampler.children, graphs['test'].n_entities, device, batch_size)
        tail_stats.add_sample(t_rank)

    print("Head stats:", head_stats.get_stats())
    print("Tail stats:", tail_stats.get_stats())
    print("Overall stats:", head_stats.combine(tail_stats).get_stats())

import code

def one_sided_rank(model, replaced_entity, other_entity, r, replace_head, positive_sampler, n_entities, device, batch_size):
    observed = positive_sampler(other_entity, r, phase='test')

    try:
        candidates = arange_excluding(n_entities, observed)
    except IndexError:
        code.interact(local=locals())

    index = index_of(candidates, replaced_entity)

    if index < 0:
        code.interact(local=locals())

    scores = torch.zeros(len(candidates), device=device)

    for batch_start in range(0, len(candidates), batch_size):
        idx = slice(batch_start, batch_start + batch_size)
        
        cand = candidates[idx]
        
        candidate_tensor = torch.from_numpy(cand).to(device, dtype=torch.long)
        other_tensor = torch.from_numpy(np.repeat(other_entity, len(cand))).to(device, dtype=torch.long)
        rel_tensor = torch.from_numpy(np.repeat(r, len(cand))).to(device, dtype=torch.long)

        if replace_head:
            scores[idx] = model((
                candidate_tensor,
                other_tensor,
                rel_tensor
            ))
        else:
            scores[idx] = model((
                other_tensor,
                candidate_tensor,
                rel_tensor
            ))

    return torch.where(torch.argsort(scores) == index)[0][0].item() + 1
