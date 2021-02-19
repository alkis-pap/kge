from contextlib import suppress

from numba import njit
import torch
import numpy as np

from .utils import timeit


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
    def __init__(self, k_hits=None):
        if k_hits is None:
            k_hits = [10]
        self.rank_sum = 0
        self.rrank_sum = 0
        self.k_hits = k_hits
        self.hits = np.zeros(len(k_hits))
        self.n = 0

    def add_sample(self, rank):
        self.rank_sum += rank
        self.rrank_sum += 1 / rank
        for i, n in enumerate(self.k_hits):
            if rank < n:
                self.hits[i] += 1
        self.n += 1

    def combine(self, other):
        result = RankingStats(self.k_hits)
        result.rank_sum = self.rank_sum + other.rank_sum
        result.rrank_sum = self.rrank_sum + other.rrank_sum
        result.n = self.n + other.n
        for i in range(len(self.hits)):
            result.hits[i] = self.hits[i] + other.hits[i]
        return result

    def get_stats(self):
        return {
            'mr': self.rank_sum / self.n,
            'mrr': self.rrank_sum / self.n,
            **{f'hits@{n}': hits / self.n for hits, n in zip(self.hits, self.k_hits)}
        }


# Evaluates a model using the entity ranking protocol.
# If the training graph is also provided, the filtered scores are returned
def evaluate(model, test_graph, device, train_graph=None, n_edges=None, batch_size=1000000, verbose=False):
    with timeit('Evaluation') if verbose else suppress():
        model.eval()
        with torch.no_grad():

            head_stats = RankingStats()
            tail_stats = RankingStats()
            
            if n_edges is None:
                n_edges = len(test_graph)

            one_sided_rank(model, test_graph.parents, head_stats, True, test_graph.n_entities, device, batch_size, train_graph.parents)
            
            one_sided_rank(model, test_graph.children, tail_stats, False,  test_graph.n_entities, device, batch_size, train_graph.children)

            return {
                'both': head_stats.combine(tail_stats).get_stats(), 
                'head': head_stats.get_stats(), 
                'tail': tail_stats.get_stats()
            }


def one_sided_rank(model, test_index, stats, replace_head,  n_entities, device, batch_size, train_index=None):
    for entity in range(len(test_index.indptr) - 1):
        
        start = test_index.indptr[entity]
        end = test_index.indptr[entity + 1]
        
        if end - start > 0:
            relation, idx = np.unique(test_index.relation[start : end], return_index=True)

            idx += start
            idx = [*idx, end]

            for i, (rel_start, rel_end) in enumerate(zip(idx[:-1], idx[1:])):

                r = relation[i]

                replaced_entities = test_index.indices[rel_start : rel_end]
                
                excluded_entities = train_index(entity, r) if train_index is not None else []

                candidates = arange_excluding(n_entities, excluded_entities)

                indices = [index_of(candidates, replaced) for replaced in replaced_entities]

                scores = torch.zeros(len(candidates), device=device)

                for batch_start in range(0, len(candidates), batch_size):
                    idx = slice(batch_start, batch_start + batch_size)
                    
                    cand = candidates[idx]
                    
                    candidate_tensor = torch.from_numpy(cand).to(device, dtype=torch.long)
                    other_tensor = torch.from_numpy(np.repeat(entity, len(cand))).to(device, dtype=torch.long)
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

                ranking = torch.argsort(scores, descending=True)

                for index in indices:
                    rank = torch.where(ranking == index)[0][0].item() + 1
                    stats.add_sample(rank)
    