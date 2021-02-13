from numba import njit
import torch
import numpy as np

# from .graph import PositiveSampler


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

import code
# Evaluates a model using the entity ranking protocol.
# If the training graph is also provided, the filtered scores are returned
def evaluate(model, test_graph, device, train_graph=None, n_edges=None, batch_size=1000000, verbose=False):
    model.eval()
    with torch.no_grad():

        head_stats = RankingStats()
        tail_stats = RankingStats()

        # positive_sampler = PositiveSampler(graphs)
        
        # perm = np.random.permutation(len(test_graph))
        
        if n_edges is None:
            n_edges = len(test_graph)

        one_sided_eval(model, test_graph.parents, head_stats, True, test_graph.n_entities, device, batch_size, train_graph.parents)
        
        one_sided_eval(model, test_graph.children, tail_stats, False,  test_graph.n_entities, device, batch_size, train_graph.children)

        # for each test triple (h,r,t):
            # find all test triples (h', r, t)
            # filter training triples (optional)
            # rank
            # find all test triples (h, r, t')
            # filter training triples (optional)
            # rank

        # tail_pairs = set()
        # head_pairs = set()


        # for i in range(n_edges):
        #     h, t, r = test_graph[perm[i]]
        #     print("%d / %d, %d, %d" % (i, n_edges, len(head_pairs), len(tail_pairs)), end='\r')
        #     # print(h, t, r)

        #     if (t, r) not in tail_pairs:
        #         test_heads = test_graph.parents(t, r)
                
        #         train_heads = train_graph.parents(t, r) if train_graph is not None else []

        #         head_ranks = one_sided_rank(model, test_heads, train_heads, t, r, True, test_graph.n_entities, device, batch_size)
                
        #         for rank in head_ranks:
        #             head_stats.add_sample(rank)

        #         tail_pairs.add((t,r))
            

        #     if (h, r) not in head_pairs:

        #         test_tails = test_graph.children(h, r)

        #         if len(test_tails) == 0:
        #             code.interact(local=locals())
                
        #         train_tails = train_graph.children(h, r) if train_graph is not None else []

        #         tail_ranks = one_sided_rank(model, test_tails, train_tails, h, r, False, test_graph.n_entities, device, batch_size)
                
        #         for rank in head_ranks:
        #             tail_stats.add_sample(rank)

        #         head_pairs.add((h,r))

        if verbose:
            print("Head stats:", head_stats.get_stats())
            print("Tail stats:", tail_stats.get_stats())
            print("Overall stats:", head_stats.combine(tail_stats).get_stats())
        
        return head_stats.combine(tail_stats).get_stats(), head_stats.get_stats(), tail_stats.get_stats()


def one_sided_eval(model, test_index, stats, replace_head,  n_entities, device, batch_size, train_index=None):
    for e in range(len(test_index.indptr) - 1):
        
        start = test_index.indptr[e]
        end = test_index.indptr[e + 1]
        
        if end - start > 0:
            relation, idx = np.unique(test_index.relation[start : end], return_index=True)

            idx += start
            idx = [*idx, end]

            for i, (rel_start, rel_end) in enumerate(zip(idx[:-1], idx[1:])):

                r = relation[i]

                replaced = test_index.indices[rel_start : rel_end]
                
                excluded = train_index(e, r) if train_index is not None else []

                problems = set(replaced) & set(excluded)
                if len(problems) > 0:
                    code.interact(local=locals())

                ranks = one_sided_rank(model, replaced, excluded, e, r, replace_head, n_entities, device, batch_size)

                for rank in ranks:
                    stats.add_sample(rank)
    


def one_sided_rank(model, replaced_entities, excluded_entities, other_entity, r, replace_head, n_entities, device, batch_size):
    # print('replaced_entities', replaced_entities)
    # print('excluded_entities', excluded_entities)

    candidates = arange_excluding(n_entities, excluded_entities)

    # print('candidates', candidates)


    indices = [index_of(candidates, e) for e in replaced_entities]

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

    ranking = torch.argsort(scores, descending=True)
    return [torch.where(ranking == index)[0][0].item() + 1 for index in indices]



# def one_sided_rank(model, replaced_entities, other_entity, r, replace_head, positive_sampler, n_entities, device, batch_size):
#     observed = positive_sampler(other_entity, r, phase='test')

#     try:
#         candidates = arange_excluding(n_entities, observed)
#     except IndexError:
#         code.interact(local=locals())

#     index = index_of(candidates, replaced_entity)

#     if index < 0:
#         code.interact(local=locals())

#     scores = torch.zeros(len(candidates), device=device)

#     for batch_start in range(0, len(candidates), batch_size):
#         idx = slice(batch_start, batch_start + batch_size)
        
#         cand = candidates[idx]
        
#         candidate_tensor = torch.from_numpy(cand).to(device, dtype=torch.long)
#         other_tensor = torch.from_numpy(np.repeat(other_entity, len(cand))).to(device, dtype=torch.long)
#         rel_tensor = torch.from_numpy(np.repeat(r, len(cand))).to(device, dtype=torch.long)

#         if replace_head:
#             scores[idx] = model((
#                 candidate_tensor,
#                 other_tensor,
#                 rel_tensor
#             ))
#         else:
#             scores[idx] = model((
#                 other_tensor,
#                 candidate_tensor,
#                 rel_tensor
#             ))

#     return torch.where(torch.argsort(scores) == index)[0][0].item() + 1
