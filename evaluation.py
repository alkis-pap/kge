from numba import njit
import torch

from graph5 import PositiveSampler

# @njit
# def arange(n, excluded=[]):
#     size = n - len(excluded)
#     result = np.empty(size)
#     j = 0
#     for i in range(size):
#         if i == excluded[j]:
#             j += 1
#         else result
import numpy as np


@njit
def index(array, item):
    for idx, val in enumerate(array):
        if val == item:
            return idx
    return -1


@njit
def candidates(n_entities, excluded):
    if len(excluded) == 0:
        return np.arange(n_entities, dtype=excluded.dtype)
    result = np.empty(n_entities - len(excluded), dtype=excluded.dtype)
    i = 0
    j = 0
    for k in range(n_entities):
        if j < len(excluded) and excluded[j] == k:
            j += 1
            continue
        result[i] = k
        i += 1
    return result



def entity_ranking(model, graphs, device, n_entities=None):
    
    positive_sampler = PositiveSampler(graphs)
    model.eval()
    if n_entities is None:
        n_entities = graphs['test'].n_entities
    print('evaluating on', n_entities, 'entities')
    with torch.no_grad():
        mrr = 0
        hits = 0
        n = 0

        print('replacing heads')

        heads = np.where(np.diff(graphs['test'].children.indptr) != 0)[0]

        indices = np.random.permutation(len(heads))

        for i in range(min(n_entities, len(heads))):
            h = heads[indices[i]]
            print(i, h, end='\r')
            # if (h > 1000):
            #     break
            t_obs, r_obs = positive_sampler.children(h, phase='test')
            rel, counts = np.unique(r_obs, return_counts=True)
            rel_bounds = np.zeros(len(rel) + 1, dtype=np.int32)
            rel_bounds[1:] = np.cumsum(counts)
            # code.interact(local=locals())
            for begin, end in zip(rel_bounds[:-1], rel_bounds[1:]):
                observed = t_obs[begin : end]
                cand = candidates(graphs['test'].n_entities, observed)
                
                head_tensor = torch.from_numpy(np.repeat(h, len(cand))).to(device, dtype=torch.long)
                t_prime = torch.from_numpy(cand).to(device, dtype=torch.long)

                scores = model((
                    head_tensor,
                    t_prime,
                    torch.from_numpy(np.repeat(r_obs[begin], len(cand))).to(device, dtype=torch.long)
                ))

                for t in graphs['test'].children(h, r_obs[begin]):
                    i = index(cand, t)
                    rank = torch.where(torch.argsort(scores) == i)[0].cpu()[0].item() + 1
                    mrr += 1 / rank
                    hits += (rank < 10) * 100
                    n += 1

        print('replacing tails')

        tails = np.where(np.diff(graphs['test'].parents.indptr) != 0)[0]

        indices = np.random.permutation(len(tails))

        for i in range(min(n_entities, len(heads))):
            t = tails[indices[i]]
            print(i, t, end='\r')
            # if (t > 1000):
            #     break
            h_obs, r_obs = positive_sampler.parents(t, phase='test')
            rel, counts = np.unique(r_obs, return_counts=True)
            rel_bounds = np.zeros(len(rel) + 1, dtype=np.int32)
            rel_bounds[1:] = np.cumsum(counts)
            for begin, end in zip(rel_bounds[:-1], rel_bounds[1:]):
                observed = h_obs[begin : end]
                cand = candidates(graphs['test'].n_entities, observed)
                
                head_tensor = torch.from_numpy(np.repeat(t, len(cand))).to(device, dtype=torch.long)
                h_prime = torch.from_numpy(cand).to(device, dtype=torch.long)

                scores = model((
                    head_tensor,
                    h_prime,
                    torch.from_numpy(np.repeat(r_obs[begin], len(cand))).to(device, dtype=torch.long)
                ))
                
                for h in graphs['test'].parents(t, r_obs[begin]):
                    i = index(cand, h)
                    rank = torch.where(torch.argsort(scores) == i)[0].cpu()[0].item() + 1
                    mrr += 1 / rank
                    hits += (rank < 10) * 100
                    n += 1

        mrr /= n
        hits /= n
    return mrr, hits
