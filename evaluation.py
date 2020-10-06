from numba import njit
import torch

from graph import PositiveSampler

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
        mrr_h = 0
        hits_h = 0
        n_h = 0

        print('replacing heads')

        heads = np.where(np.diff(graphs['test'].children.indptr) != 0)[0]

        indices = np.random.permutation(len(heads))

        for i in range(min(n_entities, len(heads))):
            h = heads[indices[i]]
            print(i, h, end='\r')

            tail, rel = graphs['test'].children(h)

            for t, r in zip(tail, rel):
                h_obs = positive_sampler.parents(t, r, phase='test')

                cand = candidates(graphs['test'].n_entities, h_obs)

                i = index(cand, h)
                
                tail_tensor = torch.from_numpy(np.repeat(t, len(cand))).to(device, dtype=torch.long)
                head_tensor = torch.from_numpy(cand).to(device, dtype=torch.long)
                rel_tensor = torch.from_numpy(np.repeat(r, len(cand))).to(device, dtype=torch.long)

                scores = model((
                    head_tensor,
                    tail_tensor,
                    rel_tensor
                ))

                rank = torch.where(torch.argsort(scores) == i)[0][0].item() + 1

                mrr_h += 1 / rank
                hits_h += (rank <= 10)
                n_h += 1


        mrr_t = 0
        hits_t = 0
        n_t = 0

        print('replacing tails')

        tails = np.where(np.diff(graphs['test'].children.indptr) != 0)[0]

        indices = np.random.permutation(len(tails))

        for i in range(min(n_entities, len(tails))):
            t = tails[indices[i]]
            print(i, t, end='\r')

            head, rel = graphs['test'].parents(t)

            for h, r in zip(head, rel):
                t_obs = positive_sampler.children(h, r, phase='test')
                cand = candidates(graphs['test'].n_entities, t_obs)

                i = index(cand, t)
                
                head_tensor = torch.from_numpy(np.repeat(h, len(cand))).to(device, dtype=torch.long)
                tail_tensor = torch.from_numpy(cand).to(device, dtype=torch.long)
                rel_tensor = torch.from_numpy(np.repeat(r, len(cand))).to(device, dtype=torch.long)

                scores = model((
                    head_tensor,
                    tail_tensor,
                    rel_tensor
                ))

                rank = torch.where(torch.argsort(scores) == i)[0][0].item() + 1

                mrr_t += 1 / rank
                hits_t += (rank <= 10)
                n_t += 1

        
    return {
        'mrr_h' : mrr_h / n_h,
        'hits_h' : 100 * hits_h / n_h,
        'mrr_t' : mrr_t / n_t,
        'hirs_t' : 100 * hits_t / n_t,
        'mrr' : (mrr_h + mrr_t) / (n_h + n_t),
        'hits' : 100 * (hits_h + hits_t) / (n_h + n_t)
    }
