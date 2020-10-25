import code

from numba import njit
import torch

from .graph import PositiveSampler

import numpy as np


@njit(debug=True)
def is_sorted(a):
    for i in range(a.size-1):
         if a[i+1] < a[i] :
               return False
    return True

@njit(debug=True)
def index(array, item):
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


def one_sided_ranking(replace_head, model, graphs, n_entities, batch_size, device):
    mr = 0
    mrr = 0
    hits = 0
    n = 0

    positive_sampler = PositiveSampler(graphs)

    if replace_head:
        neighbors = graphs['test'].children
        positive = positive_sampler.parents
    else:
        neighbors = graphs['test'].parents
        positive = positive_sampler.children

    replaced = np.where(np.diff(neighbors.indptr) != 0)[0]

    indices = np.random.permutation(len(replaced))

    for i in range(min(n_entities, len(replaced))):
        replaced_entity = replaced[indices[i]]
        print(i, replaced_entity, end='\r')

        other, rel = neighbors(replaced_entity)

        for other_entity, r in zip(other, rel):
            observed = positive(other_entity, r, phase='test')

            candidates = arange_excluding(graphs['test'].n_entities, observed)

            i = index(candidates, replaced_entity)

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

            rank = torch.where(torch.argsort(scores) == i)[0][0].item() + 1

            mr += rank
            mrr += 1 / rank
            hits += (rank <= 10)
            n += 1

    return mr, mrr, hits, n



def entity_ranking(model, graphs, device, n_entities=None, batch_size=100000):
    model.eval()
    if n_entities is None:
        n_entities = graphs['test'].n_entities
    print('evaluating on', n_entities, 'entities')
    with torch.no_grad():

        print('replacing heads')
        mr_h, mrr_h, hits_h, n_h = one_sided_ranking(True, model, graphs, n_entities, batch_size, device)

        print('replacing tails')
        mr_t, mrr_t, hits_t, n_t = one_sided_ranking(True, model, graphs, n_entities, batch_size, device)
        
    return {
        'mr_h' : mr_h / n_h,
        'mrr_h' : mrr_h / n_h,
        'hits_h' : 100 * hits_h / n_h,

        'mrr_t' : mr_t / n_t,
        'mrr_t' : mrr_t / n_t,
        'hirs_t' : 100 * hits_t / n_t,

        'mr' : (mr_h + mr_t) / (n_h + n_t),
        'mrr' : (mrr_h + mrr_t) / (n_h + n_t),
        'hits' : 100 * (hits_h + hits_t) / (n_h + n_t)
    }
