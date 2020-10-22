import torch
import torch.nn as nn
import torch.nn.functional as F



def no_op(*args, **kwargs):
    pass


class LookupEncoder(nn.Module):

    def __init__(self, n_entities, entity_dim, n_relations, relation_dim, initializer, normalizer, max_norm):
        super().__init__()
        kwargs = {}
        if max_norm:
            kwargs['max_norm'] = max_norm
        self.entity_embedding = nn.Embedding(n_entities, entity_dim, sparse=True, **kwargs)
        self.relation_embedding = nn.Embedding(n_relations, relation_dim, sparse=True, **kwargs)
        with torch.no_grad():
            initializer(self.entity_embedding.weight, self.relation_embedding.weight)
        self.normalizer = normalizer


    def forward(self, batch):
        h, t, r = batch
        with torch.no_grad():
            self.normalizer(
                self.entity_embedding.weight[h], 
                self.entity_embedding.weight[t], 
                self.relation_embedding.weight[r]
            )
            F.normalize(self.entity_embedding.weight[h], 2, out=self.entity_embedding.weight[h])
            F.normalize(self.entity_embedding.weight[t], 2, out=self.entity_embedding.weight[t])
        return (self.entity_embedding(h), self.entity_embedding(t), self.relation_embedding(r))



class ScoringFunction(nn.Module):

    def __init__(self, scoring_fn):
        super().__init__()
        self.scoring_fn = scoring_fn

    def forward(self, batch):
        return self.scoring_fn(*batch)



class EmbeddingModelBase(nn.Module):

    def __init__(self, graph, entity_dim, relation_dim, max_norm=None):
        super().__init__()

        if not hasattr(self, 'initialize'):
            self.initialize = no_op

        if not hasattr(self, 'normalize'):
            self.normalize = no_op

        self.encode = LookupEncoder(
            graph.n_entities, 
            entity_dim, 
            graph.n_relations, 
            relation_dim, 
            self.initialize, 
            self.normalize,
            max_norm
        )
        self.decode = ScoringFunction(self.score)

    def forward(self, batch):
        return self.decode(self.encode(batch))