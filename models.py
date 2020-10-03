import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class LookupEncoder(nn.Module):
    def __init__(self, n_entities, entity_dim, n_relations, relation_dim):
        super().__init__()
        # self.entity_embedding = nn.Embedding(n_entities, entity_dim, sparse=True, max_norm=1)
        self.entity_embedding = nn.Embedding(n_entities, entity_dim, sparse=True)
        self.relation_embedding = nn.Embedding(n_relations, relation_dim, sparse=True)

    def forward(self, batch):
        h, t, r = batch
        return (self.entity_embedding(h), self.entity_embedding(t), self.relation_embedding(r))


class ScoringFunction(nn.Module):
    def __init__(self, scoring_fn):
        super().__init__()
        self.scoring_fn = scoring_fn

    def forward(self, batch):
        return self.scoring_fn(*batch)


class EmbeddingModelBase(nn.Sequential):
    def __init__(self, graph, entity_dim, relation_dim):
        super().__init__(
            LookupEncoder(graph.n_entities, entity_dim, graph.n_relations, relation_dim),
            ScoringFunction(self.score)
        )


class TransE(EmbeddingModelBase):
    def __init__(self, graph, embedding_dim, norm="L2"):
        super().__init__(graph, embedding_dim, embedding_dim)
        if norm == "L1":
            self.p = 1
        else:
            self.p = 2

    def score(self, h, t, r):
        with torch.no_grad():
            F.normalize(h, 2, out=h)
            F.normalize(t, 2, out=t)
        return torch.norm(h - t + r, p=self.p, dim=len(h.shape) - 1)
