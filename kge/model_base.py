import torch
import torch.nn as nn
import torch.nn.functional as F
# pylint: disable=W0223

from .utils import Module


class EmbeddingModelBase(Module):
    
    embedding_dim: int = 50
    normalize_embeddings: bool = True
    max_norm: float = None
    
    def relation_dim(self):
        return self.embedding_dim

    def init(self, graph, device=None):
        self.entity_embedding = nn.Embedding(graph.n_entities, self.embedding_dim, sparse=False, max_norm=self.max_norm).to(device)
        self.relation_embedding = nn.Embedding(graph.n_relations, self.relation_dim(), sparse=False, max_norm=self.max_norm).to(device)
        with torch.no_grad():
            self.initialize(self.entity_embedding.weight, self.relation_embedding.weight)
        return self

    def initialize(self, entity_embeddings, relation_embeddings):
        torch.nn.init.normal_(entity_embedding)
        torch.nn.init.normal_(entity_embedding)

    def normalize(self):
        if self.normalize_embeddings:
            with torch.no_grad():
                F.normalize(self.entity_embedding.weight, p=2, out=self.entity_embedding.weight)

    def encode(self, triples):
        h, t, r = triples
        return self.entity_embedding(h), self.entity_embedding(t), self.relation_embedding(r)

    def forward(self, triples):
        return self.score(*self.encode(triples))