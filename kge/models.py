import math

import torch
import torch.nn.functional as F
# pylint: disable=W0223

from .model_base import EmbeddingModelBase


class TransE(EmbeddingModelBase):

    p: int = 2

    def score(self, h, t, r):
        return torch.norm(h - t + r, p=self.p, dim=len(h.shape) - 1)


    def initialize(self, entity_embeddings, relation_embeddings):
        x = 6 / math.sqrt(self.embedding_dim)
        torch.nn.init.uniform_(entity_embeddings, a=-x, b=x)
        torch.nn.init.uniform_(relation_embeddings, a=-x, b=x)
        F.normalize(relation_embeddings, p=2, out=relation_embeddings)
    


class Rescal(EmbeddingModelBase):
    
    def relation_dim(self):
        return self.embedding_dim ** 2

    def score(self, h, t, r):
        # view as matrix
        R = r.view(*r.shape[:-1], self.embedding_dim, self.embedding_dim)
        product = torch.matmul(h.unsqueeze(-2), R)
        return torch.matmul(product, t.unsqueeze(-1)).squeeze()

    def initialize(self, entity_embeddings, relation_embeddings):
        torch.nn.init.normal_(entity_embeddings, std=.001)
        torch.nn.init.normal_(relation_embeddings, std=.001)


class DistMult(EmbeddingModelBase):

    def score(self, h, t, r):
        return torch.sum(h * t * r, dim=-1)

    def initialize(self, entity_embeddings, relation_embeddings):
        torch.nn.init.normal_(entity_embeddings, std=.001)
        torch.nn.init.normal_(relation_embeddings, std=.001)


class ComplEx(EmbeddingModelBase):

    @staticmethod
    def complex_matmul(A_real, A_imag, B_real, B_imag):
        real = torch.matmul(A_real, B_real) - torch.matmul(A_imag, B_imag)
        imag = torch.matmul(A_real, B_imag) + torch.matmul(A_imag, B_real)
        return (real, imag)

    def score(self, h, t, r):
        h_real, h_imag = torch.unbind(h.view(*h.shape[:-1], self.embedding_dim // 2, 2), dim=-1)
        t_real, t_imag = torch.unbind(t.view(*t.shape[:-1], self.embedding_dim // 2, 2), dim=-1)
        r_real, r_imag = torch.unbind(r.view(*r.shape[:-1], self.embedding_dim // 2, 2), dim=-1)
        R_real, R_imag = (torch.diag_embed(r_real), torch.diag_embed(r_imag))
        prod_real, prod_imag = self.complex_matmul(
            h_real.unsqueeze(-2),
            h_imag.unsqueeze(-2),
            R_real,
            R_imag
        )
        result = torch.matmul(prod_real, t_real.unsqueeze(-1)) - torch.matmul(prod_imag, t_imag.unsqueeze(-1))
        return result.squeeze()

    def initialize(self, entity_embeddings, relation_embeddings):
        torch.nn.init.normal_(entity_embeddings, std=.001)
        torch.nn.init.normal_(relation_embeddings, std=.001)