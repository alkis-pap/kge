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
        # Entities: xavier_uniform
        bound = math.sqrt(6 / self.embedding_dim)
        torch.nn.init.uniform_(entity_embeddings, a=-bound, b=bound)

        # Relations: xavier_uniform + normalize
        torch.nn.init.uniform_(relation_embeddings, a=-bound, b=bound)
        F.normalize(relation_embeddings, p=2, out=relation_embeddings)
        

class DistMult(EmbeddingModelBase):

    def score(self, h, t, r):
        return torch.sum(h * t * r, dim=-1)

    def initialize(self, entity_embeddings, relation_embeddings):
        # Entities: xavier_uniform
        bound = math.sqrt(6 / self.embedding_dim)
        torch.nn.init.uniform_(entity_embeddings, a=-bound, b=bound)

        # Relations: xavier_normal + normalize
        std = math.sqrt(2 / self.embedding_dim)
        torch.nn.init.normal_(relation_embeddings, std=std)
        F.normalize(relation_embeddings, p=2, out=relation_embeddings)


class ComplEx(EmbeddingModelBase):

    @staticmethod
    def complex_matmul(A_real, A_imag, B_real, B_imag):
        real = torch.matmul(A_real, B_real) - torch.matmul(A_imag, B_imag)
        imag = torch.matmul(A_real, B_imag) + torch.matmul(A_imag, B_real)
        return (real, imag)

    def score(self, h, t, r):
        size =  self.embedding_dim // 2
        h_real, h_imag = (h[..., :size], h[..., size:])
        t_real, t_imag = (t[..., :size], t[..., size:])
        r_real, r_imag = (r[..., :size], r[..., size:])

        return sum(
            (hh * rr * tt).sum(dim=-1)
            for hh, rr, tt in [
                (h_real, r_real, t_real),
                (h_real, r_imag, t_imag),
                (h_imag, r_real, t_imag),
                (-h_imag, r_imag, t_real),
            ]
        )


    def initialize(self, entity_embeddings, relation_embeddings):
        # https://github.com/ttrouill/complex/blob/dc4eb93408d9a5288c986695b58488ac80b1cc17/efe/models.py#L481-L487
        # Both: Normal(0, 1)
        torch.nn.init.normal_(entity_embeddings, std=1)
        torch.nn.init.normal_(relation_embeddings, std=1)



class Rescal(EmbeddingModelBase):
    
    def relation_dim(self):
        return self.embedding_dim ** 2

    def score(self, h, t, r):
        # view as matrix
        R = r.view(*r.shape[:-1], self.embedding_dim, self.embedding_dim)

        product = torch.matmul(h.unsqueeze(-2), R)
        return torch.matmul(product, t.unsqueeze(-1)).squeeze()

    def initialize(self, entity_embeddings, relation_embeddings):
        # Both: Uniform(0, 1)
        torch.nn.init.uniform_(entity_embeddings)
        torch.nn.init.uniform_(relation_embeddings)