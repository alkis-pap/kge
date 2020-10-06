import math

import torch
import torch.nn.functional as F

from model_base import EmbeddingModelBase



class TransE(EmbeddingModelBase):

    def __init__(self, graph, embedding_dim, norm="L2"):
        if norm == "L1":
            self.p = 1
        else:
            self.p = 2
        self.embedding_dim = embedding_dim
        super().__init__(graph, embedding_dim, embedding_dim)


    def score(self, h, t, r):
        return torch.norm(h - t + r, p=self.p, dim=len(h.shape) - 1)


    def initialize(self, entity_embeddings, relation_embeddings):
        x = 6 / math.sqrt(self.embedding_dim)
        torch.nn.init.uniform_(entity_embeddings, a=-x, b=x)
        torch.nn.init.uniform_(relation_embeddings, a=-x, b=x)
        F.normalize(relation_embeddings, p=2, out=relation_embeddings)
    

    def normalize(self, e_h, e_t, e_r):
        F.normalize(e_h, p=2, out=e_h)
        F.normalize(e_t, p=2, out=e_t)



def complex_matmul(A_real, A_imag, B_real, B_imag):
    real = torch.matmul(A_real, B_real) - torch.matmul(A_imag, B_imag)
    imag = torch.matmul(A_real, B_imag) + torch.matmul(A_imag, B_real)
    return (real, imag)



# returns h^T * R * t
def bilinear_score(h, t, R):
    product = torch.matmul(h.unsqueeze(-2), R)
    return torch.matmul(product, t.unsqueeze(-1)).squeeze()



class Rescal(EmbeddingModelBase):

    def __init__(self, graph, embedding_dim):
        self.embedding_dim = embedding_dim
        super().__init__(graph, embedding_dim, embedding_dim ** 2)
        # super().__init__(graph, embedding_dim, embedding_dim ** 2, max_norm=1)


    def score(self, h, t, r):
        # view as matrix
        R = r.view(*r.shape[:-1], self.embedding_dim, self.embedding_dim)
        return bilinear_score(h, t, R)


    def initialize(self, entity_embeddings, relation_embeddings):
        torch.nn.init.normal_(entity_embeddings, std=.001)
        torch.nn.init.normal_(relation_embeddings, std=.001)



class DistMult(EmbeddingModelBase):

    def __init__(self, graph, embedding_dim):
        self.embedding_dim = embedding_dim
        super().__init__(graph, embedding_dim, embedding_dim)


    def score(self, h, t, r):
        return bilinear_score(h, t, torch.diag_embed(r))


    def initialize(self, entity_embeddings, relation_embeddings):
        torch.nn.init.normal_(entity_embeddings, std=.001)
        torch.nn.init.normal_(relation_embeddings, std=.001)


import code
class ComplEx(EmbeddingModelBase):

    def __init__(self, graph, embedding_dim):
        self.embedding_dim = embedding_dim
        super().__init__(graph, embedding_dim * 2, embedding_dim * 2)


    def score(self, h, t, r):
        h_real, h_imag = torch.unbind(h.view(*h.shape[:-1], self.embedding_dim, 2), dim=-1)
        t_real, t_imag = torch.unbind(t.view(*t.shape[:-1], self.embedding_dim, 2), dim=-1)
        r_real, r_imag = torch.unbind(r.view(*r.shape[:-1], self.embedding_dim, 2), dim=-1)
        R_real, R_imag = (torch.diag_embed(r_real), torch.diag_embed(r_imag))

        prod_real, prod_imag = complex_matmul(
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