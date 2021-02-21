import torch
import torch.nn as nn
# pylint: disable=W0223

import numpy as np

from .utils import Module


class PairwiseHingeLoss(Module):
    margin: float = 1

    def forward(self, scores, *_args):
        negative_scores = scores[:, 1:]
        positive_scores = scores[:, :1].expand_as(negative_scores)

        return torch.sum(torch.clamp(negative_scores + self.margin - positive_scores, min=0))


class LogisticLoss(Module):
    def forward(self, scores, *_args):
        scores = torch.clamp(scores, min=-75, max=75)

        # loss of positive examples
        loss = torch.sum(torch.log(1 + torch.exp(-scores[:, :1])))
        # loss of negative examples
        loss += torch.sum(torch.log(1 + torch.exp(scores[:, 1:])))
        return loss


class NLLMulticlass(Module):
    def forward(self, scores, *_args):
        scores = torch.clamp(scores, min=-75, max=75)

        negative_scores = scores[:, 1:]
        
        tail_replaced = negative_scores[:, : negative_scores.size(1) // 2]
        head_replaced = negative_scores[:, negative_scores.size(1) // 2 :]

        positive_scores = scores[:, :1]

        exp_scores = torch.exp(positive_scores)
        
        loss = torch.sum(torch.log(exp_scores / (exp_scores + torch.sum(torch.exp(tail_replaced), dim=1))))

        loss += torch.sum(torch.log(exp_scores / (exp_scores + torch.sum(torch.exp(head_replaced), dim=1))))

        return -loss
        

class Regularized(Module):
    criterion: nn.Module
    l_ent: float = 0.01
    l_rel: float = None
    p: int = 2

    def init(self, graph, device=None):
        self.l_rel = self.l_rel or self.l_ent
        self.e_counts = torch.from_numpy(np.diff(graph.children.indptr) + np.diff(graph.parents.indptr)).to(device)
        _, self.r_counts = np.unique(graph.relation, return_counts=True)
        self.r_counts = torch.from_numpy(self.r_counts).to(device)
        return self

    def forward(self, scores, triples, embeddings):
        e_h, e_t, e_r = (e[:, 0, :] for e in embeddings)
        loss = self.criterion(scores)
        loss += self.l_ent * torch.sum(torch.norm(e_h, dim=-1, p=self.p) / self.e_counts[triples[0, :, 0]])
        loss += self.l_ent * torch.sum(torch.norm(e_t, dim=-1, p=self.p) / self.e_counts[triples[1, :, 0]])
        loss += self.l_rel * torch.sum(torch.norm(e_r, dim=-1, p=self.p) / self.r_counts[triples[2, :, 0]])
        return loss