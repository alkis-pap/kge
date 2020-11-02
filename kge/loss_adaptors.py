import torch
import torch.nn as nn
# pylint: disable=W0223
import numpy as np


class Regularized(nn.Module):
    
    def __init__(self, criterion, graph, device, l_ent, l_rel, p=2):
        super().__init__()
        self.e_counts = torch.from_numpy(np.diff(graph.children.indptr) + np.diff(graph.parents.indptr)).to(device)
        _, self.r_counts = np.unique(graph.relation, return_counts=True)
        self.r_counts = torch.from_numpy(self.r_counts).to(device)
        self.criterion = criterion
        self.l_ent = l_ent
        self.l_rel = l_rel
        self.p = p
    

    def forward(self, scores, triples, embeddings):
        h, t, r = triples[:, :, 0]
        e_h, e_t, e_r = (e[:, 0, :] for e in embeddings)
        loss = self.criterion(scores)
        loss += torch.sum(self.l_ent * torch.norm(e_h, dim=-1, p=self.p) / self.e_counts[h])
        loss += torch.sum(self.l_ent * torch.norm(e_t, dim=-1, p=self.p) / self.e_counts[t])
        loss += torch.sum(self.l_rel * torch.norm(e_r, dim=-1, p=self.p) / self.r_counts[r])
        return loss