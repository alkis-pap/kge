import torch
import torch.nn as nn
from itertools import repeat


class PointwiseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def examples(self, n_negatives):
        return range(n_negatives)
        

class PairwiseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def examples(self, n_negatives):
        return zip(repeat(0), range(1, n_negatives))
        


class MarginBasedLoss(PairwiseLoss):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, scores):
        size = scores.size()
        return torch.sum(torch.clamp(scores[:,0].unsqueeze(1).expand(size[0], size[1] - 1) + self.gamma - scores[:,1:], min=0))
