import torch
import torch.nn as nn
# pylint: disable=W0223


class PairwiseHingeLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, scores, *_args):
        negative_scores = scores[:, 1:]
        positive_scores = scores[:, :1].expand_as(negative_scores)

        return torch.sum(torch.clamp(positive_scores + self.margin - negative_scores, min=0))


class LogisticLoss(nn.Module):
    def forward(self, scores, *_args):
        # loss of positive examples
        loss = torch.sum(torch.log(1 + torch.exp(-scores[:, 0])))
        # loss of negative examples
        loss += torch.sum(torch.log(1 + torch.exp(scores[:, 1:])))
        return loss
