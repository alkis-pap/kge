# Multi-relational graph embeddings with pytorch

## Intro

This software trains multi-relational graph embedding models on a single machine (cpu or gpu) using pytorch.
It is designed to use minimal amounts of cpu and gpu memory. The graph is stored in main memory while the model resides in device memory.

## Models

- TransE
- Rescal
- DistMult
- ComplEx

## Loss functions

- Margin-based ranking loss
- Logistic loss

## Regularization methods

- L1/L2 regularization (loss function terms)
- embedding normalization

## Optimization methods

Any* optimizer from `torch.optim` that works with sparse gradients including:

- SGD
- Adagrad
- SparseAdam

*: Pytorch currently does not support weight decay for sparse gradients.
