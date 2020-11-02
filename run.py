import argparse

import torch
import numpy as np

from kge.train import train, EpochLimit
from kge.evaluation import entity_ranking
from kge.graph import KGraph
import kge.models
import kge.loss_functions
from kge.loss_adaptors import Regularized
from kge.negative_samplers import UniformNegativeSamplerFast
from kge.utils import int_dtype_for, timeit, get_class, make_object


parser = argparse.ArgumentParser(description='Multi-relational graph embedding')

# data
parser.add_argument('--train', type=str, required=True)
parser.add_argument('--valid', type=str, default=None)
parser.add_argument('--test', type=str, default=None)
parser.add_argument('--columns', nargs=3, type=int, default=[0, 1, 2])
parser.add_argument('--entities', type=int, default=None)
parser.add_argument('--min_entity', type=int, default=0)
parser.add_argument('--relations', type=int, default=None)
parser.add_argument('--min_rel', type=int, default=0)
parser.add_argument('--out', type=str, default=None)

# model
parser.add_argument('--model', type=str, default='TransE')
parser.add_argument('--dim', type=int, default=50)
parser.add_argument('--loss', type=str, default='PairwiseHingeLoss(margin=0.5)')
parser.add_argument('--regularize', type=str, default=None, choices=['L1', 'L2'])
parser.add_argument('--reg_coeff', type=float, default=.01)

# training
parser.add_argument('--nepochs', type=int, default=100)
parser.add_argument('--optim', type=str, default='SGD')
parser.add_argument('--lr', type=float, default=.01)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--checkpoint_period', type=int, default=1)
parser.add_argument('--device', type=str, default=None)

# evaluation
parser.add_argument('--neval', type=int, default=None)

args = parser.parse_args()

if args.entities is None:
    dtypes = [np.long] * 2
else:
    dtypes = [int_dtype_for(args.entities)] * 2

if args.relations is None:
    dtypes.append(np.long)
else:
    dtypes.append(int_dtype_for(args.relations))


graphs = {}
for dataset in ['train', 'test', 'valid']:
    path = getattr(args, dataset, None)
    if path is not None:
        with timeit("loading graph: " + path):
            if path.endswith('npz'):
                graphs[dataset] = KGraph.load(path)
            else:
                graphs[dataset] = KGraph.from_csv(
                    path, 
                    args.columns, 
                    dtypes=dtypes, 
                    n_entities=args.entities, 
                    n_relations=args.relations, 
                    min_entity=args.min_entity, 
                    min_rel=args.min_rel)
        print(graphs[dataset])

if args.device is None:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(args.device)

model_cls = get_class(kge.models, args.model)
model = model_cls(graphs['train'], args.dim, p=2).to(device)

criterion = make_object(kge.loss_functions, args.loss)
if args.regularize is not None:
    criterion = Regularized(criterion, graphs['train'], device, args.reg_coeff, args.reg_coeff, p=1 if args.regularize == "L1" else 2)

optimizer_cls = get_class(torch.optim, args.optim)
optimizer = optimizer_cls(model.parameters(), lr=args.lr)

train(
    graphs,
    model=              model,
    criterion=          criterion,
    negative_sampler=   UniformNegativeSamplerFast(graphs, 2),
    optimizer=          optimizer,
    stop_condition=     EpochLimit(args.nepochs),
    device=             device,
    batch_size=         10000,
    # scheduler=          torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
    scheduler=          None,
    checkpoint_path=    args.checkpoint,
    checkpoint_period=  args.checkpoint_period
)

if args.out is not None:
    embeddings = model.entity_embedding.weight
    np.savez_compressed(
        args.out,
        entity_embeddings=model.entity_embedding.weight.numpy(),
        relation_embeddings=model.relation_embedding.weight.numpy()
    )

if 'test' in graphs:
    with timeit("Evaluation"):
        entity_ranking(model, graphs, device, getattr(args, 'neval', len(graphs['test'])))