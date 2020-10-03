import argparse

import torch
import numpy as np

from train import train, EpochLimit
from evaluation import entity_ranking
from graph5 import KGraph
from models import TransE
from loss_functions import MarginBasedLoss
from negative_samplers import UniformNegativeSampler, UniformNegativeSamplerFast
from utils import int_dtype_for


parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument('--train', type=str, required=True)
parser.add_argument('--valid', type=str, default=None)
parser.add_argument('--test', type=str, default=None)
parser.add_argument('--columns', nargs=3, type=int, default=[0, 1, 2])
parser.add_argument('--entities', type=int, default=None)
parser.add_argument('--relations', type=int, default=None)

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
        print('loading', path)
        if path.endswith('npz'):
            graphs[dataset] = KGraph.load(path)
        else:
            graphs[dataset] = KGraph.from_csv(path, args.columns, dtypes=dtypes, n_entities=args.entities, n_relations=args.relations)
        print(graphs[dataset])


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = TransE(graphs['train'], 200, 'L1')

optimizer = torch.optim.SparseAdam(model.parameters(), lr=.01)

# quit()

train(
    graphs,
    model=              model,
    criterion=          MarginBasedLoss(.5),
    negative_sampler=   UniformNegativeSamplerFast(graphs, 20, 100),
    optimizer=          optimizer,
    stop_condition=     EpochLimit(500),
    device=             device,
    batch_size=         2000,
    scheduler=          torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    # scheduler=          None
)

if 'test' in graphs:
    print('mrr', entity_ranking(model, graphs, device, 1000))