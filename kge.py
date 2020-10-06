import argparse

import torch
import numpy as np

from train import train, EpochLimit
from evaluation import entity_ranking
from graph import KGraph
from models import TransE, Rescal, ComplEx
from loss_functions import MarginBasedLoss, Regularized
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
# device = torch.device('cpu')

# model = TransE(graphs['train'], 50, 'L2')
# model = Rescal(graphs['train'], 50).to(device)
model = ComplEx(graphs['train'], 50).to(device)

optimizer = torch.optim.Adagrad(model.parameters(), lr=.01)

# quit()

train(
    graphs,
    model=              model,
    criterion=          Regularized(MarginBasedLoss(.5), graphs['train'], device, .01, .01),
    negative_sampler=   UniformNegativeSamplerFast(graphs, 10, 1500),
    optimizer=          optimizer,
    stop_condition=     EpochLimit(500),
    device=             device,
    batch_size=         1500,
    # scheduler=          torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
    scheduler=          None
)

if 'test' in graphs:
    # mrr, hits = entity_ranking(model, graphs, device, 14951)
    results = entity_ranking(model, graphs, device, 1000)
    print(results)