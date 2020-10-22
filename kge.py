import argparse
import time

import torch
import numpy as np

from train import train, EpochLimit
from evaluation import entity_ranking
from graph import KGraph
from models import *
from loss_functions import MarginBasedLoss, Regularized
from negative_samplers import UniformNegativeSamplerFast
from utils import int_dtype_for


parser = argparse.ArgumentParser(description='Multi-relational graph embedding')

parser.add_argument('--train', type=str, required=True)
parser.add_argument('--valid', type=str, default=None)
parser.add_argument('--test', type=str, default=None)
parser.add_argument('--columns', nargs=3, type=int, default=[0, 1, 2])
parser.add_argument('--entities', type=int, default=None)
parser.add_argument('--min_entity', type=int, default=0)
parser.add_argument('--relations', type=int, default=None)
parser.add_argument('--min_rel', type=int, default=0)
parser.add_argument('--neval', type=int, default=None)
parser.add_argument('--out', type=str, default=None)

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
            graphs[dataset] = KGraph.from_csv(
                path, 
                args.columns, 
                dtypes=dtypes, 
                n_entities=args.entities, 
                n_relations=args.relations, 
                min_entity=args.min_entity, 
                min_rel=args.min_rel)
        print(graphs[dataset])


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

model = TransE(graphs['train'], 150, p=2).to(device)
# model = Rescal(graphs['train'], 50).to(device)
# model = DistMult(graphs['train'], 50).to(device)
# model = ComplEx(graphs['train'], 50).to(device)

criterion = MarginBasedLoss(.5)
# criterion = Regularized(MarginBasedLoss(.5), graphs['train'], device, .01, .01)

optimizer = torch.optim.SGD(model.parameters(), lr=.01)


train(
    graphs,
    model=              model,
    criterion=          criterion,
    negative_sampler=   UniformNegativeSamplerFast(graphs, 2),
    optimizer=          optimizer,
    stop_condition=     EpochLimit(100),
    device=             device,
    batch_size=         10000,
    # scheduler=          torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
    scheduler=          None
)

if args.out is not None:
    embeddings = model.entity_embedding.weight
    np.savez_compressed(
        args.out,
        entity_embeddings=model.entity_embedding.weight.numpy(),
        relation_embeddings=model.relation_embedding.weight.numpy()
    )

if 'test' in graphs:
    t0 = time.time()
    results = entity_ranking(model, graphs, device, getattr(args, 'neval', args.entities))
    print('evaluation done in %s sec.' % (time.time() - t0,))
    print(results)