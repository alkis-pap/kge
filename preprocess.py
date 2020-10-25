import argparse

import torch
import numpy as np

from kge.train import train, EpochLimit
from kge.evaluation import entity_ranking
from kge.graph import KGraph
from kge.models import *
from kge.loss_functions import MarginBasedLoss, Regularized
from kge.egative_samplers import UniformNegativeSampler, UniformNegativeSamplerFast
from kge.utils import int_dtype_for


parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument('input', type=str)
parser.add_argument('output', type=str)
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


print('loading', args.input)
graph = KGraph.from_csv(args.input, args.columns, dtypes=dtypes, n_entities=args.entities, n_relations=args.relations)
print(graph)

print('saving', args.output)
graph.save(args.output)