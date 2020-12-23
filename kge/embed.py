import argparse

import torch
import numpy as np

import kge.models
import kge.loss_functions
from .training import train, EpochLimit
from .evaluation import entity_ranking
from .graph import KGraph
from .negative_samplers import UniformNegativeSamplerFast
from .utils import int_dtype_for, timeit, get_class, make_object
import kge.loss_adaptors


def load_graph(path, args):
    with timeit("loading graph: " + path):
        if path.endswith('npz'):
            return KGraph.load(path)
        else:
            if args.entities is None:
                dtypes = [np.long] * 2
            else:
                dtypes = [int_dtype_for(args.entities)] * 2

            if args.relations is None:
                dtypes.append(np.long)
            else:
                dtypes.append(int_dtype_for(args.relations))

            return KGraph.from_csv(
                path,
                args.columns,
                dtypes=dtypes,
                n_entities=args.entities,
                n_relations=args.relations,
                min_entity=args.min_entity,
                min_rel=args.min_rel)


def main():

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
    parser.add_argument('--model', type=str, default='TransE(50)')
    parser.add_argument('--loss', type=str, default='PairwiseHingeLoss(margin=0.5)')
    parser.add_argument('--loss_adaptor', type=str, default=None)
    parser.add_argument('--negatives', type=int, default=2)

    # training
    parser.add_argument('--nepochs', type=int, default=100)
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--checkpoint_period', type=int, default=1)
    parser.add_argument('--device', type=str, default=None)

    # evaluation
    parser.add_argument('--neval', type=int, default=None)

    args = parser.parse_args()

    graphs = {}
    graphs['train'] = load_graph(args.train, args)

    args.entities = graphs['train'].n_entities
    args.relations = graphs['train'].n_relations
    args.min_entity = graphs['train'].min_entity
    args.min_rel = graphs['train'].min_rel

    if args.test is not None:
        graphs['test'] = load_graph(args.test, args)

    if args.valid is not None:
        graphs['valid'] = load_graph(args.valid, args)
    
    for dataset in graphs:
        print(dataset, graphs[dataset])

    if args.device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device {device}")

    model = make_object(kge.models, args.model).init(graphs['train']).to(device)

    criterion = make_object(kge.loss_functions, args.loss)
    if args.loss_adaptor is not None:
        criterion = make_object(kge.loss_adaptors, args.loss_adaptor).init(criterion, graphs['train'], device)

    optimizer_cls = get_class(torch.optim, args.optim)
    optimizer = optimizer_cls(model.parameters(), lr=args.lr)

    train(
        graphs,
        model=              model,
        criterion=          criterion,
        negative_sampler=   UniformNegativeSamplerFast(graphs, args.negatives),
        optimizer=          optimizer,
        stop_condition=     EpochLimit(args.nepochs),
        device=             device,
        batch_size=         args.batch_size,
        # scheduler=          torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
        scheduler=          None,
        checkpoint_path=    args.checkpoint,
        checkpoint_period=  args.checkpoint_period
    )

    if args.out is not None:
        with torch.no_grad():
            np.savez_compressed(
                args.out.format(
                    model=model.__class__.__name__, 
                    dim=model.entity_dim, 
                    nepochs=args.nepochs,
                    loss=criterion.__class__.__name__
                ),
                entity_embeddings=model.encoder.entity_embedding.weight.cpu().numpy(),
                relation_embeddings=model.encoder.relation_embedding.weight.cpu().numpy()
            )

    if 'test' in graphs:
        with timeit("Evaluation"):
            entity_ranking(model, graphs, device, getattr(args, 'neval', len(graphs['test'])))
