import os
from contextlib import suppress
from hashlib import sha256
from operator import itemgetter
import bisect

import numpy as np
import torch
# from torch.cuda.amp import GradScaler, autocast

from .utils import timeit, strip_whitespace


# class EpochLimit(object):
#     def __init__(self, n_epochs):
#         self.n_epochs = n_epochs

#     def __call__(self, epoch, *args):
#         return epoch >= self.n_epochs
        

def train(
        graph, model, criterion, negative_sampler, optimizer, n_epochs, device,
        batch_size=100, scheduler=None, use_checkpoint=False, checkpoint_dir=None, checkpoint_period=1, verbose=False, mixed_precision=False
    ):
    
    epoch = 0
    training_loss = [None]

    # scaler = GradScaler(enabled=mixed_precision)

    checkpoint_archive_id = '\n'.join([
        f'model: {strip_whitespace(str(model))}',
        f'criterion: {strip_whitespace(str(criterion))}',
        f'optimizer: {strip_whitespace(str(optimizer))}',
        f'negative_sampler: {strip_whitespace(str(negative_sampler))}',
        f'scheduler: {strip_whitespace(str(scheduler))}',
        f'n_entities: {graph.n_entities}',
        f'n_relations: {graph.n_relations}',
        f'n_edges: {len(graph)}',
        f'batch_size: {batch_size}'
    ])
    if verbose:
        print(checkpoint_archive_id)
    
    hash_code = sha256(checkpoint_archive_id.encode('utf-8')).hexdigest()[-12:]
    
    checkpoint_dir = checkpoint_dir or '.'
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{hash_code}.pt')
    
    if use_checkpoint:
        checkpoint_archive = []
        active_checkpoint = {}
        if os.path.isfile(checkpoint_path):
            checkpoint_archive = torch.load(checkpoint_path)
            checkpoint_archive = sorted(checkpoint_archive, key=itemgetter('n_epochs'), reverse=True)
            active_checkpoint = next(filter(lambda chkpt: chkpt['n_epochs'] <= n_epochs, checkpoint_archive), None)
        if active_checkpoint:
            training_loss = active_checkpoint['training_loss']
            epoch = active_checkpoint['n_epochs']
            model.load_state_dict(active_checkpoint['model_state'])
            optimizer.load_state_dict(active_checkpoint['optimizer_state'])
            # scaler.load_state_dict(active_checkpoint['scaler_state'])
        else:
            checkpoint_archive.append(active_checkpoint)
        active_checkpoint['max_epochs'] = n_epochs

    while epoch < n_epochs:

        with timeit("Training epoch " + str(epoch)) if verbose else suppress():
            model.train(True) # training mode enables gradients

            # generate permutation
            indices = np.random.permutation(len(graph))

            total_loss = 0

            model.normalize()

            for batch_start in range(0, len(graph), batch_size):
                optimizer.zero_grad()

                idx = indices[batch_start : batch_start + batch_size]

                # positive examples
                triples = graph[idx]

                # positive + negative examples
                triples = negative_sampler(triples)

                # send to device
                triples_tensor = torch.stack([torch.from_numpy(arr).detach().to(device, dtype=torch.long) for arr in triples])

                # with autocast(enabled=mixed_precision):
                    
                # obtain embeddings
                embeddings = model.encode(triples_tensor)

                # calculate scores
                scores = model.score(*embeddings)

                # calculate loss
                loss = criterion(scores, triples_tensor, embeddings)
                # code.interact(local=locals())
                    
                # calculate gradients
                # scaler.scale(loss).backward()
                loss.backward()

                # update parameters
                optimizer.step()

                # scaler.step(optimizer)
                
                # scaler.update()


                total_loss += loss.item()
                del loss
            
            if verbose:
                if training_loss[-1] is not None:
                    print(f'training loss: {total_loss:.4E} ({100 * (total_loss - training_loss[-1]) / training_loss[-1]:+.4f} %)')
                else:
                    print(f'training loss: {total_loss:.4E}')
                
            training_loss.append(total_loss)

        if scheduler is not None:
            scheduler.step()

        epoch += 1

        if use_checkpoint and epoch % checkpoint_period == 0:
            with timeit("Updating checkpoint") if verbose else suppress():
                active_checkpoint['model_state'] = model.state_dict()
                active_checkpoint['optimizer_state'] = optimizer.state_dict()
                # active_checkpoint['scaler_state'] = scaler.state_dict()
                active_checkpoint['training_loss'] = training_loss
                active_checkpoint['n_epochs'] = epoch
                torch.save(checkpoint_archive, checkpoint_path)

