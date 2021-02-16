import os
from contextlib import suppress
from hashlib import sha256

import numpy as np
import torch

from .utils import timeit, strip_whitespace


# class EpochLimit(object):
#     def __init__(self, n_epochs):
#         self.n_epochs = n_epochs

#     def __call__(self, epoch, *args):
#         return epoch >= self.n_epochs
        

def train(
        graph, model, criterion, negative_sampler, optimizer, n_epochs, device,
        batch_size=100, scheduler=None, checkpoint=False, checkpoint_dir='.', checkpoint_period=1, verbose=False
    ):

    # graph = graphs['train']

    # # sample negatives for validation only once
    # if 'valid' in graphs:
    #     validation_data = negative_sampler(graphs['valid'][:])

    epoch = 0
    training_loss = []

    checkpoint_id = '\n'.join([
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
        print(checkpoint_id)
    hash_code = sha256(checkpoint_id.encode('utf-8')).hexdigest()[-8:]
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{hash_code}.npz')
    
    if checkpoint is not None and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        training_loss = checkpoint['training_loss']
        epoch = len(training_loss)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    while epoch < n_epochs:

        with timeit("Training epoch " + str(epoch)) if verbose else suppress():
            model.train(True) # training mode enables gradients

            # generate permutation
            indices = np.random.permutation(len(graph))

            total_loss = 0

            for batch_start in range(0, len(graph), batch_size):
                idx = indices[batch_start : batch_start + batch_size]

                # positive examples
                triples = graph[idx]

                # positive + negative examples
                triples = negative_sampler(triples)

                # send to device
                triples_tensor = torch.stack([torch.from_numpy(arr).detach().to(device, dtype=torch.long) for arr in triples])

                # obtain embeddings
                embeddings = model.encode(triples_tensor)

                # calculate scores
                scores = model.score(*embeddings)

                # calculate loss
                loss = criterion(scores, triples_tensor, embeddings)
                # code.interact(local=locals())

                optimizer.zero_grad()
                
                # calculate gradients
                loss.backward()

                # update parameters
                optimizer.step()

                total_loss += loss.item()
                del loss
            
            if verbose:
                print('training loss:', total_loss)
            training_loss.append(total_loss)

            # # validation phase
            # if 'valid' in graphs:
            #     model.train(False)
            #     validation_loss = 0
            #     with torch.no_grad():
            #         for batch_start in range(0, len(graphs['valid']), batch_size):
            #             idx = slice(batch_start, batch_start + batch_size)

            #             triples_tensor = torch.stack([torch.from_numpy(arr[idx]).to(device, dtype=torch.long) for arr in validation_data])

            #             embeddings = model.encode(triples_tensor)

            #             scores = model.score(*embeddings)

            #             loss = criterion(scores, triples_tensor, embeddings)

            #             validation_loss += loss.item()
            #             del loss

            #     print('validation loss:', validation_loss)

        if checkpoint and (epoch + 1) % checkpoint_period == 0:
            with timeit("Updating checkpoint") if verbose else suppress():
                torch.save(
                    {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'training_loss': training_loss,
                    },
                    checkpoint_path
                )

        if scheduler is not None:
            scheduler.step()

        epoch += 1
