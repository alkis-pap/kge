import signal
import time
import code

import numpy as np
import torch

class EpochLimit(object):
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs

    def __call__(self, epoch):
        return epoch >= self.n_epochs - 1
        

def train(graphs, model, criterion, negative_sampler, optimizer, stop_condition, device, batch_size=100, scheduler=None):
    print("Using device:", device)
    
    # send model parameters to device
    model = model.to(device)

    graph = graphs['train']

    # sample negatives for validation only once
    if 'valid' in graphs:
        validation_data = negative_sampler(graphs['valid'][:])

    # singal handling for early stopping with ctrl-C
    train.done = False
    def signal_handler(signal, frame):
        if train.done == True:
            quit()
        train.done = True
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)
    
    epoch = 0
    t0 = time.time()

    while not train.done:
        print("Begin training epoch:", epoch)

        model.train(True) # training mode enables gradients

        # generate permutation
        indices = np.random.permutation(len(graph))

        training_loss = 0

        ratios = []
        for batch_start in range(0, len(graph), batch_size):
            idx = indices[batch_start : batch_start + batch_size]

            # positive examples
            triples = graph[idx]

            # positive + negative examples
            triples = negative_sampler(triples)

            # send to device
            triples_tensor = torch.stack([torch.from_numpy(arr).to(device, dtype=torch.long) for arr in triples])

            # obtain embeddings
            embeddings = model.encode(triples_tensor)

            # calculate scores
            scores = model.decode(embeddings)

            # calculate loss
            loss = criterion(scores, triples_tensor, embeddings)
            
            # code.interact(local=locals())

            optimizer.zero_grad()
            
            # calculate gradients
            loss.backward()

            # update parameters
            optimizer.step()

            training_loss += loss.item()
            del loss

        print('training loss:', training_loss)

        # validation phase
        if 'valid' in graphs:
            model.train(False)
            validation_loss = 0
            with torch.no_grad():
                for batch_start in range(0, len(graphs['valid']), batch_size):
                    idx = slice(batch_start, batch_start + batch_size)

                    triples_tensor = torch.stack([torch.from_numpy(arr[idx]).to(device, dtype=torch.long) for arr in validation_data])

                    embeddings = model.encode(triples_tensor)

                    scores = model.decode(embeddings)

                    loss = criterion(scores, triples_tensor, embeddings)

                    validation_loss += loss.item()
                    del loss

            print('validation loss:', validation_loss)

        t1 = time.time()
        print("epoch done in {} sec".format(t1 - t0))
        t0 = t1

        if scheduler is not None:
            scheduler.step()

        if stop_condition(epoch):
            break

        epoch += 1

    # remove signal handler
    signal.signal(signal.SIGINT, original_sigint_handler)