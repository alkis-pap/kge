import signal
import time

import numpy as np
import torch

class EpochLimit(object):
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs

    def __call__(self, epoch):
        return epoch >= self.n_epochs
        

def train(graphs, model, criterion, negative_sampler, optimizer, stop_condition, device, batch_size=100, scheduler=None):
    print("Using device:", device)
    
    model = model.to(device)

    graph = graphs['train']

    validation_data = negative_sampler(graphs['valid'][:])

    t0 = time.time()
    epoch = 0

    # singal handling for early stopping with ctrl-C
    train.done = False
    def signal_handler(signal, frame):
        train.done = True
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    while not train.done:
        print("Begin training epoch:", epoch)

        model.train(True)
        indices = np.random.permutation(len(graph))

        ratios = []
        for batch_start in range(0, len(graph), batch_size):
            idx = indices[batch_start : batch_start + batch_size]

            data = negative_sampler(graph[idx])

            tensors = [torch.from_numpy(arr).to(device, dtype=torch.long) for arr in data]

            scores = model(tensors)

            loss = criterion(scores)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # validation phase
        if 'valid' in graphs:
            model.train(False)
            total_loss = 0
            with torch.no_grad():
                for batch_start in range(0, len(graphs['valid']), batch_size):
                    idx = slice(batch_start, batch_start + batch_size)

                    # data = negative_sampler(graphs['valid'][idx])
                    # data = validation_data[idx]

                    tensors = [torch.from_numpy(arr[idx]).to(device, dtype=torch.long) for arr in validation_data]

                    scores = model(tensors)
                    loss = criterion(scores)
                    total_loss += loss.item()
                    del loss

            print('validation loss:', total_loss)

        t1 = time.time()
        print("epoch done in {} sec".format(t1 - t0))
        t0 = t1

        if scheduler is not None:
            scheduler.step()

        if stop_condition(epoch):
            break
        epoch += 1

    signal.signal(signal.SIGINT, original_sigint_handler)