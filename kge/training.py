import os, copy, bisect
from contextlib import suppress
from hashlib import sha256
from operator import itemgetter

import numpy as np
import torch
# from torch.cuda.amp import GradScaler, autocast

from .evaluation import evaluate
from .utils import timeit, strip_whitespace



def train(
        graph, model, criterion, negative_sampler, 
        optimizer, n_epochs, device,batch_size=100, scheduler=None, 
        use_checkpoint=False, checkpoint_dir=None, checkpoint_period=1, verbose=False, 
        # validation_graph=None, validation_period=50, eval_batch_size=10000, patience=0
    ):
    
    epoch = 0
    training_loss = [None]
    
    # best_score = 0
    # best_epoch = 9999
    # best_model_state = None

    checkpoint_archive_id = '\n'.join([
        f'model: {strip_whitespace(str(model))}',
        f'criterion: {strip_whitespace(str(criterion))}',
        f'optimizer: {strip_whitespace(str(optimizer))}',
        f'negative_sampler: {strip_whitespace(str(negative_sampler))}',
        f'scheduler: {strip_whitespace(str(scheduler))}',
        f'train_graph: {strip_whitespace(str(graph))}',
        f'batch_size: {batch_size}'
        # ,
        # f'validation_graph: {strip_whitespace(str(validation_graph))}',
        # f'validation_period: {validation_period}'
    ])
    
    if verbose:
        print(checkpoint_archive_id)
    
    if use_checkpoint:
        hash_code = sha256(checkpoint_archive_id.encode('utf-8')).hexdigest()[-12:]
    
        checkpoint_dir = checkpoint_dir or '.'
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{hash_code}.pt')

        checkpoints = [{}]
        active_checkpoint = checkpoints[-1]
        if os.path.isfile(checkpoint_path):
            checkpoints = torch.load(checkpoint_path)

            for checkpoint in sorted(checkpoints, key=itemgetter('epoch'), reverse=True):
                if checkpoint['epoch'] <= n_epochs:
                    if checkpoint['epoch'] == checkpoint['n_epochs']:
                        active_checkpoint = copy.deepcopy(checkpoint)
                        checkpoints.append(active_checkpoint)
                    else:
                        active_checkpoint = checkpoint
                    break

            if active_checkpoint:
                training_loss = active_checkpoint['training_loss']
                epoch = len(training_loss)
                model.load_state_dict(active_checkpoint['model_state'])
                optimizer.load_state_dict(active_checkpoint['optimizer_state'])
        active_checkpoint['n_epochs'] = n_epochs
                # best_score = checkpoint['best_score']
                # best_epoch = checkpoint['best_epoch']
                # best_model_state = checkpoint['best_model_state']


    while epoch < n_epochs:

        # if best_model_state and epoch > best_epoch + patience * validation_period:
        #     model.load_state_dict(best_model_state)
        #     return

        with timeit("Training epoch " + str(epoch)) if verbose else suppress():
            model.train(True) # training mode enables gradients

            # generate permutation
            indices = np.random.permutation(len(graph))

            total_loss = 0

            for batch_start in range(0, len(graph), batch_size):
                optimizer.zero_grad()

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
                    
                # calculate gradients
                loss.backward()

                if torch.isnan(loss):
                    print(f'Nan detected: epoch: {epoch}, triple: [{batch_start} - {batch_start + batch_size}) ({len(graph)} total)')
                    return

                # update parameters
                optimizer.step()

                model.normalize()

                total_loss += loss.item() / scores.size(1) / len(idx)

            
            if verbose:
                if training_loss[-1] is not None:
                    print(f'training loss: {total_loss:.4E} ({100 * (total_loss - training_loss[-1]) / training_loss[-1]:+.4f} %)')
                else:
                    print(f'training loss: {total_loss:.4E}')
                
            training_loss.append(total_loss)

            if scheduler is not None:
                scheduler.step()

        # if validation_graph and (epoch + 1) % validation_period == 0:
        #     scores = evaluate(model, validation_graph, device, batch_size=eval_batch_size)
        #     if verbose:
        #         print('validation scores:', scores)
        #     validation_score = scores['both']['hits@10']
        #     if validation_score > best_score:
        #         best_score = validation_score
        #         best_epoch = epoch
        #         best_model_state = copy.deepcopy(model.state_dict())
        
        epoch += 1

        if use_checkpoint and epoch % checkpoint_period == 0:
            with timeit("Updating checkpoint") if verbose else suppress():
                active_checkpoint['model_state'] = model.state_dict()
                active_checkpoint['optimizer_state'] = optimizer.state_dict()
                active_checkpoint['training_loss'] = training_loss
                active_checkpoint['epoch'] = epoch
                    
                    # 'best_score': best_score,
                    # 'best_model_state': best_model_state,
                    # 'best_epoch': best_epoch
                torch.save(checkpoints, checkpoint_path)

        # abort if the loss has not decreased by at least 0.1 % within this window
        improvement_window = int(10 + 90 * epoch / 1000)
        if epoch > improvement_window:
            old_loss = training_loss[-improvement_window]
            if old_loss - training_loss[-1] < 0.001 * old_loss:
                # too slow
                print('aborting training: training loss is not improving.')
                return
                    
