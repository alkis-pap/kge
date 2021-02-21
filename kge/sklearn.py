import torch
import torch.nn.functional as F

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

from .graph import KGraph
from .training import train
from .negative_samplers import UniformNegativeSamplerFast
from .models import TransE
from .loss_functions import PairwiseHingeLoss
from .evaluation import evaluate


class EmbeddingEstimator(BaseEstimator):

    def __init__(
            self, 
            model=None, 
            loss=None, 
            optimizer_cls=None, 
            optimizer_args=None, 
            n_negatives=2, 
            batch_size=1000, 
            device=None, 
            n_epochs=500, 
            checkpoint_dir=None,
            validation=False,
            patience=3,
            validation_period=50,
            eval_batch_size=10000, 
            verbose=False
        ):
        
        self.model = model
        self.loss = loss
        self.n_negatives = n_negatives
        self.batch_size = batch_size
        self.optimizer_cls = optimizer_cls
        self.optimizer_args = optimizer_args
        self.device = device
        self.n_epochs = n_epochs
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose
        self.eval_batch_size = eval_batch_size
        self.validation = validation
        self.patience = patience
        self.validation_period = 50


    def fit(self, data, y=None):
        if self.validation:
            graph, validation_graph = data[0]
        else:
            graph, validation_graph = (data[0], None)

        self.model = self.model or TransE(100)
        self.loss = self.loss or PairwiseHingeLoss(margin=1)
        self.optimizer_cls = self.optimizer_cls or torch.optim.SparseAdam
        self.optimizer_args = self.optimizer_args or {'lr': 0.001}
        self.device = self.device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        negative_sampler = UniformNegativeSamplerFast(self.n_negatives)

        # init components
        self.model.init(graph, self.device)
        self.loss.init(graph, self.device)
        negative_sampler.init(graph, self.device)

        self.model.train()
        
        train(
            graph,
            self.model,
            self.loss,
            negative_sampler,
            self.optimizer_cls(self.model.parameters(), **self.optimizer_args),
            self.n_epochs,
            self.device,
            self.batch_size,
            use_checkpoint=True,
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_period=10,
            verbose=self.verbose,
            validation_graph=validation_graph,
            validation_period=self.validation_period,
            patience=self.patience
        )

        return self

    
    def evaluate(self, test_graph, train_graph=None):
        return evaluate(self.model, test_graph, self.device, train_graph, batch_size=self.eval_batch_size, verbose=self.verbose)


# class LinkPredictionEstimator(EmbeddingEstimator):

#     def __init__(self, model=None, loss=None, optimizer_cls=None, optimizer_args=None, n_negatives=2, batch_size=1000, device=None, n_epochs=500, checkpoint_dir=None, verbose=False, filtered_ranking=True, eval_batch_size=10000):
#         self.filtered_ranking = filtered_ranking
#         self.eval_batch_size = eval_batch_size
#         super().__init__(model, loss, optimizer_cls, optimizer_args, n_negatives, batch_size, device, n_epochs, checkpoint_dir, verbose)

#     def fit(self, data, y=None):
#         graph = data[0]
#         self.train_graph = graph
#         return super().fit((graph, None))

#     def predict(self, data, y=None):
#         graph = data[0]
#         train_graph = self.train_graph if self.filtered_ranking else None
#         return rank_triples(self.model, graph, self.device, train_graph, batch_size=self.eval_batch_size)


class EmbeddingTransformer(EmbeddingEstimator, TransformerMixin):

    def transform(self, data):
        _, X = data
        self.model.eval()
        with torch.no_grad():
            return self.model.entity_embedding.weight.cpu().numpy()[X]


class LinkPredictionBinaryClassifier(EmbeddingEstimator, ClassifierMixin):

    def fit(self, data, y):
        self.classes_, y = np.unique(y, return_inverse=True)

        graph, X = data

        tail = np.full_like(X, graph.n_entities)
        tail[y == 1] = graph.n_entities + 1

        graph = KGraph.from_htr(
            head=np.concatenate((graph.head, X)),
            tail=np.concatenate((graph.tail, tail)),
            relation=np.concatenate((graph.relation, np.full_like(X, graph.n_relations))),
            n_entities=graph.n_entities + 2,
            n_relations=graph.n_relations + 1
        )

        return super().fit((graph, X))

    def score(self, data):
        graph, X = data
        
        self.model.eval()
        with torch.no_grad():
            test_ids = torch.from_numpy(X).to(self.device)
            
            print(test_ids.size())

            scores = torch.empty((test_ids.size(0), 2))

            positive_scores = self.model.forward((
                test_ids,
                torch.full_like(test_ids, graph.n_entities - 1).to(self.device), 
                torch.full_like(test_ids, graph.n_relations - 1).to(self.device)
            ))

            print(positive_scores.size())

            scores[:, 0] = positive_scores

            scores[:, 1] = self.model.forward((
                test_ids,
                torch.full_like(test_ids, graph.n_entities - 2),
                torch.full_like(test_ids, graph.n_relations - 1)
            ))
        return scores


    def predict(self, data):
        D = self.score(data)
        return self.classes_[np.argmax(D, axis=1)]


    def predict_proba(self, data):
        return F.softmax(self.score(data), dim=1).numpy()

        
        

class FeatureGenerator(BaseEstimator, TransformerMixin):


    def fit(self, data, y=None):
        graph, _ = data

        self.features = np.empty((graph.n_entities, graph.n_relations * 2))

        for r in range(graph.n_relations):
            self.features[:, r] = np.bincount(graph.head[graph.relation == r], minlength=graph.n_entities)
            self.features[:, graph.n_relations + r] = np.bincount(graph.tail[graph.relation == r], minlength=graph.n_entities)

        return self

    def transform(self, data):
        _, X = data

        return self.features[X]