import argparse
import time

import torch
from torch import nn
import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split

from skorch import NeuralNetClassifier


parser = argparse.ArgumentParser(description='Entity classification')

parser.add_argument('--embeddings', type=str, default=None)
parser.add_argument('--train', type=str, required=True)
parser.add_argument('--valid', type=str, default=None)
parser.add_argument('--test', type=str, default=None)
parser.add_argument('--id_column', type=int, default=0)
parser.add_argument('--categorical_features', nargs='*', type=int, default=[])
parser.add_argument('--numerical_features', nargs='*', type=int, default=[])
parser.add_argument('--label_column', type=int, default=1)
parser.add_argument('--min_entity', type=int, default=0)

args = parser.parse_args()

data = {}

column_names = [
    'id', 
    *('C%d' % (i,) for i in range(len(args.categorical_features))),
    *('X%d' % (i,) for i in range(len(args.numerical_features))),
    'label']

columns = [args.id_column, *args.categorical_features, *args.numerical_features, args.label_column]

dtypes = {'id': np.int32, 'label': 'category'}

for i in range(len(args.categorical_features)):
    dtypes['C%d' % (i,)] = "category"

for i in range(len(args.categorical_features)):
    dtypes['X%d' % (i,)] = np.float

for dataset in ['train', 'test', 'valid']:
    path = getattr(args, dataset, None)
    if path is not None:
        data[dataset] = pd.read_csv(
            path,
            sep='\t',
            names=np.array(column_names)[np.argsort(columns)],
            usecols=columns,
            dtype=dtypes,
            header=None
        )


class MyModule(nn.Module):
    def __init__(self, input_size=3, num_units=10, nonlin=nn.ReLU()):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(input_size, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.softmax(self.output(X))
        return X



X = np.stack([
    *(data['train']['C%d' % (i,)].cat.codes.values for i in range(len(args.categorical_features))),
    *(data['train']['X%d' % (i,)].values for i in range(len(args.numerical_features)))
], 1).astype(np.float32)

y = data['train']['label'].values.astype(np.int64)

# clf = sklearn.linear_model.LogisticRegression(random_state=0)
# clf = sklearn.linear_model.SGDClassifier(loss="hinge", penalty="l2"
clf = NeuralNetClassifier(
    MyModule(),
    # device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    max_epochs=100,
    lr=0.01,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=42)

print('training...')

clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)

# acc = sklearn.model_selection.cross_val_score(clf, X, y, cv=train_test_split(), scoring='balanced_accuracy', n_jobs=-1)

print(acc)
# print(np.mean(acc))