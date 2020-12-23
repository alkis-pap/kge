import argparse

import numpy as np

import datatable as dt

from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline



def main():
    parser = argparse.ArgumentParser(description='Entity classification')

    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--label_data', type=str, required=True)
    parser.add_argument('--test_split', type=int, default=0.1)

    args = parser.parse_args()

    if args.label_data.endswith('.npz'):
        data = np.load(args.label_data)
        train_ids = data['train_ids']
        test_ids = data['test_ids']
        y_train = data['train_labels']
        y_test = data['test_labels']
    else:
        df = dt.fread(args.label_data).to_numpy()
        perm = np.random.permutation(df.shape[0])
        train_size = int(df.shape[0] * (1 - args.test_split))
        train_ids = df[perm[:train_size], 0].astype(np.int32)
        test_ids = df[perm[train_size:], 0].astype(np.int32)
        y_train = df[perm[:train_size], -1].astype(np.int32)
        y_test = df[perm[train_size:], -1].astype(np.int32)

    embeddings = np.load(args.embeddings)['entity_embeddings']

    print(train_ids.size)
    print(test_ids.size)

    X_train = embeddings[train_ids]
    X_test = embeddings[test_ids]

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    # scaler = StandardScaler().fit(X_train)

    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)


    # print(X_train, y_train)

    del embeddings

    classifiers = [
        DummyClassifier(strategy='most_frequent'),
        DecisionTreeClassifier(max_depth=5),
        GradientBoostingClassifier(random_state=0),
        MLPClassifier(alpha=1, max_iter=1000),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=8),
        KNeighborsClassifier(3, n_jobs=8),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
    ]

    for clf in classifiers:
        print(clf)
        pipeline = Pipeline([('transformer', StandardScaler()), ('estimator', clf)])
        score = cross_val_score(pipeline, X, y, cv=10, scoring='roc_auc')
        # clf.fit(X_train, y_train)
        # score = clf.score(X_test, y_test)
        print(clf, score, score.mean())



if __name__ == '__main__':
    main()
