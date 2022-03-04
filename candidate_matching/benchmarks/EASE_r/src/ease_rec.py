#======================================================================
# The EASE recommender code modified from https://github.com/Darel13712/ease_rec
# Authors: Darel13712
#          XUEPAI team
#======================================================================

from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from datetime import datetime
from .utils import evaluate_metrics


class EASE:
    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, 'uid'])
        items = self.item_enc.fit_transform(df.loc[:, 'sid'])
        return users, items

    def fit(self, train_path, lambda_: float=0.5, implicit=True):
        """
        train_path: csv data input
        lambda_: l2-regularization term
        implicit: if True, ratings are ignored and taken as 1, else normalized ratings are used
        """
        print("%s Fitting EASE model..." % datetime.now())
        df = pd.read_csv(train_path, sep="\t").astype(int)
        users, items = self._get_users_and_items(df)
        values = np.ones(df.shape[0]) if implicit else df['rating'].to_numpy() / df['rating'].max()

        X = csr_matrix((values, (users, items)))
        self.X = X

        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = B
        self.pred = X.dot(B)

    def evaluate(self, test_path, metrics):
        '''compute hitrate, recall, NDCG @ topK'''
        print("%s Evaluating metrics..." % datetime.now())
        self.pred[self.X.toarray() > 0] = -np.inf # remove clicked items in train data
        test = pd.read_csv(test_path, sep="\t").astype(int)
        test_user2items = defaultdict(list)
        for _, row in test.iterrows():
            test_user2items[row['uid']].append(row['sid'])
        return evaluate_metrics(self.pred, test_user2items, metrics)

