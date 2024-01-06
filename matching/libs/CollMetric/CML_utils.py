import math
import numpy as np
import tensorflow as tf
from scipy.sparse import lil_matrix, dok_matrix
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import argparse
import toolz

class Evaluator(object):
    def __init__(self, model, train_user_item_matrix, test_user_item_matrix):
        """
        Create a evaluator for hitrate@K, recall@K and NDCG@K evaluation
        :param model: the model we are going to evaluate
        :param train_user_item_matrix: the user-item pairs used in the training set. These pairs will be ignored
               in the metric calculation
        :param test_user_item_matrix: the held-out user-item pairs we make prediction against
        """
        self.model = model
        self.train_user_item_matrix = lil_matrix(train_user_item_matrix)
        self.test_user_item_matrix = lil_matrix(test_user_item_matrix)
        n_users = train_user_item_matrix.shape[0]
        self.user_to_test_set = {u: set(self.test_user_item_matrix.rows[u])
                                 for u in range(n_users) if self.test_user_item_matrix.rows[u]}

        if self.train_user_item_matrix is not None:
            self.user_to_train_set = {u: set(self.train_user_item_matrix.rows[u])
                                      for u in range(n_users) if self.train_user_item_matrix.rows[u]}
            self.max_train_count = max(len(row) for row in self.train_user_item_matrix.rows)
        else:
            self.max_train_count = 0

    def eval(self, sess, users, k=50):
        """
        Compute the Top-K metrics for a particular user given the predicted scores to items
        :param users: the users to eval the metrics
        :param k: compute for the top K items
        :return: hitrate@K, recall@K and NDCG@K
        """
        # compute the top (K +  Max Number Of Training Items for any user) items for each user

        _, user_tops = sess.run(tf.nn.top_k(self.model.item_scores, k + self.max_train_count),
                                {self.model.score_user_ids: users})
        hitrates = []
        recalls = []
        NDCGs = []
        Gain_tbl = (1 / np.log(np.arange(2, k + 2)))
        for user_id, tops in zip(users, user_tops):
            train_set = self.user_to_train_set.get(user_id, set())
            test_set = self.user_to_test_set.get(user_id, set())
            top_n_items = 0
            hits = 0 # how many items hit the current user's preference
            user_hit = 0 # if hit happens to this user
            u_DCG = 0
            for i in tops:
                # ignore item in the training set
                if i in train_set:
                    continue
                elif i in test_set:
                    user_hit = 1 # hit happens
                    hits += 1
                    u_DCG += Gain_tbl[top_n_items]
                top_n_items += 1
                if top_n_items == k:
                    break
            hitrates.append(user_hit)
            recalls.append(hits / float(len(test_set)))
            NDCGs.append(u_DCG / np.sum(Gain_tbl[:min(k, len(test_set))]))
        return hitrates, recalls, NDCGs


class Monitor:
    '''
    Create a monitor on validation metrics, based on which it decides whether stoping the optimization early
    '''
    def __init__(self, max_patience=5, delta=10e-6, log_file=None):
        self.counter = 0
        self.best_value = 0
        self.max_patience = max_patience
        self.patience = max_patience
        self.delta = delta
        self.log_file = log_file
        print("time,iteration,hitrate@20,recall@20,ndcg@20,hitrate@50,recall@50,ndcg@50", file=self.log_file)

    def update_monitor(self, hitrate20, recall20, ndcg20, hitrate50, recall50, ndcg50):
        self.counter += 1
        print("%s validation %d:" % (datetime.now(), self.counter))
        print("%s hitrate@20=%.4lf, recall@20=%.4lf, ndcg@20=%.4lf" % 
              (datetime.now(), hitrate20, recall20, ndcg20))
        print("%s hitrate@50=%.4lf, recall@50=%.4lf, ndcg@50=%.4lf\n" %
              (datetime.now(), hitrate50, recall50, ndcg50))
        print("%s,%d,%f,%f,%f,%f,%f,%f" % 
              (datetime.now(), self.counter, hitrate20, recall20, ndcg20, hitrate50, recall50, ndcg50),
              file=self.log_file)
        value = recall20 + ndcg20 + recall50 + ndcg50
        if value < self.best_value + self.delta:
            self.patience -= 1
            print("%s the monitor loses its patience to %d!" % (datetime.now(), self.patience))
            if self.patience == 0:
                return True
        else:
            self.patience = self.max_patience
            self.best_value = value
            return False

        
def parse_args():
    parser = argparse.ArgumentParser(description="Run CML")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name for saving result')    
    parser.add_argument('--train_data', type=str, required=True,
                        help='Training data path')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Testing data path')
    parser.add_argument('--verbose', nargs='?', type=int, default=30,
                        help='Evaluation interval.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=50000,
                        help='batch_size')
    parser.add_argument('--num_negative', nargs='?', type=int, default=20,
                       help='Number of nagative samples.')
    parser.add_argument('--max_steps', nargs='?', type=int, default=5000,
                        help='Max training step.')
    parser.add_argument('--embed_dim', type=int, default=100,
                        help='Embedding dimension.')
    parser.add_argument('--margin', type=float, default=1.9,
                        help='Margin in the hinge loss.')
    parser.add_argument('--clip_norm', type=float, default=1,
                        help='Clip the embedding so that their norm <= clip_norm.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--topK', nargs='+', type=int, default=[20, 50],
                        help='topK for hr/ndcg')
    parser.add_argument('--gpu', type=str, default='-1',
                        help='GPU index')
    return parser.parse_args()


def dataset_to_uimatrix(train_data, test_data):
    train_user_dict = defaultdict(set)
    with open(train_data) as train_file:
        for u, item_list in enumerate(train_file.readlines()):
            items = item_list.strip().split(" ")
            # ignore the first element in each line, which is the number of items the user liked. 
            for item in items[1:]:
                train_user_dict[u].add(int(item))

    test_user_dict = defaultdict(set)
    with open(test_data) as test_file:
        for u, item_list in enumerate(test_file.readlines()):
            items = item_list.strip().split(" ")
            # ignore the first element in each line, which is the number of items the user liked. 
            for item in items[1:]:
                test_user_dict[u].add(int(item))

    total_user_dict = defaultdict(set)
    for u in (set(train_user_dict.keys()) | set(test_user_dict.keys())):
        total_user_dict[u] = (train_user_dict[u] | test_user_dict[u])

    n_users = len(total_user_dict)
    n_items = max([item for items in total_user_dict.values() for item in items]) + 1

    train_user_item_matrix = dok_matrix((n_users, n_items), dtype=np.int32)
    test_user_item_matrix = dok_matrix((n_users, n_items), dtype=np.int32)
    for u, item_set in train_user_dict.items():
        for item in item_set:
            train_user_item_matrix[u, item] = 1
    for u, item_set in test_user_dict.items():
        for item in item_set:
            test_user_item_matrix[u, item] = 1
    
    return train_user_item_matrix, test_user_item_matrix