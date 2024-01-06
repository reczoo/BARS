#======================================================================
# Evaluation metrics modified from DEEM project
# Edited by XUEPAI team
#======================================================================

import numpy as np
import itertools
import os
from datetime import datetime


def evaluate_metrics(pred_matrix, test_user2items, metrics):
    num_users = len(pred_matrix)
    print("{} Evaluating metrics for {:d} users...".format(datetime.now(), num_users))
    metric_callers = []
    max_topk = 0
    for metric in metrics:
        try:
            metric_callers.append(eval(metric))
            max_topk = max(max_topk, int(metric.split("k=")[-1].strip(")")))
        except:
            raise NotImplementedError('metrics={} not implemented.'.format(metric))

    topk_items_chunk = np.argpartition(-pred_matrix, range(max_topk))[:, 0:max_topk]
    true_items_chunk = [test_user2items[user_index] for user_index in range(num_users)]
    results = [[fn(topk_items, true_items) for fn in metric_callers] \
                for topk_items, true_items in zip(topk_items_chunk, true_items_chunk)]
    average_result = np.average(np.array(results), axis=0).tolist()
    return_result = ' - '.join('{}: {:.6f}'.format(k, v) for k, v in zip(metrics, average_result))
    print('%s [Metrics] ' % datetime.now() + return_result)
    return return_result


class Recall(object):
    """Recall metric."""
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        recall = len(hit_items) / (len(true_items) + 1e-12)
        return recall


class Precision(object):
    """Precision metric."""
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        precision = len(hit_items) / (self.topk + 1e-12)
        return precision


class F1(object):
    def __init__(self, k=1):
        self.precision_k = Precision(k)
        self.recall_k = Recall(k)

    def __call__(self, topk_items, true_items):
        p = self.precision_k(topk_items, true_items)
        r = self.recall_k(topk_items, true_items)
        f1 = 2 * p * r / (p + r + 1e-12)
        return f1


class DCG(object):
    """ Calculate discounted cumulative gain
    """
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        true_items = set(true_items)
        dcg = 0
        for i, item in enumerate(topk_items):
            if item in true_items:
                dcg += 1 / np.log(2 + i)
        return dcg


class NDCG(object):
    """Normalized discounted cumulative gain metric."""
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        dcg_fn = DCG(k=self.topk)
        idcg = dcg_fn(true_items[:self.topk], true_items)
        dcg = dcg_fn(topk_items, true_items)
        return dcg / (idcg + 1e-12)


class HitRate(object):
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        hit_rate = 1 if len(hit_items) > 0 else 0
        return hit_rate






















