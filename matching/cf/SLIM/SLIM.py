import sys
sys.path.append("../../external/daisyRec/")
import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from os.path import join
from datetime import datetime
from daisy.utils.loader import load_rate, get_ur, convert_npy_mat, build_candidates_set
from daisy.utils.metrics import RecallPrecision_ATk, MRRatK_r, NDCGatK_r, HRK_r
from daisy.model.SLiMRecommender import SLIM


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple Baseline')
    parser.add_argument('--dataset', type=str, default='Gowalla')
    parser.add_argument('--topk', type=str, default='[20, 50]')
    parser.add_argument('--l1', type=float, default=1e-6)
    parser.add_argument('--alpha', type=float, default=1e-2)
    args = parser.parse_args()

    print(args)

    # load data
    TRAIN_PATH = join('../../data/' + args.dataset + '/' + args.dataset.lower() + '_x0', 'train.txt')
    TEST_PATH = join('../../data/' + args.dataset + '/' + args.dataset.lower() + '_x0', 'test.txt')

    # item popularities
    with open(TRAIN_PATH, 'r') as f:
        train_data = f.readlines()
    
    train_set = {'user':[], 'item':[], 'rating':[]}
    user_num, item_num = 0, 0
    item_set = set()
    interacted_items = {}
    for i, line in enumerate(train_data):
        line = line.strip().split(' ')
        user = int(line[0])
        interacted_items[user] = set()
        for iid in line[1:]:
            iid = int(iid)
            train_set['user'].append(user)
            train_set['item'].append(iid)
            train_set['rating'].append(1.0)

            if iid not in item_set:
                item_num += 1
                item_set.add(iid)
            interacted_items[user].add(iid)

        user_num += 1
        

    train_set = pd.DataFrame(train_set)
    model = SLIM(user_num, item_num, l1_ratio=args.l1, alpha=args.alpha)

    print('model fitting...')
    model.fit(train_set)

    print('Generate recommend list...')

    
    with open(TEST_PATH, 'r') as f:
        test_data = f.readlines()
    
    topks = eval(args.topk)
    max_k = max(topks)
    
    r = np.zeros((len(test_data), max_k))
    ground_truth = []
    hits = 0

    for i, line in enumerate(test_data):
        line = line.strip().split(' ')
        u = int(line[0])
        ground_truth_iids = [int(iid) for iid in line[1:]]
        ground_truth_set = set(ground_truth_iids)
        ground_truth.append(ground_truth_iids)
        pred_rates = []
        for iid in range(item_num):
            if iid in interacted_items[u]:
                pred_rates.append(-10000)
            else:
                pred_rates.append(model.predict(u, iid))
        rec_idx = np.argsort(pred_rates)[::-1][:max_k]
        for j, idx in enumerate(rec_idx):
            if idx in ground_truth_set:
                r[i][j] = 1
                hits += 1
        if i % 2000 == 0:
            print('{} ok, hit = {}'.format(i, hits))

    metric_results = ""
    for topk in topks:
        hit_rate = HRK_r(r, topk) / len(test_data)
        res = RecallPrecision_ATk(ground_truth, r, topk)
        recall = res['recall'] / len(test_data)
        precision = res['precision'] / len(test_data)
        ndcg = NDCGatK_r(ground_truth, r, topk) / len(test_data)
        metric_results += 'HR@{} = {}, Recall@{} = {}, NDCG@{} = {}, '.format(topk, hit_rate, topk, recall, topk, ndcg)
    
    with open("./{}_experiment_result.csv".format(args.dataset), "a+") as fout:
        print(datetime.now(), args.alpha, args.l1, metric_results, sep=",", file=fout)
    print(metric_results)
    print('Finished')
