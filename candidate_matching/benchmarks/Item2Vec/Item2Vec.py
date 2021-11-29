import sys
sys.path.append("../../external/daisyRec/")
import os
import time
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
from os.path import join
from daisy.utils.metrics import RecallPrecision_ATk,MRRatK_r,NDCGatK_r, HRK_r
from daisy.model.Item2VecRecommender import Item2Vec
from daisy.utils.data import item2vec_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple Baseline')
    parser.add_argument('--dataset', type=str, default='Gowalla')
    parser.add_argument('--topk', type=str, default='[20, 50]')
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--n_negs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--gpu_id', type=int, default=0)
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
    for line in train_data:
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

    with open(TEST_PATH, 'r') as f:
        test_data = f.readlines()
    
    test_set = {'user':[], 'item':[], 'rating':[]}
    test_ui = {}
    for line in test_data:
        line = line.strip().split(' ')
        user = int(line[0])
        test_ui[user] = []
        for iid in line[1:]:
            iid = int(iid)
            test_set['user'].append(user)
            test_set['item'].append(iid)
            test_set['rating'].append(1.0)

            if iid not in item_set:
                item_num += 1
                item_set.add(iid)
            test_ui[user].append(iid)

    test_set = pd.DataFrame(test_set)
    train_loader, vocab_size, item2idx = item2vec_data(train_set, test_set, args.window, item_num, args.batch_size)
    
    model = Item2Vec(item2idx, vocab_size, factors = args.emb_dim, epochs = args.epochs, n_negs = args.n_negs, use_cuda=True, gpu_id=args.gpu_id)
    
    print('model fitting...')
    model.fit(train_loader)

    model.build_user_vec(interacted_items)
    #model.build_user_vec(test_ui)

    topks = eval(args.topk) 
    max_k = max(topks)
    
    r = np.zeros((len(test_data), max_k))
    ground_truth = []
    hits = 0

    for i, line in enumerate(test_data):
        st = time.time()
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
        ter = time.time()
        if i % 2000 == 0:
            print('{}, {} ok, hit = {}'.format(datetime.now(), i, hits))
    
    metric_results = ""
    for topk in topks:
        hit_rate = HRK_r(r, topk) / len(test_data)
        res = RecallPrecision_ATk(ground_truth, r, topk)
        recall = res['recall'] / len(test_data)
        precision = res['precision'] / len(test_data)
        ndcg = NDCGatK_r(ground_truth, r, topk) / len(test_data)
        metric_results += 'HR@{} = {}, Recall@{} = {}, NDCG@{} = {}, '.format(topk, hit_rate, topk, recall, topk, ndcg)
    
    print(metric_results)
    
    with open("./{}_experiment_result.csv".format(args.dataset), "a+") as fout:
        print(datetime.now(), args.window, args.emb_dim, args.n_negs, args.batch_size, args.epochs, metric_results, sep=",", file=fout)
