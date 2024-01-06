"""
@author: Zhongchuan Sun
"""
import itertools
import numpy as np
import sys
import heapq
from concurrent.futures import ThreadPoolExecutor

def argmax_top_k(a, top_k=50):
    ele_idx = heapq.nlargest(top_k, zip(a, itertools.count()))
    return np.array([idx for ele, idx in ele_idx], dtype=np.intc)

def precision(rank, ground_truth):
    hits = [1 if item in ground_truth else 0 for item in rank]
    result = np.cumsum(hits, dtype=np.float)/np.arange(1, len(rank)+1)
    return result


def recall(rank, ground_truth):
    hits = [1 if item in ground_truth else 0 for item in rank]
    result = np.cumsum(hits, dtype=np.float) / len(ground_truth)
    return result

'''
----------
 Edit by liyi
----------
'''
def hitrate(rank, ground_truth):
    # 如果模型为当前用户生成的推荐列表中有一个物品是属于用户真实点击过的，那么当前用户就算被hit了
    hits = [1 if item in ground_truth else 0 for item in rank]
    # 考虑1不存在的边界条件，加入一个1
    n_hits = hits + [1]
    ind = n_hits.index(1)
    # 一旦找到一个属于真实数据的物品，又由于这是排序好的，那么之后的top_k推荐该用户都属于被hit了
    result = [0]*ind + [1]*(len(hits)-ind)
    result = np.array(result)
    return result

def map(rank, ground_truth):
    pre = precision(rank, ground_truth)
    pre = [pre[idx] if item in ground_truth else 0 for idx, item in enumerate(rank)]
    sum_pre = np.cumsum(pre, dtype=np.float32)
    gt_len = len(ground_truth)
    # len_rank = np.array([min(i, gt_len) for i in range(1, len(rank)+1)])
    result = sum_pre/gt_len
    return result


def ndcg(rank, ground_truth):
    len_rank = len(rank)
    len_gt = len(ground_truth)
    idcg_len = min(len_gt, len_rank)

    # calculate idcg
    idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
    idcg[idcg_len:] = idcg[idcg_len-1]

    # idcg = np.cumsum(1.0/np.log2(np.arange(2, len_rank+2)))
    dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
    result = dcg/idcg
    return result


def mrr(rank, ground_truth):
    last_idx = sys.maxsize
    for idx, item in enumerate(rank):
        if item in ground_truth:
            last_idx = idx
            break
    result = np.zeros(len(rank), dtype=np.float32)
    result[last_idx:] = 1.0/(last_idx+1)
    return result


def eval_score_matrix_foldout(score_matrix, test_items, top_k=50, thread_num=None):
    def _eval_one_user(idx):
        scores = score_matrix[idx]  # all scores of the test user
        test_item = test_items[idx]  # all test items of the test user

        ranking = argmax_top_k(scores, top_k)  # Top-K items
        result = []
        hr = []
        result.extend(precision(ranking, test_item))
        result.extend(recall(ranking, test_item))
        result.extend(hitrate(ranking, test_item))
        result.extend(map(ranking, test_item))
        result.extend(ndcg(ranking, test_item))
        result.extend(mrr(ranking, test_item))
        
        result = np.array(result, dtype=np.float32).flatten()
        return result

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        batch_result = executor.map(_eval_one_user, range(len(test_items)))
    '''
        ret_it = executor.map(_eval_one_user, range(len(test_items)))
        print("map")
        print(ret_it)
        print(type(ret_it))
        batch_result, batch_hr = ret_it.result()

    batch_result = []
    batch_hr = []
    for item in ret_it:
        batch_result.append(ret_it[0])
        batch_hr.append(ret_it[1])

    result = list(batch_result)  # generator to list
    hr = list(batch_hr)
    '''
    result = list(batch_result)  # generator to list
    return np.array(result)  # list to ndarray
