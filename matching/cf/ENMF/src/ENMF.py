#==============================================================
# Copied and modified based on the original source code:
# https://github.com/chenchongthu/ENMF
# Licenced to https://github.com/chenchongthu/ENMF/blob/master/LICENSE
#==============================================================

import numpy as np
import tensorflow as tf
print("tensorflow version:", tf.__version__)
import os
import pandas as pd
import scipy.sparse
import time
import sys
import argparse
from datetime import datetime
from ENMF_utils import Monitor

def parse_args():
    parser = argparse.ArgumentParser(description="Run ENMF")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name for saving result')    
    parser.add_argument('--train_data', type=str, required=True,
                        help='Training data path')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Testing data path')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.7,
                        help='dropout keep_prob')
    parser.add_argument('--negative_weight', type=float, default=0.1,
                        help='weight of non-observed data')
    parser.add_argument('--topK', nargs='+', type=int, default=[50,100,200],
                        help='topK for hr/ndcg')
    parser.add_argument('--gpu', type=str, default='-1',
                        help='GPU index')
    return parser.parse_args()


def load_data(csv_file):
    tp = pd.read_csv(csv_file, sep='\t')
    return tp


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


class ENMF:
    def __init__(self, user_num, item_num, embedding_size, max_item_pu,args):
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_size = embedding_size
        self.max_item_pu = max_item_pu
        self.weight1 = args.negative_weight
        self.lambda_bilinear = [0.0, 0.0]

    def _create_placeholders(self):
        self.input_u = tf.compat.v1.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_ur = tf.compat.v1.placeholder(tf.int32, [None, self.max_item_pu], name="input_ur")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")

    def _create_variables(self):
        self.uidW = tf.Variable(tf.compat.v1.truncated_normal(shape=[self.user_num, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="uidWg")
        self.iidW = tf.Variable(tf.compat.v1.truncated_normal(shape=[self.item_num + 1, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="iidW")

        # item domain
        self.H_i = tf.Variable(tf.constant(0.01, shape=[self.embedding_size, 1]), name="hi")

    def _create_inference(self):
        self.uid = tf.nn.embedding_lookup(self.uidW, self.input_u)
        self.uid = tf.reshape(self.uid, [-1, self.embedding_size])

        self.uid = tf.nn.dropout(self.uid, rate=1-self.dropout_keep_prob)

        self.pos_item = tf.nn.embedding_lookup(self.iidW, self.input_ur)
        self.pos_num_r = tf.cast(tf.not_equal(self.input_ur, self.item_num), 'float32')
        self.pos_item = tf.einsum('ab,abc->abc', self.pos_num_r, self.pos_item)
        self.pos_r = tf.einsum('ac,abc->abc', self.uid, self.pos_item)
        self.pos_r = tf.einsum('ajk,kl->ajl', self.pos_r, self.H_i)
        self.pos_r = tf.reshape(self.pos_r, [-1, self.max_item_pu])

    def _pre(self):
        dot = tf.einsum('ac,bc->abc', self.uid, self.iidW)
        pre = tf.einsum('ajk,kl->ajl', dot, self.H_i)
        pre = tf.reshape(pre, [-1, self.item_num + 1])
        return pre

    def _create_loss(self):
        self.loss1 = self.weight1 * tf.reduce_sum(
            tf.reduce_sum(tf.reduce_sum(tf.einsum('ab,ac->abc', self.iidW, self.iidW), 0)
                          * tf.reduce_sum(tf.einsum('ab,ac->abc', self.uid, self.uid), 0)
                          * tf.matmul(self.H_i, self.H_i, transpose_b=True), 0), 0)
        self.loss1 += tf.reduce_sum((1.0 - self.weight1) * tf.square(self.pos_r) - 2.0 * self.pos_r)
        self.l2_loss0 = tf.nn.l2_loss(self.uidW)
        self.l2_loss1 = tf.nn.l2_loss(self.iidW)
        self.loss = self.loss1 \
                    + self.lambda_bilinear[0] * self.l2_loss0 \
                    + self.lambda_bilinear[1] * self.l2_loss1

        self.reg_loss = self.lambda_bilinear[0] * self.l2_loss0 \
                        + self.lambda_bilinear[1] * self.l2_loss1

    def _build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self.pre = self._pre()


def train_step1(u_batch, y_batch, deep, sess, train_op1, args):
    """
    A single training step
    """

    feed_dict = {
        deep.input_u: u_batch,
        deep.input_ur: y_batch,
        deep.dropout_keep_prob: args.dropout,
    }
    _, loss, loss1, loss2 = sess.run(
        [train_op1, deep.loss, deep.loss1, deep.reg_loss],
        feed_dict)
    return loss, loss1, loss2


def dev_cold(u_train, i_train, test_set, train_m, test_m):
    recall100 = [[], [], [], [], []]

    ndcg100 = [[], [], [], [], []]

    user_te = [[], [], [], [], []]

    train_set = {}
    for i in range(len(u_train)):
        if train_set.has_key(u_train[i]):
            train_set[u_train[i]].append(i_train[i])
        else:
            train_set[u_train[i]] = [i_train[i]]

    for i in test_set.keys():
        if len(train_set[i]) < 9:
            user_te[0].append(i)
        elif len(train_set[i]) < 13:
            user_te[1].append(i)
        elif len(train_set[i]) < 17:
            user_te[2].append(i)
        elif len(train_set[i]) < 20:
            user_te[3].append(i)
        else:
            user_te[4].append(i)
    for l in range(len(user_te)):
        print(l)
        u = np.array(user_te[l])
        user_te2 = u[:, np.newaxis]
        ll = int(len(u) / 128) + 1
        print(u)

        for batch_num in range(ll):
            start_index = batch_num * 128
            end_index = min((batch_num + 1) * 128, len(u))
            u_batch = user_te2[start_index:end_index]

            batch_users = end_index - start_index

            feed_dict = {
                deep.input_u: u_batch,
                deep.dropout_keep_prob: 1.0,
            }

            pre = sess.run(
                deep.pre, feed_dict)

            u_b = u[start_index:end_index]

            pre = np.array(pre)
            pre = np.delete(pre, -1, axis=1)

            idx = np.zeros_like(pre, dtype=bool)
            idx[train_m[u_b].nonzero()] = True
            pre[idx] = -np.inf

            recall = []
            ndcg = []

            idx_topk_part = np.argpartition(-pre, 100, 1)
            pre_bin = np.zeros_like(pre, dtype=bool)
            pre_bin[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :100]] = True
            true_bin = np.zeros_like(pre, dtype=bool)
            true_bin[test_m[u_b].nonzero()] = True

            tmp = (np.logical_and(true_bin, pre_bin).sum(axis=1)).astype(np.float32)
            recall.append(tmp / true_bin.sum(axis=1))

            idx_topk_part = np.argpartition(-pre, 100, 1)

            topk_part = pre[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :100]]
            idx_part = np.argsort(-topk_part, axis=1)
            idx_topk = idx_topk_part[np.arange(end_index - start_index)[:, np.newaxis], idx_part]

            tp = np.log(2) / np.log(np.arange(2, 100 + 2))

            test_batch = test_m[u_b]
            DCG = (test_batch[np.arange(batch_users)[:, np.newaxis],
                              idx_topk].toarray() * tp).sum(axis=1)

            IDCG = np.array([(tp[:min(n, 100)]).sum()
                             for n in test_batch.getnnz(axis=1)])
            ndcg.append(DCG / IDCG)
            recall100[l].append(recall[0])
            ndcg100[l].append(ndcg[0])

    for l in range(len(recall100)):
        recall100[l] = np.hstack(recall100[l])
        ndcg100[l] = np.hstack(ndcg100[l])

    for l in range(len(recall100)):
        print(np.mean(recall100[l]), np.mean(ndcg100[l]))


def dev_step(test_set, train_m, test_m, deep, monitor, sess, args):
    """
    Evaluates model on a dev set
    """
    user_te = np.array(list(test_set.keys()))
    user_te2 = user_te[:, np.newaxis]

    ll = int(len(user_te) / 128) + 1

    hitrate20 = []
    hitrate50 = []
    recall20 = []
    recall50 = []
    ndcg20 = []
    ndcg50 = []

    for batch_num in range(ll):

        start_index = batch_num * 128
        end_index = min((batch_num + 1) * 128, len(user_te))
        u_batch = user_te2[start_index:end_index]

        batch_users = end_index - start_index

        feed_dict = {
            deep.input_u: u_batch,
            deep.dropout_keep_prob: 1.0,
        }

        pre = sess.run(deep.pre, feed_dict)
        u_b = user_te[start_index:end_index]
        pre = np.array(pre)
        pre = np.delete(pre, -1, axis=1)
        pre[train_m[u_b].nonzero()] = -np.inf

        recall = []
        ndcg = []
        hitrate = []
        max_topk = max(args.topK)
        idx_topk_part = np.argpartition(-pre, max_topk, 1)
        topk_part = pre[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :max_topk]]
        idx_sorted = np.argsort(-topk_part, axis=1)
        idx_max_topk = idx_topk_part[np.arange(end_index - start_index)[:, np.newaxis], idx_sorted]

        for kj in args.topK:
            idx_topk = idx_max_topk[:, 0:kj]

            ## recall & hitrate
            pre_bin = np.zeros_like(pre, dtype=bool)
            pre_bin[np.arange(batch_users)[:, np.newaxis], idx_topk] = True

            true_bin = np.zeros_like(pre, dtype=bool)
            true_bin[test_m[u_b].nonzero()] = True

            hit_table = np.logical_and(true_bin, pre_bin)
            hits = (hit_table.sum(axis=1)).astype(np.float32) # how much items hit users' preference
            user_hits = (hit_table.any(axis=1)).astype(np.float32) # if hit users
            hitrate.append(user_hits)
            recall.append(hits / (true_bin.sum(axis=1) + 1e-12))

            ## ndcg
            tp = np.log(2) / np.log(np.arange(2, kj + 2))
            test_batch = test_m[u_b]
            DCG = (test_batch[np.arange(batch_users)[:, np.newaxis],
                              idx_topk].toarray() * tp).sum(axis=1)
            IDCG = np.array([(tp[:min(n, kj)]).sum()
                             for n in test_batch.getnnz(axis=1)])
            ndcg.append(DCG / IDCG)

        hitrate20.append(hitrate[0])
        hitrate50.append(hitrate[1])
        recall20.append(recall[0])
        recall50.append(recall[1])
        ndcg20.append(ndcg[0])
        ndcg50.append(ndcg[1])

    hitrate20 = np.hstack(hitrate20)
    hitrate50 = np.hstack(hitrate50)
    recall20 = np.hstack(recall20)
    recall50 = np.hstack(recall50)
    ndcg20 = np.hstack(ndcg20)
    ndcg50 = np.hstack(ndcg50)

    early_stop = monitor.update_monitor(np.mean(hitrate20), np.mean(recall20), np.mean(ndcg20), 
                                        np.mean(hitrate50), np.mean(recall50), np.mean(ndcg50))

    return early_stop


def get_train_instances1(train_set):
    user_train, item_train = [], []
    for i in train_set.keys():
        user_train.append(i)
        item_train.append(train_set[i])

    user_train = np.array(user_train)
    item_train = np.array(item_train)
    user_train = user_train[:, np.newaxis]
    return user_train, item_train


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    random_seed = 2019
    np.random.seed(random_seed)
    tf.compat.v1.set_random_seed(random_seed)

    print('%s loading data %s' % (datetime.now(), args.train_data))
    tp_train = load_data(args.train_data)
    print('%s loading data %s' % (datetime.now(), args.test_data))
    tp_test = load_data(args.test_data)
    print("%s finish loading" % (datetime.now()))

    tp_all = tp_train.append(tp_test)
    tp_dul = pd.merge(tp_train, tp_test)
    usercount, itemcount = get_count(tp_all, 'uid'), get_count(tp_all, 'sid')
    n_users, n_items = usercount.shape[0], itemcount.shape[0]
    print("%s #users=%d, #items=%d" % (datetime.now(), n_users, n_items))

    batch_size = args.batch_size
    lr=args.lr
    embedding_size=args.embed_size
    epochs=args.epochs

    u_train = np.array(tp_train['uid'], dtype=np.int32)
    i_train = np.array(tp_train['sid'], dtype=np.int32)
    u_test = np.array(tp_test['uid'], dtype=np.int32)
    i_test = np.array(tp_test['sid'], dtype=np.int32)

    count = np.ones(len(u_train))
    train_m = scipy.sparse.csr_matrix((count, (u_train, i_train)), dtype=np.int16, shape=(n_users, n_items))
    count = np.ones(len(u_test))
    test_m = scipy.sparse.csr_matrix((count, (u_test, i_test)), dtype=np.int16, shape=(n_users, n_items))

    test_set = {}
    for i in range(len(u_test)):
        if  u_test[i] in test_set:
            test_set[u_test[i]].append(i_test[i])
        else:
            test_set[u_test[i]] = [i_test[i]]

    train_set = {}
    max_item_pu = 0
    for i in range(len(u_train)):
        if u_train[i] in train_set:
            train_set[u_train[i]].append(i_train[i])
        else:
            train_set[u_train[i]] = [i_train[i]]
    for i in train_set:
        if len(train_set[i]) > max_item_pu:
            max_item_pu = len(train_set[i])
    print("%s maximum number of items per user is %d" % (datetime.now(), max_item_pu))
    for i in train_set:
        while len(train_set[i]) < max_item_pu:
            train_set[i].append(n_items)

    print("%s begin to construct the graph" % (datetime.now()))

    log_file_name = args.dataset + "_dropout_" + str(args.dropout) + "_negative_weight_" + str(args.negative_weight) + ".csv"
    log_file = open(log_file_name, 'w')
    monitor = Monitor(log_file=log_file)
    with tf.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto()
        session_conf.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            deep = ENMF(n_users, n_items, embedding_size, max_item_pu, args)
            deep._build_graph()
            optimizer1 = tf.compat.v1.train.AdagradOptimizer(learning_rate=lr, initial_accumulator_value=1e-8).minimize(
                deep.loss)
            train_op1 = optimizer1

            sess.run(tf.compat.v1.global_variables_initializer())

            user_train1, item_train1 = get_train_instances1(train_set)

            for epoch in range(epochs):
                print('\n%s run epoch %d:' % (datetime.now(), epoch + 1), end=" ")
                start_t = time.time()

                shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
                user_train1 = user_train1[shuffle_indices]
                item_train1 = item_train1[shuffle_indices]

                ll = int(len(user_train1) / batch_size)
                loss = [0.0, 0.0, 0.0]

                for batch_num in range(ll):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, len(user_train1))

                    u_batch = user_train1[start_index:end_index]
                    i_batch = item_train1[start_index:end_index]

                    loss1, loss2, loss3 = train_step1(u_batch, i_batch, deep, sess, train_op1, args)
                    loss[0] += loss1
                    loss[1] += loss2
                    loss[2] += loss3
                print(' time=%.2fs' % (time.time() - start_t))
                print('%s loss=%.4lf, loss_no_reg=%.4lf, loss_reg=%.4lf' % 
                      (datetime.now(), loss[0] / ll, loss[1] / ll, loss[2] / ll))

                if epoch < epochs:
                    if epoch % args.verbose == 0:
                        print('\n%s evaluate at epoch %d' % (datetime.now(), epoch + 1))
                        early_stop = dev_step(test_set, train_m, test_m, deep, monitor, sess, args)
                        if early_stop:
                            print('%s early stop at epoch %d' % (datetime.now(), epoch + 1))
                            break
    log_file.close()
