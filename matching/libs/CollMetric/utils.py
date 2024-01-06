from collections import defaultdict

import numpy as np
from scipy.sparse import dok_matrix, lil_matrix
from tqdm import tqdm


def citeulike(tag_occurence_thres=10):
    user_dict = defaultdict(set)
    for u, item_list in enumerate(open("citeulike-t/users.dat").readlines()):
        items = item_list.strip().split(" ")
        # ignore the first element in each line, which is the number of items the user liked. 
        for item in items[1:]:
            user_dict[u].add(int(item))

    n_users = len(user_dict)
    n_items = max([item for items in user_dict.values() for item in items]) + 1

    user_item_matrix = dok_matrix((n_users, n_items), dtype=np.int32)
    for u, item_list in enumerate(open("citeulike-t/users.dat").readlines()):
        items = item_list.strip().split(" ")
        # ignore the first element in each line, which is the number of items the user liked. 
        for item in items[1:]:
            user_item_matrix[u, int(item)] = 1

    n_features = 0
    for l in open("citeulike-t/tag-item.dat").readlines():
        items = l.strip().split(" ")
        if len(items) >= tag_occurence_thres:
            n_features += 1
    print("{} features over tag_occurence_thres ({})".format(n_features, tag_occurence_thres))
    features = dok_matrix((n_items, n_features), dtype=np.int32)
    feature_index = 0
    for l in open("citeulike-t/tag-item.dat").readlines():
        items = l.strip().split(" ")
        if len(items) >= tag_occurence_thres:
            features[[int(i) for i in items], feature_index] = 1
            feature_index += 1

    return user_item_matrix, features


def split_data(user_item_matrix, split_ratio=(3, 1, 1), seed=1):
    # set the seed to have deterministic results
    np.random.seed(seed)
    train = dok_matrix(user_item_matrix.shape)
    validation = dok_matrix(user_item_matrix.shape)
    test = dok_matrix(user_item_matrix.shape)
    # convert it to lil format for fast row access
    user_item_matrix = lil_matrix(user_item_matrix)
    for user in tqdm(range(user_item_matrix.shape[0]), desc="Split data into train/valid/test"):
        items = list(user_item_matrix.rows[user])
        if len(items) >= 5:

            np.random.shuffle(items)

            train_count = int(len(items) * split_ratio[0] / sum(split_ratio))
            valid_count = int(len(items) * split_ratio[1] / sum(split_ratio))

            for i in items[0: train_count]:
                train[user, i] = 1
            for i in items[train_count: train_count + valid_count]:
                validation[user, i] = 1
            for i in items[train_count + valid_count:]:
                test[user, i] = 1
    print("{}/{}/{} train/valid/test samples".format(
        len(train.nonzero()[0]),
        len(validation.nonzero()[0]),
        len(test.nonzero()[0])))
    return train, validation, test
