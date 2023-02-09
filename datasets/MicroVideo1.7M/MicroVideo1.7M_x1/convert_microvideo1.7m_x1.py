import pandas as pd
import h5py
import sys
from collections import defaultdict
import numpy as np
import hashlib
from sklearn.decomposition import PCA



sequence_maxlen = 128

train = pd.read_csv("train_data.csv", dtype=object)
print("train.shape", train.shape)
# print(train.columns)
train = train.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)
item_ID = sorted(list(train["item_id"].unique()))
user_ID = sorted(list(train["user_id"].unique()))
print("Number of users: ", len(user_ID))
print("Number of items: ", len(item_ID))

clicked_items_queue = defaultdict(list)
clicked_categories_queue = defaultdict(list)
clicked_items_list = []
clicked_categories_list = []
click_time = ""
for idx, row in train.iterrows():
    if idx % 10000 == 0:
        print("Processing {} lines".format(idx))
    click_time = row['timestamp']
    user_id = row["user_id"]
    item_id = row["item_id"]
    cate_id = row["cate_id"]
    click = row['is_click']
    click_history = clicked_items_queue[user_id]
    if len(click_history) > sequence_maxlen:
        click_history = click_history[-sequence_maxlen:]
        clicked_items_queue[user_id] = click_history
    clicked_items_list.append("^".join(click_history))
    category_history = clicked_categories_queue[user_id]
    if len(category_history) > sequence_maxlen:
        category_history = category_history[-sequence_maxlen:]
        clicked_categories_queue[user_id] = category_history
    clicked_categories_list.append("^".join(category_history))
    if click == "1":
        clicked_items_queue[user_id].append(item_id)
        clicked_categories_queue[user_id].append(cate_id)

train["clicked_items"] = clicked_items_list
train["clicked_categories"] = clicked_categories_list
train.to_csv("train.csv", index=False)

test = pd.read_csv("test_data.csv", dtype=object)
print("test.shape", test.shape)
test = test.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)
test["item_id"] = test["item_id"].map(lambda x: str(len(item_ID) + int(x))) # re-map item ids of test
test_item_ID = sorted(list(test["item_id"].unique()))
test_user_ID = sorted(list(train["user_id"].unique()))
print("Number of users: ", len(test_user_ID))
print("Number of items: ", len(test_item_ID))
test["clicked_items"] = test["user_id"].map(lambda x: "^".join(clicked_items_queue[x][-sequence_maxlen:]))
test["clicked_categories"] = test["user_id"].map(lambda x: "^".join(clicked_categories_queue[x][-sequence_maxlen:]))
test.to_csv("test.csv", index=False)

# Embedding dimension reduction via PCA
train_emb = np.load("train_cover_image_feature.npy")
test_emb = np.load("test_cover_image_feature.npy")
item_emb = np.vstack([train_emb, test_emb])
pca = PCA(n_components=64)
item_emb = pca.fit_transform(item_emb)
print("item_emb.shape", item_emb.shape)

with h5py.File("item_image_emb_dim64.h5", 'w') as hf:
    hf.create_dataset("key", data=list(range(len(item_emb))))
    hf.create_dataset("value", data=item_emb)

# Check md5sum for correctness
assert("936e6612714c887e76226a60829b4e0a" == hashlib.md5(open('train.csv', 'r').read().encode('utf-8')).hexdigest())
assert("9417a18304fb62411ac27c26c5e0de56" == hashlib.md5(open('test.csv', 'r').read().encode('utf-8')).hexdigest())

print("Reproducing data succeeded!")
