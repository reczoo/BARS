""" Convert the raw `dataset.pkl` from pickle to csv format, which is
    obtained from the following paper: Li et al., Routing Micro-videos 
    via A Temporal Graph-guided Recommendation System, MM 2019.
    See https://github.com/liyongqi67/ALPINE
"""

import pickle
import numpy as np
import h5py
import pandas as pd
import hashlib


data_path = "./"
MAX_SEQ_LEN = 100  # chunk the max length of behavior sequence to 100

with open(data_path + "dataset.pkl", "rb") as f:
    train = pickle.load(f)
    test = pickle.load(f)
    pos_seq = pickle.load(f)
    neg_seq = pickle.load(f)
    pos_edge = pickle.load(f)
    neg_edge = pickle.load(f)

for part in ["train", "test"]:
    sample_list = []
    for sample in eval(part):
        user_id = sample[0][0]
        item_id = sample[0][1]
        is_click = sample[0][2]
        is_like = sample[0][3]
        is_follow = sample[0][4]
        timestamp = sample[0][5]
        pos_len = sample[1]
        neg_len = sample[2]
        pos_items = "^".join(map(str, pos_seq[user_id][0:min(pos_len, MAX_SEQ_LEN)]))
        neg_items = "^".join(map(str, neg_seq[user_id][0:min(neg_len, MAX_SEQ_LEN)]))
        sample_list.append([timestamp, user_id, item_id, is_click, is_like, is_follow, pos_items, neg_items])
    data = pd.DataFrame(sample_list, columns=["timestamp", "user_id", "item_id", "is_click", "is_like", "is_follow", "pos_items", "neg_items"])
    data.sort_values(by="timestamp", inplace=True)
    data.to_csv(f"{part}" + ".csv", index=False)

user_emb = np.load(data_path + "user_like.npy")
image_emb = np.load(data_path + "visual64_select.npy")

with h5py.File("item_visual_emb_dim64.h5", 'w') as hf:
    hf.create_dataset("key", data=list(range(len(image_emb))))
    hf.create_dataset("value", data=image_emb)

with h5py.File("user_visual_emb_dim64.h5", 'w') as hf:
    hf.create_dataset("key", data=list(range(len(user_emb))))
    hf.create_dataset("value", data=user_emb)

# Check md5sum for correctness
assert("16f13734411532cc313caf2180bfcd56" == hashlib.md5(open('train.csv', 'r').read().encode('utf-8')).hexdigest())
assert("ba26c01caaf6c65c272af11aa451fc7a" == hashlib.md5(open('test.csv', 'r').read().encode('utf-8')).hexdigest())

print("Reproducing data succeeded!")

