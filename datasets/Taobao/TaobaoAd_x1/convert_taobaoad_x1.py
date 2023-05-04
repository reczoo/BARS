# -*- coding: utf-8 -*-
"""
The raw dataset is available at https://tianchi.aliyun.com/dataset/56
The preprocessed dataset is used by the following work: 
Lyu et al., Deep Match to Rank Model for Personalized Click-Through Rate Prediction, AAAI 2020.
The preprocessing steps follow the scripts at https://aistudio.baidu.com/aistudio/projectdetail/1805731
The required data `dataset_full.zip` can be download at https://aistudio.baidu.com/aistudio/datasetdetail/81892
However, we note that the ID mapping of categorical features in `dataset_full.zip` has a buggy issue. 
Please refer to https://github.com/PaddlePaddle/PaddleRec/issues/821
Thus, we need to re-map the categorical IDs to new indices when using this dataset.
"""

import pandas as pd
import hashlib


train_path = "./work/train_sorted.csv"
test_path = "./work/test.csv"

data_parts = ["train", "test"]
for part in data_parts:
    data_df = pd.read_csv(eval(part + '_path'), header=None, dtype=object) 
    data_df.fillna("0", inplace=True)
    part_df = pd.DataFrame()
    part_df["clk"] = data_df.iloc[:, 266]
    part_df["btag_his"] = ["^".join(filter(lambda k: k != "0", x.tolist())) for x in data_df.iloc[:, 0:50].values]
    part_df["cate_his"] = ["^".join(filter(lambda k: k != "0", x.tolist())) for x in data_df.iloc[:, 50:100].values]
    part_df["brand_his"] = ["^".join(filter(lambda k: k != "0", x.tolist())) for x in data_df.iloc[:, 100:150].values]
    part_df["userid"] = data_df.iloc[:, 250]
    part_df["cms_segid"] = data_df.iloc[:, 251]
    part_df["cms_group_id"] = data_df.iloc[:, 252]
    part_df["final_gender_code"] = data_df.iloc[:, 253]
    part_df["age_level"] = data_df.iloc[:, 254]
    part_df["pvalue_level"] = data_df.iloc[:, 255]
    part_df["shopping_level"] = data_df.iloc[:, 256]
    part_df["occupation"] = data_df.iloc[:, 257]
    part_df["new_user_class_level"] = data_df.iloc[:, 258]
    part_df["adgroup_id"] = data_df.iloc[:, 259]
    part_df["cate_id"] = data_df.iloc[:, 260]
    part_df["campaign_id"] = data_df.iloc[:, 261]
    part_df["customer"] = data_df.iloc[:, 262]
    part_df["brand"] = data_df.iloc[:, 263]
    part_df["price"] = data_df.iloc[:, 264] 
    part_df["pid"] = data_df.iloc[:, 265]
    part_df["btag"] = [1] * len(data_df)
    part_df.to_csv(part + ".csv", index=False)

# Check md5sum for correctness
assert("eaabfc8629f23519b04593e26c7522fc" == hashlib.md5(open('train.csv', 'r').read().encode('utf-8')).hexdigest())
assert("f5ae6197e52385496d46e2867c1c8da1" == hashlib.md5(open('test.csv', 'r').read().encode('utf-8')).hexdigest())

print("Reproducing data succeeded!")

