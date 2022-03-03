import pickle
import pandas as pd

# cat aa ab ac > dataset.pkl from https://github.com/zhougr1993/DeepInterestNetwork

with open('dataset.pkl', 'rb') as f:
    train_set = pickle.load(f, encoding='bytes')
    test_set = pickle.load(f, encoding='bytes')
    cate_list = pickle.load(f, encoding='bytes')
    user_count, item_count, cate_count = pickle.load(f, encoding='bytes')

train_data = []
for sample in train_set:
    user_id = sample[0]
    item_id = sample[2]
    item_history = "^".join([str(i) for i in sample[1]])
    label = sample[3]
    cate_id = cate_list[item_id]
    cate_history = "^".join([str(i) for i in cate_list[sample[1]]])
    train_data.append([label, user_id, item_id, cate_id, item_history, cate_history])
train_df = pd.DataFrame(train_data, columns=['label', 'user_id', 'item_id', 'cate_id', 'item_history', 'cate_history'])
train_df.to_csv("train.csv", index=False)

test_data = []
for sample in test_set:
    user_id = sample[0]
    item_pair = sample[2]
    item_history = "^".join([str(i) for i in sample[1]])
    cate_history = "^".join([str(i) for i in cate_list[sample[1]]])
    test_data.append([1, user_id, item_pair[0], cate_list[item_pair[0]], item_history, cate_history])
    test_data.append([0, user_id, item_pair[1], cate_list[item_pair[1]], item_history, cate_history])
test_df = pd.DataFrame(test_data, columns=['label', 'user_id', 'item_id', 'cate_id', 'item_history', 'cate_history'])
test_df.to_csv("test.csv", index=False)

