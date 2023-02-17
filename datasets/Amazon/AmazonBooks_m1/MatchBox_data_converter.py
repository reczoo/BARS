import pandas as pd

user_history_dict = dict()
train_data = []
item_corpus = []
corpus_index = dict()
with open("train.txt", "r") as fid:
    for line in fid:
        splits = line.strip().split()
        user_id = splits[0]
        items = splits[1:]
        user_history_dict[user_id] = items
        for item in items:
            if item not in corpus_index:
                corpus_index[item] = len(corpus_index)
                item_corpus.append([corpus_index[item], item])
            history = user_history_dict[user_id].copy()
            history.remove(item)
            train_data.append([user_id, corpus_index[item], 1, user_id, "^".join(history)])
train = pd.DataFrame(train_data, columns=["query_index", "corpus_index", "label", "user_id", "user_history"])
print("train samples:", len(train))
train.to_csv("train.csv", index=False)

test_data = []
with open("test.txt", "r") as fid:
    for line in fid:
        splits = line.strip().split()
        user_id = splits[0]
        items = splits[1:]
        for item in items:
            if item not in corpus_index:
                corpus_index[item] = len(corpus_index)
                item_corpus.append([corpus_index[item], item])
            history = user_history_dict[user_id].copy()
            test_data.append([user_id, corpus_index[item], 1, user_id, "^".join(history)])
test = pd.DataFrame(test_data, columns=["query_index", "corpus_index", "label", "user_id", "user_history"])
print("test samples:", len(test))
test.to_csv("test.csv", index=False)

corpus = pd.DataFrame(item_corpus, columns=["corpus_index", "item_id"])
print("number of items:", len(item_corpus))
corpus = corpus.set_index("corpus_index")
corpus.to_csv("item_corpus.csv", index=False)

