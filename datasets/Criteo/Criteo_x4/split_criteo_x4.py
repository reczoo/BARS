import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


RANDOM_SEED = 2018 # Set seed for reproduction
cols = ['Label']
for i in range(1, 14):
    cols.append('I' + str(i))
for i in range(1, 27):
    cols.append('C' + str(i))

ddf = pd.read_csv('raw/train.txt', sep='\t', header=None, names=cols, dtype=object, encoding='utf-8')
X = ddf.values
y = ddf['Label'].map(lambda x: float(x)).values
print(str(len(X)) + ' lines in total')

folds = StratifiedKFold(n_splits=10, shuffle=True,
                        random_state=RANDOM_SEED).split(X, y)

fold_indexes = []
for train_id, valid_id in folds:
    fold_indexes.append(valid_id)
test_index = fold_indexes[0]
valid_index = fold_indexes[1]
train_index = np.concatenate(fold_indexes[2:])

test_df = ddf.loc[test_index, :]
test_df.to_csv('test.csv', index=False, encoding='utf-8')
valid_df = ddf.loc[valid_index, :]
valid_df.to_csv('valid.csv', index=False, encoding='utf-8')
ddf.loc[train_index, :].to_csv('train.csv', index=False, encoding='utf-8')

print('Train lines:', len(train_index))
print('Validation lines:', len(valid_index))
print('Test lines:', len(test_index))
print('Postive samples:', np.sum(y))
print('Postive ratio:', np.sum(y) / len(y))
print('Postive test:', np.sum(test_df['Label'].map(lambda x: float(x))))
print('Postive validation:', np.sum(valid_df['Label'].map(lambda x: float(x))))

