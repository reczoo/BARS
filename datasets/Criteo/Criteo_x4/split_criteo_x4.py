import numpy as np
import pandas as pd
import hashlib
from sklearn.model_selection import StratifiedKFold

"""
NOTICE: We found that even though we fix the random seed, the resulting data split can be different 
due to the potential StratifiedKFold API change in different scikit-learn versions. For 
reproduciblity, `sklearn==0.19.1` is required. We use the python environement by installing 
`Anaconda3-5.2.0-Linux-x86_64.sh`.
"""

RANDOM_SEED = 2018 # Fix seed for reproduction
cols = ['Label']
for i in range(1, 14):
    cols.append('I' + str(i))
for i in range(1, 27):
    cols.append('C' + str(i))

ddf = pd.read_csv('dac/train.txt', sep='\t', header=None, names=cols, encoding='utf-8', dtype=object)
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
print('Postive ratio:', np.sum(y) / len(y))

# Check md5sum for correctness
assert("4a53bb7cbc0e4ee25f9d6a73ed824b1a" == hashlib.md5(open('train.csv', 'r').read().encode('utf-8')).hexdigest())
assert("fba5428b22895016e790e2dec623cb56" == hashlib.md5(open('valid.csv', 'r').read().encode('utf-8')).hexdigest())
assert("cfc37da0d75c4d2d8778e76997df2976" == hashlib.md5(open('test.csv', 'r').read().encode('utf-8')).hexdigest())

print("Reproducing data succeeded!")

