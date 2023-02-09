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
ddf = pd.read_csv('train/train.csv', encoding='utf-8', dtype=object)
X = ddf.values
y = ddf['click'].map(lambda x: float(x)).values
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
assert("de3a27264cdabf66adf09df82328ccaa" == hashlib.md5(open('train.csv', 'r').read().encode('utf-8')).hexdigest())
assert("33232931d84d6452d3f956e936cab2c9" == hashlib.md5(open('valid.csv', 'r').read().encode('utf-8')).hexdigest())
assert("3ebb774a9ca74d05919b84a3d402986d" == hashlib.md5(open('test.csv', 'r').read().encode('utf-8')).hexdigest())

print("Reproducing data succeeded!")


    
