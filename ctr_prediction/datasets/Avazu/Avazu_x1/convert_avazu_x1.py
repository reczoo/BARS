""" Convert libsvm data from AFN paper to csv format """
import pandas as pd
from pathlib import Path
import gc
import hashlib

headers = ["label", "feat_1", "feat_2", "feat_3", "feat_4", "feat_5", "feat_6", "feat_7", "feat_8", "feat_9", "feat_10",
           "feat_11", "feat_12", "feat_13", "feat_14", "feat_15", "feat_16", "feat_17", "feat_18", "feat_19", "feat_20", "feat_21", "feat_22"]

data_files = ["train.libsvm", "valid.libsvm", "test.libsvm"]
for f in data_files:
    df = pd.read_csv(f, sep=" ", names=headers)
    for col in headers[1:]:
        df[col] = df[col].apply(lambda x: x.split(':')[0])
    df.to_csv(Path(f).stem + ".csv", index=False)
    del df
    gc.collect()


# Check md5sum for correctness
assert("f1114a07aea9e996842c71648e0f6395" == hashlib.md5(open('train.csv', 'r').read().encode('utf-8')).hexdigest())
assert("d9568f246357d156c4b8030fadb8b623" == hashlib.md5(open('valid.csv', 'r').read().encode('utf-8')).hexdigest())
assert("9e2fe9c48705c9315ae7a0953eb57acf" == hashlib.md5(open('test.csv', 'r').read().encode('utf-8')).hexdigest())

print("Reproducing data succeeded!")