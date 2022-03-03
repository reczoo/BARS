# Convert libsvm data from AFN [AAAI'2020] to csv format

import pandas as pd
from pathlib import Path
import gc

headers = ["label", "user_id", "item_id", "tag_id"]

data_files = ["train.libsvm", "valid.libsvm", "test.libsvm"]
for f in data_files:
    df = pd.read_csv(f, sep=" ", names=headers)
    for col in headers[1:]:
        df[col] = df[col].apply(lambda x: x.split(':')[0])
    df.to_csv(Path(f).stem + ".csv", index=False)
    del df
    gc.collect()