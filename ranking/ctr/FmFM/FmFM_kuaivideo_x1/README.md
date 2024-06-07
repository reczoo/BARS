## FmFM_kuaivideo_x1

A hands-on guide to run the FmFM model on the KuaiVideo_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)


| [Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs) |
|:-----------------------------:|:-----------:|:--------:|:--------:|-------|
### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) Gold 6278C CPU @ 2.60GHz
  GPU: Tesla V100 32G
  RAM: 755G

  ```

+ Software

  ```python
  cuda: 10.2
  python: 3.7.10
  pytorch: 1.11.0
  pandas: 1.1.5
  numpy: 1.19.5
  scipy: 1.5.2
  sklearn: 0.22.1
  pyyaml: 6.0
  h5py: 2.8.0
  tqdm: 4.64.0
  fuxictr: 2.0.1

  ```

### Dataset
Please refer to [KuaiVideo_x1](https://github.com/reczoo/Datasets/tree/main/KuaiShou/KuaiVideo_x1) to get the dataset details.

### Code

We use the [FmFM](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/FmFM) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/KuaiShou/KuaiVideo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FmFM_kuaivideo_x1_tuner_config_02](./FmFM_kuaivideo_x1_tuner_config_02). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/FmFM
    nohup python run_expid.py --config XXX/benchmarks/FmFM/FmFM_kuaivideo_x1_tuner_config_02 --expid FmFM_kuaivideo_x1_007_a04f176e --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.655220 | 0.738873 | 0.442923  |


### Logs
```python
2022-08-22 23:34:08,462 P65950 INFO Params: {
    "batch_size": "8192",
    "data_format": "csv",
    "data_root": "../data/KuaiShou/",
    "dataset_id": "kuaivideo_x1_dc7a3035",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'min_categr_count': 1, 'name': 'item_emb', 'preprocess': 'copy_from(item_id)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'pos_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'neg_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'pos_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(pos_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'neg_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(neg_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_emb'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'pos_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'neg_items'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'pos_items_emb'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'neg_items_emb'}]",
    "field_interaction_type": "matrixed",
    "gpu": "6",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "FmFM",
    "model_id": "FmFM_kuaivideo_x1_007_a04f176e",
    "model_root": "./checkpoints/FmFM_kuaivideo_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "regularizer": "5e-05",
    "save_best_only": "True",
    "seed": "20222023",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "train_data": "../data/KuaiShou/KuaiVideo_x1/train.csv",
    "valid_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "verbose": "1"
}
2022-08-22 23:34:08,463 P65950 INFO Set up feature processor...
2022-08-22 23:34:08,463 P65950 INFO Load feature_map from json: ../data/KuaiShou/kuaivideo_x1_dc7a3035/feature_map.json
2022-08-22 23:34:08,463 P65950 INFO Set column index...
2022-08-22 23:34:08,463 P65950 INFO Feature specs: {
    "group_id": "{'type': 'meta'}",
    "item_emb": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406}",
    "neg_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "neg_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "pos_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "pos_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10001, 'vocab_size': 10002}"
}
2022-08-22 23:34:17,268 P65950 INFO Total number of parameters: 52846588.
2022-08-22 23:34:17,268 P65950 INFO Loading data...
2022-08-22 23:34:17,269 P65950 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/train.h5
2022-08-22 23:34:41,070 P65950 INFO Train samples: total/10931092, blocks/1
2022-08-22 23:34:41,070 P65950 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/valid.h5
2022-08-22 23:34:46,923 P65950 INFO Validation samples: total/2730291, blocks/1
2022-08-22 23:34:46,923 P65950 INFO Loading train and validation data done.
2022-08-22 23:34:46,924 P65950 INFO Start training: 1335 batches/epoch
2022-08-22 23:34:46,924 P65950 INFO ************ Epoch=1 start ************
2022-08-22 23:41:42,378 P65950 INFO [Metrics] AUC: 0.709005 - gAUC: 0.637929
2022-08-22 23:41:42,388 P65950 INFO Save best model: monitor(max): 1.346934
2022-08-22 23:41:44,575 P65950 INFO --- 1335/1335 batches finished ---
2022-08-22 23:41:44,650 P65950 INFO Train loss: 0.469776
2022-08-22 23:41:44,650 P65950 INFO ************ Epoch=1 end ************
2022-08-22 23:48:40,661 P65950 INFO [Metrics] AUC: 0.712526 - gAUC: 0.642866
2022-08-22 23:48:40,668 P65950 INFO Save best model: monitor(max): 1.355392
2022-08-22 23:48:43,019 P65950 INFO --- 1335/1335 batches finished ---
2022-08-22 23:48:43,109 P65950 INFO Train loss: 0.459532
2022-08-22 23:48:43,110 P65950 INFO ************ Epoch=2 end ************
2022-08-22 23:55:41,148 P65950 INFO [Metrics] AUC: 0.712141 - gAUC: 0.644711
2022-08-22 23:55:41,159 P65950 INFO Save best model: monitor(max): 1.356852
2022-08-22 23:55:43,460 P65950 INFO --- 1335/1335 batches finished ---
2022-08-22 23:55:43,552 P65950 INFO Train loss: 0.455877
2022-08-22 23:55:43,553 P65950 INFO ************ Epoch=3 end ************
2022-08-23 00:02:36,305 P65950 INFO [Metrics] AUC: 0.712978 - gAUC: 0.646714
2022-08-23 00:02:36,317 P65950 INFO Save best model: monitor(max): 1.359692
2022-08-23 00:02:38,878 P65950 INFO --- 1335/1335 batches finished ---
2022-08-23 00:02:38,963 P65950 INFO Train loss: 0.454315
2022-08-23 00:02:38,964 P65950 INFO ************ Epoch=4 end ************
2022-08-23 00:09:36,355 P65950 INFO [Metrics] AUC: 0.716150 - gAUC: 0.646586
2022-08-23 00:09:36,363 P65950 INFO Save best model: monitor(max): 1.362736
2022-08-23 00:09:38,684 P65950 INFO --- 1335/1335 batches finished ---
2022-08-23 00:09:38,762 P65950 INFO Train loss: 0.453520
2022-08-23 00:09:38,762 P65950 INFO ************ Epoch=5 end ************
2022-08-23 00:16:35,594 P65950 INFO [Metrics] AUC: 0.715935 - gAUC: 0.647314
2022-08-23 00:16:35,603 P65950 INFO Save best model: monitor(max): 1.363249
2022-08-23 00:16:37,939 P65950 INFO --- 1335/1335 batches finished ---
2022-08-23 00:16:38,030 P65950 INFO Train loss: 0.453487
2022-08-23 00:16:38,030 P65950 INFO ************ Epoch=6 end ************
2022-08-23 00:23:27,003 P65950 INFO [Metrics] AUC: 0.714579 - gAUC: 0.647739
2022-08-23 00:23:27,014 P65950 INFO Monitor(max) STOP: 1.362318 !
2022-08-23 00:23:27,014 P65950 INFO Reduce learning rate on plateau: 0.000100
2022-08-23 00:23:27,015 P65950 INFO --- 1335/1335 batches finished ---
2022-08-23 00:23:27,113 P65950 INFO Train loss: 0.452704
2022-08-23 00:23:27,113 P65950 INFO ************ Epoch=7 end ************
2022-08-23 00:30:20,829 P65950 INFO [Metrics] AUC: 0.737505 - gAUC: 0.653167
2022-08-23 00:30:20,845 P65950 INFO Save best model: monitor(max): 1.390672
2022-08-23 00:30:23,070 P65950 INFO --- 1335/1335 batches finished ---
2022-08-23 00:30:23,150 P65950 INFO Train loss: 0.423589
2022-08-23 00:30:23,150 P65950 INFO ************ Epoch=8 end ************
2022-08-23 00:37:13,096 P65950 INFO [Metrics] AUC: 0.737935 - gAUC: 0.654224
2022-08-23 00:37:13,103 P65950 INFO Save best model: monitor(max): 1.392159
2022-08-23 00:37:15,477 P65950 INFO --- 1335/1335 batches finished ---
2022-08-23 00:37:15,566 P65950 INFO Train loss: 0.418201
2022-08-23 00:37:15,567 P65950 INFO ************ Epoch=9 end ************
2022-08-23 00:43:59,108 P65950 INFO [Metrics] AUC: 0.737889 - gAUC: 0.654619
2022-08-23 00:43:59,118 P65950 INFO Save best model: monitor(max): 1.392508
2022-08-23 00:44:01,495 P65950 INFO --- 1335/1335 batches finished ---
2022-08-23 00:44:01,564 P65950 INFO Train loss: 0.415755
2022-08-23 00:44:01,564 P65950 INFO ************ Epoch=10 end ************
2022-08-23 00:51:01,671 P65950 INFO [Metrics] AUC: 0.737961 - gAUC: 0.655133
2022-08-23 00:51:01,677 P65950 INFO Save best model: monitor(max): 1.393094
2022-08-23 00:51:03,972 P65950 INFO --- 1335/1335 batches finished ---
2022-08-23 00:51:04,045 P65950 INFO Train loss: 0.414010
2022-08-23 00:51:04,046 P65950 INFO ************ Epoch=11 end ************
2022-08-23 00:58:01,965 P65950 INFO [Metrics] AUC: 0.737551 - gAUC: 0.654962
2022-08-23 00:58:01,973 P65950 INFO Monitor(max) STOP: 1.392512 !
2022-08-23 00:58:01,973 P65950 INFO Reduce learning rate on plateau: 0.000010
2022-08-23 00:58:01,974 P65950 INFO --- 1335/1335 batches finished ---
2022-08-23 00:58:02,053 P65950 INFO Train loss: 0.412453
2022-08-23 00:58:02,053 P65950 INFO ************ Epoch=12 end ************
2022-08-23 01:05:07,601 P65950 INFO [Metrics] AUC: 0.738832 - gAUC: 0.655218
2022-08-23 01:05:07,609 P65950 INFO Save best model: monitor(max): 1.394050
2022-08-23 01:05:10,018 P65950 INFO --- 1335/1335 batches finished ---
2022-08-23 01:05:10,108 P65950 INFO Train loss: 0.406695
2022-08-23 01:05:10,109 P65950 INFO ************ Epoch=13 end ************
2022-08-23 01:12:13,207 P65950 INFO [Metrics] AUC: 0.738817 - gAUC: 0.655189
2022-08-23 01:12:13,215 P65950 INFO Monitor(max) STOP: 1.394006 !
2022-08-23 01:12:13,215 P65950 INFO Reduce learning rate on plateau: 0.000001
2022-08-23 01:12:13,215 P65950 INFO --- 1335/1335 batches finished ---
2022-08-23 01:12:13,304 P65950 INFO Train loss: 0.406017
2022-08-23 01:12:13,304 P65950 INFO ************ Epoch=14 end ************
2022-08-23 01:19:13,130 P65950 INFO [Metrics] AUC: 0.738873 - gAUC: 0.655220
2022-08-23 01:19:13,138 P65950 INFO Save best model: monitor(max): 1.394092
2022-08-23 01:19:15,390 P65950 INFO --- 1335/1335 batches finished ---
2022-08-23 01:19:15,488 P65950 INFO Train loss: 0.405189
2022-08-23 01:19:15,488 P65950 INFO ************ Epoch=15 end ************
2022-08-23 01:26:19,390 P65950 INFO [Metrics] AUC: 0.738846 - gAUC: 0.655203
2022-08-23 01:26:19,398 P65950 INFO Monitor(max) STOP: 1.394049 !
2022-08-23 01:26:19,398 P65950 INFO Reduce learning rate on plateau: 0.000001
2022-08-23 01:26:19,399 P65950 INFO --- 1335/1335 batches finished ---
2022-08-23 01:26:19,479 P65950 INFO Train loss: 0.405154
2022-08-23 01:26:19,479 P65950 INFO ************ Epoch=16 end ************
2022-08-23 01:33:22,798 P65950 INFO [Metrics] AUC: 0.738876 - gAUC: 0.655217
2022-08-23 01:33:22,812 P65950 INFO Monitor(max) STOP: 1.394093 !
2022-08-23 01:33:22,812 P65950 INFO Reduce learning rate on plateau: 0.000001
2022-08-23 01:33:22,812 P65950 INFO ********* Epoch==17 early stop *********
2022-08-23 01:33:22,812 P65950 INFO --- 1335/1335 batches finished ---
2022-08-23 01:33:22,910 P65950 INFO Train loss: 0.405130
2022-08-23 01:33:22,910 P65950 INFO Training finished.
2022-08-23 01:33:22,910 P65950 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/FmFM_kuaivideo_x1/kuaivideo_x1_dc7a3035/FmFM_kuaivideo_x1_007_a04f176e.model
2022-08-23 01:33:24,092 P65950 INFO ****** Validation evaluation ******
2022-08-23 01:34:48,369 P65950 INFO [Metrics] gAUC: 0.655220 - AUC: 0.738873 - logloss: 0.442923
2022-08-23 01:34:48,553 P65950 INFO ******** Test evaluation ********
2022-08-23 01:34:48,553 P65950 INFO Loading data...
2022-08-23 01:34:48,554 P65950 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/test.h5
2022-08-23 01:34:54,974 P65950 INFO Test samples: total/2730291, blocks/1
2022-08-23 01:34:54,974 P65950 INFO Loading test data done.
2022-08-23 01:36:21,158 P65950 INFO [Metrics] gAUC: 0.655220 - AUC: 0.738873 - logloss: 0.442923

```
