## DeepFM_kuaivideo_x1

A hands-on guide to run the DeepFM model on the KuaiVideo_x1 dataset.

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

We use the [DeepFM](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/DeepFM) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/KuaiShou/KuaiVideo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DeepFM_kuaivideo_x1_tuner_config_01](./DeepFM_kuaivideo_x1_tuner_config_01). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DeepFM
    nohup python run_expid.py --config XXX/benchmarks/DeepFM/DeepFM_kuaivideo_x1_tuner_config_01 --expid DeepFM_kuaivideo_x1_003_a7784cdb --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.666502 | 0.745152 | 0.440012  |


### Logs
```python
2022-08-22 23:42:45,029 P79006 INFO Params: {
    "batch_norm": "False",
    "batch_size": "8192",
    "data_format": "csv",
    "data_root": "../data/KuaiShou/",
    "dataset_id": "kuaivideo_x1_dc7a3035",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.0001",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'min_categr_count': 1, 'name': 'item_emb', 'preprocess': 'copy_from(item_id)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'pos_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'neg_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'pos_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(pos_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'neg_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(neg_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_emb'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'pos_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'neg_items'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'pos_items_emb'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'neg_items_emb'}]",
    "gpu": "2",
    "group_id": "group_id",
    "hidden_activations": "relu",
    "hidden_units": "[1024, 512, 256]",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "DeepFM",
    "model_id": "DeepFM_kuaivideo_x1_003_a7784cdb",
    "model_root": "./checkpoints/DeepFM_kuaivideo_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "20222023",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "train_data": "../data/KuaiShou/KuaiVideo_x1/train.csv",
    "valid_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "verbose": "1"
}
2022-08-22 23:42:45,030 P79006 INFO Set up feature processor...
2022-08-22 23:42:45,030 P79006 INFO Load feature_map from json: ../data/KuaiShou/kuaivideo_x1_dc7a3035/feature_map.json
2022-08-22 23:42:45,030 P79006 INFO Set column index...
2022-08-22 23:42:45,031 P79006 INFO Feature specs: {
    "group_id": "{'type': 'meta'}",
    "item_emb": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406}",
    "neg_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "neg_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "pos_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "pos_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10001, 'vocab_size': 10002}"
}
2022-08-22 23:42:53,929 P79006 INFO Total number of parameters: 53876733.
2022-08-22 23:42:53,929 P79006 INFO Loading data...
2022-08-22 23:42:53,929 P79006 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/train.h5
2022-08-22 23:43:18,398 P79006 INFO Train samples: total/10931092, blocks/1
2022-08-22 23:43:18,398 P79006 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/valid.h5
2022-08-22 23:43:24,824 P79006 INFO Validation samples: total/2730291, blocks/1
2022-08-22 23:43:24,824 P79006 INFO Loading train and validation data done.
2022-08-22 23:43:24,824 P79006 INFO Start training: 1335 batches/epoch
2022-08-22 23:43:24,824 P79006 INFO ************ Epoch=1 start ************
2022-08-22 23:50:11,606 P79006 INFO [Metrics] AUC: 0.711041 - gAUC: 0.638866
2022-08-22 23:50:11,615 P79006 INFO Save best model: monitor(max): 1.349907
2022-08-22 23:50:13,429 P79006 INFO --- 1335/1335 batches finished ---
2022-08-22 23:50:13,494 P79006 INFO Train loss: 0.527640
2022-08-22 23:50:13,495 P79006 INFO ************ Epoch=1 end ************
2022-08-22 23:56:55,575 P79006 INFO [Metrics] AUC: 0.712922 - gAUC: 0.643608
2022-08-22 23:56:55,583 P79006 INFO Save best model: monitor(max): 1.356530
2022-08-22 23:56:58,371 P79006 INFO --- 1335/1335 batches finished ---
2022-08-22 23:56:58,452 P79006 INFO Train loss: 0.469706
2022-08-22 23:56:58,452 P79006 INFO ************ Epoch=2 end ************
2022-08-23 00:03:43,067 P79006 INFO [Metrics] AUC: 0.715357 - gAUC: 0.644898
2022-08-23 00:03:43,076 P79006 INFO Save best model: monitor(max): 1.360256
2022-08-23 00:03:45,723 P79006 INFO --- 1335/1335 batches finished ---
2022-08-23 00:03:45,803 P79006 INFO Train loss: 0.465018
2022-08-23 00:03:45,803 P79006 INFO ************ Epoch=3 end ************
2022-08-23 00:10:28,661 P79006 INFO [Metrics] AUC: 0.718692 - gAUC: 0.649469
2022-08-23 00:10:28,669 P79006 INFO Save best model: monitor(max): 1.368161
2022-08-23 00:10:31,175 P79006 INFO --- 1335/1335 batches finished ---
2022-08-23 00:10:31,267 P79006 INFO Train loss: 0.462298
2022-08-23 00:10:31,267 P79006 INFO ************ Epoch=4 end ************
2022-08-23 00:17:15,223 P79006 INFO [Metrics] AUC: 0.719305 - gAUC: 0.653200
2022-08-23 00:17:15,232 P79006 INFO Save best model: monitor(max): 1.372505
2022-08-23 00:17:17,593 P79006 INFO --- 1335/1335 batches finished ---
2022-08-23 00:17:17,680 P79006 INFO Train loss: 0.460486
2022-08-23 00:17:17,680 P79006 INFO ************ Epoch=5 end ************
2022-08-23 00:24:01,593 P79006 INFO [Metrics] AUC: 0.720856 - gAUC: 0.654169
2022-08-23 00:24:01,600 P79006 INFO Save best model: monitor(max): 1.375025
2022-08-23 00:24:03,953 P79006 INFO --- 1335/1335 batches finished ---
2022-08-23 00:24:04,046 P79006 INFO Train loss: 0.459314
2022-08-23 00:24:04,047 P79006 INFO ************ Epoch=6 end ************
2022-08-23 00:30:48,040 P79006 INFO [Metrics] AUC: 0.721226 - gAUC: 0.655862
2022-08-23 00:30:48,050 P79006 INFO Save best model: monitor(max): 1.377088
2022-08-23 00:30:50,594 P79006 INFO --- 1335/1335 batches finished ---
2022-08-23 00:30:50,695 P79006 INFO Train loss: 0.458524
2022-08-23 00:30:50,696 P79006 INFO ************ Epoch=7 end ************
2022-08-23 00:37:38,067 P79006 INFO [Metrics] AUC: 0.722881 - gAUC: 0.656351
2022-08-23 00:37:38,074 P79006 INFO Save best model: monitor(max): 1.379232
2022-08-23 00:37:40,364 P79006 INFO --- 1335/1335 batches finished ---
2022-08-23 00:37:40,456 P79006 INFO Train loss: 0.457735
2022-08-23 00:37:40,456 P79006 INFO ************ Epoch=8 end ************
2022-08-23 00:44:19,204 P79006 INFO [Metrics] AUC: 0.723122 - gAUC: 0.658426
2022-08-23 00:44:19,212 P79006 INFO Save best model: monitor(max): 1.381548
2022-08-23 00:44:21,637 P79006 INFO --- 1335/1335 batches finished ---
2022-08-23 00:44:21,808 P79006 INFO Train loss: 0.457176
2022-08-23 00:44:21,808 P79006 INFO ************ Epoch=9 end ************
2022-08-23 00:51:07,182 P79006 INFO [Metrics] AUC: 0.722970 - gAUC: 0.658097
2022-08-23 00:51:07,190 P79006 INFO Monitor(max) STOP: 1.381067 !
2022-08-23 00:51:07,190 P79006 INFO Reduce learning rate on plateau: 0.000100
2022-08-23 00:51:07,190 P79006 INFO --- 1335/1335 batches finished ---
2022-08-23 00:51:07,279 P79006 INFO Train loss: 0.456702
2022-08-23 00:51:07,280 P79006 INFO ************ Epoch=10 end ************
2022-08-23 00:57:50,220 P79006 INFO [Metrics] AUC: 0.745131 - gAUC: 0.665293
2022-08-23 00:57:50,230 P79006 INFO Save best model: monitor(max): 1.410423
2022-08-23 00:57:52,706 P79006 INFO --- 1335/1335 batches finished ---
2022-08-23 00:57:52,819 P79006 INFO Train loss: 0.419560
2022-08-23 00:57:52,819 P79006 INFO ************ Epoch=11 end ************
2022-08-23 01:04:29,484 P79006 INFO [Metrics] AUC: 0.745152 - gAUC: 0.666502
2022-08-23 01:04:29,495 P79006 INFO Save best model: monitor(max): 1.411654
2022-08-23 01:04:31,796 P79006 INFO --- 1335/1335 batches finished ---
2022-08-23 01:04:32,047 P79006 INFO Train loss: 0.410638
2022-08-23 01:04:32,047 P79006 INFO ************ Epoch=12 end ************
2022-08-23 01:11:11,859 P79006 INFO [Metrics] AUC: 0.744845 - gAUC: 0.666452
2022-08-23 01:11:11,871 P79006 INFO Monitor(max) STOP: 1.411297 !
2022-08-23 01:11:11,872 P79006 INFO Reduce learning rate on plateau: 0.000010
2022-08-23 01:11:11,872 P79006 INFO --- 1335/1335 batches finished ---
2022-08-23 01:11:11,965 P79006 INFO Train loss: 0.406566
2022-08-23 01:11:11,966 P79006 INFO ************ Epoch=13 end ************
2022-08-23 01:17:42,454 P79006 INFO [Metrics] AUC: 0.744290 - gAUC: 0.665675
2022-08-23 01:17:42,460 P79006 INFO Monitor(max) STOP: 1.409965 !
2022-08-23 01:17:42,460 P79006 INFO Reduce learning rate on plateau: 0.000001
2022-08-23 01:17:42,460 P79006 INFO ********* Epoch==14 early stop *********
2022-08-23 01:17:42,460 P79006 INFO --- 1335/1335 batches finished ---
2022-08-23 01:17:42,540 P79006 INFO Train loss: 0.395855
2022-08-23 01:17:42,540 P79006 INFO Training finished.
2022-08-23 01:17:42,540 P79006 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DeepFM_kuaivideo_x1/kuaivideo_x1_dc7a3035/DeepFM_kuaivideo_x1_003_a7784cdb.model
2022-08-23 01:17:43,668 P79006 INFO ****** Validation evaluation ******
2022-08-23 01:19:02,238 P79006 INFO [Metrics] gAUC: 0.666502 - AUC: 0.745152 - logloss: 0.440012
2022-08-23 01:19:02,421 P79006 INFO ******** Test evaluation ********
2022-08-23 01:19:02,421 P79006 INFO Loading data...
2022-08-23 01:19:02,421 P79006 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/test.h5
2022-08-23 01:19:08,912 P79006 INFO Test samples: total/2730291, blocks/1
2022-08-23 01:19:08,912 P79006 INFO Loading test data done.
2022-08-23 01:20:30,887 P79006 INFO [Metrics] gAUC: 0.666502 - AUC: 0.745152 - logloss: 0.440012

```
