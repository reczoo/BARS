## DCN_kuaivideo_x1

A hands-on guide to run the DCN model on the KuaiVideo_x1 dataset.

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

We use the [DCN](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/DCN) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/KuaiShou/KuaiVideo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DCN_kuaivideo_x1_tuner_config_06](./DCN_kuaivideo_x1_tuner_config_06). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DCN
    nohup python run_expid.py --config XXX/benchmarks/DCN/DCN_kuaivideo_x1_tuner_config_06 --expid DCN_kuaivideo_x1_004_4191aa76 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.665813 | 0.746080 | 0.440783  |


### Logs
```python
2022-08-22 23:33:27,843 P12558 INFO Params: {
    "batch_norm": "False",
    "batch_size": "8192",
    "data_format": "csv",
    "data_root": "../data/KuaiShou/",
    "dataset_id": "kuaivideo_x1_dc7a3035",
    "debug_mode": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.0001",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'min_categr_count': 1, 'name': 'item_emb', 'preprocess': 'copy_from(item_id)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'pos_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'neg_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'pos_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(pos_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'neg_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(neg_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_emb'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'pos_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'neg_items'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'pos_items_emb'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'neg_items_emb'}]",
    "gpu": "3",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "DCN",
    "model_id": "DCN_kuaivideo_x1_004_4191aa76",
    "model_root": "./checkpoints/DCN_kuaivideo_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_cross_layers": "3",
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
2022-08-22 23:33:27,844 P12558 INFO Set up feature processor...
2022-08-22 23:33:27,844 P12558 INFO Load feature_map from json: ../data/KuaiShou/kuaivideo_x1_dc7a3035/feature_map.json
2022-08-22 23:33:27,845 P12558 INFO Set column index...
2022-08-22 23:33:27,845 P12558 INFO Feature specs: {
    "group_id": "{'type': 'meta'}",
    "item_emb": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406}",
    "neg_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "neg_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "pos_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "pos_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10001, 'vocab_size': 10002}"
}
2022-08-22 23:33:36,584 P12558 INFO Total number of parameters: 42245697.
2022-08-22 23:33:36,585 P12558 INFO Loading data...
2022-08-22 23:33:36,585 P12558 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/train.h5
2022-08-22 23:34:00,466 P12558 INFO Train samples: total/10931092, blocks/1
2022-08-22 23:34:00,466 P12558 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/valid.h5
2022-08-22 23:34:06,342 P12558 INFO Validation samples: total/2730291, blocks/1
2022-08-22 23:34:06,342 P12558 INFO Loading train and validation data done.
2022-08-22 23:34:06,342 P12558 INFO Start training: 1335 batches/epoch
2022-08-22 23:34:06,342 P12558 INFO ************ Epoch=1 start ************
2022-08-22 23:40:50,154 P12558 INFO [Metrics] AUC: 0.730554 - gAUC: 0.642839
2022-08-22 23:40:50,165 P12558 INFO Save best model: monitor(max): 1.373393
2022-08-22 23:40:51,940 P12558 INFO --- 1335/1335 batches finished ---
2022-08-22 23:40:52,011 P12558 INFO Train loss: 0.461223
2022-08-22 23:40:52,012 P12558 INFO ************ Epoch=1 end ************
2022-08-22 23:47:32,876 P12558 INFO [Metrics] AUC: 0.734595 - gAUC: 0.650837
2022-08-22 23:47:32,888 P12558 INFO Save best model: monitor(max): 1.385432
2022-08-22 23:47:36,305 P12558 INFO --- 1335/1335 batches finished ---
2022-08-22 23:47:36,372 P12558 INFO Train loss: 0.448062
2022-08-22 23:47:36,372 P12558 INFO ************ Epoch=2 end ************
2022-08-22 23:54:17,251 P12558 INFO [Metrics] AUC: 0.738189 - gAUC: 0.654082
2022-08-22 23:54:17,259 P12558 INFO Save best model: monitor(max): 1.392271
2022-08-22 23:54:20,156 P12558 INFO --- 1335/1335 batches finished ---
2022-08-22 23:54:20,237 P12558 INFO Train loss: 0.443761
2022-08-22 23:54:20,238 P12558 INFO ************ Epoch=3 end ************
2022-08-23 00:01:04,446 P12558 INFO [Metrics] AUC: 0.739395 - gAUC: 0.656650
2022-08-23 00:01:04,454 P12558 INFO Save best model: monitor(max): 1.396045
2022-08-23 00:01:06,776 P12558 INFO --- 1335/1335 batches finished ---
2022-08-23 00:01:06,851 P12558 INFO Train loss: 0.441324
2022-08-23 00:01:06,852 P12558 INFO ************ Epoch=4 end ************
2022-08-23 00:07:44,741 P12558 INFO [Metrics] AUC: 0.740916 - gAUC: 0.657969
2022-08-23 00:07:44,753 P12558 INFO Save best model: monitor(max): 1.398885
2022-08-23 00:07:46,981 P12558 INFO --- 1335/1335 batches finished ---
2022-08-23 00:07:47,073 P12558 INFO Train loss: 0.439654
2022-08-23 00:07:47,073 P12558 INFO ************ Epoch=5 end ************
2022-08-23 00:14:23,220 P12558 INFO [Metrics] AUC: 0.742608 - gAUC: 0.659949
2022-08-23 00:14:23,234 P12558 INFO Save best model: monitor(max): 1.402556
2022-08-23 00:14:25,884 P12558 INFO --- 1335/1335 batches finished ---
2022-08-23 00:14:25,957 P12558 INFO Train loss: 0.438370
2022-08-23 00:14:25,957 P12558 INFO ************ Epoch=6 end ************
2022-08-23 00:21:08,639 P12558 INFO [Metrics] AUC: 0.741751 - gAUC: 0.658973
2022-08-23 00:21:08,647 P12558 INFO Monitor(max) STOP: 1.400724 !
2022-08-23 00:21:08,647 P12558 INFO Reduce learning rate on plateau: 0.000100
2022-08-23 00:21:08,647 P12558 INFO --- 1335/1335 batches finished ---
2022-08-23 00:21:08,741 P12558 INFO Train loss: 0.437288
2022-08-23 00:21:08,741 P12558 INFO ************ Epoch=7 end ************
2022-08-23 00:27:49,229 P12558 INFO [Metrics] AUC: 0.745324 - gAUC: 0.664693
2022-08-23 00:27:49,236 P12558 INFO Save best model: monitor(max): 1.410017
2022-08-23 00:27:51,490 P12558 INFO --- 1335/1335 batches finished ---
2022-08-23 00:27:51,600 P12558 INFO Train loss: 0.415743
2022-08-23 00:27:51,600 P12558 INFO ************ Epoch=8 end ************
2022-08-23 00:34:22,738 P12558 INFO [Metrics] AUC: 0.746080 - gAUC: 0.665813
2022-08-23 00:34:22,751 P12558 INFO Save best model: monitor(max): 1.411893
2022-08-23 00:34:25,072 P12558 INFO --- 1335/1335 batches finished ---
2022-08-23 00:34:25,135 P12558 INFO Train loss: 0.408113
2022-08-23 00:34:25,136 P12558 INFO ************ Epoch=9 end ************
2022-08-23 00:40:42,484 P12558 INFO [Metrics] AUC: 0.744519 - gAUC: 0.664558
2022-08-23 00:40:42,498 P12558 INFO Monitor(max) STOP: 1.409077 !
2022-08-23 00:40:42,498 P12558 INFO Reduce learning rate on plateau: 0.000010
2022-08-23 00:40:42,499 P12558 INFO --- 1335/1335 batches finished ---
2022-08-23 00:40:42,581 P12558 INFO Train loss: 0.403349
2022-08-23 00:40:42,582 P12558 INFO ************ Epoch=10 end ************
2022-08-23 00:47:25,269 P12558 INFO [Metrics] AUC: 0.743031 - gAUC: 0.663927
2022-08-23 00:47:25,282 P12558 INFO Monitor(max) STOP: 1.406958 !
2022-08-23 00:47:25,283 P12558 INFO Reduce learning rate on plateau: 0.000001
2022-08-23 00:47:25,283 P12558 INFO ********* Epoch==11 early stop *********
2022-08-23 00:47:25,283 P12558 INFO --- 1335/1335 batches finished ---
2022-08-23 00:47:25,357 P12558 INFO Train loss: 0.393387
2022-08-23 00:47:25,357 P12558 INFO Training finished.
2022-08-23 00:47:25,357 P12558 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DCN_kuaivideo_x1/kuaivideo_x1_dc7a3035/DCN_kuaivideo_x1_004_4191aa76.model
2022-08-23 00:47:26,738 P12558 INFO ****** Validation evaluation ******
2022-08-23 00:48:50,731 P12558 INFO [Metrics] gAUC: 0.665813 - AUC: 0.746080 - logloss: 0.440783
2022-08-23 00:48:50,985 P12558 INFO ******** Test evaluation ********
2022-08-23 00:48:50,985 P12558 INFO Loading data...
2022-08-23 00:48:50,986 P12558 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/test.h5
2022-08-23 00:48:57,112 P12558 INFO Test samples: total/2730291, blocks/1
2022-08-23 00:48:57,112 P12558 INFO Loading test data done.
2022-08-23 00:50:14,851 P12558 INFO [Metrics] gAUC: 0.665813 - AUC: 0.746080 - logloss: 0.440783

```
