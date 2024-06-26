## DCNv2_kuaivideo_x1

A hands-on guide to run the DCNv2 model on the KuaiVideo_x1 dataset.

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

We use the [DCNv2](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/DCNv2) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/KuaiShou/KuaiVideo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DCNv2_kuaivideo_x1_tuner_config_01](./DCNv2_kuaivideo_x1_tuner_config_01). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DCNv2
    nohup python run_expid.py --config XXX/benchmarks/DCNv2/DCNv2_kuaivideo_x1_tuner_config_01 --expid DCNv2_kuaivideo_x1_008_7047bff3 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.667472 | 0.746953 | 0.437494  |


### Logs
```python
2022-08-23 10:23:10,790 P63058 INFO Params: {
    "batch_norm": "True",
    "batch_size": "8192",
    "data_format": "csv",
    "data_root": "../data/KuaiShou/",
    "dataset_id": "kuaivideo_x1_dc7a3035",
    "debug_mode": "False",
    "dnn_activations": "relu",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.0001",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'min_categr_count': 1, 'name': 'item_emb', 'preprocess': 'copy_from(item_id)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'pos_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'neg_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'pos_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(pos_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'neg_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(neg_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_emb'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'pos_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'neg_items'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'pos_items_emb'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'neg_items_emb'}]",
    "gpu": "7",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "low_rank": "32",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "DCNv2",
    "model_id": "DCNv2_kuaivideo_x1_008_7047bff3",
    "model_root": "./checkpoints/DCNv2_kuaivideo_x1/",
    "model_structure": "parallel",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_cross_layers": "2",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "parallel_dnn_hidden_units": "[1024, 512, 256]",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "20222023",
    "shuffle": "True",
    "stacked_dnn_hidden_units": "[500, 500, 500]",
    "task": "binary_classification",
    "test_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "train_data": "../data/KuaiShou/KuaiVideo_x1/train.csv",
    "use_low_rank_mixture": "False",
    "valid_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "verbose": "1"
}
2022-08-23 10:23:10,790 P63058 INFO Set up feature processor...
2022-08-23 10:23:10,791 P63058 INFO Load feature_map from json: ../data/KuaiShou/kuaivideo_x1_dc7a3035/feature_map.json
2022-08-23 10:23:10,791 P63058 INFO Set column index...
2022-08-23 10:23:10,791 P63058 INFO Feature specs: {
    "group_id": "{'type': 'meta'}",
    "item_emb": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406}",
    "neg_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "neg_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "pos_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "pos_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10001, 'vocab_size': 10002}"
}
2022-08-23 10:23:19,319 P63058 INFO Total number of parameters: 42648897.
2022-08-23 10:23:19,320 P63058 INFO Loading data...
2022-08-23 10:23:19,320 P63058 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/train.h5
2022-08-23 10:23:45,641 P63058 INFO Train samples: total/10931092, blocks/1
2022-08-23 10:23:45,641 P63058 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/valid.h5
2022-08-23 10:23:51,489 P63058 INFO Validation samples: total/2730291, blocks/1
2022-08-23 10:23:51,489 P63058 INFO Loading train and validation data done.
2022-08-23 10:23:51,490 P63058 INFO Start training: 1335 batches/epoch
2022-08-23 10:23:51,490 P63058 INFO ************ Epoch=1 start ************
2022-08-23 10:30:32,474 P63058 INFO [Metrics] AUC: 0.732557 - gAUC: 0.646713
2022-08-23 10:30:32,484 P63058 INFO Save best model: monitor(max): 1.379270
2022-08-23 10:30:34,692 P63058 INFO --- 1335/1335 batches finished ---
2022-08-23 10:30:34,753 P63058 INFO Train loss: 0.457925
2022-08-23 10:30:34,753 P63058 INFO ************ Epoch=1 end ************
2022-08-23 10:37:16,135 P63058 INFO [Metrics] AUC: 0.737319 - gAUC: 0.653437
2022-08-23 10:37:16,143 P63058 INFO Save best model: monitor(max): 1.390755
2022-08-23 10:37:18,545 P63058 INFO --- 1335/1335 batches finished ---
2022-08-23 10:37:18,627 P63058 INFO Train loss: 0.442745
2022-08-23 10:37:18,627 P63058 INFO ************ Epoch=2 end ************
2022-08-23 10:43:59,136 P63058 INFO [Metrics] AUC: 0.740819 - gAUC: 0.657624
2022-08-23 10:43:59,145 P63058 INFO Save best model: monitor(max): 1.398443
2022-08-23 10:44:02,606 P63058 INFO --- 1335/1335 batches finished ---
2022-08-23 10:44:02,691 P63058 INFO Train loss: 0.439787
2022-08-23 10:44:02,692 P63058 INFO ************ Epoch=3 end ************
2022-08-23 10:50:42,778 P63058 INFO [Metrics] AUC: 0.741125 - gAUC: 0.658397
2022-08-23 10:50:42,787 P63058 INFO Save best model: monitor(max): 1.399522
2022-08-23 10:50:45,070 P63058 INFO --- 1335/1335 batches finished ---
2022-08-23 10:50:45,149 P63058 INFO Train loss: 0.438697
2022-08-23 10:50:45,150 P63058 INFO ************ Epoch=4 end ************
2022-08-23 10:57:27,078 P63058 INFO [Metrics] AUC: 0.741792 - gAUC: 0.659205
2022-08-23 10:57:27,088 P63058 INFO Save best model: monitor(max): 1.400997
2022-08-23 10:57:29,577 P63058 INFO --- 1335/1335 batches finished ---
2022-08-23 10:57:29,667 P63058 INFO Train loss: 0.437830
2022-08-23 10:57:29,668 P63058 INFO ************ Epoch=5 end ************
2022-08-23 11:04:13,815 P63058 INFO [Metrics] AUC: 0.742669 - gAUC: 0.661432
2022-08-23 11:04:13,821 P63058 INFO Save best model: monitor(max): 1.404102
2022-08-23 11:04:16,451 P63058 INFO --- 1335/1335 batches finished ---
2022-08-23 11:04:16,527 P63058 INFO Train loss: 0.437171
2022-08-23 11:04:16,527 P63058 INFO ************ Epoch=6 end ************
2022-08-23 11:10:58,940 P63058 INFO [Metrics] AUC: 0.742916 - gAUC: 0.661971
2022-08-23 11:10:58,954 P63058 INFO Save best model: monitor(max): 1.404887
2022-08-23 11:11:01,201 P63058 INFO --- 1335/1335 batches finished ---
2022-08-23 11:11:01,288 P63058 INFO Train loss: 0.436651
2022-08-23 11:11:01,289 P63058 INFO ************ Epoch=7 end ************
2022-08-23 11:17:40,823 P63058 INFO [Metrics] AUC: 0.742337 - gAUC: 0.661560
2022-08-23 11:17:40,847 P63058 INFO Monitor(max) STOP: 1.403898 !
2022-08-23 11:17:40,847 P63058 INFO Reduce learning rate on plateau: 0.000100
2022-08-23 11:17:40,848 P63058 INFO --- 1335/1335 batches finished ---
2022-08-23 11:17:41,104 P63058 INFO Train loss: 0.436065
2022-08-23 11:17:41,105 P63058 INFO ************ Epoch=8 end ************
2022-08-23 11:24:18,789 P63058 INFO [Metrics] AUC: 0.746953 - gAUC: 0.667472
2022-08-23 11:24:18,799 P63058 INFO Save best model: monitor(max): 1.414425
2022-08-23 11:24:21,029 P63058 INFO --- 1335/1335 batches finished ---
2022-08-23 11:24:21,137 P63058 INFO Train loss: 0.413580
2022-08-23 11:24:21,137 P63058 INFO ************ Epoch=9 end ************
2022-08-23 11:30:57,853 P63058 INFO [Metrics] AUC: 0.746301 - gAUC: 0.667381
2022-08-23 11:30:57,864 P63058 INFO Monitor(max) STOP: 1.413683 !
2022-08-23 11:30:57,864 P63058 INFO Reduce learning rate on plateau: 0.000010
2022-08-23 11:30:57,865 P63058 INFO --- 1335/1335 batches finished ---
2022-08-23 11:30:57,962 P63058 INFO Train loss: 0.405960
2022-08-23 11:30:57,962 P63058 INFO ************ Epoch=10 end ************
2022-08-23 11:37:34,473 P63058 INFO [Metrics] AUC: 0.745758 - gAUC: 0.667272
2022-08-23 11:37:34,488 P63058 INFO Monitor(max) STOP: 1.413030 !
2022-08-23 11:37:34,488 P63058 INFO Reduce learning rate on plateau: 0.000001
2022-08-23 11:37:34,488 P63058 INFO ********* Epoch==11 early stop *********
2022-08-23 11:37:34,489 P63058 INFO --- 1335/1335 batches finished ---
2022-08-23 11:37:34,573 P63058 INFO Train loss: 0.396877
2022-08-23 11:37:34,573 P63058 INFO Training finished.
2022-08-23 11:37:34,573 P63058 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DCNv2_kuaivideo_x1/kuaivideo_x1_dc7a3035/DCNv2_kuaivideo_x1_008_7047bff3.model
2022-08-23 11:37:36,013 P63058 INFO ****** Validation evaluation ******
2022-08-23 11:38:58,530 P63058 INFO [Metrics] gAUC: 0.667472 - AUC: 0.746953 - logloss: 0.437494
2022-08-23 11:38:58,699 P63058 INFO ******** Test evaluation ********
2022-08-23 11:38:58,699 P63058 INFO Loading data...
2022-08-23 11:38:58,699 P63058 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/test.h5
2022-08-23 11:39:05,669 P63058 INFO Test samples: total/2730291, blocks/1
2022-08-23 11:39:05,669 P63058 INFO Loading test data done.
2022-08-23 11:40:28,774 P63058 INFO [Metrics] gAUC: 0.667472 - AUC: 0.746953 - logloss: 0.437494

```
