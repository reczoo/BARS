## DNN_kuaivideo_x1

A hands-on guide to run the DNN model on the KuaiVideo_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)


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
Please refer to the BARS dataset [KuaiVideo_x1](https://github.com/openbenchmark/BARS/blob/main/datasets/KuaiShou#KuaiVideo_x1) to get data ready.

### Code

We use the [DNN](https://github.com/xue-pai/FuxiCTR/blob/v2.0.1/model_zoo/DNN) model code from [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/KuaiShou/KuaiVideo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DNN_kuaivideo_x1_tuner_config_01](./DNN_kuaivideo_x1_tuner_config_01). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DNN
    nohup python run_expid.py --config XXX/benchmarks/DNN/DNN_kuaivideo_x1_tuner_config_01 --expid DNN_kuaivideo_x1_023_2da45570 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.665842 | 0.745264 | 0.440147  |


### Logs
```python
2022-11-19 00:07:32,702 P100588 INFO Params: {
    "batch_norm": "False",
    "batch_size": "8192",
    "data_format": "csv",
    "data_root": "../data/KuaiShou/",
    "dataset_id": "kuaivideo_x1_dc7a3035",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "5e-05",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'min_categr_count': 1, 'name': 'item_emb', 'preprocess': 'copy_from(item_id)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'pos_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'neg_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'pos_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(pos_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'neg_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(neg_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_emb'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'pos_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'neg_items'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'pos_items_emb'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'neg_items_emb'}]",
    "gpu": "7",
    "group_id": "group_id",
    "hidden_activations": "relu",
    "hidden_units": "[1024, 512, 256]",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "DNN",
    "model_id": "DNN_kuaivideo_x1_023_2da45570",
    "model_root": "./checkpoints/DNN_kuaivideo_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
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
2022-11-19 00:07:32,703 P100588 INFO Set up feature processor...
2022-11-19 00:07:32,704 P100588 INFO Load feature_map from json: ../data/KuaiShou/kuaivideo_x1_dc7a3035/feature_map.json
2022-11-19 00:07:32,704 P100588 INFO Set column index...
2022-11-19 00:07:32,704 P100588 INFO Feature specs: {
    "group_id": "{'type': 'meta'}",
    "item_emb": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406}",
    "neg_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "neg_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "pos_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "pos_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10001, 'vocab_size': 10002}"
}
2022-11-19 00:07:40,419 P100588 INFO Total number of parameters: 42242561.
2022-11-19 00:07:40,420 P100588 INFO Loading data...
2022-11-19 00:07:40,420 P100588 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/train.h5
2022-11-19 00:08:06,277 P100588 INFO Train samples: total/10931092, blocks/1
2022-11-19 00:08:06,277 P100588 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/valid.h5
2022-11-19 00:08:12,586 P100588 INFO Validation samples: total/2730291, blocks/1
2022-11-19 00:08:12,586 P100588 INFO Loading train and validation data done.
2022-11-19 00:08:12,586 P100588 INFO Start training: 1335 batches/epoch
2022-11-19 00:08:12,586 P100588 INFO ************ Epoch=1 start ************
2022-11-19 00:14:49,279 P100588 INFO [Metrics] AUC: 0.728291 - gAUC: 0.639675
2022-11-19 00:14:49,289 P100588 INFO Save best model: monitor(max): 1.367966
2022-11-19 00:14:51,099 P100588 INFO --- 1335/1335 batches finished ---
2022-11-19 00:14:51,214 P100588 INFO Train loss: 0.456392
2022-11-19 00:14:51,214 P100588 INFO ************ Epoch=1 end ************
2022-11-19 00:21:12,525 P100588 INFO [Metrics] AUC: 0.735177 - gAUC: 0.649684
2022-11-19 00:21:12,535 P100588 INFO Save best model: monitor(max): 1.384861
2022-11-19 00:21:15,001 P100588 INFO --- 1335/1335 batches finished ---
2022-11-19 00:21:15,091 P100588 INFO Train loss: 0.447351
2022-11-19 00:21:15,092 P100588 INFO ************ Epoch=2 end ************
2022-11-19 00:27:43,448 P100588 INFO [Metrics] AUC: 0.739962 - gAUC: 0.655535
2022-11-19 00:27:43,455 P100588 INFO Save best model: monitor(max): 1.395497
2022-11-19 00:27:45,695 P100588 INFO --- 1335/1335 batches finished ---
2022-11-19 00:27:45,793 P100588 INFO Train loss: 0.443181
2022-11-19 00:27:45,794 P100588 INFO ************ Epoch=3 end ************
2022-11-19 00:34:13,607 P100588 INFO [Metrics] AUC: 0.739556 - gAUC: 0.657065
2022-11-19 00:34:13,619 P100588 INFO Save best model: monitor(max): 1.396621
2022-11-19 00:34:15,823 P100588 INFO --- 1335/1335 batches finished ---
2022-11-19 00:34:15,914 P100588 INFO Train loss: 0.440347
2022-11-19 00:34:15,914 P100588 INFO ************ Epoch=4 end ************
2022-11-19 00:40:40,567 P100588 INFO [Metrics] AUC: 0.740458 - gAUC: 0.658430
2022-11-19 00:40:40,587 P100588 INFO Save best model: monitor(max): 1.398888
2022-11-19 00:40:42,886 P100588 INFO --- 1335/1335 batches finished ---
2022-11-19 00:40:42,995 P100588 INFO Train loss: 0.438206
2022-11-19 00:40:42,995 P100588 INFO ************ Epoch=5 end ************
2022-11-19 00:47:13,123 P100588 INFO [Metrics] AUC: 0.740497 - gAUC: 0.659649
2022-11-19 00:47:13,134 P100588 INFO Save best model: monitor(max): 1.400146
2022-11-19 00:47:15,294 P100588 INFO --- 1335/1335 batches finished ---
2022-11-19 00:47:15,402 P100588 INFO Train loss: 0.436731
2022-11-19 00:47:15,402 P100588 INFO ************ Epoch=6 end ************
2022-11-19 00:53:36,625 P100588 INFO [Metrics] AUC: 0.742210 - gAUC: 0.660694
2022-11-19 00:53:36,634 P100588 INFO Save best model: monitor(max): 1.402904
2022-11-19 00:53:38,776 P100588 INFO --- 1335/1335 batches finished ---
2022-11-19 00:53:38,895 P100588 INFO Train loss: 0.435793
2022-11-19 00:53:38,895 P100588 INFO ************ Epoch=7 end ************
2022-11-19 00:59:50,123 P100588 INFO [Metrics] AUC: 0.744677 - gAUC: 0.663020
2022-11-19 00:59:50,131 P100588 INFO Save best model: monitor(max): 1.407697
2022-11-19 00:59:52,403 P100588 INFO --- 1335/1335 batches finished ---
2022-11-19 00:59:52,513 P100588 INFO Train loss: 0.434744
2022-11-19 00:59:52,514 P100588 INFO ************ Epoch=8 end ************
2022-11-19 01:06:23,126 P100588 INFO [Metrics] AUC: 0.742445 - gAUC: 0.661532
2022-11-19 01:06:23,137 P100588 INFO Monitor(max) STOP: 1.403976 !
2022-11-19 01:06:23,137 P100588 INFO Reduce learning rate on plateau: 0.000100
2022-11-19 01:06:23,138 P100588 INFO --- 1335/1335 batches finished ---
2022-11-19 01:06:23,259 P100588 INFO Train loss: 0.433893
2022-11-19 01:06:23,259 P100588 INFO ************ Epoch=9 end ************
2022-11-19 01:12:23,852 P100588 INFO [Metrics] AUC: 0.745264 - gAUC: 0.665842
2022-11-19 01:12:23,867 P100588 INFO Save best model: monitor(max): 1.411107
2022-11-19 01:12:26,068 P100588 INFO --- 1335/1335 batches finished ---
2022-11-19 01:12:26,165 P100588 INFO Train loss: 0.409623
2022-11-19 01:12:26,165 P100588 INFO ************ Epoch=10 end ************
2022-11-19 01:18:07,955 P100588 INFO [Metrics] AUC: 0.745041 - gAUC: 0.666017
2022-11-19 01:18:07,963 P100588 INFO Monitor(max) STOP: 1.411058 !
2022-11-19 01:18:07,963 P100588 INFO Reduce learning rate on plateau: 0.000010
2022-11-19 01:18:07,964 P100588 INFO --- 1335/1335 batches finished ---
2022-11-19 01:18:08,063 P100588 INFO Train loss: 0.401831
2022-11-19 01:18:08,063 P100588 INFO ************ Epoch=11 end ************
2022-11-19 01:23:47,152 P100588 INFO [Metrics] AUC: 0.743544 - gAUC: 0.665123
2022-11-19 01:23:47,163 P100588 INFO Monitor(max) STOP: 1.408667 !
2022-11-19 01:23:47,163 P100588 INFO Reduce learning rate on plateau: 0.000001
2022-11-19 01:23:47,163 P100588 INFO ********* Epoch==12 early stop *********
2022-11-19 01:23:47,163 P100588 INFO --- 1335/1335 batches finished ---
2022-11-19 01:23:47,274 P100588 INFO Train loss: 0.393168
2022-11-19 01:23:47,274 P100588 INFO Training finished.
2022-11-19 01:23:47,274 P100588 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DNN_kuaivideo_x1/kuaivideo_x1_dc7a3035/DNN_kuaivideo_x1_023_2da45570.model
2022-11-19 01:23:48,467 P100588 INFO ****** Validation evaluation ******
2022-11-19 01:25:03,811 P100588 INFO [Metrics] gAUC: 0.665842 - AUC: 0.745264 - logloss: 0.440147
2022-11-19 01:25:04,000 P100588 INFO ******** Test evaluation ********
2022-11-19 01:25:04,000 P100588 INFO Loading data...
2022-11-19 01:25:04,000 P100588 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/test.h5
2022-11-19 01:25:11,267 P100588 INFO Test samples: total/2730291, blocks/1
2022-11-19 01:25:11,267 P100588 INFO Loading test data done.
2022-11-19 01:26:22,291 P100588 INFO [Metrics] gAUC: 0.665842 - AUC: 0.745264 - logloss: 0.440147

```
