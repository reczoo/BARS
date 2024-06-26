## xDeepFM_kuaivideo_x1

A hands-on guide to run the xDeepFM model on the KuaiVideo_x1 dataset.

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

We use the [xDeepFM](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/xDeepFM) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/KuaiShou/KuaiVideo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [xDeepFM_kuaivideo_x1_tuner_config_02](./xDeepFM_kuaivideo_x1_tuner_config_02). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/xDeepFM
    nohup python run_expid.py --config XXX/benchmarks/xDeepFM/xDeepFM_kuaivideo_x1_tuner_config_02 --expid xDeepFM_kuaivideo_x1_016_0372eae9 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.669617 | 0.747075 | 0.437683  |


### Logs
```python
2022-08-23 21:40:21,181 P81121 INFO Params: {
    "batch_norm": "True",
    "batch_size": "8192",
    "cin_hidden_units": "[64, 64]",
    "data_format": "csv",
    "data_root": "../data/KuaiShou/",
    "dataset_id": "kuaivideo_x1_dc7a3035",
    "debug_mode": "False",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.0005",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'min_categr_count': 1, 'name': 'item_emb', 'preprocess': 'copy_from(item_id)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'pos_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'neg_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'pos_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(pos_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'neg_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(neg_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_emb'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'pos_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'neg_items'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'pos_items_emb'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'neg_items_emb'}]",
    "gpu": "5",
    "group_id": "group_id",
    "hidden_activations": "relu",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "xDeepFM",
    "model_id": "xDeepFM_kuaivideo_x1_016_0372eae9",
    "model_root": "./checkpoints/xDeepFM_kuaivideo_x1/",
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
2022-08-23 21:40:21,182 P81121 INFO Set up feature processor...
2022-08-23 21:40:21,182 P81121 INFO Load feature_map from json: ../data/KuaiShou/kuaivideo_x1_dc7a3035/feature_map.json
2022-08-23 21:40:21,183 P81121 INFO Set column index...
2022-08-23 21:40:21,183 P81121 INFO Feature specs: {
    "group_id": "{'type': 'meta'}",
    "item_emb": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406}",
    "neg_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "neg_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "pos_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "pos_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10001, 'vocab_size': 10002}"
}
2022-08-23 21:40:28,526 P81121 INFO Total number of parameters: 53912381.
2022-08-23 21:40:28,526 P81121 INFO Loading data...
2022-08-23 21:40:28,526 P81121 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/train.h5
2022-08-23 21:40:56,661 P81121 INFO Train samples: total/10931092, blocks/1
2022-08-23 21:40:56,662 P81121 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/valid.h5
2022-08-23 21:41:03,965 P81121 INFO Validation samples: total/2730291, blocks/1
2022-08-23 21:41:03,965 P81121 INFO Loading train and validation data done.
2022-08-23 21:41:03,965 P81121 INFO Start training: 1335 batches/epoch
2022-08-23 21:41:03,965 P81121 INFO ************ Epoch=1 start ************
2022-08-23 21:48:11,186 P81121 INFO [Metrics] AUC: 0.712296 - gAUC: 0.638810
2022-08-23 21:48:11,200 P81121 INFO Save best model: monitor(max): 1.351106
2022-08-23 21:48:13,066 P81121 INFO --- 1335/1335 batches finished ---
2022-08-23 21:48:13,146 P81121 INFO Train loss: 0.494716
2022-08-23 21:48:13,146 P81121 INFO ************ Epoch=1 end ************
2022-08-23 21:55:20,280 P81121 INFO [Metrics] AUC: 0.716236 - gAUC: 0.642465
2022-08-23 21:55:20,289 P81121 INFO Save best model: monitor(max): 1.358701
2022-08-23 21:55:22,704 P81121 INFO --- 1335/1335 batches finished ---
2022-08-23 21:55:22,774 P81121 INFO Train loss: 0.478587
2022-08-23 21:55:22,774 P81121 INFO ************ Epoch=2 end ************
2022-08-23 22:02:26,934 P81121 INFO [Metrics] AUC: 0.717490 - gAUC: 0.645723
2022-08-23 22:02:26,947 P81121 INFO Save best model: monitor(max): 1.363213
2022-08-23 22:02:29,316 P81121 INFO --- 1335/1335 batches finished ---
2022-08-23 22:02:29,400 P81121 INFO Train loss: 0.476762
2022-08-23 22:02:29,400 P81121 INFO ************ Epoch=3 end ************
2022-08-23 22:09:31,166 P81121 INFO [Metrics] AUC: 0.719699 - gAUC: 0.648761
2022-08-23 22:09:31,175 P81121 INFO Save best model: monitor(max): 1.368460
2022-08-23 22:09:33,451 P81121 INFO --- 1335/1335 batches finished ---
2022-08-23 22:09:33,525 P81121 INFO Train loss: 0.474716
2022-08-23 22:09:33,525 P81121 INFO ************ Epoch=4 end ************
2022-08-23 22:16:35,911 P81121 INFO [Metrics] AUC: 0.719828 - gAUC: 0.651837
2022-08-23 22:16:35,921 P81121 INFO Save best model: monitor(max): 1.371666
2022-08-23 22:16:38,225 P81121 INFO --- 1335/1335 batches finished ---
2022-08-23 22:16:38,318 P81121 INFO Train loss: 0.472735
2022-08-23 22:16:38,318 P81121 INFO ************ Epoch=5 end ************
2022-08-23 22:23:39,416 P81121 INFO [Metrics] AUC: 0.721639 - gAUC: 0.651887
2022-08-23 22:23:39,425 P81121 INFO Save best model: monitor(max): 1.373527
2022-08-23 22:23:41,699 P81121 INFO --- 1335/1335 batches finished ---
2022-08-23 22:23:41,781 P81121 INFO Train loss: 0.471303
2022-08-23 22:23:41,781 P81121 INFO ************ Epoch=6 end ************
2022-08-23 22:30:40,452 P81121 INFO [Metrics] AUC: 0.721042 - gAUC: 0.652801
2022-08-23 22:30:40,459 P81121 INFO Save best model: monitor(max): 1.373843
2022-08-23 22:30:42,836 P81121 INFO --- 1335/1335 batches finished ---
2022-08-23 22:30:42,922 P81121 INFO Train loss: 0.469841
2022-08-23 22:30:42,922 P81121 INFO ************ Epoch=7 end ************
2022-08-23 22:37:41,249 P81121 INFO [Metrics] AUC: 0.721959 - gAUC: 0.653582
2022-08-23 22:37:41,258 P81121 INFO Save best model: monitor(max): 1.375541
2022-08-23 22:37:43,543 P81121 INFO --- 1335/1335 batches finished ---
2022-08-23 22:37:43,639 P81121 INFO Train loss: 0.468844
2022-08-23 22:37:43,639 P81121 INFO ************ Epoch=8 end ************
2022-08-23 22:44:23,782 P81121 INFO [Metrics] AUC: 0.721402 - gAUC: 0.654343
2022-08-23 22:44:23,788 P81121 INFO Save best model: monitor(max): 1.375745
2022-08-23 22:44:26,119 P81121 INFO --- 1335/1335 batches finished ---
2022-08-23 22:44:26,197 P81121 INFO Train loss: 0.467852
2022-08-23 22:44:26,197 P81121 INFO ************ Epoch=9 end ************
2022-08-23 22:50:30,188 P81121 INFO [Metrics] AUC: 0.722602 - gAUC: 0.654904
2022-08-23 22:50:30,195 P81121 INFO Save best model: monitor(max): 1.377506
2022-08-23 22:50:32,350 P81121 INFO --- 1335/1335 batches finished ---
2022-08-23 22:50:32,437 P81121 INFO Train loss: 0.467016
2022-08-23 22:50:32,437 P81121 INFO ************ Epoch=10 end ************
2022-08-23 22:56:26,071 P81121 INFO [Metrics] AUC: 0.725514 - gAUC: 0.656380
2022-08-23 22:56:26,078 P81121 INFO Save best model: monitor(max): 1.381894
2022-08-23 22:56:28,364 P81121 INFO --- 1335/1335 batches finished ---
2022-08-23 22:56:28,438 P81121 INFO Train loss: 0.466199
2022-08-23 22:56:28,438 P81121 INFO ************ Epoch=11 end ************
2022-08-23 23:02:15,203 P81121 INFO [Metrics] AUC: 0.722788 - gAUC: 0.655674
2022-08-23 23:02:15,209 P81121 INFO Monitor(max) STOP: 1.378463 !
2022-08-23 23:02:15,209 P81121 INFO Reduce learning rate on plateau: 0.000100
2022-08-23 23:02:15,209 P81121 INFO --- 1335/1335 batches finished ---
2022-08-23 23:02:15,290 P81121 INFO Train loss: 0.465686
2022-08-23 23:02:15,291 P81121 INFO ************ Epoch=12 end ************
2022-08-23 23:07:59,084 P81121 INFO [Metrics] AUC: 0.746248 - gAUC: 0.666062
2022-08-23 23:07:59,091 P81121 INFO Save best model: monitor(max): 1.412310
2022-08-23 23:08:01,340 P81121 INFO --- 1335/1335 batches finished ---
2022-08-23 23:08:01,432 P81121 INFO Train loss: 0.427631
2022-08-23 23:08:01,433 P81121 INFO ************ Epoch=13 end ************
2022-08-23 23:13:45,662 P81121 INFO [Metrics] AUC: 0.747075 - gAUC: 0.668974
2022-08-23 23:13:45,672 P81121 INFO Save best model: monitor(max): 1.416049
2022-08-23 23:13:47,828 P81121 INFO --- 1335/1335 batches finished ---
2022-08-23 23:13:47,904 P81121 INFO Train loss: 0.418746
2022-08-23 23:13:47,904 P81121 INFO ************ Epoch=14 end ************
2022-08-23 23:19:27,779 P81121 INFO [Metrics] AUC: 0.747075 - gAUC: 0.669617
2022-08-23 23:19:27,789 P81121 INFO Save best model: monitor(max): 1.416692
2022-08-23 23:19:30,071 P81121 INFO --- 1335/1335 batches finished ---
2022-08-23 23:19:30,144 P81121 INFO Train loss: 0.414130
2022-08-23 23:19:30,144 P81121 INFO ************ Epoch=15 end ************
2022-08-23 23:25:10,539 P81121 INFO [Metrics] AUC: 0.746097 - gAUC: 0.668952
2022-08-23 23:25:10,546 P81121 INFO Monitor(max) STOP: 1.415048 !
2022-08-23 23:25:10,546 P81121 INFO Reduce learning rate on plateau: 0.000010
2022-08-23 23:25:10,547 P81121 INFO --- 1335/1335 batches finished ---
2022-08-23 23:25:10,629 P81121 INFO Train loss: 0.410089
2022-08-23 23:25:10,629 P81121 INFO ************ Epoch=16 end ************
2022-08-23 23:30:57,626 P81121 INFO [Metrics] AUC: 0.745830 - gAUC: 0.668803
2022-08-23 23:30:57,635 P81121 INFO Monitor(max) STOP: 1.414633 !
2022-08-23 23:30:57,635 P81121 INFO Reduce learning rate on plateau: 0.000001
2022-08-23 23:30:57,635 P81121 INFO ********* Epoch==17 early stop *********
2022-08-23 23:30:57,636 P81121 INFO --- 1335/1335 batches finished ---
2022-08-23 23:30:57,704 P81121 INFO Train loss: 0.392105
2022-08-23 23:30:57,704 P81121 INFO Training finished.
2022-08-23 23:30:57,705 P81121 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/xDeepFM_kuaivideo_x1/kuaivideo_x1_dc7a3035/xDeepFM_kuaivideo_x1_016_0372eae9.model
2022-08-23 23:30:59,092 P81121 INFO ****** Validation evaluation ******
2022-08-23 23:31:20,679 P81121 INFO [Metrics] gAUC: 0.669617 - AUC: 0.747075 - logloss: 0.437683
2022-08-23 23:31:20,834 P81121 INFO ******** Test evaluation ********
2022-08-23 23:31:20,835 P81121 INFO Loading data...
2022-08-23 23:31:20,835 P81121 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/test.h5
2022-08-23 23:31:27,107 P81121 INFO Test samples: total/2730291, blocks/1
2022-08-23 23:31:27,107 P81121 INFO Loading test data done.
2022-08-23 23:31:47,404 P81121 INFO [Metrics] gAUC: 0.669617 - AUC: 0.747075 - logloss: 0.437683

```
