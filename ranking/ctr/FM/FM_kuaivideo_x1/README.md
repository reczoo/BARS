## FM_kuaivideo_x1

A hands-on guide to run the FM model on the KuaiVideo_x1 dataset.

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

We use the [FM](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/FM) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/KuaiShou/KuaiVideo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FM_kuaivideo_x1_tuner_config_02](./FM_kuaivideo_x1_tuner_config_02). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/FM
    nohup python run_expid.py --config XXX/benchmarks/FM/FM_kuaivideo_x1_tuner_config_02 --expid FM_kuaivideo_x1_012_35007647 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.659864 | 0.741790 | 0.439771  |


### Logs
```python
2022-08-23 02:41:20,556 P58316 INFO Params: {
    "batch_size": "4096",
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
    "gpu": "6",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "FM",
    "model_id": "FM_kuaivideo_x1_012_35007647",
    "model_root": "./checkpoints/FM_kuaivideo_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-05",
    "save_best_only": "True",
    "seed": "20222023",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "train_data": "../data/KuaiShou/KuaiVideo_x1/train.csv",
    "valid_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "verbose": "1"
}
2022-08-23 02:41:20,557 P58316 INFO Set up feature processor...
2022-08-23 02:41:20,557 P58316 INFO Load feature_map from json: ../data/KuaiShou/kuaivideo_x1_dc7a3035/feature_map.json
2022-08-23 02:41:20,558 P58316 INFO Set column index...
2022-08-23 02:41:20,558 P58316 INFO Feature specs: {
    "group_id": "{'type': 'meta'}",
    "item_emb": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406}",
    "neg_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "neg_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "pos_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "pos_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10001, 'vocab_size': 10002}"
}
2022-08-23 02:41:28,072 P58316 INFO Total number of parameters: 52760572.
2022-08-23 02:41:28,072 P58316 INFO Loading data...
2022-08-23 02:41:28,072 P58316 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/train.h5
2022-08-23 02:41:53,123 P58316 INFO Train samples: total/10931092, blocks/1
2022-08-23 02:41:53,124 P58316 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/valid.h5
2022-08-23 02:41:59,970 P58316 INFO Validation samples: total/2730291, blocks/1
2022-08-23 02:41:59,970 P58316 INFO Loading train and validation data done.
2022-08-23 02:41:59,970 P58316 INFO Start training: 2669 batches/epoch
2022-08-23 02:41:59,970 P58316 INFO ************ Epoch=1 start ************
2022-08-23 02:54:26,643 P58316 INFO [Metrics] AUC: 0.703334 - gAUC: 0.640784
2022-08-23 02:54:26,653 P58316 INFO Save best model: monitor(max): 1.344118
2022-08-23 02:54:28,561 P58316 INFO --- 2669/2669 batches finished ---
2022-08-23 02:54:28,658 P58316 INFO Train loss: 0.509706
2022-08-23 02:54:28,658 P58316 INFO ************ Epoch=1 end ************
2022-08-23 03:06:45,068 P58316 INFO [Metrics] AUC: 0.703126 - gAUC: 0.644995
2022-08-23 03:06:45,080 P58316 INFO Save best model: monitor(max): 1.348121
2022-08-23 03:06:47,459 P58316 INFO --- 2669/2669 batches finished ---
2022-08-23 03:06:47,540 P58316 INFO Train loss: 0.464370
2022-08-23 03:06:47,540 P58316 INFO ************ Epoch=2 end ************
2022-08-23 03:19:06,096 P58316 INFO [Metrics] AUC: 0.707544 - gAUC: 0.644233
2022-08-23 03:19:06,103 P58316 INFO Save best model: monitor(max): 1.351777
2022-08-23 03:19:08,433 P58316 INFO --- 2669/2669 batches finished ---
2022-08-23 03:19:08,527 P58316 INFO Train loss: 0.462275
2022-08-23 03:19:08,527 P58316 INFO ************ Epoch=3 end ************
2022-08-23 03:31:35,428 P58316 INFO [Metrics] AUC: 0.707941 - gAUC: 0.645428
2022-08-23 03:31:35,439 P58316 INFO Save best model: monitor(max): 1.353369
2022-08-23 03:31:37,893 P58316 INFO --- 2669/2669 batches finished ---
2022-08-23 03:31:37,989 P58316 INFO Train loss: 0.461490
2022-08-23 03:31:37,989 P58316 INFO ************ Epoch=4 end ************
2022-08-23 03:43:50,889 P58316 INFO [Metrics] AUC: 0.706139 - gAUC: 0.646535
2022-08-23 03:43:50,898 P58316 INFO Monitor(max) STOP: 1.352674 !
2022-08-23 03:43:50,898 P58316 INFO Reduce learning rate on plateau: 0.000100
2022-08-23 03:43:50,899 P58316 INFO --- 2669/2669 batches finished ---
2022-08-23 03:43:50,991 P58316 INFO Train loss: 0.460610
2022-08-23 03:43:50,991 P58316 INFO ************ Epoch=5 end ************
2022-08-23 03:56:07,046 P58316 INFO [Metrics] AUC: 0.736390 - gAUC: 0.654879
2022-08-23 03:56:07,052 P58316 INFO Save best model: monitor(max): 1.391269
2022-08-23 03:56:09,480 P58316 INFO --- 2669/2669 batches finished ---
2022-08-23 03:56:09,565 P58316 INFO Train loss: 0.421676
2022-08-23 03:56:09,565 P58316 INFO ************ Epoch=6 end ************
2022-08-23 04:08:24,394 P58316 INFO [Metrics] AUC: 0.738808 - gAUC: 0.657660
2022-08-23 04:08:24,401 P58316 INFO Save best model: monitor(max): 1.396468
2022-08-23 04:08:26,776 P58316 INFO --- 2669/2669 batches finished ---
2022-08-23 04:08:26,861 P58316 INFO Train loss: 0.413702
2022-08-23 04:08:26,862 P58316 INFO ************ Epoch=7 end ************
2022-08-23 04:20:54,087 P58316 INFO [Metrics] AUC: 0.738844 - gAUC: 0.658319
2022-08-23 04:20:54,094 P58316 INFO Save best model: monitor(max): 1.397163
2022-08-23 04:20:56,416 P58316 INFO --- 2669/2669 batches finished ---
2022-08-23 04:20:56,523 P58316 INFO Train loss: 0.410588
2022-08-23 04:20:56,524 P58316 INFO ************ Epoch=8 end ************
2022-08-23 04:33:16,117 P58316 INFO [Metrics] AUC: 0.739526 - gAUC: 0.659158
2022-08-23 04:33:16,123 P58316 INFO Save best model: monitor(max): 1.398684
2022-08-23 04:33:18,538 P58316 INFO --- 2669/2669 batches finished ---
2022-08-23 04:33:18,643 P58316 INFO Train loss: 0.408357
2022-08-23 04:33:18,643 P58316 INFO ************ Epoch=9 end ************
2022-08-23 04:45:33,960 P58316 INFO [Metrics] AUC: 0.739467 - gAUC: 0.659229
2022-08-23 04:45:33,969 P58316 INFO Save best model: monitor(max): 1.398696
2022-08-23 04:45:36,337 P58316 INFO --- 2669/2669 batches finished ---
2022-08-23 04:45:36,427 P58316 INFO Train loss: 0.406525
2022-08-23 04:45:36,428 P58316 INFO ************ Epoch=10 end ************
2022-08-23 04:57:54,018 P58316 INFO [Metrics] AUC: 0.738293 - gAUC: 0.658995
2022-08-23 04:57:54,024 P58316 INFO Monitor(max) STOP: 1.397288 !
2022-08-23 04:57:54,024 P58316 INFO Reduce learning rate on plateau: 0.000010
2022-08-23 04:57:54,024 P58316 INFO --- 2669/2669 batches finished ---
2022-08-23 04:57:54,116 P58316 INFO Train loss: 0.404923
2022-08-23 04:57:54,117 P58316 INFO ************ Epoch=11 end ************
2022-08-23 05:10:20,319 P58316 INFO [Metrics] AUC: 0.741472 - gAUC: 0.659837
2022-08-23 05:10:20,325 P58316 INFO Save best model: monitor(max): 1.401309
2022-08-23 05:10:22,671 P58316 INFO --- 2669/2669 batches finished ---
2022-08-23 05:10:22,772 P58316 INFO Train loss: 0.396365
2022-08-23 05:10:22,772 P58316 INFO ************ Epoch=12 end ************
2022-08-23 05:22:33,440 P58316 INFO [Metrics] AUC: 0.741764 - gAUC: 0.659864
2022-08-23 05:22:33,447 P58316 INFO Save best model: monitor(max): 1.401628
2022-08-23 05:22:35,782 P58316 INFO --- 2669/2669 batches finished ---
2022-08-23 05:22:35,867 P58316 INFO Train loss: 0.395439
2022-08-23 05:22:35,867 P58316 INFO ************ Epoch=13 end ************
2022-08-23 05:34:51,747 P58316 INFO [Metrics] AUC: 0.741790 - gAUC: 0.659864
2022-08-23 05:34:51,756 P58316 INFO Save best model: monitor(max): 1.401653
2022-08-23 05:34:54,073 P58316 INFO --- 2669/2669 batches finished ---
2022-08-23 05:34:54,193 P58316 INFO Train loss: 0.395050
2022-08-23 05:34:54,193 P58316 INFO ************ Epoch=14 end ************
2022-08-23 05:47:19,176 P58316 INFO [Metrics] AUC: 0.741639 - gAUC: 0.659726
2022-08-23 05:47:19,182 P58316 INFO Monitor(max) STOP: 1.401365 !
2022-08-23 05:47:19,182 P58316 INFO Reduce learning rate on plateau: 0.000001
2022-08-23 05:47:19,183 P58316 INFO --- 2669/2669 batches finished ---
2022-08-23 05:47:19,323 P58316 INFO Train loss: 0.394749
2022-08-23 05:47:19,323 P58316 INFO ************ Epoch=15 end ************
2022-08-23 05:59:48,455 P58316 INFO [Metrics] AUC: 0.741740 - gAUC: 0.659836
2022-08-23 05:59:48,463 P58316 INFO Monitor(max) STOP: 1.401576 !
2022-08-23 05:59:48,463 P58316 INFO Reduce learning rate on plateau: 0.000001
2022-08-23 05:59:48,464 P58316 INFO ********* Epoch==16 early stop *********
2022-08-23 05:59:48,464 P58316 INFO --- 2669/2669 batches finished ---
2022-08-23 05:59:48,568 P58316 INFO Train loss: 0.393534
2022-08-23 05:59:48,568 P58316 INFO Training finished.
2022-08-23 05:59:48,568 P58316 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/FM_kuaivideo_x1/kuaivideo_x1_dc7a3035/FM_kuaivideo_x1_012_35007647.model
2022-08-23 05:59:50,269 P58316 INFO ****** Validation evaluation ******
2022-08-23 06:02:21,830 P58316 INFO [Metrics] gAUC: 0.659864 - AUC: 0.741790 - logloss: 0.439771
2022-08-23 06:02:22,034 P58316 INFO ******** Test evaluation ********
2022-08-23 06:02:22,034 P58316 INFO Loading data...
2022-08-23 06:02:22,034 P58316 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/test.h5
2022-08-23 06:02:28,504 P58316 INFO Test samples: total/2730291, blocks/1
2022-08-23 06:02:28,505 P58316 INFO Loading test data done.
2022-08-23 06:05:00,480 P58316 INFO [Metrics] gAUC: 0.659864 - AUC: 0.741790 - logloss: 0.439771

```
