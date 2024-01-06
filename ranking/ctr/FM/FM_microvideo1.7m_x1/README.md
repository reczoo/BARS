## FM_microvideo1.7m_x1

A hands-on guide to run the FM model on the MicroVideo1.7M_x1 dataset.

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
Please refer to the BARS dataset [MicroVideo1.7M_x1](https://github.com/openbenchmark/BARS/blob/main/datasets/MicroVideo1.7M#MicroVideo17M_x1) to get data ready.

### Code

We use the [FM](https://github.com/xue-pai/FuxiCTR/blob/v2.0.1/model_zoo/FM) model code from [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/MicroVideo1.7M/MicroVideo1.7M_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FM_microvideo1.7m_x1_tuner_config_05](./FM_microvideo1.7m_x1_tuner_config_05). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/FM
    nohup python run_expid.py --config XXX/benchmarks/FM/FM_microvideo1.7m_x1_tuner_config_05 --expid FM_microvideo1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.672374 | 0.718639 | 0.415241  |


### Logs
```python
2022-08-19 14:19:17,788 P74022 INFO Params: {
    "batch_size": "2048",
    "data_format": "csv",
    "data_root": "../data/MicroVideo1.7M/",
    "dataset_id": "microvideo1.7m_x1_0d855fe6",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'name': 'clicked_items', 'padding': 'pre', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'clicked_categories', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'clicked_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'clicked_categories'}]",
    "gpu": "2",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FM",
    "model_id": "FM_microvideo1.7m_x1_003_ffc302aa",
    "model_root": "./checkpoints/FM_microvideo1.7m_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "regularizer": "0.0001",
    "save_best_only": "True",
    "seed": "2022",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "train_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/train.csv",
    "valid_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "verbose": "1"
}
2022-08-19 14:19:17,789 P74022 INFO Set up feature processor...
2022-08-19 14:19:17,789 P74022 INFO Load feature_map from json: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/feature_map.json
2022-08-19 14:19:17,790 P74022 INFO Set column index...
2022-08-19 14:19:17,790 P74022 INFO Feature specs: {
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514}",
    "clicked_categories": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514, 'max_len': 100}",
    "clicked_items": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'max_len': 100}",
    "group_id": "{'type': 'meta'}",
    "item_id": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10987, 'vocab_size': 10988}"
}
2022-08-19 14:19:27,049 P74022 INFO Total number of parameters: 4166101.
2022-08-19 14:19:27,050 P74022 INFO Loading data...
2022-08-19 14:19:27,051 P74022 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/train.h5
2022-08-19 14:19:37,258 P74022 INFO Train samples: total/8970309, blocks/1
2022-08-19 14:19:37,258 P74022 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/valid.h5
2022-08-19 14:19:41,804 P74022 INFO Validation samples: total/3767308, blocks/1
2022-08-19 14:19:41,805 P74022 INFO Loading train and validation data done.
2022-08-19 14:19:41,805 P74022 INFO Start training: 4381 batches/epoch
2022-08-19 14:19:41,805 P74022 INFO ************ Epoch=1 start ************
2022-08-19 14:36:04,469 P74022 INFO [Metrics] AUC: 0.706684 - gAUC: 0.657126
2022-08-19 14:36:04,484 P74022 INFO Save best model: monitor(max): 1.363810
2022-08-19 14:36:06,098 P74022 INFO --- 4381/4381 batches finished ---
2022-08-19 14:36:06,162 P74022 INFO Train loss: 0.468795
2022-08-19 14:36:06,162 P74022 INFO ************ Epoch=1 end ************
2022-08-19 14:52:21,366 P74022 INFO [Metrics] AUC: 0.709939 - gAUC: 0.663045
2022-08-19 14:52:21,376 P74022 INFO Save best model: monitor(max): 1.372984
2022-08-19 14:52:23,443 P74022 INFO --- 4381/4381 batches finished ---
2022-08-19 14:52:23,528 P74022 INFO Train loss: 0.445092
2022-08-19 14:52:23,529 P74022 INFO ************ Epoch=2 end ************
2022-08-19 15:08:49,748 P74022 INFO [Metrics] AUC: 0.710688 - gAUC: 0.665049
2022-08-19 15:08:49,756 P74022 INFO Save best model: monitor(max): 1.375736
2022-08-19 15:08:51,773 P74022 INFO --- 4381/4381 batches finished ---
2022-08-19 15:08:51,859 P74022 INFO Train loss: 0.443456
2022-08-19 15:08:51,859 P74022 INFO ************ Epoch=3 end ************
2022-08-19 15:25:06,700 P74022 INFO [Metrics] AUC: 0.711112 - gAUC: 0.665326
2022-08-19 15:25:06,711 P74022 INFO Save best model: monitor(max): 1.376438
2022-08-19 15:25:08,693 P74022 INFO --- 4381/4381 batches finished ---
2022-08-19 15:25:08,787 P74022 INFO Train loss: 0.442641
2022-08-19 15:25:08,787 P74022 INFO ************ Epoch=4 end ************
2022-08-19 15:41:29,851 P74022 INFO [Metrics] AUC: 0.710817 - gAUC: 0.665805
2022-08-19 15:41:29,857 P74022 INFO Save best model: monitor(max): 1.376622
2022-08-19 15:41:32,034 P74022 INFO --- 4381/4381 batches finished ---
2022-08-19 15:41:32,133 P74022 INFO Train loss: 0.442103
2022-08-19 15:41:32,133 P74022 INFO ************ Epoch=5 end ************
2022-08-19 15:57:55,749 P74022 INFO [Metrics] AUC: 0.712690 - gAUC: 0.667453
2022-08-19 15:57:55,757 P74022 INFO Save best model: monitor(max): 1.380143
2022-08-19 15:57:57,889 P74022 INFO --- 4381/4381 batches finished ---
2022-08-19 15:57:57,971 P74022 INFO Train loss: 0.441712
2022-08-19 15:57:57,971 P74022 INFO ************ Epoch=6 end ************
2022-08-19 16:14:25,658 P74022 INFO [Metrics] AUC: 0.711518 - gAUC: 0.666742
2022-08-19 16:14:25,666 P74022 INFO Monitor(max) STOP: 1.378260 !
2022-08-19 16:14:25,666 P74022 INFO Reduce learning rate on plateau: 0.000050
2022-08-19 16:14:25,667 P74022 INFO --- 4381/4381 batches finished ---
2022-08-19 16:14:25,829 P74022 INFO Train loss: 0.441400
2022-08-19 16:14:25,829 P74022 INFO ************ Epoch=7 end ************
2022-08-19 16:29:42,571 P74022 INFO [Metrics] AUC: 0.717063 - gAUC: 0.671743
2022-08-19 16:29:42,581 P74022 INFO Save best model: monitor(max): 1.388806
2022-08-19 16:29:44,645 P74022 INFO --- 4381/4381 batches finished ---
2022-08-19 16:29:44,726 P74022 INFO Train loss: 0.432686
2022-08-19 16:29:44,726 P74022 INFO ************ Epoch=8 end ************
2022-08-19 16:41:53,647 P74022 INFO [Metrics] AUC: 0.718092 - gAUC: 0.672273
2022-08-19 16:41:53,655 P74022 INFO Save best model: monitor(max): 1.390365
2022-08-19 16:41:55,694 P74022 INFO --- 4381/4381 batches finished ---
2022-08-19 16:41:55,775 P74022 INFO Train loss: 0.430311
2022-08-19 16:41:55,775 P74022 INFO ************ Epoch=9 end ************
2022-08-19 16:50:46,049 P74022 INFO [Metrics] AUC: 0.718267 - gAUC: 0.672212
2022-08-19 16:50:46,060 P74022 INFO Save best model: monitor(max): 1.390478
2022-08-19 16:50:48,040 P74022 INFO --- 4381/4381 batches finished ---
2022-08-19 16:50:48,117 P74022 INFO Train loss: 0.429427
2022-08-19 16:50:48,117 P74022 INFO ************ Epoch=10 end ************
2022-08-19 16:55:56,712 P74022 INFO [Metrics] AUC: 0.718363 - gAUC: 0.672047
2022-08-19 16:55:56,723 P74022 INFO Monitor(max) STOP: 1.390410 !
2022-08-19 16:55:56,723 P74022 INFO Reduce learning rate on plateau: 0.000005
2022-08-19 16:55:56,724 P74022 INFO --- 4381/4381 batches finished ---
2022-08-19 16:55:56,800 P74022 INFO Train loss: 0.428967
2022-08-19 16:55:56,801 P74022 INFO ************ Epoch=11 end ************
2022-08-19 17:00:04,524 P74022 INFO [Metrics] AUC: 0.718639 - gAUC: 0.672374
2022-08-19 17:00:04,530 P74022 INFO Save best model: monitor(max): 1.391012
2022-08-19 17:00:06,307 P74022 INFO --- 4381/4381 batches finished ---
2022-08-19 17:00:06,397 P74022 INFO Train loss: 0.427133
2022-08-19 17:00:06,398 P74022 INFO ************ Epoch=12 end ************
2022-08-19 17:02:55,026 P74022 INFO [Metrics] AUC: 0.718633 - gAUC: 0.672347
2022-08-19 17:02:55,031 P74022 INFO Monitor(max) STOP: 1.390980 !
2022-08-19 17:02:55,031 P74022 INFO Reduce learning rate on plateau: 0.000001
2022-08-19 17:02:55,032 P74022 INFO --- 4381/4381 batches finished ---
2022-08-19 17:02:55,088 P74022 INFO Train loss: 0.427021
2022-08-19 17:02:55,089 P74022 INFO ************ Epoch=13 end ************
2022-08-19 17:04:27,360 P74022 INFO [Metrics] AUC: 0.718662 - gAUC: 0.672334
2022-08-19 17:04:27,365 P74022 INFO Monitor(max) STOP: 1.390996 !
2022-08-19 17:04:27,365 P74022 INFO Reduce learning rate on plateau: 0.000001
2022-08-19 17:04:27,365 P74022 INFO ********* Epoch==14 early stop *********
2022-08-19 17:04:27,366 P74022 INFO --- 4381/4381 batches finished ---
2022-08-19 17:04:27,436 P74022 INFO Train loss: 0.426803
2022-08-19 17:04:27,436 P74022 INFO Training finished.
2022-08-19 17:04:27,436 P74022 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/FM_microvideo1.7m_x1/microvideo1.7m_x1_0d855fe6/FM_microvideo1.7m_x1_003_ffc302aa.model
2022-08-19 17:04:28,095 P74022 INFO ****** Validation evaluation ******
2022-08-19 17:04:48,757 P74022 INFO [Metrics] gAUC: 0.672374 - AUC: 0.718639 - logloss: 0.415241
2022-08-19 17:04:48,885 P74022 INFO ******** Test evaluation ********
2022-08-19 17:04:48,885 P74022 INFO Loading data...
2022-08-19 17:04:48,885 P74022 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/test.h5
2022-08-19 17:04:52,843 P74022 INFO Test samples: total/3767308, blocks/1
2022-08-19 17:04:52,843 P74022 INFO Loading test data done.
2022-08-19 17:05:13,286 P74022 INFO [Metrics] gAUC: 0.672374 - AUC: 0.718639 - logloss: 0.415241

```
