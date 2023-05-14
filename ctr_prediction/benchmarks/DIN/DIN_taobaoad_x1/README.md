## DIN_taobaoad_x1

A hands-on guide to run the DIN model on the TaobaoAd_x1 dataset.

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
Please refer to the BARS dataset [TaobaoAd_x1](https://github.com/openbenchmark/BARS/blob/main/datasets/Taobao#TaobaoAd_x1) to get data ready.

### Code

We use the [DIN](https://github.com/xue-pai/FuxiCTR/blob/v2.0.1/model_zoo/DIN) model code from [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Taobao/TaobaoAd_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DIN_taobaoad_x1_tuner_config_07](./DIN_taobaoad_x1_tuner_config_07). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DIN
    nohup python run_expid.py --config XXX/benchmarks/DIN/DIN_taobaoad_x1_tuner_config_07 --expid DIN_taobaoad_x1_012_13c23a36 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.576459 | 0.652399 | 0.192445  |


### Logs
```python
2022-08-12 06:12:59,499 P99655 INFO Params: {
    "attention_dropout": "0.4",
    "attention_hidden_activations": "ReLU",
    "attention_hidden_units": "[512, 256]",
    "attention_output_activation": "None",
    "batch_norm": "False",
    "batch_size": "8192",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobaoad_x1_2753db8a",
    "debug_mode": "False",
    "din_sequence_field": "('cate_his', 'brand_his', 'btag_his')",
    "din_target_field": "('cate_id', 'brand', 'btag')",
    "din_use_softmax": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[512, 256, 128]",
    "early_stop_patience": "1",
    "embedding_dim": "32",
    "embedding_regularizer": "5e-06",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(userid)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'pid', 'btag'], 'type': 'categorical'}, {'active': True, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'cate_his', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'brand_his', 'padding': 'pre', 'share_embedding': 'brand', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'btag_his', 'padding': 'pre', 'share_embedding': 'btag', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "[{'feature_encoder': None, 'name': ['cate_his', 'brand_his', 'btag_his']}]",
    "gpu": "3",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "DIN",
    "model_id": "DIN_taobaoad_x1_012_13c23a36",
    "model_root": "./checkpoints/DIN_taobaoad_x1/",
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
    "test_data": "../data/Taobao/TaobaoAd_x1/test.csv",
    "train_data": "../data/Taobao/TaobaoAd_x1/train.csv",
    "valid_data": "../data/Taobao/TaobaoAd_x1/test.csv",
    "verbose": "1"
}
2022-08-12 06:12:59,500 P99655 INFO Set up feature processor...
2022-08-12 06:12:59,500 P99655 INFO Load feature_map from json: ../data/Taobao/taobaoad_x1_2753db8a/feature_map.json
2022-08-12 06:12:59,500 P99655 INFO Set column index...
2022-08-12 06:12:59,500 P99655 INFO Feature specs: {
    "adgroup_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 246850, 'vocab_size': 246851}",
    "age_level": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 8, 'vocab_size': 9}",
    "brand": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 308869, 'vocab_size': 308870}",
    "brand_his": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'brand', 'padding_idx': 0, 'oov_idx': 308869, 'vocab_size': 308870, 'max_len': 50}",
    "btag": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 5, 'vocab_size': 6}",
    "btag_his": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'btag', 'padding_idx': 0, 'oov_idx': 5, 'vocab_size': 6, 'max_len': 50}",
    "campaign_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 191770, 'vocab_size': 191771}",
    "cate_his": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 11329, 'vocab_size': 11330, 'max_len': 50}",
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 11329, 'vocab_size': 11330}",
    "cms_group_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 14, 'vocab_size': 15}",
    "cms_segid": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 98, 'vocab_size': 99}",
    "customer": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 132135, 'vocab_size': 132136}",
    "final_gender_code": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 3, 'vocab_size': 4}",
    "group_id": "{'type': 'meta'}",
    "new_user_class_level": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 6, 'vocab_size': 7}",
    "occupation": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 3, 'vocab_size': 4}",
    "pid": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 3, 'vocab_size': 4}",
    "price": "{'source': '', 'type': 'numeric'}",
    "pvalue_level": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 5, 'vocab_size': 6}",
    "shopping_level": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 4, 'vocab_size': 5}",
    "userid": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 403957, 'vocab_size': 403958}"
}
2022-08-12 06:13:07,077 P99655 INFO Total number of parameters: 42263682.
2022-08-12 06:13:07,078 P99655 INFO Loading data...
2022-08-12 06:13:07,078 P99655 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/train.h5
2022-08-12 06:13:32,840 P99655 INFO Train samples: total/21929911, blocks/1
2022-08-12 06:13:32,840 P99655 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/valid.h5
2022-08-12 06:13:36,682 P99655 INFO Validation samples: total/3099515, blocks/1
2022-08-12 06:13:36,682 P99655 INFO Loading train and validation data done.
2022-08-12 06:13:36,682 P99655 INFO Start training: 2677 batches/epoch
2022-08-12 06:13:36,682 P99655 INFO ************ Epoch=1 start ************
2022-08-12 06:33:09,608 P99655 INFO [Metrics] AUC: 0.648455 - gAUC: 0.568912
2022-08-12 06:33:09,866 P99655 INFO Save best model: monitor(max): 1.217367
2022-08-12 06:33:10,263 P99655 INFO --- 2677/2677 batches finished ---
2022-08-12 06:33:10,331 P99655 INFO Train loss: 0.205849
2022-08-12 06:33:10,331 P99655 INFO ************ Epoch=1 end ************
2022-08-12 06:52:25,925 P99655 INFO [Metrics] AUC: 0.651967 - gAUC: 0.574210
2022-08-12 06:52:26,193 P99655 INFO Save best model: monitor(max): 1.226177
2022-08-12 06:52:26,521 P99655 INFO --- 2677/2677 batches finished ---
2022-08-12 06:52:26,622 P99655 INFO Train loss: 0.202492
2022-08-12 06:52:26,622 P99655 INFO ************ Epoch=2 end ************
2022-08-12 07:10:14,890 P99655 INFO [Metrics] AUC: 0.652399 - gAUC: 0.576459
2022-08-12 07:10:15,115 P99655 INFO Save best model: monitor(max): 1.228857
2022-08-12 07:10:15,413 P99655 INFO --- 2677/2677 batches finished ---
2022-08-12 07:10:15,535 P99655 INFO Train loss: 0.200582
2022-08-12 07:10:15,535 P99655 INFO ************ Epoch=3 end ************
2022-08-12 07:20:35,163 P99655 INFO [Metrics] AUC: 0.650382 - gAUC: 0.575051
2022-08-12 07:20:35,379 P99655 INFO Monitor(max) STOP: 1.225433 !
2022-08-12 07:20:35,379 P99655 INFO Reduce learning rate on plateau: 0.000100
2022-08-12 07:20:35,379 P99655 INFO ********* Epoch==4 early stop *********
2022-08-12 07:20:35,380 P99655 INFO --- 2677/2677 batches finished ---
2022-08-12 07:20:35,511 P99655 INFO Train loss: 0.200417
2022-08-12 07:20:35,511 P99655 INFO Training finished.
2022-08-12 07:20:35,511 P99655 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DIN_taobaoad_x1/taobaoad_x1_2753db8a/DIN_taobaoad_x1_012_13c23a36.model
2022-08-12 07:20:35,825 P99655 INFO ****** Validation evaluation ******
2022-08-12 07:22:05,141 P99655 INFO [Metrics] gAUC: 0.576459 - AUC: 0.652399 - logloss: 0.192445
2022-08-12 07:22:05,591 P99655 INFO ******** Test evaluation ********
2022-08-12 07:22:05,592 P99655 INFO Loading data...
2022-08-12 07:22:05,592 P99655 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/test.h5
2022-08-12 07:22:09,232 P99655 INFO Test samples: total/3099515, blocks/1
2022-08-12 07:22:09,232 P99655 INFO Loading test data done.
2022-08-12 07:23:37,336 P99655 INFO [Metrics] gAUC: 0.576459 - AUC: 0.652399 - logloss: 0.192445

```
