## DNN_taobaoad_x1

A hands-on guide to run the DNN model on the TaobaoAd_x1 dataset.

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

We use the [DNN](https://github.com/xue-pai/FuxiCTR/blob/v2.0.1/model_zoo/DNN) model code from [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Taobao/TaobaoAd_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DNN_taobao_x1_tuner_config_01](./DNN_taobao_x1_tuner_config_01). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DNN
    nohup python run_expid.py --config XXX/benchmarks/DNN/DNN_taobao_x1_tuner_config_01 --expid DNN_taobaoad_x1_009_0f4cc5a8 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.573269 | 0.646715 | 0.193133  |


### Logs
```python
2022-11-18 15:48:17,385 P79058 INFO Params: {
    "batch_norm": "False",
    "batch_size": "8192",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobaoad_x1_2753db8a",
    "debug_mode": "False",
    "early_stop_patience": "1",
    "embedding_dim": "32",
    "embedding_regularizer": "5e-06",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(userid)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'pid', 'btag'], 'type': 'categorical'}, {'active': True, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'cate_his', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'brand_his', 'padding': 'pre', 'share_embedding': 'brand', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'btag_his', 'padding': 'pre', 'share_embedding': 'btag', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "None",
    "gpu": "0",
    "group_id": "group_id",
    "hidden_activations": "relu",
    "hidden_units": "[512, 256, 128]",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "DNN",
    "model_id": "DNN_taobaoad_x1_009_0f4cc5a8",
    "model_root": "./checkpoints/DNN_taobaoad_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0",
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
2022-11-18 15:48:17,386 P79058 INFO Set up feature processor...
2022-11-18 15:48:17,386 P79058 INFO Load feature_map from json: ../data/Taobao/taobaoad_x1_2753db8a/feature_map.json
2022-11-18 15:48:17,387 P79058 INFO Set column index...
2022-11-18 15:48:17,387 P79058 INFO Feature specs: {
    "adgroup_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 246850, 'vocab_size': 246851}",
    "age_level": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 8, 'vocab_size': 9}",
    "brand": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 308869, 'vocab_size': 308870}",
    "brand_his": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'brand', 'padding_idx': 0, 'oov_idx': 308869, 'vocab_size': 308870, 'max_len': 50}",
    "btag": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 5, 'vocab_size': 6}",
    "btag_his": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'btag', 'padding_idx': 0, 'oov_idx': 5, 'vocab_size': 6, 'max_len': 50}",
    "campaign_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 191770, 'vocab_size': 191771}",
    "cate_his": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 11329, 'vocab_size': 11330, 'max_len': 50}",
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
2022-11-18 15:48:24,511 P79058 INFO Total number of parameters: 41934977.
2022-11-18 15:48:24,512 P79058 INFO Loading data...
2022-11-18 15:48:24,513 P79058 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/train.h5
2022-11-18 15:50:23,686 P79058 INFO Train samples: total/21929911, blocks/1
2022-11-18 15:50:23,689 P79058 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/valid.h5
2022-11-18 15:50:28,395 P79058 INFO Validation samples: total/3099515, blocks/1
2022-11-18 15:50:28,395 P79058 INFO Loading train and validation data done.
2022-11-18 15:50:28,395 P79058 INFO Start training: 2677 batches/epoch
2022-11-18 15:50:28,395 P79058 INFO ************ Epoch=1 start ************
2022-11-18 16:08:07,490 P79058 INFO [Metrics] AUC: 0.644183 - gAUC: 0.565629
2022-11-18 16:08:07,932 P79058 INFO Save best model: monitor(max): 1.209813
2022-11-18 16:08:08,255 P79058 INFO --- 2677/2677 batches finished ---
2022-11-18 16:08:08,430 P79058 INFO Train loss: 0.205119
2022-11-18 16:08:08,431 P79058 INFO ************ Epoch=1 end ************
2022-11-18 16:25:54,940 P79058 INFO [Metrics] AUC: 0.646987 - gAUC: 0.571721
2022-11-18 16:25:55,416 P79058 INFO Save best model: monitor(max): 1.218708
2022-11-18 16:25:55,855 P79058 INFO --- 2677/2677 batches finished ---
2022-11-18 16:25:56,117 P79058 INFO Train loss: 0.202012
2022-11-18 16:25:56,117 P79058 INFO ************ Epoch=2 end ************
2022-11-18 16:43:31,004 P79058 INFO [Metrics] AUC: 0.647597 - gAUC: 0.571997
2022-11-18 16:43:31,444 P79058 INFO Save best model: monitor(max): 1.219594
2022-11-18 16:43:31,875 P79058 INFO --- 2677/2677 batches finished ---
2022-11-18 16:43:32,228 P79058 INFO Train loss: 0.200657
2022-11-18 16:43:32,229 P79058 INFO ************ Epoch=3 end ************
2022-11-18 17:00:54,846 P79058 INFO [Metrics] AUC: 0.646715 - gAUC: 0.573269
2022-11-18 17:00:55,210 P79058 INFO Save best model: monitor(max): 1.219984
2022-11-18 17:00:55,708 P79058 INFO --- 2677/2677 batches finished ---
2022-11-18 17:00:56,012 P79058 INFO Train loss: 0.200546
2022-11-18 17:00:56,013 P79058 INFO ************ Epoch=4 end ************
2022-11-18 17:17:03,342 P79058 INFO [Metrics] AUC: 0.646843 - gAUC: 0.572112
2022-11-18 17:17:03,746 P79058 INFO Monitor(max) STOP: 1.218955 !
2022-11-18 17:17:03,747 P79058 INFO Reduce learning rate on plateau: 0.000100
2022-11-18 17:17:03,747 P79058 INFO ********* Epoch==5 early stop *********
2022-11-18 17:17:03,748 P79058 INFO --- 2677/2677 batches finished ---
2022-11-18 17:17:03,971 P79058 INFO Train loss: 0.201048
2022-11-18 17:17:03,975 P79058 INFO Training finished.
2022-11-18 17:17:03,975 P79058 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DNN_taobaoad_x1/taobaoad_x1_2753db8a/DNN_taobaoad_x1_009_0f4cc5a8.model
2022-11-18 17:17:04,156 P79058 INFO ****** Validation evaluation ******
2022-11-18 17:20:30,853 P79058 INFO [Metrics] gAUC: 0.573269 - AUC: 0.646715 - logloss: 0.193133
2022-11-18 17:20:31,833 P79058 INFO ******** Test evaluation ********
2022-11-18 17:20:31,834 P79058 INFO Loading data...
2022-11-18 17:20:31,834 P79058 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/test.h5
2022-11-18 17:20:36,576 P79058 INFO Test samples: total/3099515, blocks/1
2022-11-18 17:20:36,576 P79058 INFO Loading test data done.
2022-11-18 17:23:50,176 P79058 INFO [Metrics] gAUC: 0.573269 - AUC: 0.646715 - logloss: 0.193133

```
