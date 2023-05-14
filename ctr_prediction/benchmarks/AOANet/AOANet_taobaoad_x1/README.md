## AOANet_taobaoad_x1

A hands-on guide to run the AOANet model on the TaobaoAd_x1 dataset.

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

We use the [AOANet](https://github.com/xue-pai/FuxiCTR/blob/v2.0.1/model_zoo/AOANet) model code from [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Taobao/TaobaoAd_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AOANet_taobaoad_x1_tuner_config_01](./AOANet_taobaoad_x1_tuner_config_01). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/AOANet
    nohup python run_expid.py --config XXX/benchmarks/AOANet/AOANet_taobaoad_x1_tuner_config_01 --expid AOANet_taobaoad_x1_005_26a70b74 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.573251 | 0.650041 | 0.192513  |


### Logs
```python
2022-08-12 17:05:58,078 P1479 INFO Params: {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobaoad_x1_2753db8a",
    "debug_mode": "False",
    "dnn_hidden_activations": "ReLU",
    "dnn_hidden_units": "[512, 256, 128]",
    "early_stop_patience": "1",
    "embedding_dim": "32",
    "embedding_regularizer": "5e-06",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(userid)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'pid', 'btag'], 'type': 'categorical'}, {'active': True, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'cate_his', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'brand_his', 'padding': 'pre', 'share_embedding': 'brand', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'btag_his', 'padding': 'pre', 'share_embedding': 'btag', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "None",
    "gpu": "4",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "AOANet",
    "model_id": "AOANet_taobaoad_x1_005_26a70b74",
    "model_root": "./checkpoints/AOANet_taobaoad_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_interaction_layers": "1",
    "num_subspaces": "4",
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
2022-08-12 17:05:58,079 P1479 INFO Set up feature processor...
2022-08-12 17:05:58,079 P1479 INFO Load feature_map from json: ../data/Taobao/taobaoad_x1_2753db8a/feature_map.json
2022-08-12 17:05:58,079 P1479 INFO Set column index...
2022-08-12 17:05:58,079 P1479 INFO Feature specs: {
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
2022-08-12 17:06:03,634 P1479 INFO Total number of parameters: 41940929.
2022-08-12 17:06:03,635 P1479 INFO Loading data...
2022-08-12 17:06:03,635 P1479 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/train.h5
2022-08-12 17:06:28,258 P1479 INFO Train samples: total/21929911, blocks/1
2022-08-12 17:06:28,258 P1479 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/valid.h5
2022-08-12 17:06:31,861 P1479 INFO Validation samples: total/3099515, blocks/1
2022-08-12 17:06:31,861 P1479 INFO Loading train and validation data done.
2022-08-12 17:06:31,861 P1479 INFO Start training: 5354 batches/epoch
2022-08-12 17:06:31,861 P1479 INFO ************ Epoch=1 start ************
2022-08-12 17:31:03,220 P1479 INFO [Metrics] AUC: 0.641506 - gAUC: 0.566738
2022-08-12 17:31:03,447 P1479 INFO Save best model: monitor(max): 1.208244
2022-08-12 17:31:03,720 P1479 INFO --- 5354/5354 batches finished ---
2022-08-12 17:31:03,793 P1479 INFO Train loss: 0.205435
2022-08-12 17:31:03,793 P1479 INFO ************ Epoch=1 end ************
2022-08-12 17:55:52,252 P1479 INFO [Metrics] AUC: 0.647308 - gAUC: 0.570766
2022-08-12 17:55:52,507 P1479 INFO Save best model: monitor(max): 1.218073
2022-08-12 17:55:52,817 P1479 INFO --- 5354/5354 batches finished ---
2022-08-12 17:55:52,904 P1479 INFO Train loss: 0.202786
2022-08-12 17:55:52,904 P1479 INFO ************ Epoch=2 end ************
2022-08-12 18:20:55,458 P1479 INFO [Metrics] AUC: 0.648460 - gAUC: 0.571239
2022-08-12 18:20:55,709 P1479 INFO Save best model: monitor(max): 1.219699
2022-08-12 18:20:56,026 P1479 INFO --- 5354/5354 batches finished ---
2022-08-12 18:20:56,145 P1479 INFO Train loss: 0.201769
2022-08-12 18:20:56,145 P1479 INFO ************ Epoch=3 end ************
2022-08-12 18:46:03,774 P1479 INFO [Metrics] AUC: 0.648851 - gAUC: 0.572579
2022-08-12 18:46:04,044 P1479 INFO Save best model: monitor(max): 1.221430
2022-08-12 18:46:04,389 P1479 INFO --- 5354/5354 batches finished ---
2022-08-12 18:46:04,486 P1479 INFO Train loss: 0.201912
2022-08-12 18:46:04,486 P1479 INFO ************ Epoch=4 end ************
2022-08-12 19:11:41,330 P1479 INFO [Metrics] AUC: 0.649485 - gAUC: 0.572345
2022-08-12 19:11:41,606 P1479 INFO Save best model: monitor(max): 1.221830
2022-08-12 19:11:41,963 P1479 INFO --- 5354/5354 batches finished ---
2022-08-12 19:11:42,089 P1479 INFO Train loss: 0.202482
2022-08-12 19:11:42,090 P1479 INFO ************ Epoch=5 end ************
2022-08-12 19:36:35,115 P1479 INFO [Metrics] AUC: 0.650041 - gAUC: 0.573251
2022-08-12 19:36:35,407 P1479 INFO Save best model: monitor(max): 1.223292
2022-08-12 19:36:35,784 P1479 INFO --- 5354/5354 batches finished ---
2022-08-12 19:36:35,879 P1479 INFO Train loss: 0.203183
2022-08-12 19:36:35,880 P1479 INFO ************ Epoch=6 end ************
2022-08-12 20:01:37,714 P1479 INFO [Metrics] AUC: 0.648511 - gAUC: 0.570984
2022-08-12 20:01:37,957 P1479 INFO Monitor(max) STOP: 1.219495 !
2022-08-12 20:01:37,957 P1479 INFO Reduce learning rate on plateau: 0.000100
2022-08-12 20:01:37,957 P1479 INFO ********* Epoch==7 early stop *********
2022-08-12 20:01:37,958 P1479 INFO --- 5354/5354 batches finished ---
2022-08-12 20:01:38,059 P1479 INFO Train loss: 0.203955
2022-08-12 20:01:38,060 P1479 INFO Training finished.
2022-08-12 20:01:38,060 P1479 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/AOANet_taobaoad_x1/taobaoad_x1_2753db8a/AOANet_taobaoad_x1_005_26a70b74.model
2022-08-12 20:01:38,365 P1479 INFO ****** Validation evaluation ******
2022-08-12 20:04:15,560 P1479 INFO [Metrics] gAUC: 0.573251 - AUC: 0.650041 - logloss: 0.192513
2022-08-12 20:04:15,944 P1479 INFO ******** Test evaluation ********
2022-08-12 20:04:15,945 P1479 INFO Loading data...
2022-08-12 20:04:15,945 P1479 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/test.h5
2022-08-12 20:04:19,819 P1479 INFO Test samples: total/3099515, blocks/1
2022-08-12 20:04:19,819 P1479 INFO Loading test data done.
2022-08-12 20:07:02,810 P1479 INFO [Metrics] gAUC: 0.573251 - AUC: 0.650041 - logloss: 0.192513

```
