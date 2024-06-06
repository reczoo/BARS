## APG_DeepFM_taobaoad_x1

A hands-on guide to run the APG model on the TaobaoAd_x1 dataset.

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
  fuxictr: 2.0.3

  ```

### Dataset
Please refer to [TaobaoAd_x1](https://github.com/reczoo/Datasets/tree/main/Taobao/TaobaoAd_x1) to get the dataset details.

### Code

We use the [APG](https://github.com/reczoo/FuxiCTR/tree/v2.0.3/model_zoo/APG) model code from [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/tree/v2.0.3) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.3.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.3
    ```

2. Create a data directory and put the downloaded data files in `../data/Taobao/TaobaoAd_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [APG_DeepFM_taobaoad_x1_tuner_config_04](./APG_DeepFM_taobaoad_x1_tuner_config_04). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/APG
    nohup python run_expid.py --config YOUR_PATH/APG/APG_DeepFM_taobaoad_x1_tuner_config_04 --expid APG_DeepFM_taobaoad_x1_003_d01b9650 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.569266 | 0.640834 | 0.196771  |


### Logs
```python
2023-05-31 09:47:22,882 P26433 INFO Params: {
    "batch_norm": "False",
    "batch_size": "4096",
    "condition_features": "None",
    "condition_mode": "self-wise",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobaoad_x1_2753db8a",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "32",
    "embedding_regularizer": "5e-06",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(userid)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'pid', 'btag'], 'type': 'categorical'}, {'active': True, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'cate_his', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'brand_his', 'padding': 'pre', 'share_embedding': 'brand', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'btag_his', 'padding': 'pre', 'share_embedding': 'btag', 'splitter': '^', 'type': 'sequence'}]",
    "feature_config": "None",
    "feature_specs": "None",
    "generate_bias": "True",
    "gpu": "2",
    "group_id": "group_id",
    "hidden_activations": "relu",
    "hidden_units": "[512, 256, 128]",
    "hypernet_config": "{'dropout_rates': 0, 'hidden_activations': 'relu', 'hidden_units': [512]}",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "APG_DeepFM",
    "model_id": "APG_DeepFM_taobaoad_x1_003_d01b9650",
    "model_root": "./checkpoints/APG_DeepFM_taobaoad_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "new_condition_emb": "False",
    "num_workers": "3",
    "optimizer": "adam",
    "overparam_p": "None",
    "pickle_feature_encoder": "True",
    "rank_k": "[128, 64, 32]",
    "save_best_only": "True",
    "seed": "20222023",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Taobao/TaobaoAd_x1/test.csv",
    "train_data": "../data/Taobao/TaobaoAd_x1/train.csv",
    "use_features": "None",
    "valid_data": "../data/Taobao/TaobaoAd_x1/test.csv",
    "verbose": "1"
}
2023-05-31 09:47:22,882 P26433 INFO Set up feature processor...
2023-05-31 09:47:22,883 P26433 WARNING Skip rebuilding ../data/Taobao/taobaoad_x1_2753db8a/feature_map.json. Please delete it manually if rebuilding is required.
2023-05-31 09:47:22,883 P26433 INFO Load feature_map from json: ../data/Taobao/taobaoad_x1_2753db8a/feature_map.json
2023-05-31 09:47:22,883 P26433 INFO Set column index...
2023-05-31 09:47:22,883 P26433 INFO Feature specs: {
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
2023-05-31 09:47:27,774 P26433 INFO Total number of parameters: 55480372.
2023-05-31 09:47:27,775 P26433 INFO Loading data...
2023-05-31 09:47:27,775 P26433 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/train.h5
2023-05-31 09:47:55,417 P26433 INFO Train samples: total/21929911, blocks/1
2023-05-31 09:47:55,417 P26433 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/valid.h5
2023-05-31 09:47:59,190 P26433 INFO Validation samples: total/3099515, blocks/1
2023-05-31 09:47:59,190 P26433 INFO Loading train and validation data done.
2023-05-31 09:47:59,190 P26433 INFO Start training: 5354 batches/epoch
2023-05-31 09:47:59,190 P26433 INFO ************ Epoch=1 start ************
2023-05-31 09:55:29,060 P26433 INFO Train loss: 0.207752
2023-05-31 09:55:29,060 P26433 INFO Evaluation @epoch 1 - batch 5354: 
2023-05-31 09:57:01,042 P26433 INFO [Metrics] AUC: 0.622413 - gAUC: 0.557634
2023-05-31 09:57:01,048 P26433 INFO Save best model: monitor(max)=1.180048
2023-05-31 09:57:01,523 P26433 INFO ************ Epoch=1 end ************
2023-05-31 10:04:34,097 P26433 INFO Train loss: 0.206166
2023-05-31 10:04:34,097 P26433 INFO Evaluation @epoch 2 - batch 5354: 
2023-05-31 10:06:01,343 P26433 INFO [Metrics] AUC: 0.626363 - gAUC: 0.562221
2023-05-31 10:06:01,344 P26433 INFO Save best model: monitor(max)=1.188584
2023-05-31 10:06:01,844 P26433 INFO ************ Epoch=2 end ************
2023-05-31 10:13:33,139 P26433 INFO Train loss: 0.205646
2023-05-31 10:13:33,139 P26433 INFO Evaluation @epoch 3 - batch 5354: 
2023-05-31 10:15:09,268 P26433 INFO [Metrics] AUC: 0.630111 - gAUC: 0.564107
2023-05-31 10:15:09,270 P26433 INFO Save best model: monitor(max)=1.194218
2023-05-31 10:15:09,804 P26433 INFO ************ Epoch=3 end ************
2023-05-31 10:22:45,585 P26433 INFO Train loss: 0.205778
2023-05-31 10:22:45,585 P26433 INFO Evaluation @epoch 4 - batch 5354: 
2023-05-31 10:24:18,808 P26433 INFO [Metrics] AUC: 0.627708 - gAUC: 0.563474
2023-05-31 10:24:18,809 P26433 INFO Monitor(max)=1.191182 STOP!
2023-05-31 10:24:18,809 P26433 INFO Reduce learning rate on plateau: 0.000100
2023-05-31 10:24:18,931 P26433 INFO ************ Epoch=4 end ************
2023-05-31 10:31:58,131 P26433 INFO Train loss: 0.192036
2023-05-31 10:31:58,131 P26433 INFO Evaluation @epoch 5 - batch 5354: 
2023-05-31 10:33:30,682 P26433 INFO [Metrics] AUC: 0.640834 - gAUC: 0.569266
2023-05-31 10:33:30,683 P26433 INFO Save best model: monitor(max)=1.210100
2023-05-31 10:33:31,148 P26433 INFO ************ Epoch=5 end ************
2023-05-31 10:41:11,836 P26433 INFO Train loss: 0.186426
2023-05-31 10:41:11,836 P26433 INFO Evaluation @epoch 6 - batch 5354: 
2023-05-31 10:42:48,687 P26433 INFO [Metrics] AUC: 0.637207 - gAUC: 0.565047
2023-05-31 10:42:48,689 P26433 INFO Monitor(max)=1.202254 STOP!
2023-05-31 10:42:48,689 P26433 INFO Reduce learning rate on plateau: 0.000010
2023-05-31 10:42:48,784 P26433 INFO ************ Epoch=6 end ************
2023-05-31 10:50:23,231 P26433 INFO Train loss: 0.166509
2023-05-31 10:50:23,231 P26433 INFO Evaluation @epoch 7 - batch 5354: 
2023-05-31 10:52:02,384 P26433 INFO [Metrics] AUC: 0.621497 - gAUC: 0.553312
2023-05-31 10:52:02,385 P26433 INFO Monitor(max)=1.174809 STOP!
2023-05-31 10:52:02,385 P26433 INFO Reduce learning rate on plateau: 0.000001
2023-05-31 10:52:02,385 P26433 INFO ********* Epoch==7 early stop *********
2023-05-31 10:52:02,500 P26433 INFO Training finished.
2023-05-31 10:52:02,501 P26433 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/APG_DeepFM_taobaoad_x1/taobaoad_x1_2753db8a/APG_DeepFM_taobaoad_x1_003_d01b9650.model
2023-05-31 10:52:02,727 P26433 INFO ****** Validation evaluation ******
2023-05-31 10:53:29,702 P26433 INFO [Metrics] gAUC: 0.569266 - AUC: 0.640834 - logloss: 0.196771
2023-05-31 10:53:29,871 P26433 INFO ******** Test evaluation ********
2023-05-31 10:53:29,872 P26433 INFO Loading data...
2023-05-31 10:53:29,872 P26433 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/test.h5
2023-05-31 10:53:33,937 P26433 INFO Test samples: total/3099515, blocks/1
2023-05-31 10:53:33,937 P26433 INFO Loading test data done.
2023-05-31 10:55:12,912 P26433 INFO [Metrics] gAUC: 0.569266 - AUC: 0.640834 - logloss: 0.196771

```
