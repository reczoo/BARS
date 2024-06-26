## DMIN_taobaoad_x1

A hands-on guide to run the DMIN model on the TaobaoAd_x1 dataset.

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

We use the [DMIN](https://github.com/reczoo/FuxiCTR/tree/v2.0.3/model_zoo/DMIN) model code from [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/tree/v2.0.3) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.3.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.3
    ```

2. Create a data directory and put the downloaded data files in `../data/Taobao/TaobaoAd_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DMIN_taobaoad_x1_tuner_config_05](./DMIN_taobaoad_x1_tuner_config_05). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DMIN
    nohup python run_expid.py --config YOUR_PATH/DMIN/DMIN_taobaoad_x1_tuner_config_05 --expid DMIN_taobaoad_x1_011_0be611e8 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.577000 | 0.651103 | 0.192813  |


### Logs
```python
2023-05-23 11:38:17,711 P27781 INFO Params: {
    "attention_activation": "ReLU",
    "attention_dropout": "0.2",
    "attention_hidden_units": "[512, 256]",
    "aux_activation": "ReLU",
    "aux_hidden_units": "[100, 50]",
    "aux_loss_lambda": "0",
    "batch_norm": "False",
    "batch_size": "8192",
    "bn_only_once": "False",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobaoad_x1_bf8c47ea",
    "debug_mode": "False",
    "dnn_activations": "ReLU",
    "dnn_hidden_units": "[512, 256, 128]",
    "early_stop_patience": "2",
    "embedding_dim": "32",
    "embedding_regularizer": "5e-06",
    "enable_sum_pooling": "False",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(userid)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'pid', 'btag'], 'type': 'categorical'}, {'active': True, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'cate_his', 'padding': 'post', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'brand_his', 'padding': 'post', 'share_embedding': 'brand', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'btag_his', 'padding': 'post', 'share_embedding': 'btag', 'splitter': '^', 'type': 'sequence'}]",
    "feature_config": "None",
    "feature_specs": "[{'feature_encoder': None, 'name': ['cate_his', 'brand_his', 'btag_his']}]",
    "gpu": "0",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "layer_norm": "True",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "DMIN",
    "model_id": "DMIN_taobaoad_x1_011_0be611e8",
    "model_root": "./checkpoints/DMIN_taobaoad_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "neg_seq_field": "None",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "1",
    "num_workers": "3",
    "optimizer": "adam",
    "pickle_feature_encoder": "True",
    "pos_emb_dim": "4",
    "save_best_only": "True",
    "seed": "20222023",
    "sequence_field": "('cate_his', 'brand_his', 'btag_his')",
    "shuffle": "True",
    "target_field": "('cate_id', 'brand', 'btag')",
    "task": "binary_classification",
    "test_data": "../data/Taobao/TaobaoAd_x1/test.csv",
    "train_data": "../data/Taobao/TaobaoAd_x1/train.csv",
    "use_behavior_refiner": "False",
    "use_features": "None",
    "use_pos_emb": "True",
    "valid_data": "../data/Taobao/TaobaoAd_x1/test.csv",
    "verbose": "1"
}
2023-05-23 11:38:17,713 P27781 INFO Set up feature processor...
2023-05-23 11:38:17,713 P27781 WARNING Skip rebuilding ../data/Taobao/taobaoad_x1_bf8c47ea/feature_map.json. Please delete it manually if rebuilding is required.
2023-05-23 11:38:17,713 P27781 INFO Load feature_map from json: ../data/Taobao/taobaoad_x1_bf8c47ea/feature_map.json
2023-05-23 11:38:17,714 P27781 INFO Set column index...
2023-05-23 11:38:17,714 P27781 INFO Feature specs: {
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
2023-05-23 11:38:21,653 P27781 INFO Total number of parameters: 42347786.
2023-05-23 11:38:21,654 P27781 INFO Loading data...
2023-05-23 11:38:21,654 P27781 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_bf8c47ea/train.h5
2023-05-23 11:38:49,862 P27781 INFO Train samples: total/21929911, blocks/1
2023-05-23 11:38:49,862 P27781 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_bf8c47ea/valid.h5
2023-05-23 11:38:53,632 P27781 INFO Validation samples: total/3099515, blocks/1
2023-05-23 11:38:53,632 P27781 INFO Loading train and validation data done.
2023-05-23 11:38:53,632 P27781 INFO Start training: 2677 batches/epoch
2023-05-23 11:38:53,632 P27781 INFO ************ Epoch=1 start ************
2023-05-23 11:47:12,354 P27781 INFO Train loss: 0.204894
2023-05-23 11:47:12,354 P27781 INFO Evaluation @epoch 1 - batch 2677: 
2023-05-23 11:48:58,068 P27781 INFO [Metrics] AUC: 0.639346 - gAUC: 0.564971
2023-05-23 11:48:58,069 P27781 INFO Save best model: monitor(max)=1.204317
2023-05-23 11:48:58,389 P27781 INFO ************ Epoch=1 end ************
2023-05-23 11:57:36,966 P27781 INFO Train loss: 0.201418
2023-05-23 11:57:36,966 P27781 INFO Evaluation @epoch 2 - batch 2677: 
2023-05-23 11:59:20,585 P27781 INFO [Metrics] AUC: 0.650592 - gAUC: 0.572662
2023-05-23 11:59:20,587 P27781 INFO Save best model: monitor(max)=1.223254
2023-05-23 11:59:21,045 P27781 INFO ************ Epoch=2 end ************
2023-05-23 12:07:36,123 P27781 INFO Train loss: 0.198432
2023-05-23 12:07:36,124 P27781 INFO Evaluation @epoch 3 - batch 2677: 
2023-05-23 12:09:18,991 P27781 INFO [Metrics] AUC: 0.651607 - gAUC: 0.575599
2023-05-23 12:09:18,993 P27781 INFO Save best model: monitor(max)=1.227206
2023-05-23 12:09:19,407 P27781 INFO ************ Epoch=3 end ************
2023-05-23 12:17:38,650 P27781 INFO Train loss: 0.197723
2023-05-23 12:17:38,651 P27781 INFO Evaluation @epoch 4 - batch 2677: 
2023-05-23 12:19:16,429 P27781 INFO [Metrics] AUC: 0.651103 - gAUC: 0.577000
2023-05-23 12:19:16,430 P27781 INFO Save best model: monitor(max)=1.228104
2023-05-23 12:19:16,865 P27781 INFO ************ Epoch=4 end ************
2023-05-23 12:27:46,877 P27781 INFO Train loss: 0.197705
2023-05-23 12:27:46,877 P27781 INFO Evaluation @epoch 5 - batch 2677: 
2023-05-23 12:29:25,193 P27781 INFO [Metrics] AUC: 0.651449 - gAUC: 0.575664
2023-05-23 12:29:25,194 P27781 INFO Monitor(max)=1.227113 STOP!
2023-05-23 12:29:25,194 P27781 INFO Reduce learning rate on plateau: 0.000100
2023-05-23 12:29:25,288 P27781 INFO ************ Epoch=5 end ************
2023-05-23 12:37:40,904 P27781 INFO Train loss: 0.184534
2023-05-23 12:37:40,904 P27781 INFO Evaluation @epoch 6 - batch 2677: 
2023-05-23 12:39:17,180 P27781 INFO [Metrics] AUC: 0.644481 - gAUC: 0.572521
2023-05-23 12:39:17,181 P27781 INFO Monitor(max)=1.217002 STOP!
2023-05-23 12:39:17,181 P27781 INFO Reduce learning rate on plateau: 0.000010
2023-05-23 12:39:17,182 P27781 INFO ********* Epoch==6 early stop *********
2023-05-23 12:39:17,284 P27781 INFO Training finished.
2023-05-23 12:39:17,284 P27781 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DMIN_taobaoad_x1/taobaoad_x1_bf8c47ea/DMIN_taobaoad_x1_011_0be611e8.model
2023-05-23 12:39:17,452 P27781 INFO ****** Validation evaluation ******
2023-05-23 12:40:52,661 P27781 INFO [Metrics] gAUC: 0.577000 - AUC: 0.651103 - logloss: 0.192813
2023-05-23 12:40:52,807 P27781 INFO ******** Test evaluation ********
2023-05-23 12:40:52,807 P27781 INFO Loading data...
2023-05-23 12:40:52,807 P27781 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_bf8c47ea/test.h5
2023-05-23 12:40:57,028 P27781 INFO Test samples: total/3099515, blocks/1
2023-05-23 12:40:57,029 P27781 INFO Loading test data done.
2023-05-23 12:42:30,215 P27781 INFO [Metrics] gAUC: 0.577000 - AUC: 0.651103 - logloss: 0.192813

```
