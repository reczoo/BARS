## BST_taobaoad_x1

A hands-on guide to run the BST model on the TaobaoAd_x1 dataset.

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

We use the [BST](https://github.com/xue-pai/FuxiCTR/blob/v2.0.1/model_zoo/BST) model code from [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Taobao/TaobaoAd_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [BST_taobaoad_x1_tuner_config_03](./BST_taobaoad_x1_tuner_config_03). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/BST
    nohup python run_expid.py --config XXX/benchmarks/BST/BST_taobaoad_x1_tuner_config_03 --expid BST_taobaoad_x1_021_e30ae99a --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.576304 | 0.651131 | 0.192842  |


### Logs
```python
2022-09-08 00:41:26,292 P5446 INFO Params: {
    "attention_dropout": "0.2",
    "batch_norm": "False",
    "batch_size": "8192",
    "bst_sequence_field": "('cate_his', 'brand_his', 'btag_his')",
    "bst_target_field": "('cate_id', 'brand', 'btag')",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobaoad_x1_2753db8a",
    "debug_mode": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[512, 256, 128]",
    "early_stop_patience": "1",
    "embedding_dim": "32",
    "embedding_regularizer": "5e-06",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(userid)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'pid', 'btag'], 'type': 'categorical'}, {'active': True, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'cate_his', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'brand_his', 'padding': 'pre', 'share_embedding': 'brand', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'btag_his', 'padding': 'pre', 'share_embedding': 'btag', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "[{'feature_encoder': None, 'name': ['cate_his', 'brand_his', 'btag_his']}]",
    "gpu": "7",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "BST",
    "model_id": "BST_taobaoad_x1_021_e30ae99a",
    "model_root": "./checkpoints/BST_taobaoad_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_heads": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "20222023",
    "seq_pooling_type": "mean",
    "shuffle": "True",
    "stacked_transformer_layers": "1",
    "task": "binary_classification",
    "test_data": "../data/Taobao/TaobaoAd_x1/test.csv",
    "train_data": "../data/Taobao/TaobaoAd_x1/train.csv",
    "use_causal_mask": "False",
    "use_position_emb": "True",
    "use_residual": "True",
    "valid_data": "../data/Taobao/TaobaoAd_x1/test.csv",
    "verbose": "1"
}
2022-09-08 00:41:26,293 P5446 INFO Set up feature processor...
2022-09-08 00:41:26,294 P5446 INFO Load feature_map from json: ../data/Taobao/taobaoad_x1_2753db8a/feature_map.json
2022-09-08 00:41:26,294 P5446 INFO Set column index...
2022-09-08 00:41:26,294 P5446 INFO Feature specs: {
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
2022-09-08 00:41:31,189 P5446 INFO Total number of parameters: 42052065.
2022-09-08 00:41:31,189 P5446 INFO Loading data...
2022-09-08 00:41:31,189 P5446 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/train.h5
2022-09-08 00:42:00,325 P5446 INFO Train samples: total/21929911, blocks/1
2022-09-08 00:42:00,325 P5446 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/valid.h5
2022-09-08 00:42:04,524 P5446 INFO Validation samples: total/3099515, blocks/1
2022-09-08 00:42:04,524 P5446 INFO Loading train and validation data done.
2022-09-08 00:42:04,524 P5446 INFO Start training: 2677 batches/epoch
2022-09-08 00:42:04,524 P5446 INFO ************ Epoch=1 start ************
2022-09-08 01:03:36,380 P5446 INFO [Metrics] AUC: 0.641308 - gAUC: 0.564395
2022-09-08 01:03:36,748 P5446 INFO Save best model: monitor(max): 1.205703
2022-09-08 01:03:37,060 P5446 INFO --- 2677/2677 batches finished ---
2022-09-08 01:03:37,168 P5446 INFO Train loss: 0.204707
2022-09-08 01:03:37,168 P5446 INFO ************ Epoch=1 end ************
2022-09-08 01:24:36,364 P5446 INFO [Metrics] AUC: 0.648767 - gAUC: 0.570378
2022-09-08 01:24:36,675 P5446 INFO Save best model: monitor(max): 1.219145
2022-09-08 01:24:37,000 P5446 INFO --- 2677/2677 batches finished ---
2022-09-08 01:24:37,128 P5446 INFO Train loss: 0.202000
2022-09-08 01:24:37,128 P5446 INFO ************ Epoch=2 end ************
2022-09-08 01:45:33,037 P5446 INFO [Metrics] AUC: 0.650646 - gAUC: 0.574043
2022-09-08 01:45:33,367 P5446 INFO Save best model: monitor(max): 1.224688
2022-09-08 01:45:33,704 P5446 INFO --- 2677/2677 batches finished ---
2022-09-08 01:45:33,935 P5446 INFO Train loss: 0.199897
2022-09-08 01:45:33,936 P5446 INFO ************ Epoch=3 end ************
2022-09-08 02:02:52,526 P5446 INFO [Metrics] AUC: 0.650397 - gAUC: 0.574924
2022-09-08 02:02:52,890 P5446 INFO Save best model: monitor(max): 1.225321
2022-09-08 02:02:53,219 P5446 INFO --- 2677/2677 batches finished ---
2022-09-08 02:02:53,397 P5446 INFO Train loss: 0.199440
2022-09-08 02:02:53,398 P5446 INFO ************ Epoch=4 end ************
2022-09-08 02:16:21,316 P5446 INFO [Metrics] AUC: 0.651131 - gAUC: 0.576304
2022-09-08 02:16:21,616 P5446 INFO Save best model: monitor(max): 1.227435
2022-09-08 02:16:22,065 P5446 INFO --- 2677/2677 batches finished ---
2022-09-08 02:16:22,210 P5446 INFO Train loss: 0.199731
2022-09-08 02:16:22,211 P5446 INFO ************ Epoch=5 end ************
2022-09-08 02:28:57,280 P5446 INFO [Metrics] AUC: 0.650616 - gAUC: 0.575283
2022-09-08 02:28:57,569 P5446 INFO Monitor(max) STOP: 1.225899 !
2022-09-08 02:28:57,570 P5446 INFO Reduce learning rate on plateau: 0.000100
2022-09-08 02:28:57,570 P5446 INFO ********* Epoch==6 early stop *********
2022-09-08 02:28:57,571 P5446 INFO --- 2677/2677 batches finished ---
2022-09-08 02:28:57,708 P5446 INFO Train loss: 0.200221
2022-09-08 02:28:57,708 P5446 INFO Training finished.
2022-09-08 02:28:57,708 P5446 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/BST_taobaoad_x1/taobaoad_x1_2753db8a/BST_taobaoad_x1_021_e30ae99a.model
2022-09-08 02:28:58,223 P5446 INFO ****** Validation evaluation ******
2022-09-08 02:31:18,867 P5446 INFO [Metrics] gAUC: 0.576304 - AUC: 0.651131 - logloss: 0.192842
2022-09-08 02:31:19,334 P5446 INFO ******** Test evaluation ********
2022-09-08 02:31:19,335 P5446 INFO Loading data...
2022-09-08 02:31:19,335 P5446 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/test.h5
2022-09-08 02:31:23,638 P5446 INFO Test samples: total/3099515, blocks/1
2022-09-08 02:31:23,638 P5446 INFO Loading test data done.
2022-09-08 02:33:45,056 P5446 INFO [Metrics] gAUC: 0.576304 - AUC: 0.651131 - logloss: 0.192842

```
