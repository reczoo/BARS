## APG_DCNv2_taobaoad_x1

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

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [APG_DCNv2_taobaoad_x1_tuner_config_01](./APG_DCNv2_taobaoad_x1_tuner_config_01). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/APG
    nohup python run_expid.py --config YOUR_PATH/APG/APG_DCNv2_taobaoad_x1_tuner_config_01 --expid APG_DCNv2_taobaoad_x1_030_5fbce754 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.575258 | 0.649595 | 0.192880  |


### Logs
```python
2023-05-31 23:26:32,257 P100601 INFO Params: {
    "batch_norm": "False",
    "batch_size": "8192",
    "condition_features": "None",
    "condition_mode": "self-wise",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobaoad_x1_2753db8a",
    "debug_mode": "False",
    "dnn_activations": "relu",
    "early_stop_patience": "1",
    "embedding_dim": "32",
    "embedding_regularizer": "5e-06",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(userid)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'pid', 'btag'], 'type': 'categorical'}, {'active': True, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'cate_his', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'brand_his', 'padding': 'pre', 'share_embedding': 'brand', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'btag_his', 'padding': 'pre', 'share_embedding': 'btag', 'splitter': '^', 'type': 'sequence'}]",
    "feature_config": "None",
    "feature_specs": "None",
    "generate_bias": "True",
    "gpu": "7",
    "group_id": "group_id",
    "hypernet_config": "{'dropout_rates': 0.1, 'hidden_activations': 'relu', 'hidden_units': []}",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "low_rank": "32",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "APG_DCNv2",
    "model_id": "APG_DCNv2_taobaoad_x1_030_5fbce754",
    "model_root": "./checkpoints/APG_DCNv2_taobaoad_x1/",
    "model_structure": "parallel",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "new_condition_emb": "False",
    "num_cross_layers": "4",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "overparam_p": "[128, 64, 32]",
    "parallel_dnn_hidden_units": "[512, 256, 128]",
    "pickle_feature_encoder": "True",
    "rank_k": "[32, 16, 8]",
    "save_best_only": "True",
    "seed": "20222023",
    "shuffle": "True",
    "stacked_dnn_hidden_units": "[500, 500, 500]",
    "task": "binary_classification",
    "test_data": "../data/Taobao/TaobaoAd_x1/test.csv",
    "train_data": "../data/Taobao/TaobaoAd_x1/train.csv",
    "use_features": "None",
    "use_low_rank_mixture": "False",
    "valid_data": "../data/Taobao/TaobaoAd_x1/test.csv",
    "verbose": "1"
}
2023-05-31 23:26:32,258 P100601 INFO Set up feature processor...
2023-05-31 23:26:32,259 P100601 WARNING Skip rebuilding ../data/Taobao/taobaoad_x1_2753db8a/feature_map.json. Please delete it manually if rebuilding is required.
2023-05-31 23:26:32,259 P100601 INFO Load feature_map from json: ../data/Taobao/taobaoad_x1_2753db8a/feature_map.json
2023-05-31 23:26:32,259 P100601 INFO Set column index...
2023-05-31 23:26:32,259 P100601 INFO Feature specs: {
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
2023-05-31 23:26:36,793 P100601 INFO Total number of parameters: 44600385.
2023-05-31 23:26:36,793 P100601 INFO Loading data...
2023-05-31 23:26:36,794 P100601 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/train.h5
2023-05-31 23:27:03,523 P100601 INFO Train samples: total/21929911, blocks/1
2023-05-31 23:27:03,523 P100601 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/valid.h5
2023-05-31 23:27:07,188 P100601 INFO Validation samples: total/3099515, blocks/1
2023-05-31 23:27:07,188 P100601 INFO Loading train and validation data done.
2023-05-31 23:27:07,188 P100601 INFO Start training: 2677 batches/epoch
2023-05-31 23:27:07,188 P100601 INFO ************ Epoch=1 start ************
2023-05-31 23:30:46,547 P100601 INFO Train loss: 0.204391
2023-05-31 23:30:46,548 P100601 INFO Evaluation @epoch 1 - batch 2677: 
2023-05-31 23:32:17,157 P100601 INFO [Metrics] AUC: 0.643445 - gAUC: 0.567565
2023-05-31 23:32:17,159 P100601 INFO Save best model: monitor(max)=1.211010
2023-05-31 23:32:17,488 P100601 INFO ************ Epoch=1 end ************
2023-05-31 23:35:43,502 P100601 INFO Train loss: 0.200060
2023-05-31 23:35:43,503 P100601 INFO Evaluation @epoch 2 - batch 2677: 
2023-05-31 23:37:18,926 P100601 INFO [Metrics] AUC: 0.646287 - gAUC: 0.572210
2023-05-31 23:37:18,928 P100601 INFO Save best model: monitor(max)=1.218497
2023-05-31 23:37:19,393 P100601 INFO ************ Epoch=2 end ************
2023-05-31 23:41:13,610 P100601 INFO Train loss: 0.198432
2023-05-31 23:41:13,611 P100601 INFO Evaluation @epoch 3 - batch 2677: 
2023-05-31 23:42:48,272 P100601 INFO [Metrics] AUC: 0.648910 - gAUC: 0.574236
2023-05-31 23:42:48,273 P100601 INFO Save best model: monitor(max)=1.223146
2023-05-31 23:42:48,781 P100601 INFO ************ Epoch=3 end ************
2023-05-31 23:48:06,450 P100601 INFO Train loss: 0.198764
2023-05-31 23:48:06,451 P100601 INFO Evaluation @epoch 4 - batch 2677: 
2023-05-31 23:49:40,605 P100601 INFO [Metrics] AUC: 0.649595 - gAUC: 0.575258
2023-05-31 23:49:40,606 P100601 INFO Save best model: monitor(max)=1.224853
2023-05-31 23:49:41,023 P100601 INFO ************ Epoch=4 end ************
2023-05-31 23:52:51,199 P100601 INFO Train loss: 0.199276
2023-05-31 23:52:51,200 P100601 INFO Evaluation @epoch 5 - batch 2677: 
2023-05-31 23:54:15,672 P100601 INFO [Metrics] AUC: 0.648382 - gAUC: 0.573086
2023-05-31 23:54:15,673 P100601 INFO Monitor(max)=1.221469 STOP!
2023-05-31 23:54:15,673 P100601 INFO Reduce learning rate on plateau: 0.000100
2023-05-31 23:54:15,673 P100601 INFO ********* Epoch==5 early stop *********
2023-05-31 23:54:15,799 P100601 INFO Training finished.
2023-05-31 23:54:15,799 P100601 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/APG_DCNv2_taobaoad_x1/taobaoad_x1_2753db8a/APG_DCNv2_taobaoad_x1_030_5fbce754.model
2023-05-31 23:54:15,941 P100601 INFO ****** Validation evaluation ******
2023-05-31 23:55:50,427 P100601 INFO [Metrics] gAUC: 0.575258 - AUC: 0.649595 - logloss: 0.192880
2023-05-31 23:55:50,613 P100601 INFO ******** Test evaluation ********
2023-05-31 23:55:50,613 P100601 INFO Loading data...
2023-05-31 23:55:50,613 P100601 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/test.h5
2023-05-31 23:55:54,481 P100601 INFO Test samples: total/3099515, blocks/1
2023-05-31 23:55:54,481 P100601 INFO Loading test data done.
2023-05-31 23:57:27,124 P100601 INFO [Metrics] gAUC: 0.575258 - AUC: 0.649595 - logloss: 0.192880

```
