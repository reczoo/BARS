## DeepFM_taobaoad_x1

A hands-on guide to run the DeepFM model on the TaobaoAd_x1 dataset.

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
Please refer to [TaobaoAd_x1](https://github.com/reczoo/Datasets/tree/main/Taobao/TaobaoAd_x1) to get the dataset details.

### Code

We use the [DeepFM](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/DeepFM) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Taobao/TaobaoAd_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DeepFM_taobaoad_x1_tuner_config_01](./DeepFM_taobaoad_x1_tuner_config_01). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DeepFM
    nohup python run_expid.py --config XXX/benchmarks/DeepFM/DeepFM_taobaoad_x1_tuner_config_01 --expid DeepFM_taobaoad_x1_009_afab5940 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.569739 | 0.636426 | 0.196501  |


### Logs
```python
2022-08-11 23:19:47,028 P81583 INFO Params: {
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
    "model": "DeepFM",
    "model_id": "DeepFM_taobaoad_x1_009_afab5940",
    "model_root": "./checkpoints/DeepFM_taobaoad_x1/",
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
2022-08-11 23:19:47,028 P81583 INFO Set up feature processor...
2022-08-11 23:19:47,029 P81583 INFO Load feature_map from json: ../data/Taobao/taobaoad_x1_2753db8a/feature_map.json
2022-08-11 23:19:47,029 P81583 INFO Set column index...
2022-08-11 23:19:47,029 P81583 INFO Feature specs: {
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
2022-08-11 23:19:54,241 P81583 INFO Total number of parameters: 43550260.
2022-08-11 23:19:54,241 P81583 INFO Loading data...
2022-08-11 23:19:54,241 P81583 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/train.h5
2022-08-11 23:24:57,741 P81583 INFO Train samples: total/21929911, blocks/1
2022-08-11 23:24:57,741 P81583 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/valid.h5
2022-08-11 23:25:01,600 P81583 INFO Validation samples: total/3099515, blocks/1
2022-08-11 23:25:01,600 P81583 INFO Loading train and validation data done.
2022-08-11 23:25:01,600 P81583 INFO Start training: 2677 batches/epoch
2022-08-11 23:25:01,600 P81583 INFO ************ Epoch=1 start ************
2022-08-11 23:44:08,953 P81583 INFO [Metrics] AUC: 0.629708 - gAUC: 0.563763
2022-08-11 23:44:09,254 P81583 INFO Save best model: monitor(max): 1.193471
2022-08-11 23:44:09,522 P81583 INFO --- 2677/2677 batches finished ---
2022-08-11 23:44:09,814 P81583 INFO Train loss: 0.205954
2022-08-11 23:44:09,814 P81583 INFO ************ Epoch=1 end ************
2022-08-12 00:03:21,527 P81583 INFO [Metrics] AUC: 0.634989 - gAUC: 0.567199
2022-08-12 00:03:21,814 P81583 INFO Save best model: monitor(max): 1.202189
2022-08-12 00:03:22,189 P81583 INFO --- 2677/2677 batches finished ---
2022-08-12 00:03:22,503 P81583 INFO Train loss: 0.203679
2022-08-12 00:03:22,503 P81583 INFO ************ Epoch=2 end ************
2022-08-12 00:22:28,583 P81583 INFO [Metrics] AUC: 0.635269 - gAUC: 0.569396
2022-08-12 00:22:28,940 P81583 INFO Save best model: monitor(max): 1.204666
2022-08-12 00:22:29,411 P81583 INFO --- 2677/2677 batches finished ---
2022-08-12 00:22:29,737 P81583 INFO Train loss: 0.202543
2022-08-12 00:22:29,737 P81583 INFO ************ Epoch=3 end ************
2022-08-12 00:41:28,302 P81583 INFO [Metrics] AUC: 0.636580 - gAUC: 0.568187
2022-08-12 00:41:28,786 P81583 INFO Save best model: monitor(max): 1.204767
2022-08-12 00:41:29,165 P81583 INFO --- 2677/2677 batches finished ---
2022-08-12 00:41:29,513 P81583 INFO Train loss: 0.202689
2022-08-12 00:41:29,514 P81583 INFO ************ Epoch=4 end ************
2022-08-12 00:59:06,375 P81583 INFO [Metrics] AUC: 0.636426 - gAUC: 0.569739
2022-08-12 00:59:06,703 P81583 INFO Save best model: monitor(max): 1.206164
2022-08-12 00:59:07,115 P81583 INFO --- 2677/2677 batches finished ---
2022-08-12 00:59:07,453 P81583 INFO Train loss: 0.203714
2022-08-12 00:59:07,454 P81583 INFO ************ Epoch=5 end ************
2022-08-12 01:16:15,415 P81583 INFO [Metrics] AUC: 0.633918 - gAUC: 0.566834
2022-08-12 01:16:15,704 P81583 INFO Monitor(max) STOP: 1.200752 !
2022-08-12 01:16:15,704 P81583 INFO Reduce learning rate on plateau: 0.000100
2022-08-12 01:16:15,705 P81583 INFO ********* Epoch==6 early stop *********
2022-08-12 01:16:15,706 P81583 INFO --- 2677/2677 batches finished ---
2022-08-12 01:16:16,070 P81583 INFO Train loss: 0.205098
2022-08-12 01:16:16,070 P81583 INFO Training finished.
2022-08-12 01:16:16,070 P81583 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DeepFM_taobaoad_x1/taobaoad_x1_2753db8a/DeepFM_taobaoad_x1_009_afab5940.model
2022-08-12 01:16:16,236 P81583 INFO ****** Validation evaluation ******
2022-08-12 01:19:40,864 P81583 INFO [Metrics] gAUC: 0.569739 - AUC: 0.636426 - logloss: 0.196501
2022-08-12 01:19:42,484 P81583 INFO ******** Test evaluation ********
2022-08-12 01:19:42,484 P81583 INFO Loading data...
2022-08-12 01:19:42,484 P81583 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/test.h5
2022-08-12 01:19:46,933 P81583 INFO Test samples: total/3099515, blocks/1
2022-08-12 01:19:46,933 P81583 INFO Loading test data done.
2022-08-12 01:22:56,266 P81583 INFO [Metrics] gAUC: 0.569739 - AUC: 0.636426 - logloss: 0.196501

```
