## AutoInt_taobaoad_x1

A hands-on guide to run the AutoInt model on the TaobaoAd_x1 dataset.

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

We use the [AutoInt](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/AutoInt) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Taobao/TaobaoAd_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt+_taobaoad_x1_tuner_config_02](./AutoInt+_taobaoad_x1_tuner_config_02). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/AutoInt
    nohup python run_expid.py --config XXX/benchmarks/AutoInt/AutoInt+_taobaoad_x1_tuner_config_02 --expid AutoInt_taobaoad_x1_012_778cf7e5 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.574437 | 0.648629 | 0.193020  |


### Logs
```python
2022-08-20 21:22:32,206 P53256 INFO Params: {
    "attention_dim": "128",
    "attention_layers": "3",
    "batch_norm": "False",
    "batch_size": "8192",
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
    "feature_specs": "None",
    "gpu": "3",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "AutoInt",
    "model_id": "AutoInt_taobaoad_x1_012_778cf7e5",
    "model_root": "./checkpoints/AutoInt_taobaoad_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_heads": "1",
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
    "use_residual": "True",
    "use_scale": "False",
    "use_wide": "False",
    "valid_data": "../data/Taobao/TaobaoAd_x1/test.csv",
    "verbose": "1"
}
2022-08-20 21:22:32,207 P53256 INFO Set up feature processor...
2022-08-20 21:22:32,207 P53256 INFO Load feature_map from json: ../data/Taobao/taobaoad_x1_2753db8a/feature_map.json
2022-08-20 21:22:32,207 P53256 INFO Set column index...
2022-08-20 21:22:32,207 P53256 INFO Feature specs: {
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
2022-08-20 21:22:41,359 P53256 INFO Total number of parameters: 42052226.
2022-08-20 21:22:41,359 P53256 INFO Loading data...
2022-08-20 21:22:41,360 P53256 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/train.h5
2022-08-20 21:23:40,558 P53256 INFO Train samples: total/21929911, blocks/1
2022-08-20 21:23:40,559 P53256 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/valid.h5
2022-08-20 21:23:45,697 P53256 INFO Validation samples: total/3099515, blocks/1
2022-08-20 21:23:45,697 P53256 INFO Loading train and validation data done.
2022-08-20 21:23:45,698 P53256 INFO Start training: 2677 batches/epoch
2022-08-20 21:23:45,698 P53256 INFO ************ Epoch=1 start ************
2022-08-20 21:41:18,011 P53256 INFO [Metrics] AUC: 0.641299 - gAUC: 0.566756
2022-08-20 21:41:18,306 P53256 INFO Save best model: monitor(max): 1.208055
2022-08-20 21:41:18,636 P53256 INFO --- 2677/2677 batches finished ---
2022-08-20 21:41:18,749 P53256 INFO Train loss: 0.205843
2022-08-20 21:41:18,749 P53256 INFO ************ Epoch=1 end ************
2022-08-20 21:58:51,128 P53256 INFO [Metrics] AUC: 0.646146 - gAUC: 0.571447
2022-08-20 21:58:51,480 P53256 INFO Save best model: monitor(max): 1.217592
2022-08-20 21:58:51,885 P53256 INFO --- 2677/2677 batches finished ---
2022-08-20 21:58:52,091 P53256 INFO Train loss: 0.203748
2022-08-20 21:58:52,091 P53256 INFO ************ Epoch=2 end ************
2022-08-20 22:16:31,706 P53256 INFO [Metrics] AUC: 0.647586 - gAUC: 0.572421
2022-08-20 22:16:32,190 P53256 INFO Save best model: monitor(max): 1.220007
2022-08-20 22:16:32,512 P53256 INFO --- 2677/2677 batches finished ---
2022-08-20 22:16:32,655 P53256 INFO Train loss: 0.201247
2022-08-20 22:16:32,655 P53256 INFO ************ Epoch=3 end ************
2022-08-20 22:33:30,844 P53256 INFO [Metrics] AUC: 0.648629 - gAUC: 0.574437
2022-08-20 22:33:31,151 P53256 INFO Save best model: monitor(max): 1.223065
2022-08-20 22:33:31,540 P53256 INFO --- 2677/2677 batches finished ---
2022-08-20 22:33:31,727 P53256 INFO Train loss: 0.202221
2022-08-20 22:33:31,727 P53256 INFO ************ Epoch=4 end ************
2022-08-20 22:48:37,270 P53256 INFO [Metrics] AUC: 0.648068 - gAUC: 0.573248
2022-08-20 22:48:37,584 P53256 INFO Monitor(max) STOP: 1.221317 !
2022-08-20 22:48:37,584 P53256 INFO Reduce learning rate on plateau: 0.000100
2022-08-20 22:48:37,584 P53256 INFO ********* Epoch==5 early stop *********
2022-08-20 22:48:37,585 P53256 INFO --- 2677/2677 batches finished ---
2022-08-20 22:48:37,758 P53256 INFO Train loss: 0.204838
2022-08-20 22:48:37,758 P53256 INFO Training finished.
2022-08-20 22:48:37,758 P53256 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/AutoInt_taobaoad_x1/taobaoad_x1_2753db8a/AutoInt_taobaoad_x1_012_778cf7e5.model
2022-08-20 22:48:38,147 P53256 INFO ****** Validation evaluation ******
2022-08-20 22:51:29,465 P53256 INFO [Metrics] gAUC: 0.574437 - AUC: 0.648629 - logloss: 0.193020
2022-08-20 22:51:30,237 P53256 INFO ******** Test evaluation ********
2022-08-20 22:51:30,238 P53256 INFO Loading data...
2022-08-20 22:51:30,238 P53256 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/test.h5
2022-08-20 22:51:34,908 P53256 INFO Test samples: total/3099515, blocks/1
2022-08-20 22:51:34,908 P53256 INFO Loading test data done.
2022-08-20 22:54:18,085 P53256 INFO [Metrics] gAUC: 0.574437 - AUC: 0.648629 - logloss: 0.193020

```
