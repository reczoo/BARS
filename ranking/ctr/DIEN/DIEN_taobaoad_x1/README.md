## DIEN_taobaoad_x1

A hands-on guide to run the DIEN model on the TaobaoAd_x1 dataset.

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
  fuxictr: 2.0.2

  ```

### Dataset
Please refer to [TaobaoAd_x1](https://github.com/reczoo/Datasets/tree/main/Taobao/TaobaoAd_x1) to get the dataset details.

### Code

We use the [DIEN](https://github.com/reczoo/FuxiCTR/blob/v2.0.2/model_zoo/DIEN) model code from [FuxiCTR-v2.0.2](https://github.com/reczoo/FuxiCTR/tree/v2.0.2) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.2.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.2
    ```

2. Create a data directory and put the downloaded data files in `../data/Taobao/TaobaoAd_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DIEN_taobaoad_x1_tuner_config_01](./DIEN_taobaoad_x1_tuner_config_01). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DIEN
    nohup python run_expid.py --config XXX/benchmarks/DIEN/DIEN_taobaoad_x1_tuner_config_01 --expid DIEN_taobaoad_x1_019_5d9b3874 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.576916 | 0.652937 | 0.192840  |


### Logs
```python
2023-05-12 13:23:35,969 P40695 INFO Params: {
    "attention_activation": "Dice",
    "attention_dropout": "0",
    "attention_hidden_units": "[512, 256]",
    "attention_type": "din_attention",
    "aux_activation": "ReLU",
    "aux_hidden_units": "[100, 50]",
    "aux_loss_alpha": "0",
    "batch_norm": "False",
    "batch_size": "8192",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobaoad_x1_bf8c47ea",
    "debug_mode": "False",
    "dien_neg_seq_field": "[]",
    "dien_sequence_field": "('cate_his', 'brand_his', 'btag_his')",
    "dien_target_field": "('cate_id', 'brand', 'btag')",
    "dnn_activations": "ReLU",
    "dnn_hidden_units": "[512, 256, 128]",
    "early_stop_patience": "1",
    "embedding_dim": "32",
    "embedding_regularizer": "5e-06",
    "enable_sum_pooling": "False",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(userid)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'pid', 'btag'], 'type': 'categorical'}, {'active': True, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'cate_his', 'padding': 'post', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'brand_his', 'padding': 'post', 'share_embedding': 'brand', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 50, 'name': 'btag_his', 'padding': 'post', 'share_embedding': 'btag', 'splitter': '^', 'type': 'sequence'}]",
    "feature_config": "None",
    "feature_specs": "[{'feature_encoder': None, 'name': ['cate_his', 'brand_his', 'btag_his']}]",
    "gpu": "5",
    "group_id": "group_id",
    "gru_type": "AUGRU",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "DIEN",
    "model_id": "DIEN_taobaoad_x1_019_5d9b3874",
    "model_root": "./checkpoints/DIEN_taobaoad_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "20222023",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Taobao/TaobaoAd_x1/test.csv",
    "train_data": "../data/Taobao/TaobaoAd_x1/train.csv",
    "use_attention_softmax": "True",
    "use_features": "None",
    "valid_data": "../data/Taobao/TaobaoAd_x1/test.csv",
    "verbose": "1"
}
2023-05-12 13:23:35,970 P40695 INFO Set up feature processor...
2023-05-12 13:23:35,971 P40695 WARNING Skip rebuilding ../data/Taobao/taobaoad_x1_bf8c47ea/feature_map.json. Please delete it manually if rebuilding is required.
2023-05-12 13:23:35,971 P40695 INFO Load feature_map from json: ../data/Taobao/taobaoad_x1_bf8c47ea/feature_map.json
2023-05-12 13:23:35,971 P40695 INFO Set column index...
2023-05-12 13:23:35,971 P40695 INFO Feature specs: {
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
2023-05-12 13:23:40,498 P40695 INFO Total number of parameters: 42376194.
2023-05-12 13:23:40,499 P40695 INFO Loading data...
2023-05-12 13:23:40,499 P40695 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_bf8c47ea/train.h5
2023-05-12 13:24:07,676 P40695 INFO Train samples: total/21929911, blocks/1
2023-05-12 13:24:07,676 P40695 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_bf8c47ea/valid.h5
2023-05-12 13:24:11,682 P40695 INFO Validation samples: total/3099515, blocks/1
2023-05-12 13:24:11,682 P40695 INFO Loading train and validation data done.
2023-05-12 13:24:11,682 P40695 INFO Start training: 2677 batches/epoch
2023-05-12 13:24:11,682 P40695 INFO ************ Epoch=1 start ************
2023-05-12 13:43:28,981 P40695 INFO Train loss: 0.206096
2023-05-12 13:43:28,981 P40695 INFO Evaluation @epoch 1 - batch 2677: 
2023-05-12 13:45:40,996 P40695 INFO [Metrics] AUC: 0.644660 - gAUC: 0.569318
2023-05-12 13:45:40,998 P40695 INFO Save best model: monitor(max)=1.213978
2023-05-12 13:45:41,319 P40695 INFO ************ Epoch=1 end ************
2023-05-12 14:05:04,771 P40695 INFO Train loss: 0.202818
2023-05-12 14:05:04,772 P40695 INFO Evaluation @epoch 2 - batch 2677: 
2023-05-12 14:07:13,341 P40695 INFO [Metrics] AUC: 0.651441 - gAUC: 0.573494
2023-05-12 14:07:13,342 P40695 INFO Save best model: monitor(max)=1.224936
2023-05-12 14:07:13,745 P40695 INFO ************ Epoch=2 end ************
2023-05-12 14:26:26,322 P40695 INFO Train loss: 0.200345
2023-05-12 14:26:26,322 P40695 INFO Evaluation @epoch 3 - batch 2677: 
2023-05-12 14:28:31,878 P40695 INFO [Metrics] AUC: 0.652077 - gAUC: 0.575660
2023-05-12 14:28:31,879 P40695 INFO Save best model: monitor(max)=1.227736
2023-05-12 14:28:32,313 P40695 INFO ************ Epoch=3 end ************
2023-05-12 14:47:11,265 P40695 INFO Train loss: 0.200253
2023-05-12 14:47:11,266 P40695 INFO Evaluation @epoch 4 - batch 2677: 
2023-05-12 14:49:02,086 P40695 INFO [Metrics] AUC: 0.652161 - gAUC: 0.576367
2023-05-12 14:49:02,088 P40695 INFO Save best model: monitor(max)=1.228528
2023-05-12 14:49:02,510 P40695 INFO ************ Epoch=4 end ************
2023-05-12 15:07:14,926 P40695 INFO Train loss: 0.200552
2023-05-12 15:07:14,927 P40695 INFO Evaluation @epoch 5 - batch 2677: 
2023-05-12 15:09:04,691 P40695 INFO [Metrics] AUC: 0.652937 - gAUC: 0.576916
2023-05-12 15:09:04,692 P40695 INFO Save best model: monitor(max)=1.229853
2023-05-12 15:09:05,188 P40695 INFO ************ Epoch=5 end ************
2023-05-12 15:26:51,051 P40695 INFO Train loss: 0.200870
2023-05-12 15:26:51,052 P40695 INFO Evaluation @epoch 6 - batch 2677: 
2023-05-12 15:28:36,128 P40695 INFO [Metrics] AUC: 0.650501 - gAUC: 0.574232
2023-05-12 15:28:36,129 P40695 INFO Monitor(max)=1.224733 STOP!
2023-05-12 15:28:36,129 P40695 INFO Reduce learning rate on plateau: 0.000100
2023-05-12 15:28:36,129 P40695 INFO ********* Epoch==6 early stop *********
2023-05-12 15:28:36,274 P40695 INFO Training finished.
2023-05-12 15:28:36,275 P40695 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DIEN_taobaoad_x1/taobaoad_x1_bf8c47ea/DIEN_taobaoad_x1_019_5d9b3874.model
2023-05-12 15:28:36,423 P40695 INFO ****** Validation evaluation ******
2023-05-12 15:30:23,176 P40695 INFO [Metrics] gAUC: 0.576916 - AUC: 0.652937 - logloss: 0.192840
2023-05-12 15:30:23,322 P40695 INFO ******** Test evaluation ********
2023-05-12 15:30:23,322 P40695 INFO Loading data...
2023-05-12 15:30:23,322 P40695 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_bf8c47ea/test.h5
2023-05-12 15:30:27,047 P40695 INFO Test samples: total/3099515, blocks/1
2023-05-12 15:30:27,047 P40695 INFO Loading test data done.
2023-05-12 15:32:13,367 P40695 INFO [Metrics] gAUC: 0.576916 - AUC: 0.652937 - logloss: 0.192840

```
