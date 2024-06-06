## PPNet_taobao_x1

A hands-on guide to run the PPNet model on the TaobaoAd_x1 dataset.

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

We use the [PPNet](https://github.com/reczoo/FuxiCTR/tree/v2.0.3/model_zoo/PPNet) model code from [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/tree/v2.0.3) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.3.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.3
    ```

2. Create a data directory and put the downloaded data files in `../data/Taobao/TaobaoAd_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [PPNet_taobaoad_x1_tuner_config_01](./PPNet_taobaoad_x1_tuner_config_01). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/PPNet
    nohup python run_expid.py --config YOUR_PATH/PPNet/PPNet_taobaoad_x1_tuner_config_01 --expid PPNet_taobaoad_x1_023_a59da105 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.572597 | 0.647442 | 0.194488  |


### Logs
```python
2023-06-02 02:02:39,172 P84789 INFO Params: {
    "batch_norm": "False",
    "batch_size": "4096",
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
    "gate_emb_dim": "32",
    "gate_hidden_dim": "256",
    "gate_priors": "['userid', 'cms_group_id', 'adgroup_id']",
    "gpu": "7",
    "group_id": "group_id",
    "hidden_activations": "ReLU",
    "hidden_units": "[512, 256, 128]",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "PPNet",
    "model_id": "PPNet_taobaoad_x1_023_a59da105",
    "model_root": "./checkpoints/PPNet_taobaoad_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0",
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
    "use_features": "None",
    "valid_data": "../data/Taobao/TaobaoAd_x1/test.csv",
    "verbose": "1"
}
2023-06-02 02:02:39,190 P84789 INFO Set up feature processor...
2023-06-02 02:02:39,191 P84789 WARNING Skip rebuilding ../data/Taobao/taobaoad_x1_2753db8a/feature_map.json. Please delete it manually if rebuilding is required.
2023-06-02 02:02:39,191 P84789 INFO Load feature_map from json: ../data/Taobao/taobaoad_x1_2753db8a/feature_map.json
2023-06-02 02:02:39,191 P84789 INFO Set column index...
2023-06-02 02:02:39,191 P84789 INFO Feature specs: {
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
2023-06-02 02:02:43,991 P84789 INFO Total number of parameters: 63557633.
2023-06-02 02:02:43,991 P84789 INFO Loading data...
2023-06-02 02:02:43,992 P84789 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/train.h5
2023-06-02 02:03:13,986 P84789 INFO Train samples: total/21929911, blocks/1
2023-06-02 02:03:13,986 P84789 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/valid.h5
2023-06-02 02:03:17,855 P84789 INFO Validation samples: total/3099515, blocks/1
2023-06-02 02:03:17,855 P84789 INFO Loading train and validation data done.
2023-06-02 02:03:17,855 P84789 INFO Start training: 5354 batches/epoch
2023-06-02 02:03:17,855 P84789 INFO ************ Epoch=1 start ************
2023-06-02 02:18:34,306 P84789 INFO Train loss: 0.205051
2023-06-02 02:18:34,307 P84789 INFO Evaluation @epoch 1 - batch 5354: 
2023-06-02 02:21:49,287 P84789 INFO [Metrics] AUC: 0.641968 - gAUC: 0.564675
2023-06-02 02:21:49,289 P84789 INFO Save best model: monitor(max)=1.206643
2023-06-02 02:21:49,762 P84789 INFO ************ Epoch=1 end ************
2023-06-02 02:36:54,632 P84789 INFO Train loss: 0.200883
2023-06-02 02:36:54,632 P84789 INFO Evaluation @epoch 2 - batch 5354: 
2023-06-02 02:40:14,103 P84789 INFO [Metrics] AUC: 0.644788 - gAUC: 0.568505
2023-06-02 02:40:14,105 P84789 INFO Save best model: monitor(max)=1.213293
2023-06-02 02:40:14,809 P84789 INFO ************ Epoch=2 end ************
2023-06-02 02:55:21,620 P84789 INFO Train loss: 0.199641
2023-06-02 02:55:21,621 P84789 INFO Evaluation @epoch 3 - batch 5354: 
2023-06-02 02:58:39,958 P84789 INFO [Metrics] AUC: 0.646491 - gAUC: 0.568941
2023-06-02 02:58:39,960 P84789 INFO Save best model: monitor(max)=1.215432
2023-06-02 02:58:40,590 P84789 INFO ************ Epoch=3 end ************
2023-06-02 03:13:42,846 P84789 INFO Train loss: 0.199421
2023-06-02 03:13:42,855 P84789 INFO Evaluation @epoch 4 - batch 5354: 
2023-06-02 03:16:58,483 P84789 INFO [Metrics] AUC: 0.646835 - gAUC: 0.570554
2023-06-02 03:16:58,485 P84789 INFO Save best model: monitor(max)=1.217389
2023-06-02 03:16:59,225 P84789 INFO ************ Epoch=4 end ************
2023-06-02 03:31:51,794 P84789 INFO Train loss: 0.199368
2023-06-02 03:31:51,795 P84789 INFO Evaluation @epoch 5 - batch 5354: 
2023-06-02 03:35:00,080 P84789 INFO [Metrics] AUC: 0.646718 - gAUC: 0.569567
2023-06-02 03:35:00,082 P84789 INFO Monitor(max)=1.216284 STOP!
2023-06-02 03:35:00,082 P84789 INFO Reduce learning rate on plateau: 0.000100
2023-06-02 03:35:00,211 P84789 INFO ************ Epoch=5 end ************
2023-06-02 03:49:05,090 P84789 INFO Train loss: 0.189274
2023-06-02 03:49:05,091 P84789 INFO Evaluation @epoch 6 - batch 5354: 
2023-06-02 03:52:15,449 P84789 INFO [Metrics] AUC: 0.647442 - gAUC: 0.572597
2023-06-02 03:52:15,452 P84789 INFO Save best model: monitor(max)=1.220039
2023-06-02 03:52:16,000 P84789 INFO ************ Epoch=6 end ************
2023-06-02 04:04:57,999 P84789 INFO Train loss: 0.186120
2023-06-02 04:04:58,000 P84789 INFO Evaluation @epoch 7 - batch 5354: 
2023-06-02 04:07:53,403 P84789 INFO [Metrics] AUC: 0.646037 - gAUC: 0.571588
2023-06-02 04:07:53,404 P84789 INFO Monitor(max)=1.217625 STOP!
2023-06-02 04:07:53,404 P84789 INFO Reduce learning rate on plateau: 0.000010
2023-06-02 04:07:53,520 P84789 INFO ************ Epoch=7 end ************
2023-06-02 04:20:43,348 P84789 INFO Train loss: 0.176549
2023-06-02 04:20:43,348 P84789 INFO Evaluation @epoch 8 - batch 5354: 
2023-06-02 04:23:42,541 P84789 INFO [Metrics] AUC: 0.640906 - gAUC: 0.566632
2023-06-02 04:23:42,542 P84789 INFO Monitor(max)=1.207538 STOP!
2023-06-02 04:23:42,543 P84789 INFO Reduce learning rate on plateau: 0.000001
2023-06-02 04:23:42,543 P84789 INFO ********* Epoch==8 early stop *********
2023-06-02 04:23:42,695 P84789 INFO Training finished.
2023-06-02 04:23:42,696 P84789 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/PPNet_taobaoad_x1/taobaoad_x1_2753db8a/PPNet_taobaoad_x1_023_a59da105.model
2023-06-02 04:23:42,929 P84789 INFO ****** Validation evaluation ******
2023-06-02 04:26:49,560 P84789 INFO [Metrics] gAUC: 0.572597 - AUC: 0.647442 - logloss: 0.194488
2023-06-02 04:26:49,718 P84789 INFO ******** Test evaluation ********
2023-06-02 04:26:49,718 P84789 INFO Loading data...
2023-06-02 04:26:49,718 P84789 INFO Loading data from h5: ../data/Taobao/taobaoad_x1_2753db8a/test.h5
2023-06-02 04:26:53,701 P84789 INFO Test samples: total/3099515, blocks/1
2023-06-02 04:26:53,701 P84789 INFO Loading test data done.
2023-06-02 04:29:56,282 P84789 INFO [Metrics] gAUC: 0.572597 - AUC: 0.647442 - logloss: 0.194488

```
