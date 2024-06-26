## APG_DeepFM_amazonelectronics_x1

A hands-on guide to run the APG model on the AmazonElectronics_x1 dataset.

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
Please refer to [AmazonElectronics_x1](https://github.com/reczoo/Datasets/tree/main/Amazon/AmazonElectronics_x1) to get the dataset details.

### Code

We use the [APG](https://github.com/reczoo/FuxiCTR/tree/v2.0.3/model_zoo/APG) model code from [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/tree/v2.0.3) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.3.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.3
    ```

2. Create a data directory and put the downloaded data files in `../data/Amazon/AmazonElectronics_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [APG_DeepFM_amazonelectronics_x1_tuner_config_08](./APG_DeepFM_amazonelectronics_x1_tuner_config_08). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/APG
    nohup python run_expid.py --config YOUR_PATH/APG/APG_DeepFM_amazonelectronics_x1_tuner_config_08 --expid APG_DeepFM_amazonelectronics_x1_001_daf845b0 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.878817 | 0.881180 | 0.439346  |


### Logs
```python
2023-05-31 07:46:22,585 P50072 INFO Params: {
    "batch_norm": "True",
    "batch_size": "1024",
    "condition_features": "None",
    "condition_mode": "self-wise",
    "data_format": "csv",
    "data_root": "../data/Amazon/",
    "dataset_id": "amazonelectronics_x1_b7a43f49",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.005",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'user_id', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'item_history', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'cate_history', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}]",
    "feature_config": "None",
    "feature_specs": "None",
    "generate_bias": "True",
    "gpu": "0",
    "group_id": "user_id",
    "hidden_activations": "relu",
    "hidden_units": "[1024, 512, 256]",
    "hypernet_config": "{'dropout_rates': 0.1, 'hidden_activations': 'relu', 'hidden_units': []}",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "APG_DeepFM",
    "model_id": "APG_DeepFM_amazonelectronics_x1_001_daf845b0",
    "model_root": "./checkpoints/APG_DeepFM_amazonelectronics_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "new_condition_emb": "False",
    "num_workers": "3",
    "optimizer": "adam",
    "overparam_p": "None",
    "pickle_feature_encoder": "True",
    "rank_k": "[32, 16, 8]",
    "save_best_only": "True",
    "seed": "20222023",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "train_data": "../data/Amazon/AmazonElectronics_x1/train.csv",
    "use_features": "None",
    "valid_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "verbose": "1"
}
2023-05-31 07:46:22,585 P50072 INFO Set up feature processor...
2023-05-31 07:46:22,585 P50072 WARNING Skip rebuilding ../data/Amazon/amazonelectronics_x1_b7a43f49/feature_map.json. Please delete it manually if rebuilding is required.
2023-05-31 07:46:22,585 P50072 INFO Load feature_map from json: ../data/Amazon/amazonelectronics_x1_b7a43f49/feature_map.json
2023-05-31 07:46:22,586 P50072 INFO Set column index...
2023-05-31 07:46:22,586 P50072 INFO Feature specs: {
    "cate_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803, 'max_len': 100}",
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803}",
    "item_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003, 'max_len': 100}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003}",
    "user_id": "{'type': 'meta'}"
}
2023-05-31 07:46:27,353 P50072 INFO Total number of parameters: 5764414.
2023-05-31 07:46:27,354 P50072 INFO Loading data...
2023-05-31 07:46:27,354 P50072 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/train.h5
2023-05-31 07:46:30,303 P50072 INFO Train samples: total/2608764, blocks/1
2023-05-31 07:46:30,303 P50072 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/valid.h5
2023-05-31 07:46:30,714 P50072 INFO Validation samples: total/384806, blocks/1
2023-05-31 07:46:30,714 P50072 INFO Loading train and validation data done.
2023-05-31 07:46:30,714 P50072 INFO Start training: 2548 batches/epoch
2023-05-31 07:46:30,715 P50072 INFO ************ Epoch=1 start ************
2023-05-31 07:51:16,222 P50072 INFO Train loss: 0.638213
2023-05-31 07:51:16,222 P50072 INFO Evaluation @epoch 1 - batch 2548: 
2023-05-31 07:52:48,320 P50072 INFO [Metrics] AUC: 0.831274 - gAUC: 0.829119
2023-05-31 07:52:48,321 P50072 INFO Save best model: monitor(max)=1.660393
2023-05-31 07:52:48,458 P50072 INFO ************ Epoch=1 end ************
2023-05-31 07:57:29,405 P50072 INFO Train loss: 0.593494
2023-05-31 07:57:29,405 P50072 INFO Evaluation @epoch 2 - batch 2548: 
2023-05-31 07:59:03,390 P50072 INFO [Metrics] AUC: 0.845894 - gAUC: 0.844046
2023-05-31 07:59:03,391 P50072 INFO Save best model: monitor(max)=1.689940
2023-05-31 07:59:03,585 P50072 INFO ************ Epoch=2 end ************
2023-05-31 08:03:49,117 P50072 INFO Train loss: 0.578527
2023-05-31 08:03:49,118 P50072 INFO Evaluation @epoch 3 - batch 2548: 
2023-05-31 08:05:26,905 P50072 INFO [Metrics] AUC: 0.850897 - gAUC: 0.848376
2023-05-31 08:05:26,906 P50072 INFO Save best model: monitor(max)=1.699273
2023-05-31 08:05:27,068 P50072 INFO ************ Epoch=3 end ************
2023-05-31 08:10:08,489 P50072 INFO Train loss: 0.572567
2023-05-31 08:10:08,490 P50072 INFO Evaluation @epoch 4 - batch 2548: 
2023-05-31 08:11:43,438 P50072 INFO [Metrics] AUC: 0.851587 - gAUC: 0.848802
2023-05-31 08:11:43,439 P50072 INFO Save best model: monitor(max)=1.700389
2023-05-31 08:11:43,613 P50072 INFO ************ Epoch=4 end ************
2023-05-31 08:16:26,946 P50072 INFO Train loss: 0.569565
2023-05-31 08:16:26,947 P50072 INFO Evaluation @epoch 5 - batch 2548: 
2023-05-31 08:18:00,913 P50072 INFO [Metrics] AUC: 0.854329 - gAUC: 0.851759
2023-05-31 08:18:00,915 P50072 INFO Save best model: monitor(max)=1.706088
2023-05-31 08:18:01,082 P50072 INFO ************ Epoch=5 end ************
2023-05-31 08:22:40,250 P50072 INFO Train loss: 0.567909
2023-05-31 08:22:40,250 P50072 INFO Evaluation @epoch 6 - batch 2548: 
2023-05-31 08:24:14,192 P50072 INFO [Metrics] AUC: 0.856020 - gAUC: 0.854103
2023-05-31 08:24:14,194 P50072 INFO Save best model: monitor(max)=1.710123
2023-05-31 08:24:14,366 P50072 INFO ************ Epoch=6 end ************
2023-05-31 08:28:58,830 P50072 INFO Train loss: 0.566446
2023-05-31 08:28:58,831 P50072 INFO Evaluation @epoch 7 - batch 2548: 
2023-05-31 08:30:34,543 P50072 INFO [Metrics] AUC: 0.856143 - gAUC: 0.854082
2023-05-31 08:30:34,544 P50072 INFO Save best model: monitor(max)=1.710226
2023-05-31 08:30:34,693 P50072 INFO ************ Epoch=7 end ************
2023-05-31 08:35:17,623 P50072 INFO Train loss: 0.566004
2023-05-31 08:35:17,623 P50072 INFO Evaluation @epoch 8 - batch 2548: 
2023-05-31 08:36:50,699 P50072 INFO [Metrics] AUC: 0.855911 - gAUC: 0.853999
2023-05-31 08:36:50,700 P50072 INFO Monitor(max)=1.709910 STOP!
2023-05-31 08:36:50,700 P50072 INFO Reduce learning rate on plateau: 0.000050
2023-05-31 08:36:50,793 P50072 INFO ************ Epoch=8 end ************
2023-05-31 08:41:31,260 P50072 INFO Train loss: 0.469837
2023-05-31 08:41:31,261 P50072 INFO Evaluation @epoch 9 - batch 2548: 
2023-05-31 08:43:04,068 P50072 INFO [Metrics] AUC: 0.876257 - gAUC: 0.873936
2023-05-31 08:43:04,070 P50072 INFO Save best model: monitor(max)=1.750194
2023-05-31 08:43:04,236 P50072 INFO ************ Epoch=9 end ************
2023-05-31 08:47:44,677 P50072 INFO Train loss: 0.425284
2023-05-31 08:47:44,677 P50072 INFO Evaluation @epoch 10 - batch 2548: 
2023-05-31 08:49:16,247 P50072 INFO [Metrics] AUC: 0.879939 - gAUC: 0.877549
2023-05-31 08:49:16,248 P50072 INFO Save best model: monitor(max)=1.757488
2023-05-31 08:49:16,415 P50072 INFO ************ Epoch=10 end ************
2023-05-31 08:53:48,424 P50072 INFO Train loss: 0.405618
2023-05-31 08:53:48,424 P50072 INFO Evaluation @epoch 11 - batch 2548: 
2023-05-31 08:55:10,413 P50072 INFO [Metrics] AUC: 0.881180 - gAUC: 0.878817
2023-05-31 08:55:10,414 P50072 INFO Save best model: monitor(max)=1.759997
2023-05-31 08:55:10,570 P50072 INFO ************ Epoch=11 end ************
2023-05-31 08:58:40,712 P50072 INFO Train loss: 0.393057
2023-05-31 08:58:40,712 P50072 INFO Evaluation @epoch 12 - batch 2548: 
2023-05-31 08:59:40,360 P50072 INFO [Metrics] AUC: 0.880131 - gAUC: 0.878022
2023-05-31 08:59:40,361 P50072 INFO Monitor(max)=1.758153 STOP!
2023-05-31 08:59:40,361 P50072 INFO Reduce learning rate on plateau: 0.000005
2023-05-31 08:59:40,427 P50072 INFO ************ Epoch=12 end ************
2023-05-31 09:01:35,997 P50072 INFO Train loss: 0.334959
2023-05-31 09:01:35,998 P50072 INFO Evaluation @epoch 13 - batch 2548: 
2023-05-31 09:02:26,838 P50072 INFO [Metrics] AUC: 0.874610 - gAUC: 0.873510
2023-05-31 09:02:26,839 P50072 INFO Monitor(max)=1.748120 STOP!
2023-05-31 09:02:26,839 P50072 INFO Reduce learning rate on plateau: 0.000001
2023-05-31 09:02:26,839 P50072 INFO ********* Epoch==13 early stop *********
2023-05-31 09:02:26,903 P50072 INFO Training finished.
2023-05-31 09:02:26,904 P50072 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/APG_DeepFM_amazonelectronics_x1/amazonelectronics_x1_b7a43f49/APG_DeepFM_amazonelectronics_x1_001_daf845b0.model
2023-05-31 09:02:26,933 P50072 INFO ****** Validation evaluation ******
2023-05-31 09:03:13,140 P50072 INFO [Metrics] gAUC: 0.878817 - AUC: 0.881180 - logloss: 0.439346
2023-05-31 09:03:13,225 P50072 INFO ******** Test evaluation ********
2023-05-31 09:03:13,225 P50072 INFO Loading data...
2023-05-31 09:03:13,225 P50072 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/test.h5
2023-05-31 09:03:13,671 P50072 INFO Test samples: total/384806, blocks/1
2023-05-31 09:03:13,672 P50072 INFO Loading test data done.
2023-05-31 09:03:56,850 P50072 INFO [Metrics] gAUC: 0.878817 - AUC: 0.881180 - logloss: 0.439346

```
