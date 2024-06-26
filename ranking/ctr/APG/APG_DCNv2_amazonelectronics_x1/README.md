## APG_DCNv2_amazonelectronics_x1

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

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [APG_DCNv2_amazonelectronics_x1_tuner_config_03](./APG_DCNv2_amazonelectronics_x1_tuner_config_03). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/APG
    nohup python run_expid.py --config YOUR_PATH/APG/APG_DCNv2_amazonelectronics_x1_tuner_config_03 --expid APG_DCNv2_amazonelectronics_x1_015_a0cca3e4 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.879420 | 0.882007 | 0.437314  |


### Logs
```python
2023-05-31 09:46:57,297 P12611 INFO Params: {
    "batch_norm": "True",
    "batch_size": "1024",
    "condition_features": "None",
    "condition_mode": "self-wise",
    "data_format": "csv",
    "data_root": "../data/Amazon/",
    "dataset_id": "amazonelectronics_x1_b7a43f49",
    "debug_mode": "False",
    "dnn_activations": "relu",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.005",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'user_id', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'item_history', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'cate_history', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}]",
    "feature_config": "None",
    "feature_specs": "None",
    "generate_bias": "True",
    "gpu": "6",
    "group_id": "user_id",
    "hypernet_config": "{'dropout_rates': 0.1, 'hidden_activations': 'relu', 'hidden_units': []}",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "low_rank": "32",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "APG_DCNv2",
    "model_id": "APG_DCNv2_amazonelectronics_x1_015_a0cca3e4",
    "model_root": "./checkpoints/APG_DCNv2_amazonelectronics_x1/",
    "model_structure": "parallel",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "new_condition_emb": "False",
    "num_cross_layers": "2",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "overparam_p": "[32, 16, 8]",
    "parallel_dnn_hidden_units": "[1024, 512, 256]",
    "pickle_feature_encoder": "True",
    "rank_k": "[32, 16, 8]",
    "save_best_only": "True",
    "seed": "2022",
    "shuffle": "True",
    "stacked_dnn_hidden_units": "[500, 500, 500]",
    "task": "binary_classification",
    "test_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "train_data": "../data/Amazon/AmazonElectronics_x1/train.csv",
    "use_features": "None",
    "use_low_rank_mixture": "False",
    "valid_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "verbose": "1"
}
2023-05-31 09:46:57,297 P12611 INFO Set up feature processor...
2023-05-31 09:46:57,297 P12611 WARNING Skip rebuilding ../data/Amazon/amazonelectronics_x1_b7a43f49/feature_map.json. Please delete it manually if rebuilding is required.
2023-05-31 09:46:57,298 P12611 INFO Load feature_map from json: ../data/Amazon/amazonelectronics_x1_b7a43f49/feature_map.json
2023-05-31 09:46:57,298 P12611 INFO Set column index...
2023-05-31 09:46:57,298 P12611 INFO Feature specs: {
    "cate_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803, 'max_len': 100}",
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803}",
    "item_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003, 'max_len': 100}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003}",
    "user_id": "{'type': 'meta'}"
}
2023-05-31 09:47:03,303 P12611 INFO Total number of parameters: 5771329.
2023-05-31 09:47:03,304 P12611 INFO Loading data...
2023-05-31 09:47:03,304 P12611 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/train.h5
2023-05-31 09:47:06,616 P12611 INFO Train samples: total/2608764, blocks/1
2023-05-31 09:47:06,617 P12611 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/valid.h5
2023-05-31 09:47:07,029 P12611 INFO Validation samples: total/384806, blocks/1
2023-05-31 09:47:07,029 P12611 INFO Loading train and validation data done.
2023-05-31 09:47:07,029 P12611 INFO Start training: 2548 batches/epoch
2023-05-31 09:47:07,029 P12611 INFO ************ Epoch=1 start ************
2023-05-31 09:52:05,224 P12611 INFO Train loss: 0.637748
2023-05-31 09:52:05,224 P12611 INFO Evaluation @epoch 1 - batch 2548: 
2023-05-31 09:53:38,497 P12611 INFO [Metrics] AUC: 0.834987 - gAUC: 0.833734
2023-05-31 09:53:38,498 P12611 INFO Save best model: monitor(max)=1.668721
2023-05-31 09:53:38,600 P12611 INFO ************ Epoch=1 end ************
2023-05-31 09:58:32,438 P12611 INFO Train loss: 0.595188
2023-05-31 09:58:32,439 P12611 INFO Evaluation @epoch 2 - batch 2548: 
2023-05-31 10:00:06,624 P12611 INFO [Metrics] AUC: 0.847464 - gAUC: 0.844956
2023-05-31 10:00:06,627 P12611 INFO Save best model: monitor(max)=1.692420
2023-05-31 10:00:06,797 P12611 INFO ************ Epoch=2 end ************
2023-05-31 10:05:02,904 P12611 INFO Train loss: 0.578905
2023-05-31 10:05:02,905 P12611 INFO Evaluation @epoch 3 - batch 2548: 
2023-05-31 10:06:36,308 P12611 INFO [Metrics] AUC: 0.852364 - gAUC: 0.850231
2023-05-31 10:06:36,309 P12611 INFO Save best model: monitor(max)=1.702595
2023-05-31 10:06:36,509 P12611 INFO ************ Epoch=3 end ************
2023-05-31 10:11:31,521 P12611 INFO Train loss: 0.573118
2023-05-31 10:11:31,525 P12611 INFO Evaluation @epoch 4 - batch 2548: 
2023-05-31 10:13:05,624 P12611 INFO [Metrics] AUC: 0.854765 - gAUC: 0.851972
2023-05-31 10:13:05,625 P12611 INFO Save best model: monitor(max)=1.706737
2023-05-31 10:13:05,741 P12611 INFO ************ Epoch=4 end ************
2023-05-31 10:18:04,625 P12611 INFO Train loss: 0.569993
2023-05-31 10:18:04,625 P12611 INFO Evaluation @epoch 5 - batch 2548: 
2023-05-31 10:19:39,244 P12611 INFO [Metrics] AUC: 0.856304 - gAUC: 0.854181
2023-05-31 10:19:39,245 P12611 INFO Save best model: monitor(max)=1.710485
2023-05-31 10:19:39,372 P12611 INFO ************ Epoch=5 end ************
2023-05-31 10:24:36,877 P12611 INFO Train loss: 0.568825
2023-05-31 10:24:36,878 P12611 INFO Evaluation @epoch 6 - batch 2548: 
2023-05-31 10:26:09,168 P12611 INFO [Metrics] AUC: 0.855471 - gAUC: 0.853173
2023-05-31 10:26:09,248 P12611 INFO Monitor(max)=1.708644 STOP!
2023-05-31 10:26:09,248 P12611 INFO Reduce learning rate on plateau: 0.000050
2023-05-31 10:26:09,353 P12611 INFO ************ Epoch=6 end ************
2023-05-31 10:30:59,981 P12611 INFO Train loss: 0.472181
2023-05-31 10:30:59,981 P12611 INFO Evaluation @epoch 7 - batch 2548: 
2023-05-31 10:32:35,254 P12611 INFO [Metrics] AUC: 0.876567 - gAUC: 0.873755
2023-05-31 10:32:35,256 P12611 INFO Save best model: monitor(max)=1.750322
2023-05-31 10:32:35,391 P12611 INFO ************ Epoch=7 end ************
2023-05-31 10:37:11,148 P12611 INFO Train loss: 0.427074
2023-05-31 10:37:11,149 P12611 INFO Evaluation @epoch 8 - batch 2548: 
2023-05-31 10:38:38,473 P12611 INFO [Metrics] AUC: 0.880211 - gAUC: 0.877897
2023-05-31 10:38:38,475 P12611 INFO Save best model: monitor(max)=1.758108
2023-05-31 10:38:38,601 P12611 INFO ************ Epoch=8 end ************
2023-05-31 10:43:10,243 P12611 INFO Train loss: 0.407486
2023-05-31 10:43:10,243 P12611 INFO Evaluation @epoch 9 - batch 2548: 
2023-05-31 10:44:33,860 P12611 INFO [Metrics] AUC: 0.882007 - gAUC: 0.879420
2023-05-31 10:44:33,860 P12611 INFO Save best model: monitor(max)=1.761427
2023-05-31 10:44:33,990 P12611 INFO ************ Epoch=9 end ************
2023-05-31 10:48:39,854 P12611 INFO Train loss: 0.394435
2023-05-31 10:48:39,855 P12611 INFO Evaluation @epoch 10 - batch 2548: 
2023-05-31 10:49:55,253 P12611 INFO [Metrics] AUC: 0.880280 - gAUC: 0.877923
2023-05-31 10:49:55,254 P12611 INFO Monitor(max)=1.758203 STOP!
2023-05-31 10:49:55,254 P12611 INFO Reduce learning rate on plateau: 0.000005
2023-05-31 10:49:55,331 P12611 INFO ************ Epoch=10 end ************
2023-05-31 10:53:22,343 P12611 INFO Train loss: 0.332896
2023-05-31 10:53:22,344 P12611 INFO Evaluation @epoch 11 - batch 2548: 
2023-05-31 10:54:28,346 P12611 INFO [Metrics] AUC: 0.874604 - gAUC: 0.872393
2023-05-31 10:54:28,347 P12611 INFO Monitor(max)=1.746997 STOP!
2023-05-31 10:54:28,347 P12611 INFO Reduce learning rate on plateau: 0.000001
2023-05-31 10:54:28,347 P12611 INFO ********* Epoch==11 early stop *********
2023-05-31 10:54:28,415 P12611 INFO Training finished.
2023-05-31 10:54:28,415 P12611 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/APG_DCNv2_amazonelectronics_x1/amazonelectronics_x1_b7a43f49/APG_DCNv2_amazonelectronics_x1_015_a0cca3e4.model
2023-05-31 10:54:28,452 P12611 INFO ****** Validation evaluation ******
2023-05-31 10:55:32,159 P12611 INFO [Metrics] gAUC: 0.879420 - AUC: 0.882007 - logloss: 0.437314
2023-05-31 10:55:32,252 P12611 INFO ******** Test evaluation ********
2023-05-31 10:55:32,253 P12611 INFO Loading data...
2023-05-31 10:55:32,253 P12611 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/test.h5
2023-05-31 10:55:32,694 P12611 INFO Test samples: total/384806, blocks/1
2023-05-31 10:55:32,694 P12611 INFO Loading test data done.
2023-05-31 10:56:41,092 P12611 INFO [Metrics] gAUC: 0.879420 - AUC: 0.882007 - logloss: 0.437314

```
