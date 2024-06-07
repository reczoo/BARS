## BST_amazonelectronics_x1

A hands-on guide to run the BST model on the AmazonElectronics_x1 dataset.

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
Please refer to [AmazonElectronics_x1](https://github.com/reczoo/Datasets/tree/main/Amazon/AmazonElectronics_x1) to get the dataset details.

### Code

We use the [BST](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/BST) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Amazon/AmazonElectronics_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [BST_amazonelectronics_x1_tuner_config_09](./BST_amazonelectronics_x1_tuner_config_09). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/BST
    nohup python run_expid.py --config XXX/benchmarks/BST/BST_amazonelectronics_x1_tuner_config_09 --expid BST_amazonelectronics_x1_044_3f4303a7 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.884108 | 0.886424 | 0.430077  |


### Logs
```python
2022-09-06 23:45:38,836 P80232 INFO Params: {
    "attention_dropout": "0.1",
    "batch_norm": "True",
    "batch_size": "1024",
    "bst_sequence_field": "('item_history', 'cate_history')",
    "bst_target_field": "('item_id', 'cate_id')",
    "data_format": "csv",
    "data_root": "../data/Amazon/",
    "dataset_id": "amazonelectronics_x1_b7a43f49",
    "debug_mode": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.005",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'user_id', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'item_history', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'cate_history', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "[{'feature_encoder': None, 'name': 'item_history'}, {'feature_encoder': None, 'name': 'cate_history'}]",
    "gpu": "7",
    "group_id": "user_id",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "layer_norm": "False",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "BST",
    "model_id": "BST_amazonelectronics_x1_044_3f4303a7",
    "model_root": "./checkpoints/BST_amazonelectronics_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_heads": "8",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "20222023",
    "seq_pooling_type": "target",
    "shuffle": "True",
    "stacked_transformer_layers": "1",
    "task": "binary_classification",
    "test_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "train_data": "../data/Amazon/AmazonElectronics_x1/train.csv",
    "use_causal_mask": "False",
    "use_position_emb": "True",
    "use_residual": "True",
    "valid_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "verbose": "1"
}
2022-09-06 23:45:38,837 P80232 INFO Set up feature processor...
2022-09-06 23:45:38,837 P80232 INFO Load feature_map from json: ../data/Amazon/amazonelectronics_x1_b7a43f49/feature_map.json
2022-09-06 23:45:38,837 P80232 INFO Set column index...
2022-09-06 23:45:38,838 P80232 INFO Feature specs: {
    "cate_history": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803, 'max_len': 100}",
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803}",
    "item_history": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003, 'max_len': 100}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003}",
    "user_id": "{'type': 'meta'}"
}
2022-09-06 23:45:45,055 P80232 INFO Total number of parameters: 5301057.
2022-09-06 23:45:45,055 P80232 INFO Loading data...
2022-09-06 23:45:45,055 P80232 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/train.h5
2022-09-06 23:45:49,382 P80232 INFO Train samples: total/2608764, blocks/1
2022-09-06 23:45:49,382 P80232 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/valid.h5
2022-09-06 23:45:49,905 P80232 INFO Validation samples: total/384806, blocks/1
2022-09-06 23:45:49,905 P80232 INFO Loading train and validation data done.
2022-09-06 23:45:49,905 P80232 INFO Start training: 2548 batches/epoch
2022-09-06 23:45:49,905 P80232 INFO ************ Epoch=1 start ************
2022-09-07 00:02:01,274 P80232 INFO [Metrics] AUC: 0.815693 - gAUC: 0.814052
2022-09-07 00:02:01,489 P80232 INFO Save best model: monitor(max): 1.629745
2022-09-07 00:02:01,554 P80232 INFO --- 2548/2548 batches finished ---
2022-09-07 00:02:01,857 P80232 INFO Train loss: 0.637424
2022-09-07 00:02:01,857 P80232 INFO ************ Epoch=1 end ************
2022-09-07 00:18:03,823 P80232 INFO [Metrics] AUC: 0.833753 - gAUC: 0.830725
2022-09-07 00:18:04,005 P80232 INFO Save best model: monitor(max): 1.664478
2022-09-07 00:18:04,057 P80232 INFO --- 2548/2548 batches finished ---
2022-09-07 00:18:04,163 P80232 INFO Train loss: 0.581576
2022-09-07 00:18:04,171 P80232 INFO ************ Epoch=2 end ************
2022-09-07 00:34:02,896 P80232 INFO [Metrics] AUC: 0.852171 - gAUC: 0.850413
2022-09-07 00:34:03,104 P80232 INFO Save best model: monitor(max): 1.702584
2022-09-07 00:34:03,157 P80232 INFO --- 2548/2548 batches finished ---
2022-09-07 00:34:03,366 P80232 INFO Train loss: 0.555085
2022-09-07 00:34:03,367 P80232 INFO ************ Epoch=3 end ************
2022-09-07 00:50:00,765 P80232 INFO [Metrics] AUC: 0.853999 - gAUC: 0.852112
2022-09-07 00:50:00,971 P80232 INFO Save best model: monitor(max): 1.706112
2022-09-07 00:50:01,037 P80232 INFO --- 2548/2548 batches finished ---
2022-09-07 00:50:01,148 P80232 INFO Train loss: 0.546868
2022-09-07 00:50:01,148 P80232 INFO ************ Epoch=4 end ************
2022-09-07 01:05:23,105 P80232 INFO [Metrics] AUC: 0.850815 - gAUC: 0.848428
2022-09-07 01:05:23,377 P80232 INFO Monitor(max) STOP: 1.699243 !
2022-09-07 01:05:23,377 P80232 INFO Reduce learning rate on plateau: 0.000050
2022-09-07 01:05:23,378 P80232 INFO --- 2548/2548 batches finished ---
2022-09-07 01:05:23,535 P80232 INFO Train loss: 0.542949
2022-09-07 01:05:23,536 P80232 INFO ************ Epoch=5 end ************
2022-09-07 01:20:19,918 P80232 INFO [Metrics] AUC: 0.874655 - gAUC: 0.871452
2022-09-07 01:20:20,256 P80232 INFO Save best model: monitor(max): 1.746107
2022-09-07 01:20:20,337 P80232 INFO --- 2548/2548 batches finished ---
2022-09-07 01:20:20,441 P80232 INFO Train loss: 0.467796
2022-09-07 01:20:20,442 P80232 INFO ************ Epoch=6 end ************
2022-09-07 01:35:07,587 P80232 INFO [Metrics] AUC: 0.877825 - gAUC: 0.874794
2022-09-07 01:35:07,758 P80232 INFO Save best model: monitor(max): 1.752619
2022-09-07 01:35:07,811 P80232 INFO --- 2548/2548 batches finished ---
2022-09-07 01:35:07,921 P80232 INFO Train loss: 0.432608
2022-09-07 01:35:07,922 P80232 INFO ************ Epoch=7 end ************
2022-09-07 01:48:40,769 P80232 INFO [Metrics] AUC: 0.880581 - gAUC: 0.877647
2022-09-07 01:48:40,915 P80232 INFO Save best model: monitor(max): 1.758229
2022-09-07 01:48:40,959 P80232 INFO --- 2548/2548 batches finished ---
2022-09-07 01:48:41,070 P80232 INFO Train loss: 0.417337
2022-09-07 01:48:41,071 P80232 INFO ************ Epoch=8 end ************
2022-09-07 02:01:26,172 P80232 INFO [Metrics] AUC: 0.880613 - gAUC: 0.877803
2022-09-07 02:01:26,405 P80232 INFO Save best model: monitor(max): 1.758416
2022-09-07 02:01:26,449 P80232 INFO --- 2548/2548 batches finished ---
2022-09-07 02:01:26,540 P80232 INFO Train loss: 0.408079
2022-09-07 02:01:26,541 P80232 INFO ************ Epoch=9 end ************
2022-09-07 02:12:01,122 P80232 INFO [Metrics] AUC: 0.886424 - gAUC: 0.884108
2022-09-07 02:12:01,281 P80232 INFO Save best model: monitor(max): 1.770532
2022-09-07 02:12:01,327 P80232 INFO --- 2548/2548 batches finished ---
2022-09-07 02:12:01,423 P80232 INFO Train loss: 0.401457
2022-09-07 02:12:01,424 P80232 INFO ************ Epoch=10 end ************
2022-09-07 02:20:20,449 P80232 INFO [Metrics] AUC: 0.883617 - gAUC: 0.881348
2022-09-07 02:20:20,605 P80232 INFO Monitor(max) STOP: 1.764965 !
2022-09-07 02:20:20,605 P80232 INFO Reduce learning rate on plateau: 0.000005
2022-09-07 02:20:20,606 P80232 INFO --- 2548/2548 batches finished ---
2022-09-07 02:20:20,671 P80232 INFO Train loss: 0.396509
2022-09-07 02:20:20,672 P80232 INFO ************ Epoch=11 end ************
2022-09-07 02:24:27,090 P80232 INFO [Metrics] AUC: 0.880776 - gAUC: 0.878697
2022-09-07 02:24:27,243 P80232 INFO Monitor(max) STOP: 1.759474 !
2022-09-07 02:24:27,243 P80232 INFO Reduce learning rate on plateau: 0.000001
2022-09-07 02:24:27,243 P80232 INFO ********* Epoch==12 early stop *********
2022-09-07 02:24:27,243 P80232 INFO --- 2548/2548 batches finished ---
2022-09-07 02:24:27,324 P80232 INFO Train loss: 0.353063
2022-09-07 02:24:27,324 P80232 INFO Training finished.
2022-09-07 02:24:27,324 P80232 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/BST_amazonelectronics_x1/amazonelectronics_x1_b7a43f49/BST_amazonelectronics_x1_044_3f4303a7.model
2022-09-07 02:24:27,529 P80232 INFO ****** Validation evaluation ******
2022-09-07 02:25:14,401 P80232 INFO [Metrics] gAUC: 0.884108 - AUC: 0.886424 - logloss: 0.430077
2022-09-07 02:25:14,623 P80232 INFO ******** Test evaluation ********
2022-09-07 02:25:14,623 P80232 INFO Loading data...
2022-09-07 02:25:14,623 P80232 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/test.h5
2022-09-07 02:25:15,064 P80232 INFO Test samples: total/384806, blocks/1
2022-09-07 02:25:15,064 P80232 INFO Loading test data done.
2022-09-07 02:26:01,326 P80232 INFO [Metrics] gAUC: 0.884108 - AUC: 0.886424 - logloss: 0.430077

```
