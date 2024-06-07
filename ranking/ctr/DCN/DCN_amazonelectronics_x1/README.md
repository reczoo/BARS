## DCN_amazonelectronics_x1

A hands-on guide to run the DCN model on the AmazonElectronics_x1 dataset.

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

We use the [DCN](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/DCN) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Amazon/AmazonElectronics_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DCN_amazonelectronics_x1_tuner_config_04](./DCN_amazonelectronics_x1_tuner_config_04). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DCN
    nohup python run_expid.py --config XXX/benchmarks/DCN/DCN_amazonelectronics_x1_tuner_config_04 --expid DCN_amazonelectronics_x1_029_366c6259 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.877829 | 0.880079 | 0.442113  |


### Logs
```python
2022-08-11 09:04:16,566 P12493 INFO Params: {
    "batch_norm": "False",
    "batch_size": "1024",
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
    "feature_specs": "None",
    "gpu": "3",
    "group_id": "user_id",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCN",
    "model_id": "DCN_amazonelectronics_x1_029_366c6259",
    "model_root": "./checkpoints/DCN_amazonelectronics_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_cross_layers": "6",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "20222023",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "train_data": "../data/Amazon/AmazonElectronics_x1/train.csv",
    "valid_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "verbose": "1"
}
2022-08-11 09:04:16,567 P12493 INFO Set up feature processor...
2022-08-11 09:04:16,567 P12493 INFO Load feature_map from json: ../data/Amazon/amazonelectronics_x1_b7a43f49/feature_map.json
2022-08-11 09:04:16,567 P12493 INFO Set column index...
2022-08-11 09:04:16,568 P12493 INFO Feature specs: {
    "cate_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803, 'max_len': 100}",
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803}",
    "item_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003, 'max_len': 100}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003}",
    "user_id": "{'type': 'meta'}"
}
2022-08-11 09:04:21,275 P12493 INFO Total number of parameters: 5006465.
2022-08-11 09:04:21,276 P12493 INFO Loading data...
2022-08-11 09:04:21,276 P12493 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/train.h5
2022-08-11 09:04:25,542 P12493 INFO Train samples: total/2608764, blocks/1
2022-08-11 09:04:25,542 P12493 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/valid.h5
2022-08-11 09:04:26,112 P12493 INFO Validation samples: total/384806, blocks/1
2022-08-11 09:04:26,112 P12493 INFO Loading train and validation data done.
2022-08-11 09:04:26,112 P12493 INFO Start training: 2548 batches/epoch
2022-08-11 09:04:26,112 P12493 INFO ************ Epoch=1 start ************
2022-08-11 09:15:04,600 P12493 INFO [Metrics] AUC: 0.826419 - gAUC: 0.823771
2022-08-11 09:15:04,779 P12493 INFO Save best model: monitor(max): 1.650190
2022-08-11 09:15:04,853 P12493 INFO --- 2548/2548 batches finished ---
2022-08-11 09:15:04,957 P12493 INFO Train loss: 0.610813
2022-08-11 09:15:04,958 P12493 INFO ************ Epoch=1 end ************
2022-08-11 09:25:51,930 P12493 INFO [Metrics] AUC: 0.839604 - gAUC: 0.837529
2022-08-11 09:25:52,124 P12493 INFO Save best model: monitor(max): 1.677133
2022-08-11 09:25:52,206 P12493 INFO --- 2548/2548 batches finished ---
2022-08-11 09:25:52,346 P12493 INFO Train loss: 0.584337
2022-08-11 09:25:52,347 P12493 INFO ************ Epoch=2 end ************
2022-08-11 09:36:38,733 P12493 INFO [Metrics] AUC: 0.845385 - gAUC: 0.842830
2022-08-11 09:36:38,948 P12493 INFO Save best model: monitor(max): 1.688215
2022-08-11 09:36:39,003 P12493 INFO --- 2548/2548 batches finished ---
2022-08-11 09:36:39,145 P12493 INFO Train loss: 0.579937
2022-08-11 09:36:39,146 P12493 INFO ************ Epoch=3 end ************
2022-08-11 09:47:17,209 P12493 INFO [Metrics] AUC: 0.850628 - gAUC: 0.847918
2022-08-11 09:47:17,373 P12493 INFO Save best model: monitor(max): 1.698546
2022-08-11 09:47:17,437 P12493 INFO --- 2548/2548 batches finished ---
2022-08-11 09:47:17,549 P12493 INFO Train loss: 0.573019
2022-08-11 09:47:17,549 P12493 INFO ************ Epoch=4 end ************
2022-08-11 09:58:06,028 P12493 INFO [Metrics] AUC: 0.851834 - gAUC: 0.848557
2022-08-11 09:58:06,173 P12493 INFO Save best model: monitor(max): 1.700392
2022-08-11 09:58:06,217 P12493 INFO --- 2548/2548 batches finished ---
2022-08-11 09:58:06,320 P12493 INFO Train loss: 0.569427
2022-08-11 09:58:06,320 P12493 INFO ************ Epoch=5 end ************
2022-08-11 10:08:50,015 P12493 INFO [Metrics] AUC: 0.852228 - gAUC: 0.849259
2022-08-11 10:08:50,164 P12493 INFO Save best model: monitor(max): 1.701487
2022-08-11 10:08:50,216 P12493 INFO --- 2548/2548 batches finished ---
2022-08-11 10:08:50,308 P12493 INFO Train loss: 0.567037
2022-08-11 10:08:50,308 P12493 INFO ************ Epoch=6 end ************
2022-08-11 10:19:32,878 P12493 INFO [Metrics] AUC: 0.855475 - gAUC: 0.852700
2022-08-11 10:19:33,013 P12493 INFO Save best model: monitor(max): 1.708175
2022-08-11 10:19:33,116 P12493 INFO --- 2548/2548 batches finished ---
2022-08-11 10:19:33,219 P12493 INFO Train loss: 0.565674
2022-08-11 10:19:33,219 P12493 INFO ************ Epoch=7 end ************
2022-08-11 10:30:12,223 P12493 INFO [Metrics] AUC: 0.855073 - gAUC: 0.853365
2022-08-11 10:30:12,405 P12493 INFO Save best model: monitor(max): 1.708438
2022-08-11 10:30:12,458 P12493 INFO --- 2548/2548 batches finished ---
2022-08-11 10:30:12,555 P12493 INFO Train loss: 0.566867
2022-08-11 10:30:12,555 P12493 INFO ************ Epoch=8 end ************
2022-08-11 10:40:17,285 P12493 INFO [Metrics] AUC: 0.856521 - gAUC: 0.854446
2022-08-11 10:40:17,469 P12493 INFO Save best model: monitor(max): 1.710967
2022-08-11 10:40:17,516 P12493 INFO --- 2548/2548 batches finished ---
2022-08-11 10:40:17,632 P12493 INFO Train loss: 0.567879
2022-08-11 10:40:17,632 P12493 INFO ************ Epoch=9 end ************
2022-08-11 10:49:27,573 P12493 INFO [Metrics] AUC: 0.857900 - gAUC: 0.855590
2022-08-11 10:49:27,721 P12493 INFO Save best model: monitor(max): 1.713490
2022-08-11 10:49:27,759 P12493 INFO --- 2548/2548 batches finished ---
2022-08-11 10:49:27,874 P12493 INFO Train loss: 0.568293
2022-08-11 10:49:27,874 P12493 INFO ************ Epoch=10 end ************
2022-08-11 10:57:26,886 P12493 INFO [Metrics] AUC: 0.855907 - gAUC: 0.852996
2022-08-11 10:57:27,038 P12493 INFO Monitor(max) STOP: 1.708903 !
2022-08-11 10:57:27,039 P12493 INFO Reduce learning rate on plateau: 0.000050
2022-08-11 10:57:27,039 P12493 INFO --- 2548/2548 batches finished ---
2022-08-11 10:57:27,155 P12493 INFO Train loss: 0.570178
2022-08-11 10:57:27,156 P12493 INFO ************ Epoch=11 end ************
2022-08-11 11:03:57,458 P12493 INFO [Metrics] AUC: 0.874097 - gAUC: 0.871145
2022-08-11 11:03:57,610 P12493 INFO Save best model: monitor(max): 1.745242
2022-08-11 11:03:57,673 P12493 INFO --- 2548/2548 batches finished ---
2022-08-11 11:03:57,767 P12493 INFO Train loss: 0.473938
2022-08-11 11:03:57,767 P12493 INFO ************ Epoch=12 end ************
2022-08-11 11:09:33,769 P12493 INFO [Metrics] AUC: 0.878136 - gAUC: 0.875267
2022-08-11 11:09:33,908 P12493 INFO Save best model: monitor(max): 1.753403
2022-08-11 11:09:33,948 P12493 INFO --- 2548/2548 batches finished ---
2022-08-11 11:09:34,041 P12493 INFO Train loss: 0.430298
2022-08-11 11:09:34,041 P12493 INFO ************ Epoch=13 end ************
2022-08-11 11:13:24,434 P12493 INFO [Metrics] AUC: 0.879009 - gAUC: 0.876956
2022-08-11 11:13:24,594 P12493 INFO Save best model: monitor(max): 1.755965
2022-08-11 11:13:24,651 P12493 INFO --- 2548/2548 batches finished ---
2022-08-11 11:13:24,751 P12493 INFO Train loss: 0.413001
2022-08-11 11:13:24,751 P12493 INFO ************ Epoch=14 end ************
2022-08-11 11:17:11,620 P12493 INFO [Metrics] AUC: 0.880079 - gAUC: 0.877829
2022-08-11 11:17:11,769 P12493 INFO Save best model: monitor(max): 1.757908
2022-08-11 11:17:11,815 P12493 INFO --- 2548/2548 batches finished ---
2022-08-11 11:17:11,900 P12493 INFO Train loss: 0.402607
2022-08-11 11:17:11,900 P12493 INFO ************ Epoch=15 end ************
2022-08-11 11:20:07,628 P12493 INFO [Metrics] AUC: 0.879470 - gAUC: 0.877128
2022-08-11 11:20:07,794 P12493 INFO Monitor(max) STOP: 1.756597 !
2022-08-11 11:20:07,795 P12493 INFO Reduce learning rate on plateau: 0.000005
2022-08-11 11:20:07,795 P12493 INFO --- 2548/2548 batches finished ---
2022-08-11 11:20:07,875 P12493 INFO Train loss: 0.394758
2022-08-11 11:20:07,875 P12493 INFO ************ Epoch=16 end ************
2022-08-11 11:22:42,775 P12493 INFO [Metrics] AUC: 0.875841 - gAUC: 0.873713
2022-08-11 11:22:42,929 P12493 INFO Monitor(max) STOP: 1.749554 !
2022-08-11 11:22:42,929 P12493 INFO Reduce learning rate on plateau: 0.000001
2022-08-11 11:22:42,929 P12493 INFO ********* Epoch==17 early stop *********
2022-08-11 11:22:42,930 P12493 INFO --- 2548/2548 batches finished ---
2022-08-11 11:22:43,018 P12493 INFO Train loss: 0.344606
2022-08-11 11:22:43,018 P12493 INFO Training finished.
2022-08-11 11:22:43,018 P12493 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DCN_amazonelectronics_x1/amazonelectronics_x1_b7a43f49/DCN_amazonelectronics_x1_029_366c6259.model
2022-08-11 11:22:43,235 P12493 INFO ****** Validation evaluation ******
2022-08-11 11:23:34,162 P12493 INFO [Metrics] gAUC: 0.877829 - AUC: 0.880079 - logloss: 0.442113
2022-08-11 11:23:34,383 P12493 INFO ******** Test evaluation ********
2022-08-11 11:23:34,383 P12493 INFO Loading data...
2022-08-11 11:23:34,383 P12493 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/test.h5
2022-08-11 11:23:34,801 P12493 INFO Test samples: total/384806, blocks/1
2022-08-11 11:23:34,801 P12493 INFO Loading test data done.
2022-08-11 11:24:13,959 P12493 INFO [Metrics] gAUC: 0.877829 - AUC: 0.880079 - logloss: 0.442113

```
