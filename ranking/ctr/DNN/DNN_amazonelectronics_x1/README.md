## DNN_amazonelectronics_x1

A hands-on guide to run the DNN model on the AmazonElectronics_x1 dataset.

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

We use the [DNN](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/DNN) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Amazon/AmazonElectronics_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DNN_amazonelectronics_x1_tuner_config_01](./DNN_amazonelectronics_x1_tuner_config_01). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DNN
    nohup python run_expid.py --config XXX/benchmarks/DNN/DNN_amazonelectronics_x1_tuner_config_01 --expid DNN_amazonelectronics_x1_005_0a5f1f4b --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.878957 | 0.881546 | 0.440110  |


### Logs
```python
2023-05-14 07:36:07,564 P14474 INFO Params: {
    "batch_norm": "True",
    "batch_size": "1024",
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
    "feature_specs": "None",
    "gpu": "4",
    "group_id": "user_id",
    "hidden_activations": "relu",
    "hidden_units": "[1024, 512, 256]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DNN",
    "model_id": "DNN_amazonelectronics_x1_005_0a5f1f4b",
    "model_root": "./checkpoints/DNN_amazonelectronics_x1/",
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
    "test_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "train_data": "../data/Amazon/AmazonElectronics_x1/train.csv",
    "valid_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "verbose": "1"
}
2023-05-14 07:36:07,565 P14474 INFO Set up feature processor...
2023-05-14 07:36:07,565 P14474 WARNING Skip rebuilding ../data/Amazon/amazonelectronics_x1_b7a43f49/feature_map.json. Please delete it manually if rebuilding is required.
2023-05-14 07:36:07,565 P14474 INFO Load feature_map from json: ../data/Amazon/amazonelectronics_x1_b7a43f49/feature_map.json
2023-05-14 07:36:07,566 P14474 INFO Set column index...
2023-05-14 07:36:07,566 P14474 INFO Feature specs: {
    "cate_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803, 'max_len': 100}",
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803}",
    "item_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003, 'max_len': 100}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003}",
    "user_id": "{'type': 'meta'}"
}
2023-05-14 07:36:14,202 P14474 INFO Total number of parameters: 5006721.
2023-05-14 07:36:14,203 P14474 INFO Loading data...
2023-05-14 07:36:14,203 P14474 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/train.h5
2023-05-14 07:36:17,678 P14474 INFO Train samples: total/2608764, blocks/1
2023-05-14 07:36:17,678 P14474 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/valid.h5
2023-05-14 07:36:18,130 P14474 INFO Validation samples: total/384806, blocks/1
2023-05-14 07:36:18,130 P14474 INFO Loading train and validation data done.
2023-05-14 07:36:18,199 P14474 INFO Start training: 2548 batches/epoch
2023-05-14 07:36:18,199 P14474 INFO ************ Epoch=1 start ************
2023-05-14 07:44:23,678 P14474 INFO Train loss: 0.640565
2023-05-14 07:44:23,678 P14474 INFO Evaluation @epoch 1 - batch 2548: 
2023-05-14 07:47:02,662 P14474 INFO [Metrics] AUC: 0.807212 - gAUC: 0.807976
2023-05-14 07:47:02,664 P14474 INFO Save best model: monitor(max)=1.615188
2023-05-14 07:47:02,746 P14474 INFO ************ Epoch=1 end ************
2023-05-14 07:55:22,051 P14474 INFO Train loss: 0.592809
2023-05-14 07:55:22,057 P14474 INFO Evaluation @epoch 2 - batch 2548: 
2023-05-14 07:57:58,912 P14474 INFO [Metrics] AUC: 0.849016 - gAUC: 0.846910
2023-05-14 07:57:58,913 P14474 INFO Save best model: monitor(max)=1.695926
2023-05-14 07:57:59,029 P14474 INFO ************ Epoch=2 end ************
2023-05-14 08:06:12,975 P14474 INFO Train loss: 0.574785
2023-05-14 08:06:12,979 P14474 INFO Evaluation @epoch 3 - batch 2548: 
2023-05-14 08:08:56,681 P14474 INFO [Metrics] AUC: 0.853434 - gAUC: 0.850210
2023-05-14 08:08:56,682 P14474 INFO Save best model: monitor(max)=1.703645
2023-05-14 08:08:56,868 P14474 INFO ************ Epoch=3 end ************
2023-05-14 08:17:15,584 P14474 INFO Train loss: 0.569511
2023-05-14 08:17:15,585 P14474 INFO Evaluation @epoch 4 - batch 2548: 
2023-05-14 08:19:52,923 P14474 INFO [Metrics] AUC: 0.853828 - gAUC: 0.851317
2023-05-14 08:19:52,923 P14474 INFO Save best model: monitor(max)=1.705145
2023-05-14 08:19:53,068 P14474 INFO ************ Epoch=4 end ************
2023-05-14 08:28:09,551 P14474 INFO Train loss: 0.565181
2023-05-14 08:28:09,555 P14474 INFO Evaluation @epoch 5 - batch 2548: 
2023-05-14 08:30:48,615 P14474 INFO [Metrics] AUC: 0.857014 - gAUC: 0.854654
2023-05-14 08:30:48,616 P14474 INFO Save best model: monitor(max)=1.711668
2023-05-14 08:30:48,828 P14474 INFO ************ Epoch=5 end ************
2023-05-14 08:38:53,871 P14474 INFO Train loss: 0.564680
2023-05-14 08:38:53,871 P14474 INFO Evaluation @epoch 6 - batch 2548: 
2023-05-14 08:41:25,968 P14474 INFO [Metrics] AUC: 0.857812 - gAUC: 0.855070
2023-05-14 08:41:25,974 P14474 INFO Save best model: monitor(max)=1.712882
2023-05-14 08:41:26,107 P14474 INFO ************ Epoch=6 end ************
2023-05-14 08:49:04,907 P14474 INFO Train loss: 0.563134
2023-05-14 08:49:04,907 P14474 INFO Evaluation @epoch 7 - batch 2548: 
2023-05-14 08:51:25,064 P14474 INFO [Metrics] AUC: 0.860156 - gAUC: 0.858064
2023-05-14 08:51:25,070 P14474 INFO Save best model: monitor(max)=1.718220
2023-05-14 08:51:25,237 P14474 INFO ************ Epoch=7 end ************
2023-05-14 08:58:26,514 P14474 INFO Train loss: 0.562746
2023-05-14 08:58:26,514 P14474 INFO Evaluation @epoch 8 - batch 2548: 
2023-05-14 09:00:35,464 P14474 INFO [Metrics] AUC: 0.860808 - gAUC: 0.858058
2023-05-14 09:00:35,465 P14474 INFO Save best model: monitor(max)=1.718866
2023-05-14 09:00:35,588 P14474 INFO ************ Epoch=8 end ************
2023-05-14 09:07:06,208 P14474 INFO Train loss: 0.562858
2023-05-14 09:07:06,209 P14474 INFO Evaluation @epoch 9 - batch 2548: 
2023-05-14 09:09:12,919 P14474 INFO [Metrics] AUC: 0.857386 - gAUC: 0.855065
2023-05-14 09:09:12,920 P14474 INFO Monitor(max)=1.712451 STOP!
2023-05-14 09:09:12,920 P14474 INFO Reduce learning rate on plateau: 0.000050
2023-05-14 09:09:12,979 P14474 INFO ************ Epoch=9 end ************
2023-05-14 09:15:44,999 P14474 INFO Train loss: 0.466091
2023-05-14 09:15:45,000 P14474 INFO Evaluation @epoch 10 - batch 2548: 
2023-05-14 09:17:52,414 P14474 INFO [Metrics] AUC: 0.876534 - gAUC: 0.873952
2023-05-14 09:17:52,415 P14474 INFO Save best model: monitor(max)=1.750486
2023-05-14 09:17:52,550 P14474 INFO ************ Epoch=10 end ************
2023-05-14 09:24:15,432 P14474 INFO Train loss: 0.420790
2023-05-14 09:24:15,433 P14474 INFO Evaluation @epoch 11 - batch 2548: 
2023-05-14 09:26:11,275 P14474 INFO [Metrics] AUC: 0.880901 - gAUC: 0.878510
2023-05-14 09:26:11,284 P14474 INFO Save best model: monitor(max)=1.759411
2023-05-14 09:26:11,412 P14474 INFO ************ Epoch=11 end ************
2023-05-14 09:32:22,281 P14474 INFO Train loss: 0.400774
2023-05-14 09:32:22,281 P14474 INFO Evaluation @epoch 12 - batch 2548: 
2023-05-14 09:34:15,214 P14474 INFO [Metrics] AUC: 0.881546 - gAUC: 0.878957
2023-05-14 09:34:15,215 P14474 INFO Save best model: monitor(max)=1.760503
2023-05-14 09:34:15,371 P14474 INFO ************ Epoch=12 end ************
2023-05-14 09:40:06,445 P14474 INFO Train loss: 0.387786
2023-05-14 09:40:06,445 P14474 INFO Evaluation @epoch 13 - batch 2548: 
2023-05-14 09:41:52,325 P14474 INFO [Metrics] AUC: 0.880033 - gAUC: 0.877637
2023-05-14 09:41:52,326 P14474 INFO Monitor(max)=1.757670 STOP!
2023-05-14 09:41:52,326 P14474 INFO Reduce learning rate on plateau: 0.000005
2023-05-14 09:41:52,405 P14474 INFO ************ Epoch=13 end ************
2023-05-14 09:46:41,471 P14474 INFO Train loss: 0.334238
2023-05-14 09:46:41,472 P14474 INFO Evaluation @epoch 14 - batch 2548: 
2023-05-14 09:47:59,038 P14474 INFO [Metrics] AUC: 0.878295 - gAUC: 0.875943
2023-05-14 09:47:59,039 P14474 INFO Monitor(max)=1.754237 STOP!
2023-05-14 09:47:59,039 P14474 INFO Reduce learning rate on plateau: 0.000001
2023-05-14 09:47:59,039 P14474 INFO ********* Epoch==14 early stop *********
2023-05-14 09:47:59,110 P14474 INFO Training finished.
2023-05-14 09:47:59,111 P14474 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DNN_amazonelectronics_x1/amazonelectronics_x1_b7a43f49/DNN_amazonelectronics_x1_005_0a5f1f4b.model
2023-05-14 09:47:59,146 P14474 INFO ****** Validation evaluation ******
2023-05-14 09:49:16,403 P14474 INFO [Metrics] gAUC: 0.878957 - AUC: 0.881546 - logloss: 0.440110
2023-05-14 09:49:16,498 P14474 INFO ******** Test evaluation ********
2023-05-14 09:49:16,499 P14474 INFO Loading data...
2023-05-14 09:49:16,499 P14474 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/test.h5
2023-05-14 09:49:16,900 P14474 INFO Test samples: total/384806, blocks/1
2023-05-14 09:49:16,900 P14474 INFO Loading test data done.
2023-05-14 09:50:25,901 P14474 INFO [Metrics] gAUC: 0.878957 - AUC: 0.881546 - logloss: 0.440110

```
