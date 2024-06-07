## MaskNet_avazu_x4_001

A hands-on guide to run the MaskNet model on the Avazu_x4 dataset.

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
  CUDA: 10.2
  python: 3.7.10
  pytorch: 1.10.2+cu102
  pandas: 1.1.5
  numpy: 1.19.5
  scipy: 1.5.2
  sklearn: 0.22.1
  pyyaml: 6.0.1
  h5py: 2.8.0
  tqdm: 4.64.0
  keras_preprocessing: 1.1.2
  fuxictr: 2.2.0

  ```

### Dataset
Please refer to [Avazu_x4]([Avazu_x4](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x4)) to get the dataset details.

### Code

We use the [MaskNet](https://github.com/reczoo/FuxiCTR/tree/v2.2.0/model_zoo/MaskNet) model code from [FuxiCTR-v2.2.0](https://github.com/reczoo/FuxiCTR/tree/v2.2.0) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.2.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.2.0.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.2.0
    ```

2. Create a data directory and put the downloaded data files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [MaskNet_avazu_x4_tuner_config_03](./MaskNet_avazu_x4_tuner_config_03). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/MaskNet
    nohup python run_expid.py --config YOUR_PATH/MaskNet/MaskNet_avazu_x4_tuner_config_03 --expid MaskNet_avazu_x4_001_019_541571c0 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.794382 | 0.371189  |


### Logs
```python
2024-02-22 20:06:23,961 P316688 INFO Params: {
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_001_a31210da",
    "debug_mode": "False",
    "dnn_hidden_activations": "relu",
    "dnn_hidden_units": "[2000, 2000, 2000]",
    "early_stop_patience": "1",
    "emb_layernorm": "False",
    "embedding_dim": "16",
    "embedding_regularizer": "1e-09",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': False, 'dtype': 'str', 'name': 'id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'type': 'categorical'}]",
    "feature_config": "None",
    "feature_specs": "None",
    "gpu": "1",
    "group_id": "None",
    "label_col": "{'dtype': 'float', 'name': 'click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "2",
    "model": "MaskNet",
    "model_id": "MaskNet_avazu_x4_001_019_541571c0",
    "model_root": "./checkpoints/",
    "model_type": "ParallelMaskNet",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_layernorm": "False",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_block_dim": "500",
    "parallel_num_blocks": "6",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "0.8",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x4/test.csv",
    "train_data": "../data/Avazu/Avazu_x4/train.csv",
    "use_features": "None",
    "valid_data": "../data/Avazu/Avazu_x4/valid.csv",
    "verbose": "1"
}
2024-02-22 20:06:23,962 P316688 INFO Set up feature processor...
2024-02-22 20:06:23,963 P316688 WARNING Skip rebuilding ../data/Avazu/avazu_x4_001_a31210da/feature_map.json. Please delete it manually if rebuilding is required.
2024-02-22 20:06:23,963 P316688 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_a31210da/feature_map.json
2024-02-22 20:06:23,963 P316688 INFO Set column index...
2024-02-22 20:06:23,963 P316688 INFO Feature specs: {
    "C1": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 9}",
    "C14": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 2556}",
    "C15": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 10}",
    "C16": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 11}",
    "C17": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 434}",
    "C18": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 6}",
    "C19": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 70}",
    "C20": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 173}",
    "C21": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 62}",
    "app_category": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 33}",
    "app_domain": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 400}",
    "app_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 6545}",
    "banner_pos": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 9}",
    "device_conn_type": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 6}",
    "device_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 820509}",
    "device_ip": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 2903322}",
    "device_model": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 7259}",
    "device_type": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 7}",
    "hour": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 26}",
    "site_category": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 27}",
    "site_domain": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 5461}",
    "site_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 4051}",
    "weekday": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 9}",
    "weekend": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 4}"
}
2024-02-22 20:06:28,906 P316688 INFO Total number of parameters: 76594787.
2024-02-22 20:06:28,907 P316688 INFO Loading datasets...
2024-02-22 20:06:49,843 P316688 INFO Train samples: total/32343172, blocks/1
2024-02-22 20:06:52,314 P316688 INFO Validation samples: total/4042897, blocks/1
2024-02-22 20:06:52,314 P316688 INFO Loading train and validation data done.
2024-02-22 20:06:52,314 P316688 INFO Start training: 3235 batches/epoch
2024-02-22 20:06:52,314 P316688 INFO ************ Epoch=1 start ************
2024-02-22 20:13:41,048 P316688 INFO Train loss: 0.379669
2024-02-22 20:13:41,048 P316688 INFO Evaluation @epoch 1 - batch 3235: 
2024-02-22 20:14:00,960 P316688 INFO [Metrics] AUC: 0.794253 - logloss: 0.371254
2024-02-22 20:14:00,963 P316688 INFO Save best model: monitor(max)=0.423000
2024-02-22 20:14:01,495 P316688 INFO ************ Epoch=1 end ************
2024-02-22 20:20:51,541 P316688 INFO Train loss: 0.329410
2024-02-22 20:20:51,542 P316688 INFO Evaluation @epoch 2 - batch 3235: 
2024-02-22 20:21:11,342 P316688 INFO [Metrics] AUC: 0.788830 - logloss: 0.379190
2024-02-22 20:21:11,346 P316688 INFO Monitor(max)=0.409639 STOP!
2024-02-22 20:21:11,346 P316688 INFO Reduce learning rate on plateau: 0.000100
2024-02-22 20:21:11,346 P316688 INFO ********* Epoch==2 early stop *********
2024-02-22 20:21:11,397 P316688 INFO Training finished.
2024-02-22 20:21:11,397 P316688 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/avazu_x4_001_a31210da/MaskNet_avazu_x4_001_019_541571c0.model
2024-02-22 20:21:11,641 P316688 INFO ****** Validation evaluation ******
2024-02-22 20:21:31,510 P316688 INFO [Metrics] AUC: 0.794253 - logloss: 0.371254
2024-02-22 20:21:31,627 P316688 INFO ******** Test evaluation ********
2024-02-22 20:21:31,627 P316688 INFO Loading datasets...
2024-02-22 20:21:34,114 P316688 INFO Test samples: total/4042898, blocks/1
2024-02-22 20:21:34,114 P316688 INFO Loading test data done.
2024-02-22 20:21:53,769 P316688 INFO [Metrics] AUC: 0.794382 - logloss: 0.371189

```
