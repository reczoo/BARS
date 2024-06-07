## FinalMLP_avazu_x4_001

A hands-on guide to run the FinalMLP model on the Avazu_x4 dataset.

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

We use the [FinalMLP](https://github.com/reczoo/FuxiCTR/tree/v2.2.0/model_zoo/FinalMLP) model code from [FuxiCTR-v2.2.0](https://github.com/reczoo/FuxiCTR/tree/v2.2.0) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.2.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.2.0.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.2.0
    ```

2. Create a data directory and put the downloaded data files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FinalMLP_avazu_x4_tuner_config_03](./FinalMLP_avazu_x4_tuner_config_03). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/FinalMLP
    nohup python run_expid.py --config YOUR_PATH/FinalMLP/FinalMLP_avazu_x4_tuner_config_03 --expid FinalMLP_avazu_x4_001_006_a7c95fe1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.793537 | 0.371862  |


### Logs
```python
2024-02-22 12:55:35,485 P1084821 INFO Params: {
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_001_a31210da",
    "debug_mode": "False",
    "early_stop_patience": "1",
    "embedding_dim": "16",
    "embedding_regularizer": "1e-09",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': False, 'dtype': 'str', 'name': 'id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'type': 'categorical'}]",
    "feature_config": "None",
    "feature_specs": "None",
    "fs1_context": "[]",
    "fs2_context": "[]",
    "fs_hidden_units": "[1000]",
    "gpu": "5",
    "group_id": "None",
    "label_col": "{'dtype': 'float', 'name': 'click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "2",
    "mlp1_batch_norm": "True",
    "mlp1_dropout": "0",
    "mlp1_hidden_activations": "relu",
    "mlp1_hidden_units": "[2000, 2000, 2000]",
    "mlp2_batch_norm": "False",
    "mlp2_dropout": "0",
    "mlp2_hidden_activations": "relu",
    "mlp2_hidden_units": "[500]",
    "model": "FinalMLP",
    "model_id": "FinalMLP_avazu_x4_001_006_a7c95fe1",
    "model_root": "./checkpoints/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "num_heads": "20",
    "num_workers": "3",
    "optimizer": "adam",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x4/test.csv",
    "train_data": "../data/Avazu/Avazu_x4/train.csv",
    "use_features": "None",
    "use_fs": "False",
    "valid_data": "../data/Avazu/Avazu_x4/valid.csv",
    "verbose": "1"
}
2024-02-22 12:55:35,486 P1084821 INFO Set up feature processor...
2024-02-22 12:55:35,486 P1084821 WARNING Skip rebuilding ../data/Avazu/avazu_x4_001_a31210da/feature_map.json. Please delete it manually if rebuilding is required.
2024-02-22 12:55:35,487 P1084821 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_a31210da/feature_map.json
2024-02-22 12:55:35,487 P1084821 INFO Set column index...
2024-02-22 12:55:35,487 P1084821 INFO Feature specs: {
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
2024-02-22 12:55:41,390 P1084821 INFO Total number of parameters: 69046986.
2024-02-22 12:55:41,391 P1084821 INFO Loading datasets...
2024-02-22 12:56:01,114 P1084821 INFO Train samples: total/32343172, blocks/1
2024-02-22 12:56:03,573 P1084821 INFO Validation samples: total/4042897, blocks/1
2024-02-22 12:56:03,573 P1084821 INFO Loading train and validation data done.
2024-02-22 12:56:03,574 P1084821 INFO Start training: 3235 batches/epoch
2024-02-22 12:56:03,574 P1084821 INFO ************ Epoch=1 start ************
2024-02-22 13:02:12,158 P1084821 INFO Train loss: 0.382539
2024-02-22 13:02:12,158 P1084821 INFO Evaluation @epoch 1 - batch 3235: 
2024-02-22 13:02:29,486 P1084821 INFO [Metrics] AUC: 0.793356 - logloss: 0.371948
2024-02-22 13:02:29,489 P1084821 INFO Save best model: monitor(max)=0.421408
2024-02-22 13:02:30,208 P1084821 INFO ************ Epoch=1 end ************
2024-02-22 13:08:38,065 P1084821 INFO Train loss: 0.331726
2024-02-22 13:08:38,065 P1084821 INFO Evaluation @epoch 2 - batch 3235: 
2024-02-22 13:08:55,613 P1084821 INFO [Metrics] AUC: 0.789218 - logloss: 0.380356
2024-02-22 13:08:55,619 P1084821 INFO Monitor(max)=0.408862 STOP!
2024-02-22 13:08:55,619 P1084821 INFO Reduce learning rate on plateau: 0.000100
2024-02-22 13:08:55,619 P1084821 INFO ********* Epoch==2 early stop *********
2024-02-22 13:08:55,671 P1084821 INFO Training finished.
2024-02-22 13:08:55,671 P1084821 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/avazu_x4_001_a31210da/FinalMLP_avazu_x4_001_006_a7c95fe1.model
2024-02-22 13:08:55,895 P1084821 INFO ****** Validation evaluation ******
2024-02-22 13:09:13,147 P1084821 INFO [Metrics] AUC: 0.793356 - logloss: 0.371948
2024-02-22 13:09:13,251 P1084821 INFO ******** Test evaluation ********
2024-02-22 13:09:13,251 P1084821 INFO Loading datasets...
2024-02-22 13:09:15,854 P1084821 INFO Test samples: total/4042898, blocks/1
2024-02-22 13:09:15,854 P1084821 INFO Loading test data done.
2024-02-22 13:09:33,591 P1084821 INFO [Metrics] AUC: 0.793537 - logloss: 0.371862

```
