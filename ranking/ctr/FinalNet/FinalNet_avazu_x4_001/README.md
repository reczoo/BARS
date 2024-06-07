## FinalNet_avazu_x4_001

A hands-on guide to run the FinalNet model on the Avazu_x4 dataset.

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

We use the [FinalNet](https://github.com/reczoo/FuxiCTR/tree/v2.2.0/model_zoo/FinalNet) model code from [FuxiCTR-v2.2.0](https://github.com/reczoo/FuxiCTR/tree/v2.2.0) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.2.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.2.0.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.2.0
    ```

2. Create a data directory and put the downloaded data files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FinalNet_avazu_x4_tuner_config_04](./FinalNet_avazu_x4_tuner_config_04). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/FinalNet
    nohup python run_expid.py --config YOUR_PATH/FinalNet/FinalNet_avazu_x4_tuner_config_04 --expid FinalNet_avazu_x4_001_015_4b405413 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.794116 | 0.371254  |


### Logs
```python
2024-02-22 13:22:42,272 P1566359 INFO Params: {
    "batch_norm": "True",
    "batch_size": "8192",
    "block1_dropout": "0",
    "block1_hidden_activations": "ReLU",
    "block1_hidden_units": "[2000, 2000, 2000]",
    "block2_dropout": "0.1",
    "block2_hidden_activations": "ReLU",
    "block2_hidden_units": "[500]",
    "block_type": "2B",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_001_a31210da",
    "debug_mode": "False",
    "early_stop_patience": "1",
    "embedding_dim": "16",
    "embedding_regularizer": "0",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': False, 'dtype': 'str', 'name': 'id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'type': 'categorical'}]",
    "feature_specs": "None",
    "gpu": "6",
    "group_id": "None",
    "label_col": "{'dtype': 'float', 'name': 'click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "2",
    "model": "FinalNet",
    "model_id": "FinalNet_avazu_x4_001_015_4b405413",
    "model_root": "./checkpoints/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "residual_type": "concat",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x4/test.csv",
    "train_data": "../data/Avazu/Avazu_x4/train.csv",
    "use_feature_gating": "True",
    "valid_data": "../data/Avazu/Avazu_x4/valid.csv",
    "verbose": "1"
}
2024-02-22 13:22:42,273 P1566359 INFO Set up feature processor...
2024-02-22 13:22:42,273 P1566359 WARNING Skip rebuilding ../data/Avazu/avazu_x4_001_a31210da/feature_map.json. Please delete it manually if rebuilding is required.
2024-02-22 13:22:42,273 P1566359 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_a31210da/feature_map.json
2024-02-22 13:22:42,273 P1566359 INFO Set column index...
2024-02-22 13:22:42,274 P1566359 INFO Feature specs: {
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
2024-02-22 13:22:46,770 P1566359 INFO Total number of parameters: 69766586.
2024-02-22 13:22:46,770 P1566359 INFO Loading datasets...
2024-02-22 13:23:07,420 P1566359 INFO Train samples: total/32343172, blocks/1
2024-02-22 13:23:09,945 P1566359 INFO Validation samples: total/4042897, blocks/1
2024-02-22 13:23:09,946 P1566359 INFO Loading train and validation data done.
2024-02-22 13:23:09,946 P1566359 INFO Start training: 3949 batches/epoch
2024-02-22 13:23:09,946 P1566359 INFO ************ Epoch=1 start ************
2024-02-22 13:28:17,207 P1566359 INFO Train loss: 0.379626
2024-02-22 13:28:17,207 P1566359 INFO Evaluation @epoch 1 - batch 3949: 
2024-02-22 13:28:33,023 P1566359 INFO [Metrics] AUC: 0.793961 - logloss: 0.371360
2024-02-22 13:28:33,026 P1566359 INFO Save best model: monitor(max)=0.422601
2024-02-22 13:28:33,641 P1566359 INFO ************ Epoch=1 end ************
2024-02-22 13:33:41,213 P1566359 INFO Train loss: 0.330151
2024-02-22 13:33:41,213 P1566359 INFO Evaluation @epoch 2 - batch 3949: 
2024-02-22 13:33:57,482 P1566359 INFO [Metrics] AUC: 0.789077 - logloss: 0.381398
2024-02-22 13:33:57,485 P1566359 INFO Monitor(max)=0.407679 STOP!
2024-02-22 13:33:57,485 P1566359 INFO Reduce learning rate on plateau: 0.000100
2024-02-22 13:33:57,485 P1566359 INFO ********* Epoch==2 early stop *********
2024-02-22 13:33:57,530 P1566359 INFO Training finished.
2024-02-22 13:33:57,531 P1566359 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/avazu_x4_001_a31210da/FinalNet_avazu_x4_001_015_4b405413.model
2024-02-22 13:33:57,720 P1566359 INFO ****** Validation evaluation ******
2024-02-22 13:34:13,540 P1566359 INFO [Metrics] AUC: 0.793961 - logloss: 0.371360
2024-02-22 13:34:13,621 P1566359 INFO ******** Test evaluation ********
2024-02-22 13:34:13,621 P1566359 INFO Loading datasets...
2024-02-22 13:34:16,143 P1566359 INFO Test samples: total/4042898, blocks/1
2024-02-22 13:34:16,143 P1566359 INFO Loading test data done.
2024-02-22 13:34:31,959 P1566359 INFO [Metrics] AUC: 0.794116 - logloss: 0.371254

```
