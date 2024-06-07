## FINAL_2B_avazu_x1

A hands-on guide to run the FINAL model on the Avazu_x1 dataset.

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
Please refer to the BARS dataset [Avazu_x1](https://github.com/openbenchmark/BARS/blob/main/datasets/Avazu#Avazu_x1) to get data ready.

### Code

We use the [FINAL](https://github.com/reczoo/FuxiCTR/blob/v2.0.2/model_zoo/FINAL) model code from [FuxiCTR-v2.0.2](https://github.com/reczoo/FuxiCTR/tree/v2.0.2) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.2.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.2
    ```

2. Create a data directory and put the downloaded data files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FINAL_2B_avazu_x1_tuner_config_04](./FINAL_2B_avazu_x1_tuner_config_04). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/FINAL
    nohup python run_expid.py --config XXX/benchmarks/FINAL/FINAL_2B_avazu_x1_tuner_config_04 --expid FINAL_avazu_x1_023_9d3d37bd --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.767159 | 0.365334  |


### Logs
```python
2022-12-31 16:20:08,154 P21850 INFO Params: {
    "batch_size": "4096",
    "block1_dropout": "0.3",
    "block1_hidden_activations": "ReLU",
    "block1_hidden_units": "[800]",
    "block2_dropout": "0.2",
    "block2_hidden_activations": "ReLU",
    "block2_hidden_units": "[800, 800]",
    "block_type": "2B",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_0bbde04e",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "10",
    "embedding_regularizer": "0.05",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "feature_specs": "None",
    "gpu": "7",
    "group_id": "None",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FINAL",
    "model_id": "FINAL_avazu_x1_023_9d3d37bd",
    "model_root": "./checkpoints/FINAL_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "norm_type": "BN",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_field_gate": "True",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "1"
}
2022-12-31 16:20:08,156 P21850 INFO Load feature_map from json: ../data/Avazu/avazu_x1_0bbde04e/feature_map.json
2022-12-31 16:20:08,156 P21850 INFO Set column index...
2022-12-31 16:20:08,156 P21850 INFO Feature specs: {
    "feat_1": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 8, 'vocab_size': 9}",
    "feat_10": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 1048284, 'vocab_size': 1048285}",
    "feat_11": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 6514, 'vocab_size': 6515}",
    "feat_12": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 5, 'vocab_size': 6}",
    "feat_13": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 5, 'vocab_size': 6}",
    "feat_14": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 1939, 'vocab_size': 1940}",
    "feat_15": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 9, 'vocab_size': 10}",
    "feat_16": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10, 'vocab_size': 11}",
    "feat_17": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 348, 'vocab_size': 349}",
    "feat_18": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 5, 'vocab_size': 6}",
    "feat_19": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 60, 'vocab_size': 61}",
    "feat_2": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 8, 'vocab_size': 9}",
    "feat_20": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 170, 'vocab_size': 171}",
    "feat_21": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 51, 'vocab_size': 52}",
    "feat_22": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 25, 'vocab_size': 26}",
    "feat_3": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 3479, 'vocab_size': 3480}",
    "feat_4": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 4270, 'vocab_size': 4271}",
    "feat_5": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 25, 'vocab_size': 26}",
    "feat_6": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 4863, 'vocab_size': 4864}",
    "feat_7": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 304, 'vocab_size': 305}",
    "feat_8": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 32, 'vocab_size': 33}",
    "feat_9": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 228185, 'vocab_size': 228186}"
}
2022-12-31 16:20:11,594 P21850 INFO Total number of parameters: 14163518.
2022-12-31 16:20:11,594 P21850 INFO Loading data...
2022-12-31 16:20:11,594 P21850 INFO Loading data from h5: ../data/Avazu/avazu_x1_0bbde04e/train.h5
2022-12-31 16:20:23,586 P21850 INFO Train samples: total/28300276, blocks/1
2022-12-31 16:20:23,586 P21850 INFO Loading data from h5: ../data/Avazu/avazu_x1_0bbde04e/valid.h5
2022-12-31 16:20:25,306 P21850 INFO Validation samples: total/4042897, blocks/1
2022-12-31 16:20:25,306 P21850 INFO Loading train and validation data done.
2022-12-31 16:20:25,306 P21850 INFO Start training: 6910 batches/epoch
2022-12-31 16:20:25,306 P21850 INFO ************ Epoch=1 start ************
2022-12-31 16:24:14,382 P21850 INFO [Metrics] AUC: 0.728069
2022-12-31 16:24:14,384 P21850 INFO Save best model: monitor(max): 0.728069
2022-12-31 16:24:14,482 P21850 INFO --- 6910/6910 batches finished ---
2022-12-31 16:24:14,521 P21850 INFO Train loss @epoch 1: 1.250046
2022-12-31 16:24:14,521 P21850 INFO ************ Epoch=1 end ************
2022-12-31 16:28:02,538 P21850 INFO [Metrics] AUC: 0.740246
2022-12-31 16:28:02,541 P21850 INFO Save best model: monitor(max): 0.740246
2022-12-31 16:28:02,645 P21850 INFO --- 6910/6910 batches finished ---
2022-12-31 16:28:02,685 P21850 INFO Train loss @epoch 2: 1.244454
2022-12-31 16:28:02,685 P21850 INFO ************ Epoch=2 end ************
2022-12-31 16:31:50,665 P21850 INFO [Metrics] AUC: 0.738322
2022-12-31 16:31:50,668 P21850 INFO Monitor(max) STOP: 0.738322 !
2022-12-31 16:31:50,668 P21850 INFO Reduce learning rate on plateau: 0.000100
2022-12-31 16:31:50,669 P21850 INFO --- 6910/6910 batches finished ---
2022-12-31 16:31:50,710 P21850 INFO Train loss @epoch 3: 1.243133
2022-12-31 16:31:50,710 P21850 INFO ************ Epoch=3 end ************
2022-12-31 16:35:44,914 P21850 INFO [Metrics] AUC: 0.745343
2022-12-31 16:35:44,919 P21850 INFO Save best model: monitor(max): 0.745343
2022-12-31 16:35:45,029 P21850 INFO --- 6910/6910 batches finished ---
2022-12-31 16:35:45,084 P21850 INFO Train loss @epoch 4: 1.205469
2022-12-31 16:35:45,084 P21850 INFO ************ Epoch=4 end ************
2022-12-31 16:39:34,126 P21850 INFO [Metrics] AUC: 0.747210
2022-12-31 16:39:34,129 P21850 INFO Save best model: monitor(max): 0.747210
2022-12-31 16:39:34,245 P21850 INFO --- 6910/6910 batches finished ---
2022-12-31 16:39:34,292 P21850 INFO Train loss @epoch 5: 1.204703
2022-12-31 16:39:34,292 P21850 INFO ************ Epoch=5 end ************
2022-12-31 16:43:22,054 P21850 INFO [Metrics] AUC: 0.747260
2022-12-31 16:43:22,057 P21850 INFO Save best model: monitor(max): 0.747260
2022-12-31 16:43:22,164 P21850 INFO --- 6910/6910 batches finished ---
2022-12-31 16:43:22,222 P21850 INFO Train loss @epoch 6: 1.204205
2022-12-31 16:43:22,222 P21850 INFO ************ Epoch=6 end ************
2022-12-31 16:47:08,847 P21850 INFO [Metrics] AUC: 0.747293
2022-12-31 16:47:08,850 P21850 INFO Save best model: monitor(max): 0.747293
2022-12-31 16:47:08,952 P21850 INFO --- 6910/6910 batches finished ---
2022-12-31 16:47:09,000 P21850 INFO Train loss @epoch 7: 1.203820
2022-12-31 16:47:09,001 P21850 INFO ************ Epoch=7 end ************
2022-12-31 16:50:54,223 P21850 INFO [Metrics] AUC: 0.745874
2022-12-31 16:50:54,225 P21850 INFO Monitor(max) STOP: 0.745874 !
2022-12-31 16:50:54,225 P21850 INFO Reduce learning rate on plateau: 0.000010
2022-12-31 16:50:54,226 P21850 INFO --- 6910/6910 batches finished ---
2022-12-31 16:50:54,269 P21850 INFO Train loss @epoch 8: 1.203478
2022-12-31 16:50:54,269 P21850 INFO ************ Epoch=8 end ************
2022-12-31 16:54:38,444 P21850 INFO [Metrics] AUC: 0.748371
2022-12-31 16:54:38,447 P21850 INFO Save best model: monitor(max): 0.748371
2022-12-31 16:54:38,550 P21850 INFO --- 6910/6910 batches finished ---
2022-12-31 16:54:38,592 P21850 INFO Train loss @epoch 9: 1.182463
2022-12-31 16:54:38,592 P21850 INFO ************ Epoch=9 end ************
2022-12-31 16:58:26,514 P21850 INFO [Metrics] AUC: 0.747002
2022-12-31 16:58:26,518 P21850 INFO Monitor(max) STOP: 0.747002 !
2022-12-31 16:58:26,519 P21850 INFO Reduce learning rate on plateau: 0.000001
2022-12-31 16:58:26,519 P21850 INFO --- 6910/6910 batches finished ---
2022-12-31 16:58:26,561 P21850 INFO Train loss @epoch 10: 1.173606
2022-12-31 16:58:26,562 P21850 INFO ************ Epoch=10 end ************
2022-12-31 17:02:07,498 P21850 INFO [Metrics] AUC: 0.744956
2022-12-31 17:02:07,501 P21850 INFO Monitor(max) STOP: 0.744956 !
2022-12-31 17:02:07,501 P21850 INFO Reduce learning rate on plateau: 0.000001
2022-12-31 17:02:07,501 P21850 INFO ********* Epoch==11 early stop *********
2022-12-31 17:02:07,502 P21850 INFO --- 6910/6910 batches finished ---
2022-12-31 17:02:07,545 P21850 INFO Train loss @epoch 11: 1.162006
2022-12-31 17:02:07,546 P21850 INFO Training finished.
2022-12-31 17:02:07,546 P21850 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/FINAL_avazu_x1/avazu_x1_0bbde04e/FINAL_avazu_x1_023_9d3d37bd.model
2022-12-31 17:02:07,604 P21850 INFO ****** Validation evaluation ******
2022-12-31 17:02:18,166 P21850 INFO [Metrics] AUC: 0.748371 - logloss: 0.395190
2022-12-31 17:02:18,250 P21850 INFO ******** Test evaluation ********
2022-12-31 17:02:18,250 P21850 INFO Loading data...
2022-12-31 17:02:18,251 P21850 INFO Loading data from h5: ../data/Avazu/avazu_x1_0bbde04e/test.h5
2022-12-31 17:02:21,734 P21850 INFO Test samples: total/8085794, blocks/1
2022-12-31 17:02:21,734 P21850 INFO Loading test data done.
2022-12-31 17:02:45,337 P21850 INFO [Metrics] AUC: 0.767159 - logloss: 0.365334

```
