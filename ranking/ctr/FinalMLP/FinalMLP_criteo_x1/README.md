## FinalMLP_criteo_x1

A hands-on guide to run the FinalMLP model on the Criteo_x1 dataset.

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
Please refer to the BARS dataset [Criteo_x1](https://github.com/openbenchmark/BARS/blob/main/datasets/Criteo#Criteo_x1) to get data ready.

### Code

We use the [FinalMLP](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/FinalMLP) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FinalMLP_criteo_x1_tuner_config_07](./FinalMLP_criteo_x1_tuner_config_07). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/FinalMLP
    nohup python run_expid.py --config XXX/benchmarks/FinalMLP/FinalMLP_criteo_x1_tuner_config_07 --expid FinalMLP_criteo_x1_004_d5d36917 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.814902 | 0.437025  |


### Logs
```python
2022-12-18 10:57:07,848 P6713 INFO Params: {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_49102577",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "feature_specs": "None",
    "fs1_context": "['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13']",
    "fs2_context": "['C6']",
    "fs_hidden_units": "[800]",
    "gpu": "3",
    "group_id": "None",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "mlp1_batch_norm": "True",
    "mlp1_dropout": "0.2",
    "mlp1_hidden_activations": "relu",
    "mlp1_hidden_units": "[400, 400, 400]",
    "mlp2_batch_norm": "True",
    "mlp2_dropout": "0.3",
    "mlp2_hidden_activations": "relu",
    "mlp2_hidden_units": "[1000]",
    "model": "FinalMLP",
    "model_id": "FinalMLP_criteo_x1_004_d5d36917",
    "model_root": "./checkpoints/FinalMLP_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "num_heads": "200",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_fs": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "0"
}
2022-12-18 10:57:07,848 P6713 INFO Load feature_map from json: ../data/Criteo/criteo_x1_49102577/feature_map.json
2022-12-18 10:57:07,849 P6713 INFO Set column index...
2022-12-18 10:57:07,849 P6713 INFO Feature specs: {
    "C1": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 1461, 'vocab_size': 1462}",
    "C10": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 54711, 'vocab_size': 54712}",
    "C11": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 5349, 'vocab_size': 5350}",
    "C12": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 409748, 'vocab_size': 409749}",
    "C13": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 3181, 'vocab_size': 3182}",
    "C14": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 28, 'vocab_size': 29}",
    "C15": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 12499, 'vocab_size': 12500}",
    "C16": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 365810, 'vocab_size': 365811}",
    "C17": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 11, 'vocab_size': 12}",
    "C18": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 4933, 'vocab_size': 4934}",
    "C19": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 2095, 'vocab_size': 2096}",
    "C2": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 559, 'vocab_size': 560}",
    "C20": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 5, 'vocab_size': 6}",
    "C21": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 397980, 'vocab_size': 397981}",
    "C22": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 19, 'vocab_size': 20}",
    "C23": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 16, 'vocab_size': 17}",
    "C24": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 88607, 'vocab_size': 88608}",
    "C25": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 97, 'vocab_size': 98}",
    "C26": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 64072, 'vocab_size': 64073}",
    "C3": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 413423, 'vocab_size': 413424}",
    "C4": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 248542, 'vocab_size': 248543}",
    "C5": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 306, 'vocab_size': 307}",
    "C6": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 22, 'vocab_size': 23}",
    "C7": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 12191, 'vocab_size': 12192}",
    "C8": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 634, 'vocab_size': 635}",
    "C9": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 4, 'vocab_size': 5}",
    "I1": "{'source': '', 'type': 'numeric'}",
    "I10": "{'source': '', 'type': 'numeric'}",
    "I11": "{'source': '', 'type': 'numeric'}",
    "I12": "{'source': '', 'type': 'numeric'}",
    "I13": "{'source': '', 'type': 'numeric'}",
    "I2": "{'source': '', 'type': 'numeric'}",
    "I3": "{'source': '', 'type': 'numeric'}",
    "I4": "{'source': '', 'type': 'numeric'}",
    "I5": "{'source': '', 'type': 'numeric'}",
    "I6": "{'source': '', 'type': 'numeric'}",
    "I7": "{'source': '', 'type': 'numeric'}",
    "I8": "{'source': '', 'type': 'numeric'}",
    "I9": "{'source': '', 'type': 'numeric'}"
}
2022-12-18 10:57:13,146 P6713 INFO Total number of parameters: 22478162.
2022-12-18 10:57:13,147 P6713 INFO Loading data...
2022-12-18 10:57:13,147 P6713 INFO Loading data from h5: ../data/Criteo/criteo_x1_49102577/train.h5
2022-12-18 10:57:42,980 P6713 INFO Train samples: total/33003326, blocks/1
2022-12-18 10:57:42,980 P6713 INFO Loading data from h5: ../data/Criteo/criteo_x1_49102577/valid.h5
2022-12-18 10:57:48,569 P6713 INFO Validation samples: total/8250124, blocks/1
2022-12-18 10:57:48,569 P6713 INFO Loading train and validation data done.
2022-12-18 10:57:48,569 P6713 INFO Start training: 8058 batches/epoch
2022-12-18 10:57:48,569 P6713 INFO ************ Epoch=1 start ************
2022-12-18 11:06:34,746 P6713 INFO [Metrics] AUC: 0.804593
2022-12-18 11:06:34,748 P6713 INFO Save best model: monitor(max): 0.804593
2022-12-18 11:06:34,896 P6713 INFO --- 8058/8058 batches finished ---
2022-12-18 11:06:34,948 P6713 INFO Train loss @epoch 1: 0.463153
2022-12-18 11:06:34,948 P6713 INFO ************ Epoch=1 end ************
2022-12-18 11:15:18,499 P6713 INFO [Metrics] AUC: 0.807331
2022-12-18 11:15:18,501 P6713 INFO Save best model: monitor(max): 0.807331
2022-12-18 11:15:18,648 P6713 INFO --- 8058/8058 batches finished ---
2022-12-18 11:15:18,695 P6713 INFO Train loss @epoch 2: 0.455107
2022-12-18 11:15:18,695 P6713 INFO ************ Epoch=2 end ************
2022-12-18 11:24:02,727 P6713 INFO [Metrics] AUC: 0.808377
2022-12-18 11:24:02,728 P6713 INFO Save best model: monitor(max): 0.808377
2022-12-18 11:24:02,893 P6713 INFO --- 8058/8058 batches finished ---
2022-12-18 11:24:02,943 P6713 INFO Train loss @epoch 3: 0.453147
2022-12-18 11:24:02,943 P6713 INFO ************ Epoch=3 end ************
2022-12-18 11:32:46,953 P6713 INFO [Metrics] AUC: 0.809333
2022-12-18 11:32:46,954 P6713 INFO Save best model: monitor(max): 0.809333
2022-12-18 11:32:47,102 P6713 INFO --- 8058/8058 batches finished ---
2022-12-18 11:32:47,151 P6713 INFO Train loss @epoch 4: 0.452195
2022-12-18 11:32:47,151 P6713 INFO ************ Epoch=4 end ************
2022-12-18 11:41:30,321 P6713 INFO [Metrics] AUC: 0.809723
2022-12-18 11:41:30,322 P6713 INFO Save best model: monitor(max): 0.809723
2022-12-18 11:41:30,469 P6713 INFO --- 8058/8058 batches finished ---
2022-12-18 11:41:30,519 P6713 INFO Train loss @epoch 5: 0.451618
2022-12-18 11:41:30,519 P6713 INFO ************ Epoch=5 end ************
2022-12-18 11:50:13,325 P6713 INFO [Metrics] AUC: 0.810162
2022-12-18 11:50:13,327 P6713 INFO Save best model: monitor(max): 0.810162
2022-12-18 11:50:13,471 P6713 INFO --- 8058/8058 batches finished ---
2022-12-18 11:50:13,520 P6713 INFO Train loss @epoch 6: 0.451191
2022-12-18 11:50:13,520 P6713 INFO ************ Epoch=6 end ************
2022-12-18 11:58:59,344 P6713 INFO [Metrics] AUC: 0.810490
2022-12-18 11:58:59,346 P6713 INFO Save best model: monitor(max): 0.810490
2022-12-18 11:58:59,501 P6713 INFO --- 8058/8058 batches finished ---
2022-12-18 11:58:59,554 P6713 INFO Train loss @epoch 7: 0.450843
2022-12-18 11:58:59,554 P6713 INFO ************ Epoch=7 end ************
2022-12-18 12:07:49,659 P6713 INFO [Metrics] AUC: 0.810449
2022-12-18 12:07:49,661 P6713 INFO Monitor(max) STOP: 0.810449 !
2022-12-18 12:07:49,661 P6713 INFO Reduce learning rate on plateau: 0.000100
2022-12-18 12:07:49,662 P6713 INFO --- 8058/8058 batches finished ---
2022-12-18 12:07:49,719 P6713 INFO Train loss @epoch 8: 0.450561
2022-12-18 12:07:49,719 P6713 INFO ************ Epoch=8 end ************
2022-12-18 12:16:38,199 P6713 INFO [Metrics] AUC: 0.813866
2022-12-18 12:16:38,201 P6713 INFO Save best model: monitor(max): 0.813866
2022-12-18 12:16:38,345 P6713 INFO --- 8058/8058 batches finished ---
2022-12-18 12:16:38,399 P6713 INFO Train loss @epoch 9: 0.440473
2022-12-18 12:16:38,399 P6713 INFO ************ Epoch=9 end ************
2022-12-18 12:25:29,980 P6713 INFO [Metrics] AUC: 0.814427
2022-12-18 12:25:29,982 P6713 INFO Save best model: monitor(max): 0.814427
2022-12-18 12:25:30,138 P6713 INFO --- 8058/8058 batches finished ---
2022-12-18 12:25:30,186 P6713 INFO Train loss @epoch 10: 0.436416
2022-12-18 12:25:30,186 P6713 INFO ************ Epoch=10 end ************
2022-12-18 12:34:23,502 P6713 INFO [Metrics] AUC: 0.814554
2022-12-18 12:34:23,504 P6713 INFO Save best model: monitor(max): 0.814554
2022-12-18 12:34:23,647 P6713 INFO --- 8058/8058 batches finished ---
2022-12-18 12:34:23,696 P6713 INFO Train loss @epoch 11: 0.434451
2022-12-18 12:34:23,696 P6713 INFO ************ Epoch=11 end ************
2022-12-18 12:43:15,962 P6713 INFO [Metrics] AUC: 0.814430
2022-12-18 12:43:15,964 P6713 INFO Monitor(max) STOP: 0.814430 !
2022-12-18 12:43:15,964 P6713 INFO Reduce learning rate on plateau: 0.000010
2022-12-18 12:43:15,965 P6713 INFO --- 8058/8058 batches finished ---
2022-12-18 12:43:16,013 P6713 INFO Train loss @epoch 12: 0.432828
2022-12-18 12:43:16,013 P6713 INFO ************ Epoch=12 end ************
2022-12-18 12:52:05,939 P6713 INFO [Metrics] AUC: 0.813827
2022-12-18 12:52:05,940 P6713 INFO Monitor(max) STOP: 0.813827 !
2022-12-18 12:52:05,940 P6713 INFO Reduce learning rate on plateau: 0.000001
2022-12-18 12:52:05,940 P6713 INFO ********* Epoch==13 early stop *********
2022-12-18 12:52:05,942 P6713 INFO --- 8058/8058 batches finished ---
2022-12-18 12:52:05,996 P6713 INFO Train loss @epoch 13: 0.427918
2022-12-18 12:52:05,996 P6713 INFO Training finished.
2022-12-18 12:52:05,996 P6713 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/FinalMLP_criteo_x1/criteo_x1_49102577/FinalMLP_criteo_x1_004_d5d36917.model
2022-12-18 12:52:06,084 P6713 INFO ****** Validation evaluation ******
2022-12-18 12:52:43,628 P6713 INFO [Metrics] AUC: 0.814554 - logloss: 0.437523
2022-12-18 12:52:43,727 P6713 INFO ******** Test evaluation ********
2022-12-18 12:52:43,727 P6713 INFO Loading data...
2022-12-18 12:52:43,727 P6713 INFO Loading data from h5: ../data/Criteo/criteo_x1_49102577/test.h5
2022-12-18 12:52:47,098 P6713 INFO Test samples: total/4587167, blocks/1
2022-12-18 12:52:47,099 P6713 INFO Loading test data done.
2022-12-18 12:53:08,356 P6713 INFO [Metrics] AUC: 0.814902 - logloss: 0.437025

```
