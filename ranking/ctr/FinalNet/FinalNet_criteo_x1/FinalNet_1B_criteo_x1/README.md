## FINAL_1B_criteo_x1

A hands-on guide to run the FINAL model on the Criteo_x1 dataset.

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
Please refer to the BARS dataset [Criteo_x1](https://github.com/openbenchmark/BARS/blob/main/datasets/Criteo#Criteo_x1) to get data ready.

### Code

We use the [FINAL](https://github.com/reczoo/FuxiCTR/blob/v2.0.2/model_zoo/FINAL) model code from [FuxiCTR-v2.0.2](https://github.com/reczoo/FuxiCTR/tree/v2.0.2) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.2.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.2
    ```

2. Create a data directory and put the downloaded data files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FINAL_1B_criteo_x1_tuner_config_04](./FINAL_1B_criteo_x1_tuner_config_04). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/FINAL
    nohup python run_expid.py --config XXX/benchmarks/FINAL/FINAL_1B_criteo_x1_tuner_config_04 --expid FINAL_criteo_x1_007_b4783421 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.814403 | 0.437479  |


### Logs
```python
2022-12-31 23:02:46,357 P57556 INFO Params: {
    "batch_size": "4096",
    "block1_dropout": "0.2",
    "block1_hidden_activations": "ReLU",
    "block1_hidden_units": "[1000, 1000]",
    "block2_dropout": "0",
    "block2_hidden_activations": "None",
    "block2_hidden_units": "[64, 64, 64]",
    "block_type": "1B",
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
    "gpu": "6",
    "group_id": "None",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FINAL",
    "model_id": "FINAL_criteo_x1_007_b4783421",
    "model_root": "./checkpoints/FINAL_criteo_x1/",
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
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_field_gate": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "1"
}
2022-12-31 23:02:46,358 P57556 INFO Load feature_map from json: ../data/Criteo/criteo_x1_49102577/feature_map.json
2022-12-31 23:02:46,358 P57556 INFO Set column index...
2022-12-31 23:02:46,359 P57556 INFO Feature specs: {
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
2022-12-31 23:02:51,714 P57556 INFO Total number of parameters: 22651981.
2022-12-31 23:02:51,714 P57556 INFO Loading data...
2022-12-31 23:02:51,714 P57556 INFO Loading data from h5: ../data/Criteo/criteo_x1_49102577/train.h5
2022-12-31 23:03:19,334 P57556 INFO Train samples: total/33003326, blocks/1
2022-12-31 23:03:19,334 P57556 INFO Loading data from h5: ../data/Criteo/criteo_x1_49102577/valid.h5
2022-12-31 23:03:25,240 P57556 INFO Validation samples: total/8250124, blocks/1
2022-12-31 23:03:25,240 P57556 INFO Loading train and validation data done.
2022-12-31 23:03:25,240 P57556 INFO Start training: 8058 batches/epoch
2022-12-31 23:03:25,240 P57556 INFO ************ Epoch=1 start ************
2022-12-31 23:09:34,624 P57556 INFO [Metrics] AUC: 0.804092
2022-12-31 23:09:34,625 P57556 INFO Save best model: monitor(max): 0.804092
2022-12-31 23:09:34,759 P57556 INFO --- 8058/8058 batches finished ---
2022-12-31 23:09:34,810 P57556 INFO Train loss @epoch 1: 0.462683
2022-12-31 23:09:34,810 P57556 INFO ************ Epoch=1 end ************
2022-12-31 23:15:42,833 P57556 INFO [Metrics] AUC: 0.806820
2022-12-31 23:15:42,835 P57556 INFO Save best model: monitor(max): 0.806820
2022-12-31 23:15:42,978 P57556 INFO --- 8058/8058 batches finished ---
2022-12-31 23:15:43,029 P57556 INFO Train loss @epoch 2: 0.455929
2022-12-31 23:15:43,029 P57556 INFO ************ Epoch=2 end ************
2022-12-31 23:21:52,579 P57556 INFO [Metrics] AUC: 0.808197
2022-12-31 23:21:52,580 P57556 INFO Save best model: monitor(max): 0.808197
2022-12-31 23:21:52,726 P57556 INFO --- 8058/8058 batches finished ---
2022-12-31 23:21:52,802 P57556 INFO Train loss @epoch 3: 0.454131
2022-12-31 23:21:52,802 P57556 INFO ************ Epoch=3 end ************
2022-12-31 23:28:00,094 P57556 INFO [Metrics] AUC: 0.808861
2022-12-31 23:28:00,096 P57556 INFO Save best model: monitor(max): 0.808861
2022-12-31 23:28:00,246 P57556 INFO --- 8058/8058 batches finished ---
2022-12-31 23:28:00,296 P57556 INFO Train loss @epoch 4: 0.453171
2022-12-31 23:28:00,296 P57556 INFO ************ Epoch=4 end ************
2022-12-31 23:34:09,392 P57556 INFO [Metrics] AUC: 0.809494
2022-12-31 23:34:09,394 P57556 INFO Save best model: monitor(max): 0.809494
2022-12-31 23:34:09,539 P57556 INFO --- 8058/8058 batches finished ---
2022-12-31 23:34:09,590 P57556 INFO Train loss @epoch 5: 0.452536
2022-12-31 23:34:09,590 P57556 INFO ************ Epoch=5 end ************
2022-12-31 23:40:18,902 P57556 INFO [Metrics] AUC: 0.809739
2022-12-31 23:40:18,903 P57556 INFO Save best model: monitor(max): 0.809739
2022-12-31 23:40:19,043 P57556 INFO --- 8058/8058 batches finished ---
2022-12-31 23:40:19,092 P57556 INFO Train loss @epoch 6: 0.452069
2022-12-31 23:40:19,092 P57556 INFO ************ Epoch=6 end ************
2022-12-31 23:46:29,246 P57556 INFO [Metrics] AUC: 0.810117
2022-12-31 23:46:29,247 P57556 INFO Save best model: monitor(max): 0.810117
2022-12-31 23:46:29,405 P57556 INFO --- 8058/8058 batches finished ---
2022-12-31 23:46:29,456 P57556 INFO Train loss @epoch 7: 0.451698
2022-12-31 23:46:29,457 P57556 INFO ************ Epoch=7 end ************
2022-12-31 23:52:43,033 P57556 INFO [Metrics] AUC: 0.810134
2022-12-31 23:52:43,035 P57556 INFO Save best model: monitor(max): 0.810134
2022-12-31 23:52:43,194 P57556 INFO --- 8058/8058 batches finished ---
2022-12-31 23:52:43,245 P57556 INFO Train loss @epoch 8: 0.451348
2022-12-31 23:52:43,245 P57556 INFO ************ Epoch=8 end ************
2022-12-31 23:58:57,190 P57556 INFO [Metrics] AUC: 0.810418
2022-12-31 23:58:57,191 P57556 INFO Save best model: monitor(max): 0.810418
2022-12-31 23:58:57,351 P57556 INFO --- 8058/8058 batches finished ---
2022-12-31 23:58:57,410 P57556 INFO Train loss @epoch 9: 0.451102
2022-12-31 23:58:57,410 P57556 INFO ************ Epoch=9 end ************
2023-01-01 00:05:12,034 P57556 INFO [Metrics] AUC: 0.810630
2023-01-01 00:05:12,036 P57556 INFO Save best model: monitor(max): 0.810630
2023-01-01 00:05:12,179 P57556 INFO --- 8058/8058 batches finished ---
2023-01-01 00:05:12,252 P57556 INFO Train loss @epoch 10: 0.450840
2023-01-01 00:05:12,252 P57556 INFO ************ Epoch=10 end ************
2023-01-01 00:11:21,114 P57556 INFO [Metrics] AUC: 0.810702
2023-01-01 00:11:21,115 P57556 INFO Save best model: monitor(max): 0.810702
2023-01-01 00:11:21,257 P57556 INFO --- 8058/8058 batches finished ---
2023-01-01 00:11:21,334 P57556 INFO Train loss @epoch 11: 0.450579
2023-01-01 00:11:21,334 P57556 INFO ************ Epoch=11 end ************
2023-01-01 00:17:32,221 P57556 INFO [Metrics] AUC: 0.810674
2023-01-01 00:17:32,223 P57556 INFO Monitor(max) STOP: 0.810674 !
2023-01-01 00:17:32,223 P57556 INFO Reduce learning rate on plateau: 0.000100
2023-01-01 00:17:32,223 P57556 INFO --- 8058/8058 batches finished ---
2023-01-01 00:17:32,278 P57556 INFO Train loss @epoch 12: 0.450351
2023-01-01 00:17:32,278 P57556 INFO ************ Epoch=12 end ************
2023-01-01 00:23:37,268 P57556 INFO [Metrics] AUC: 0.813578
2023-01-01 00:23:37,269 P57556 INFO Save best model: monitor(max): 0.813578
2023-01-01 00:23:37,416 P57556 INFO --- 8058/8058 batches finished ---
2023-01-01 00:23:37,478 P57556 INFO Train loss @epoch 13: 0.439393
2023-01-01 00:23:37,478 P57556 INFO ************ Epoch=13 end ************
2023-01-01 00:29:42,833 P57556 INFO [Metrics] AUC: 0.813988
2023-01-01 00:29:42,835 P57556 INFO Save best model: monitor(max): 0.813988
2023-01-01 00:29:42,976 P57556 INFO --- 8058/8058 batches finished ---
2023-01-01 00:29:43,027 P57556 INFO Train loss @epoch 14: 0.435142
2023-01-01 00:29:43,027 P57556 INFO ************ Epoch=14 end ************
2023-01-01 00:35:49,216 P57556 INFO [Metrics] AUC: 0.814054
2023-01-01 00:35:49,218 P57556 INFO Save best model: monitor(max): 0.814054
2023-01-01 00:35:49,371 P57556 INFO --- 8058/8058 batches finished ---
2023-01-01 00:35:49,433 P57556 INFO Train loss @epoch 15: 0.433198
2023-01-01 00:35:49,434 P57556 INFO ************ Epoch=15 end ************
2023-01-01 00:41:53,186 P57556 INFO [Metrics] AUC: 0.813869
2023-01-01 00:41:53,187 P57556 INFO Monitor(max) STOP: 0.813869 !
2023-01-01 00:41:53,188 P57556 INFO Reduce learning rate on plateau: 0.000010
2023-01-01 00:41:53,188 P57556 INFO --- 8058/8058 batches finished ---
2023-01-01 00:41:53,241 P57556 INFO Train loss @epoch 16: 0.431728
2023-01-01 00:41:53,242 P57556 INFO ************ Epoch=16 end ************
2023-01-01 00:47:49,855 P57556 INFO [Metrics] AUC: 0.813425
2023-01-01 00:47:49,856 P57556 INFO Monitor(max) STOP: 0.813425 !
2023-01-01 00:47:49,857 P57556 INFO Reduce learning rate on plateau: 0.000001
2023-01-01 00:47:49,857 P57556 INFO ********* Epoch==17 early stop *********
2023-01-01 00:47:49,857 P57556 INFO --- 8058/8058 batches finished ---
2023-01-01 00:47:49,921 P57556 INFO Train loss @epoch 17: 0.427319
2023-01-01 00:47:49,921 P57556 INFO Training finished.
2023-01-01 00:47:49,921 P57556 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/FINAL_criteo_x1/criteo_x1_49102577/FINAL_criteo_x1_007_b4783421.model
2023-01-01 00:47:50,004 P57556 INFO ****** Validation evaluation ******
2023-01-01 00:48:17,824 P57556 INFO [Metrics] AUC: 0.814054 - logloss: 0.437986
2023-01-01 00:48:17,932 P57556 INFO ******** Test evaluation ********
2023-01-01 00:48:17,932 P57556 INFO Loading data...
2023-01-01 00:48:17,932 P57556 INFO Loading data from h5: ../data/Criteo/criteo_x1_49102577/test.h5
2023-01-01 00:48:21,345 P57556 INFO Test samples: total/4587167, blocks/1
2023-01-01 00:48:21,345 P57556 INFO Loading test data done.
2023-01-01 00:48:36,861 P57556 INFO [Metrics] AUC: 0.814403 - logloss: 0.437479

```
