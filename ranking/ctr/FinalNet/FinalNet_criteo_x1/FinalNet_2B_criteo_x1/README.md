## FINAL_2B_criteo_x1

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

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FINAL_2B_criteo_x1_tuner_config_01](./FINAL_2B_criteo_x1_tuner_config_01). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/FINAL
    nohup python run_expid.py --config XXX/benchmarks/FINAL/FINAL_2B_criteo_x1_tuner_config_01 --expid FINAL_criteo_x1_028_b6c861e4 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.815438 | 0.436520  |


### Logs
```python
2022-12-31 14:18:16,161 P55824 INFO Params: {
    "batch_size": "4096",
    "block1_dropout": "0.1",
    "block1_hidden_activations": "ReLU",
    "block1_hidden_units": "[400, 400]",
    "block2_dropout": "0",
    "block2_hidden_activations": "ReLU",
    "block2_hidden_units": "[400, 400]",
    "block_type": "2B",
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
    "gpu": "2",
    "group_id": "None",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FINAL",
    "model_id": "FINAL_criteo_x1_028_b6c861e4",
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
2022-12-31 14:18:16,162 P55824 INFO Load feature_map from json: ../data/Criteo/criteo_x1_49102577/feature_map.json
2022-12-31 14:18:16,162 P55824 INFO Set column index...
2022-12-31 14:18:16,162 P55824 INFO Feature specs: {
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
2022-12-31 14:18:19,719 P55824 INFO Total number of parameters: 21658582.
2022-12-31 14:18:19,719 P55824 INFO Loading data...
2022-12-31 14:18:19,719 P55824 INFO Loading data from h5: ../data/Criteo/criteo_x1_49102577/train.h5
2022-12-31 14:18:43,925 P55824 INFO Train samples: total/33003326, blocks/1
2022-12-31 14:18:43,925 P55824 INFO Loading data from h5: ../data/Criteo/criteo_x1_49102577/valid.h5
2022-12-31 14:18:49,931 P55824 INFO Validation samples: total/8250124, blocks/1
2022-12-31 14:18:49,931 P55824 INFO Loading train and validation data done.
2022-12-31 14:18:49,931 P55824 INFO Start training: 8058 batches/epoch
2022-12-31 14:18:49,931 P55824 INFO ************ Epoch=1 start ************
2022-12-31 14:24:58,689 P55824 INFO [Metrics] AUC: 0.804045
2022-12-31 14:24:58,691 P55824 INFO Save best model: monitor(max): 0.804045
2022-12-31 14:24:58,813 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 14:24:58,864 P55824 INFO Train loss @epoch 1: 1.365760
2022-12-31 14:24:58,864 P55824 INFO ************ Epoch=1 end ************
2022-12-31 14:31:08,437 P55824 INFO [Metrics] AUC: 0.806607
2022-12-31 14:31:08,438 P55824 INFO Save best model: monitor(max): 0.806607
2022-12-31 14:31:08,581 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 14:31:08,635 P55824 INFO Train loss @epoch 2: 1.346502
2022-12-31 14:31:08,635 P55824 INFO ************ Epoch=2 end ************
2022-12-31 14:37:12,783 P55824 INFO [Metrics] AUC: 0.807782
2022-12-31 14:37:12,784 P55824 INFO Save best model: monitor(max): 0.807782
2022-12-31 14:37:12,918 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 14:37:12,971 P55824 INFO Train loss @epoch 3: 1.341058
2022-12-31 14:37:12,971 P55824 INFO ************ Epoch=3 end ************
2022-12-31 14:43:14,978 P55824 INFO [Metrics] AUC: 0.808539
2022-12-31 14:43:14,980 P55824 INFO Save best model: monitor(max): 0.808539
2022-12-31 14:43:15,115 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 14:43:15,172 P55824 INFO Train loss @epoch 4: 1.338090
2022-12-31 14:43:15,172 P55824 INFO ************ Epoch=4 end ************
2022-12-31 14:49:16,540 P55824 INFO [Metrics] AUC: 0.809245
2022-12-31 14:49:16,542 P55824 INFO Save best model: monitor(max): 0.809245
2022-12-31 14:49:16,673 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 14:49:16,726 P55824 INFO Train loss @epoch 5: 1.336240
2022-12-31 14:49:16,726 P55824 INFO ************ Epoch=5 end ************
2022-12-31 14:55:16,601 P55824 INFO [Metrics] AUC: 0.809501
2022-12-31 14:55:16,603 P55824 INFO Save best model: monitor(max): 0.809501
2022-12-31 14:55:16,736 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 14:55:16,790 P55824 INFO Train loss @epoch 6: 1.334750
2022-12-31 14:55:16,790 P55824 INFO ************ Epoch=6 end ************
2022-12-31 15:01:16,587 P55824 INFO [Metrics] AUC: 0.809854
2022-12-31 15:01:16,589 P55824 INFO Save best model: monitor(max): 0.809854
2022-12-31 15:01:16,724 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 15:01:16,790 P55824 INFO Train loss @epoch 7: 1.333630
2022-12-31 15:01:16,790 P55824 INFO ************ Epoch=7 end ************
2022-12-31 15:07:19,506 P55824 INFO [Metrics] AUC: 0.810019
2022-12-31 15:07:19,508 P55824 INFO Save best model: monitor(max): 0.810019
2022-12-31 15:07:19,648 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 15:07:19,713 P55824 INFO Train loss @epoch 8: 1.332669
2022-12-31 15:07:19,713 P55824 INFO ************ Epoch=8 end ************
2022-12-31 15:13:22,185 P55824 INFO [Metrics] AUC: 0.810288
2022-12-31 15:13:22,187 P55824 INFO Save best model: monitor(max): 0.810288
2022-12-31 15:13:22,323 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 15:13:22,376 P55824 INFO Train loss @epoch 9: 1.331825
2022-12-31 15:13:22,376 P55824 INFO ************ Epoch=9 end ************
2022-12-31 15:19:25,899 P55824 INFO [Metrics] AUC: 0.810392
2022-12-31 15:19:25,900 P55824 INFO Save best model: monitor(max): 0.810392
2022-12-31 15:19:26,039 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 15:19:26,091 P55824 INFO Train loss @epoch 10: 1.331006
2022-12-31 15:19:26,091 P55824 INFO ************ Epoch=10 end ************
2022-12-31 15:25:31,489 P55824 INFO [Metrics] AUC: 0.810647
2022-12-31 15:25:31,490 P55824 INFO Save best model: monitor(max): 0.810647
2022-12-31 15:25:31,627 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 15:25:31,680 P55824 INFO Train loss @epoch 11: 1.330436
2022-12-31 15:25:31,680 P55824 INFO ************ Epoch=11 end ************
2022-12-31 15:31:43,924 P55824 INFO [Metrics] AUC: 0.810714
2022-12-31 15:31:43,926 P55824 INFO Save best model: monitor(max): 0.810714
2022-12-31 15:31:44,074 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 15:31:44,145 P55824 INFO Train loss @epoch 12: 1.329828
2022-12-31 15:31:44,145 P55824 INFO ************ Epoch=12 end ************
2022-12-31 15:38:00,469 P55824 INFO [Metrics] AUC: 0.810785
2022-12-31 15:38:00,471 P55824 INFO Save best model: monitor(max): 0.810785
2022-12-31 15:38:00,623 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 15:38:00,692 P55824 INFO Train loss @epoch 13: 1.329275
2022-12-31 15:38:00,693 P55824 INFO ************ Epoch=13 end ************
2022-12-31 15:44:12,313 P55824 INFO [Metrics] AUC: 0.810808
2022-12-31 15:44:12,315 P55824 INFO Save best model: monitor(max): 0.810808
2022-12-31 15:44:12,450 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 15:44:12,522 P55824 INFO Train loss @epoch 14: 1.328723
2022-12-31 15:44:12,522 P55824 INFO ************ Epoch=14 end ************
2022-12-31 15:50:28,078 P55824 INFO [Metrics] AUC: 0.811010
2022-12-31 15:50:28,080 P55824 INFO Save best model: monitor(max): 0.811010
2022-12-31 15:50:28,216 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 15:50:28,283 P55824 INFO Train loss @epoch 15: 1.328237
2022-12-31 15:50:28,283 P55824 INFO ************ Epoch=15 end ************
2022-12-31 15:56:41,494 P55824 INFO [Metrics] AUC: 0.811013
2022-12-31 15:56:41,496 P55824 INFO Save best model: monitor(max): 0.811013
2022-12-31 15:56:41,632 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 15:56:41,698 P55824 INFO Train loss @epoch 16: 1.327812
2022-12-31 15:56:41,698 P55824 INFO ************ Epoch=16 end ************
2022-12-31 16:02:56,864 P55824 INFO [Metrics] AUC: 0.811077
2022-12-31 16:02:56,866 P55824 INFO Save best model: monitor(max): 0.811077
2022-12-31 16:02:57,008 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 16:02:57,072 P55824 INFO Train loss @epoch 17: 1.327422
2022-12-31 16:02:57,072 P55824 INFO ************ Epoch=17 end ************
2022-12-31 16:09:08,701 P55824 INFO [Metrics] AUC: 0.811132
2022-12-31 16:09:08,703 P55824 INFO Save best model: monitor(max): 0.811132
2022-12-31 16:09:08,838 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 16:09:08,897 P55824 INFO Train loss @epoch 18: 1.327053
2022-12-31 16:09:08,898 P55824 INFO ************ Epoch=18 end ************
2022-12-31 16:15:19,422 P55824 INFO [Metrics] AUC: 0.811207
2022-12-31 16:15:19,424 P55824 INFO Save best model: monitor(max): 0.811207
2022-12-31 16:15:19,561 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 16:15:19,618 P55824 INFO Train loss @epoch 19: 1.326685
2022-12-31 16:15:19,618 P55824 INFO ************ Epoch=19 end ************
2022-12-31 16:21:29,268 P55824 INFO [Metrics] AUC: 0.811334
2022-12-31 16:21:29,269 P55824 INFO Save best model: monitor(max): 0.811334
2022-12-31 16:21:29,406 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 16:21:29,476 P55824 INFO Train loss @epoch 20: 1.326408
2022-12-31 16:21:29,476 P55824 INFO ************ Epoch=20 end ************
2022-12-31 16:27:41,055 P55824 INFO [Metrics] AUC: 0.811377
2022-12-31 16:27:41,057 P55824 INFO Save best model: monitor(max): 0.811377
2022-12-31 16:27:41,194 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 16:27:41,263 P55824 INFO Train loss @epoch 21: 1.326081
2022-12-31 16:27:41,263 P55824 INFO ************ Epoch=21 end ************
2022-12-31 16:33:49,083 P55824 INFO [Metrics] AUC: 0.811383
2022-12-31 16:33:49,084 P55824 INFO Save best model: monitor(max): 0.811383
2022-12-31 16:33:49,218 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 16:33:49,275 P55824 INFO Train loss @epoch 22: 1.325760
2022-12-31 16:33:49,276 P55824 INFO ************ Epoch=22 end ************
2022-12-31 16:39:55,077 P55824 INFO [Metrics] AUC: 0.811385
2022-12-31 16:39:55,079 P55824 INFO Save best model: monitor(max): 0.811385
2022-12-31 16:39:55,209 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 16:39:55,268 P55824 INFO Train loss @epoch 23: 1.325518
2022-12-31 16:39:55,268 P55824 INFO ************ Epoch=23 end ************
2022-12-31 16:46:01,654 P55824 INFO [Metrics] AUC: 0.811512
2022-12-31 16:46:01,656 P55824 INFO Save best model: monitor(max): 0.811512
2022-12-31 16:46:01,788 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 16:46:01,853 P55824 INFO Train loss @epoch 24: 1.325232
2022-12-31 16:46:01,854 P55824 INFO ************ Epoch=24 end ************
2022-12-31 16:52:07,914 P55824 INFO [Metrics] AUC: 0.811570
2022-12-31 16:52:07,916 P55824 INFO Save best model: monitor(max): 0.811570
2022-12-31 16:52:08,061 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 16:52:08,129 P55824 INFO Train loss @epoch 25: 1.324979
2022-12-31 16:52:08,130 P55824 INFO ************ Epoch=25 end ************
2022-12-31 16:58:08,917 P55824 INFO [Metrics] AUC: 0.811509
2022-12-31 16:58:08,919 P55824 INFO Monitor(max) STOP: 0.811509 !
2022-12-31 16:58:08,919 P55824 INFO Reduce learning rate on plateau: 0.000100
2022-12-31 16:58:08,920 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 16:58:08,985 P55824 INFO Train loss @epoch 26: 1.324730
2022-12-31 16:58:08,985 P55824 INFO ************ Epoch=26 end ************
2022-12-31 17:04:13,214 P55824 INFO [Metrics] AUC: 0.814493
2022-12-31 17:04:13,216 P55824 INFO Save best model: monitor(max): 0.814493
2022-12-31 17:04:13,344 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 17:04:13,405 P55824 INFO Train loss @epoch 27: 1.303181
2022-12-31 17:04:13,405 P55824 INFO ************ Epoch=27 end ************
2022-12-31 17:10:17,812 P55824 INFO [Metrics] AUC: 0.815003
2022-12-31 17:10:17,814 P55824 INFO Save best model: monitor(max): 0.815003
2022-12-31 17:10:17,945 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 17:10:18,004 P55824 INFO Train loss @epoch 28: 1.294512
2022-12-31 17:10:18,004 P55824 INFO ************ Epoch=28 end ************
2022-12-31 17:16:19,987 P55824 INFO [Metrics] AUC: 0.815126
2022-12-31 17:16:19,989 P55824 INFO Save best model: monitor(max): 0.815126
2022-12-31 17:16:20,122 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 17:16:20,181 P55824 INFO Train loss @epoch 29: 1.290084
2022-12-31 17:16:20,181 P55824 INFO ************ Epoch=29 end ************
2022-12-31 17:22:25,444 P55824 INFO [Metrics] AUC: 0.815087
2022-12-31 17:22:25,446 P55824 INFO Monitor(max) STOP: 0.815087 !
2022-12-31 17:22:25,446 P55824 INFO Reduce learning rate on plateau: 0.000010
2022-12-31 17:22:25,447 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 17:22:25,504 P55824 INFO Train loss @epoch 30: 1.286551
2022-12-31 17:22:25,504 P55824 INFO ************ Epoch=30 end ************
2022-12-31 17:28:30,446 P55824 INFO [Metrics] AUC: 0.814877
2022-12-31 17:28:30,448 P55824 INFO Monitor(max) STOP: 0.814877 !
2022-12-31 17:28:30,449 P55824 INFO Reduce learning rate on plateau: 0.000001
2022-12-31 17:28:30,449 P55824 INFO ********* Epoch==31 early stop *********
2022-12-31 17:28:30,450 P55824 INFO --- 8058/8058 batches finished ---
2022-12-31 17:28:30,517 P55824 INFO Train loss @epoch 31: 1.276923
2022-12-31 17:28:30,517 P55824 INFO Training finished.
2022-12-31 17:28:30,517 P55824 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/FINAL_criteo_x1/criteo_x1_49102577/FINAL_criteo_x1_028_b6c861e4.model
2022-12-31 17:28:30,594 P55824 INFO ****** Validation evaluation ******
2022-12-31 17:28:56,827 P55824 INFO [Metrics] AUC: 0.815126 - logloss: 0.436976
2022-12-31 17:28:56,948 P55824 INFO ******** Test evaluation ********
2022-12-31 17:28:56,948 P55824 INFO Loading data...
2022-12-31 17:28:56,948 P55824 INFO Loading data from h5: ../data/Criteo/criteo_x1_49102577/test.h5
2022-12-31 17:29:00,621 P55824 INFO Test samples: total/4587167, blocks/1
2022-12-31 17:29:00,621 P55824 INFO Loading test data done.
2022-12-31 17:29:15,923 P55824 INFO [Metrics] AUC: 0.815438 - logloss: 0.436520

```
