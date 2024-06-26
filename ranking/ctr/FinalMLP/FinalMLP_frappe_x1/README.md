## FinalMLP_frappe_x1

A hands-on guide to run the FinalMLP model on the Frappe_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)


| [Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs) |
|:-----------------------------:|:-----------:|:--------:|:--------:|-------|
### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
  GPU: Tesla P100 16G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 10.0
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  fuxictr: 2.0.1
  ```

### Dataset
Please refer to the BARS dataset [Frappe_x1](https://github.com/openbenchmark/BARS/blob/main/datasets/Frappe#Frappe_x1) to get data ready.

### Code

We use the [FinalMLP](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/FinalMLP) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FinalMLP_frappe_x1_tuner_config_16](./FinalMLP_frappe_x1_tuner_config_16). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/FinalMLP
    nohup python run_expid.py --config XXX/benchmarks/FinalMLP/FinalMLP_frappe_x1_tuner_config_16 --expid FinalMLP_frappe_x1_004_e1ab402f --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.986072 | 0.148415  |


### Logs
```python
2022-12-17 14:26:33,332 P29088 INFO Params: {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_47e6e0df",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "10",
    "embedding_regularizer": "0.05",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "feature_specs": "None",
    "fs1_context": "['user']",
    "fs2_context": "['item']",
    "fs_hidden_units": "[400]",
    "gpu": "1",
    "group_id": "None",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "mlp1_batch_norm": "True",
    "mlp1_dropout": "0.4",
    "mlp1_hidden_activations": "relu",
    "mlp1_hidden_units": "[400]",
    "mlp2_batch_norm": "True",
    "mlp2_dropout": "0.4",
    "mlp2_hidden_activations": "relu",
    "mlp2_hidden_units": "[100]",
    "model": "FinalMLP",
    "model_id": "FinalMLP_frappe_x1_004_e1ab402f",
    "model_root": "./checkpoints/FinalMLP_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "num_heads": "5",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Frappe/Frappe_x1/test.csv",
    "train_data": "../data/Frappe/Frappe_x1/train.csv",
    "use_fs": "True",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1"
}
2022-12-17 14:26:33,333 P29088 INFO Load feature_map from json: ../data/Frappe/frappe_x1_47e6e0df/feature_map.json
2022-12-17 14:26:33,333 P29088 INFO Set column index...
2022-12-17 14:26:33,334 P29088 INFO Feature specs: {
    "city": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 234, 'vocab_size': 235}",
    "cost": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 3, 'vocab_size': 4}",
    "country": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 81, 'vocab_size': 82}",
    "daytime": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 8, 'vocab_size': 9}",
    "homework": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 4, 'vocab_size': 5}",
    "isweekend": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 3, 'vocab_size': 4}",
    "item": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 4083, 'vocab_size': 4084}",
    "user": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 955, 'vocab_size': 956}",
    "weather": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10, 'vocab_size': 11}",
    "weekday": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 8, 'vocab_size': 9}"
}
2022-12-17 14:26:38,263 P29088 INFO Total number of parameters: 253392.
2022-12-17 14:26:38,264 P29088 INFO Loading data...
2022-12-17 14:26:38,264 P29088 INFO Loading data from h5: ../data/Frappe/frappe_x1_47e6e0df/train.h5
2022-12-17 14:26:38,296 P29088 INFO Train samples: total/202027, blocks/1
2022-12-17 14:26:38,296 P29088 INFO Loading data from h5: ../data/Frappe/frappe_x1_47e6e0df/valid.h5
2022-12-17 14:26:38,304 P29088 INFO Validation samples: total/57722, blocks/1
2022-12-17 14:26:38,304 P29088 INFO Loading train and validation data done.
2022-12-17 14:26:38,305 P29088 INFO Start training: 50 batches/epoch
2022-12-17 14:26:38,305 P29088 INFO ************ Epoch=1 start ************
2022-12-17 14:26:52,754 P29088 INFO [Metrics] AUC: 0.940814
2022-12-17 14:26:52,755 P29088 INFO Save best model: monitor(max): 0.940814
2022-12-17 14:26:52,761 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:26:52,794 P29088 INFO Train loss @epoch 1: 0.375609
2022-12-17 14:26:52,794 P29088 INFO ************ Epoch=1 end ************
2022-12-17 14:27:07,173 P29088 INFO [Metrics] AUC: 0.962458
2022-12-17 14:27:07,174 P29088 INFO Save best model: monitor(max): 0.962458
2022-12-17 14:27:07,178 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:27:07,215 P29088 INFO Train loss @epoch 2: 0.277278
2022-12-17 14:27:07,215 P29088 INFO ************ Epoch=2 end ************
2022-12-17 14:27:19,594 P29088 INFO [Metrics] AUC: 0.971601
2022-12-17 14:27:19,595 P29088 INFO Save best model: monitor(max): 0.971601
2022-12-17 14:27:19,599 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:27:19,632 P29088 INFO Train loss @epoch 3: 0.239565
2022-12-17 14:27:19,632 P29088 INFO ************ Epoch=3 end ************
2022-12-17 14:27:34,350 P29088 INFO [Metrics] AUC: 0.974917
2022-12-17 14:27:34,350 P29088 INFO Save best model: monitor(max): 0.974917
2022-12-17 14:27:34,355 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:27:34,388 P29088 INFO Train loss @epoch 4: 0.220704
2022-12-17 14:27:34,388 P29088 INFO ************ Epoch=4 end ************
2022-12-17 14:27:48,103 P29088 INFO [Metrics] AUC: 0.978341
2022-12-17 14:27:48,104 P29088 INFO Save best model: monitor(max): 0.978341
2022-12-17 14:27:48,108 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:27:48,145 P29088 INFO Train loss @epoch 5: 0.207423
2022-12-17 14:27:48,145 P29088 INFO ************ Epoch=5 end ************
2022-12-17 14:28:01,925 P29088 INFO [Metrics] AUC: 0.978471
2022-12-17 14:28:01,926 P29088 INFO Save best model: monitor(max): 0.978471
2022-12-17 14:28:01,930 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:28:01,961 P29088 INFO Train loss @epoch 6: 0.199262
2022-12-17 14:28:01,961 P29088 INFO ************ Epoch=6 end ************
2022-12-17 14:28:16,471 P29088 INFO [Metrics] AUC: 0.980229
2022-12-17 14:28:16,472 P29088 INFO Save best model: monitor(max): 0.980229
2022-12-17 14:28:16,476 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:28:16,511 P29088 INFO Train loss @epoch 7: 0.192794
2022-12-17 14:28:16,511 P29088 INFO ************ Epoch=7 end ************
2022-12-17 14:28:30,222 P29088 INFO [Metrics] AUC: 0.980285
2022-12-17 14:28:30,222 P29088 INFO Save best model: monitor(max): 0.980285
2022-12-17 14:28:30,226 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:28:30,262 P29088 INFO Train loss @epoch 8: 0.185644
2022-12-17 14:28:30,262 P29088 INFO ************ Epoch=8 end ************
2022-12-17 14:28:44,046 P29088 INFO [Metrics] AUC: 0.981008
2022-12-17 14:28:44,046 P29088 INFO Save best model: monitor(max): 0.981008
2022-12-17 14:28:44,051 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:28:44,088 P29088 INFO Train loss @epoch 9: 0.182929
2022-12-17 14:28:44,088 P29088 INFO ************ Epoch=9 end ************
2022-12-17 14:28:59,336 P29088 INFO [Metrics] AUC: 0.981085
2022-12-17 14:28:59,336 P29088 INFO Save best model: monitor(max): 0.981085
2022-12-17 14:28:59,340 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:28:59,377 P29088 INFO Train loss @epoch 10: 0.179877
2022-12-17 14:28:59,378 P29088 INFO ************ Epoch=10 end ************
2022-12-17 14:29:13,062 P29088 INFO [Metrics] AUC: 0.981973
2022-12-17 14:29:13,062 P29088 INFO Save best model: monitor(max): 0.981973
2022-12-17 14:29:13,067 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:29:13,099 P29088 INFO Train loss @epoch 11: 0.177397
2022-12-17 14:29:13,100 P29088 INFO ************ Epoch=11 end ************
2022-12-17 14:29:26,688 P29088 INFO [Metrics] AUC: 0.982296
2022-12-17 14:29:26,689 P29088 INFO Save best model: monitor(max): 0.982296
2022-12-17 14:29:26,693 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:29:26,734 P29088 INFO Train loss @epoch 12: 0.175585
2022-12-17 14:29:26,734 P29088 INFO ************ Epoch=12 end ************
2022-12-17 14:29:41,147 P29088 INFO [Metrics] AUC: 0.982388
2022-12-17 14:29:41,148 P29088 INFO Save best model: monitor(max): 0.982388
2022-12-17 14:29:41,152 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:29:41,184 P29088 INFO Train loss @epoch 13: 0.171989
2022-12-17 14:29:41,184 P29088 INFO ************ Epoch=13 end ************
2022-12-17 14:29:54,784 P29088 INFO [Metrics] AUC: 0.982856
2022-12-17 14:29:54,784 P29088 INFO Save best model: monitor(max): 0.982856
2022-12-17 14:29:54,789 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:29:54,824 P29088 INFO Train loss @epoch 14: 0.171455
2022-12-17 14:29:54,824 P29088 INFO ************ Epoch=14 end ************
2022-12-17 14:30:08,693 P29088 INFO [Metrics] AUC: 0.982275
2022-12-17 14:30:08,693 P29088 INFO Monitor(max) STOP: 0.982275 !
2022-12-17 14:30:08,693 P29088 INFO Reduce learning rate on plateau: 0.000100
2022-12-17 14:30:08,693 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:30:08,732 P29088 INFO Train loss @epoch 15: 0.169575
2022-12-17 14:30:08,733 P29088 INFO ************ Epoch=15 end ************
2022-12-17 14:30:23,210 P29088 INFO [Metrics] AUC: 0.984281
2022-12-17 14:30:23,210 P29088 INFO Save best model: monitor(max): 0.984281
2022-12-17 14:30:23,214 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:30:23,252 P29088 INFO Train loss @epoch 16: 0.141549
2022-12-17 14:30:23,252 P29088 INFO ************ Epoch=16 end ************
2022-12-17 14:30:37,641 P29088 INFO [Metrics] AUC: 0.985418
2022-12-17 14:30:37,641 P29088 INFO Save best model: monitor(max): 0.985418
2022-12-17 14:30:37,645 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:30:37,681 P29088 INFO Train loss @epoch 17: 0.121648
2022-12-17 14:30:37,681 P29088 INFO ************ Epoch=17 end ************
2022-12-17 14:30:51,300 P29088 INFO [Metrics] AUC: 0.985996
2022-12-17 14:30:51,301 P29088 INFO Save best model: monitor(max): 0.985996
2022-12-17 14:30:51,305 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:30:51,353 P29088 INFO Train loss @epoch 18: 0.109507
2022-12-17 14:30:51,353 P29088 INFO ************ Epoch=18 end ************
2022-12-17 14:31:05,572 P29088 INFO [Metrics] AUC: 0.986277
2022-12-17 14:31:05,572 P29088 INFO Save best model: monitor(max): 0.986277
2022-12-17 14:31:05,576 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:31:05,612 P29088 INFO Train loss @epoch 19: 0.100590
2022-12-17 14:31:05,612 P29088 INFO ************ Epoch=19 end ************
2022-12-17 14:31:19,169 P29088 INFO [Metrics] AUC: 0.986426
2022-12-17 14:31:19,170 P29088 INFO Save best model: monitor(max): 0.986426
2022-12-17 14:31:19,174 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:31:19,210 P29088 INFO Train loss @epoch 20: 0.094059
2022-12-17 14:31:19,210 P29088 INFO ************ Epoch=20 end ************
2022-12-17 14:31:32,739 P29088 INFO [Metrics] AUC: 0.986507
2022-12-17 14:31:32,739 P29088 INFO Save best model: monitor(max): 0.986507
2022-12-17 14:31:32,743 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:31:32,802 P29088 INFO Train loss @epoch 21: 0.088697
2022-12-17 14:31:32,802 P29088 INFO ************ Epoch=21 end ************
2022-12-17 14:31:46,923 P29088 INFO [Metrics] AUC: 0.986510
2022-12-17 14:31:46,923 P29088 INFO Save best model: monitor(max): 0.986510
2022-12-17 14:31:46,927 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:31:46,962 P29088 INFO Train loss @epoch 22: 0.084178
2022-12-17 14:31:46,962 P29088 INFO ************ Epoch=22 end ************
2022-12-17 14:32:00,926 P29088 INFO [Metrics] AUC: 0.986507
2022-12-17 14:32:00,926 P29088 INFO Monitor(max) STOP: 0.986507 !
2022-12-17 14:32:00,926 P29088 INFO Reduce learning rate on plateau: 0.000010
2022-12-17 14:32:00,926 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:32:00,966 P29088 INFO Train loss @epoch 23: 0.080563
2022-12-17 14:32:00,967 P29088 INFO ************ Epoch=23 end ************
2022-12-17 14:32:14,727 P29088 INFO [Metrics] AUC: 0.986548
2022-12-17 14:32:14,727 P29088 INFO Save best model: monitor(max): 0.986548
2022-12-17 14:32:14,731 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:32:14,766 P29088 INFO Train loss @epoch 24: 0.075752
2022-12-17 14:32:14,766 P29088 INFO ************ Epoch=24 end ************
2022-12-17 14:32:29,173 P29088 INFO [Metrics] AUC: 0.986554
2022-12-17 14:32:29,173 P29088 INFO Save best model: monitor(max): 0.986554
2022-12-17 14:32:29,177 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:32:29,209 P29088 INFO Train loss @epoch 25: 0.075708
2022-12-17 14:32:29,209 P29088 INFO ************ Epoch=25 end ************
2022-12-17 14:32:42,732 P29088 INFO [Metrics] AUC: 0.986577
2022-12-17 14:32:42,732 P29088 INFO Save best model: monitor(max): 0.986577
2022-12-17 14:32:42,737 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:32:42,778 P29088 INFO Train loss @epoch 26: 0.075111
2022-12-17 14:32:42,778 P29088 INFO ************ Epoch=26 end ************
2022-12-17 14:32:56,599 P29088 INFO [Metrics] AUC: 0.986582
2022-12-17 14:32:56,599 P29088 INFO Save best model: monitor(max): 0.986582
2022-12-17 14:32:56,604 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:32:56,636 P29088 INFO Train loss @epoch 27: 0.074745
2022-12-17 14:32:56,636 P29088 INFO ************ Epoch=27 end ************
2022-12-17 14:33:10,914 P29088 INFO [Metrics] AUC: 0.986614
2022-12-17 14:33:10,914 P29088 INFO Save best model: monitor(max): 0.986614
2022-12-17 14:33:10,918 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:33:10,958 P29088 INFO Train loss @epoch 28: 0.073972
2022-12-17 14:33:10,958 P29088 INFO ************ Epoch=28 end ************
2022-12-17 14:33:24,745 P29088 INFO [Metrics] AUC: 0.986630
2022-12-17 14:33:24,746 P29088 INFO Save best model: monitor(max): 0.986630
2022-12-17 14:33:24,750 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:33:24,784 P29088 INFO Train loss @epoch 29: 0.073906
2022-12-17 14:33:24,784 P29088 INFO ************ Epoch=29 end ************
2022-12-17 14:33:38,311 P29088 INFO [Metrics] AUC: 0.986631
2022-12-17 14:33:38,312 P29088 INFO Save best model: monitor(max): 0.986631
2022-12-17 14:33:38,316 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:33:38,360 P29088 INFO Train loss @epoch 30: 0.073600
2022-12-17 14:33:38,360 P29088 INFO ************ Epoch=30 end ************
2022-12-17 14:33:53,223 P29088 INFO [Metrics] AUC: 0.986629
2022-12-17 14:33:53,224 P29088 INFO Monitor(max) STOP: 0.986629 !
2022-12-17 14:33:53,224 P29088 INFO Reduce learning rate on plateau: 0.000001
2022-12-17 14:33:53,224 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:33:53,256 P29088 INFO Train loss @epoch 31: 0.072644
2022-12-17 14:33:53,256 P29088 INFO ************ Epoch=31 end ************
2022-12-17 14:34:07,106 P29088 INFO [Metrics] AUC: 0.986622
2022-12-17 14:34:07,106 P29088 INFO Monitor(max) STOP: 0.986622 !
2022-12-17 14:34:07,106 P29088 INFO Reduce learning rate on plateau: 0.000001
2022-12-17 14:34:07,106 P29088 INFO ********* Epoch==32 early stop *********
2022-12-17 14:34:07,107 P29088 INFO --- 50/50 batches finished ---
2022-12-17 14:34:07,142 P29088 INFO Train loss @epoch 32: 0.071815
2022-12-17 14:34:07,142 P29088 INFO Training finished.
2022-12-17 14:34:07,142 P29088 INFO Load best model: /home/FuxiCTRv2/benchmark/checkpoints/FinalMLP_frappe_x1/frappe_x1_47e6e0df/FinalMLP_frappe_x1_004_e1ab402f.model
2022-12-17 14:34:07,179 P29088 INFO ****** Validation evaluation ******
2022-12-17 14:34:08,077 P29088 INFO [Metrics] AUC: 0.986631 - logloss: 0.144852
2022-12-17 14:34:08,115 P29088 INFO ******** Test evaluation ********
2022-12-17 14:34:08,115 P29088 INFO Loading data...
2022-12-17 14:34:08,115 P29088 INFO Loading data from h5: ../data/Frappe/frappe_x1_47e6e0df/test.h5
2022-12-17 14:34:08,122 P29088 INFO Test samples: total/28860, blocks/1
2022-12-17 14:34:08,122 P29088 INFO Loading test data done.
2022-12-17 14:34:08,619 P29088 INFO [Metrics] AUC: 0.986072 - logloss: 0.148415

```
