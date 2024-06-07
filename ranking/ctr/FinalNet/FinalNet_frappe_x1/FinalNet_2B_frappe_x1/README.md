## FINAL_2B_frappe_x1

A hands-on guide to run the FINAL model on the Frappe_x1 dataset.

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
  fuxictr: 2.0.2
  ```

### Dataset
Please refer to the BARS dataset [Frappe_x1](https://github.com/openbenchmark/BARS/blob/main/datasets/Frappe#Frappe_x1) to get data ready.

### Code

We use the [FINAL](https://github.com/reczoo/FuxiCTR/blob/v2.0.2/model_zoo/FINAL) model code from [FuxiCTR-v2.0.2](https://github.com/reczoo/FuxiCTR/tree/v2.0.2) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.2.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.2
    ```

2. Create a data directory and put the downloaded data files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FINAL_2B_frappe_x1_tuner_config_02](./FINAL_2B_frappe_x1_tuner_config_02). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/FINAL
    nohup python run_expid.py --config XXX/benchmarks/FINAL/FINAL_2B_frappe_x1_tuner_config_02 --expid FINAL_frappe_x1_003_c3722b71 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.985744 | 0.131070  |


### Logs
```python
2023-01-06 22:45:43,999 P37903 INFO Params: {
    "batch_size": "4096",
    "block1_dropout": "0.1",
    "block1_hidden_activations": "ReLU",
    "block1_hidden_units": "[400, 400]",
    "block2_dropout": "0.1",
    "block2_hidden_activations": "ReLU",
    "block2_hidden_units": "[400]",
    "block_type": "2B",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_47e6e0df",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "10",
    "embedding_regularizer": "0.1",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "feature_specs": "None",
    "gpu": "0",
    "group_id": "None",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FINAL",
    "model_id": "FINAL_frappe_x1_003_c3722b71",
    "model_root": "./checkpoints/FINAL_frappe_x1/",
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
    "test_data": "../data/Frappe/Frappe_x1/test.csv",
    "train_data": "../data/Frappe/Frappe_x1/train.csv",
    "use_field_gate": "True",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1"
}
2023-01-06 22:45:44,001 P37903 INFO Load feature_map from json: ../data/Frappe/frappe_x1_47e6e0df/feature_map.json
2023-01-06 22:45:44,001 P37903 INFO Set column index...
2023-01-06 22:45:44,001 P37903 INFO Feature specs: {
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
2023-01-06 22:45:48,703 P37903 INFO Total number of parameters: 338502.
2023-01-06 22:45:48,704 P37903 INFO Loading data...
2023-01-06 22:45:48,704 P37903 INFO Loading data from h5: ../data/Frappe/frappe_x1_47e6e0df/train.h5
2023-01-06 22:45:48,743 P37903 INFO Train samples: total/202027, blocks/1
2023-01-06 22:45:48,743 P37903 INFO Loading data from h5: ../data/Frappe/frappe_x1_47e6e0df/valid.h5
2023-01-06 22:45:48,755 P37903 INFO Validation samples: total/57722, blocks/1
2023-01-06 22:45:48,755 P37903 INFO Loading train and validation data done.
2023-01-06 22:45:48,755 P37903 INFO Start training: 50 batches/epoch
2023-01-06 22:45:48,755 P37903 INFO ************ Epoch=1 start ************
2023-01-06 22:45:55,189 P37903 INFO [Metrics] AUC: 0.931314
2023-01-06 22:45:55,190 P37903 INFO Save best model: monitor(max): 0.931314
2023-01-06 22:45:55,197 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:45:55,264 P37903 INFO Train loss @epoch 1: 1.303795
2023-01-06 22:45:55,264 P37903 INFO ************ Epoch=1 end ************
2023-01-06 22:46:02,368 P37903 INFO [Metrics] AUC: 0.944991
2023-01-06 22:46:02,368 P37903 INFO Save best model: monitor(max): 0.944991
2023-01-06 22:46:02,375 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:46:02,455 P37903 INFO Train loss @epoch 2: 0.869754
2023-01-06 22:46:02,455 P37903 INFO ************ Epoch=2 end ************
2023-01-06 22:46:11,004 P37903 INFO [Metrics] AUC: 0.960037
2023-01-06 22:46:11,004 P37903 INFO Save best model: monitor(max): 0.960037
2023-01-06 22:46:11,011 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:46:11,071 P37903 INFO Train loss @epoch 3: 0.797433
2023-01-06 22:46:11,071 P37903 INFO ************ Epoch=3 end ************
2023-01-06 22:46:21,202 P37903 INFO [Metrics] AUC: 0.966038
2023-01-06 22:46:21,202 P37903 INFO Save best model: monitor(max): 0.966038
2023-01-06 22:46:21,210 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:46:21,272 P37903 INFO Train loss @epoch 4: 0.716508
2023-01-06 22:46:21,272 P37903 INFO ************ Epoch=4 end ************
2023-01-06 22:46:31,753 P37903 INFO [Metrics] AUC: 0.970877
2023-01-06 22:46:31,753 P37903 INFO Save best model: monitor(max): 0.970877
2023-01-06 22:46:31,761 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:46:31,887 P37903 INFO Train loss @epoch 5: 0.650016
2023-01-06 22:46:31,887 P37903 INFO ************ Epoch=5 end ************
2023-01-06 22:46:40,752 P37903 INFO [Metrics] AUC: 0.974044
2023-01-06 22:46:40,752 P37903 INFO Save best model: monitor(max): 0.974044
2023-01-06 22:46:40,762 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:46:40,829 P37903 INFO Train loss @epoch 6: 0.601324
2023-01-06 22:46:40,829 P37903 INFO ************ Epoch=6 end ************
2023-01-06 22:46:50,976 P37903 INFO [Metrics] AUC: 0.975111
2023-01-06 22:46:50,976 P37903 INFO Save best model: monitor(max): 0.975111
2023-01-06 22:46:50,983 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:46:51,044 P37903 INFO Train loss @epoch 7: 0.569696
2023-01-06 22:46:51,044 P37903 INFO ************ Epoch=7 end ************
2023-01-06 22:47:01,236 P37903 INFO [Metrics] AUC: 0.976444
2023-01-06 22:47:01,237 P37903 INFO Save best model: monitor(max): 0.976444
2023-01-06 22:47:01,244 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:47:01,320 P37903 INFO Train loss @epoch 8: 0.542129
2023-01-06 22:47:01,320 P37903 INFO ************ Epoch=8 end ************
2023-01-06 22:47:10,177 P37903 INFO [Metrics] AUC: 0.977678
2023-01-06 22:47:10,178 P37903 INFO Save best model: monitor(max): 0.977678
2023-01-06 22:47:10,184 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:47:10,251 P37903 INFO Train loss @epoch 9: 0.525614
2023-01-06 22:47:10,251 P37903 INFO ************ Epoch=9 end ************
2023-01-06 22:47:19,767 P37903 INFO [Metrics] AUC: 0.978224
2023-01-06 22:47:19,768 P37903 INFO Save best model: monitor(max): 0.978224
2023-01-06 22:47:19,774 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:47:19,842 P37903 INFO Train loss @epoch 10: 0.512716
2023-01-06 22:47:19,842 P37903 INFO ************ Epoch=10 end ************
2023-01-06 22:47:29,830 P37903 INFO [Metrics] AUC: 0.978376
2023-01-06 22:47:29,831 P37903 INFO Save best model: monitor(max): 0.978376
2023-01-06 22:47:29,838 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:47:29,933 P37903 INFO Train loss @epoch 11: 0.496952
2023-01-06 22:47:29,933 P37903 INFO ************ Epoch=11 end ************
2023-01-06 22:47:38,818 P37903 INFO [Metrics] AUC: 0.979660
2023-01-06 22:47:38,818 P37903 INFO Save best model: monitor(max): 0.979660
2023-01-06 22:47:38,825 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:47:38,894 P37903 INFO Train loss @epoch 12: 0.484935
2023-01-06 22:47:38,895 P37903 INFO ************ Epoch=12 end ************
2023-01-06 22:47:46,336 P37903 INFO [Metrics] AUC: 0.980664
2023-01-06 22:47:46,336 P37903 INFO Save best model: monitor(max): 0.980664
2023-01-06 22:47:46,350 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:47:46,418 P37903 INFO Train loss @epoch 13: 0.473727
2023-01-06 22:47:46,419 P37903 INFO ************ Epoch=13 end ************
2023-01-06 22:47:56,233 P37903 INFO [Metrics] AUC: 0.980411
2023-01-06 22:47:56,234 P37903 INFO Monitor(max) STOP: 0.980411 !
2023-01-06 22:47:56,234 P37903 INFO Reduce learning rate on plateau: 0.000100
2023-01-06 22:47:56,234 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:47:56,310 P37903 INFO Train loss @epoch 14: 0.465223
2023-01-06 22:47:56,310 P37903 INFO ************ Epoch=14 end ************
2023-01-06 22:48:03,433 P37903 INFO [Metrics] AUC: 0.984199
2023-01-06 22:48:03,434 P37903 INFO Save best model: monitor(max): 0.984199
2023-01-06 22:48:03,442 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:48:03,521 P37903 INFO Train loss @epoch 15: 0.409544
2023-01-06 22:48:03,522 P37903 INFO ************ Epoch=15 end ************
2023-01-06 22:48:09,545 P37903 INFO [Metrics] AUC: 0.985565
2023-01-06 22:48:09,545 P37903 INFO Save best model: monitor(max): 0.985565
2023-01-06 22:48:09,552 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:48:09,614 P37903 INFO Train loss @epoch 16: 0.345579
2023-01-06 22:48:09,614 P37903 INFO ************ Epoch=16 end ************
2023-01-06 22:48:13,961 P37903 INFO [Metrics] AUC: 0.986155
2023-01-06 22:48:13,962 P37903 INFO Save best model: monitor(max): 0.986155
2023-01-06 22:48:13,966 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:48:14,033 P37903 INFO Train loss @epoch 17: 0.312664
2023-01-06 22:48:14,033 P37903 INFO ************ Epoch=17 end ************
2023-01-06 22:48:17,949 P37903 INFO [Metrics] AUC: 0.986371
2023-01-06 22:48:17,949 P37903 INFO Save best model: monitor(max): 0.986371
2023-01-06 22:48:17,956 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:48:18,028 P37903 INFO Train loss @epoch 18: 0.293378
2023-01-06 22:48:18,028 P37903 INFO ************ Epoch=18 end ************
2023-01-06 22:48:24,435 P37903 INFO [Metrics] AUC: 0.986521
2023-01-06 22:48:24,436 P37903 INFO Save best model: monitor(max): 0.986521
2023-01-06 22:48:24,442 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:48:24,519 P37903 INFO Train loss @epoch 19: 0.281362
2023-01-06 22:48:24,519 P37903 INFO ************ Epoch=19 end ************
2023-01-06 22:48:31,276 P37903 INFO [Metrics] AUC: 0.986597
2023-01-06 22:48:31,276 P37903 INFO Save best model: monitor(max): 0.986597
2023-01-06 22:48:31,283 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:48:31,367 P37903 INFO Train loss @epoch 20: 0.271261
2023-01-06 22:48:31,367 P37903 INFO ************ Epoch=20 end ************
2023-01-06 22:48:36,886 P37903 INFO [Metrics] AUC: 0.986581
2023-01-06 22:48:36,886 P37903 INFO Monitor(max) STOP: 0.986581 !
2023-01-06 22:48:36,886 P37903 INFO Reduce learning rate on plateau: 0.000010
2023-01-06 22:48:36,887 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:48:36,960 P37903 INFO Train loss @epoch 21: 0.263937
2023-01-06 22:48:36,960 P37903 INFO ************ Epoch=21 end ************
2023-01-06 22:48:42,727 P37903 INFO [Metrics] AUC: 0.986748
2023-01-06 22:48:42,728 P37903 INFO Save best model: monitor(max): 0.986748
2023-01-06 22:48:42,732 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:48:42,779 P37903 INFO Train loss @epoch 22: 0.251688
2023-01-06 22:48:42,779 P37903 INFO ************ Epoch=22 end ************
2023-01-06 22:48:49,412 P37903 INFO [Metrics] AUC: 0.986735
2023-01-06 22:48:49,413 P37903 INFO Monitor(max) STOP: 0.986735 !
2023-01-06 22:48:49,413 P37903 INFO Reduce learning rate on plateau: 0.000001
2023-01-06 22:48:49,413 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:48:49,479 P37903 INFO Train loss @epoch 23: 0.249140
2023-01-06 22:48:49,479 P37903 INFO ************ Epoch=23 end ************
2023-01-06 22:48:56,623 P37903 INFO [Metrics] AUC: 0.986747
2023-01-06 22:48:56,624 P37903 INFO Monitor(max) STOP: 0.986747 !
2023-01-06 22:48:56,624 P37903 INFO Reduce learning rate on plateau: 0.000001
2023-01-06 22:48:56,624 P37903 INFO ********* Epoch==24 early stop *********
2023-01-06 22:48:56,624 P37903 INFO --- 50/50 batches finished ---
2023-01-06 22:48:56,690 P37903 INFO Train loss @epoch 24: 0.247568
2023-01-06 22:48:56,690 P37903 INFO Training finished.
2023-01-06 22:48:56,690 P37903 INFO Load best model: /home/FuxiCTR/benchmark/checkpoints/FINAL_frappe_x1/frappe_x1_47e6e0df/FINAL_frappe_x1_003_c3722b71.model
2023-01-06 22:48:56,706 P37903 INFO ****** Validation evaluation ******
2023-01-06 22:48:57,429 P37903 INFO [Metrics] AUC: 0.986748 - logloss: 0.127132
2023-01-06 22:48:57,511 P37903 INFO ******** Test evaluation ********
2023-01-06 22:48:57,512 P37903 INFO Loading data...
2023-01-06 22:48:57,512 P37903 INFO Loading data from h5: ../data/Frappe/frappe_x1_47e6e0df/test.h5
2023-01-06 22:48:57,523 P37903 INFO Test samples: total/28860, blocks/1
2023-01-06 22:48:57,523 P37903 INFO Loading test data done.
2023-01-06 22:48:58,044 P37903 INFO [Metrics] AUC: 0.985744 - logloss: 0.131070

```
