## FinalMLP_movielenslatest_x1

A hands-on guide to run the FinalMLP model on the MovielensLatest_x1 dataset.

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
Please refer to the BARS dataset [MovielensLatest_x1](https://github.com/openbenchmark/BARS/blob/main/datasets/MovieLens#MovielensLatest_x1) to get data ready.

### Code

We use the [FinalMLP](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/FinalMLP) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/MovieLens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FinalMLP_movielenslatest_x1_tuner_config_06](./FinalMLP_movielenslatest_x1_tuner_config_06). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/FinalMLP
    nohup python run_expid.py --config XXX/benchmarks/FinalMLP/FinalMLP_movielenslatest_x1_tuner_config_06 --expid FinalMLP_movielenslatest_x1_004_498f3e4f --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.971968 | 0.196645  |


### Logs
```python
2022-12-17 14:05:16,676 P24328 INFO Params: {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_233328b6",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "feature_specs": "None",
    "fs1_context": "[]",
    "fs2_context": "[]",
    "fs_hidden_units": "[800]",
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
    "mlp2_dropout": "0.2",
    "mlp2_hidden_activations": "relu",
    "mlp2_hidden_units": "[800]",
    "model": "FinalMLP",
    "model_id": "FinalMLP_movielenslatest_x1_004_498f3e4f",
    "model_root": "./checkpoints/FinalMLP_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "num_heads": "10",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Movielens/MovielensLatest_x1/test.csv",
    "train_data": "../data/Movielens/MovielensLatest_x1/train.csv",
    "use_fs": "True",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1"
}
2022-12-17 14:05:16,677 P24328 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_233328b6/feature_map.json
2022-12-17 14:05:16,677 P24328 INFO Set column index...
2022-12-17 14:05:16,677 P24328 INFO Feature specs: {
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 23605, 'vocab_size': 23606}",
    "tag_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 49658, 'vocab_size': 49659}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 16976, 'vocab_size': 16977}"
}
2022-12-17 14:05:21,131 P24328 INFO Total number of parameters: 1040902.
2022-12-17 14:05:21,131 P24328 INFO Loading data...
2022-12-17 14:05:21,131 P24328 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_233328b6/train.h5
2022-12-17 14:05:21,192 P24328 INFO Train samples: total/1404801, blocks/1
2022-12-17 14:05:21,192 P24328 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_233328b6/valid.h5
2022-12-17 14:05:21,204 P24328 INFO Validation samples: total/401372, blocks/1
2022-12-17 14:05:21,204 P24328 INFO Loading train and validation data done.
2022-12-17 14:05:21,204 P24328 INFO Start training: 343 batches/epoch
2022-12-17 14:05:21,204 P24328 INFO ************ Epoch=1 start ************
2022-12-17 14:06:29,679 P24328 INFO [Metrics] AUC: 0.939571
2022-12-17 14:06:29,679 P24328 INFO Save best model: monitor(max): 0.939571
2022-12-17 14:06:29,686 P24328 INFO --- 343/343 batches finished ---
2022-12-17 14:06:29,752 P24328 INFO Train loss @epoch 1: 0.380939
2022-12-17 14:06:29,752 P24328 INFO ************ Epoch=1 end ************
2022-12-17 14:07:39,822 P24328 INFO [Metrics] AUC: 0.947405
2022-12-17 14:07:39,823 P24328 INFO Save best model: monitor(max): 0.947405
2022-12-17 14:07:39,830 P24328 INFO --- 343/343 batches finished ---
2022-12-17 14:07:39,873 P24328 INFO Train loss @epoch 2: 0.372074
2022-12-17 14:07:39,873 P24328 INFO ************ Epoch=2 end ************
2022-12-17 14:08:50,763 P24328 INFO [Metrics] AUC: 0.949848
2022-12-17 14:08:50,764 P24328 INFO Save best model: monitor(max): 0.949848
2022-12-17 14:08:50,771 P24328 INFO --- 343/343 batches finished ---
2022-12-17 14:08:50,813 P24328 INFO Train loss @epoch 3: 0.375706
2022-12-17 14:08:50,813 P24328 INFO ************ Epoch=3 end ************
2022-12-17 14:10:01,751 P24328 INFO [Metrics] AUC: 0.951878
2022-12-17 14:10:01,752 P24328 INFO Save best model: monitor(max): 0.951878
2022-12-17 14:10:01,759 P24328 INFO --- 343/343 batches finished ---
2022-12-17 14:10:01,799 P24328 INFO Train loss @epoch 4: 0.375421
2022-12-17 14:10:01,799 P24328 INFO ************ Epoch=4 end ************
2022-12-17 14:11:13,769 P24328 INFO [Metrics] AUC: 0.953459
2022-12-17 14:11:13,769 P24328 INFO Save best model: monitor(max): 0.953459
2022-12-17 14:11:13,777 P24328 INFO --- 343/343 batches finished ---
2022-12-17 14:11:13,818 P24328 INFO Train loss @epoch 5: 0.375694
2022-12-17 14:11:13,818 P24328 INFO ************ Epoch=5 end ************
2022-12-17 14:12:25,320 P24328 INFO [Metrics] AUC: 0.954723
2022-12-17 14:12:25,321 P24328 INFO Save best model: monitor(max): 0.954723
2022-12-17 14:12:25,331 P24328 INFO --- 343/343 batches finished ---
2022-12-17 14:12:25,375 P24328 INFO Train loss @epoch 6: 0.375326
2022-12-17 14:12:25,375 P24328 INFO ************ Epoch=6 end ************
2022-12-17 14:13:36,428 P24328 INFO [Metrics] AUC: 0.955611
2022-12-17 14:13:36,428 P24328 INFO Save best model: monitor(max): 0.955611
2022-12-17 14:13:36,435 P24328 INFO --- 343/343 batches finished ---
2022-12-17 14:13:36,473 P24328 INFO Train loss @epoch 7: 0.375191
2022-12-17 14:13:36,473 P24328 INFO ************ Epoch=7 end ************
2022-12-17 14:14:47,343 P24328 INFO [Metrics] AUC: 0.955632
2022-12-17 14:14:47,343 P24328 INFO Save best model: monitor(max): 0.955632
2022-12-17 14:14:47,350 P24328 INFO --- 343/343 batches finished ---
2022-12-17 14:14:47,396 P24328 INFO Train loss @epoch 8: 0.371717
2022-12-17 14:14:47,396 P24328 INFO ************ Epoch=8 end ************
2022-12-17 14:15:58,334 P24328 INFO [Metrics] AUC: 0.956352
2022-12-17 14:15:58,334 P24328 INFO Save best model: monitor(max): 0.956352
2022-12-17 14:15:58,341 P24328 INFO --- 343/343 batches finished ---
2022-12-17 14:15:58,378 P24328 INFO Train loss @epoch 9: 0.370084
2022-12-17 14:15:58,379 P24328 INFO ************ Epoch=9 end ************
2022-12-17 14:17:08,776 P24328 INFO [Metrics] AUC: 0.956738
2022-12-17 14:17:08,777 P24328 INFO Save best model: monitor(max): 0.956738
2022-12-17 14:17:08,784 P24328 INFO --- 343/343 batches finished ---
2022-12-17 14:17:08,822 P24328 INFO Train loss @epoch 10: 0.368781
2022-12-17 14:17:08,822 P24328 INFO ************ Epoch=10 end ************
2022-12-17 14:18:19,486 P24328 INFO [Metrics] AUC: 0.957527
2022-12-17 14:18:19,486 P24328 INFO Save best model: monitor(max): 0.957527
2022-12-17 14:18:19,494 P24328 INFO --- 343/343 batches finished ---
2022-12-17 14:18:19,532 P24328 INFO Train loss @epoch 11: 0.368817
2022-12-17 14:18:19,533 P24328 INFO ************ Epoch=11 end ************
2022-12-17 14:19:30,784 P24328 INFO [Metrics] AUC: 0.958320
2022-12-17 14:19:30,785 P24328 INFO Save best model: monitor(max): 0.958320
2022-12-17 14:19:30,793 P24328 INFO --- 343/343 batches finished ---
2022-12-17 14:19:30,834 P24328 INFO Train loss @epoch 12: 0.367380
2022-12-17 14:19:30,834 P24328 INFO ************ Epoch=12 end ************
2022-12-17 14:20:42,131 P24328 INFO [Metrics] AUC: 0.957900
2022-12-17 14:20:42,132 P24328 INFO Monitor(max) STOP: 0.957900 !
2022-12-17 14:20:42,132 P24328 INFO Reduce learning rate on plateau: 0.000100
2022-12-17 14:20:42,132 P24328 INFO --- 343/343 batches finished ---
2022-12-17 14:20:42,180 P24328 INFO Train loss @epoch 13: 0.366062
2022-12-17 14:20:42,180 P24328 INFO ************ Epoch=13 end ************
2022-12-17 14:21:54,370 P24328 INFO [Metrics] AUC: 0.969740
2022-12-17 14:21:54,370 P24328 INFO Save best model: monitor(max): 0.969740
2022-12-17 14:21:54,378 P24328 INFO --- 343/343 batches finished ---
2022-12-17 14:21:54,439 P24328 INFO Train loss @epoch 14: 0.266173
2022-12-17 14:21:54,439 P24328 INFO ************ Epoch=14 end ************
2022-12-17 14:23:06,927 P24328 INFO [Metrics] AUC: 0.972100
2022-12-17 14:23:06,928 P24328 INFO Save best model: monitor(max): 0.972100
2022-12-17 14:23:06,935 P24328 INFO --- 343/343 batches finished ---
2022-12-17 14:23:06,973 P24328 INFO Train loss @epoch 15: 0.182185
2022-12-17 14:23:06,973 P24328 INFO ************ Epoch=15 end ************
2022-12-17 14:24:20,268 P24328 INFO [Metrics] AUC: 0.971802
2022-12-17 14:24:20,269 P24328 INFO Monitor(max) STOP: 0.971802 !
2022-12-17 14:24:20,269 P24328 INFO Reduce learning rate on plateau: 0.000010
2022-12-17 14:24:20,269 P24328 INFO --- 343/343 batches finished ---
2022-12-17 14:24:20,319 P24328 INFO Train loss @epoch 16: 0.139268
2022-12-17 14:24:20,319 P24328 INFO ************ Epoch=16 end ************
2022-12-17 14:25:29,170 P24328 INFO [Metrics] AUC: 0.972058
2022-12-17 14:25:29,170 P24328 INFO Monitor(max) STOP: 0.972058 !
2022-12-17 14:25:29,170 P24328 INFO Reduce learning rate on plateau: 0.000001
2022-12-17 14:25:29,171 P24328 INFO ********* Epoch==17 early stop *********
2022-12-17 14:25:29,171 P24328 INFO --- 343/343 batches finished ---
2022-12-17 14:25:29,210 P24328 INFO Train loss @epoch 17: 0.106034
2022-12-17 14:25:29,210 P24328 INFO Training finished.
2022-12-17 14:25:29,210 P24328 INFO Load best model: /home/FuxiCTRv2/benchmark/checkpoints/FinalMLP_movielenslatest_x1/movielenslatest_x1_233328b6/FinalMLP_movielenslatest_x1_004_498f3e4f.model
2022-12-17 14:25:29,249 P24328 INFO ****** Validation evaluation ******
2022-12-17 14:25:33,216 P24328 INFO [Metrics] AUC: 0.972100 - logloss: 0.196004
2022-12-17 14:25:33,258 P24328 INFO ******** Test evaluation ********
2022-12-17 14:25:33,258 P24328 INFO Loading data...
2022-12-17 14:25:33,258 P24328 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_233328b6/test.h5
2022-12-17 14:25:33,265 P24328 INFO Test samples: total/200686, blocks/1
2022-12-17 14:25:33,265 P24328 INFO Loading test data done.
2022-12-17 14:25:35,310 P24328 INFO [Metrics] AUC: 0.971968 - logloss: 0.196645

```
