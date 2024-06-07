## FINAL_1B_avazu_x1

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

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FINAL_1B_avazu_x1_tuner_config_01](./FINAL_1B_avazu_x1_tuner_config_01). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/FINAL
    nohup python run_expid.py --config XXX/benchmarks/FINAL/FINAL_1B_avazu_x1_tuner_config_01 --expid FINAL_avazu_x1_030_3ad8933e --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.766067 | 0.366463  |


### Logs
```python
2022-12-31 08:58:07,301 P81977 INFO Params: {
    "batch_size": "4096",
    "block1_dropout": "0.2",
    "block1_hidden_activations": "ReLU",
    "block1_hidden_units": "[400, 400, 400]",
    "block2_dropout": "0",
    "block2_hidden_activations": "None",
    "block2_hidden_units": "[64, 64, 64]",
    "block_type": "1B",
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
    "gpu": "5",
    "group_id": "None",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FINAL",
    "model_id": "FINAL_avazu_x1_030_3ad8933e",
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
    "use_field_gate": "False",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "1"
}
2022-12-31 08:58:07,302 P81977 INFO Load feature_map from json: ../data/Avazu/avazu_x1_0bbde04e/feature_map.json
2022-12-31 08:58:07,302 P81977 INFO Set column index...
2022-12-31 08:58:07,302 P81977 INFO Feature specs: {
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
2022-12-31 08:58:10,768 P81977 INFO Total number of parameters: 13398211.
2022-12-31 08:58:10,768 P81977 INFO Loading data...
2022-12-31 08:58:10,768 P81977 INFO Loading data from h5: ../data/Avazu/avazu_x1_0bbde04e/train.h5
2022-12-31 08:58:23,223 P81977 INFO Train samples: total/28300276, blocks/1
2022-12-31 08:58:23,223 P81977 INFO Loading data from h5: ../data/Avazu/avazu_x1_0bbde04e/valid.h5
2022-12-31 08:58:25,077 P81977 INFO Validation samples: total/4042897, blocks/1
2022-12-31 08:58:25,078 P81977 INFO Loading train and validation data done.
2022-12-31 08:58:25,078 P81977 INFO Start training: 6910 batches/epoch
2022-12-31 08:58:25,078 P81977 INFO ************ Epoch=1 start ************
2022-12-31 09:01:59,846 P81977 INFO [Metrics] AUC: 0.737371
2022-12-31 09:01:59,849 P81977 INFO Save best model: monitor(max): 0.737371
2022-12-31 09:01:59,947 P81977 INFO --- 6910/6910 batches finished ---
2022-12-31 09:02:00,007 P81977 INFO Train loss @epoch 1: 0.441603
2022-12-31 09:02:00,007 P81977 INFO ************ Epoch=1 end ************
2022-12-31 09:05:32,490 P81977 INFO [Metrics] AUC: 0.734711
2022-12-31 09:05:32,493 P81977 INFO Monitor(max) STOP: 0.734711 !
2022-12-31 09:05:32,493 P81977 INFO Reduce learning rate on plateau: 0.000100
2022-12-31 09:05:32,494 P81977 INFO --- 6910/6910 batches finished ---
2022-12-31 09:05:32,556 P81977 INFO Train loss @epoch 2: 0.436825
2022-12-31 09:05:32,556 P81977 INFO ************ Epoch=2 end ************
2022-12-31 09:09:04,454 P81977 INFO [Metrics] AUC: 0.745579
2022-12-31 09:09:04,457 P81977 INFO Save best model: monitor(max): 0.745579
2022-12-31 09:09:04,563 P81977 INFO --- 6910/6910 batches finished ---
2022-12-31 09:09:04,628 P81977 INFO Train loss @epoch 3: 0.409492
2022-12-31 09:09:04,628 P81977 INFO ************ Epoch=3 end ************
2022-12-31 09:12:38,390 P81977 INFO [Metrics] AUC: 0.747427
2022-12-31 09:12:38,392 P81977 INFO Save best model: monitor(max): 0.747427
2022-12-31 09:12:38,499 P81977 INFO --- 6910/6910 batches finished ---
2022-12-31 09:12:38,561 P81977 INFO Train loss @epoch 4: 0.411459
2022-12-31 09:12:38,561 P81977 INFO ************ Epoch=4 end ************
2022-12-31 09:16:11,190 P81977 INFO [Metrics] AUC: 0.746381
2022-12-31 09:16:11,193 P81977 INFO Monitor(max) STOP: 0.746381 !
2022-12-31 09:16:11,193 P81977 INFO Reduce learning rate on plateau: 0.000010
2022-12-31 09:16:11,194 P81977 INFO --- 6910/6910 batches finished ---
2022-12-31 09:16:11,247 P81977 INFO Train loss @epoch 5: 0.412222
2022-12-31 09:16:11,247 P81977 INFO ************ Epoch=5 end ************
2022-12-31 09:19:44,808 P81977 INFO [Metrics] AUC: 0.747866
2022-12-31 09:19:44,812 P81977 INFO Save best model: monitor(max): 0.747866
2022-12-31 09:19:44,917 P81977 INFO --- 6910/6910 batches finished ---
2022-12-31 09:19:44,976 P81977 INFO Train loss @epoch 6: 0.397319
2022-12-31 09:19:44,976 P81977 INFO ************ Epoch=6 end ************
2022-12-31 09:23:18,543 P81977 INFO [Metrics] AUC: 0.744631
2022-12-31 09:23:18,549 P81977 INFO Monitor(max) STOP: 0.744631 !
2022-12-31 09:23:18,549 P81977 INFO Reduce learning rate on plateau: 0.000001
2022-12-31 09:23:18,550 P81977 INFO --- 6910/6910 batches finished ---
2022-12-31 09:23:18,614 P81977 INFO Train loss @epoch 7: 0.394690
2022-12-31 09:23:18,615 P81977 INFO ************ Epoch=7 end ************
2022-12-31 09:26:50,083 P81977 INFO [Metrics] AUC: 0.739957
2022-12-31 09:26:50,086 P81977 INFO Monitor(max) STOP: 0.739957 !
2022-12-31 09:26:50,086 P81977 INFO Reduce learning rate on plateau: 0.000001
2022-12-31 09:26:50,086 P81977 INFO ********* Epoch==8 early stop *********
2022-12-31 09:26:50,089 P81977 INFO --- 6910/6910 batches finished ---
2022-12-31 09:26:50,161 P81977 INFO Train loss @epoch 8: 0.387489
2022-12-31 09:26:50,161 P81977 INFO Training finished.
2022-12-31 09:26:50,161 P81977 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/FINAL_avazu_x1/avazu_x1_0bbde04e/FINAL_avazu_x1_030_3ad8933e.model
2022-12-31 09:26:50,217 P81977 INFO ****** Validation evaluation ******
2022-12-31 09:27:00,732 P81977 INFO [Metrics] AUC: 0.747866 - logloss: 0.395675
2022-12-31 09:27:00,828 P81977 INFO ******** Test evaluation ********
2022-12-31 09:27:00,828 P81977 INFO Loading data...
2022-12-31 09:27:00,828 P81977 INFO Loading data from h5: ../data/Avazu/avazu_x1_0bbde04e/test.h5
2022-12-31 09:27:04,302 P81977 INFO Test samples: total/8085794, blocks/1
2022-12-31 09:27:04,302 P81977 INFO Loading test data done.
2022-12-31 09:27:26,159 P81977 INFO [Metrics] AUC: 0.766067 - logloss: 0.366463

```
