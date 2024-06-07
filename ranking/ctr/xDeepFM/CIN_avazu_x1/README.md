## CIN_avazu_x1

A hands-on guide to run the xDeepFM model on the Avazu_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
  GPU: Tesla P100 16G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 11.4
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  fuxictr: 1.1.0
  ```

### Dataset
Dataset ID: [Avazu_x1](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [xDeepFM](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/xDeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CIN_avazu_x1_tuner_config_01](./CIN_avazu_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CIN_avazu_x1
    nohup python run_expid.py --config ./CIN_avazu_x1_tuner_config_01 --expid xDeepFM_avazu_x1_003_7f3fed32 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.762390 | 0.367299  |


### Logs
```python
2022-01-19 15:04:47,056 P1299 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "cin_layer_units": "[64, 64, 64]",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "dnn_hidden_units": "None",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "xDeepFM",
    "model_id": "xDeepFM_avazu_x1_003_7f3fed32",
    "model_root": "./Avazu/xDeepFM_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-19 15:04:47,057 P1299 INFO Set up feature encoder...
2022-01-19 15:04:47,057 P1299 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-01-19 15:04:47,057 P1299 INFO Loading data...
2022-01-19 15:04:47,059 P1299 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-01-19 15:04:49,469 P1299 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-01-19 15:04:49,808 P1299 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-01-19 15:04:49,808 P1299 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-01-19 15:04:49,808 P1299 INFO Loading train data done.
2022-01-19 15:04:54,207 P1299 INFO Total number of parameters: 14496174.
2022-01-19 15:04:54,207 P1299 INFO Start training: 6910 batches/epoch
2022-01-19 15:04:54,207 P1299 INFO ************ Epoch=1 start ************
2022-01-19 15:37:59,970 P1299 INFO [Metrics] AUC: 0.736006 - logloss: 0.403859
2022-01-19 15:37:59,974 P1299 INFO Save best model: monitor(max): 0.736006
2022-01-19 15:38:00,038 P1299 INFO --- 6910/6910 batches finished ---
2022-01-19 15:38:00,068 P1299 INFO Train loss: 0.429377
2022-01-19 15:38:00,068 P1299 INFO ************ Epoch=1 end ************
2022-01-19 16:07:37,698 P1299 INFO [Metrics] AUC: 0.740224 - logloss: 0.399523
2022-01-19 16:07:37,703 P1299 INFO Save best model: monitor(max): 0.740224
2022-01-19 16:07:37,803 P1299 INFO --- 6910/6910 batches finished ---
2022-01-19 16:07:37,839 P1299 INFO Train loss: 0.427476
2022-01-19 16:07:37,840 P1299 INFO ************ Epoch=2 end ************
2022-01-19 16:37:13,072 P1299 INFO [Metrics] AUC: 0.741092 - logloss: 0.399354
2022-01-19 16:37:13,076 P1299 INFO Save best model: monitor(max): 0.741092
2022-01-19 16:37:13,166 P1299 INFO --- 6910/6910 batches finished ---
2022-01-19 16:37:13,202 P1299 INFO Train loss: 0.427457
2022-01-19 16:37:13,202 P1299 INFO ************ Epoch=3 end ************
2022-01-19 17:06:47,583 P1299 INFO [Metrics] AUC: 0.738909 - logloss: 0.399946
2022-01-19 17:06:47,586 P1299 INFO Monitor(max) STOP: 0.738909 !
2022-01-19 17:06:47,586 P1299 INFO Reduce learning rate on plateau: 0.000100
2022-01-19 17:06:47,586 P1299 INFO --- 6910/6910 batches finished ---
2022-01-19 17:06:47,629 P1299 INFO Train loss: 0.426773
2022-01-19 17:06:47,630 P1299 INFO ************ Epoch=4 end ************
2022-01-19 17:36:18,310 P1299 INFO [Metrics] AUC: 0.746848 - logloss: 0.395632
2022-01-19 17:36:18,312 P1299 INFO Save best model: monitor(max): 0.746848
2022-01-19 17:36:18,400 P1299 INFO --- 6910/6910 batches finished ---
2022-01-19 17:36:18,433 P1299 INFO Train loss: 0.406090
2022-01-19 17:36:18,433 P1299 INFO ************ Epoch=5 end ************
2022-01-19 18:05:50,805 P1299 INFO [Metrics] AUC: 0.746120 - logloss: 0.395994
2022-01-19 18:05:50,808 P1299 INFO Monitor(max) STOP: 0.746120 !
2022-01-19 18:05:50,808 P1299 INFO Reduce learning rate on plateau: 0.000010
2022-01-19 18:05:50,809 P1299 INFO --- 6910/6910 batches finished ---
2022-01-19 18:05:50,844 P1299 INFO Train loss: 0.407529
2022-01-19 18:05:50,845 P1299 INFO ************ Epoch=6 end ************
2022-01-19 18:35:21,572 P1299 INFO [Metrics] AUC: 0.747134 - logloss: 0.395617
2022-01-19 18:35:21,575 P1299 INFO Save best model: monitor(max): 0.747134
2022-01-19 18:35:21,664 P1299 INFO --- 6910/6910 batches finished ---
2022-01-19 18:35:21,702 P1299 INFO Train loss: 0.397857
2022-01-19 18:35:21,702 P1299 INFO ************ Epoch=7 end ************
2022-01-19 19:04:52,874 P1299 INFO [Metrics] AUC: 0.745046 - logloss: 0.396780
2022-01-19 19:04:52,877 P1299 INFO Monitor(max) STOP: 0.745046 !
2022-01-19 19:04:52,877 P1299 INFO Reduce learning rate on plateau: 0.000001
2022-01-19 19:04:52,877 P1299 INFO --- 6910/6910 batches finished ---
2022-01-19 19:04:52,925 P1299 INFO Train loss: 0.394690
2022-01-19 19:04:52,925 P1299 INFO ************ Epoch=8 end ************
2022-01-19 19:34:25,141 P1299 INFO [Metrics] AUC: 0.744240 - logloss: 0.397411
2022-01-19 19:34:25,144 P1299 INFO Monitor(max) STOP: 0.744240 !
2022-01-19 19:34:25,144 P1299 INFO Reduce learning rate on plateau: 0.000001
2022-01-19 19:34:25,144 P1299 INFO Early stopping at epoch=9
2022-01-19 19:34:25,144 P1299 INFO --- 6910/6910 batches finished ---
2022-01-19 19:34:25,189 P1299 INFO Train loss: 0.390291
2022-01-19 19:34:25,189 P1299 INFO Training finished.
2022-01-19 19:34:25,189 P1299 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Avazu/xDeepFM_avazu_x1/avazu_x1_3fb65689/xDeepFM_avazu_x1_003_7f3fed32.model
2022-01-19 19:34:25,260 P1299 INFO ****** Validation evaluation ******
2022-01-19 19:34:48,041 P1299 INFO [Metrics] AUC: 0.747134 - logloss: 0.395617
2022-01-19 19:34:48,076 P1299 INFO ******** Test evaluation ********
2022-01-19 19:34:48,076 P1299 INFO Loading data...
2022-01-19 19:34:48,077 P1299 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-01-19 19:34:48,807 P1299 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-01-19 19:34:48,808 P1299 INFO Loading test data done.
2022-01-19 19:35:34,730 P1299 INFO [Metrics] AUC: 0.762390 - logloss: 0.367299

```
