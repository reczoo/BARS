## CrossNet_criteo_x1

A hands-on guide to run the DCN model on the Criteo_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

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
Dataset ID: [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [DCN](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DCN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CrossNet_criteo_x1_tuner_config_01](./CrossNet_criteo_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CrossNet_criteo_x1
    nohup python run_expid.py --config ./CrossNet_criteo_x1_tuner_config_01 --expid DCN_criteo_x1_012_32e10ead --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.802017 | 0.449093  |


### Logs
```python
2022-01-21 13:29:17,165 P11228 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "crossing_layers": "9",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "None",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCN",
    "model_id": "DCN_criteo_x1_012_32e10ead",
    "model_root": "./Criteo/DCN_criteo_x1/",
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
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-21 13:29:17,166 P11228 INFO Set up feature encoder...
2022-01-21 13:29:17,166 P11228 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-01-21 13:29:17,166 P11228 INFO Loading data...
2022-01-21 13:29:17,169 P11228 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-01-21 13:29:21,840 P11228 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-01-21 13:29:23,146 P11228 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-01-21 13:29:23,147 P11228 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-01-21 13:29:23,147 P11228 INFO Loading train data done.
2022-01-21 13:29:26,857 P11228 INFO Total number of parameters: 20870571.
2022-01-21 13:29:26,857 P11228 INFO Start training: 8058 batches/epoch
2022-01-21 13:29:26,857 P11228 INFO ************ Epoch=1 start ************
2022-01-21 13:50:06,965 P11228 INFO [Metrics] AUC: 0.794210 - logloss: 0.456605
2022-01-21 13:50:06,967 P11228 INFO Save best model: monitor(max): 0.794210
2022-01-21 13:50:07,041 P11228 INFO --- 8058/8058 batches finished ---
2022-01-21 13:50:07,095 P11228 INFO Train loss: 0.468046
2022-01-21 13:50:07,095 P11228 INFO ************ Epoch=1 end ************
2022-01-21 14:10:43,544 P11228 INFO [Metrics] AUC: 0.795742 - logloss: 0.454693
2022-01-21 14:10:43,546 P11228 INFO Save best model: monitor(max): 0.795742
2022-01-21 14:10:43,680 P11228 INFO --- 8058/8058 batches finished ---
2022-01-21 14:10:43,723 P11228 INFO Train loss: 0.462185
2022-01-21 14:10:43,723 P11228 INFO ************ Epoch=2 end ************
2022-01-21 14:31:18,493 P11228 INFO [Metrics] AUC: 0.795956 - logloss: 0.454494
2022-01-21 14:31:18,494 P11228 INFO Save best model: monitor(max): 0.795956
2022-01-21 14:31:18,648 P11228 INFO --- 8058/8058 batches finished ---
2022-01-21 14:31:18,687 P11228 INFO Train loss: 0.461463
2022-01-21 14:31:18,687 P11228 INFO ************ Epoch=3 end ************
2022-01-21 14:51:52,633 P11228 INFO [Metrics] AUC: 0.796611 - logloss: 0.453911
2022-01-21 14:51:52,635 P11228 INFO Save best model: monitor(max): 0.796611
2022-01-21 14:51:52,757 P11228 INFO --- 8058/8058 batches finished ---
2022-01-21 14:51:52,798 P11228 INFO Train loss: 0.461151
2022-01-21 14:51:52,798 P11228 INFO ************ Epoch=4 end ************
2022-01-21 15:12:24,797 P11228 INFO [Metrics] AUC: 0.796813 - logloss: 0.454291
2022-01-21 15:12:24,799 P11228 INFO Save best model: monitor(max): 0.796813
2022-01-21 15:12:24,916 P11228 INFO --- 8058/8058 batches finished ---
2022-01-21 15:12:24,957 P11228 INFO Train loss: 0.460921
2022-01-21 15:12:24,957 P11228 INFO ************ Epoch=5 end ************
2022-01-21 15:32:57,242 P11228 INFO [Metrics] AUC: 0.797036 - logloss: 0.453558
2022-01-21 15:32:57,244 P11228 INFO Save best model: monitor(max): 0.797036
2022-01-21 15:32:57,379 P11228 INFO --- 8058/8058 batches finished ---
2022-01-21 15:32:57,422 P11228 INFO Train loss: 0.460714
2022-01-21 15:32:57,422 P11228 INFO ************ Epoch=6 end ************
2022-01-21 15:53:33,884 P11228 INFO [Metrics] AUC: 0.797105 - logloss: 0.453733
2022-01-21 15:53:33,885 P11228 INFO Save best model: monitor(max): 0.797105
2022-01-21 15:53:34,001 P11228 INFO --- 8058/8058 batches finished ---
2022-01-21 15:53:34,042 P11228 INFO Train loss: 0.460623
2022-01-21 15:53:34,042 P11228 INFO ************ Epoch=7 end ************
2022-01-21 16:14:06,175 P11228 INFO [Metrics] AUC: 0.797175 - logloss: 0.453563
2022-01-21 16:14:06,176 P11228 INFO Save best model: monitor(max): 0.797175
2022-01-21 16:14:06,307 P11228 INFO --- 8058/8058 batches finished ---
2022-01-21 16:14:06,347 P11228 INFO Train loss: 0.460514
2022-01-21 16:14:06,347 P11228 INFO ************ Epoch=8 end ************
2022-01-21 16:34:41,533 P11228 INFO [Metrics] AUC: 0.797086 - logloss: 0.453491
2022-01-21 16:34:41,534 P11228 INFO Monitor(max) STOP: 0.797086 !
2022-01-21 16:34:41,534 P11228 INFO Reduce learning rate on plateau: 0.000100
2022-01-21 16:34:41,534 P11228 INFO --- 8058/8058 batches finished ---
2022-01-21 16:34:41,576 P11228 INFO Train loss: 0.460425
2022-01-21 16:34:41,576 P11228 INFO ************ Epoch=9 end ************
2022-01-21 16:55:10,791 P11228 INFO [Metrics] AUC: 0.800694 - logloss: 0.450387
2022-01-21 16:55:10,793 P11228 INFO Save best model: monitor(max): 0.800694
2022-01-21 16:55:10,928 P11228 INFO --- 8058/8058 batches finished ---
2022-01-21 16:55:10,966 P11228 INFO Train loss: 0.452914
2022-01-21 16:55:10,966 P11228 INFO ************ Epoch=10 end ************
2022-01-21 17:15:38,204 P11228 INFO [Metrics] AUC: 0.801290 - logloss: 0.449948
2022-01-21 17:15:38,206 P11228 INFO Save best model: monitor(max): 0.801290
2022-01-21 17:15:38,336 P11228 INFO --- 8058/8058 batches finished ---
2022-01-21 17:15:38,381 P11228 INFO Train loss: 0.450301
2022-01-21 17:15:38,381 P11228 INFO ************ Epoch=11 end ************
2022-01-21 17:35:38,065 P11228 INFO [Metrics] AUC: 0.801466 - logloss: 0.449804
2022-01-21 17:35:38,066 P11228 INFO Save best model: monitor(max): 0.801466
2022-01-21 17:35:38,199 P11228 INFO --- 8058/8058 batches finished ---
2022-01-21 17:35:38,242 P11228 INFO Train loss: 0.449035
2022-01-21 17:35:38,242 P11228 INFO ************ Epoch=12 end ************
2022-01-21 17:56:03,014 P11228 INFO [Metrics] AUC: 0.801667 - logloss: 0.449561
2022-01-21 17:56:03,015 P11228 INFO Save best model: monitor(max): 0.801667
2022-01-21 17:56:03,136 P11228 INFO --- 8058/8058 batches finished ---
2022-01-21 17:56:03,174 P11228 INFO Train loss: 0.448102
2022-01-21 17:56:03,174 P11228 INFO ************ Epoch=13 end ************
2022-01-21 18:16:26,839 P11228 INFO [Metrics] AUC: 0.801657 - logloss: 0.449624
2022-01-21 18:16:26,841 P11228 INFO Monitor(max) STOP: 0.801657 !
2022-01-21 18:16:26,841 P11228 INFO Reduce learning rate on plateau: 0.000010
2022-01-21 18:16:26,841 P11228 INFO --- 8058/8058 batches finished ---
2022-01-21 18:16:26,879 P11228 INFO Train loss: 0.447391
2022-01-21 18:16:26,880 P11228 INFO ************ Epoch=14 end ************
2022-01-21 18:36:51,155 P11228 INFO [Metrics] AUC: 0.801594 - logloss: 0.449795
2022-01-21 18:36:51,157 P11228 INFO Monitor(max) STOP: 0.801594 !
2022-01-21 18:36:51,157 P11228 INFO Reduce learning rate on plateau: 0.000001
2022-01-21 18:36:51,157 P11228 INFO Early stopping at epoch=15
2022-01-21 18:36:51,157 P11228 INFO --- 8058/8058 batches finished ---
2022-01-21 18:36:51,195 P11228 INFO Train loss: 0.444232
2022-01-21 18:36:51,195 P11228 INFO Training finished.
2022-01-21 18:36:51,195 P11228 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Criteo/DCN_criteo_x1/criteo_x1_7b681156/DCN_criteo_x1_012_32e10ead.model
2022-01-21 18:36:51,283 P11228 INFO ****** Validation evaluation ******
2022-01-21 18:37:21,444 P11228 INFO [Metrics] AUC: 0.801667 - logloss: 0.449561
2022-01-21 18:37:21,515 P11228 INFO ******** Test evaluation ********
2022-01-21 18:37:21,516 P11228 INFO Loading data...
2022-01-21 18:37:21,516 P11228 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-01-21 18:37:22,249 P11228 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-01-21 18:37:22,249 P11228 INFO Loading test data done.
2022-01-21 18:37:38,410 P11228 INFO [Metrics] AUC: 0.802017 - logloss: 0.449093

```
