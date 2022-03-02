## DeepIM_criteo_x1

A hands-on guide to run the DeepIM model on the Criteo_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) Gold 6278C CPU @ 2.60GHz
  GPU: Tesla V100 32G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 10.2
  python: 3.6.4
  pytorch: 1.0.0
  pandas: 0.22.0
  numpy: 1.19.2
  scipy: 1.5.4
  sklearn: 0.22.1
  pyyaml: 5.4.1
  h5py: 2.8.0
  tqdm: 4.60.0
  fuxictr: 1.1.0

  ```

### Dataset
Dataset ID: [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [DeepIM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DeepIM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DeepIM_criteo_x1_tuner_config_01](./DeepIM_criteo_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DeepIM_criteo_x1
    nohup python run_expid.py --config ./DeepIM_criteo_x1_tuner_config_01 --expid DeepIM_criteo_x1_001_6de9e773 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.813927 | 0.437959  |
| 2 | 0.813964 | 0.437948  |
| 3 | 0.813801 | 0.438088  |
| 4 | 0.813840 | 0.438001  |
| 5 | 0.813867 | 0.438036  |
| | | | 
| Avg | 0.813880 | 0.438006 |
| Std | &#177;0.00005878 | &#177;0.00005142 |


### Logs
```python
2022-02-09 08:04:26,195 P42567 INFO {
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "im_batch_norm": "True",
    "im_order": "3",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DeepIM",
    "model_id": "DeepIM_criteo_x1_001_6de9e773",
    "model_root": "./Criteo/DeepIM_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_batch_norm": "True",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
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
    "verbose": "0",
    "version": "pytorch"
}
2022-02-09 08:04:26,196 P42567 INFO Set up feature encoder...
2022-02-09 08:04:26,196 P42567 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-02-09 08:04:26,196 P42567 INFO Loading data...
2022-02-09 08:04:26,197 P42567 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-02-09 08:04:30,939 P42567 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-02-09 08:04:32,137 P42567 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-02-09 08:04:32,137 P42567 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-02-09 08:04:32,137 P42567 INFO Loading train data done.
2022-02-09 08:04:36,868 P42567 INFO Total number of parameters: 21343252.
2022-02-09 08:04:36,868 P42567 INFO Start training: 8058 batches/epoch
2022-02-09 08:04:36,868 P42567 INFO ************ Epoch=1 start ************
2022-02-09 08:11:33,785 P42567 INFO [Metrics] AUC: 0.804013 - logloss: 0.447266
2022-02-09 08:11:33,787 P42567 INFO Save best model: monitor(max): 0.804013
2022-02-09 08:11:33,868 P42567 INFO --- 8058/8058 batches finished ---
2022-02-09 08:11:33,917 P42567 INFO Train loss: 0.462306
2022-02-09 08:11:33,918 P42567 INFO ************ Epoch=1 end ************
2022-02-09 08:18:31,941 P42567 INFO [Metrics] AUC: 0.806196 - logloss: 0.445214
2022-02-09 08:18:31,943 P42567 INFO Save best model: monitor(max): 0.806196
2022-02-09 08:18:32,035 P42567 INFO --- 8058/8058 batches finished ---
2022-02-09 08:18:32,075 P42567 INFO Train loss: 0.456381
2022-02-09 08:18:32,075 P42567 INFO ************ Epoch=2 end ************
2022-02-09 08:25:33,509 P42567 INFO [Metrics] AUC: 0.807399 - logloss: 0.444086
2022-02-09 08:25:33,510 P42567 INFO Save best model: monitor(max): 0.807399
2022-02-09 08:25:33,600 P42567 INFO --- 8058/8058 batches finished ---
2022-02-09 08:25:33,650 P42567 INFO Train loss: 0.454917
2022-02-09 08:25:33,651 P42567 INFO ************ Epoch=3 end ************
2022-02-09 08:32:33,727 P42567 INFO [Metrics] AUC: 0.807991 - logloss: 0.443497
2022-02-09 08:32:33,729 P42567 INFO Save best model: monitor(max): 0.807991
2022-02-09 08:32:33,831 P42567 INFO --- 8058/8058 batches finished ---
2022-02-09 08:32:33,876 P42567 INFO Train loss: 0.454223
2022-02-09 08:32:33,876 P42567 INFO ************ Epoch=4 end ************
2022-02-09 08:39:31,154 P42567 INFO [Metrics] AUC: 0.808421 - logloss: 0.443139
2022-02-09 08:39:31,155 P42567 INFO Save best model: monitor(max): 0.808421
2022-02-09 08:39:31,243 P42567 INFO --- 8058/8058 batches finished ---
2022-02-09 08:39:31,291 P42567 INFO Train loss: 0.453751
2022-02-09 08:39:31,291 P42567 INFO ************ Epoch=5 end ************
2022-02-09 08:46:27,883 P42567 INFO [Metrics] AUC: 0.808926 - logloss: 0.442683
2022-02-09 08:46:27,885 P42567 INFO Save best model: monitor(max): 0.808926
2022-02-09 08:46:27,973 P42567 INFO --- 8058/8058 batches finished ---
2022-02-09 08:46:28,013 P42567 INFO Train loss: 0.453390
2022-02-09 08:46:28,013 P42567 INFO ************ Epoch=6 end ************
2022-02-09 08:53:26,999 P42567 INFO [Metrics] AUC: 0.809099 - logloss: 0.442545
2022-02-09 08:53:27,000 P42567 INFO Save best model: monitor(max): 0.809099
2022-02-09 08:53:27,101 P42567 INFO --- 8058/8058 batches finished ---
2022-02-09 08:53:27,142 P42567 INFO Train loss: 0.453081
2022-02-09 08:53:27,143 P42567 INFO ************ Epoch=7 end ************
2022-02-09 09:00:24,329 P42567 INFO [Metrics] AUC: 0.809317 - logloss: 0.442312
2022-02-09 09:00:24,331 P42567 INFO Save best model: monitor(max): 0.809317
2022-02-09 09:00:24,431 P42567 INFO --- 8058/8058 batches finished ---
2022-02-09 09:00:24,472 P42567 INFO Train loss: 0.452867
2022-02-09 09:00:24,472 P42567 INFO ************ Epoch=8 end ************
2022-02-09 09:07:19,850 P42567 INFO [Metrics] AUC: 0.809491 - logloss: 0.442103
2022-02-09 09:07:19,851 P42567 INFO Save best model: monitor(max): 0.809491
2022-02-09 09:07:19,954 P42567 INFO --- 8058/8058 batches finished ---
2022-02-09 09:07:19,997 P42567 INFO Train loss: 0.452661
2022-02-09 09:07:19,997 P42567 INFO ************ Epoch=9 end ************
2022-02-09 09:14:13,564 P42567 INFO [Metrics] AUC: 0.809643 - logloss: 0.441985
2022-02-09 09:14:13,566 P42567 INFO Save best model: monitor(max): 0.809643
2022-02-09 09:14:13,668 P42567 INFO --- 8058/8058 batches finished ---
2022-02-09 09:14:13,714 P42567 INFO Train loss: 0.452521
2022-02-09 09:14:13,714 P42567 INFO ************ Epoch=10 end ************
2022-02-09 09:21:11,880 P42567 INFO [Metrics] AUC: 0.809822 - logloss: 0.441964
2022-02-09 09:21:11,882 P42567 INFO Save best model: monitor(max): 0.809822
2022-02-09 09:21:11,984 P42567 INFO --- 8058/8058 batches finished ---
2022-02-09 09:21:12,025 P42567 INFO Train loss: 0.452354
2022-02-09 09:21:12,025 P42567 INFO ************ Epoch=11 end ************
2022-02-09 09:28:07,689 P42567 INFO [Metrics] AUC: 0.809632 - logloss: 0.441955
2022-02-09 09:28:07,691 P42567 INFO Monitor(max) STOP: 0.809632 !
2022-02-09 09:28:07,691 P42567 INFO Reduce learning rate on plateau: 0.000100
2022-02-09 09:28:07,691 P42567 INFO --- 8058/8058 batches finished ---
2022-02-09 09:28:07,743 P42567 INFO Train loss: 0.452221
2022-02-09 09:28:07,743 P42567 INFO ************ Epoch=12 end ************
2022-02-09 09:35:06,122 P42567 INFO [Metrics] AUC: 0.812987 - logloss: 0.439029
2022-02-09 09:35:06,124 P42567 INFO Save best model: monitor(max): 0.812987
2022-02-09 09:35:06,216 P42567 INFO --- 8058/8058 batches finished ---
2022-02-09 09:35:06,268 P42567 INFO Train loss: 0.441680
2022-02-09 09:35:06,268 P42567 INFO ************ Epoch=13 end ************
2022-02-09 09:42:02,807 P42567 INFO [Metrics] AUC: 0.813457 - logloss: 0.438514
2022-02-09 09:42:02,808 P42567 INFO Save best model: monitor(max): 0.813457
2022-02-09 09:42:02,914 P42567 INFO --- 8058/8058 batches finished ---
2022-02-09 09:42:02,963 P42567 INFO Train loss: 0.437604
2022-02-09 09:42:02,963 P42567 INFO ************ Epoch=14 end ************
2022-02-09 09:49:01,764 P42567 INFO [Metrics] AUC: 0.813607 - logloss: 0.438447
2022-02-09 09:49:01,765 P42567 INFO Save best model: monitor(max): 0.813607
2022-02-09 09:49:01,871 P42567 INFO --- 8058/8058 batches finished ---
2022-02-09 09:49:01,911 P42567 INFO Train loss: 0.435759
2022-02-09 09:49:01,912 P42567 INFO ************ Epoch=15 end ************
2022-02-09 09:55:58,043 P42567 INFO [Metrics] AUC: 0.813595 - logloss: 0.438474
2022-02-09 09:55:58,045 P42567 INFO Monitor(max) STOP: 0.813595 !
2022-02-09 09:55:58,045 P42567 INFO Reduce learning rate on plateau: 0.000010
2022-02-09 09:55:58,045 P42567 INFO --- 8058/8058 batches finished ---
2022-02-09 09:55:58,095 P42567 INFO Train loss: 0.434354
2022-02-09 09:55:58,095 P42567 INFO ************ Epoch=16 end ************
2022-02-09 10:02:48,555 P42567 INFO [Metrics] AUC: 0.813125 - logloss: 0.439168
2022-02-09 10:02:48,557 P42567 INFO Monitor(max) STOP: 0.813125 !
2022-02-09 10:02:48,557 P42567 INFO Reduce learning rate on plateau: 0.000001
2022-02-09 10:02:48,557 P42567 INFO Early stopping at epoch=17
2022-02-09 10:02:48,557 P42567 INFO --- 8058/8058 batches finished ---
2022-02-09 10:02:48,604 P42567 INFO Train loss: 0.430058
2022-02-09 10:02:48,604 P42567 INFO Training finished.
2022-02-09 10:02:48,605 P42567 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/DeepIM_criteo_x1/criteo_x1_7b681156/DeepIM_criteo_x1_001_6de9e773.model
2022-02-09 10:02:48,675 P42567 INFO ****** Validation evaluation ******
2022-02-09 10:03:13,506 P42567 INFO [Metrics] AUC: 0.813607 - logloss: 0.438447
2022-02-09 10:03:13,591 P42567 INFO ******** Test evaluation ********
2022-02-09 10:03:13,591 P42567 INFO Loading data...
2022-02-09 10:03:13,592 P42567 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-02-09 10:03:14,419 P42567 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-02-09 10:03:14,419 P42567 INFO Loading test data done.
2022-02-09 10:03:29,265 P42567 INFO [Metrics] AUC: 0.813927 - logloss: 0.437959

```
