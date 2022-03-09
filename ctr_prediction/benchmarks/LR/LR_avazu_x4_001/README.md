## LR_avazu_x4_001

A hands-on guide to run the LR model on the Avazu_x4_001 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

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
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [Avazu_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [LR](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/LR.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [LR_avazu_x4_tuner_config_01](./LR_avazu_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd LR_avazu_x4_001
    nohup python run_expid.py --config ./LR_avazu_x4_tuner_config_01 --expid LR_avazu_x4_005_93043d62 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.381525 | 0.777457  |


### Logs
```python
2020-05-02 23:34:53,553 P11439 INFO {
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'str', 'name': 'id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "2",
    "model": "LR",
    "model_id": "LR_avazu_x4_005_f828a69c",
    "model_root": "./Avazu/LR_avazu/min2/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-08",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x4/test.csv",
    "train_data": "../data/Avazu/Avazu_x4/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x4/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-05-02 23:34:53,554 P11439 INFO Set up feature encoder...
2020-05-02 23:34:53,554 P11439 INFO Load feature_encoder from pickle: ../data/Avazu/avazu_x4_3bbbc4c9/feature_encoder.pkl
2020-05-02 23:34:55,083 P11439 INFO Loading data...
2020-05-02 23:34:55,086 P11439 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-05-02 23:34:57,898 P11439 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-05-02 23:34:59,200 P11439 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-05-02 23:34:59,319 P11439 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-05-02 23:34:59,319 P11439 INFO Loading train data done.
2020-05-02 23:35:02,209 P11439 INFO **** Start training: 3235 batches/epoch ****
2020-05-02 23:40:05,126 P11439 INFO [Metrics] logloss: 0.393119 - AUC: 0.757035
2020-05-02 23:40:05,127 P11439 INFO Save best model: monitor(max): 0.363916
2020-05-02 23:40:05,140 P11439 INFO --- 3235/3235 batches finished ---
2020-05-02 23:40:05,189 P11439 INFO Train loss: 0.403267
2020-05-02 23:40:05,189 P11439 INFO ************ Epoch=1 end ************
2020-05-02 23:45:07,848 P11439 INFO [Metrics] logloss: 0.389312 - AUC: 0.764028
2020-05-02 23:45:07,853 P11439 INFO Save best model: monitor(max): 0.374716
2020-05-02 23:45:07,870 P11439 INFO --- 3235/3235 batches finished ---
2020-05-02 23:45:07,921 P11439 INFO Train loss: 0.387846
2020-05-02 23:45:07,922 P11439 INFO ************ Epoch=2 end ************
2020-05-02 23:50:08,584 P11439 INFO [Metrics] logloss: 0.387043 - AUC: 0.768059
2020-05-02 23:50:08,588 P11439 INFO Save best model: monitor(max): 0.381017
2020-05-02 23:50:08,604 P11439 INFO --- 3235/3235 batches finished ---
2020-05-02 23:50:08,653 P11439 INFO Train loss: 0.382057
2020-05-02 23:50:08,654 P11439 INFO ************ Epoch=3 end ************
2020-05-02 23:55:05,923 P11439 INFO [Metrics] logloss: 0.385525 - AUC: 0.770807
2020-05-02 23:55:05,927 P11439 INFO Save best model: monitor(max): 0.385283
2020-05-02 23:55:05,944 P11439 INFO --- 3235/3235 batches finished ---
2020-05-02 23:55:05,994 P11439 INFO Train loss: 0.377714
2020-05-02 23:55:05,994 P11439 INFO ************ Epoch=4 end ************
2020-05-03 00:00:05,799 P11439 INFO [Metrics] logloss: 0.384396 - AUC: 0.772697
2020-05-03 00:00:05,802 P11439 INFO Save best model: monitor(max): 0.388301
2020-05-03 00:00:05,822 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 00:00:05,874 P11439 INFO Train loss: 0.374262
2020-05-03 00:00:05,874 P11439 INFO ************ Epoch=5 end ************
2020-05-03 00:05:05,835 P11439 INFO [Metrics] logloss: 0.383585 - AUC: 0.774049
2020-05-03 00:05:05,838 P11439 INFO Save best model: monitor(max): 0.390464
2020-05-03 00:05:05,856 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 00:05:05,905 P11439 INFO Train loss: 0.371462
2020-05-03 00:05:05,905 P11439 INFO ************ Epoch=6 end ************
2020-05-03 00:10:06,721 P11439 INFO [Metrics] logloss: 0.383031 - AUC: 0.774999
2020-05-03 00:10:06,725 P11439 INFO Save best model: monitor(max): 0.391968
2020-05-03 00:10:06,742 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 00:10:06,796 P11439 INFO Train loss: 0.369152
2020-05-03 00:10:06,796 P11439 INFO ************ Epoch=7 end ************
2020-05-03 00:15:04,722 P11439 INFO [Metrics] logloss: 0.382582 - AUC: 0.775722
2020-05-03 00:15:04,725 P11439 INFO Save best model: monitor(max): 0.393140
2020-05-03 00:15:04,742 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 00:15:04,792 P11439 INFO Train loss: 0.367224
2020-05-03 00:15:04,792 P11439 INFO ************ Epoch=8 end ************
2020-05-03 00:20:02,584 P11439 INFO [Metrics] logloss: 0.382207 - AUC: 0.776290
2020-05-03 00:20:02,588 P11439 INFO Save best model: monitor(max): 0.394083
2020-05-03 00:20:02,607 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 00:20:02,657 P11439 INFO Train loss: 0.365599
2020-05-03 00:20:02,657 P11439 INFO ************ Epoch=9 end ************
2020-05-03 00:25:04,080 P11439 INFO [Metrics] logloss: 0.382026 - AUC: 0.776603
2020-05-03 00:25:04,084 P11439 INFO Save best model: monitor(max): 0.394577
2020-05-03 00:25:04,102 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 00:25:04,154 P11439 INFO Train loss: 0.364236
2020-05-03 00:25:04,154 P11439 INFO ************ Epoch=10 end ************
2020-05-03 00:30:05,670 P11439 INFO [Metrics] logloss: 0.381793 - AUC: 0.776895
2020-05-03 00:30:05,674 P11439 INFO Save best model: monitor(max): 0.395103
2020-05-03 00:30:05,691 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 00:30:05,741 P11439 INFO Train loss: 0.363069
2020-05-03 00:30:05,742 P11439 INFO ************ Epoch=11 end ************
2020-05-03 00:35:05,874 P11439 INFO [Metrics] logloss: 0.381677 - AUC: 0.777102
2020-05-03 00:35:05,878 P11439 INFO Save best model: monitor(max): 0.395424
2020-05-03 00:35:05,897 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 00:35:05,949 P11439 INFO Train loss: 0.362078
2020-05-03 00:35:05,949 P11439 INFO ************ Epoch=12 end ************
2020-05-03 00:40:05,317 P11439 INFO [Metrics] logloss: 0.381606 - AUC: 0.777197
2020-05-03 00:40:05,321 P11439 INFO Save best model: monitor(max): 0.395591
2020-05-03 00:40:05,340 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 00:40:05,393 P11439 INFO Train loss: 0.361228
2020-05-03 00:40:05,393 P11439 INFO ************ Epoch=13 end ************
2020-05-03 00:45:04,108 P11439 INFO [Metrics] logloss: 0.381563 - AUC: 0.777274
2020-05-03 00:45:04,112 P11439 INFO Save best model: monitor(max): 0.395712
2020-05-03 00:45:04,130 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 00:45:04,183 P11439 INFO Train loss: 0.360484
2020-05-03 00:45:04,183 P11439 INFO ************ Epoch=14 end ************
2020-05-03 00:50:05,411 P11439 INFO [Metrics] logloss: 0.381568 - AUC: 0.777384
2020-05-03 00:50:05,415 P11439 INFO Save best model: monitor(max): 0.395816
2020-05-03 00:50:05,434 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 00:50:05,485 P11439 INFO Train loss: 0.359846
2020-05-03 00:50:05,485 P11439 INFO ************ Epoch=15 end ************
2020-05-03 00:55:09,200 P11439 INFO [Metrics] logloss: 0.381607 - AUC: 0.777384
2020-05-03 00:55:09,204 P11439 INFO Monitor(max) STOP: 0.395777 !
2020-05-03 00:55:09,205 P11439 INFO Reduce learning rate on plateau: 0.000100
2020-05-03 00:55:09,205 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 00:55:09,258 P11439 INFO Train loss: 0.359299
2020-05-03 00:55:09,258 P11439 INFO ************ Epoch=16 end ************
2020-05-03 01:00:11,176 P11439 INFO [Metrics] logloss: 0.381499 - AUC: 0.777451
2020-05-03 01:00:11,179 P11439 INFO Save best model: monitor(max): 0.395951
2020-05-03 01:00:11,196 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 01:00:11,248 P11439 INFO Train loss: 0.357537
2020-05-03 01:00:11,249 P11439 INFO ************ Epoch=17 end ************
2020-05-03 01:05:13,249 P11439 INFO [Metrics] logloss: 0.381492 - AUC: 0.777495
2020-05-03 01:05:13,252 P11439 INFO Save best model: monitor(max): 0.396003
2020-05-03 01:05:13,269 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 01:05:13,321 P11439 INFO Train loss: 0.357470
2020-05-03 01:05:13,321 P11439 INFO ************ Epoch=18 end ************
2020-05-03 01:10:10,822 P11439 INFO [Metrics] logloss: 0.381504 - AUC: 0.777488
2020-05-03 01:10:10,825 P11439 INFO Monitor(max) STOP: 0.395985 !
2020-05-03 01:10:10,825 P11439 INFO Reduce learning rate on plateau: 0.000010
2020-05-03 01:10:10,825 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 01:10:10,875 P11439 INFO Train loss: 0.357430
2020-05-03 01:10:10,875 P11439 INFO ************ Epoch=19 end ************
2020-05-03 01:15:09,611 P11439 INFO [Metrics] logloss: 0.381491 - AUC: 0.777498
2020-05-03 01:15:09,614 P11439 INFO Save best model: monitor(max): 0.396007
2020-05-03 01:15:09,631 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 01:15:09,681 P11439 INFO Train loss: 0.357247
2020-05-03 01:15:09,681 P11439 INFO ************ Epoch=20 end ************
2020-05-03 01:20:09,343 P11439 INFO [Metrics] logloss: 0.381490 - AUC: 0.777502
2020-05-03 01:20:09,346 P11439 INFO Save best model: monitor(max): 0.396012
2020-05-03 01:20:09,363 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 01:20:09,414 P11439 INFO Train loss: 0.357241
2020-05-03 01:20:09,414 P11439 INFO ************ Epoch=21 end ************
2020-05-03 01:25:09,391 P11439 INFO [Metrics] logloss: 0.381490 - AUC: 0.777501
2020-05-03 01:25:09,394 P11439 INFO Monitor(max) STOP: 0.396011 !
2020-05-03 01:25:09,394 P11439 INFO Reduce learning rate on plateau: 0.000001
2020-05-03 01:25:09,394 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 01:25:09,446 P11439 INFO Train loss: 0.357239
2020-05-03 01:25:09,446 P11439 INFO ************ Epoch=22 end ************
2020-05-03 01:30:06,033 P11439 INFO [Metrics] logloss: 0.381490 - AUC: 0.777501
2020-05-03 01:30:06,037 P11439 INFO Monitor(max) STOP: 0.396011 !
2020-05-03 01:30:06,037 P11439 INFO Reduce learning rate on plateau: 0.000001
2020-05-03 01:30:06,037 P11439 INFO Early stopping at epoch=23
2020-05-03 01:30:06,037 P11439 INFO --- 3235/3235 batches finished ---
2020-05-03 01:30:06,085 P11439 INFO Train loss: 0.357217
2020-05-03 01:30:06,085 P11439 INFO Training finished.
2020-05-03 01:30:06,085 P11439 INFO Load best model: /home/XXX/benchmarks/Avazu/LR_avazu/min2/avazu_x4_3bbbc4c9/LR_avazu_x4_005_f828a69c_model.ckpt
2020-05-03 01:30:06,110 P11439 INFO ****** Train/validation evaluation ******
2020-05-03 01:30:32,199 P11439 INFO [Metrics] logloss: 0.381490 - AUC: 0.777502
2020-05-03 01:30:32,316 P11439 INFO ******** Test evaluation ********
2020-05-03 01:30:32,317 P11439 INFO Loading data...
2020-05-03 01:30:32,317 P11439 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-05-03 01:30:32,809 P11439 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-05-03 01:30:32,809 P11439 INFO Loading test data done.
2020-05-03 01:30:59,413 P11439 INFO [Metrics] logloss: 0.381525 - AUC: 0.777457

```
