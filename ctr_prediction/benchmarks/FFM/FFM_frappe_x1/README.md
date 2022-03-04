## FFM_frappe_x1

A hands-on guide to run the FFM model on the Frappe_x1 dataset.

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
Dataset ID: [Frappe_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Frappe/README.md#Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [FFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FFM_frappe_x1_tuner_config_01](./FFM_frappe_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FFM_frappe_x1
    nohup python run_expid.py --config ./FFM_frappe_x1_tuner_config_01 --expid FFM_frappe_x1_007_b8a83e47 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.978620 | 0.186032  |


### Logs
```python
2021-03-15 23:40:34,377 P35819 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_7f91d67a",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FFM",
    "model_id": "FFM_frappe_x1_007_1e29f4c0",
    "model_root": "./Frappe/FFM_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-06",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Frappe/Frappe_x1/test.csv",
    "train_data": "../data/Frappe/Frappe_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2021-03-15 23:40:34,377 P35819 INFO Set up feature encoder...
2021-03-15 23:40:34,377 P35819 INFO Load feature_encoder from pickle: ../data/Frappe/frappe_x1_7f91d67a/feature_encoder.pkl
2021-03-15 23:40:34,422 P35819 INFO Total number of parameters: 490400.
2021-03-15 23:40:34,422 P35819 INFO Loading data...
2021-03-15 23:40:34,425 P35819 INFO Loading data from h5: ../data/Frappe/frappe_x1_7f91d67a/train.h5
2021-03-15 23:40:34,437 P35819 INFO Loading data from h5: ../data/Frappe/frappe_x1_7f91d67a/valid.h5
2021-03-15 23:40:34,441 P35819 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2021-03-15 23:40:34,442 P35819 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2021-03-15 23:40:34,442 P35819 INFO Loading train data done.
2021-03-15 23:40:37,577 P35819 INFO Start training: 50 batches/epoch
2021-03-15 23:40:37,577 P35819 INFO ************ Epoch=1 start ************
2021-03-15 23:41:00,948 P35819 INFO [Metrics] AUC: 0.920168 - logloss: 0.566003
2021-03-15 23:41:00,949 P35819 INFO Save best model: monitor(max): 0.920168
2021-03-15 23:41:00,961 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:41:01,014 P35819 INFO Train loss: 0.634316
2021-03-15 23:41:01,014 P35819 INFO ************ Epoch=1 end ************
2021-03-15 23:41:24,269 P35819 INFO [Metrics] AUC: 0.931866 - logloss: 0.390165
2021-03-15 23:41:24,271 P35819 INFO Save best model: monitor(max): 0.931866
2021-03-15 23:41:24,283 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:41:24,341 P35819 INFO Train loss: 0.479657
2021-03-15 23:41:24,341 P35819 INFO ************ Epoch=2 end ************
2021-03-15 23:41:47,657 P35819 INFO [Metrics] AUC: 0.935397 - logloss: 0.305626
2021-03-15 23:41:47,658 P35819 INFO Save best model: monitor(max): 0.935397
2021-03-15 23:41:47,671 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:41:47,742 P35819 INFO Train loss: 0.335937
2021-03-15 23:41:47,743 P35819 INFO ************ Epoch=3 end ************
2021-03-15 23:42:11,042 P35819 INFO [Metrics] AUC: 0.938628 - logloss: 0.284615
2021-03-15 23:42:11,043 P35819 INFO Save best model: monitor(max): 0.938628
2021-03-15 23:42:11,055 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:42:11,111 P35819 INFO Train loss: 0.286447
2021-03-15 23:42:11,111 P35819 INFO ************ Epoch=4 end ************
2021-03-15 23:42:34,354 P35819 INFO [Metrics] AUC: 0.940596 - logloss: 0.277226
2021-03-15 23:42:34,354 P35819 INFO Save best model: monitor(max): 0.940596
2021-03-15 23:42:34,366 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:42:34,440 P35819 INFO Train loss: 0.270624
2021-03-15 23:42:34,441 P35819 INFO ************ Epoch=5 end ************
2021-03-15 23:42:57,759 P35819 INFO [Metrics] AUC: 0.941724 - logloss: 0.273391
2021-03-15 23:42:57,760 P35819 INFO Save best model: monitor(max): 0.941724
2021-03-15 23:42:57,772 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:42:57,830 P35819 INFO Train loss: 0.262896
2021-03-15 23:42:57,830 P35819 INFO ************ Epoch=6 end ************
2021-03-15 23:43:21,010 P35819 INFO [Metrics] AUC: 0.942825 - logloss: 0.270482
2021-03-15 23:43:21,010 P35819 INFO Save best model: monitor(max): 0.942825
2021-03-15 23:43:21,018 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:43:21,075 P35819 INFO Train loss: 0.257308
2021-03-15 23:43:21,075 P35819 INFO ************ Epoch=7 end ************
2021-03-15 23:43:44,326 P35819 INFO [Metrics] AUC: 0.943768 - logloss: 0.267881
2021-03-15 23:43:44,327 P35819 INFO Save best model: monitor(max): 0.943768
2021-03-15 23:43:44,339 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:43:44,380 P35819 INFO Train loss: 0.252608
2021-03-15 23:43:44,381 P35819 INFO ************ Epoch=8 end ************
2021-03-15 23:44:07,670 P35819 INFO [Metrics] AUC: 0.944894 - logloss: 0.265145
2021-03-15 23:44:07,671 P35819 INFO Save best model: monitor(max): 0.944894
2021-03-15 23:44:07,679 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:44:07,722 P35819 INFO Train loss: 0.248501
2021-03-15 23:44:07,722 P35819 INFO ************ Epoch=9 end ************
2021-03-15 23:44:30,943 P35819 INFO [Metrics] AUC: 0.946070 - logloss: 0.262360
2021-03-15 23:44:30,944 P35819 INFO Save best model: monitor(max): 0.946070
2021-03-15 23:44:30,956 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:44:31,011 P35819 INFO Train loss: 0.243660
2021-03-15 23:44:31,011 P35819 INFO ************ Epoch=10 end ************
2021-03-15 23:44:54,229 P35819 INFO [Metrics] AUC: 0.947353 - logloss: 0.259254
2021-03-15 23:44:54,230 P35819 INFO Save best model: monitor(max): 0.947353
2021-03-15 23:44:54,239 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:44:54,281 P35819 INFO Train loss: 0.239445
2021-03-15 23:44:54,281 P35819 INFO ************ Epoch=11 end ************
2021-03-15 23:45:17,613 P35819 INFO [Metrics] AUC: 0.948523 - logloss: 0.256501
2021-03-15 23:45:17,614 P35819 INFO Save best model: monitor(max): 0.948523
2021-03-15 23:45:17,622 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:45:17,683 P35819 INFO Train loss: 0.234844
2021-03-15 23:45:17,684 P35819 INFO ************ Epoch=12 end ************
2021-03-15 23:45:40,920 P35819 INFO [Metrics] AUC: 0.949706 - logloss: 0.253646
2021-03-15 23:45:40,921 P35819 INFO Save best model: monitor(max): 0.949706
2021-03-15 23:45:40,933 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:45:40,979 P35819 INFO Train loss: 0.230562
2021-03-15 23:45:40,979 P35819 INFO ************ Epoch=13 end ************
2021-03-15 23:46:03,989 P35819 INFO [Metrics] AUC: 0.950903 - logloss: 0.250779
2021-03-15 23:46:03,990 P35819 INFO Save best model: monitor(max): 0.950903
2021-03-15 23:46:04,002 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:46:04,051 P35819 INFO Train loss: 0.226006
2021-03-15 23:46:04,051 P35819 INFO ************ Epoch=14 end ************
2021-03-15 23:46:27,322 P35819 INFO [Metrics] AUC: 0.952049 - logloss: 0.248036
2021-03-15 23:46:27,323 P35819 INFO Save best model: monitor(max): 0.952049
2021-03-15 23:46:27,337 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:46:27,382 P35819 INFO Train loss: 0.221502
2021-03-15 23:46:27,383 P35819 INFO ************ Epoch=15 end ************
2021-03-15 23:46:50,644 P35819 INFO [Metrics] AUC: 0.953141 - logloss: 0.245356
2021-03-15 23:46:50,644 P35819 INFO Save best model: monitor(max): 0.953141
2021-03-15 23:46:50,656 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:46:50,701 P35819 INFO Train loss: 0.217480
2021-03-15 23:46:50,702 P35819 INFO ************ Epoch=16 end ************
2021-03-15 23:47:13,977 P35819 INFO [Metrics] AUC: 0.954241 - logloss: 0.242728
2021-03-15 23:47:13,978 P35819 INFO Save best model: monitor(max): 0.954241
2021-03-15 23:47:13,990 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:47:14,049 P35819 INFO Train loss: 0.212428
2021-03-15 23:47:14,050 P35819 INFO ************ Epoch=17 end ************
2021-03-15 23:47:37,355 P35819 INFO [Metrics] AUC: 0.955317 - logloss: 0.240084
2021-03-15 23:47:37,355 P35819 INFO Save best model: monitor(max): 0.955317
2021-03-15 23:47:37,364 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:47:37,405 P35819 INFO Train loss: 0.207942
2021-03-15 23:47:37,406 P35819 INFO ************ Epoch=18 end ************
2021-03-15 23:48:00,735 P35819 INFO [Metrics] AUC: 0.956327 - logloss: 0.237599
2021-03-15 23:48:00,736 P35819 INFO Save best model: monitor(max): 0.956327
2021-03-15 23:48:00,749 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:48:00,796 P35819 INFO Train loss: 0.203592
2021-03-15 23:48:00,796 P35819 INFO ************ Epoch=19 end ************
2021-03-15 23:48:23,901 P35819 INFO [Metrics] AUC: 0.957324 - logloss: 0.235034
2021-03-15 23:48:23,901 P35819 INFO Save best model: monitor(max): 0.957324
2021-03-15 23:48:23,909 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:48:23,975 P35819 INFO Train loss: 0.199036
2021-03-15 23:48:23,975 P35819 INFO ************ Epoch=20 end ************
2021-03-15 23:48:47,303 P35819 INFO [Metrics] AUC: 0.958290 - logloss: 0.232498
2021-03-15 23:48:47,304 P35819 INFO Save best model: monitor(max): 0.958290
2021-03-15 23:48:47,316 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:48:47,362 P35819 INFO Train loss: 0.194892
2021-03-15 23:48:47,363 P35819 INFO ************ Epoch=21 end ************
2021-03-15 23:49:10,633 P35819 INFO [Metrics] AUC: 0.959285 - logloss: 0.229938
2021-03-15 23:49:10,634 P35819 INFO Save best model: monitor(max): 0.959285
2021-03-15 23:49:10,646 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:49:10,724 P35819 INFO Train loss: 0.189851
2021-03-15 23:49:10,724 P35819 INFO ************ Epoch=22 end ************
2021-03-15 23:49:33,920 P35819 INFO [Metrics] AUC: 0.960101 - logloss: 0.227793
2021-03-15 23:49:33,920 P35819 INFO Save best model: monitor(max): 0.960101
2021-03-15 23:49:33,932 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:49:34,011 P35819 INFO Train loss: 0.185458
2021-03-15 23:49:34,011 P35819 INFO ************ Epoch=23 end ************
2021-03-15 23:49:57,163 P35819 INFO [Metrics] AUC: 0.960993 - logloss: 0.225339
2021-03-15 23:49:57,164 P35819 INFO Save best model: monitor(max): 0.960993
2021-03-15 23:49:57,176 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:49:57,249 P35819 INFO Train loss: 0.181118
2021-03-15 23:49:57,249 P35819 INFO ************ Epoch=24 end ************
2021-03-15 23:50:20,489 P35819 INFO [Metrics] AUC: 0.961772 - logloss: 0.223334
2021-03-15 23:50:20,490 P35819 INFO Save best model: monitor(max): 0.961772
2021-03-15 23:50:20,498 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:50:20,571 P35819 INFO Train loss: 0.177035
2021-03-15 23:50:20,571 P35819 INFO ************ Epoch=25 end ************
2021-03-15 23:50:43,807 P35819 INFO [Metrics] AUC: 0.962601 - logloss: 0.220965
2021-03-15 23:50:43,807 P35819 INFO Save best model: monitor(max): 0.962601
2021-03-15 23:50:43,816 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:50:43,855 P35819 INFO Train loss: 0.172199
2021-03-15 23:50:43,856 P35819 INFO ************ Epoch=26 end ************
2021-03-15 23:51:07,164 P35819 INFO [Metrics] AUC: 0.963414 - logloss: 0.218780
2021-03-15 23:51:07,165 P35819 INFO Save best model: monitor(max): 0.963414
2021-03-15 23:51:07,180 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:51:07,270 P35819 INFO Train loss: 0.167882
2021-03-15 23:51:07,271 P35819 INFO ************ Epoch=27 end ************
2021-03-15 23:51:30,139 P35819 INFO [Metrics] AUC: 0.964080 - logloss: 0.216841
2021-03-15 23:51:30,139 P35819 INFO Save best model: monitor(max): 0.964080
2021-03-15 23:51:30,149 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:51:30,219 P35819 INFO Train loss: 0.163660
2021-03-15 23:51:30,219 P35819 INFO ************ Epoch=28 end ************
2021-03-15 23:51:53,476 P35819 INFO [Metrics] AUC: 0.964826 - logloss: 0.214584
2021-03-15 23:51:53,477 P35819 INFO Save best model: monitor(max): 0.964826
2021-03-15 23:51:53,485 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:51:53,533 P35819 INFO Train loss: 0.159375
2021-03-15 23:51:53,534 P35819 INFO ************ Epoch=29 end ************
2021-03-15 23:52:16,750 P35819 INFO [Metrics] AUC: 0.965507 - logloss: 0.212650
2021-03-15 23:52:16,751 P35819 INFO Save best model: monitor(max): 0.965507
2021-03-15 23:52:16,759 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:52:16,816 P35819 INFO Train loss: 0.155068
2021-03-15 23:52:16,817 P35819 INFO ************ Epoch=30 end ************
2021-03-15 23:52:40,095 P35819 INFO [Metrics] AUC: 0.966198 - logloss: 0.210402
2021-03-15 23:52:40,096 P35819 INFO Save best model: monitor(max): 0.966198
2021-03-15 23:52:40,104 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:52:40,140 P35819 INFO Train loss: 0.150686
2021-03-15 23:52:40,140 P35819 INFO ************ Epoch=31 end ************
2021-03-15 23:53:03,450 P35819 INFO [Metrics] AUC: 0.966849 - logloss: 0.208379
2021-03-15 23:53:03,451 P35819 INFO Save best model: monitor(max): 0.966849
2021-03-15 23:53:03,463 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:53:03,531 P35819 INFO Train loss: 0.146881
2021-03-15 23:53:03,531 P35819 INFO ************ Epoch=32 end ************
2021-03-15 23:53:26,893 P35819 INFO [Metrics] AUC: 0.967551 - logloss: 0.206147
2021-03-15 23:53:26,894 P35819 INFO Save best model: monitor(max): 0.967551
2021-03-15 23:53:26,906 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:53:26,977 P35819 INFO Train loss: 0.142245
2021-03-15 23:53:26,977 P35819 INFO ************ Epoch=33 end ************
2021-03-15 23:53:50,103 P35819 INFO [Metrics] AUC: 0.968211 - logloss: 0.204090
2021-03-15 23:53:50,104 P35819 INFO Save best model: monitor(max): 0.968211
2021-03-15 23:53:50,112 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:53:50,199 P35819 INFO Train loss: 0.137817
2021-03-15 23:53:50,199 P35819 INFO ************ Epoch=34 end ************
2021-03-15 23:54:13,572 P35819 INFO [Metrics] AUC: 0.968820 - logloss: 0.202093
2021-03-15 23:54:13,572 P35819 INFO Save best model: monitor(max): 0.968820
2021-03-15 23:54:13,584 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:54:13,634 P35819 INFO Train loss: 0.133972
2021-03-15 23:54:13,634 P35819 INFO ************ Epoch=35 end ************
2021-03-15 23:54:36,957 P35819 INFO [Metrics] AUC: 0.969382 - logloss: 0.200139
2021-03-15 23:54:36,958 P35819 INFO Save best model: monitor(max): 0.969382
2021-03-15 23:54:36,970 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:54:37,010 P35819 INFO Train loss: 0.129661
2021-03-15 23:54:37,011 P35819 INFO ************ Epoch=36 end ************
2021-03-15 23:55:00,348 P35819 INFO [Metrics] AUC: 0.969976 - logloss: 0.198252
2021-03-15 23:55:00,349 P35819 INFO Save best model: monitor(max): 0.969976
2021-03-15 23:55:00,357 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:55:00,404 P35819 INFO Train loss: 0.125849
2021-03-15 23:55:00,404 P35819 INFO ************ Epoch=37 end ************
2021-03-15 23:55:23,633 P35819 INFO [Metrics] AUC: 0.970519 - logloss: 0.196425
2021-03-15 23:55:23,634 P35819 INFO Save best model: monitor(max): 0.970519
2021-03-15 23:55:23,642 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:55:23,698 P35819 INFO Train loss: 0.121723
2021-03-15 23:55:23,699 P35819 INFO ************ Epoch=38 end ************
2021-03-15 23:55:46,830 P35819 INFO [Metrics] AUC: 0.971041 - logloss: 0.194674
2021-03-15 23:55:46,831 P35819 INFO Save best model: monitor(max): 0.971041
2021-03-15 23:55:46,843 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:55:46,911 P35819 INFO Train loss: 0.117951
2021-03-15 23:55:46,911 P35819 INFO ************ Epoch=39 end ************
2021-03-15 23:56:10,148 P35819 INFO [Metrics] AUC: 0.971537 - logloss: 0.192984
2021-03-15 23:56:10,148 P35819 INFO Save best model: monitor(max): 0.971537
2021-03-15 23:56:10,160 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:56:10,208 P35819 INFO Train loss: 0.114261
2021-03-15 23:56:10,208 P35819 INFO ************ Epoch=40 end ************
2021-03-15 23:56:33,596 P35819 INFO [Metrics] AUC: 0.972004 - logloss: 0.191613
2021-03-15 23:56:33,597 P35819 INFO Save best model: monitor(max): 0.972004
2021-03-15 23:56:33,609 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:56:33,673 P35819 INFO Train loss: 0.110568
2021-03-15 23:56:33,673 P35819 INFO ************ Epoch=41 end ************
2021-03-15 23:56:57,136 P35819 INFO [Metrics] AUC: 0.972432 - logloss: 0.190091
2021-03-15 23:56:57,137 P35819 INFO Save best model: monitor(max): 0.972432
2021-03-15 23:56:57,149 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:56:57,232 P35819 INFO Train loss: 0.106619
2021-03-15 23:56:57,233 P35819 INFO ************ Epoch=42 end ************
2021-03-15 23:57:20,815 P35819 INFO [Metrics] AUC: 0.972834 - logloss: 0.188806
2021-03-15 23:57:20,816 P35819 INFO Save best model: monitor(max): 0.972834
2021-03-15 23:57:20,828 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:57:20,889 P35819 INFO Train loss: 0.103495
2021-03-15 23:57:20,889 P35819 INFO ************ Epoch=43 end ************
2021-03-15 23:57:44,389 P35819 INFO [Metrics] AUC: 0.973272 - logloss: 0.187528
2021-03-15 23:57:44,390 P35819 INFO Save best model: monitor(max): 0.973272
2021-03-15 23:57:44,402 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:57:44,445 P35819 INFO Train loss: 0.100365
2021-03-15 23:57:44,445 P35819 INFO ************ Epoch=44 end ************
2021-03-15 23:58:07,951 P35819 INFO [Metrics] AUC: 0.973637 - logloss: 0.186141
2021-03-15 23:58:07,952 P35819 INFO Save best model: monitor(max): 0.973637
2021-03-15 23:58:07,964 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:58:08,035 P35819 INFO Train loss: 0.097138
2021-03-15 23:58:08,035 P35819 INFO ************ Epoch=45 end ************
2021-03-15 23:58:31,622 P35819 INFO [Metrics] AUC: 0.973975 - logloss: 0.185171
2021-03-15 23:58:31,623 P35819 INFO Save best model: monitor(max): 0.973975
2021-03-15 23:58:31,631 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:58:31,689 P35819 INFO Train loss: 0.094156
2021-03-15 23:58:31,689 P35819 INFO ************ Epoch=46 end ************
2021-03-15 23:58:55,309 P35819 INFO [Metrics] AUC: 0.974334 - logloss: 0.183878
2021-03-15 23:58:55,310 P35819 INFO Save best model: monitor(max): 0.974334
2021-03-15 23:58:55,322 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:58:55,386 P35819 INFO Train loss: 0.090924
2021-03-15 23:58:55,387 P35819 INFO ************ Epoch=47 end ************
2021-03-15 23:59:19,071 P35819 INFO [Metrics] AUC: 0.974636 - logloss: 0.183094
2021-03-15 23:59:19,072 P35819 INFO Save best model: monitor(max): 0.974636
2021-03-15 23:59:19,080 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:59:19,156 P35819 INFO Train loss: 0.088228
2021-03-15 23:59:19,157 P35819 INFO ************ Epoch=48 end ************
2021-03-15 23:59:42,812 P35819 INFO [Metrics] AUC: 0.974965 - logloss: 0.182178
2021-03-15 23:59:42,813 P35819 INFO Save best model: monitor(max): 0.974965
2021-03-15 23:59:42,821 P35819 INFO --- 50/50 batches finished ---
2021-03-15 23:59:42,896 P35819 INFO Train loss: 0.085503
2021-03-15 23:59:42,896 P35819 INFO ************ Epoch=49 end ************
2021-03-16 00:00:06,505 P35819 INFO [Metrics] AUC: 0.975226 - logloss: 0.181446
2021-03-16 00:00:06,506 P35819 INFO Save best model: monitor(max): 0.975226
2021-03-16 00:00:06,515 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:00:06,574 P35819 INFO Train loss: 0.082552
2021-03-16 00:00:06,574 P35819 INFO ************ Epoch=50 end ************
2021-03-16 00:00:30,169 P35819 INFO [Metrics] AUC: 0.975518 - logloss: 0.180914
2021-03-16 00:00:30,170 P35819 INFO Save best model: monitor(max): 0.975518
2021-03-16 00:00:30,178 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:00:30,224 P35819 INFO Train loss: 0.080249
2021-03-16 00:00:30,224 P35819 INFO ************ Epoch=51 end ************
2021-03-16 00:00:53,907 P35819 INFO [Metrics] AUC: 0.975802 - logloss: 0.179967
2021-03-16 00:00:53,907 P35819 INFO Save best model: monitor(max): 0.975802
2021-03-16 00:00:53,920 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:00:53,955 P35819 INFO Train loss: 0.077577
2021-03-16 00:00:53,955 P35819 INFO ************ Epoch=52 end ************
2021-03-16 00:01:17,642 P35819 INFO [Metrics] AUC: 0.976068 - logloss: 0.179343
2021-03-16 00:01:17,643 P35819 INFO Save best model: monitor(max): 0.976068
2021-03-16 00:01:17,655 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:01:17,720 P35819 INFO Train loss: 0.075251
2021-03-16 00:01:17,720 P35819 INFO ************ Epoch=53 end ************
2021-03-16 00:01:41,349 P35819 INFO [Metrics] AUC: 0.976266 - logloss: 0.178914
2021-03-16 00:01:41,350 P35819 INFO Save best model: monitor(max): 0.976266
2021-03-16 00:01:41,363 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:01:41,431 P35819 INFO Train loss: 0.072901
2021-03-16 00:01:41,432 P35819 INFO ************ Epoch=54 end ************
2021-03-16 00:02:05,083 P35819 INFO [Metrics] AUC: 0.976477 - logloss: 0.178494
2021-03-16 00:02:05,084 P35819 INFO Save best model: monitor(max): 0.976477
2021-03-16 00:02:05,096 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:02:05,159 P35819 INFO Train loss: 0.070662
2021-03-16 00:02:05,160 P35819 INFO ************ Epoch=55 end ************
2021-03-16 00:02:29,064 P35819 INFO [Metrics] AUC: 0.976694 - logloss: 0.177885
2021-03-16 00:02:29,064 P35819 INFO Save best model: monitor(max): 0.976694
2021-03-16 00:02:29,073 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:02:29,138 P35819 INFO Train loss: 0.068590
2021-03-16 00:02:29,139 P35819 INFO ************ Epoch=56 end ************
2021-03-16 00:02:52,734 P35819 INFO [Metrics] AUC: 0.976898 - logloss: 0.177569
2021-03-16 00:02:52,735 P35819 INFO Save best model: monitor(max): 0.976898
2021-03-16 00:02:52,743 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:02:52,781 P35819 INFO Train loss: 0.066451
2021-03-16 00:02:52,782 P35819 INFO ************ Epoch=57 end ************
2021-03-16 00:03:16,477 P35819 INFO [Metrics] AUC: 0.977097 - logloss: 0.177222
2021-03-16 00:03:16,478 P35819 INFO Save best model: monitor(max): 0.977097
2021-03-16 00:03:16,486 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:03:16,534 P35819 INFO Train loss: 0.064519
2021-03-16 00:03:16,534 P35819 INFO ************ Epoch=58 end ************
2021-03-16 00:03:40,265 P35819 INFO [Metrics] AUC: 0.977228 - logloss: 0.177022
2021-03-16 00:03:40,266 P35819 INFO Save best model: monitor(max): 0.977228
2021-03-16 00:03:40,288 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:03:40,345 P35819 INFO Train loss: 0.062581
2021-03-16 00:03:40,345 P35819 INFO ************ Epoch=59 end ************
2021-03-16 00:04:03,938 P35819 INFO [Metrics] AUC: 0.977410 - logloss: 0.176941
2021-03-16 00:04:03,938 P35819 INFO Save best model: monitor(max): 0.977410
2021-03-16 00:04:03,951 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:04:04,055 P35819 INFO Train loss: 0.060756
2021-03-16 00:04:04,055 P35819 INFO ************ Epoch=60 end ************
2021-03-16 00:04:27,727 P35819 INFO [Metrics] AUC: 0.977564 - logloss: 0.176888
2021-03-16 00:04:27,728 P35819 INFO Save best model: monitor(max): 0.977564
2021-03-16 00:04:27,736 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:04:27,828 P35819 INFO Train loss: 0.059080
2021-03-16 00:04:27,829 P35819 INFO ************ Epoch=61 end ************
2021-03-16 00:04:51,383 P35819 INFO [Metrics] AUC: 0.977710 - logloss: 0.176734
2021-03-16 00:04:51,384 P35819 INFO Save best model: monitor(max): 0.977710
2021-03-16 00:04:51,396 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:04:51,455 P35819 INFO Train loss: 0.057438
2021-03-16 00:04:51,455 P35819 INFO ************ Epoch=62 end ************
2021-03-16 00:05:14,996 P35819 INFO [Metrics] AUC: 0.977848 - logloss: 0.176627
2021-03-16 00:05:14,997 P35819 INFO Save best model: monitor(max): 0.977848
2021-03-16 00:05:15,009 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:05:15,060 P35819 INFO Train loss: 0.055910
2021-03-16 00:05:15,060 P35819 INFO ************ Epoch=63 end ************
2021-03-16 00:05:38,737 P35819 INFO [Metrics] AUC: 0.977961 - logloss: 0.176757
2021-03-16 00:05:38,738 P35819 INFO Save best model: monitor(max): 0.977961
2021-03-16 00:05:38,767 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:05:38,830 P35819 INFO Train loss: 0.054351
2021-03-16 00:05:38,831 P35819 INFO ************ Epoch=64 end ************
2021-03-16 00:06:02,483 P35819 INFO [Metrics] AUC: 0.978078 - logloss: 0.176895
2021-03-16 00:06:02,484 P35819 INFO Save best model: monitor(max): 0.978078
2021-03-16 00:06:02,496 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:06:02,540 P35819 INFO Train loss: 0.053003
2021-03-16 00:06:02,540 P35819 INFO ************ Epoch=65 end ************
2021-03-16 00:06:26,156 P35819 INFO [Metrics] AUC: 0.978163 - logloss: 0.177120
2021-03-16 00:06:26,157 P35819 INFO Save best model: monitor(max): 0.978163
2021-03-16 00:06:26,165 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:06:26,235 P35819 INFO Train loss: 0.051635
2021-03-16 00:06:26,235 P35819 INFO ************ Epoch=66 end ************
2021-03-16 00:06:49,659 P35819 INFO [Metrics] AUC: 0.978268 - logloss: 0.177446
2021-03-16 00:06:49,659 P35819 INFO Save best model: monitor(max): 0.978268
2021-03-16 00:06:49,672 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:06:49,714 P35819 INFO Train loss: 0.050388
2021-03-16 00:06:49,714 P35819 INFO ************ Epoch=67 end ************
2021-03-16 00:07:13,258 P35819 INFO [Metrics] AUC: 0.978355 - logloss: 0.177500
2021-03-16 00:07:13,259 P35819 INFO Save best model: monitor(max): 0.978355
2021-03-16 00:07:13,271 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:07:13,324 P35819 INFO Train loss: 0.048874
2021-03-16 00:07:13,324 P35819 INFO ************ Epoch=68 end ************
2021-03-16 00:07:36,740 P35819 INFO [Metrics] AUC: 0.978425 - logloss: 0.177841
2021-03-16 00:07:36,741 P35819 INFO Save best model: monitor(max): 0.978425
2021-03-16 00:07:36,752 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:07:36,787 P35819 INFO Train loss: 0.047582
2021-03-16 00:07:36,788 P35819 INFO ************ Epoch=69 end ************
2021-03-16 00:08:00,339 P35819 INFO [Metrics] AUC: 0.978527 - logloss: 0.178061
2021-03-16 00:08:00,340 P35819 INFO Save best model: monitor(max): 0.978527
2021-03-16 00:08:00,352 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:08:00,387 P35819 INFO Train loss: 0.046532
2021-03-16 00:08:00,387 P35819 INFO ************ Epoch=70 end ************
2021-03-16 00:08:23,938 P35819 INFO [Metrics] AUC: 0.978598 - logloss: 0.178091
2021-03-16 00:08:23,939 P35819 INFO Save best model: monitor(max): 0.978598
2021-03-16 00:08:23,951 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:08:23,988 P35819 INFO Train loss: 0.045310
2021-03-16 00:08:23,988 P35819 INFO ************ Epoch=71 end ************
2021-03-16 00:08:47,538 P35819 INFO [Metrics] AUC: 0.978669 - logloss: 0.178601
2021-03-16 00:08:47,539 P35819 INFO Save best model: monitor(max): 0.978669
2021-03-16 00:08:47,547 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:08:47,583 P35819 INFO Train loss: 0.044286
2021-03-16 00:08:47,583 P35819 INFO ************ Epoch=72 end ************
2021-03-16 00:09:10,814 P35819 INFO [Metrics] AUC: 0.978710 - logloss: 0.178965
2021-03-16 00:09:10,815 P35819 INFO Save best model: monitor(max): 0.978710
2021-03-16 00:09:10,827 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:09:10,871 P35819 INFO Train loss: 0.043228
2021-03-16 00:09:10,872 P35819 INFO ************ Epoch=73 end ************
2021-03-16 00:09:33,962 P35819 INFO [Metrics] AUC: 0.978775 - logloss: 0.179502
2021-03-16 00:09:33,963 P35819 INFO Save best model: monitor(max): 0.978775
2021-03-16 00:09:33,975 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:09:34,019 P35819 INFO Train loss: 0.042347
2021-03-16 00:09:34,020 P35819 INFO ************ Epoch=74 end ************
2021-03-16 00:09:56,777 P35819 INFO [Metrics] AUC: 0.978807 - logloss: 0.179842
2021-03-16 00:09:56,778 P35819 INFO Save best model: monitor(max): 0.978807
2021-03-16 00:09:56,786 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:09:56,821 P35819 INFO Train loss: 0.041186
2021-03-16 00:09:56,821 P35819 INFO ************ Epoch=75 end ************
2021-03-16 00:10:19,335 P35819 INFO [Metrics] AUC: 0.978867 - logloss: 0.180467
2021-03-16 00:10:19,335 P35819 INFO Save best model: monitor(max): 0.978867
2021-03-16 00:10:19,344 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:10:19,377 P35819 INFO Train loss: 0.040147
2021-03-16 00:10:19,377 P35819 INFO ************ Epoch=76 end ************
2021-03-16 00:10:41,922 P35819 INFO [Metrics] AUC: 0.978885 - logloss: 0.180705
2021-03-16 00:10:41,923 P35819 INFO Save best model: monitor(max): 0.978885
2021-03-16 00:10:41,931 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:10:41,972 P35819 INFO Train loss: 0.039301
2021-03-16 00:10:41,973 P35819 INFO ************ Epoch=77 end ************
2021-03-16 00:11:04,208 P35819 INFO [Metrics] AUC: 0.978929 - logloss: 0.181093
2021-03-16 00:11:04,208 P35819 INFO Save best model: monitor(max): 0.978929
2021-03-16 00:11:04,250 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:11:04,292 P35819 INFO Train loss: 0.038409
2021-03-16 00:11:04,293 P35819 INFO ************ Epoch=78 end ************
2021-03-16 00:11:27,104 P35819 INFO [Metrics] AUC: 0.978974 - logloss: 0.181693
2021-03-16 00:11:27,105 P35819 INFO Save best model: monitor(max): 0.978974
2021-03-16 00:11:27,113 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:11:27,154 P35819 INFO Train loss: 0.037635
2021-03-16 00:11:27,155 P35819 INFO ************ Epoch=79 end ************
2021-03-16 00:11:49,974 P35819 INFO [Metrics] AUC: 0.978997 - logloss: 0.182241
2021-03-16 00:11:49,974 P35819 INFO Save best model: monitor(max): 0.978997
2021-03-16 00:11:49,986 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:11:50,021 P35819 INFO Train loss: 0.036851
2021-03-16 00:11:50,022 P35819 INFO ************ Epoch=80 end ************
2021-03-16 00:12:12,809 P35819 INFO [Metrics] AUC: 0.979020 - logloss: 0.182910
2021-03-16 00:12:12,810 P35819 INFO Save best model: monitor(max): 0.979020
2021-03-16 00:12:12,822 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:12:12,864 P35819 INFO Train loss: 0.035974
2021-03-16 00:12:12,865 P35819 INFO ************ Epoch=81 end ************
2021-03-16 00:12:35,611 P35819 INFO [Metrics] AUC: 0.979055 - logloss: 0.183072
2021-03-16 00:12:35,612 P35819 INFO Save best model: monitor(max): 0.979055
2021-03-16 00:12:35,624 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:12:35,673 P35819 INFO Train loss: 0.035215
2021-03-16 00:12:35,673 P35819 INFO ************ Epoch=82 end ************
2021-03-16 00:12:58,503 P35819 INFO [Metrics] AUC: 0.979037 - logloss: 0.183872
2021-03-16 00:12:58,504 P35819 INFO Monitor(max) STOP: 0.979037 !
2021-03-16 00:12:58,504 P35819 INFO Reduce learning rate on plateau: 0.000100
2021-03-16 00:12:58,504 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:12:58,544 P35819 INFO Train loss: 0.034570
2021-03-16 00:12:58,544 P35819 INFO ************ Epoch=83 end ************
2021-03-16 00:13:21,393 P35819 INFO [Metrics] AUC: 0.979044 - logloss: 0.183997
2021-03-16 00:13:21,393 P35819 INFO Monitor(max) STOP: 0.979044 !
2021-03-16 00:13:21,394 P35819 INFO Reduce learning rate on plateau: 0.000010
2021-03-16 00:13:21,394 P35819 INFO Early stopping at epoch=84
2021-03-16 00:13:21,394 P35819 INFO --- 50/50 batches finished ---
2021-03-16 00:13:21,434 P35819 INFO Train loss: 0.033254
2021-03-16 00:13:21,434 P35819 INFO Training finished.
2021-03-16 00:13:21,434 P35819 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Frappe/FFM_frappe_x1/frappe_x1_7f91d67a/FFM_frappe_x1_007_1e29f4c0_model.ckpt
2021-03-16 00:13:21,627 P35819 INFO ****** Train/validation evaluation ******
2021-03-16 00:13:22,073 P35819 INFO [Metrics] AUC: 0.979055 - logloss: 0.183072
2021-03-16 00:13:22,114 P35819 INFO ******** Test evaluation ********
2021-03-16 00:13:22,114 P35819 INFO Loading data...
2021-03-16 00:13:22,115 P35819 INFO Loading data from h5: ../data/Frappe/frappe_x1_7f91d67a/test.h5
2021-03-16 00:13:22,119 P35819 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2021-03-16 00:13:22,119 P35819 INFO Loading test data done.
2021-03-16 00:13:22,413 P35819 INFO [Metrics] AUC: 0.978620 - logloss: 0.186032

```
