## EDCN_movielenslatest_x1

A hands-on guide to run the EDCN model on the MovielensLatest_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index

[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs) | [Revision History](#Revision-History)

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
  fuxictr: 1.2.2
  ```

### Dataset

Dataset ID: [MovielensLatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Movielens#MovielensLatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.2](https://github.com/reczoo/FuxiCTR/tree/v1.2.2) for this experiment. See the model code: [EDCN](https://github.com/reczoo/FuxiCTR/blob/v1.2.2/fuxictr/pytorch/models/EDCN.py).

Running steps:

1. Download [FuxiCTR-v1.2.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.2.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [fuxictr_version.py](./fuxictr_version.py#L3) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [EDCN_movielenslatest_x1_tuner_config_02](./EDCN_movielenslatest_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd EDCN_movielenslatest_x1
   nohup python run_expid.py --config ./EDCN_movielenslatest_x1_tuner_config_02 --expid EDCN_movielenslatest_x1_003_b42bf948 --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

| AUC      | logloss  |
|:--------:|:--------:|
| 0.967133 | 0.212202 |

### Logs

```python
2022-06-17 12:12:21,461 P25515 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "bridge_type": "hadamard_product",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "ReLU",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "EDCN",
    "model_id": "EDCN_movielenslatest_x1_003_b42bf948",
    "model_root": "./Movielens/EDCN_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_cross_layers": "2",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "temperature": "1",
    "test_data": "../data/Movielens/MovielensLatest_x1/test.csv",
    "train_data": "../data/Movielens/MovielensLatest_x1/train.csv",
    "use_hdf5": "True",
    "use_regulation_module": "False",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-06-17 12:12:21,462 P25515 INFO Set up feature encoder...
2022-06-17 12:12:21,462 P25515 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-06-17 12:12:21,462 P25515 INFO Loading data...
2022-06-17 12:12:21,465 P25515 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-06-17 12:12:21,499 P25515 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-06-17 12:12:21,507 P25515 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-06-17 12:12:21,507 P25515 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-06-17 12:12:21,507 P25515 INFO Loading train data done.
2022-06-17 12:12:25,078 P25515 INFO Total number of parameters: 904701.
2022-06-17 12:12:25,079 P25515 INFO Start training: 343 batches/epoch
2022-06-17 12:12:25,079 P25515 INFO ************ Epoch=1 start ************
2022-06-17 12:12:36,481 P25515 INFO [Metrics] AUC: 0.933634 - logloss: 0.290183
2022-06-17 12:12:36,482 P25515 INFO Save best model: monitor(max): 0.933634
2022-06-17 12:12:36,495 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:12:36,541 P25515 INFO Train loss: 0.419454
2022-06-17 12:12:36,541 P25515 INFO ************ Epoch=1 end ************
2022-06-17 12:12:47,367 P25515 INFO [Metrics] AUC: 0.939138 - logloss: 0.277084
2022-06-17 12:12:47,368 P25515 INFO Save best model: monitor(max): 0.939138
2022-06-17 12:12:47,376 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:12:47,425 P25515 INFO Train loss: 0.371564
2022-06-17 12:12:47,425 P25515 INFO ************ Epoch=2 end ************
2022-06-17 12:12:57,906 P25515 INFO [Metrics] AUC: 0.943257 - logloss: 0.268084
2022-06-17 12:12:57,907 P25515 INFO Save best model: monitor(max): 0.943257
2022-06-17 12:12:57,916 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:12:57,956 P25515 INFO Train loss: 0.375999
2022-06-17 12:12:57,956 P25515 INFO ************ Epoch=3 end ************
2022-06-17 12:13:08,483 P25515 INFO [Metrics] AUC: 0.945390 - logloss: 0.262248
2022-06-17 12:13:08,484 P25515 INFO Save best model: monitor(max): 0.945390
2022-06-17 12:13:08,490 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:13:08,532 P25515 INFO Train loss: 0.380016
2022-06-17 12:13:08,532 P25515 INFO ************ Epoch=4 end ************
2022-06-17 12:13:18,563 P25515 INFO [Metrics] AUC: 0.946697 - logloss: 0.259171
2022-06-17 12:13:18,564 P25515 INFO Save best model: monitor(max): 0.946697
2022-06-17 12:13:18,573 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:13:18,612 P25515 INFO Train loss: 0.380180
2022-06-17 12:13:18,612 P25515 INFO ************ Epoch=5 end ************
2022-06-17 12:13:28,946 P25515 INFO [Metrics] AUC: 0.947954 - logloss: 0.255879
2022-06-17 12:13:28,947 P25515 INFO Save best model: monitor(max): 0.947954
2022-06-17 12:13:28,955 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:13:29,019 P25515 INFO Train loss: 0.380986
2022-06-17 12:13:29,020 P25515 INFO ************ Epoch=6 end ************
2022-06-17 12:13:39,411 P25515 INFO [Metrics] AUC: 0.948742 - logloss: 0.254098
2022-06-17 12:13:39,412 P25515 INFO Save best model: monitor(max): 0.948742
2022-06-17 12:13:39,421 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:13:39,461 P25515 INFO Train loss: 0.381804
2022-06-17 12:13:39,461 P25515 INFO ************ Epoch=7 end ************
2022-06-17 12:13:50,036 P25515 INFO [Metrics] AUC: 0.949946 - logloss: 0.251128
2022-06-17 12:13:50,037 P25515 INFO Save best model: monitor(max): 0.949946
2022-06-17 12:13:50,046 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:13:50,103 P25515 INFO Train loss: 0.382055
2022-06-17 12:13:50,103 P25515 INFO ************ Epoch=8 end ************
2022-06-17 12:14:01,128 P25515 INFO [Metrics] AUC: 0.950708 - logloss: 0.249096
2022-06-17 12:14:01,128 P25515 INFO Save best model: monitor(max): 0.950708
2022-06-17 12:14:01,135 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:14:01,182 P25515 INFO Train loss: 0.381378
2022-06-17 12:14:01,182 P25515 INFO ************ Epoch=9 end ************
2022-06-17 12:14:12,378 P25515 INFO [Metrics] AUC: 0.950998 - logloss: 0.248674
2022-06-17 12:14:12,379 P25515 INFO Save best model: monitor(max): 0.950998
2022-06-17 12:14:12,388 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:14:12,435 P25515 INFO Train loss: 0.381386
2022-06-17 12:14:12,435 P25515 INFO ************ Epoch=10 end ************
2022-06-17 12:14:23,398 P25515 INFO [Metrics] AUC: 0.951889 - logloss: 0.245546
2022-06-17 12:14:23,399 P25515 INFO Save best model: monitor(max): 0.951889
2022-06-17 12:14:23,408 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:14:23,450 P25515 INFO Train loss: 0.380847
2022-06-17 12:14:23,450 P25515 INFO ************ Epoch=11 end ************
2022-06-17 12:14:34,844 P25515 INFO [Metrics] AUC: 0.951935 - logloss: 0.245423
2022-06-17 12:14:34,845 P25515 INFO Save best model: monitor(max): 0.951935
2022-06-17 12:14:34,854 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:14:34,915 P25515 INFO Train loss: 0.380140
2022-06-17 12:14:34,915 P25515 INFO ************ Epoch=12 end ************
2022-06-17 12:14:45,648 P25515 INFO [Metrics] AUC: 0.953329 - logloss: 0.241963
2022-06-17 12:14:45,649 P25515 INFO Save best model: monitor(max): 0.953329
2022-06-17 12:14:45,658 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:14:45,736 P25515 INFO Train loss: 0.379753
2022-06-17 12:14:45,736 P25515 INFO ************ Epoch=13 end ************
2022-06-17 12:14:55,791 P25515 INFO [Metrics] AUC: 0.953397 - logloss: 0.241601
2022-06-17 12:14:55,792 P25515 INFO Save best model: monitor(max): 0.953397
2022-06-17 12:14:55,799 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:14:55,844 P25515 INFO Train loss: 0.378183
2022-06-17 12:14:55,845 P25515 INFO ************ Epoch=14 end ************
2022-06-17 12:15:05,947 P25515 INFO [Metrics] AUC: 0.953892 - logloss: 0.240063
2022-06-17 12:15:05,948 P25515 INFO Save best model: monitor(max): 0.953892
2022-06-17 12:15:05,957 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:15:06,018 P25515 INFO Train loss: 0.377304
2022-06-17 12:15:06,018 P25515 INFO ************ Epoch=15 end ************
2022-06-17 12:15:16,193 P25515 INFO [Metrics] AUC: 0.954159 - logloss: 0.240188
2022-06-17 12:15:16,194 P25515 INFO Save best model: monitor(max): 0.954159
2022-06-17 12:15:16,203 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:15:16,264 P25515 INFO Train loss: 0.376969
2022-06-17 12:15:16,264 P25515 INFO ************ Epoch=16 end ************
2022-06-17 12:15:26,760 P25515 INFO [Metrics] AUC: 0.954100 - logloss: 0.240116
2022-06-17 12:15:26,761 P25515 INFO Monitor(max) STOP: 0.954100 !
2022-06-17 12:15:26,761 P25515 INFO Reduce learning rate on plateau: 0.000100
2022-06-17 12:15:26,761 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:15:26,804 P25515 INFO Train loss: 0.376650
2022-06-17 12:15:26,805 P25515 INFO ************ Epoch=17 end ************
2022-06-17 12:15:36,751 P25515 INFO [Metrics] AUC: 0.965472 - logloss: 0.210414
2022-06-17 12:15:36,752 P25515 INFO Save best model: monitor(max): 0.965472
2022-06-17 12:15:36,761 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:15:36,806 P25515 INFO Train loss: 0.281611
2022-06-17 12:15:36,806 P25515 INFO ************ Epoch=18 end ************
2022-06-17 12:15:46,924 P25515 INFO [Metrics] AUC: 0.967075 - logloss: 0.212382
2022-06-17 12:15:46,925 P25515 INFO Save best model: monitor(max): 0.967075
2022-06-17 12:15:46,931 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:15:46,972 P25515 INFO Train loss: 0.200979
2022-06-17 12:15:46,972 P25515 INFO ************ Epoch=19 end ************
2022-06-17 12:15:56,798 P25515 INFO [Metrics] AUC: 0.966340 - logloss: 0.228757
2022-06-17 12:15:56,799 P25515 INFO Monitor(max) STOP: 0.966340 !
2022-06-17 12:15:56,799 P25515 INFO Reduce learning rate on plateau: 0.000010
2022-06-17 12:15:56,799 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:15:56,870 P25515 INFO Train loss: 0.161790
2022-06-17 12:15:56,871 P25515 INFO ************ Epoch=20 end ************
2022-06-17 12:16:07,127 P25515 INFO [Metrics] AUC: 0.966700 - logloss: 0.232543
2022-06-17 12:16:07,128 P25515 INFO Monitor(max) STOP: 0.966700 !
2022-06-17 12:16:07,128 P25515 INFO Reduce learning rate on plateau: 0.000001
2022-06-17 12:16:07,128 P25515 INFO Early stopping at epoch=21
2022-06-17 12:16:07,128 P25515 INFO --- 343/343 batches finished ---
2022-06-17 12:16:07,168 P25515 INFO Train loss: 0.127184
2022-06-17 12:16:07,169 P25515 INFO Training finished.
2022-06-17 12:16:07,169 P25515 INFO Load best model: /home/FuxiCTR/benchmarks_local/Movielens/EDCN_movielenslatest_x1/movielenslatest_x1_cd32d937/EDCN_movielenslatest_x1_003_b42bf948.model
2022-06-17 12:16:07,198 P25515 INFO ****** Validation evaluation ******
2022-06-17 12:16:08,525 P25515 INFO [Metrics] AUC: 0.967075 - logloss: 0.212382
2022-06-17 12:16:08,569 P25515 INFO ******** Test evaluation ********
2022-06-17 12:16:08,570 P25515 INFO Loading data...
2022-06-17 12:16:08,570 P25515 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-06-17 12:16:08,576 P25515 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-06-17 12:16:08,576 P25515 INFO Loading test data done.
2022-06-17 12:16:09,231 P25515 INFO [Metrics] AUC: 0.967133 - logloss: 0.212202
```

### Revision History

- [Version 1](https://github.com/openbenchmark/BARS/tree/88d3a0faa4565e975141ae89a52d35d3a8b56eda/ctr_prediction/benchmarks/EDCN/EDCN_movielenslatest_x1): deprecated due to bug fix [#29](https://github.com/reczoo/FuxiCTR/issues/29) of FuxiCTR.
