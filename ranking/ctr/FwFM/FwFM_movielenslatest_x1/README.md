## FwFM_movielenslatest_x1

A hands-on guide to run the FwFM model on the MovielensLatest_x1 dataset.

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
Dataset ID: [MovielensLatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/MovieLens#MovielensLatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [FwFM](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FwFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FwFM_movielenslatest_x1_tuner_config_02](./FwFM_movielenslatest_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FwFM_movielenslatest_x1
    nohup python run_expid.py --config ./FwFM_movielenslatest_x1_tuner_config_02 --expid FwFM_movielenslatest_x1_006_e527bbd6 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.955799 | 0.242620  |


### Logs
```python
2022-01-25 13:54:27,525 P43856 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "embedding_dim": "10",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "linear_type": "FeLV",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FwFM",
    "model_id": "FwFM_movielenslatest_x1_006_e527bbd6",
    "model_root": "./Movielens/FwFM_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "5e-06",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Movielens/MovielensLatest_x1/test.csv",
    "train_data": "../data/Movielens/MovielensLatest_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-25 13:54:27,526 P43856 INFO Set up feature encoder...
2022-01-25 13:54:27,526 P43856 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-01-25 13:54:27,526 P43856 INFO Loading data...
2022-01-25 13:54:27,529 P43856 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-01-25 13:54:28,248 P43856 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-01-25 13:54:28,259 P43856 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-01-25 13:54:28,259 P43856 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-01-25 13:54:28,259 P43856 INFO Loading train data done.
2022-01-25 13:55:10,424 P43856 INFO Total number of parameters: 1804784.
2022-01-25 13:55:10,424 P43856 INFO Start training: 343 batches/epoch
2022-01-25 13:55:10,424 P43856 INFO ************ Epoch=1 start ************
2022-01-25 13:55:26,643 P43856 INFO [Metrics] AUC: 0.857053 - logloss: 0.482405
2022-01-25 13:55:26,643 P43856 INFO Save best model: monitor(max): 0.857053
2022-01-25 13:55:26,655 P43856 INFO --- 343/343 batches finished ---
2022-01-25 13:55:26,722 P43856 INFO Train loss: 0.605152
2022-01-25 13:55:26,722 P43856 INFO ************ Epoch=1 end ************
2022-01-25 13:55:39,308 P43856 INFO [Metrics] AUC: 0.925707 - logloss: 0.334313
2022-01-25 13:55:39,308 P43856 INFO Save best model: monitor(max): 0.925707
2022-01-25 13:55:39,318 P43856 INFO --- 343/343 batches finished ---
2022-01-25 13:55:39,380 P43856 INFO Train loss: 0.405395
2022-01-25 13:55:39,380 P43856 INFO ************ Epoch=2 end ************
2022-01-25 13:55:51,880 P43856 INFO [Metrics] AUC: 0.942385 - logloss: 0.280649
2022-01-25 13:55:51,880 P43856 INFO Save best model: monitor(max): 0.942385
2022-01-25 13:55:51,890 P43856 INFO --- 343/343 batches finished ---
2022-01-25 13:55:51,963 P43856 INFO Train loss: 0.314460
2022-01-25 13:55:51,964 P43856 INFO ************ Epoch=3 end ************
2022-01-25 13:56:04,084 P43856 INFO [Metrics] AUC: 0.948237 - logloss: 0.259972
2022-01-25 13:56:04,085 P43856 INFO Save best model: monitor(max): 0.948237
2022-01-25 13:56:04,098 P43856 INFO --- 343/343 batches finished ---
2022-01-25 13:56:04,169 P43856 INFO Train loss: 0.276651
2022-01-25 13:56:04,169 P43856 INFO ************ Epoch=4 end ************
2022-01-25 13:56:16,649 P43856 INFO [Metrics] AUC: 0.951234 - logloss: 0.249929
2022-01-25 13:56:16,650 P43856 INFO Save best model: monitor(max): 0.951234
2022-01-25 13:56:16,664 P43856 INFO --- 343/343 batches finished ---
2022-01-25 13:56:16,720 P43856 INFO Train loss: 0.254924
2022-01-25 13:56:16,720 P43856 INFO ************ Epoch=5 end ************
2022-01-25 13:56:29,800 P43856 INFO [Metrics] AUC: 0.953068 - logloss: 0.244246
2022-01-25 13:56:29,801 P43856 INFO Save best model: monitor(max): 0.953068
2022-01-25 13:56:29,816 P43856 INFO --- 343/343 batches finished ---
2022-01-25 13:56:29,889 P43856 INFO Train loss: 0.238786
2022-01-25 13:56:29,890 P43856 INFO ************ Epoch=6 end ************
2022-01-25 13:56:41,898 P43856 INFO [Metrics] AUC: 0.954283 - logloss: 0.240922
2022-01-25 13:56:41,899 P43856 INFO Save best model: monitor(max): 0.954283
2022-01-25 13:56:41,911 P43856 INFO --- 343/343 batches finished ---
2022-01-25 13:56:41,972 P43856 INFO Train loss: 0.225328
2022-01-25 13:56:41,973 P43856 INFO ************ Epoch=7 end ************
2022-01-25 13:56:56,363 P43856 INFO [Metrics] AUC: 0.955018 - logloss: 0.239211
2022-01-25 13:56:56,364 P43856 INFO Save best model: monitor(max): 0.955018
2022-01-25 13:56:56,374 P43856 INFO --- 343/343 batches finished ---
2022-01-25 13:56:56,437 P43856 INFO Train loss: 0.213537
2022-01-25 13:56:56,437 P43856 INFO ************ Epoch=8 end ************
2022-01-25 13:57:07,516 P43856 INFO [Metrics] AUC: 0.955468 - logloss: 0.238645
2022-01-25 13:57:07,516 P43856 INFO Save best model: monitor(max): 0.955468
2022-01-25 13:57:07,527 P43856 INFO --- 343/343 batches finished ---
2022-01-25 13:57:07,589 P43856 INFO Train loss: 0.202920
2022-01-25 13:57:07,590 P43856 INFO ************ Epoch=9 end ************
2022-01-25 13:57:19,900 P43856 INFO [Metrics] AUC: 0.955801 - logloss: 0.238970
2022-01-25 13:57:19,901 P43856 INFO Save best model: monitor(max): 0.955801
2022-01-25 13:57:19,913 P43856 INFO --- 343/343 batches finished ---
2022-01-25 13:57:19,964 P43856 INFO Train loss: 0.193248
2022-01-25 13:57:19,964 P43856 INFO ************ Epoch=10 end ************
2022-01-25 13:57:32,037 P43856 INFO [Metrics] AUC: 0.955880 - logloss: 0.240202
2022-01-25 13:57:32,037 P43856 INFO Save best model: monitor(max): 0.955880
2022-01-25 13:57:32,049 P43856 INFO --- 343/343 batches finished ---
2022-01-25 13:57:32,116 P43856 INFO Train loss: 0.184399
2022-01-25 13:57:32,116 P43856 INFO ************ Epoch=11 end ************
2022-01-25 13:57:40,104 P43856 INFO [Metrics] AUC: 0.955890 - logloss: 0.242119
2022-01-25 13:57:40,105 P43856 INFO Save best model: monitor(max): 0.955890
2022-01-25 13:57:40,117 P43856 INFO --- 343/343 batches finished ---
2022-01-25 13:57:40,179 P43856 INFO Train loss: 0.176247
2022-01-25 13:57:40,179 P43856 INFO ************ Epoch=12 end ************
2022-01-25 13:57:49,829 P43856 INFO [Metrics] AUC: 0.955803 - logloss: 0.244583
2022-01-25 13:57:49,830 P43856 INFO Monitor(max) STOP: 0.955803 !
2022-01-25 13:57:49,830 P43856 INFO Reduce learning rate on plateau: 0.000100
2022-01-25 13:57:49,830 P43856 INFO --- 343/343 batches finished ---
2022-01-25 13:57:49,905 P43856 INFO Train loss: 0.168766
2022-01-25 13:57:49,905 P43856 INFO ************ Epoch=13 end ************
2022-01-25 13:57:56,438 P43856 INFO [Metrics] AUC: 0.955720 - logloss: 0.245005
2022-01-25 13:57:56,439 P43856 INFO Monitor(max) STOP: 0.955720 !
2022-01-25 13:57:56,439 P43856 INFO Reduce learning rate on plateau: 0.000010
2022-01-25 13:57:56,439 P43856 INFO Early stopping at epoch=14
2022-01-25 13:57:56,439 P43856 INFO --- 343/343 batches finished ---
2022-01-25 13:57:56,492 P43856 INFO Train loss: 0.157290
2022-01-25 13:57:56,492 P43856 INFO Training finished.
2022-01-25 13:57:56,881 P43856 INFO Load best model: /home/XXX/benchmarks/Movielens/FwFM_movielenslatest_x1/movielenslatest_x1_cd32d937/FwFM_movielenslatest_x1_006_e527bbd6.model
2022-01-25 13:58:22,122 P43856 INFO ****** Validation evaluation ******
2022-01-25 13:58:25,329 P43856 INFO [Metrics] AUC: 0.955890 - logloss: 0.242119
2022-01-25 13:58:25,399 P43856 INFO ******** Test evaluation ********
2022-01-25 13:58:25,399 P43856 INFO Loading data...
2022-01-25 13:58:25,399 P43856 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-01-25 13:58:25,404 P43856 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-01-25 13:58:25,404 P43856 INFO Loading test data done.
2022-01-25 13:58:27,832 P43856 INFO [Metrics] AUC: 0.955799 - logloss: 0.242620

```
