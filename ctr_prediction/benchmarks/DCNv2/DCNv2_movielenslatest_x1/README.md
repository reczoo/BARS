## DCNv2_movielenslatest_x1

A hands-on guide to run the DCNv2 model on the Movielenslatest_x1 dataset.

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
Dataset ID: [Movielenslatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/MovieLens/README.md#Movielenslatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DCNv2](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DCNv2.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DCNv2_movielenslatest_x1_tuner_config_01](./DCNv2_movielenslatest_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DCNv2_movielenslatest_x1
    nohup python run_expid.py --config ./DCNv2_movielenslatest_x1_tuner_config_01 --expid DCNv2_movielenslatest_x1_001_98ea1c72 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.969098 | 0.214736  |
| 2 | 0.968205 | 0.217647  |
| 3 | 0.968267 | 0.216530  |
| 4 | 0.968876 | 0.214814  |
| 5 | 0.968838 | 0.216954  |
| Avg | 0.968657 | 0.216136 |
| Std | &#177;0.00035542 | &#177;0.00116749 |


### Logs
```python
2022-02-11 00:50:52,917 P22270 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "dnn_activations": "relu",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "low_rank": "32",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCNv2",
    "model_id": "DCNv2_movielenslatest_x1_001_98ea1c72",
    "model_root": "./Movielens/DCN_movielenslatest_x1/",
    "model_structure": "parallel",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_cross_layers": "5",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_dnn_hidden_units": "[400, 400, 400]",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "stacked_dnn_hidden_units": "[500, 500, 500]",
    "task": "binary_classification",
    "test_data": "../data/Movielens/MovielensLatest_x1/test.csv",
    "train_data": "../data/Movielens/MovielensLatest_x1/train.csv",
    "use_hdf5": "True",
    "use_low_rank_mixture": "False",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-02-11 00:50:52,926 P22270 INFO Set up feature encoder...
2022-02-11 00:50:52,926 P22270 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-02-11 00:50:52,926 P22270 INFO Loading data...
2022-02-11 00:50:52,947 P22270 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-02-11 00:50:53,156 P22270 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-02-11 00:50:53,261 P22270 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-02-11 00:50:53,261 P22270 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-02-11 00:50:53,261 P22270 INFO Loading train data done.
2022-02-11 00:51:17,253 P22270 INFO Total number of parameters: 1243071.
2022-02-11 00:51:17,283 P22270 INFO Start training: 343 batches/epoch
2022-02-11 00:51:17,283 P22270 INFO ************ Epoch=1 start ************
2022-02-11 00:52:51,887 P22270 INFO [Metrics] AUC: 0.935047 - logloss: 0.293483
2022-02-11 00:52:51,889 P22270 INFO Save best model: monitor(max): 0.935047
2022-02-11 00:52:51,976 P22270 INFO --- 343/343 batches finished ---
2022-02-11 00:52:52,465 P22270 INFO Train loss: 0.382016
2022-02-11 00:52:52,465 P22270 INFO ************ Epoch=1 end ************
2022-02-11 00:54:36,765 P22270 INFO [Metrics] AUC: 0.945515 - logloss: 0.264626
2022-02-11 00:54:36,776 P22270 INFO Save best model: monitor(max): 0.945515
2022-02-11 00:54:36,864 P22270 INFO --- 343/343 batches finished ---
2022-02-11 00:54:37,318 P22270 INFO Train loss: 0.371308
2022-02-11 00:54:37,319 P22270 INFO ************ Epoch=2 end ************
2022-02-11 00:56:22,941 P22270 INFO [Metrics] AUC: 0.948241 - logloss: 0.258100
2022-02-11 00:56:22,950 P22270 INFO Save best model: monitor(max): 0.948241
2022-02-11 00:56:22,999 P22270 INFO --- 343/343 batches finished ---
2022-02-11 00:56:23,305 P22270 INFO Train loss: 0.370473
2022-02-11 00:56:23,305 P22270 INFO ************ Epoch=3 end ************
2022-02-11 00:58:04,200 P22270 INFO [Metrics] AUC: 0.950849 - logloss: 0.251910
2022-02-11 00:58:04,222 P22270 INFO Save best model: monitor(max): 0.950849
2022-02-11 00:58:04,296 P22270 INFO --- 343/343 batches finished ---
2022-02-11 00:58:04,638 P22270 INFO Train loss: 0.372997
2022-02-11 00:58:04,638 P22270 INFO ************ Epoch=4 end ************
2022-02-11 00:59:48,631 P22270 INFO [Metrics] AUC: 0.952387 - logloss: 0.247171
2022-02-11 00:59:48,683 P22270 INFO Save best model: monitor(max): 0.952387
2022-02-11 00:59:48,805 P22270 INFO --- 343/343 batches finished ---
2022-02-11 00:59:49,206 P22270 INFO Train loss: 0.375214
2022-02-11 00:59:49,206 P22270 INFO ************ Epoch=5 end ************
2022-02-11 01:01:25,207 P22270 INFO [Metrics] AUC: 0.953951 - logloss: 0.239795
2022-02-11 01:01:25,209 P22270 INFO Save best model: monitor(max): 0.953951
2022-02-11 01:01:25,319 P22270 INFO --- 343/343 batches finished ---
2022-02-11 01:01:25,782 P22270 INFO Train loss: 0.376620
2022-02-11 01:01:25,782 P22270 INFO ************ Epoch=6 end ************
2022-02-11 01:03:08,844 P22270 INFO [Metrics] AUC: 0.954787 - logloss: 0.238054
2022-02-11 01:03:08,876 P22270 INFO Save best model: monitor(max): 0.954787
2022-02-11 01:03:08,921 P22270 INFO --- 343/343 batches finished ---
2022-02-11 01:03:09,261 P22270 INFO Train loss: 0.377080
2022-02-11 01:03:09,261 P22270 INFO ************ Epoch=7 end ************
2022-02-11 01:04:50,940 P22270 INFO [Metrics] AUC: 0.955095 - logloss: 0.237576
2022-02-11 01:04:50,941 P22270 INFO Save best model: monitor(max): 0.955095
2022-02-11 01:04:51,065 P22270 INFO --- 343/343 batches finished ---
2022-02-11 01:04:51,452 P22270 INFO Train loss: 0.377740
2022-02-11 01:04:51,453 P22270 INFO ************ Epoch=8 end ************
2022-02-11 01:06:27,497 P22270 INFO [Metrics] AUC: 0.955781 - logloss: 0.236796
2022-02-11 01:06:27,524 P22270 INFO Save best model: monitor(max): 0.955781
2022-02-11 01:06:27,616 P22270 INFO --- 343/343 batches finished ---
2022-02-11 01:06:27,973 P22270 INFO Train loss: 0.377764
2022-02-11 01:06:27,974 P22270 INFO ************ Epoch=9 end ************
2022-02-11 01:08:12,233 P22270 INFO [Metrics] AUC: 0.956402 - logloss: 0.233103
2022-02-11 01:08:12,258 P22270 INFO Save best model: monitor(max): 0.956402
2022-02-11 01:08:12,416 P22270 INFO --- 343/343 batches finished ---
2022-02-11 01:08:12,862 P22270 INFO Train loss: 0.379035
2022-02-11 01:08:12,862 P22270 INFO ************ Epoch=10 end ************
2022-02-11 01:09:48,890 P22270 INFO [Metrics] AUC: 0.956446 - logloss: 0.233880
2022-02-11 01:09:48,891 P22270 INFO Save best model: monitor(max): 0.956446
2022-02-11 01:09:49,069 P22270 INFO --- 343/343 batches finished ---
2022-02-11 01:09:49,517 P22270 INFO Train loss: 0.379735
2022-02-11 01:09:49,517 P22270 INFO ************ Epoch=11 end ************
2022-02-11 01:10:53,866 P22270 INFO [Metrics] AUC: 0.956138 - logloss: 0.234126
2022-02-11 01:10:53,877 P22270 INFO Monitor(max) STOP: 0.956138 !
2022-02-11 01:10:53,877 P22270 INFO Reduce learning rate on plateau: 0.000100
2022-02-11 01:10:53,877 P22270 INFO --- 343/343 batches finished ---
2022-02-11 01:10:54,044 P22270 INFO Train loss: 0.379465
2022-02-11 01:10:54,044 P22270 INFO ************ Epoch=12 end ************
2022-02-11 01:12:00,109 P22270 INFO [Metrics] AUC: 0.967827 - logloss: 0.205812
2022-02-11 01:12:00,126 P22270 INFO Save best model: monitor(max): 0.967827
2022-02-11 01:12:00,187 P22270 INFO --- 343/343 batches finished ---
2022-02-11 01:12:00,474 P22270 INFO Train loss: 0.277900
2022-02-11 01:12:00,474 P22270 INFO ************ Epoch=13 end ************
2022-02-11 01:12:57,188 P22270 INFO [Metrics] AUC: 0.969117 - logloss: 0.215212
2022-02-11 01:12:57,211 P22270 INFO Save best model: monitor(max): 0.969117
2022-02-11 01:12:57,322 P22270 INFO --- 343/343 batches finished ---
2022-02-11 01:12:57,690 P22270 INFO Train loss: 0.189714
2022-02-11 01:12:57,691 P22270 INFO ************ Epoch=14 end ************
2022-02-11 01:14:06,648 P22270 INFO [Metrics] AUC: 0.967712 - logloss: 0.238754
2022-02-11 01:14:06,654 P22270 INFO Monitor(max) STOP: 0.967712 !
2022-02-11 01:14:06,654 P22270 INFO Reduce learning rate on plateau: 0.000010
2022-02-11 01:14:06,654 P22270 INFO --- 343/343 batches finished ---
2022-02-11 01:14:06,884 P22270 INFO Train loss: 0.143900
2022-02-11 01:14:06,885 P22270 INFO ************ Epoch=15 end ************
2022-02-11 01:15:14,989 P22270 INFO [Metrics] AUC: 0.967278 - logloss: 0.251545
2022-02-11 01:15:15,011 P22270 INFO Monitor(max) STOP: 0.967278 !
2022-02-11 01:15:15,058 P22270 INFO Reduce learning rate on plateau: 0.000001
2022-02-11 01:15:15,058 P22270 INFO Early stopping at epoch=16
2022-02-11 01:15:15,058 P22270 INFO --- 343/343 batches finished ---
2022-02-11 01:15:15,310 P22270 INFO Train loss: 0.111731
2022-02-11 01:15:15,311 P22270 INFO Training finished.
2022-02-11 01:15:15,311 P22270 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Movielens/DCN_movielenslatest_x1/movielenslatest_x1_cd32d937/DCNv2_movielenslatest_x1_001_98ea1c72.model
2022-02-11 01:15:15,378 P22270 INFO ****** Validation evaluation ******
2022-02-11 01:15:30,314 P22270 INFO [Metrics] AUC: 0.969117 - logloss: 0.215212
2022-02-11 01:15:31,205 P22270 INFO ******** Test evaluation ********
2022-02-11 01:15:31,206 P22270 INFO Loading data...
2022-02-11 01:15:31,207 P22270 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-02-11 01:15:31,234 P22270 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-02-11 01:15:31,234 P22270 INFO Loading test data done.
2022-02-11 01:15:36,083 P22270 INFO [Metrics] AUC: 0.969098 - logloss: 0.214736

```
