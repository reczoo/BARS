## DeepIM_movielenslatest_x1

A hands-on guide to run the DeepIM model on the Movielenslatest_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DeepIM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DeepIM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DeepIM_movielenslatest_x1_tuner_config_01](./DeepIM_movielenslatest_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd DeepIM_movielenslatest_x1
   nohup python run_expid.py --config ./DeepIM_movielenslatest_x1_tuner_config_01 --expid DeepIM_movielenslatest_x1_022_eb1c9e99 --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

Total 5 runs:

| Runs | AUC              | logloss          |
|:----:|:----------------:|:----------------:|
| 1    | 0.969316         | 0.209861         |
| 2    | 0.967859         | 0.212252         |
| 3    | 0.968758         | 0.212782         |
| 4    | 0.967749         | 0.212312         |
| 5    | 0.968553         | 0.212095         |
| Avg  | 0.968447         | 0.211860         |
| Std  | &#177;0.00058242 | &#177;0.00102560 |

### Logs

```python
2022-11-04 15:11:43,876 P30497 INFO {
    "batch_size": "4096",
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
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "im_batch_norm": "True",
    "im_order": "5",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DeepIM",
    "model_id": "DeepIM_movielenslatest_x1_022_eb1c9e99",
    "model_root": "./Movielens/DeepIM_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_batch_norm": "True",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
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
2022-11-04 15:11:43,877 P30497 INFO Set up feature encoder...
2022-11-04 15:11:43,877 P30497 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-11-04 15:11:43,877 P30497 INFO Loading data...
2022-11-04 15:11:43,879 P30497 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-11-04 15:11:43,905 P30497 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-11-04 15:11:43,913 P30497 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-11-04 15:11:43,913 P30497 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-11-04 15:11:43,913 P30497 INFO Loading train data done.
2022-11-04 15:11:47,233 P30497 INFO Total number of parameters: 1238542.
2022-11-04 15:11:47,234 P30497 INFO Start training: 343 batches/epoch
2022-11-04 15:11:47,234 P30497 INFO ************ Epoch=1 start ************
2022-11-04 15:12:01,761 P30497 INFO [Metrics] AUC: 0.934311 - logloss: 0.292014
2022-11-04 15:12:01,762 P30497 INFO Save best model: monitor(max): 0.934311
2022-11-04 15:12:01,771 P30497 INFO --- 343/343 batches finished ---
2022-11-04 15:12:01,812 P30497 INFO Train loss: 0.379546
2022-11-04 15:12:01,813 P30497 INFO ************ Epoch=1 end ************
2022-11-04 15:12:16,275 P30497 INFO [Metrics] AUC: 0.944055 - logloss: 0.272700
2022-11-04 15:12:16,276 P30497 INFO Save best model: monitor(max): 0.944055
2022-11-04 15:12:16,284 P30497 INFO --- 343/343 batches finished ---
2022-11-04 15:12:16,345 P30497 INFO Train loss: 0.359097
2022-11-04 15:12:16,345 P30497 INFO ************ Epoch=2 end ************
2022-11-04 15:12:30,945 P30497 INFO [Metrics] AUC: 0.947627 - logloss: 0.257598
2022-11-04 15:12:30,946 P30497 INFO Save best model: monitor(max): 0.947627
2022-11-04 15:12:30,957 P30497 INFO --- 343/343 batches finished ---
2022-11-04 15:12:31,006 P30497 INFO Train loss: 0.360707
2022-11-04 15:12:31,007 P30497 INFO ************ Epoch=3 end ************
2022-11-04 15:12:45,464 P30497 INFO [Metrics] AUC: 0.949276 - logloss: 0.252871
2022-11-04 15:12:45,464 P30497 INFO Save best model: monitor(max): 0.949276
2022-11-04 15:12:45,472 P30497 INFO --- 343/343 batches finished ---
2022-11-04 15:12:45,548 P30497 INFO Train loss: 0.366489
2022-11-04 15:12:45,548 P30497 INFO ************ Epoch=4 end ************
2022-11-04 15:12:57,395 P30497 INFO [Metrics] AUC: 0.950254 - logloss: 0.249001
2022-11-04 15:12:57,396 P30497 INFO Save best model: monitor(max): 0.950254
2022-11-04 15:12:57,404 P30497 INFO --- 343/343 batches finished ---
2022-11-04 15:12:57,451 P30497 INFO Train loss: 0.370444
2022-11-04 15:12:57,451 P30497 INFO ************ Epoch=5 end ************
2022-11-04 15:13:10,477 P30497 INFO [Metrics] AUC: 0.951529 - logloss: 0.246459
2022-11-04 15:13:10,478 P30497 INFO Save best model: monitor(max): 0.951529
2022-11-04 15:13:10,487 P30497 INFO --- 343/343 batches finished ---
2022-11-04 15:13:10,537 P30497 INFO Train loss: 0.374334
2022-11-04 15:13:10,537 P30497 INFO ************ Epoch=6 end ************
2022-11-04 15:13:25,282 P30497 INFO [Metrics] AUC: 0.953046 - logloss: 0.242244
2022-11-04 15:13:25,283 P30497 INFO Save best model: monitor(max): 0.953046
2022-11-04 15:13:25,291 P30497 INFO --- 343/343 batches finished ---
2022-11-04 15:13:25,325 P30497 INFO Train loss: 0.375143
2022-11-04 15:13:25,325 P30497 INFO ************ Epoch=7 end ************
2022-11-04 15:13:38,872 P30497 INFO [Metrics] AUC: 0.953973 - logloss: 0.240622
2022-11-04 15:13:38,873 P30497 INFO Save best model: monitor(max): 0.953973
2022-11-04 15:13:38,883 P30497 INFO --- 343/343 batches finished ---
2022-11-04 15:13:38,932 P30497 INFO Train loss: 0.377144
2022-11-04 15:13:38,932 P30497 INFO ************ Epoch=8 end ************
2022-11-04 15:13:49,784 P30497 INFO [Metrics] AUC: 0.954090 - logloss: 0.238773
2022-11-04 15:13:49,785 P30497 INFO Save best model: monitor(max): 0.954090
2022-11-04 15:13:49,793 P30497 INFO --- 343/343 batches finished ---
2022-11-04 15:13:49,856 P30497 INFO Train loss: 0.375621
2022-11-04 15:13:49,857 P30497 INFO ************ Epoch=9 end ************
2022-11-04 15:14:01,310 P30497 INFO [Metrics] AUC: 0.955372 - logloss: 0.235571
2022-11-04 15:14:01,311 P30497 INFO Save best model: monitor(max): 0.955372
2022-11-04 15:14:01,322 P30497 INFO --- 343/343 batches finished ---
2022-11-04 15:14:01,366 P30497 INFO Train loss: 0.376009
2022-11-04 15:14:01,366 P30497 INFO ************ Epoch=10 end ************
2022-11-04 15:14:12,257 P30497 INFO [Metrics] AUC: 0.955174 - logloss: 0.236176
2022-11-04 15:14:12,258 P30497 INFO Monitor(max) STOP: 0.955174 !
2022-11-04 15:14:12,258 P30497 INFO Reduce learning rate on plateau: 0.000100
2022-11-04 15:14:12,258 P30497 INFO --- 343/343 batches finished ---
2022-11-04 15:14:12,307 P30497 INFO Train loss: 0.376239
2022-11-04 15:14:12,307 P30497 INFO ************ Epoch=11 end ************
2022-11-04 15:14:23,247 P30497 INFO [Metrics] AUC: 0.967495 - logloss: 0.205305
2022-11-04 15:14:23,247 P30497 INFO Save best model: monitor(max): 0.967495
2022-11-04 15:14:23,258 P30497 INFO --- 343/343 batches finished ---
2022-11-04 15:14:23,297 P30497 INFO Train loss: 0.278775
2022-11-04 15:14:23,297 P30497 INFO ************ Epoch=12 end ************
2022-11-04 15:14:34,036 P30497 INFO [Metrics] AUC: 0.969260 - logloss: 0.209989
2022-11-04 15:14:34,037 P30497 INFO Save best model: monitor(max): 0.969260
2022-11-04 15:14:34,045 P30497 INFO --- 343/343 batches finished ---
2022-11-04 15:14:34,117 P30497 INFO Train loss: 0.193920
2022-11-04 15:14:34,117 P30497 INFO ************ Epoch=13 end ************
2022-11-04 15:14:45,144 P30497 INFO [Metrics] AUC: 0.968248 - logloss: 0.230470
2022-11-04 15:14:45,145 P30497 INFO Monitor(max) STOP: 0.968248 !
2022-11-04 15:14:45,145 P30497 INFO Reduce learning rate on plateau: 0.000010
2022-11-04 15:14:45,145 P30497 INFO --- 343/343 batches finished ---
2022-11-04 15:14:45,211 P30497 INFO Train loss: 0.150512
2022-11-04 15:14:45,212 P30497 INFO ************ Epoch=14 end ************
2022-11-04 15:14:56,225 P30497 INFO [Metrics] AUC: 0.967887 - logloss: 0.244538
2022-11-04 15:14:56,226 P30497 INFO Monitor(max) STOP: 0.967887 !
2022-11-04 15:14:56,226 P30497 INFO Reduce learning rate on plateau: 0.000001
2022-11-04 15:14:56,226 P30497 INFO Early stopping at epoch=15
2022-11-04 15:14:56,226 P30497 INFO --- 343/343 batches finished ---
2022-11-04 15:14:56,269 P30497 INFO Train loss: 0.117196
2022-11-04 15:14:56,270 P30497 INFO Training finished.
2022-11-04 15:14:56,270 P30497 INFO Load best model: /home/FuxiCTR/benchmarks/Movielens/DeepIM_movielenslatest_x1/movielenslatest_x1_cd32d937/DeepIM_movielenslatest_x1_022_eb1c9e99.model
2022-11-04 15:14:56,291 P30497 INFO ****** Validation evaluation ******
2022-11-04 15:14:57,689 P30497 INFO [Metrics] AUC: 0.969260 - logloss: 0.209989
2022-11-04 15:14:57,741 P30497 INFO ******** Test evaluation ********
2022-11-04 15:14:57,741 P30497 INFO Loading data...
2022-11-04 15:14:57,742 P30497 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-11-04 15:14:57,747 P30497 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-11-04 15:14:57,747 P30497 INFO Loading test data done.
2022-11-04 15:14:58,427 P30497 INFO [Metrics] AUC: 0.969316 - logloss: 0.209861
```
