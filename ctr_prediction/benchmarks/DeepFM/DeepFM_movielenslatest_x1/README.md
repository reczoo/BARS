## DeepFM_movielenslatest_x1

A hands-on guide to run the DeepFM model on the Movielenslatest_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DeepFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DeepFM_movielenslatest_x1_tuner_config_02](./DeepFM_movielenslatest_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DeepFM_movielenslatest_x1
    nohup python run_expid.py --config ./DeepFM_movielenslatest_x1_tuner_config_02 --expid DeepFM_movielenslatest_x1_001_0f6d2e8e --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.968515 | 0.213002  |
| 2 | 0.968224 | 0.215447  |
| 3 | 0.968233 | 0.214846  |
| 4 | 0.968092 | 0.214043  |
| 5 | 0.968050 | 0.211613  |
| Avg | 0.968223 | 0.213790 |
| Std | &#177;0.00016276 | &#177;0.00136272 |


### Logs
```python
2022-02-10 00:52:53,547 P7806 INFO {
    "batch_norm": "True",
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
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DeepFM",
    "model_id": "DeepFM_movielenslatest_x1_001_0f6d2e8e",
    "model_root": "./Movielens/DeepFM_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
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
2022-02-10 00:52:53,548 P7806 INFO Set up feature encoder...
2022-02-10 00:52:53,548 P7806 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-02-10 00:52:53,548 P7806 INFO Loading data...
2022-02-10 00:52:53,551 P7806 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-02-10 00:52:53,590 P7806 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-02-10 00:52:53,602 P7806 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-02-10 00:52:53,603 P7806 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-02-10 00:52:53,603 P7806 INFO Loading train data done.
2022-02-10 00:52:57,869 P7806 INFO Total number of parameters: 1328630.
2022-02-10 00:52:57,870 P7806 INFO Start training: 343 batches/epoch
2022-02-10 00:52:57,870 P7806 INFO ************ Epoch=1 start ************
2022-02-10 00:53:34,792 P7806 INFO [Metrics] AUC: 0.933626 - logloss: 0.294289
2022-02-10 00:53:34,792 P7806 INFO Save best model: monitor(max): 0.933626
2022-02-10 00:53:34,800 P7806 INFO --- 343/343 batches finished ---
2022-02-10 00:53:34,836 P7806 INFO Train loss: 0.383940
2022-02-10 00:53:34,836 P7806 INFO ************ Epoch=1 end ************
2022-02-10 00:54:12,104 P7806 INFO [Metrics] AUC: 0.944673 - logloss: 0.268256
2022-02-10 00:54:12,105 P7806 INFO Save best model: monitor(max): 0.944673
2022-02-10 00:54:12,121 P7806 INFO --- 343/343 batches finished ---
2022-02-10 00:54:12,186 P7806 INFO Train loss: 0.364249
2022-02-10 00:54:12,187 P7806 INFO ************ Epoch=2 end ************
2022-02-10 00:54:49,084 P7806 INFO [Metrics] AUC: 0.948251 - logloss: 0.261106
2022-02-10 00:54:49,084 P7806 INFO Save best model: monitor(max): 0.948251
2022-02-10 00:54:49,093 P7806 INFO --- 343/343 batches finished ---
2022-02-10 00:54:49,135 P7806 INFO Train loss: 0.368224
2022-02-10 00:54:49,136 P7806 INFO ************ Epoch=3 end ************
2022-02-10 00:55:26,166 P7806 INFO [Metrics] AUC: 0.949748 - logloss: 0.252908
2022-02-10 00:55:26,166 P7806 INFO Save best model: monitor(max): 0.949748
2022-02-10 00:55:26,175 P7806 INFO --- 343/343 batches finished ---
2022-02-10 00:55:26,232 P7806 INFO Train loss: 0.370964
2022-02-10 00:55:26,232 P7806 INFO ************ Epoch=4 end ************
2022-02-10 00:56:03,067 P7806 INFO [Metrics] AUC: 0.950725 - logloss: 0.249454
2022-02-10 00:56:03,068 P7806 INFO Save best model: monitor(max): 0.950725
2022-02-10 00:56:03,077 P7806 INFO --- 343/343 batches finished ---
2022-02-10 00:56:03,134 P7806 INFO Train loss: 0.374899
2022-02-10 00:56:03,134 P7806 INFO ************ Epoch=5 end ************
2022-02-10 00:56:40,110 P7806 INFO [Metrics] AUC: 0.951914 - logloss: 0.244771
2022-02-10 00:56:40,110 P7806 INFO Save best model: monitor(max): 0.951914
2022-02-10 00:56:40,119 P7806 INFO --- 343/343 batches finished ---
2022-02-10 00:56:40,168 P7806 INFO Train loss: 0.376905
2022-02-10 00:56:40,168 P7806 INFO ************ Epoch=6 end ************
2022-02-10 00:57:17,245 P7806 INFO [Metrics] AUC: 0.952396 - logloss: 0.249258
2022-02-10 00:57:17,245 P7806 INFO Save best model: monitor(max): 0.952396
2022-02-10 00:57:17,254 P7806 INFO --- 343/343 batches finished ---
2022-02-10 00:57:17,318 P7806 INFO Train loss: 0.379039
2022-02-10 00:57:17,318 P7806 INFO ************ Epoch=7 end ************
2022-02-10 00:57:54,167 P7806 INFO [Metrics] AUC: 0.953777 - logloss: 0.240847
2022-02-10 00:57:54,168 P7806 INFO Save best model: monitor(max): 0.953777
2022-02-10 00:57:54,177 P7806 INFO --- 343/343 batches finished ---
2022-02-10 00:57:54,217 P7806 INFO Train loss: 0.380648
2022-02-10 00:57:54,217 P7806 INFO ************ Epoch=8 end ************
2022-02-10 00:58:27,284 P7806 INFO [Metrics] AUC: 0.954308 - logloss: 0.239431
2022-02-10 00:58:27,285 P7806 INFO Save best model: monitor(max): 0.954308
2022-02-10 00:58:27,294 P7806 INFO --- 343/343 batches finished ---
2022-02-10 00:58:27,343 P7806 INFO Train loss: 0.382599
2022-02-10 00:58:27,344 P7806 INFO ************ Epoch=9 end ************
2022-02-10 00:59:00,512 P7806 INFO [Metrics] AUC: 0.954788 - logloss: 0.237974
2022-02-10 00:59:00,513 P7806 INFO Save best model: monitor(max): 0.954788
2022-02-10 00:59:00,522 P7806 INFO --- 343/343 batches finished ---
2022-02-10 00:59:00,589 P7806 INFO Train loss: 0.381046
2022-02-10 00:59:00,589 P7806 INFO ************ Epoch=10 end ************
2022-02-10 00:59:33,841 P7806 INFO [Metrics] AUC: 0.955321 - logloss: 0.236506
2022-02-10 00:59:33,841 P7806 INFO Save best model: monitor(max): 0.955321
2022-02-10 00:59:33,850 P7806 INFO --- 343/343 batches finished ---
2022-02-10 00:59:33,891 P7806 INFO Train loss: 0.381104
2022-02-10 00:59:33,891 P7806 INFO ************ Epoch=11 end ************
2022-02-10 01:00:07,372 P7806 INFO [Metrics] AUC: 0.955086 - logloss: 0.238763
2022-02-10 01:00:07,373 P7806 INFO Monitor(max) STOP: 0.955086 !
2022-02-10 01:00:07,373 P7806 INFO Reduce learning rate on plateau: 0.000100
2022-02-10 01:00:07,373 P7806 INFO --- 343/343 batches finished ---
2022-02-10 01:00:07,420 P7806 INFO Train loss: 0.381737
2022-02-10 01:00:07,420 P7806 INFO ************ Epoch=12 end ************
2022-02-10 01:00:36,072 P7806 INFO [Metrics] AUC: 0.966958 - logloss: 0.207044
2022-02-10 01:00:36,073 P7806 INFO Save best model: monitor(max): 0.966958
2022-02-10 01:00:36,082 P7806 INFO --- 343/343 batches finished ---
2022-02-10 01:00:36,124 P7806 INFO Train loss: 0.280365
2022-02-10 01:00:36,124 P7806 INFO ************ Epoch=13 end ************
2022-02-10 01:01:01,896 P7806 INFO [Metrics] AUC: 0.968768 - logloss: 0.212037
2022-02-10 01:01:01,896 P7806 INFO Save best model: monitor(max): 0.968768
2022-02-10 01:01:01,905 P7806 INFO --- 343/343 batches finished ---
2022-02-10 01:01:01,957 P7806 INFO Train loss: 0.194116
2022-02-10 01:01:01,959 P7806 INFO ************ Epoch=14 end ************
2022-02-10 01:01:24,001 P7806 INFO [Metrics] AUC: 0.967715 - logloss: 0.233148
2022-02-10 01:01:24,002 P7806 INFO Monitor(max) STOP: 0.967715 !
2022-02-10 01:01:24,002 P7806 INFO Reduce learning rate on plateau: 0.000010
2022-02-10 01:01:24,002 P7806 INFO --- 343/343 batches finished ---
2022-02-10 01:01:24,040 P7806 INFO Train loss: 0.150472
2022-02-10 01:01:24,040 P7806 INFO ************ Epoch=15 end ************
2022-02-10 01:01:44,809 P7806 INFO [Metrics] AUC: 0.967179 - logloss: 0.247217
2022-02-10 01:01:44,810 P7806 INFO Monitor(max) STOP: 0.967179 !
2022-02-10 01:01:44,810 P7806 INFO Reduce learning rate on plateau: 0.000001
2022-02-10 01:01:44,810 P7806 INFO Early stopping at epoch=16
2022-02-10 01:01:44,810 P7806 INFO --- 343/343 batches finished ---
2022-02-10 01:01:44,845 P7806 INFO Train loss: 0.117001
2022-02-10 01:01:44,846 P7806 INFO Training finished.
2022-02-10 01:01:44,846 P7806 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Movielens/DeepFM_movielenslatest_x1/movielenslatest_x1_cd32d937/DeepFM_movielenslatest_x1_001_0f6d2e8e.model
2022-02-10 01:01:44,863 P7806 INFO ****** Validation evaluation ******
2022-02-10 01:01:46,249 P7806 INFO [Metrics] AUC: 0.968768 - logloss: 0.212037
2022-02-10 01:01:46,294 P7806 INFO ******** Test evaluation ********
2022-02-10 01:01:46,294 P7806 INFO Loading data...
2022-02-10 01:01:46,295 P7806 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-02-10 01:01:46,300 P7806 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-02-10 01:01:46,300 P7806 INFO Loading test data done.
2022-02-10 01:01:47,081 P7806 INFO [Metrics] AUC: 0.968515 - logloss: 0.213002

```
