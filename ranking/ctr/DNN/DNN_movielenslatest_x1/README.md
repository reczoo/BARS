## DNN_movielenslatest_x1

A hands-on guide to run the DNN model on the Movielenslatest_x1 dataset.

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
Dataset ID: [MovielensLatest_x1](https://github.com/reczoo/Datasets/tree/main/MovieLens/MovielensLatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DNN](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DNN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DNN_movielenslatest_x1_tuner_config_02](./DNN_movielenslatest_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DNN_movielenslatest_x1
    nohup python run_expid.py --config ./DNN_movielenslatest_x1_tuner_config_02 --expid DNN_movielenslatest_x1_001_338b5be6 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.967764 | 0.238757  |
| 2 | 0.967363 | 0.241220  |
| 3 | 0.968299 | 0.237786  |
| 4 | 0.967606 | 0.239932  |
| 5 | 0.967385 | 0.236088  |
| Avg | 0.967683 | 0.238757 |
| Std | &#177;0.00034151 | &#177;0.00176144 |


### Logs
```python
2022-02-10 00:52:15,296 P6999 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.005",
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
    "model": "DNN",
    "model_id": "DNN_movielenslatest_x1_001_338b5be6",
    "model_root": "./Movielens/DNN_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.4",
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
2022-02-10 00:52:15,296 P6999 INFO Set up feature encoder...
2022-02-10 00:52:15,297 P6999 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-02-10 00:52:15,297 P6999 INFO Loading data...
2022-02-10 00:52:15,299 P6999 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-02-10 00:52:15,334 P6999 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-02-10 00:52:15,346 P6999 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-02-10 00:52:15,346 P6999 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-02-10 00:52:15,346 P6999 INFO Loading train data done.
2022-02-10 00:52:19,348 P6999 INFO Total number of parameters: 1574392.
2022-02-10 00:52:19,349 P6999 INFO Start training: 343 batches/epoch
2022-02-10 00:52:19,349 P6999 INFO ************ Epoch=1 start ************
2022-02-10 00:52:48,144 P6999 INFO [Metrics] AUC: 0.931433 - logloss: 0.300571
2022-02-10 00:52:48,144 P6999 INFO Save best model: monitor(max): 0.931433
2022-02-10 00:52:48,154 P6999 INFO --- 343/343 batches finished ---
2022-02-10 00:52:48,193 P6999 INFO Train loss: 0.375315
2022-02-10 00:52:48,193 P6999 INFO ************ Epoch=1 end ************
2022-02-10 00:53:30,598 P6999 INFO [Metrics] AUC: 0.943804 - logloss: 0.273493
2022-02-10 00:53:30,598 P6999 INFO Save best model: monitor(max): 0.943804
2022-02-10 00:53:30,609 P6999 INFO --- 343/343 batches finished ---
2022-02-10 00:53:30,652 P6999 INFO Train loss: 0.339772
2022-02-10 00:53:30,652 P6999 INFO ************ Epoch=2 end ************
2022-02-10 00:54:17,055 P6999 INFO [Metrics] AUC: 0.952665 - logloss: 0.248138
2022-02-10 00:54:17,056 P6999 INFO Save best model: monitor(max): 0.952665
2022-02-10 00:54:17,067 P6999 INFO --- 343/343 batches finished ---
2022-02-10 00:54:17,113 P6999 INFO Train loss: 0.339227
2022-02-10 00:54:17,113 P6999 INFO ************ Epoch=3 end ************
2022-02-10 00:55:04,236 P6999 INFO [Metrics] AUC: 0.955387 - logloss: 0.240477
2022-02-10 00:55:04,237 P6999 INFO Save best model: monitor(max): 0.955387
2022-02-10 00:55:04,248 P6999 INFO --- 343/343 batches finished ---
2022-02-10 00:55:04,293 P6999 INFO Train loss: 0.338165
2022-02-10 00:55:04,294 P6999 INFO ************ Epoch=4 end ************
2022-02-10 00:55:50,951 P6999 INFO [Metrics] AUC: 0.958281 - logloss: 0.233453
2022-02-10 00:55:50,951 P6999 INFO Save best model: monitor(max): 0.958281
2022-02-10 00:55:50,963 P6999 INFO --- 343/343 batches finished ---
2022-02-10 00:55:51,026 P6999 INFO Train loss: 0.338842
2022-02-10 00:55:51,026 P6999 INFO ************ Epoch=5 end ************
2022-02-10 00:56:38,092 P6999 INFO [Metrics] AUC: 0.958514 - logloss: 0.230451
2022-02-10 00:56:38,092 P6999 INFO Save best model: monitor(max): 0.958514
2022-02-10 00:56:38,103 P6999 INFO --- 343/343 batches finished ---
2022-02-10 00:56:38,144 P6999 INFO Train loss: 0.338615
2022-02-10 00:56:38,144 P6999 INFO ************ Epoch=6 end ************
2022-02-10 00:57:24,539 P6999 INFO [Metrics] AUC: 0.959488 - logloss: 0.231087
2022-02-10 00:57:24,540 P6999 INFO Save best model: monitor(max): 0.959488
2022-02-10 00:57:24,551 P6999 INFO --- 343/343 batches finished ---
2022-02-10 00:57:24,623 P6999 INFO Train loss: 0.338980
2022-02-10 00:57:24,624 P6999 INFO ************ Epoch=7 end ************
2022-02-10 00:58:08,898 P6999 INFO [Metrics] AUC: 0.960392 - logloss: 0.227187
2022-02-10 00:58:08,899 P6999 INFO Save best model: monitor(max): 0.960392
2022-02-10 00:58:08,910 P6999 INFO --- 343/343 batches finished ---
2022-02-10 00:58:08,952 P6999 INFO Train loss: 0.339117
2022-02-10 00:58:08,952 P6999 INFO ************ Epoch=8 end ************
2022-02-10 00:58:50,085 P6999 INFO [Metrics] AUC: 0.960713 - logloss: 0.225267
2022-02-10 00:58:50,086 P6999 INFO Save best model: monitor(max): 0.960713
2022-02-10 00:58:50,097 P6999 INFO --- 343/343 batches finished ---
2022-02-10 00:58:50,141 P6999 INFO Train loss: 0.339475
2022-02-10 00:58:50,141 P6999 INFO ************ Epoch=9 end ************
2022-02-10 00:59:31,475 P6999 INFO [Metrics] AUC: 0.960831 - logloss: 0.223652
2022-02-10 00:59:31,475 P6999 INFO Save best model: monitor(max): 0.960831
2022-02-10 00:59:31,486 P6999 INFO --- 343/343 batches finished ---
2022-02-10 00:59:31,557 P6999 INFO Train loss: 0.338052
2022-02-10 00:59:31,558 P6999 INFO ************ Epoch=10 end ************
2022-02-10 01:00:11,165 P6999 INFO [Metrics] AUC: 0.961430 - logloss: 0.221364
2022-02-10 01:00:11,165 P6999 INFO Save best model: monitor(max): 0.961430
2022-02-10 01:00:11,176 P6999 INFO --- 343/343 batches finished ---
2022-02-10 01:00:11,219 P6999 INFO Train loss: 0.338125
2022-02-10 01:00:11,219 P6999 INFO ************ Epoch=11 end ************
2022-02-10 01:00:46,522 P6999 INFO [Metrics] AUC: 0.961520 - logloss: 0.222912
2022-02-10 01:00:46,523 P6999 INFO Save best model: monitor(max): 0.961520
2022-02-10 01:00:46,534 P6999 INFO --- 343/343 batches finished ---
2022-02-10 01:00:46,573 P6999 INFO Train loss: 0.337482
2022-02-10 01:00:46,573 P6999 INFO ************ Epoch=12 end ************
2022-02-10 01:01:14,931 P6999 INFO [Metrics] AUC: 0.961850 - logloss: 0.220074
2022-02-10 01:01:14,932 P6999 INFO Save best model: monitor(max): 0.961850
2022-02-10 01:01:14,942 P6999 INFO --- 343/343 batches finished ---
2022-02-10 01:01:14,979 P6999 INFO Train loss: 0.337399
2022-02-10 01:01:14,979 P6999 INFO ************ Epoch=13 end ************
2022-02-10 01:01:39,533 P6999 INFO [Metrics] AUC: 0.962071 - logloss: 0.219876
2022-02-10 01:01:39,534 P6999 INFO Save best model: monitor(max): 0.962071
2022-02-10 01:01:39,544 P6999 INFO --- 343/343 batches finished ---
2022-02-10 01:01:39,582 P6999 INFO Train loss: 0.336847
2022-02-10 01:01:39,583 P6999 INFO ************ Epoch=14 end ************
2022-02-10 01:02:00,167 P6999 INFO [Metrics] AUC: 0.961938 - logloss: 0.218345
2022-02-10 01:02:00,167 P6999 INFO Monitor(max) STOP: 0.961938 !
2022-02-10 01:02:00,167 P6999 INFO Reduce learning rate on plateau: 0.000100
2022-02-10 01:02:00,167 P6999 INFO --- 343/343 batches finished ---
2022-02-10 01:02:00,206 P6999 INFO Train loss: 0.335474
2022-02-10 01:02:00,206 P6999 INFO ************ Epoch=15 end ************
2022-02-10 01:02:16,352 P6999 INFO [Metrics] AUC: 0.967371 - logloss: 0.217296
2022-02-10 01:02:16,352 P6999 INFO Save best model: monitor(max): 0.967371
2022-02-10 01:02:16,362 P6999 INFO --- 343/343 batches finished ---
2022-02-10 01:02:16,412 P6999 INFO Train loss: 0.248857
2022-02-10 01:02:16,412 P6999 INFO ************ Epoch=16 end ************
2022-02-10 01:02:31,833 P6999 INFO [Metrics] AUC: 0.967808 - logloss: 0.238691
2022-02-10 01:02:31,834 P6999 INFO Save best model: monitor(max): 0.967808
2022-02-10 01:02:31,844 P6999 INFO --- 343/343 batches finished ---
2022-02-10 01:02:31,880 P6999 INFO Train loss: 0.172643
2022-02-10 01:02:31,881 P6999 INFO ************ Epoch=17 end ************
2022-02-10 01:02:47,495 P6999 INFO [Metrics] AUC: 0.966264 - logloss: 0.273494
2022-02-10 01:02:47,495 P6999 INFO Monitor(max) STOP: 0.966264 !
2022-02-10 01:02:47,495 P6999 INFO Reduce learning rate on plateau: 0.000010
2022-02-10 01:02:47,495 P6999 INFO --- 343/343 batches finished ---
2022-02-10 01:02:47,535 P6999 INFO Train loss: 0.130550
2022-02-10 01:02:47,535 P6999 INFO ************ Epoch=18 end ************
2022-02-10 01:03:02,707 P6999 INFO [Metrics] AUC: 0.966047 - logloss: 0.282961
2022-02-10 01:03:02,707 P6999 INFO Monitor(max) STOP: 0.966047 !
2022-02-10 01:03:02,707 P6999 INFO Reduce learning rate on plateau: 0.000001
2022-02-10 01:03:02,707 P6999 INFO Early stopping at epoch=19
2022-02-10 01:03:02,708 P6999 INFO --- 343/343 batches finished ---
2022-02-10 01:03:02,743 P6999 INFO Train loss: 0.104320
2022-02-10 01:03:02,744 P6999 INFO Training finished.
2022-02-10 01:03:02,744 P6999 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Movielens/DNN_movielenslatest_x1/movielenslatest_x1_cd32d937/DNN_movielenslatest_x1_001_338b5be6.model
2022-02-10 01:03:02,760 P6999 INFO ****** Validation evaluation ******
2022-02-10 01:03:03,978 P6999 INFO [Metrics] AUC: 0.967808 - logloss: 0.238691
2022-02-10 01:03:04,012 P6999 INFO ******** Test evaluation ********
2022-02-10 01:03:04,012 P6999 INFO Loading data...
2022-02-10 01:03:04,013 P6999 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-02-10 01:03:04,017 P6999 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-02-10 01:03:04,017 P6999 INFO Loading test data done.
2022-02-10 01:03:04,669 P6999 INFO [Metrics] AUC: 0.967764 - logloss: 0.238757

```
