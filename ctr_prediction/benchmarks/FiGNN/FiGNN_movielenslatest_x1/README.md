## FiGNN_movielenslatest_x1

A hands-on guide to run the FiGNN model on the Movielenslatest_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
  GPU: Tesla P100 16G
  RAM: 503G

  ```

+ Software

  ```python
  CUDA: 10.2
  python: 3.6.4
  pytorch: 1.0.0
  pandas: 0.22.0
  numpy: 1.19.1
  scipy: 1.2.2
  sklearn: 0.19.1
  pyyaml: 6.0
  h5py: 2.8.0
  tqdm: 4.28.1
  fuxictr: 1.1.0
  ```

### Dataset
Dataset ID: [Movielenslatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Movielenslatest/README.md#Movielenslatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [FiGNN](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FiGNN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FiGNN_movielenslatest_x1_tuner_config_02](./FiGNN_movielenslatest_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FiGNN_movielenslatest_x1
    nohup python run_expid.py --config ./FiGNN_movielenslatest_x1_tuner_config_02 --expid FiGNN_movielenslatest_x1_005_73cc20f9 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.952476 | 0.255931  |


### Logs
```python
2022-01-30 14:09:11,248 P144370 INFO {
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
    "gnn_layers": "13",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FiGNN",
    "model_id": "FiGNN_movielenslatest_x1_005_73cc20f9",
    "model_root": "./Movielens/FiGNN_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reuse_graph_layer": "False",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Movielens/MovielensLatest_x1/test.csv",
    "train_data": "../data/Movielens/MovielensLatest_x1/train.csv",
    "use_gru": "False",
    "use_hdf5": "True",
    "use_residual": "True",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-30 14:09:11,249 P144370 INFO Set up feature encoder...
2022-01-30 14:09:11,249 P144370 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-01-30 14:09:11,250 P144370 INFO Loading data...
2022-01-30 14:09:11,252 P144370 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-01-30 14:09:11,283 P144370 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-01-30 14:09:11,292 P144370 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-01-30 14:09:11,292 P144370 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-01-30 14:09:11,292 P144370 INFO Loading train data done.
2022-01-30 14:09:14,968 P144370 INFO Total number of parameters: 910440.
2022-01-30 14:09:14,969 P144370 INFO Start training: 343 batches/epoch
2022-01-30 14:09:14,969 P144370 INFO ************ Epoch=1 start ************
2022-01-30 14:11:07,910 P144370 INFO [Metrics] AUC: 0.923442 - logloss: 0.312623
2022-01-30 14:11:07,910 P144370 INFO Save best model: monitor(max): 0.923442
2022-01-30 14:11:07,918 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:11:07,961 P144370 INFO Train loss: 0.399469
2022-01-30 14:11:07,961 P144370 INFO ************ Epoch=1 end ************
2022-01-30 14:13:00,629 P144370 INFO [Metrics] AUC: 0.927625 - logloss: 0.305546
2022-01-30 14:13:00,629 P144370 INFO Save best model: monitor(max): 0.927625
2022-01-30 14:13:00,637 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:13:00,680 P144370 INFO Train loss: 0.356182
2022-01-30 14:13:00,680 P144370 INFO ************ Epoch=2 end ************
2022-01-30 14:14:53,395 P144370 INFO [Metrics] AUC: 0.930771 - logloss: 0.299925
2022-01-30 14:14:53,396 P144370 INFO Save best model: monitor(max): 0.930771
2022-01-30 14:14:53,404 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:14:53,447 P144370 INFO Train loss: 0.346050
2022-01-30 14:14:53,447 P144370 INFO ************ Epoch=3 end ************
2022-01-30 14:16:46,154 P144370 INFO [Metrics] AUC: 0.933006 - logloss: 0.296211
2022-01-30 14:16:46,155 P144370 INFO Save best model: monitor(max): 0.933006
2022-01-30 14:16:46,163 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:16:46,205 P144370 INFO Train loss: 0.336733
2022-01-30 14:16:46,205 P144370 INFO ************ Epoch=4 end ************
2022-01-30 14:18:38,454 P144370 INFO [Metrics] AUC: 0.935608 - logloss: 0.289599
2022-01-30 14:18:38,454 P144370 INFO Save best model: monitor(max): 0.935608
2022-01-30 14:18:38,462 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:18:38,508 P144370 INFO Train loss: 0.337660
2022-01-30 14:18:38,509 P144370 INFO ************ Epoch=5 end ************
2022-01-30 14:20:33,264 P144370 INFO [Metrics] AUC: 0.936703 - logloss: 0.286358
2022-01-30 14:20:33,265 P144370 INFO Save best model: monitor(max): 0.936703
2022-01-30 14:20:33,273 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:20:33,315 P144370 INFO Train loss: 0.341741
2022-01-30 14:20:33,315 P144370 INFO ************ Epoch=6 end ************
2022-01-30 14:22:27,143 P144370 INFO [Metrics] AUC: 0.938212 - logloss: 0.283172
2022-01-30 14:22:27,144 P144370 INFO Save best model: monitor(max): 0.938212
2022-01-30 14:22:27,151 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:22:27,195 P144370 INFO Train loss: 0.337850
2022-01-30 14:22:27,196 P144370 INFO ************ Epoch=7 end ************
2022-01-30 14:24:19,849 P144370 INFO [Metrics] AUC: 0.939737 - logloss: 0.278994
2022-01-30 14:24:19,849 P144370 INFO Save best model: monitor(max): 0.939737
2022-01-30 14:24:19,857 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:24:19,904 P144370 INFO Train loss: 0.334483
2022-01-30 14:24:19,904 P144370 INFO ************ Epoch=8 end ************
2022-01-30 14:26:12,649 P144370 INFO [Metrics] AUC: 0.940539 - logloss: 0.277398
2022-01-30 14:26:12,649 P144370 INFO Save best model: monitor(max): 0.940539
2022-01-30 14:26:12,658 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:26:12,700 P144370 INFO Train loss: 0.332032
2022-01-30 14:26:12,701 P144370 INFO ************ Epoch=9 end ************
2022-01-30 14:28:05,686 P144370 INFO [Metrics] AUC: 0.941661 - logloss: 0.275401
2022-01-30 14:28:05,687 P144370 INFO Save best model: monitor(max): 0.941661
2022-01-30 14:28:05,695 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:28:05,737 P144370 INFO Train loss: 0.329526
2022-01-30 14:28:05,737 P144370 INFO ************ Epoch=10 end ************
2022-01-30 14:29:58,591 P144370 INFO [Metrics] AUC: 0.941805 - logloss: 0.275073
2022-01-30 14:29:58,592 P144370 INFO Save best model: monitor(max): 0.941805
2022-01-30 14:29:58,601 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:29:58,644 P144370 INFO Train loss: 0.326775
2022-01-30 14:29:58,644 P144370 INFO ************ Epoch=11 end ************
2022-01-30 14:31:53,730 P144370 INFO [Metrics] AUC: 0.942406 - logloss: 0.273679
2022-01-30 14:31:53,731 P144370 INFO Save best model: monitor(max): 0.942406
2022-01-30 14:31:53,739 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:31:53,782 P144370 INFO Train loss: 0.324983
2022-01-30 14:31:53,782 P144370 INFO ************ Epoch=12 end ************
2022-01-30 14:33:47,379 P144370 INFO [Metrics] AUC: 0.942633 - logloss: 0.273496
2022-01-30 14:33:47,380 P144370 INFO Save best model: monitor(max): 0.942633
2022-01-30 14:33:47,389 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:33:47,440 P144370 INFO Train loss: 0.324413
2022-01-30 14:33:47,440 P144370 INFO ************ Epoch=13 end ************
2022-01-30 14:35:41,433 P144370 INFO [Metrics] AUC: 0.943212 - logloss: 0.271988
2022-01-30 14:35:41,433 P144370 INFO Save best model: monitor(max): 0.943212
2022-01-30 14:35:41,441 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:35:41,482 P144370 INFO Train loss: 0.324231
2022-01-30 14:35:41,482 P144370 INFO ************ Epoch=14 end ************
2022-01-30 14:37:34,897 P144370 INFO [Metrics] AUC: 0.943312 - logloss: 0.273023
2022-01-30 14:37:34,897 P144370 INFO Save best model: monitor(max): 0.943312
2022-01-30 14:37:34,906 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:37:34,950 P144370 INFO Train loss: 0.325013
2022-01-30 14:37:34,950 P144370 INFO ************ Epoch=15 end ************
2022-01-30 14:39:31,129 P144370 INFO [Metrics] AUC: 0.943474 - logloss: 0.271737
2022-01-30 14:39:31,130 P144370 INFO Save best model: monitor(max): 0.943474
2022-01-30 14:39:31,139 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:39:31,187 P144370 INFO Train loss: 0.324266
2022-01-30 14:39:31,188 P144370 INFO ************ Epoch=16 end ************
2022-01-30 14:41:27,425 P144370 INFO [Metrics] AUC: 0.943561 - logloss: 0.271724
2022-01-30 14:41:27,425 P144370 INFO Save best model: monitor(max): 0.943561
2022-01-30 14:41:27,434 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:41:27,479 P144370 INFO Train loss: 0.325677
2022-01-30 14:41:27,480 P144370 INFO ************ Epoch=17 end ************
2022-01-30 14:43:23,720 P144370 INFO [Metrics] AUC: 0.943515 - logloss: 0.271433
2022-01-30 14:43:23,721 P144370 INFO Monitor(max) STOP: 0.943515 !
2022-01-30 14:43:23,721 P144370 INFO Reduce learning rate on plateau: 0.000100
2022-01-30 14:43:23,721 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:43:23,770 P144370 INFO Train loss: 0.327054
2022-01-30 14:43:23,771 P144370 INFO ************ Epoch=18 end ************
2022-01-30 14:45:20,359 P144370 INFO [Metrics] AUC: 0.951086 - logloss: 0.256450
2022-01-30 14:45:20,361 P144370 INFO Save best model: monitor(max): 0.951086
2022-01-30 14:45:20,369 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:45:20,429 P144370 INFO Train loss: 0.277358
2022-01-30 14:45:20,429 P144370 INFO ************ Epoch=19 end ************
2022-01-30 14:47:05,248 P144370 INFO [Metrics] AUC: 0.952423 - logloss: 0.256050
2022-01-30 14:47:05,249 P144370 INFO Save best model: monitor(max): 0.952423
2022-01-30 14:47:05,257 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:47:05,301 P144370 INFO Train loss: 0.248293
2022-01-30 14:47:05,301 P144370 INFO ************ Epoch=20 end ************
2022-01-30 14:48:34,782 P144370 INFO [Metrics] AUC: 0.952161 - logloss: 0.262081
2022-01-30 14:48:34,783 P144370 INFO Monitor(max) STOP: 0.952161 !
2022-01-30 14:48:34,783 P144370 INFO Reduce learning rate on plateau: 0.000010
2022-01-30 14:48:34,783 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:48:34,831 P144370 INFO Train loss: 0.234663
2022-01-30 14:48:34,832 P144370 INFO ************ Epoch=21 end ************
2022-01-30 14:50:03,625 P144370 INFO [Metrics] AUC: 0.951856 - logloss: 0.270343
2022-01-30 14:50:03,625 P144370 INFO Monitor(max) STOP: 0.951856 !
2022-01-30 14:50:03,625 P144370 INFO Reduce learning rate on plateau: 0.000001
2022-01-30 14:50:03,625 P144370 INFO Early stopping at epoch=22
2022-01-30 14:50:03,625 P144370 INFO --- 343/343 batches finished ---
2022-01-30 14:50:03,671 P144370 INFO Train loss: 0.210410
2022-01-30 14:50:03,671 P144370 INFO Training finished.
2022-01-30 14:50:03,671 P144370 INFO Load best model: /home/ma-user/XXX/benchmarks/Movielens/FiGNN_movielenslatest_x1/movielenslatest_x1_cd32d937/FiGNN_movielenslatest_x1_005_73cc20f9.model
2022-01-30 14:50:03,793 P144370 INFO ****** Validation evaluation ******
2022-01-30 14:50:09,511 P144370 INFO [Metrics] AUC: 0.952423 - logloss: 0.256050
2022-01-30 14:50:09,560 P144370 INFO ******** Test evaluation ********
2022-01-30 14:50:09,560 P144370 INFO Loading data...
2022-01-30 14:50:09,561 P144370 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-01-30 14:50:09,565 P144370 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-01-30 14:50:09,565 P144370 INFO Loading test data done.
2022-01-30 14:50:12,337 P144370 INFO [Metrics] AUC: 0.952476 - logloss: 0.255931

```
