## NFM_criteo_x1

A hands-on guide to run the NFM model on the Criteo_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

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
Dataset ID: [Criteo_x1](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [NFM](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/NFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [NFM_criteo_x1_tuner_config_02](./NFM_criteo_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd NFM_criteo_x1
    nohup python run_expid.py --config ./NFM_criteo_x1_tuner_config_02 --expid NFM_criteo_x1_002_e07a761d --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.805309 | 0.445929  |


### Logs
```python
2022-01-27 10:27:09,557 P790 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.0001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "NFM",
    "model_id": "NFM_criteo_x1_002_e07a761d",
    "model_root": "./Criteo/NFM_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
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
2022-01-27 10:27:09,557 P790 INFO Set up feature encoder...
2022-01-27 10:27:09,557 P790 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-01-27 10:27:09,558 P790 INFO Loading data...
2022-01-27 10:27:09,559 P790 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-01-27 10:27:14,130 P790 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-01-27 10:27:15,286 P790 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-01-27 10:27:15,287 P790 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-01-27 10:27:15,287 P790 INFO Loading train data done.
2022-01-27 10:27:19,499 P790 INFO Total number of parameters: 23275077.
2022-01-27 10:27:19,499 P790 INFO Start training: 8058 batches/epoch
2022-01-27 10:27:19,499 P790 INFO ************ Epoch=1 start ************
2022-01-27 10:36:45,038 P790 INFO [Metrics] AUC: 0.789969 - logloss: 0.459614
2022-01-27 10:36:45,040 P790 INFO Save best model: monitor(max): 0.789969
2022-01-27 10:36:45,283 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 10:36:45,335 P790 INFO Train loss: 0.482604
2022-01-27 10:36:45,335 P790 INFO ************ Epoch=1 end ************
2022-01-27 10:46:06,927 P790 INFO [Metrics] AUC: 0.791132 - logloss: 0.458416
2022-01-27 10:46:06,929 P790 INFO Save best model: monitor(max): 0.791132
2022-01-27 10:46:07,051 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 10:46:07,101 P790 INFO Train loss: 0.479225
2022-01-27 10:46:07,101 P790 INFO ************ Epoch=2 end ************
2022-01-27 10:55:22,741 P790 INFO [Metrics] AUC: 0.791435 - logloss: 0.458381
2022-01-27 10:55:22,743 P790 INFO Save best model: monitor(max): 0.791435
2022-01-27 10:55:22,853 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 10:55:22,906 P790 INFO Train loss: 0.478869
2022-01-27 10:55:22,906 P790 INFO ************ Epoch=3 end ************
2022-01-27 11:04:42,903 P790 INFO [Metrics] AUC: 0.791579 - logloss: 0.458558
2022-01-27 11:04:42,905 P790 INFO Save best model: monitor(max): 0.791579
2022-01-27 11:04:43,039 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 11:04:43,090 P790 INFO Train loss: 0.478779
2022-01-27 11:04:43,090 P790 INFO ************ Epoch=4 end ************
2022-01-27 11:14:06,779 P790 INFO [Metrics] AUC: 0.791553 - logloss: 0.458013
2022-01-27 11:14:06,781 P790 INFO Monitor(max) STOP: 0.791553 !
2022-01-27 11:14:06,781 P790 INFO Reduce learning rate on plateau: 0.000100
2022-01-27 11:14:06,781 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 11:14:06,827 P790 INFO Train loss: 0.478696
2022-01-27 11:14:06,827 P790 INFO ************ Epoch=5 end ************
2022-01-27 11:23:28,518 P790 INFO [Metrics] AUC: 0.801002 - logloss: 0.450060
2022-01-27 11:23:28,520 P790 INFO Save best model: monitor(max): 0.801002
2022-01-27 11:23:28,647 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 11:23:28,693 P790 INFO Train loss: 0.459200
2022-01-27 11:23:28,694 P790 INFO ************ Epoch=6 end ************
2022-01-27 11:32:45,744 P790 INFO [Metrics] AUC: 0.802119 - logloss: 0.449031
2022-01-27 11:32:45,746 P790 INFO Save best model: monitor(max): 0.802119
2022-01-27 11:32:45,853 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 11:32:45,901 P790 INFO Train loss: 0.454980
2022-01-27 11:32:45,902 P790 INFO ************ Epoch=7 end ************
2022-01-27 11:41:57,916 P790 INFO [Metrics] AUC: 0.802542 - logloss: 0.448560
2022-01-27 11:41:57,918 P790 INFO Save best model: monitor(max): 0.802542
2022-01-27 11:41:58,028 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 11:41:58,067 P790 INFO Train loss: 0.453968
2022-01-27 11:41:58,067 P790 INFO ************ Epoch=8 end ************
2022-01-27 11:51:12,903 P790 INFO [Metrics] AUC: 0.802760 - logloss: 0.448382
2022-01-27 11:51:12,904 P790 INFO Save best model: monitor(max): 0.802760
2022-01-27 11:51:13,005 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 11:51:13,054 P790 INFO Train loss: 0.453428
2022-01-27 11:51:13,054 P790 INFO ************ Epoch=9 end ************
2022-01-27 12:00:23,329 P790 INFO [Metrics] AUC: 0.802878 - logloss: 0.448259
2022-01-27 12:00:23,331 P790 INFO Save best model: monitor(max): 0.802878
2022-01-27 12:00:23,432 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 12:00:23,471 P790 INFO Train loss: 0.453103
2022-01-27 12:00:23,471 P790 INFO ************ Epoch=10 end ************
2022-01-27 12:09:37,095 P790 INFO [Metrics] AUC: 0.803062 - logloss: 0.448113
2022-01-27 12:09:37,096 P790 INFO Save best model: monitor(max): 0.803062
2022-01-27 12:09:37,225 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 12:09:37,264 P790 INFO Train loss: 0.452849
2022-01-27 12:09:37,265 P790 INFO ************ Epoch=11 end ************
2022-01-27 12:18:49,823 P790 INFO [Metrics] AUC: 0.803159 - logloss: 0.448101
2022-01-27 12:18:49,825 P790 INFO Save best model: monitor(max): 0.803159
2022-01-27 12:18:49,947 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 12:18:49,999 P790 INFO Train loss: 0.452696
2022-01-27 12:18:49,999 P790 INFO ************ Epoch=12 end ************
2022-01-27 12:28:02,366 P790 INFO [Metrics] AUC: 0.803144 - logloss: 0.447975
2022-01-27 12:28:02,367 P790 INFO Monitor(max) STOP: 0.803144 !
2022-01-27 12:28:02,368 P790 INFO Reduce learning rate on plateau: 0.000010
2022-01-27 12:28:02,368 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 12:28:02,422 P790 INFO Train loss: 0.452560
2022-01-27 12:28:02,423 P790 INFO ************ Epoch=13 end ************
2022-01-27 12:37:11,519 P790 INFO [Metrics] AUC: 0.804622 - logloss: 0.446711
2022-01-27 12:37:11,520 P790 INFO Save best model: monitor(max): 0.804622
2022-01-27 12:37:11,621 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 12:37:11,672 P790 INFO Train loss: 0.446513
2022-01-27 12:37:11,673 P790 INFO ************ Epoch=14 end ************
2022-01-27 12:46:23,864 P790 INFO [Metrics] AUC: 0.804868 - logloss: 0.446495
2022-01-27 12:46:23,866 P790 INFO Save best model: monitor(max): 0.804868
2022-01-27 12:46:23,966 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 12:46:24,015 P790 INFO Train loss: 0.444930
2022-01-27 12:46:24,016 P790 INFO ************ Epoch=15 end ************
2022-01-27 12:55:37,183 P790 INFO [Metrics] AUC: 0.804941 - logloss: 0.446372
2022-01-27 12:55:37,184 P790 INFO Save best model: monitor(max): 0.804941
2022-01-27 12:55:37,288 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 12:55:37,338 P790 INFO Train loss: 0.444239
2022-01-27 12:55:37,338 P790 INFO ************ Epoch=16 end ************
2022-01-27 13:04:46,074 P790 INFO [Metrics] AUC: 0.804984 - logloss: 0.446363
2022-01-27 13:04:46,075 P790 INFO Save best model: monitor(max): 0.804984
2022-01-27 13:04:46,185 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 13:04:46,237 P790 INFO Train loss: 0.443742
2022-01-27 13:04:46,237 P790 INFO ************ Epoch=17 end ************
2022-01-27 13:13:54,384 P790 INFO [Metrics] AUC: 0.804965 - logloss: 0.446411
2022-01-27 13:13:54,386 P790 INFO Monitor(max) STOP: 0.804965 !
2022-01-27 13:13:54,386 P790 INFO Reduce learning rate on plateau: 0.000001
2022-01-27 13:13:54,386 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 13:13:54,434 P790 INFO Train loss: 0.443329
2022-01-27 13:13:54,435 P790 INFO ************ Epoch=18 end ************
2022-01-27 13:23:02,060 P790 INFO [Metrics] AUC: 0.804924 - logloss: 0.446448
2022-01-27 13:23:02,062 P790 INFO Monitor(max) STOP: 0.804924 !
2022-01-27 13:23:02,062 P790 INFO Reduce learning rate on plateau: 0.000001
2022-01-27 13:23:02,062 P790 INFO Early stopping at epoch=19
2022-01-27 13:23:02,062 P790 INFO --- 8058/8058 batches finished ---
2022-01-27 13:23:02,108 P790 INFO Train loss: 0.441685
2022-01-27 13:23:02,108 P790 INFO Training finished.
2022-01-27 13:23:02,108 P790 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/NFM_criteo_x1/criteo_x1_7b681156/NFM_criteo_x1_002_e07a761d.model
2022-01-27 13:23:04,810 P790 INFO ****** Validation evaluation ******
2022-01-27 13:23:31,566 P790 INFO [Metrics] AUC: 0.804984 - logloss: 0.446363
2022-01-27 13:23:31,651 P790 INFO ******** Test evaluation ********
2022-01-27 13:23:31,652 P790 INFO Loading data...
2022-01-27 13:23:31,652 P790 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-01-27 13:23:32,427 P790 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-01-27 13:23:32,427 P790 INFO Loading test data done.
2022-01-27 13:23:48,059 P790 INFO [Metrics] AUC: 0.805309 - logloss: 0.445929

```
