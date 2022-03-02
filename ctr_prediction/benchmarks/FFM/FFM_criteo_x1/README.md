## FFM_criteo_x1

A hands-on guide to run the FFM model on the Criteo_x1 dataset.

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
Dataset ID: [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [FFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FFM_criteo_x1_tuner_config_01](./FFM_criteo_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FFM_criteo_x1
    nohup python run_expid.py --config ./FFM_criteo_x1_tuner_config_01 --expid FFMv2_criteo_x1_004_9c79983e --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.806024 | 0.445553  |


### Logs
```python
2022-03-02 09:09:51,863 P50145 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "embedding_dim": "5",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FFMv2",
    "model_id": "FFMv2_criteo_x1_004_9c79983e",
    "model_root": "./Criteo/FFM_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-06",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-03-02 09:09:51,864 P50145 INFO Set up feature encoder...
2022-03-02 09:09:51,872 P50145 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-03-02 09:09:51,881 P50145 INFO Loading data...
2022-03-02 09:09:51,883 P50145 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-03-02 09:10:01,189 P50145 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-03-02 09:10:03,425 P50145 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-03-02 09:10:03,426 P50145 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-03-02 09:10:03,426 P50145 INFO Loading train data done.
2022-03-02 09:10:16,147 P50145 INFO Total number of parameters: 398486357.
2022-03-02 09:10:16,147 P50145 INFO Start training: 8058 batches/epoch
2022-03-02 09:10:16,147 P50145 INFO ************ Epoch=1 start ************
2022-03-02 09:37:28,225 P50145 INFO [Metrics] AUC: 0.802158 - logloss: 0.449176
2022-03-02 09:37:28,226 P50145 INFO Save best model: monitor(max): 0.802158
2022-03-02 09:37:29,765 P50145 INFO --- 8058/8058 batches finished ---
2022-03-02 09:37:29,803 P50145 INFO Train loss: 0.464259
2022-03-02 09:37:29,803 P50145 INFO ************ Epoch=1 end ************
2022-03-02 10:04:41,657 P50145 INFO [Metrics] AUC: 0.803524 - logloss: 0.447939
2022-03-02 10:04:41,659 P50145 INFO Save best model: monitor(max): 0.803524
2022-03-02 10:04:44,769 P50145 INFO --- 8058/8058 batches finished ---
2022-03-02 10:04:44,810 P50145 INFO Train loss: 0.457541
2022-03-02 10:04:44,810 P50145 INFO ************ Epoch=2 end ************
2022-03-02 10:32:06,360 P50145 INFO [Metrics] AUC: 0.804030 - logloss: 0.447534
2022-03-02 10:32:06,362 P50145 INFO Save best model: monitor(max): 0.804030
2022-03-02 10:32:09,515 P50145 INFO --- 8058/8058 batches finished ---
2022-03-02 10:32:09,557 P50145 INFO Train loss: 0.456167
2022-03-02 10:32:09,557 P50145 INFO ************ Epoch=3 end ************
2022-03-02 10:59:28,192 P50145 INFO [Metrics] AUC: 0.804104 - logloss: 0.447507
2022-03-02 10:59:28,194 P50145 INFO Save best model: monitor(max): 0.804104
2022-03-02 10:59:31,325 P50145 INFO --- 8058/8058 batches finished ---
2022-03-02 10:59:31,363 P50145 INFO Train loss: 0.455502
2022-03-02 10:59:31,363 P50145 INFO ************ Epoch=4 end ************
2022-03-02 11:26:53,442 P50145 INFO [Metrics] AUC: 0.804063 - logloss: 0.447481
2022-03-02 11:26:53,443 P50145 INFO Monitor(max) STOP: 0.804063 !
2022-03-02 11:26:53,443 P50145 INFO Reduce learning rate on plateau: 0.000100
2022-03-02 11:26:53,443 P50145 INFO --- 8058/8058 batches finished ---
2022-03-02 11:26:53,484 P50145 INFO Train loss: 0.455166
2022-03-02 11:26:53,484 P50145 INFO ************ Epoch=5 end ************
2022-03-02 11:54:12,609 P50145 INFO [Metrics] AUC: 0.805779 - logloss: 0.445920
2022-03-02 11:54:12,611 P50145 INFO Save best model: monitor(max): 0.805779
2022-03-02 11:54:15,756 P50145 INFO --- 8058/8058 batches finished ---
2022-03-02 11:54:15,795 P50145 INFO Train loss: 0.441719
2022-03-02 11:54:15,795 P50145 INFO ************ Epoch=6 end ************
2022-03-02 12:21:34,292 P50145 INFO [Metrics] AUC: 0.805576 - logloss: 0.446153
2022-03-02 12:21:34,294 P50145 INFO Monitor(max) STOP: 0.805576 !
2022-03-02 12:21:34,294 P50145 INFO Reduce learning rate on plateau: 0.000010
2022-03-02 12:21:34,294 P50145 INFO --- 8058/8058 batches finished ---
2022-03-02 12:21:34,337 P50145 INFO Train loss: 0.438513
2022-03-02 12:21:34,337 P50145 INFO ************ Epoch=7 end ************
2022-03-02 12:48:57,456 P50145 INFO [Metrics] AUC: 0.805575 - logloss: 0.446160
2022-03-02 12:48:57,458 P50145 INFO Monitor(max) STOP: 0.805575 !
2022-03-02 12:48:57,458 P50145 INFO Reduce learning rate on plateau: 0.000001
2022-03-02 12:48:57,458 P50145 INFO Early stopping at epoch=8
2022-03-02 12:48:57,458 P50145 INFO --- 8058/8058 batches finished ---
2022-03-02 12:48:57,499 P50145 INFO Train loss: 0.435023
2022-03-02 12:48:57,499 P50145 INFO Training finished.
2022-03-02 12:48:57,499 P50145 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Criteo/FFM_criteo_x1/criteo_x1_7b681156/FFMv2_criteo_x1_004_9c79983e.model
2022-03-02 12:48:58,574 P50145 INFO ****** Validation evaluation ******
2022-03-02 12:49:58,866 P50145 INFO [Metrics] AUC: 0.805779 - logloss: 0.445920
2022-03-02 12:49:58,942 P50145 INFO ******** Test evaluation ********
2022-03-02 12:49:58,943 P50145 INFO Loading data...
2022-03-02 12:49:58,943 P50145 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-03-02 12:50:00,199 P50145 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-03-02 12:50:00,200 P50145 INFO Loading test data done.
2022-03-02 12:50:33,982 P50145 INFO [Metrics] AUC: 0.806024 - logloss: 0.445553

```
