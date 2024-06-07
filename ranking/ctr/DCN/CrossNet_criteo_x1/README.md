## CrossNet_criteo_x1

A hands-on guide to run the DCN model on the Criteo_x1 dataset.

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
Dataset ID: [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo#Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DCN](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DCN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CrossNet_criteo_x1_tuner_config_03](./CrossNet_criteo_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CrossNet_criteo_x1
    nohup python run_expid.py --config ./CrossNet_criteo_x1_tuner_config_03 --expid DCN_criteo_x1_001_95179395 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.804147 | 0.447206  |


### Logs
```python
2022-01-22 13:50:29,195 P32119 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "crossing_layers": "12",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "None",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCN",
    "model_id": "DCN_criteo_x1_001_95179395",
    "model_root": "./Criteo/DCN_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
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
    "verbose": "1",
    "version": "pytorch"
}
2022-01-22 13:50:29,195 P32119 INFO Set up feature encoder...
2022-01-22 13:50:29,196 P32119 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-01-22 13:50:29,196 P32119 INFO Loading data...
2022-01-22 13:50:29,198 P32119 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-01-22 13:50:33,682 P32119 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-01-22 13:50:34,810 P32119 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-01-22 13:50:34,810 P32119 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-01-22 13:50:34,810 P32119 INFO Loading train data done.
2022-01-22 13:50:39,495 P32119 INFO Total number of parameters: 20872911.
2022-01-22 13:50:39,495 P32119 INFO Start training: 8058 batches/epoch
2022-01-22 13:50:39,495 P32119 INFO ************ Epoch=1 start ************
2022-01-22 14:06:15,356 P32119 INFO [Metrics] AUC: 0.793362 - logloss: 0.456760
2022-01-22 14:06:15,358 P32119 INFO Save best model: monitor(max): 0.793362
2022-01-22 14:06:15,431 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 14:06:15,476 P32119 INFO Train loss: 0.467689
2022-01-22 14:06:15,476 P32119 INFO ************ Epoch=1 end ************
2022-01-22 14:21:52,468 P32119 INFO [Metrics] AUC: 0.795489 - logloss: 0.455006
2022-01-22 14:21:52,470 P32119 INFO Save best model: monitor(max): 0.795489
2022-01-22 14:21:52,636 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 14:21:52,693 P32119 INFO Train loss: 0.462050
2022-01-22 14:21:52,693 P32119 INFO ************ Epoch=2 end ************
2022-01-22 14:37:24,847 P32119 INFO [Metrics] AUC: 0.796730 - logloss: 0.453806
2022-01-22 14:37:24,848 P32119 INFO Save best model: monitor(max): 0.796730
2022-01-22 14:37:25,009 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 14:37:25,059 P32119 INFO Train loss: 0.461084
2022-01-22 14:37:25,059 P32119 INFO ************ Epoch=3 end ************
2022-01-22 14:52:59,505 P32119 INFO [Metrics] AUC: 0.797286 - logloss: 0.453251
2022-01-22 14:52:59,507 P32119 INFO Save best model: monitor(max): 0.797286
2022-01-22 14:52:59,656 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 14:52:59,700 P32119 INFO Train loss: 0.460754
2022-01-22 14:52:59,700 P32119 INFO ************ Epoch=4 end ************
2022-01-22 15:08:33,564 P32119 INFO [Metrics] AUC: 0.797541 - logloss: 0.452961
2022-01-22 15:08:33,565 P32119 INFO Save best model: monitor(max): 0.797541
2022-01-22 15:08:33,741 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 15:08:33,787 P32119 INFO Train loss: 0.460553
2022-01-22 15:08:33,788 P32119 INFO ************ Epoch=5 end ************
2022-01-22 15:24:08,579 P32119 INFO [Metrics] AUC: 0.797777 - logloss: 0.452763
2022-01-22 15:24:08,581 P32119 INFO Save best model: monitor(max): 0.797777
2022-01-22 15:24:08,747 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 15:24:08,791 P32119 INFO Train loss: 0.460388
2022-01-22 15:24:08,791 P32119 INFO ************ Epoch=6 end ************
2022-01-22 15:39:40,256 P32119 INFO [Metrics] AUC: 0.798042 - logloss: 0.452715
2022-01-22 15:39:40,257 P32119 INFO Save best model: monitor(max): 0.798042
2022-01-22 15:39:40,400 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 15:39:40,450 P32119 INFO Train loss: 0.460269
2022-01-22 15:39:40,451 P32119 INFO ************ Epoch=7 end ************
2022-01-22 15:55:07,742 P32119 INFO [Metrics] AUC: 0.798346 - logloss: 0.452655
2022-01-22 15:55:07,744 P32119 INFO Save best model: monitor(max): 0.798346
2022-01-22 15:55:07,886 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 15:55:07,929 P32119 INFO Train loss: 0.460147
2022-01-22 15:55:07,929 P32119 INFO ************ Epoch=8 end ************
2022-01-22 16:10:33,649 P32119 INFO [Metrics] AUC: 0.798647 - logloss: 0.452045
2022-01-22 16:10:33,650 P32119 INFO Save best model: monitor(max): 0.798647
2022-01-22 16:10:33,937 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 16:10:33,979 P32119 INFO Train loss: 0.460037
2022-01-22 16:10:33,979 P32119 INFO ************ Epoch=9 end ************
2022-01-22 16:25:55,180 P32119 INFO [Metrics] AUC: 0.798734 - logloss: 0.452117
2022-01-22 16:25:55,181 P32119 INFO Save best model: monitor(max): 0.798734
2022-01-22 16:25:55,354 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 16:25:55,398 P32119 INFO Train loss: 0.459969
2022-01-22 16:25:55,398 P32119 INFO ************ Epoch=10 end ************
2022-01-22 16:41:13,291 P32119 INFO [Metrics] AUC: 0.798879 - logloss: 0.452343
2022-01-22 16:41:13,293 P32119 INFO Save best model: monitor(max): 0.798879
2022-01-22 16:41:13,440 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 16:41:13,482 P32119 INFO Train loss: 0.459825
2022-01-22 16:41:13,482 P32119 INFO ************ Epoch=11 end ************
2022-01-22 16:56:31,320 P32119 INFO [Metrics] AUC: 0.799004 - logloss: 0.452083
2022-01-22 16:56:31,322 P32119 INFO Save best model: monitor(max): 0.799004
2022-01-22 16:56:31,445 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 16:56:31,497 P32119 INFO Train loss: 0.459749
2022-01-22 16:56:31,497 P32119 INFO ************ Epoch=12 end ************
2022-01-22 17:11:46,884 P32119 INFO [Metrics] AUC: 0.798969 - logloss: 0.452558
2022-01-22 17:11:46,886 P32119 INFO Monitor(max) STOP: 0.798969 !
2022-01-22 17:11:46,886 P32119 INFO Reduce learning rate on plateau: 0.000100
2022-01-22 17:11:46,886 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 17:11:46,930 P32119 INFO Train loss: 0.459695
2022-01-22 17:11:46,930 P32119 INFO ************ Epoch=13 end ************
2022-01-22 17:27:02,572 P32119 INFO [Metrics] AUC: 0.802562 - logloss: 0.448619
2022-01-22 17:27:02,573 P32119 INFO Save best model: monitor(max): 0.802562
2022-01-22 17:27:02,726 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 17:27:02,771 P32119 INFO Train loss: 0.451300
2022-01-22 17:27:02,771 P32119 INFO ************ Epoch=14 end ************
2022-01-22 17:42:14,503 P32119 INFO [Metrics] AUC: 0.803274 - logloss: 0.448020
2022-01-22 17:42:14,505 P32119 INFO Save best model: monitor(max): 0.803274
2022-01-22 17:42:14,632 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 17:42:14,677 P32119 INFO Train loss: 0.448364
2022-01-22 17:42:14,677 P32119 INFO ************ Epoch=15 end ************
2022-01-22 17:50:35,047 P32119 INFO [Metrics] AUC: 0.803562 - logloss: 0.448028
2022-01-22 17:50:35,048 P32119 INFO Save best model: monitor(max): 0.803562
2022-01-22 17:50:35,167 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 17:50:35,220 P32119 INFO Train loss: 0.446960
2022-01-22 17:50:35,221 P32119 INFO ************ Epoch=16 end ************
2022-01-22 17:58:34,377 P32119 INFO [Metrics] AUC: 0.803779 - logloss: 0.447682
2022-01-22 17:58:34,378 P32119 INFO Save best model: monitor(max): 0.803779
2022-01-22 17:58:34,487 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 17:58:34,536 P32119 INFO Train loss: 0.445949
2022-01-22 17:58:34,536 P32119 INFO ************ Epoch=17 end ************
2022-01-22 18:06:27,403 P32119 INFO [Metrics] AUC: 0.803688 - logloss: 0.447696
2022-01-22 18:06:27,404 P32119 INFO Monitor(max) STOP: 0.803688 !
2022-01-22 18:06:27,405 P32119 INFO Reduce learning rate on plateau: 0.000010
2022-01-22 18:06:27,405 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 18:06:27,452 P32119 INFO Train loss: 0.445124
2022-01-22 18:06:27,452 P32119 INFO ************ Epoch=18 end ************
2022-01-22 18:14:26,287 P32119 INFO [Metrics] AUC: 0.803618 - logloss: 0.447941
2022-01-22 18:14:26,288 P32119 INFO Monitor(max) STOP: 0.803618 !
2022-01-22 18:14:26,288 P32119 INFO Reduce learning rate on plateau: 0.000001
2022-01-22 18:14:26,288 P32119 INFO Early stopping at epoch=19
2022-01-22 18:14:26,289 P32119 INFO --- 8058/8058 batches finished ---
2022-01-22 18:14:26,335 P32119 INFO Train loss: 0.441598
2022-01-22 18:14:26,335 P32119 INFO Training finished.
2022-01-22 18:14:26,335 P32119 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Criteo/DCN_criteo_x1/criteo_x1_7b681156/DCN_criteo_x1_001_95179395.model
2022-01-22 18:14:26,408 P32119 INFO ****** Validation evaluation ******
2022-01-22 18:14:52,302 P32119 INFO [Metrics] AUC: 0.803779 - logloss: 0.447682
2022-01-22 18:14:52,341 P32119 INFO ******** Test evaluation ********
2022-01-22 18:14:52,341 P32119 INFO Loading data...
2022-01-22 18:14:52,342 P32119 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-01-22 18:14:52,931 P32119 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-01-22 18:14:52,932 P32119 INFO Loading test data done.
2022-01-22 18:15:07,775 P32119 INFO [Metrics] AUC: 0.804147 - logloss: 0.447206

```
