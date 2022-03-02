## DCN_criteo_x1

A hands-on guide to run the DCN model on the Criteo_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

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
Dataset ID: [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [DCN](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DCN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DCN_criteo_x1_tuner_config_03](./DCN_criteo_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DCN_criteo_x1
    nohup python run_expid.py --config ./DCN_criteo_x1_tuner_config_03 --expid DCN_criteo_x1_001_fa7fcfea --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.813798 | 0.438057  |
| 2 | 0.813826 | 0.438040  |
| 3 | 0.813791 | 0.438028  |
| 4 | 0.813723 | 0.438122  |
| 5 | 0.813695 | 0.438129  |
| | | | 
| Avg | 0.813767 | 0.438075 |
| Std | &#177;0.00004927 | &#177;0.00004215 |


### Logs
```python
2022-02-08 10:23:28,157 P102709 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "crossing_layers": "3",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
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
    "model_id": "DCN_criteo_x1_001_fa7fcfea",
    "model_root": "./Criteo/DCN_criteo_x1/",
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
2022-02-08 10:23:28,158 P102709 INFO Set up feature encoder...
2022-02-08 10:23:28,158 P102709 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-02-08 10:23:28,159 P102709 INFO Loading data...
2022-02-08 10:23:28,160 P102709 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-02-08 10:23:32,720 P102709 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-02-08 10:23:33,845 P102709 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-02-08 10:23:33,845 P102709 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-02-08 10:23:33,845 P102709 INFO Loading train data done.
2022-02-08 10:23:38,274 P102709 INFO Total number of parameters: 21343491.
2022-02-08 10:23:38,274 P102709 INFO Start training: 8058 batches/epoch
2022-02-08 10:23:38,274 P102709 INFO ************ Epoch=1 start ************
2022-02-08 10:29:32,735 P102709 INFO [Metrics] AUC: 0.804038 - logloss: 0.447081
2022-02-08 10:29:32,736 P102709 INFO Save best model: monitor(max): 0.804038
2022-02-08 10:29:32,821 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 10:29:32,857 P102709 INFO Train loss: 0.461727
2022-02-08 10:29:32,857 P102709 INFO ************ Epoch=1 end ************
2022-02-08 10:35:32,111 P102709 INFO [Metrics] AUC: 0.806501 - logloss: 0.445216
2022-02-08 10:35:32,112 P102709 INFO Save best model: monitor(max): 0.806501
2022-02-08 10:35:32,207 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 10:35:32,252 P102709 INFO Train loss: 0.456087
2022-02-08 10:35:32,252 P102709 INFO ************ Epoch=2 end ************
2022-02-08 10:41:23,642 P102709 INFO [Metrics] AUC: 0.807541 - logloss: 0.444275
2022-02-08 10:41:23,643 P102709 INFO Save best model: monitor(max): 0.807541
2022-02-08 10:41:23,730 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 10:41:23,767 P102709 INFO Train loss: 0.454675
2022-02-08 10:41:23,767 P102709 INFO ************ Epoch=3 end ************
2022-02-08 10:47:17,846 P102709 INFO [Metrics] AUC: 0.808202 - logloss: 0.443317
2022-02-08 10:47:17,848 P102709 INFO Save best model: monitor(max): 0.808202
2022-02-08 10:47:17,945 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 10:47:17,988 P102709 INFO Train loss: 0.454002
2022-02-08 10:47:17,988 P102709 INFO ************ Epoch=4 end ************
2022-02-08 10:53:15,742 P102709 INFO [Metrics] AUC: 0.808534 - logloss: 0.443053
2022-02-08 10:53:15,744 P102709 INFO Save best model: monitor(max): 0.808534
2022-02-08 10:53:15,835 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 10:53:15,876 P102709 INFO Train loss: 0.453592
2022-02-08 10:53:15,876 P102709 INFO ************ Epoch=5 end ************
2022-02-08 10:59:09,462 P102709 INFO [Metrics] AUC: 0.808895 - logloss: 0.442666
2022-02-08 10:59:09,463 P102709 INFO Save best model: monitor(max): 0.808895
2022-02-08 10:59:09,568 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 10:59:09,611 P102709 INFO Train loss: 0.453279
2022-02-08 10:59:09,611 P102709 INFO ************ Epoch=6 end ************
2022-02-08 11:05:04,853 P102709 INFO [Metrics] AUC: 0.808925 - logloss: 0.442752
2022-02-08 11:05:04,855 P102709 INFO Save best model: monitor(max): 0.808925
2022-02-08 11:05:04,950 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 11:05:04,992 P102709 INFO Train loss: 0.453029
2022-02-08 11:05:04,992 P102709 INFO ************ Epoch=7 end ************
2022-02-08 11:10:59,012 P102709 INFO [Metrics] AUC: 0.809075 - logloss: 0.442800
2022-02-08 11:10:59,013 P102709 INFO Save best model: monitor(max): 0.809075
2022-02-08 11:10:59,108 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 11:10:59,155 P102709 INFO Train loss: 0.452842
2022-02-08 11:10:59,155 P102709 INFO ************ Epoch=8 end ************
2022-02-08 11:16:54,080 P102709 INFO [Metrics] AUC: 0.809259 - logloss: 0.442595
2022-02-08 11:16:54,081 P102709 INFO Save best model: monitor(max): 0.809259
2022-02-08 11:16:54,175 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 11:16:54,215 P102709 INFO Train loss: 0.452654
2022-02-08 11:16:54,215 P102709 INFO ************ Epoch=9 end ************
2022-02-08 11:22:47,460 P102709 INFO [Metrics] AUC: 0.809303 - logloss: 0.442359
2022-02-08 11:22:47,461 P102709 INFO Save best model: monitor(max): 0.809303
2022-02-08 11:22:47,557 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 11:22:47,605 P102709 INFO Train loss: 0.452532
2022-02-08 11:22:47,605 P102709 INFO ************ Epoch=10 end ************
2022-02-08 11:28:41,358 P102709 INFO [Metrics] AUC: 0.809438 - logloss: 0.442187
2022-02-08 11:28:41,359 P102709 INFO Save best model: monitor(max): 0.809438
2022-02-08 11:28:41,453 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 11:28:41,489 P102709 INFO Train loss: 0.452405
2022-02-08 11:28:41,490 P102709 INFO ************ Epoch=11 end ************
2022-02-08 11:34:35,416 P102709 INFO [Metrics] AUC: 0.809448 - logloss: 0.442190
2022-02-08 11:34:35,418 P102709 INFO Save best model: monitor(max): 0.809448
2022-02-08 11:34:35,528 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 11:34:35,566 P102709 INFO Train loss: 0.452277
2022-02-08 11:34:35,566 P102709 INFO ************ Epoch=12 end ************
2022-02-08 11:40:27,620 P102709 INFO [Metrics] AUC: 0.809520 - logloss: 0.442118
2022-02-08 11:40:27,621 P102709 INFO Save best model: monitor(max): 0.809520
2022-02-08 11:40:27,724 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 11:40:27,769 P102709 INFO Train loss: 0.452206
2022-02-08 11:40:27,769 P102709 INFO ************ Epoch=13 end ************
2022-02-08 11:46:20,758 P102709 INFO [Metrics] AUC: 0.809599 - logloss: 0.442031
2022-02-08 11:46:20,759 P102709 INFO Save best model: monitor(max): 0.809599
2022-02-08 11:46:20,860 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 11:46:20,899 P102709 INFO Train loss: 0.452133
2022-02-08 11:46:20,899 P102709 INFO ************ Epoch=14 end ************
2022-02-08 11:52:12,620 P102709 INFO [Metrics] AUC: 0.809656 - logloss: 0.441997
2022-02-08 11:52:12,621 P102709 INFO Save best model: monitor(max): 0.809656
2022-02-08 11:52:12,736 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 11:52:12,774 P102709 INFO Train loss: 0.452063
2022-02-08 11:52:12,774 P102709 INFO ************ Epoch=15 end ************
2022-02-08 11:58:04,475 P102709 INFO [Metrics] AUC: 0.809752 - logloss: 0.441842
2022-02-08 11:58:04,476 P102709 INFO Save best model: monitor(max): 0.809752
2022-02-08 11:58:04,578 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 11:58:04,617 P102709 INFO Train loss: 0.451991
2022-02-08 11:58:04,618 P102709 INFO ************ Epoch=16 end ************
2022-02-08 12:03:53,787 P102709 INFO [Metrics] AUC: 0.809755 - logloss: 0.442277
2022-02-08 12:03:53,788 P102709 INFO Save best model: monitor(max): 0.809755
2022-02-08 12:03:53,896 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 12:03:53,936 P102709 INFO Train loss: 0.451891
2022-02-08 12:03:53,936 P102709 INFO ************ Epoch=17 end ************
2022-02-08 12:09:41,345 P102709 INFO [Metrics] AUC: 0.810012 - logloss: 0.441691
2022-02-08 12:09:41,346 P102709 INFO Save best model: monitor(max): 0.810012
2022-02-08 12:09:41,459 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 12:09:41,499 P102709 INFO Train loss: 0.451834
2022-02-08 12:09:41,499 P102709 INFO ************ Epoch=18 end ************
2022-02-08 12:15:21,752 P102709 INFO [Metrics] AUC: 0.809787 - logloss: 0.441869
2022-02-08 12:15:21,753 P102709 INFO Monitor(max) STOP: 0.809787 !
2022-02-08 12:15:21,753 P102709 INFO Reduce learning rate on plateau: 0.000100
2022-02-08 12:15:21,754 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 12:15:21,794 P102709 INFO Train loss: 0.451798
2022-02-08 12:15:21,794 P102709 INFO ************ Epoch=19 end ************
2022-02-08 12:20:59,893 P102709 INFO [Metrics] AUC: 0.812867 - logloss: 0.438988
2022-02-08 12:20:59,894 P102709 INFO Save best model: monitor(max): 0.812867
2022-02-08 12:20:59,985 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 12:21:00,024 P102709 INFO Train loss: 0.441128
2022-02-08 12:21:00,024 P102709 INFO ************ Epoch=20 end ************
2022-02-08 12:26:38,859 P102709 INFO [Metrics] AUC: 0.813315 - logloss: 0.438618
2022-02-08 12:26:38,860 P102709 INFO Save best model: monitor(max): 0.813315
2022-02-08 12:26:38,965 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 12:26:39,013 P102709 INFO Train loss: 0.437111
2022-02-08 12:26:39,013 P102709 INFO ************ Epoch=21 end ************
2022-02-08 12:32:16,858 P102709 INFO [Metrics] AUC: 0.813480 - logloss: 0.438517
2022-02-08 12:32:16,859 P102709 INFO Save best model: monitor(max): 0.813480
2022-02-08 12:32:16,976 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 12:32:17,020 P102709 INFO Train loss: 0.435371
2022-02-08 12:32:17,021 P102709 INFO ************ Epoch=22 end ************
2022-02-08 12:37:55,785 P102709 INFO [Metrics] AUC: 0.813472 - logloss: 0.438570
2022-02-08 12:37:55,787 P102709 INFO Monitor(max) STOP: 0.813472 !
2022-02-08 12:37:55,787 P102709 INFO Reduce learning rate on plateau: 0.000010
2022-02-08 12:37:55,787 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 12:37:55,833 P102709 INFO Train loss: 0.434110
2022-02-08 12:37:55,834 P102709 INFO ************ Epoch=23 end ************
2022-02-08 12:43:33,989 P102709 INFO [Metrics] AUC: 0.813176 - logloss: 0.439033
2022-02-08 12:43:33,990 P102709 INFO Monitor(max) STOP: 0.813176 !
2022-02-08 12:43:33,990 P102709 INFO Reduce learning rate on plateau: 0.000001
2022-02-08 12:43:33,991 P102709 INFO Early stopping at epoch=24
2022-02-08 12:43:33,991 P102709 INFO --- 8058/8058 batches finished ---
2022-02-08 12:43:34,041 P102709 INFO Train loss: 0.430381
2022-02-08 12:43:34,042 P102709 INFO Training finished.
2022-02-08 12:43:34,042 P102709 INFO Load best model: /cache/FuxiCTR/benchmarks_modelarts/Criteo/DCN_criteo_x1/criteo_x1_7b681156/DCN_criteo_x1_001_fa7fcfea.model
2022-02-08 12:43:34,107 P102709 INFO ****** Validation evaluation ******
2022-02-08 12:43:57,914 P102709 INFO [Metrics] AUC: 0.813480 - logloss: 0.438517
2022-02-08 12:43:57,992 P102709 INFO ******** Test evaluation ********
2022-02-08 12:43:57,993 P102709 INFO Loading data...
2022-02-08 12:43:57,993 P102709 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-02-08 12:43:58,771 P102709 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-02-08 12:43:58,771 P102709 INFO Loading test data done.
2022-02-08 12:44:13,341 P102709 INFO [Metrics] AUC: 0.813798 - logloss: 0.438057

```
