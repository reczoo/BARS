## FiBiNET_criteo_x1

A hands-on guide to run the FiBiNET model on the Criteo_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [FiBiNET](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FiBiNET.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FiBiNET_criteo_x1_tuner_config_03](./FiBiNET_criteo_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FiBiNET_criteo_x1
    nohup python run_expid.py --config ./FiBiNET_criteo_x1_tuner_config_03 --expid FiBiNET_criteo_x1_022_19ca3145 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.813010 | 0.438826  |


### Logs
```python
2022-01-26 21:02:40,760 P24905 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "bilinear_type": "field_interaction",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "5",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FiBiNET",
    "model_id": "FiBiNET_criteo_x1_022_19ca3145",
    "model_root": "./Criteo/FiBiNET_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.5",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "12",
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
2022-01-26 21:02:40,761 P24905 INFO Set up feature encoder...
2022-01-26 21:02:40,761 P24905 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-01-26 21:02:40,761 P24905 INFO Loading data...
2022-01-26 21:02:40,762 P24905 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-01-26 21:02:45,755 P24905 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-01-26 21:02:47,443 P24905 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-01-26 21:02:47,444 P24905 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-01-26 21:02:47,444 P24905 INFO Loading train data done.
2022-01-26 21:02:55,369 P24905 INFO Total number of parameters: 29273411.
2022-01-26 21:02:55,369 P24905 INFO Start training: 8058 batches/epoch
2022-01-26 21:02:55,369 P24905 INFO ************ Epoch=1 start ************
2022-01-27 02:52:58,367 P24905 INFO [Metrics] AUC: 0.804734 - logloss: 0.446802
2022-01-27 02:52:58,369 P24905 INFO Save best model: monitor(max): 0.804734
2022-01-27 02:52:58,748 P24905 INFO --- 8058/8058 batches finished ---
2022-01-27 02:52:58,796 P24905 INFO Train loss: 0.463223
2022-01-27 02:52:58,799 P24905 INFO ************ Epoch=1 end ************
2022-01-27 08:41:10,562 P24905 INFO [Metrics] AUC: 0.806694 - logloss: 0.444863
2022-01-27 08:41:10,564 P24905 INFO Save best model: monitor(max): 0.806694
2022-01-27 08:41:10,760 P24905 INFO --- 8058/8058 batches finished ---
2022-01-27 08:41:10,820 P24905 INFO Train loss: 0.458331
2022-01-27 08:41:10,821 P24905 INFO ************ Epoch=2 end ************
2022-01-27 14:29:11,233 P24905 INFO [Metrics] AUC: 0.807428 - logloss: 0.444110
2022-01-27 14:29:11,234 P24905 INFO Save best model: monitor(max): 0.807428
2022-01-27 14:29:11,407 P24905 INFO --- 8058/8058 batches finished ---
2022-01-27 14:29:11,465 P24905 INFO Train loss: 0.457180
2022-01-27 14:29:11,467 P24905 INFO ************ Epoch=3 end ************
2022-01-27 20:21:26,350 P24905 INFO [Metrics] AUC: 0.807553 - logloss: 0.444230
2022-01-27 20:21:26,351 P24905 INFO Save best model: monitor(max): 0.807553
2022-01-27 20:21:26,527 P24905 INFO --- 8058/8058 batches finished ---
2022-01-27 20:21:26,584 P24905 INFO Train loss: 0.456538
2022-01-27 20:21:26,587 P24905 INFO ************ Epoch=4 end ************
2022-01-27 23:33:03,924 P24905 INFO [Metrics] AUC: 0.807543 - logloss: 0.444385
2022-01-27 23:33:03,925 P24905 INFO Monitor(max) STOP: 0.807543 !
2022-01-27 23:33:03,925 P24905 INFO Reduce learning rate on plateau: 0.000100
2022-01-27 23:33:03,925 P24905 INFO --- 8058/8058 batches finished ---
2022-01-27 23:33:03,982 P24905 INFO Train loss: 0.456162
2022-01-27 23:33:03,984 P24905 INFO ************ Epoch=5 end ************
2022-01-28 00:47:11,829 P24905 INFO [Metrics] AUC: 0.811937 - logloss: 0.440024
2022-01-28 00:47:11,830 P24905 INFO Save best model: monitor(max): 0.811937
2022-01-28 00:47:11,986 P24905 INFO --- 8058/8058 batches finished ---
2022-01-28 00:47:12,047 P24905 INFO Train loss: 0.444398
2022-01-28 00:47:12,049 P24905 INFO ************ Epoch=6 end ************
2022-01-28 02:01:03,630 P24905 INFO [Metrics] AUC: 0.812579 - logloss: 0.439391
2022-01-28 02:01:03,631 P24905 INFO Save best model: monitor(max): 0.812579
2022-01-28 02:01:03,812 P24905 INFO --- 8058/8058 batches finished ---
2022-01-28 02:01:03,867 P24905 INFO Train loss: 0.439721
2022-01-28 02:01:03,869 P24905 INFO ************ Epoch=7 end ************
2022-01-28 03:14:58,309 P24905 INFO [Metrics] AUC: 0.812741 - logloss: 0.439211
2022-01-28 03:14:58,310 P24905 INFO Save best model: monitor(max): 0.812741
2022-01-28 03:14:58,498 P24905 INFO --- 8058/8058 batches finished ---
2022-01-28 03:14:58,553 P24905 INFO Train loss: 0.437548
2022-01-28 03:14:58,556 P24905 INFO ************ Epoch=8 end ************
2022-01-28 04:29:37,260 P24905 INFO [Metrics] AUC: 0.812610 - logloss: 0.439299
2022-01-28 04:29:37,261 P24905 INFO Monitor(max) STOP: 0.812610 !
2022-01-28 04:29:37,261 P24905 INFO Reduce learning rate on plateau: 0.000010
2022-01-28 04:29:37,261 P24905 INFO --- 8058/8058 batches finished ---
2022-01-28 04:29:37,318 P24905 INFO Train loss: 0.435817
2022-01-28 04:29:37,321 P24905 INFO ************ Epoch=9 end ************
2022-01-28 05:44:05,669 P24905 INFO [Metrics] AUC: 0.812144 - logloss: 0.439911
2022-01-28 05:44:05,670 P24905 INFO Monitor(max) STOP: 0.812144 !
2022-01-28 05:44:05,670 P24905 INFO Reduce learning rate on plateau: 0.000001
2022-01-28 05:44:05,671 P24905 INFO Early stopping at epoch=10
2022-01-28 05:44:05,671 P24905 INFO --- 8058/8058 batches finished ---
2022-01-28 05:44:05,733 P24905 INFO Train loss: 0.430854
2022-01-28 05:44:05,736 P24905 INFO Training finished.
2022-01-28 05:44:05,736 P24905 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/FiBiNET_criteo_x1/criteo_x1_7b681156/FiBiNET_criteo_x1_022_19ca3145.model
2022-01-28 05:44:10,623 P24905 INFO ****** Validation evaluation ******
2022-01-28 05:48:13,373 P24905 INFO [Metrics] AUC: 0.812741 - logloss: 0.439211
2022-01-28 05:48:13,443 P24905 INFO ******** Test evaluation ********
2022-01-28 05:48:13,443 P24905 INFO Loading data...
2022-01-28 05:48:13,443 P24905 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-01-28 05:48:14,236 P24905 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-01-28 05:48:14,236 P24905 INFO Loading test data done.
2022-01-28 05:50:38,029 P24905 INFO [Metrics] AUC: 0.813010 - logloss: 0.438826

```
