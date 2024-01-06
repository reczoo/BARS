## DCNv2_criteo_x1

A hands-on guide to run the DCNv2 model on the Criteo_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DCNv2](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DCNv2.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DCNv2_criteo_x1_tuner_config_04](./DCNv2_criteo_x1_tuner_config_04). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DCNv2_criteo_x1
    nohup python run_expid.py --config ./DCNv2_criteo_x1_tuner_config_04 --expid DCNv2_criteo_x1_001_28c60688 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.814195 | 0.437788  |
| 2 | 0.813970 | 0.438088  |
| 3 | 0.813831 | 0.438203  |
| 4 | 0.814079 | 0.437967  |
| 5 | 0.813620 | 0.438391  |
| Avg | 0.813939 | 0.438087 |
| Std | &#177;0.00019976 | &#177;0.00020478 |


### Logs
```python
2022-02-08 14:57:52,912 P10110 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "dnn_activations": "relu",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "low_rank": "32",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCNv2",
    "model_id": "DCNv2_criteo_x1_001_28c60688",
    "model_root": "./Criteo/DCNv2_criteo_x1/",
    "model_structure": "parallel",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_cross_layers": "3",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_dnn_hidden_units": "[500, 500, 500]",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "stacked_dnn_hidden_units": "[400, 400, 400]",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_hdf5": "True",
    "use_low_rank_mixture": "False",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-02-08 14:57:52,912 P10110 INFO Set up feature encoder...
2022-02-08 14:57:52,913 P10110 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-02-08 14:57:52,913 P10110 INFO Loading data...
2022-02-08 14:57:52,914 P10110 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-02-08 14:57:57,665 P10110 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-02-08 14:57:58,910 P10110 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-02-08 14:57:58,910 P10110 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-02-08 14:57:58,910 P10110 INFO Loading train data done.
2022-02-08 14:58:02,932 P10110 INFO Total number of parameters: 22018021.
2022-02-08 14:58:02,932 P10110 INFO Start training: 8058 batches/epoch
2022-02-08 14:58:02,932 P10110 INFO ************ Epoch=1 start ************
2022-02-08 15:04:22,718 P10110 INFO [Metrics] AUC: 0.805664 - logloss: 0.445767
2022-02-08 15:04:22,719 P10110 INFO Save best model: monitor(max): 0.805664
2022-02-08 15:04:22,803 P10110 INFO --- 8058/8058 batches finished ---
2022-02-08 15:04:22,849 P10110 INFO Train loss: 0.460206
2022-02-08 15:04:22,849 P10110 INFO ************ Epoch=1 end ************
2022-02-08 15:10:39,708 P10110 INFO [Metrics] AUC: 0.807963 - logloss: 0.443604
2022-02-08 15:10:39,709 P10110 INFO Save best model: monitor(max): 0.807963
2022-02-08 15:10:39,802 P10110 INFO --- 8058/8058 batches finished ---
2022-02-08 15:10:39,843 P10110 INFO Train loss: 0.453986
2022-02-08 15:10:39,843 P10110 INFO ************ Epoch=2 end ************
2022-02-08 15:16:54,905 P10110 INFO [Metrics] AUC: 0.809111 - logloss: 0.442599
2022-02-08 15:16:54,907 P10110 INFO Save best model: monitor(max): 0.809111
2022-02-08 15:16:54,999 P10110 INFO --- 8058/8058 batches finished ---
2022-02-08 15:16:55,038 P10110 INFO Train loss: 0.452322
2022-02-08 15:16:55,039 P10110 INFO ************ Epoch=3 end ************
2022-02-08 15:23:10,697 P10110 INFO [Metrics] AUC: 0.809730 - logloss: 0.442035
2022-02-08 15:23:10,698 P10110 INFO Save best model: monitor(max): 0.809730
2022-02-08 15:23:10,800 P10110 INFO --- 8058/8058 batches finished ---
2022-02-08 15:23:10,843 P10110 INFO Train loss: 0.451461
2022-02-08 15:23:10,844 P10110 INFO ************ Epoch=4 end ************
2022-02-08 15:29:25,861 P10110 INFO [Metrics] AUC: 0.810019 - logloss: 0.441650
2022-02-08 15:29:25,862 P10110 INFO Save best model: monitor(max): 0.810019
2022-02-08 15:29:25,964 P10110 INFO --- 8058/8058 batches finished ---
2022-02-08 15:29:26,017 P10110 INFO Train loss: 0.450873
2022-02-08 15:29:26,017 P10110 INFO ************ Epoch=5 end ************
2022-02-08 15:35:37,373 P10110 INFO [Metrics] AUC: 0.810351 - logloss: 0.441365
2022-02-08 15:35:37,374 P10110 INFO Save best model: monitor(max): 0.810351
2022-02-08 15:35:37,466 P10110 INFO --- 8058/8058 batches finished ---
2022-02-08 15:35:37,518 P10110 INFO Train loss: 0.450467
2022-02-08 15:35:37,518 P10110 INFO ************ Epoch=6 end ************
2022-02-08 15:41:50,641 P10110 INFO [Metrics] AUC: 0.810259 - logloss: 0.441482
2022-02-08 15:41:50,642 P10110 INFO Monitor(max) STOP: 0.810259 !
2022-02-08 15:41:50,642 P10110 INFO Reduce learning rate on plateau: 0.000100
2022-02-08 15:41:50,642 P10110 INFO --- 8058/8058 batches finished ---
2022-02-08 15:41:50,694 P10110 INFO Train loss: 0.450085
2022-02-08 15:41:50,694 P10110 INFO ************ Epoch=7 end ************
2022-02-08 15:48:03,145 P10110 INFO [Metrics] AUC: 0.813417 - logloss: 0.438588
2022-02-08 15:48:03,146 P10110 INFO Save best model: monitor(max): 0.813417
2022-02-08 15:48:03,254 P10110 INFO --- 8058/8058 batches finished ---
2022-02-08 15:48:03,309 P10110 INFO Train loss: 0.439067
2022-02-08 15:48:03,309 P10110 INFO ************ Epoch=8 end ************
2022-02-08 15:54:15,049 P10110 INFO [Metrics] AUC: 0.813831 - logloss: 0.438283
2022-02-08 15:54:15,050 P10110 INFO Save best model: monitor(max): 0.813831
2022-02-08 15:54:15,150 P10110 INFO --- 8058/8058 batches finished ---
2022-02-08 15:54:15,201 P10110 INFO Train loss: 0.435239
2022-02-08 15:54:15,202 P10110 INFO ************ Epoch=9 end ************
2022-02-08 16:00:27,628 P10110 INFO [Metrics] AUC: 0.813887 - logloss: 0.438259
2022-02-08 16:00:27,630 P10110 INFO Save best model: monitor(max): 0.813887
2022-02-08 16:00:27,724 P10110 INFO --- 8058/8058 batches finished ---
2022-02-08 16:00:27,778 P10110 INFO Train loss: 0.433561
2022-02-08 16:00:27,778 P10110 INFO ************ Epoch=10 end ************
2022-02-08 16:06:49,425 P10110 INFO [Metrics] AUC: 0.813798 - logloss: 0.438414
2022-02-08 16:06:49,426 P10110 INFO Monitor(max) STOP: 0.813798 !
2022-02-08 16:06:49,426 P10110 INFO Reduce learning rate on plateau: 0.000010
2022-02-08 16:06:49,426 P10110 INFO --- 8058/8058 batches finished ---
2022-02-08 16:06:49,480 P10110 INFO Train loss: 0.432303
2022-02-08 16:06:49,480 P10110 INFO ************ Epoch=11 end ************
2022-02-08 16:13:07,014 P10110 INFO [Metrics] AUC: 0.813577 - logloss: 0.438834
2022-02-08 16:13:07,015 P10110 INFO Monitor(max) STOP: 0.813577 !
2022-02-08 16:13:07,016 P10110 INFO Reduce learning rate on plateau: 0.000001
2022-02-08 16:13:07,016 P10110 INFO Early stopping at epoch=12
2022-02-08 16:13:07,016 P10110 INFO --- 8058/8058 batches finished ---
2022-02-08 16:13:07,067 P10110 INFO Train loss: 0.428313
2022-02-08 16:13:07,067 P10110 INFO Training finished.
2022-02-08 16:13:07,067 P10110 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/DCNv2_criteo_x1/criteo_x1_7b681156/DCNv2_criteo_x1_001_28c60688.model
2022-02-08 16:13:07,163 P10110 INFO ****** Validation evaluation ******
2022-02-08 16:13:32,989 P10110 INFO [Metrics] AUC: 0.813887 - logloss: 0.438259
2022-02-08 16:13:33,075 P10110 INFO ******** Test evaluation ********
2022-02-08 16:13:33,075 P10110 INFO Loading data...
2022-02-08 16:13:33,076 P10110 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-02-08 16:13:33,677 P10110 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-02-08 16:13:33,677 P10110 INFO Loading test data done.
2022-02-08 16:13:48,821 P10110 INFO [Metrics] AUC: 0.814195 - logloss: 0.437788

```
