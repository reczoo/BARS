## HFM_avazu_x1

A hands-on guide to run the HFM model on the Avazu_x1 dataset.

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
Dataset ID: [Avazu_x1](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [HFM](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/HFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HFM_avazu_x1_tuner_config_01](./HFM_avazu_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HFM_avazu_x1
    nohup python run_expid.py --config ./HFM_avazu_x1_tuner_config_01 --expid HFM_avazu_x1_007_bf21aada --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.762124 | 0.367616  |


### Logs
```python
2022-01-27 13:13:14,972 P90921 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.0001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "6",
    "hidden_activations": "relu",
    "hidden_units": "[64, 64, 64]",
    "interaction_type": "circular_correlation",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "HFM",
    "model_id": "HFM_avazu_x1_007_bf21aada",
    "model_root": "./Avazu/HFM_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_dnn": "False",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-01-27 13:13:14,973 P90921 INFO Set up feature encoder...
2022-01-27 13:13:14,973 P90921 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-01-27 13:13:14,973 P90921 INFO Loading data...
2022-01-27 13:13:14,974 P90921 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-01-27 13:13:17,392 P90921 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-01-27 13:13:17,724 P90921 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-01-27 13:13:17,724 P90921 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-01-27 13:13:17,724 P90921 INFO Loading train data done.
2022-01-27 13:13:23,463 P90921 INFO Total number of parameters: 14284600.
2022-01-27 13:13:23,463 P90921 INFO Start training: 6910 batches/epoch
2022-01-27 13:13:23,463 P90921 INFO ************ Epoch=1 start ************
2022-01-27 13:21:54,802 P90921 INFO [Metrics] AUC: 0.741769 - logloss: 0.398226
2022-01-27 13:21:54,805 P90921 INFO Save best model: monitor(max): 0.741769
2022-01-27 13:21:55,041 P90921 INFO --- 6910/6910 batches finished ---
2022-01-27 13:21:55,080 P90921 INFO Train loss: 0.411714
2022-01-27 13:21:55,080 P90921 INFO ************ Epoch=1 end ************
2022-01-27 13:30:23,790 P90921 INFO [Metrics] AUC: 0.744189 - logloss: 0.396990
2022-01-27 13:30:23,792 P90921 INFO Save best model: monitor(max): 0.744189
2022-01-27 13:30:23,864 P90921 INFO --- 6910/6910 batches finished ---
2022-01-27 13:30:23,907 P90921 INFO Train loss: 0.409285
2022-01-27 13:30:23,907 P90921 INFO ************ Epoch=2 end ************
2022-01-27 13:38:53,776 P90921 INFO [Metrics] AUC: 0.744416 - logloss: 0.398849
2022-01-27 13:38:53,778 P90921 INFO Save best model: monitor(max): 0.744416
2022-01-27 13:38:53,854 P90921 INFO --- 6910/6910 batches finished ---
2022-01-27 13:38:53,901 P90921 INFO Train loss: 0.409359
2022-01-27 13:38:53,902 P90921 INFO ************ Epoch=3 end ************
2022-01-27 13:47:23,396 P90921 INFO [Metrics] AUC: 0.744625 - logloss: 0.397605
2022-01-27 13:47:23,399 P90921 INFO Save best model: monitor(max): 0.744625
2022-01-27 13:47:23,463 P90921 INFO --- 6910/6910 batches finished ---
2022-01-27 13:47:23,511 P90921 INFO Train loss: 0.409353
2022-01-27 13:47:23,511 P90921 INFO ************ Epoch=4 end ************
2022-01-27 13:55:50,695 P90921 INFO [Metrics] AUC: 0.744716 - logloss: 0.396756
2022-01-27 13:55:50,698 P90921 INFO Save best model: monitor(max): 0.744716
2022-01-27 13:55:50,766 P90921 INFO --- 6910/6910 batches finished ---
2022-01-27 13:55:50,806 P90921 INFO Train loss: 0.409111
2022-01-27 13:55:50,807 P90921 INFO ************ Epoch=5 end ************
2022-01-27 14:04:19,350 P90921 INFO [Metrics] AUC: 0.744069 - logloss: 0.396889
2022-01-27 14:04:19,352 P90921 INFO Monitor(max) STOP: 0.744069 !
2022-01-27 14:04:19,353 P90921 INFO Reduce learning rate on plateau: 0.000100
2022-01-27 14:04:19,353 P90921 INFO --- 6910/6910 batches finished ---
2022-01-27 14:04:19,398 P90921 INFO Train loss: 0.409002
2022-01-27 14:04:19,398 P90921 INFO ************ Epoch=6 end ************
2022-01-27 14:12:48,761 P90921 INFO [Metrics] AUC: 0.746115 - logloss: 0.396682
2022-01-27 14:12:48,765 P90921 INFO Save best model: monitor(max): 0.746115
2022-01-27 14:12:48,834 P90921 INFO --- 6910/6910 batches finished ---
2022-01-27 14:12:48,877 P90921 INFO Train loss: 0.398428
2022-01-27 14:12:48,877 P90921 INFO ************ Epoch=7 end ************
2022-01-27 14:21:17,796 P90921 INFO [Metrics] AUC: 0.744325 - logloss: 0.396911
2022-01-27 14:21:17,800 P90921 INFO Monitor(max) STOP: 0.744325 !
2022-01-27 14:21:17,800 P90921 INFO Reduce learning rate on plateau: 0.000010
2022-01-27 14:21:17,800 P90921 INFO --- 6910/6910 batches finished ---
2022-01-27 14:21:17,843 P90921 INFO Train loss: 0.395847
2022-01-27 14:21:17,843 P90921 INFO ************ Epoch=8 end ************
2022-01-27 14:29:45,699 P90921 INFO [Metrics] AUC: 0.743978 - logloss: 0.397735
2022-01-27 14:29:45,702 P90921 INFO Monitor(max) STOP: 0.743978 !
2022-01-27 14:29:45,703 P90921 INFO Reduce learning rate on plateau: 0.000001
2022-01-27 14:29:45,703 P90921 INFO Early stopping at epoch=9
2022-01-27 14:29:45,703 P90921 INFO --- 6910/6910 batches finished ---
2022-01-27 14:29:45,745 P90921 INFO Train loss: 0.392396
2022-01-27 14:29:45,746 P90921 INFO Training finished.
2022-01-27 14:29:45,746 P90921 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/HFM_avazu_x1/avazu_x1_3fb65689/HFM_avazu_x1_007_bf21aada.model
2022-01-27 14:29:49,068 P90921 INFO ****** Validation evaluation ******
2022-01-27 14:30:04,675 P90921 INFO [Metrics] AUC: 0.746115 - logloss: 0.396682
2022-01-27 14:30:04,736 P90921 INFO ******** Test evaluation ********
2022-01-27 14:30:04,737 P90921 INFO Loading data...
2022-01-27 14:30:04,738 P90921 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-01-27 14:30:05,651 P90921 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-01-27 14:30:05,651 P90921 INFO Loading test data done.
2022-01-27 14:30:39,580 P90921 INFO [Metrics] AUC: 0.762124 - logloss: 0.367616

```
