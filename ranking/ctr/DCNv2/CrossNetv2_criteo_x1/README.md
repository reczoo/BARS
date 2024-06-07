## CrossNetv2_criteo_x1

A hands-on guide to run the DCNv2 model on the Criteo_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DCNv2](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DCNv2.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CrossNetv2_criteo_x1_tuner_config_01](./CrossNetv2_criteo_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CrossNetv2_criteo_x1
    nohup python run_expid.py --config ./CrossNetv2_criteo_x1_tuner_config_01 --expid DCNv2_criteo_x1_008_43e17586 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.812727 | 0.439273  |


### Logs
```python
2022-01-21 13:08:34,850 P838 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
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
    "gpu": "7",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "low_rank": "32",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCNv2",
    "model_id": "DCNv2_criteo_x1_008_43e17586",
    "model_root": "./Criteo/DCNv2_criteo_x1/",
    "model_structure": "crossnet_only",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_cross_layers": "5",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_dnn_hidden_units": "[500, 500, 500]",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "stacked_dnn_hidden_units": "[500, 500, 500]",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_hdf5": "True",
    "use_low_rank_mixture": "False",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-21 13:08:34,851 P838 INFO Set up feature encoder...
2022-01-21 13:08:34,851 P838 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-01-21 13:08:34,851 P838 INFO Loading data...
2022-01-21 13:08:34,853 P838 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-01-21 13:08:39,630 P838 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-01-21 13:08:40,887 P838 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-01-21 13:08:40,887 P838 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-01-21 13:08:40,887 P838 INFO Loading train data done.
2022-01-21 13:08:47,396 P838 INFO Total number of parameters: 21626001.
2022-01-21 13:08:47,396 P838 INFO Start training: 8058 batches/epoch
2022-01-21 13:08:47,396 P838 INFO ************ Epoch=1 start ************
2022-01-21 13:20:15,336 P838 INFO [Metrics] AUC: 0.805415 - logloss: 0.445949
2022-01-21 13:20:15,338 P838 INFO Save best model: monitor(max): 0.805415
2022-01-21 13:20:15,611 P838 INFO --- 8058/8058 batches finished ---
2022-01-21 13:20:15,648 P838 INFO Train loss: 0.460166
2022-01-21 13:20:15,648 P838 INFO ************ Epoch=1 end ************
2022-01-21 13:31:44,402 P838 INFO [Metrics] AUC: 0.807542 - logloss: 0.444014
2022-01-21 13:31:44,403 P838 INFO Save best model: monitor(max): 0.807542
2022-01-21 13:31:44,501 P838 INFO --- 8058/8058 batches finished ---
2022-01-21 13:31:44,543 P838 INFO Train loss: 0.453695
2022-01-21 13:31:44,543 P838 INFO ************ Epoch=2 end ************
2022-01-21 13:43:09,510 P838 INFO [Metrics] AUC: 0.808471 - logloss: 0.443366
2022-01-21 13:43:09,512 P838 INFO Save best model: monitor(max): 0.808471
2022-01-21 13:43:09,610 P838 INFO --- 8058/8058 batches finished ---
2022-01-21 13:43:09,652 P838 INFO Train loss: 0.452368
2022-01-21 13:43:09,652 P838 INFO ************ Epoch=3 end ************
2022-01-21 13:54:33,232 P838 INFO [Metrics] AUC: 0.809029 - logloss: 0.442669
2022-01-21 13:54:33,233 P838 INFO Save best model: monitor(max): 0.809029
2022-01-21 13:54:33,347 P838 INFO --- 8058/8058 batches finished ---
2022-01-21 13:54:33,389 P838 INFO Train loss: 0.451598
2022-01-21 13:54:33,389 P838 INFO ************ Epoch=4 end ************
2022-01-21 14:05:50,169 P838 INFO [Metrics] AUC: 0.809012 - logloss: 0.442708
2022-01-21 14:05:50,171 P838 INFO Monitor(max) STOP: 0.809012 !
2022-01-21 14:05:50,171 P838 INFO Reduce learning rate on plateau: 0.000100
2022-01-21 14:05:50,171 P838 INFO --- 8058/8058 batches finished ---
2022-01-21 14:05:50,213 P838 INFO Train loss: 0.451008
2022-01-21 14:05:50,213 P838 INFO ************ Epoch=5 end ************
2022-01-21 14:16:57,754 P838 INFO [Metrics] AUC: 0.812119 - logloss: 0.439936
2022-01-21 14:16:57,755 P838 INFO Save best model: monitor(max): 0.812119
2022-01-21 14:16:57,863 P838 INFO --- 8058/8058 batches finished ---
2022-01-21 14:16:57,906 P838 INFO Train loss: 0.440113
2022-01-21 14:16:57,906 P838 INFO ************ Epoch=6 end ************
2022-01-21 14:28:00,035 P838 INFO [Metrics] AUC: 0.812447 - logloss: 0.439698
2022-01-21 14:28:00,037 P838 INFO Save best model: monitor(max): 0.812447
2022-01-21 14:28:00,143 P838 INFO --- 8058/8058 batches finished ---
2022-01-21 14:28:00,187 P838 INFO Train loss: 0.436292
2022-01-21 14:28:00,187 P838 INFO ************ Epoch=7 end ************
2022-01-21 14:38:59,200 P838 INFO [Metrics] AUC: 0.812431 - logloss: 0.439811
2022-01-21 14:38:59,201 P838 INFO Monitor(max) STOP: 0.812431 !
2022-01-21 14:38:59,201 P838 INFO Reduce learning rate on plateau: 0.000010
2022-01-21 14:38:59,201 P838 INFO --- 8058/8058 batches finished ---
2022-01-21 14:38:59,242 P838 INFO Train loss: 0.434514
2022-01-21 14:38:59,242 P838 INFO ************ Epoch=8 end ************
2022-01-21 14:49:53,533 P838 INFO [Metrics] AUC: 0.812175 - logloss: 0.440318
2022-01-21 14:49:53,535 P838 INFO Monitor(max) STOP: 0.812175 !
2022-01-21 14:49:53,535 P838 INFO Reduce learning rate on plateau: 0.000001
2022-01-21 14:49:53,535 P838 INFO Early stopping at epoch=9
2022-01-21 14:49:53,535 P838 INFO --- 8058/8058 batches finished ---
2022-01-21 14:49:53,577 P838 INFO Train loss: 0.430529
2022-01-21 14:49:53,578 P838 INFO Training finished.
2022-01-21 14:49:53,578 P838 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/DCNv2_criteo_x1/criteo_x1_7b681156/DCNv2_criteo_x1_008_43e17586.model
2022-01-21 14:49:58,555 P838 INFO ****** Validation evaluation ******
2022-01-21 14:50:23,813 P838 INFO [Metrics] AUC: 0.812447 - logloss: 0.439698
2022-01-21 14:50:23,889 P838 INFO ******** Test evaluation ********
2022-01-21 14:50:23,889 P838 INFO Loading data...
2022-01-21 14:50:23,889 P838 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-01-21 14:50:24,674 P838 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-01-21 14:50:24,674 P838 INFO Loading test data done.
2022-01-21 14:50:39,865 P838 INFO [Metrics] AUC: 0.812727 - logloss: 0.439273

```
