## WideDeep_avazu_x1

A hands-on guide to run the WideDeep model on the Avazu_x1 dataset.

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
Dataset ID: [Avazu_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [WideDeep](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/WideDeep.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [WideDeep_avazu_x1_tuner_config_03](./WideDeep_avazu_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd WideDeep_avazu_x1
    nohup python run_expid.py --config ./WideDeep_avazu_x1_tuner_config_03 --expid WideDeep_avazu_x1_002_d8c146ac --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.764917 | 0.366495  |
| 2 | 0.764663 | 0.366701  |
| 3 | 0.765479 | 0.366452  |
| 4 | 0.763954 | 0.367605  |
| 5 | 0.764176 | 0.367386  |
| Avg | 0.764638 | 0.366928 |
| Std | &#177;0.00054167 | &#177;0.00047617 |


### Logs
```python
2022-02-06 16:42:19,258 P75877 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.005",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "WideDeep",
    "model_id": "WideDeep_avazu_x1_002_d8c146ac",
    "model_root": "./Avazu/WideDeep_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
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
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-02-06 16:42:19,259 P75877 INFO Set up feature encoder...
2022-02-06 16:42:19,259 P75877 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-02-06 16:42:19,259 P75877 INFO Loading data...
2022-02-06 16:42:19,260 P75877 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-02-06 16:42:21,753 P75877 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-02-06 16:42:22,135 P75877 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-02-06 16:42:22,135 P75877 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-02-06 16:42:22,135 P75877 INFO Loading train data done.
2022-02-06 16:42:25,871 P75877 INFO Total number of parameters: 14696590.
2022-02-06 16:42:25,871 P75877 INFO Start training: 6910 batches/epoch
2022-02-06 16:42:25,871 P75877 INFO ************ Epoch=1 start ************
2022-02-06 16:49:45,980 P75877 INFO [Metrics] AUC: 0.742079 - logloss: 0.398475
2022-02-06 16:49:45,982 P75877 INFO Save best model: monitor(max): 0.742079
2022-02-06 16:49:46,205 P75877 INFO --- 6910/6910 batches finished ---
2022-02-06 16:49:46,253 P75877 INFO Train loss: 0.429063
2022-02-06 16:49:46,253 P75877 INFO ************ Epoch=1 end ************
2022-02-06 16:57:10,604 P75877 INFO [Metrics] AUC: 0.745307 - logloss: 0.396935
2022-02-06 16:57:10,607 P75877 INFO Save best model: monitor(max): 0.745307
2022-02-06 16:57:10,674 P75877 INFO --- 6910/6910 batches finished ---
2022-02-06 16:57:10,716 P75877 INFO Train loss: 0.427836
2022-02-06 16:57:10,716 P75877 INFO ************ Epoch=2 end ************
2022-02-06 17:04:32,356 P75877 INFO [Metrics] AUC: 0.744534 - logloss: 0.397424
2022-02-06 17:04:32,360 P75877 INFO Monitor(max) STOP: 0.744534 !
2022-02-06 17:04:32,360 P75877 INFO Reduce learning rate on plateau: 0.000100
2022-02-06 17:04:32,360 P75877 INFO --- 6910/6910 batches finished ---
2022-02-06 17:04:32,405 P75877 INFO Train loss: 0.428318
2022-02-06 17:04:32,405 P75877 INFO ************ Epoch=3 end ************
2022-02-06 17:11:50,192 P75877 INFO [Metrics] AUC: 0.745535 - logloss: 0.396204
2022-02-06 17:11:50,195 P75877 INFO Save best model: monitor(max): 0.745535
2022-02-06 17:11:50,267 P75877 INFO --- 6910/6910 batches finished ---
2022-02-06 17:11:50,325 P75877 INFO Train loss: 0.402562
2022-02-06 17:11:50,325 P75877 INFO ************ Epoch=4 end ************
2022-02-06 17:19:08,434 P75877 INFO [Metrics] AUC: 0.745824 - logloss: 0.395891
2022-02-06 17:19:08,437 P75877 INFO Save best model: monitor(max): 0.745824
2022-02-06 17:19:08,509 P75877 INFO --- 6910/6910 batches finished ---
2022-02-06 17:19:08,570 P75877 INFO Train loss: 0.402307
2022-02-06 17:19:08,570 P75877 INFO ************ Epoch=5 end ************
2022-02-06 17:26:26,427 P75877 INFO [Metrics] AUC: 0.746129 - logloss: 0.395687
2022-02-06 17:26:26,430 P75877 INFO Save best model: monitor(max): 0.746129
2022-02-06 17:26:26,495 P75877 INFO --- 6910/6910 batches finished ---
2022-02-06 17:26:26,554 P75877 INFO Train loss: 0.402231
2022-02-06 17:26:26,555 P75877 INFO ************ Epoch=6 end ************
2022-02-06 17:33:43,050 P75877 INFO [Metrics] AUC: 0.745665 - logloss: 0.396153
2022-02-06 17:33:43,053 P75877 INFO Monitor(max) STOP: 0.745665 !
2022-02-06 17:33:43,053 P75877 INFO Reduce learning rate on plateau: 0.000010
2022-02-06 17:33:43,053 P75877 INFO --- 6910/6910 batches finished ---
2022-02-06 17:33:43,104 P75877 INFO Train loss: 0.402205
2022-02-06 17:33:43,105 P75877 INFO ************ Epoch=7 end ************
2022-02-06 17:41:00,959 P75877 INFO [Metrics] AUC: 0.740420 - logloss: 0.399509
2022-02-06 17:41:00,963 P75877 INFO Monitor(max) STOP: 0.740420 !
2022-02-06 17:41:00,963 P75877 INFO Reduce learning rate on plateau: 0.000001
2022-02-06 17:41:00,963 P75877 INFO Early stopping at epoch=8
2022-02-06 17:41:00,963 P75877 INFO --- 6910/6910 batches finished ---
2022-02-06 17:41:01,026 P75877 INFO Train loss: 0.389451
2022-02-06 17:41:01,027 P75877 INFO Training finished.
2022-02-06 17:41:01,027 P75877 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/WideDeep_avazu_x1/avazu_x1_3fb65689/WideDeep_avazu_x1_002_d8c146ac.model
2022-02-06 17:41:04,634 P75877 INFO ****** Validation evaluation ******
2022-02-06 17:41:15,454 P75877 INFO [Metrics] AUC: 0.746129 - logloss: 0.395687
2022-02-06 17:41:15,519 P75877 INFO ******** Test evaluation ********
2022-02-06 17:41:15,519 P75877 INFO Loading data...
2022-02-06 17:41:15,520 P75877 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-02-06 17:41:16,373 P75877 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-02-06 17:41:16,373 P75877 INFO Loading test data done.
2022-02-06 17:41:39,819 P75877 INFO [Metrics] AUC: 0.764917 - logloss: 0.366495

```
