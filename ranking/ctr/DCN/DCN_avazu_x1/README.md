## DCN_avazu_x1

A hands-on guide to run the DCN model on the Avazu_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DCN](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DCN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DCN_avazu_x1_tuner_config_03](./DCN_avazu_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DCN_avazu_x1
    nohup python run_expid.py --config ./DCN_avazu_x1_tuner_config_03 --expid DCN_avazu_x1_004_e58d35c1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.765181 | 0.366485  |
| 2 | 0.761728 | 0.368109  |
| 3 | 0.764153 | 0.366988  |
| 4 | 0.764464 | 0.366932  |
| 5 | 0.763515 | 0.367715  |
| Avg | 0.763808 | 0.367246 |
| Std | &#177;0.00117019 | &#177;0.00058464 |


### Logs
```python
2022-02-09 00:54:48,138 P734 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "crossing_layers": "3",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "3",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCN",
    "model_id": "DCN_avazu_x1_004_e58d35c1",
    "model_root": "./Avazu/DCN_avazu_x1/",
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
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-02-09 00:54:48,139 P734 INFO Set up feature encoder...
2022-02-09 00:54:48,139 P734 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-02-09 00:54:48,139 P734 INFO Loading data...
2022-02-09 00:54:48,140 P734 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-02-09 00:54:50,502 P734 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-02-09 00:54:50,824 P734 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-02-09 00:54:50,825 P734 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-02-09 00:54:50,825 P734 INFO Loading train data done.
2022-02-09 00:54:54,916 P734 INFO Total number of parameters: 13397131.
2022-02-09 00:54:54,916 P734 INFO Start training: 6910 batches/epoch
2022-02-09 00:54:54,917 P734 INFO ************ Epoch=1 start ************
2022-02-09 01:04:07,081 P734 INFO [Metrics] AUC: 0.736263 - logloss: 0.401565
2022-02-09 01:04:07,084 P734 INFO Save best model: monitor(max): 0.736263
2022-02-09 01:04:07,311 P734 INFO --- 6910/6910 batches finished ---
2022-02-09 01:04:07,345 P734 INFO Train loss: 0.437337
2022-02-09 01:04:07,345 P734 INFO ************ Epoch=1 end ************
2022-02-09 01:13:19,081 P734 INFO [Metrics] AUC: 0.730804 - logloss: 0.404337
2022-02-09 01:13:19,084 P734 INFO Monitor(max) STOP: 0.730804 !
2022-02-09 01:13:19,084 P734 INFO Reduce learning rate on plateau: 0.000100
2022-02-09 01:13:19,084 P734 INFO --- 6910/6910 batches finished ---
2022-02-09 01:13:19,120 P734 INFO Train loss: 0.438976
2022-02-09 01:13:19,120 P734 INFO ************ Epoch=2 end ************
2022-02-09 01:22:32,539 P734 INFO [Metrics] AUC: 0.743859 - logloss: 0.398189
2022-02-09 01:22:32,541 P734 INFO Save best model: monitor(max): 0.743859
2022-02-09 01:22:32,613 P734 INFO --- 6910/6910 batches finished ---
2022-02-09 01:22:32,647 P734 INFO Train loss: 0.410227
2022-02-09 01:22:32,647 P734 INFO ************ Epoch=3 end ************
2022-02-09 01:31:45,614 P734 INFO [Metrics] AUC: 0.746237 - logloss: 0.397325
2022-02-09 01:31:45,617 P734 INFO Save best model: monitor(max): 0.746237
2022-02-09 01:31:45,689 P734 INFO --- 6910/6910 batches finished ---
2022-02-09 01:31:45,727 P734 INFO Train loss: 0.412729
2022-02-09 01:31:45,727 P734 INFO ************ Epoch=4 end ************
2022-02-09 01:40:59,799 P734 INFO [Metrics] AUC: 0.746186 - logloss: 0.396691
2022-02-09 01:40:59,802 P734 INFO Monitor(max) STOP: 0.746186 !
2022-02-09 01:40:59,802 P734 INFO Reduce learning rate on plateau: 0.000010
2022-02-09 01:40:59,803 P734 INFO --- 6910/6910 batches finished ---
2022-02-09 01:40:59,842 P734 INFO Train loss: 0.413467
2022-02-09 01:40:59,842 P734 INFO ************ Epoch=5 end ************
2022-02-09 01:50:09,735 P734 INFO [Metrics] AUC: 0.747655 - logloss: 0.395564
2022-02-09 01:50:09,738 P734 INFO Save best model: monitor(max): 0.747655
2022-02-09 01:50:09,804 P734 INFO --- 6910/6910 batches finished ---
2022-02-09 01:50:09,840 P734 INFO Train loss: 0.398316
2022-02-09 01:50:09,840 P734 INFO ************ Epoch=6 end ************
2022-02-09 01:59:15,502 P734 INFO [Metrics] AUC: 0.744314 - logloss: 0.397735
2022-02-09 01:59:15,505 P734 INFO Monitor(max) STOP: 0.744314 !
2022-02-09 01:59:15,505 P734 INFO Reduce learning rate on plateau: 0.000001
2022-02-09 01:59:15,505 P734 INFO --- 6910/6910 batches finished ---
2022-02-09 01:59:15,537 P734 INFO Train loss: 0.395764
2022-02-09 01:59:15,537 P734 INFO ************ Epoch=7 end ************
2022-02-09 02:08:18,833 P734 INFO [Metrics] AUC: 0.741345 - logloss: 0.399801
2022-02-09 02:08:18,836 P734 INFO Monitor(max) STOP: 0.741345 !
2022-02-09 02:08:18,836 P734 INFO Reduce learning rate on plateau: 0.000001
2022-02-09 02:08:18,836 P734 INFO Early stopping at epoch=8
2022-02-09 02:08:18,836 P734 INFO --- 6910/6910 batches finished ---
2022-02-09 02:08:18,869 P734 INFO Train loss: 0.388288
2022-02-09 02:08:18,869 P734 INFO Training finished.
2022-02-09 02:08:18,869 P734 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/DCN_avazu_x1/avazu_x1_3fb65689/DCN_avazu_x1_004_e58d35c1.model
2022-02-09 02:08:23,407 P734 INFO ****** Validation evaluation ******
2022-02-09 02:08:34,994 P734 INFO [Metrics] AUC: 0.747655 - logloss: 0.395564
2022-02-09 02:08:35,093 P734 INFO ******** Test evaluation ********
2022-02-09 02:08:35,094 P734 INFO Loading data...
2022-02-09 02:08:35,094 P734 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-02-09 02:08:35,789 P734 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-02-09 02:08:35,789 P734 INFO Loading test data done.
2022-02-09 02:09:01,181 P734 INFO [Metrics] AUC: 0.765181 - logloss: 0.366485

```
