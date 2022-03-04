## CrossNetv2_avazu_x1

A hands-on guide to run the DCNv2 model on the Avazu_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DCNv2](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DCNv2.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CrossNetv2_avazu_x1_tuner_config_01](./CrossNetv2_avazu_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CrossNetv2_avazu_x1
    nohup python run_expid.py --config ./CrossNetv2_avazu_x1_tuner_config_01 --expid DCNv2_avazu_x1_011_e4cbb525 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.762511 | 0.368057  |


### Logs
```python
2022-01-22 15:17:59,116 P810 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "dnn_activations": "relu",
    "embedding_dim": "10",
    "embedding_regularizer": "0.0001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "2",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "low_rank": "32",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCNv2",
    "model_id": "DCNv2_avazu_x1_011_e4cbb525",
    "model_root": "./Avazu/DCNv2_avazu_x1/",
    "model_structure": "crossnet_only",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_cross_layers": "4",
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
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_hdf5": "True",
    "use_low_rank_mixture": "False",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-01-22 15:17:59,116 P810 INFO Set up feature encoder...
2022-01-22 15:17:59,116 P810 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-01-22 15:17:59,117 P810 INFO Loading data...
2022-01-22 15:17:59,118 P810 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-01-22 15:18:01,585 P810 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-01-22 15:18:01,942 P810 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-01-22 15:18:01,942 P810 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-01-22 15:18:01,942 P810 INFO Loading train data done.
2022-01-22 15:18:07,877 P810 INFO Total number of parameters: 13180691.
2022-01-22 15:18:07,878 P810 INFO Start training: 6910 batches/epoch
2022-01-22 15:18:07,878 P810 INFO ************ Epoch=1 start ************
2022-01-22 15:26:54,209 P810 INFO [Metrics] AUC: 0.744306 - logloss: 0.398425
2022-01-22 15:26:54,212 P810 INFO Save best model: monitor(max): 0.744306
2022-01-22 15:26:54,432 P810 INFO --- 6910/6910 batches finished ---
2022-01-22 15:26:54,479 P810 INFO Train loss: 0.409269
2022-01-22 15:26:54,479 P810 INFO ************ Epoch=1 end ************
2022-01-22 15:35:45,034 P810 INFO [Metrics] AUC: 0.744847 - logloss: 0.396355
2022-01-22 15:35:45,039 P810 INFO Save best model: monitor(max): 0.744847
2022-01-22 15:35:45,111 P810 INFO --- 6910/6910 batches finished ---
2022-01-22 15:35:45,159 P810 INFO Train loss: 0.405218
2022-01-22 15:35:45,159 P810 INFO ************ Epoch=2 end ************
2022-01-22 15:44:36,868 P810 INFO [Metrics] AUC: 0.746125 - logloss: 0.397151
2022-01-22 15:44:36,871 P810 INFO Save best model: monitor(max): 0.746125
2022-01-22 15:44:36,948 P810 INFO --- 6910/6910 batches finished ---
2022-01-22 15:44:36,991 P810 INFO Train loss: 0.404487
2022-01-22 15:44:36,991 P810 INFO ************ Epoch=3 end ************
2022-01-22 15:53:28,383 P810 INFO [Metrics] AUC: 0.747329 - logloss: 0.396326
2022-01-22 15:53:28,386 P810 INFO Save best model: monitor(max): 0.747329
2022-01-22 15:53:28,458 P810 INFO --- 6910/6910 batches finished ---
2022-01-22 15:53:28,497 P810 INFO Train loss: 0.404287
2022-01-22 15:53:28,498 P810 INFO ************ Epoch=4 end ************
2022-01-22 16:02:16,672 P810 INFO [Metrics] AUC: 0.744175 - logloss: 0.396730
2022-01-22 16:02:16,677 P810 INFO Monitor(max) STOP: 0.744175 !
2022-01-22 16:02:16,677 P810 INFO Reduce learning rate on plateau: 0.000100
2022-01-22 16:02:16,677 P810 INFO --- 6910/6910 batches finished ---
2022-01-22 16:02:16,718 P810 INFO Train loss: 0.404125
2022-01-22 16:02:16,719 P810 INFO ************ Epoch=5 end ************
2022-01-22 16:06:26,198 P810 INFO [Metrics] AUC: 0.744130 - logloss: 0.397730
2022-01-22 16:06:26,203 P810 INFO Monitor(max) STOP: 0.744130 !
2022-01-22 16:06:26,203 P810 INFO Reduce learning rate on plateau: 0.000010
2022-01-22 16:06:26,203 P810 INFO Early stopping at epoch=6
2022-01-22 16:06:26,203 P810 INFO --- 6910/6910 batches finished ---
2022-01-22 16:06:26,248 P810 INFO Train loss: 0.390893
2022-01-22 16:06:26,249 P810 INFO Training finished.
2022-01-22 16:06:26,249 P810 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/DCNv2_avazu_x1/avazu_x1_3fb65689/DCNv2_avazu_x1_011_e4cbb525.model
2022-01-22 16:06:31,819 P810 INFO ****** Validation evaluation ******
2022-01-22 16:06:42,987 P810 INFO [Metrics] AUC: 0.747329 - logloss: 0.396326
2022-01-22 16:06:43,078 P810 INFO ******** Test evaluation ********
2022-01-22 16:06:43,078 P810 INFO Loading data...
2022-01-22 16:06:43,078 P810 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-01-22 16:06:43,979 P810 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-01-22 16:06:43,980 P810 INFO Loading test data done.
2022-01-22 16:07:10,372 P810 INFO [Metrics] AUC: 0.762511 - logloss: 0.368057

```
