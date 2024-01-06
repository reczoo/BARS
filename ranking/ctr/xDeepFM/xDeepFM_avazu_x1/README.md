## xDeepFM_avazu_x1

A hands-on guide to run the xDeepFM model on the Avazu_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [xDeepFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/xDeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [xDeepFM_avazu_x1_tuner_config_07](./xDeepFM_avazu_x1_tuner_config_07). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd xDeepFM_avazu_x1
    nohup python run_expid.py --config ./xDeepFM_avazu_x1_tuner_config_07 --expid xDeepFM_avazu_x1_002_937cff08 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.764730 | 0.367068  |
| 2 | 0.763954 | 0.367703  |
| 3 | 0.764540 | 0.366576  |
| 4 | 0.764217 | 0.367269  |
| 5 | 0.763584 | 0.367166  |
| Avg | 0.764205 | 0.367156 |
| Std | &#177;0.00040906 | &#177;0.00036228 |


### Logs
```python
2022-01-21 07:29:06,902 P799 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "cin_layer_units": "[64]",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "xDeepFM",
    "model_id": "xDeepFM_avazu_x1_002_937cff08",
    "model_root": "./Avazu/xDeepFM_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
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
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-21 07:29:06,903 P799 INFO Set up feature encoder...
2022-01-21 07:29:06,903 P799 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-01-21 07:29:06,903 P799 INFO Loading data...
2022-01-21 07:29:06,905 P799 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-01-21 07:29:09,338 P799 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-01-21 07:29:09,657 P799 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-01-21 07:29:09,657 P799 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-01-21 07:29:09,657 P799 INFO Loading train data done.
2022-01-21 07:29:16,041 P799 INFO Total number of parameters: 14727695.
2022-01-21 07:29:16,042 P799 INFO Start training: 6910 batches/epoch
2022-01-21 07:29:16,042 P799 INFO ************ Epoch=1 start ************
2022-01-21 07:48:20,785 P799 INFO [Metrics] AUC: 0.741322 - logloss: 0.400246
2022-01-21 07:48:20,788 P799 INFO Save best model: monitor(max): 0.741322
2022-01-21 07:48:21,107 P799 INFO --- 6910/6910 batches finished ---
2022-01-21 07:48:21,150 P799 INFO Train loss: 0.432228
2022-01-21 07:48:21,151 P799 INFO ************ Epoch=1 end ************
2022-01-21 08:07:26,469 P799 INFO [Metrics] AUC: 0.745349 - logloss: 0.397551
2022-01-21 08:07:26,472 P799 INFO Save best model: monitor(max): 0.745349
2022-01-21 08:07:26,543 P799 INFO --- 6910/6910 batches finished ---
2022-01-21 08:07:26,592 P799 INFO Train loss: 0.429237
2022-01-21 08:07:26,592 P799 INFO ************ Epoch=2 end ************
2022-01-21 08:26:30,331 P799 INFO [Metrics] AUC: 0.741173 - logloss: 0.398976
2022-01-21 08:26:30,333 P799 INFO Monitor(max) STOP: 0.741173 !
2022-01-21 08:26:30,333 P799 INFO Reduce learning rate on plateau: 0.000100
2022-01-21 08:26:30,334 P799 INFO --- 6910/6910 batches finished ---
2022-01-21 08:26:30,393 P799 INFO Train loss: 0.428893
2022-01-21 08:26:30,394 P799 INFO ************ Epoch=3 end ************
2022-01-21 08:45:34,941 P799 INFO [Metrics] AUC: 0.746297 - logloss: 0.396029
2022-01-21 08:45:34,945 P799 INFO Save best model: monitor(max): 0.746297
2022-01-21 08:45:35,024 P799 INFO --- 6910/6910 batches finished ---
2022-01-21 08:45:35,069 P799 INFO Train loss: 0.404743
2022-01-21 08:45:35,069 P799 INFO ************ Epoch=4 end ************
2022-01-21 09:04:38,528 P799 INFO [Metrics] AUC: 0.743723 - logloss: 0.397231
2022-01-21 09:04:38,530 P799 INFO Monitor(max) STOP: 0.743723 !
2022-01-21 09:04:38,530 P799 INFO Reduce learning rate on plateau: 0.000010
2022-01-21 09:04:38,530 P799 INFO --- 6910/6910 batches finished ---
2022-01-21 09:04:38,573 P799 INFO Train loss: 0.405646
2022-01-21 09:04:38,573 P799 INFO ************ Epoch=5 end ************
2022-01-21 09:23:41,854 P799 INFO [Metrics] AUC: 0.743161 - logloss: 0.398305
2022-01-21 09:23:41,856 P799 INFO Monitor(max) STOP: 0.743161 !
2022-01-21 09:23:41,856 P799 INFO Reduce learning rate on plateau: 0.000001
2022-01-21 09:23:41,856 P799 INFO Early stopping at epoch=6
2022-01-21 09:23:41,856 P799 INFO --- 6910/6910 batches finished ---
2022-01-21 09:23:41,900 P799 INFO Train loss: 0.393103
2022-01-21 09:23:41,900 P799 INFO Training finished.
2022-01-21 09:23:41,901 P799 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/xDeepFM_avazu_x1/avazu_x1_3fb65689/xDeepFM_avazu_x1_002_937cff08.model
2022-01-21 09:23:46,425 P799 INFO ****** Validation evaluation ******
2022-01-21 09:23:58,290 P799 INFO [Metrics] AUC: 0.746297 - logloss: 0.396029
2022-01-21 09:23:58,361 P799 INFO ******** Test evaluation ********
2022-01-21 09:23:58,362 P799 INFO Loading data...
2022-01-21 09:23:58,362 P799 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-01-21 09:23:59,045 P799 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-01-21 09:23:59,045 P799 INFO Loading test data done.
2022-01-21 09:24:24,148 P799 INFO [Metrics] AUC: 0.764730 - logloss: 0.367068

```
