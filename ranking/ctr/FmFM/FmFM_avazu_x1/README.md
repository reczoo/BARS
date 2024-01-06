## FmFM_avazu_x1

A hands-on guide to run the FmFM model on the Avazu_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

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
Dataset ID: [Avazu_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [FmFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FmFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FmFM_avazu_x1_tuner_config_01](./FmFM_avazu_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FmFM_avazu_x1
    nohup python run_expid.py --config ./FmFM_avazu_x1_tuner_config_01 --expid FmFM_avazu_x1_002_a09a4da9 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.759545 | 0.368913  |


### Logs
```python
2022-01-19 19:29:28,862 P42722 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "embedding_dim": "10",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "field_interaction_type": "matrixed",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FmFM",
    "model_id": "FmFM_avazu_x1_002_a09a4da9",
    "model_root": "./Avazu/FmFM_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-05",
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
2022-01-19 19:29:28,862 P42722 INFO Set up feature encoder...
2022-01-19 19:29:28,862 P42722 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-01-19 19:29:28,863 P42722 INFO Loading data...
2022-01-19 19:29:28,865 P42722 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-01-19 19:29:31,760 P42722 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-01-19 19:29:32,187 P42722 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-01-19 19:29:32,187 P42722 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-01-19 19:29:32,187 P42722 INFO Loading train data done.
2022-01-19 19:29:36,911 P42722 INFO Total number of parameters: 14307690.
2022-01-19 19:29:36,911 P42722 INFO Start training: 6910 batches/epoch
2022-01-19 19:29:36,911 P42722 INFO ************ Epoch=1 start ************
2022-01-19 20:40:31,042 P42722 INFO [Metrics] AUC: 0.742740 - logloss: 0.400485
2022-01-19 20:40:31,046 P42722 INFO Save best model: monitor(max): 0.742740
2022-01-19 20:40:31,118 P42722 INFO --- 6910/6910 batches finished ---
2022-01-19 20:40:31,188 P42722 INFO Train loss: 0.407180
2022-01-19 20:40:31,188 P42722 INFO ************ Epoch=1 end ************
2022-01-19 21:52:58,432 P42722 INFO [Metrics] AUC: 0.743770 - logloss: 0.397679
2022-01-19 21:52:58,441 P42722 INFO Save best model: monitor(max): 0.743770
2022-01-19 21:52:58,546 P42722 INFO --- 6910/6910 batches finished ---
2022-01-19 21:52:58,596 P42722 INFO Train loss: 0.402016
2022-01-19 21:52:58,596 P42722 INFO ************ Epoch=2 end ************
2022-01-19 22:57:12,211 P42722 INFO [Metrics] AUC: 0.743921 - logloss: 0.398118
2022-01-19 22:57:12,214 P42722 INFO Save best model: monitor(max): 0.743921
2022-01-19 22:57:12,309 P42722 INFO --- 6910/6910 batches finished ---
2022-01-19 22:57:12,359 P42722 INFO Train loss: 0.398866
2022-01-19 22:57:12,359 P42722 INFO ************ Epoch=3 end ************
2022-01-19 23:51:30,314 P42722 INFO [Metrics] AUC: 0.743831 - logloss: 0.397041
2022-01-19 23:51:30,318 P42722 INFO Monitor(max) STOP: 0.743831 !
2022-01-19 23:51:30,318 P42722 INFO Reduce learning rate on plateau: 0.000100
2022-01-19 23:51:30,318 P42722 INFO --- 6910/6910 batches finished ---
2022-01-19 23:51:30,371 P42722 INFO Train loss: 0.397408
2022-01-19 23:51:30,372 P42722 INFO ************ Epoch=4 end ************
2022-01-20 00:03:53,148 P42722 INFO [Metrics] AUC: 0.738686 - logloss: 0.401159
2022-01-20 00:03:53,153 P42722 INFO Monitor(max) STOP: 0.738686 !
2022-01-20 00:03:53,153 P42722 INFO Reduce learning rate on plateau: 0.000010
2022-01-20 00:03:53,153 P42722 INFO Early stopping at epoch=5
2022-01-20 00:03:53,153 P42722 INFO --- 6910/6910 batches finished ---
2022-01-20 00:03:53,202 P42722 INFO Train loss: 0.386411
2022-01-20 00:03:53,202 P42722 INFO Training finished.
2022-01-20 00:03:53,202 P42722 INFO Load best model: /home/XXX/benchmarks/Avazu/FmFM_avazu_x1/avazu_x1_3fb65689/FmFM_avazu_x1_002_a09a4da9.model
2022-01-20 00:03:58,405 P42722 INFO ****** Validation evaluation ******
2022-01-20 00:04:19,456 P42722 INFO [Metrics] AUC: 0.743921 - logloss: 0.398118
2022-01-20 00:04:19,515 P42722 INFO ******** Test evaluation ********
2022-01-20 00:04:19,515 P42722 INFO Loading data...
2022-01-20 00:04:19,515 P42722 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-01-20 00:04:21,286 P42722 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-01-20 00:04:21,286 P42722 INFO Loading test data done.
2022-01-20 00:05:03,752 P42722 INFO [Metrics] AUC: 0.759545 - logloss: 0.368913

```
