## LR_avazu_x1

A hands-on guide to run the LR model on the Avazu_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [LR](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/LR.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [LR_avazu_x1_tuner_config_01](./LR_avazu_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd LR_avazu_x1
    nohup python run_expid.py --config ./LR_avazu_x1_tuner_config_01 --expid LR_avazu_x1_007_c7701820 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.751625 | 0.373527  |


### Logs
```python
2022-01-26 09:21:43,959 P57363 INFO {
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "6",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "LR",
    "model_id": "LR_avazu_x1_007_c7701820",
    "model_root": "./Avazu/LR_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "5e-07",
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
2022-01-26 09:21:43,961 P57363 INFO Set up feature encoder...
2022-01-26 09:21:43,961 P57363 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-01-26 09:21:43,961 P57363 INFO Loading data...
2022-01-26 09:21:43,963 P57363 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-01-26 09:21:46,083 P57363 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-01-26 09:21:46,399 P57363 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-01-26 09:21:46,399 P57363 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-01-26 09:21:46,399 P57363 INFO Loading train data done.
2022-01-26 09:21:51,070 P57363 INFO Total number of parameters: 1298600.
2022-01-26 09:21:51,070 P57363 INFO Start training: 6910 batches/epoch
2022-01-26 09:21:51,070 P57363 INFO ************ Epoch=1 start ************
2022-01-26 09:25:01,094 P57363 INFO [Metrics] AUC: 0.734799 - logloss: 0.403314
2022-01-26 09:25:01,096 P57363 INFO Save best model: monitor(max): 0.734799
2022-01-26 09:25:01,255 P57363 INFO --- 6910/6910 batches finished ---
2022-01-26 09:25:01,290 P57363 INFO Train loss: 0.407915
2022-01-26 09:25:01,290 P57363 INFO ************ Epoch=1 end ************
2022-01-26 09:28:08,942 P57363 INFO [Metrics] AUC: 0.735522 - logloss: 0.402475
2022-01-26 09:28:08,944 P57363 INFO Save best model: monitor(max): 0.735522
2022-01-26 09:28:08,954 P57363 INFO --- 6910/6910 batches finished ---
2022-01-26 09:28:09,002 P57363 INFO Train loss: 0.400620
2022-01-26 09:28:09,002 P57363 INFO ************ Epoch=2 end ************
2022-01-26 09:31:15,936 P57363 INFO [Metrics] AUC: 0.734518 - logloss: 0.403539
2022-01-26 09:31:15,938 P57363 INFO Monitor(max) STOP: 0.734518 !
2022-01-26 09:31:15,938 P57363 INFO Reduce learning rate on plateau: 0.000100
2022-01-26 09:31:15,938 P57363 INFO --- 6910/6910 batches finished ---
2022-01-26 09:31:15,974 P57363 INFO Train loss: 0.400006
2022-01-26 09:31:15,974 P57363 INFO ************ Epoch=3 end ************
2022-01-26 09:34:22,009 P57363 INFO [Metrics] AUC: 0.735172 - logloss: 0.402830
2022-01-26 09:34:22,010 P57363 INFO Monitor(max) STOP: 0.735172 !
2022-01-26 09:34:22,010 P57363 INFO Reduce learning rate on plateau: 0.000010
2022-01-26 09:34:22,010 P57363 INFO Early stopping at epoch=4
2022-01-26 09:34:22,010 P57363 INFO --- 6910/6910 batches finished ---
2022-01-26 09:34:22,058 P57363 INFO Train loss: 0.398741
2022-01-26 09:34:22,059 P57363 INFO Training finished.
2022-01-26 09:34:22,059 P57363 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/LR_avazu_x1/avazu_x1_3fb65689/LR_avazu_x1_007_c7701820.model
2022-01-26 09:34:25,206 P57363 INFO ****** Validation evaluation ******
2022-01-26 09:34:35,572 P57363 INFO [Metrics] AUC: 0.735522 - logloss: 0.402475
2022-01-26 09:34:35,641 P57363 INFO ******** Test evaluation ********
2022-01-26 09:34:35,642 P57363 INFO Loading data...
2022-01-26 09:34:35,642 P57363 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-01-26 09:34:36,565 P57363 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-01-26 09:34:36,565 P57363 INFO Loading test data done.
2022-01-26 09:35:03,283 P57363 INFO [Metrics] AUC: 0.751625 - logloss: 0.373527

```
