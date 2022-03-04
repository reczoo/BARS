## DeepFM_avazu_x1

A hands-on guide to run the DeepFM model on the Avazu_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DeepFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DeepFM_avazu_x1_tuner_config_02](./DeepFM_avazu_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DeepFM_avazu_x1
    nohup python run_expid.py --config ./DeepFM_avazu_x1_tuner_config_02 --expid DeepFM_avazu_x1_004_514a2b87 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.764834 | 0.366694  |
| 2 | 0.764245 | 0.367372  |
| 3 | 0.764242 | 0.367065  |
| 4 | 0.764264 | 0.367133  |
| 5 | 0.765346 | 0.366591  |
| | | | 
| Avg | 0.764586 | 0.366971 |
| Std | &#177;0.00044213 | &#177;0.00028879 |


### Logs
```python
2022-01-19 00:02:14,415 P4432 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.1",
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
    "model": "DeepFM",
    "model_id": "DeepFM_avazu_x1_004_514a2b87",
    "model_root": "./Avazu/DeepFM_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
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
2022-01-19 00:02:14,416 P4432 INFO Set up feature encoder...
2022-01-19 00:02:14,416 P4432 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-01-19 00:02:14,416 P4432 INFO Loading data...
2022-01-19 00:02:14,419 P4432 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-01-19 00:02:18,314 P4432 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-01-19 00:02:18,665 P4432 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-01-19 00:02:18,665 P4432 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-01-19 00:02:18,665 P4432 INFO Loading train data done.
2022-01-19 00:02:21,853 P4432 INFO Total number of parameters: 14696590.
2022-01-19 00:02:21,853 P4432 INFO Start training: 6910 batches/epoch
2022-01-19 00:02:21,853 P4432 INFO ************ Epoch=1 start ************
2022-01-19 00:21:27,614 P4432 INFO [Metrics] AUC: 0.737313 - logloss: 0.401376
2022-01-19 00:21:27,618 P4432 INFO Save best model: monitor(max): 0.737313
2022-01-19 00:21:27,690 P4432 INFO --- 6910/6910 batches finished ---
2022-01-19 00:21:27,721 P4432 INFO Train loss: 0.446882
2022-01-19 00:21:27,721 P4432 INFO ************ Epoch=1 end ************
2022-01-19 00:40:32,580 P4432 INFO [Metrics] AUC: 0.735205 - logloss: 0.401558
2022-01-19 00:40:32,583 P4432 INFO Monitor(max) STOP: 0.735205 !
2022-01-19 00:40:32,583 P4432 INFO Reduce learning rate on plateau: 0.000100
2022-01-19 00:40:32,583 P4432 INFO --- 6910/6910 batches finished ---
2022-01-19 00:40:32,611 P4432 INFO Train loss: 0.443180
2022-01-19 00:40:32,611 P4432 INFO ************ Epoch=2 end ************
2022-01-19 00:59:35,181 P4432 INFO [Metrics] AUC: 0.743520 - logloss: 0.397635
2022-01-19 00:59:35,185 P4432 INFO Save best model: monitor(max): 0.743520
2022-01-19 00:59:35,306 P4432 INFO --- 6910/6910 batches finished ---
2022-01-19 00:59:35,342 P4432 INFO Train loss: 0.412606
2022-01-19 00:59:35,343 P4432 INFO ************ Epoch=3 end ************
2022-01-19 01:18:35,980 P4432 INFO [Metrics] AUC: 0.743583 - logloss: 0.397054
2022-01-19 01:18:35,983 P4432 INFO Save best model: monitor(max): 0.743583
2022-01-19 01:18:36,072 P4432 INFO --- 6910/6910 batches finished ---
2022-01-19 01:18:36,100 P4432 INFO Train loss: 0.414403
2022-01-19 01:18:36,100 P4432 INFO ************ Epoch=4 end ************
2022-01-19 01:37:38,199 P4432 INFO [Metrics] AUC: 0.745535 - logloss: 0.396151
2022-01-19 01:37:38,201 P4432 INFO Save best model: monitor(max): 0.745535
2022-01-19 01:37:38,295 P4432 INFO --- 6910/6910 batches finished ---
2022-01-19 01:37:38,322 P4432 INFO Train loss: 0.415673
2022-01-19 01:37:38,322 P4432 INFO ************ Epoch=5 end ************
2022-01-19 01:56:39,014 P4432 INFO [Metrics] AUC: 0.743507 - logloss: 0.397280
2022-01-19 01:56:39,016 P4432 INFO Monitor(max) STOP: 0.743507 !
2022-01-19 01:56:39,016 P4432 INFO Reduce learning rate on plateau: 0.000010
2022-01-19 01:56:39,017 P4432 INFO --- 6910/6910 batches finished ---
2022-01-19 01:56:39,046 P4432 INFO Train loss: 0.416631
2022-01-19 01:56:39,046 P4432 INFO ************ Epoch=6 end ************
2022-01-19 02:15:39,051 P4432 INFO [Metrics] AUC: 0.746620 - logloss: 0.395728
2022-01-19 02:15:39,055 P4432 INFO Save best model: monitor(max): 0.746620
2022-01-19 02:15:39,163 P4432 INFO --- 6910/6910 batches finished ---
2022-01-19 02:15:39,192 P4432 INFO Train loss: 0.399585
2022-01-19 02:15:39,193 P4432 INFO ************ Epoch=7 end ************
2022-01-19 02:34:38,607 P4432 INFO [Metrics] AUC: 0.746226 - logloss: 0.395861
2022-01-19 02:34:38,609 P4432 INFO Monitor(max) STOP: 0.746226 !
2022-01-19 02:34:38,610 P4432 INFO Reduce learning rate on plateau: 0.000001
2022-01-19 02:34:38,610 P4432 INFO --- 6910/6910 batches finished ---
2022-01-19 02:34:38,645 P4432 INFO Train loss: 0.398252
2022-01-19 02:34:38,645 P4432 INFO ************ Epoch=8 end ************
2022-01-19 02:53:36,044 P4432 INFO [Metrics] AUC: 0.743626 - logloss: 0.397129
2022-01-19 02:53:36,048 P4432 INFO Monitor(max) STOP: 0.743626 !
2022-01-19 02:53:36,048 P4432 INFO Reduce learning rate on plateau: 0.000001
2022-01-19 02:53:36,048 P4432 INFO Early stopping at epoch=9
2022-01-19 02:53:36,048 P4432 INFO --- 6910/6910 batches finished ---
2022-01-19 02:53:36,082 P4432 INFO Train loss: 0.391931
2022-01-19 02:53:36,082 P4432 INFO Training finished.
2022-01-19 02:53:36,082 P4432 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Avazu/DeepFM_avazu_x1/avazu_x1_3fb65689/DeepFM_avazu_x1_004_514a2b87.model
2022-01-19 02:53:39,365 P4432 INFO ****** Validation evaluation ******
2022-01-19 02:53:54,595 P4432 INFO [Metrics] AUC: 0.746620 - logloss: 0.395728
2022-01-19 02:53:54,669 P4432 INFO ******** Test evaluation ********
2022-01-19 02:53:54,669 P4432 INFO Loading data...
2022-01-19 02:53:54,670 P4432 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-01-19 02:53:55,380 P4432 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-01-19 02:53:55,381 P4432 INFO Loading test data done.
2022-01-19 02:54:20,558 P4432 INFO [Metrics] AUC: 0.764834 - logloss: 0.366694

```
