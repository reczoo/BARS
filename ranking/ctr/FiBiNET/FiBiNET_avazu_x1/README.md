## FiBiNET_avazu_x1

A hands-on guide to run the FiBiNET model on the Avazu_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [FiBiNET](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FiBiNET.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FiBiNET_avazu_x1_tuner_config_06](./FiBiNET_avazu_x1_tuner_config_06). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FiBiNET_avazu_x1
    nohup python run_expid.py --config ./FiBiNET_avazu_x1_tuner_config_06 --expid FiBiNET_avazu_x1_002_f09a284c --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.764531 | 0.367292  |


### Logs
```python
2022-01-26 08:11:57,295 P35997 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "bilinear_type": "field_all",
    "data_block_size": "-1",
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
    "model": "FiBiNET",
    "model_id": "FiBiNET_avazu_x1_002_f09a284c",
    "model_root": "./Avazu/FiBiNET_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "3",
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
2022-01-26 08:11:57,296 P35997 INFO Set up feature encoder...
2022-01-26 08:11:57,297 P35997 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-01-26 08:11:57,297 P35997 INFO Loading data...
2022-01-26 08:11:57,298 P35997 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-01-26 08:11:59,489 P35997 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-01-26 08:11:59,823 P35997 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-01-26 08:11:59,824 P35997 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-01-26 08:11:59,824 P35997 INFO Loading train data done.
2022-01-26 08:12:05,090 P35997 INFO Total number of parameters: 16456998.
2022-01-26 08:12:05,091 P35997 INFO Start training: 6910 batches/epoch
2022-01-26 08:12:05,091 P35997 INFO ************ Epoch=1 start ************
2022-01-26 08:49:48,866 P35997 INFO [Metrics] AUC: 0.694862 - logloss: 0.442176
2022-01-26 08:49:48,869 P35997 INFO Save best model: monitor(max): 0.694862
2022-01-26 08:49:49,102 P35997 INFO --- 6910/6910 batches finished ---
2022-01-26 08:49:49,135 P35997 INFO Train loss: 0.443377
2022-01-26 08:49:49,135 P35997 INFO ************ Epoch=1 end ************
2022-01-26 09:27:29,738 P35997 INFO [Metrics] AUC: 0.694253 - logloss: 0.450116
2022-01-26 09:27:29,740 P35997 INFO Monitor(max) STOP: 0.694253 !
2022-01-26 09:27:29,740 P35997 INFO Reduce learning rate on plateau: 0.000100
2022-01-26 09:27:29,740 P35997 INFO --- 6910/6910 batches finished ---
2022-01-26 09:27:29,781 P35997 INFO Train loss: 0.441145
2022-01-26 09:27:29,782 P35997 INFO ************ Epoch=2 end ************
2022-01-26 10:05:02,487 P35997 INFO [Metrics] AUC: 0.731436 - logloss: 0.404059
2022-01-26 10:05:02,489 P35997 INFO Save best model: monitor(max): 0.731436
2022-01-26 10:05:02,580 P35997 INFO --- 6910/6910 batches finished ---
2022-01-26 10:05:02,626 P35997 INFO Train loss: 0.408753
2022-01-26 10:05:02,626 P35997 INFO ************ Epoch=3 end ************
2022-01-26 10:42:31,706 P35997 INFO [Metrics] AUC: 0.734940 - logloss: 0.402441
2022-01-26 10:42:31,709 P35997 INFO Save best model: monitor(max): 0.734940
2022-01-26 10:42:31,785 P35997 INFO --- 6910/6910 batches finished ---
2022-01-26 10:42:31,829 P35997 INFO Train loss: 0.410734
2022-01-26 10:42:31,830 P35997 INFO ************ Epoch=4 end ************
2022-01-26 11:19:58,960 P35997 INFO [Metrics] AUC: 0.732611 - logloss: 0.405762
2022-01-26 11:19:58,964 P35997 INFO Monitor(max) STOP: 0.732611 !
2022-01-26 11:19:58,964 P35997 INFO Reduce learning rate on plateau: 0.000010
2022-01-26 11:19:58,964 P35997 INFO --- 6910/6910 batches finished ---
2022-01-26 11:19:59,002 P35997 INFO Train loss: 0.412090
2022-01-26 11:19:59,003 P35997 INFO ************ Epoch=5 end ************
2022-01-26 11:57:23,659 P35997 INFO [Metrics] AUC: 0.747779 - logloss: 0.394999
2022-01-26 11:57:23,662 P35997 INFO Save best model: monitor(max): 0.747779
2022-01-26 11:57:23,742 P35997 INFO --- 6910/6910 batches finished ---
2022-01-26 11:57:23,790 P35997 INFO Train loss: 0.397874
2022-01-26 11:57:23,791 P35997 INFO ************ Epoch=6 end ************
2022-01-26 12:34:46,953 P35997 INFO [Metrics] AUC: 0.743894 - logloss: 0.396379
2022-01-26 12:34:46,955 P35997 INFO Monitor(max) STOP: 0.743894 !
2022-01-26 12:34:46,955 P35997 INFO Reduce learning rate on plateau: 0.000001
2022-01-26 12:34:46,955 P35997 INFO --- 6910/6910 batches finished ---
2022-01-26 12:34:46,995 P35997 INFO Train loss: 0.396184
2022-01-26 12:34:46,996 P35997 INFO ************ Epoch=7 end ************
2022-01-26 13:12:11,459 P35997 INFO [Metrics] AUC: 0.744298 - logloss: 0.396660
2022-01-26 13:12:11,461 P35997 INFO Monitor(max) STOP: 0.744298 !
2022-01-26 13:12:11,461 P35997 INFO Reduce learning rate on plateau: 0.000001
2022-01-26 13:12:11,461 P35997 INFO Early stopping at epoch=8
2022-01-26 13:12:11,461 P35997 INFO --- 6910/6910 batches finished ---
2022-01-26 13:12:11,504 P35997 INFO Train loss: 0.390987
2022-01-26 13:12:11,505 P35997 INFO Training finished.
2022-01-26 13:12:11,505 P35997 INFO Load best model: /cache/FuxiCTR/benchmarks_modelarts/Avazu/FiBiNET_avazu_x1/avazu_x1_3fb65689/FiBiNET_avazu_x1_002_f09a284c.model
2022-01-26 13:12:17,093 P35997 INFO ****** Validation evaluation ******
2022-01-26 13:12:56,334 P35997 INFO [Metrics] AUC: 0.747779 - logloss: 0.394999
2022-01-26 13:12:56,397 P35997 INFO ******** Test evaluation ********
2022-01-26 13:12:56,397 P35997 INFO Loading data...
2022-01-26 13:12:56,397 P35997 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-01-26 13:12:57,265 P35997 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-01-26 13:12:57,265 P35997 INFO Loading test data done.
2022-01-26 13:14:16,753 P35997 INFO [Metrics] AUC: 0.764514 - logloss: 0.367047

```
