## FFM_avazu_x1

A hands-on guide to run the FFM model on the Avazu_x1 dataset.

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

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [FFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FFM_avazu_x1_tuner_config_01](./FFM_avazu_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FFM_avazu_x1
    nohup python run_expid.py --config ./FFM_avazu_x1_tuner_config_01 --expid FFMv2_avazu_x1_003_2c82d4bd --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.762534 | 0.367564  |


### Logs
```python
2021-03-22 10:03:25,209 P55897 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_83355fc7",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FFM_v2",
    "model_id": "FFM_v2_avazu_x1_003_e7bb5b13",
    "model_root": "./Avazu/FFM_avazu/FFM_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-05",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2021-03-22 10:03:25,209 P55897 INFO Set up feature encoder...
2021-03-22 10:03:25,209 P55897 INFO Load feature_encoder from pickle: ../data/Avazu/avazu_x1_83355fc7/feature_encoder.pkl
2021-03-22 10:03:32,019 P55897 INFO Total number of parameters: 274004390.
2021-03-22 10:03:32,020 P55897 INFO Loading data...
2021-03-22 10:03:32,022 P55897 INFO Loading data from h5: ../data/Avazu/avazu_x1_83355fc7/train.h5
2021-03-22 10:03:34,526 P55897 INFO Loading data from h5: ../data/Avazu/avazu_x1_83355fc7/valid.h5
2021-03-22 10:03:34,867 P55897 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2021-03-22 10:03:34,867 P55897 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2021-03-22 10:03:34,867 P55897 INFO Loading train data done.
2021-03-22 10:03:37,884 P55897 INFO Start training: 6910 batches/epoch
2021-03-22 10:03:37,885 P55897 INFO ************ Epoch=1 start ************
2021-03-22 10:21:17,064 P55897 INFO [Metrics] AUC: 0.745524 - logloss: 0.397833
2021-03-22 10:21:17,066 P55897 INFO Save best model: monitor(max): 0.745524
2021-03-22 10:21:18,348 P55897 INFO --- 6910/6910 batches finished ---
2021-03-22 10:21:18,393 P55897 INFO Train loss: 0.409041
2021-03-22 10:21:18,393 P55897 INFO ************ Epoch=1 end ************
2021-03-22 10:39:01,917 P55897 INFO [Metrics] AUC: 0.744636 - logloss: 0.398278
2021-03-22 10:39:01,920 P55897 INFO Monitor(max) STOP: 0.744636 !
2021-03-22 10:39:01,921 P55897 INFO Reduce learning rate on plateau: 0.000100
2021-03-22 10:39:01,921 P55897 INFO --- 6910/6910 batches finished ---
2021-03-22 10:39:02,001 P55897 INFO Train loss: 0.404549
2021-03-22 10:39:02,001 P55897 INFO ************ Epoch=2 end ************
2021-03-22 10:56:47,371 P55897 INFO [Metrics] AUC: 0.746299 - logloss: 0.397378
2021-03-22 10:56:47,374 P55897 INFO Save best model: monitor(max): 0.746299
2021-03-22 10:56:49,670 P55897 INFO --- 6910/6910 batches finished ---
2021-03-22 10:56:49,747 P55897 INFO Train loss: 0.395094
2021-03-22 10:56:49,747 P55897 INFO ************ Epoch=3 end ************
2021-03-22 11:14:31,138 P55897 INFO [Metrics] AUC: 0.746163 - logloss: 0.396957
2021-03-22 11:14:31,140 P55897 INFO Monitor(max) STOP: 0.746163 !
2021-03-22 11:14:31,140 P55897 INFO Reduce learning rate on plateau: 0.000010
2021-03-22 11:14:31,140 P55897 INFO --- 6910/6910 batches finished ---
2021-03-22 11:14:31,219 P55897 INFO Train loss: 0.393051
2021-03-22 11:14:31,219 P55897 INFO ************ Epoch=4 end ************
2021-03-22 11:32:10,337 P55897 INFO [Metrics] AUC: 0.746075 - logloss: 0.397145
2021-03-22 11:32:10,339 P55897 INFO Monitor(max) STOP: 0.746075 !
2021-03-22 11:32:10,339 P55897 INFO Reduce learning rate on plateau: 0.000001
2021-03-22 11:32:10,339 P55897 INFO Early stopping at epoch=5
2021-03-22 11:32:10,340 P55897 INFO --- 6910/6910 batches finished ---
2021-03-22 11:32:10,415 P55897 INFO Train loss: 0.390754
2021-03-22 11:32:10,415 P55897 INFO Training finished.
2021-03-22 11:32:10,415 P55897 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Avazu/FFM_avazu/FFM_avazu_x1/avazu_x1_83355fc7/FFM_v2_avazu_x1_003_e7bb5b13_model.ckpt
2021-03-22 11:32:11,843 P55897 INFO ****** Train/validation evaluation ******
2021-03-22 11:32:27,464 P55897 INFO [Metrics] AUC: 0.746299 - logloss: 0.397378
2021-03-22 11:32:27,520 P55897 INFO ******** Test evaluation ********
2021-03-22 11:32:27,520 P55897 INFO Loading data...
2021-03-22 11:32:27,521 P55897 INFO Loading data from h5: ../data/Avazu/avazu_x1_83355fc7/test.h5
2021-03-22 11:32:28,228 P55897 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2021-03-22 11:32:28,228 P55897 INFO Loading test data done.
2021-03-22 11:32:58,400 P55897 INFO [Metrics] AUC: 0.762534 - logloss: 0.367564

```
