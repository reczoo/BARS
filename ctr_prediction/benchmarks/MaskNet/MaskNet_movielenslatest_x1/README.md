## MaskNet_movielenslatest_x1

A hands-on guide to run the MaskNet model on the Movielenslatest_x1 dataset.

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
  fuxictr: 1.1.1
  ```

### Dataset
Dataset ID: [Movielenslatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/MovieLens/README.md#Movielenslatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.1](https://github.com/xue-pai/FuxiCTR/tree/v1.1.1) for this experiment. See the model code: [MaskNet](https://github.com/xue-pai/FuxiCTR/blob/v1.1.1/fuxictr/pytorch/models/MaskNet.py).

Running steps:

1. Download [FuxiCTR-v1.1.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [MaskNet_movielenslatest_x1_tuner_config_02](./MaskNet_movielenslatest_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd MaskNet_movielenslatest_x1
    nohup python run_expid.py --config ./MaskNet_movielenslatest_x1_tuner_config_02 --expid MaskNet_movielenslatest_x1_007_d44f3aed --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.968655 | 0.226563  |
| 2 | 0.968582 | 0.224182  |
| 3 | 0.967955 | 0.222666  |
| 4 | 0.968199 | 0.228566  |
| 5 | 0.968071 | 0.222331  |
| | | | 
| Avg | 0.968292 | 0.224862 |
| Std | &#177;0.00027818 | &#177;0.00237923 |


### Logs
```python
2022-01-29 11:17:31,720 P8307 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "dnn_hidden_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "emb_layernorm": "True",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "MaskNet",
    "model_id": "MaskNet_movielenslatest_x1_007_d44f3aed",
    "model_root": "./Movielens/MaskNet_movielenslatest_x1/",
    "model_type": "SerialMaskNet",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.5",
    "net_layernorm": "False",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_block_dim": "64",
    "parallel_num_blocks": "1",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "0.5",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Movielens/MovielensLatest_x1/test.csv",
    "train_data": "../data/Movielens/MovielensLatest_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-29 11:17:31,720 P8307 INFO Set up feature encoder...
2022-01-29 11:17:31,721 P8307 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-01-29 11:17:31,721 P8307 INFO Loading data...
2022-01-29 11:17:31,723 P8307 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-01-29 11:17:31,751 P8307 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-01-29 11:17:31,759 P8307 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-01-29 11:17:31,760 P8307 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-01-29 11:17:31,760 P8307 INFO Loading train data done.
2022-01-29 11:17:34,917 P8307 INFO Total number of parameters: 1408996.
2022-01-29 11:17:34,918 P8307 INFO Start training: 343 batches/epoch
2022-01-29 11:17:34,918 P8307 INFO ************ Epoch=1 start ************
2022-01-29 11:17:45,467 P8307 INFO [Metrics] AUC: 0.934451 - logloss: 0.291797
2022-01-29 11:17:45,468 P8307 INFO Save best model: monitor(max): 0.934451
2022-01-29 11:17:45,475 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:17:45,516 P8307 INFO Train loss: 0.425826
2022-01-29 11:17:45,516 P8307 INFO ************ Epoch=1 end ************
2022-01-29 11:17:51,955 P8307 INFO [Metrics] AUC: 0.942202 - logloss: 0.272248
2022-01-29 11:17:51,956 P8307 INFO Save best model: monitor(max): 0.942202
2022-01-29 11:17:51,964 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:17:52,003 P8307 INFO Train loss: 0.401172
2022-01-29 11:17:52,003 P8307 INFO ************ Epoch=2 end ************
2022-01-29 11:17:58,332 P8307 INFO [Metrics] AUC: 0.945456 - logloss: 0.263593
2022-01-29 11:17:58,332 P8307 INFO Save best model: monitor(max): 0.945456
2022-01-29 11:17:58,341 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:17:58,381 P8307 INFO Train loss: 0.400195
2022-01-29 11:17:58,381 P8307 INFO ************ Epoch=3 end ************
2022-01-29 11:18:07,427 P8307 INFO [Metrics] AUC: 0.948211 - logloss: 0.256923
2022-01-29 11:18:07,428 P8307 INFO Save best model: monitor(max): 0.948211
2022-01-29 11:18:07,437 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:18:07,476 P8307 INFO Train loss: 0.400181
2022-01-29 11:18:07,477 P8307 INFO ************ Epoch=4 end ************
2022-01-29 11:18:16,563 P8307 INFO [Metrics] AUC: 0.949875 - logloss: 0.252333
2022-01-29 11:18:16,564 P8307 INFO Save best model: monitor(max): 0.949875
2022-01-29 11:18:16,572 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:18:16,612 P8307 INFO Train loss: 0.399246
2022-01-29 11:18:16,612 P8307 INFO ************ Epoch=5 end ************
2022-01-29 11:18:25,575 P8307 INFO [Metrics] AUC: 0.951367 - logloss: 0.248042
2022-01-29 11:18:25,575 P8307 INFO Save best model: monitor(max): 0.951367
2022-01-29 11:18:25,583 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:18:25,625 P8307 INFO Train loss: 0.399238
2022-01-29 11:18:25,625 P8307 INFO ************ Epoch=6 end ************
2022-01-29 11:18:34,473 P8307 INFO [Metrics] AUC: 0.951837 - logloss: 0.246448
2022-01-29 11:18:34,474 P8307 INFO Save best model: monitor(max): 0.951837
2022-01-29 11:18:34,486 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:18:34,527 P8307 INFO Train loss: 0.399176
2022-01-29 11:18:34,527 P8307 INFO ************ Epoch=7 end ************
2022-01-29 11:18:43,485 P8307 INFO [Metrics] AUC: 0.953351 - logloss: 0.242882
2022-01-29 11:18:43,486 P8307 INFO Save best model: monitor(max): 0.953351
2022-01-29 11:18:43,494 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:18:43,534 P8307 INFO Train loss: 0.398524
2022-01-29 11:18:43,534 P8307 INFO ************ Epoch=8 end ************
2022-01-29 11:18:52,343 P8307 INFO [Metrics] AUC: 0.953960 - logloss: 0.241020
2022-01-29 11:18:52,343 P8307 INFO Save best model: monitor(max): 0.953960
2022-01-29 11:18:52,352 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:18:52,392 P8307 INFO Train loss: 0.396503
2022-01-29 11:18:52,392 P8307 INFO ************ Epoch=9 end ************
2022-01-29 11:19:01,140 P8307 INFO [Metrics] AUC: 0.954321 - logloss: 0.239495
2022-01-29 11:19:01,141 P8307 INFO Save best model: monitor(max): 0.954321
2022-01-29 11:19:01,149 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:19:01,190 P8307 INFO Train loss: 0.395025
2022-01-29 11:19:01,191 P8307 INFO ************ Epoch=10 end ************
2022-01-29 11:19:09,997 P8307 INFO [Metrics] AUC: 0.955070 - logloss: 0.238484
2022-01-29 11:19:09,998 P8307 INFO Save best model: monitor(max): 0.955070
2022-01-29 11:19:10,006 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:19:10,049 P8307 INFO Train loss: 0.393201
2022-01-29 11:19:10,049 P8307 INFO ************ Epoch=11 end ************
2022-01-29 11:19:18,936 P8307 INFO [Metrics] AUC: 0.955216 - logloss: 0.238256
2022-01-29 11:19:18,936 P8307 INFO Save best model: monitor(max): 0.955216
2022-01-29 11:19:18,945 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:19:18,984 P8307 INFO Train loss: 0.392307
2022-01-29 11:19:18,984 P8307 INFO ************ Epoch=12 end ************
2022-01-29 11:19:27,863 P8307 INFO [Metrics] AUC: 0.955621 - logloss: 0.236567
2022-01-29 11:19:27,863 P8307 INFO Save best model: monitor(max): 0.955621
2022-01-29 11:19:27,872 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:19:27,933 P8307 INFO Train loss: 0.393110
2022-01-29 11:19:27,933 P8307 INFO ************ Epoch=13 end ************
2022-01-29 11:19:36,787 P8307 INFO [Metrics] AUC: 0.956290 - logloss: 0.234109
2022-01-29 11:19:36,788 P8307 INFO Save best model: monitor(max): 0.956290
2022-01-29 11:19:36,796 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:19:36,835 P8307 INFO Train loss: 0.391854
2022-01-29 11:19:36,836 P8307 INFO ************ Epoch=14 end ************
2022-01-29 11:19:45,843 P8307 INFO [Metrics] AUC: 0.957038 - logloss: 0.233020
2022-01-29 11:19:45,844 P8307 INFO Save best model: monitor(max): 0.957038
2022-01-29 11:19:45,853 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:19:45,898 P8307 INFO Train loss: 0.391823
2022-01-29 11:19:45,898 P8307 INFO ************ Epoch=15 end ************
2022-01-29 11:19:55,145 P8307 INFO [Metrics] AUC: 0.956466 - logloss: 0.233735
2022-01-29 11:19:55,145 P8307 INFO Monitor(max) STOP: 0.956466 !
2022-01-29 11:19:55,145 P8307 INFO Reduce learning rate on plateau: 0.000100
2022-01-29 11:19:55,145 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:19:55,186 P8307 INFO Train loss: 0.391781
2022-01-29 11:19:55,186 P8307 INFO ************ Epoch=16 end ************
2022-01-29 11:20:04,301 P8307 INFO [Metrics] AUC: 0.967182 - logloss: 0.210530
2022-01-29 11:20:04,302 P8307 INFO Save best model: monitor(max): 0.967182
2022-01-29 11:20:04,311 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:20:04,350 P8307 INFO Train loss: 0.285841
2022-01-29 11:20:04,351 P8307 INFO ************ Epoch=17 end ************
2022-01-29 11:20:13,818 P8307 INFO [Metrics] AUC: 0.968945 - logloss: 0.224821
2022-01-29 11:20:13,818 P8307 INFO Save best model: monitor(max): 0.968945
2022-01-29 11:20:13,827 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:20:13,867 P8307 INFO Train loss: 0.192273
2022-01-29 11:20:13,867 P8307 INFO ************ Epoch=18 end ************
2022-01-29 11:20:23,078 P8307 INFO [Metrics] AUC: 0.968032 - logloss: 0.252728
2022-01-29 11:20:23,078 P8307 INFO Monitor(max) STOP: 0.968032 !
2022-01-29 11:20:23,078 P8307 INFO Reduce learning rate on plateau: 0.000010
2022-01-29 11:20:23,078 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:20:23,117 P8307 INFO Train loss: 0.143570
2022-01-29 11:20:23,117 P8307 INFO ************ Epoch=19 end ************
2022-01-29 11:20:32,526 P8307 INFO [Metrics] AUC: 0.967954 - logloss: 0.307460
2022-01-29 11:20:32,527 P8307 INFO Monitor(max) STOP: 0.967954 !
2022-01-29 11:20:32,527 P8307 INFO Reduce learning rate on plateau: 0.000001
2022-01-29 11:20:32,527 P8307 INFO Early stopping at epoch=20
2022-01-29 11:20:32,527 P8307 INFO --- 343/343 batches finished ---
2022-01-29 11:20:32,567 P8307 INFO Train loss: 0.109295
2022-01-29 11:20:32,567 P8307 INFO Training finished.
2022-01-29 11:20:32,567 P8307 INFO Load best model: /home/XXX/benchmarks/Movielens/MaskNet_movielenslatest_x1/movielenslatest_x1_cd32d937/MaskNet_movielenslatest_x1_007_d44f3aed.model
2022-01-29 11:20:35,941 P8307 INFO ****** Validation evaluation ******
2022-01-29 11:20:37,276 P8307 INFO [Metrics] AUC: 0.968945 - logloss: 0.224821
2022-01-29 11:20:37,315 P8307 INFO ******** Test evaluation ********
2022-01-29 11:20:37,315 P8307 INFO Loading data...
2022-01-29 11:20:37,316 P8307 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-01-29 11:20:37,320 P8307 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-01-29 11:20:37,320 P8307 INFO Loading test data done.
2022-01-29 11:20:38,078 P8307 INFO [Metrics] AUC: 0.968655 - logloss: 0.226563

```
