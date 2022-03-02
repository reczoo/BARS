## FFM_movielenslatest_x1

A hands-on guide to run the FFM model on the Movielenslatest_x1 dataset.

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
Dataset ID: [Movielenslatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Movielenslatest/README.md#Movielenslatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [FFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FFM_movielenslatest_x1_tuner_config_03](./FFM_movielenslatest_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FFM_movielenslatest_x1
    nohup python run_expid.py --config ./FFM_movielenslatest_x1_tuner_config_03 --expid FFM_movielenslatest_x1_001_79d12e82 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.955627 | 0.242157  |


### Logs
```python
2021-03-15 22:56:14,183 P27777 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/MovielensLatest/",
    "dataset_id": "movielenslatest_x0_bcd26aed",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FFM",
    "model_id": "FFM_movielenslatest_x0_001_33a2d50b",
    "model_root": "./MovielensLatest/FFM_movielenslatest_x0/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-06",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/MovielensLatest/MovielensLatest_x0/test.csv",
    "train_data": "../data/MovielensLatest/MovielensLatest_x0/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/MovielensLatest/MovielensLatest_x0/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2021-03-15 22:56:14,184 P27777 INFO Set up feature encoder...
2021-03-15 22:56:14,184 P27777 INFO Load feature_encoder from pickle: ../data/MovielensLatest/movielenslatest_x0_bcd26aed/feature_encoder.pkl
2021-03-15 22:56:15,162 P27777 INFO Total number of parameters: 1895020.
2021-03-15 22:56:15,162 P27777 INFO Loading data...
2021-03-15 22:56:15,165 P27777 INFO Loading data from h5: ../data/MovielensLatest/movielenslatest_x0_bcd26aed/train.h5
2021-03-15 22:56:15,193 P27777 INFO Loading data from h5: ../data/MovielensLatest/movielenslatest_x0_bcd26aed/valid.h5
2021-03-15 22:56:15,201 P27777 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2021-03-15 22:56:15,202 P27777 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2021-03-15 22:56:15,202 P27777 INFO Loading train data done.
2021-03-15 22:56:18,694 P27777 INFO Start training: 343 batches/epoch
2021-03-15 22:56:18,695 P27777 INFO ************ Epoch=1 start ************
2021-03-15 22:56:29,295 P27777 INFO [Metrics] AUC: 0.918588 - logloss: 0.419209
2021-03-15 22:56:29,296 P27777 INFO Save best model: monitor(max): 0.918588
2021-03-15 22:56:29,308 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:56:29,355 P27777 INFO Train loss: 0.570728
2021-03-15 22:56:29,355 P27777 INFO ************ Epoch=1 end ************
2021-03-15 22:56:39,892 P27777 INFO [Metrics] AUC: 0.933624 - logloss: 0.309145
2021-03-15 22:56:39,894 P27777 INFO Save best model: monitor(max): 0.933624
2021-03-15 22:56:39,908 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:56:39,965 P27777 INFO Train loss: 0.348651
2021-03-15 22:56:39,965 P27777 INFO ************ Epoch=2 end ************
2021-03-15 22:56:50,338 P27777 INFO [Metrics] AUC: 0.939323 - logloss: 0.285040
2021-03-15 22:56:50,339 P27777 INFO Save best model: monitor(max): 0.939323
2021-03-15 22:56:50,353 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:56:50,421 P27777 INFO Train loss: 0.291125
2021-03-15 22:56:50,421 P27777 INFO ************ Epoch=3 end ************
2021-03-15 22:57:00,748 P27777 INFO [Metrics] AUC: 0.942345 - logloss: 0.275311
2021-03-15 22:57:00,749 P27777 INFO Save best model: monitor(max): 0.942345
2021-03-15 22:57:00,762 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:57:00,813 P27777 INFO Train loss: 0.271902
2021-03-15 22:57:00,813 P27777 INFO ************ Epoch=4 end ************
2021-03-15 22:57:10,508 P27777 INFO [Metrics] AUC: 0.944346 - logloss: 0.269661
2021-03-15 22:57:10,509 P27777 INFO Save best model: monitor(max): 0.944346
2021-03-15 22:57:10,522 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:57:10,591 P27777 INFO Train loss: 0.260572
2021-03-15 22:57:10,591 P27777 INFO ************ Epoch=5 end ************
2021-03-15 22:57:20,704 P27777 INFO [Metrics] AUC: 0.945837 - logloss: 0.265656
2021-03-15 22:57:20,705 P27777 INFO Save best model: monitor(max): 0.945837
2021-03-15 22:57:20,719 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:57:20,831 P27777 INFO Train loss: 0.252100
2021-03-15 22:57:20,832 P27777 INFO ************ Epoch=6 end ************
2021-03-15 22:57:30,911 P27777 INFO [Metrics] AUC: 0.947080 - logloss: 0.262475
2021-03-15 22:57:30,912 P27777 INFO Save best model: monitor(max): 0.947080
2021-03-15 22:57:30,924 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:57:31,003 P27777 INFO Train loss: 0.245056
2021-03-15 22:57:31,003 P27777 INFO ************ Epoch=7 end ************
2021-03-15 22:57:40,905 P27777 INFO [Metrics] AUC: 0.948134 - logloss: 0.259791
2021-03-15 22:57:40,906 P27777 INFO Save best model: monitor(max): 0.948134
2021-03-15 22:57:40,917 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:57:40,961 P27777 INFO Train loss: 0.238826
2021-03-15 22:57:40,961 P27777 INFO ************ Epoch=8 end ************
2021-03-15 22:57:50,437 P27777 INFO [Metrics] AUC: 0.949070 - logloss: 0.257385
2021-03-15 22:57:50,438 P27777 INFO Save best model: monitor(max): 0.949070
2021-03-15 22:57:50,453 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:57:50,545 P27777 INFO Train loss: 0.233089
2021-03-15 22:57:50,546 P27777 INFO ************ Epoch=9 end ************
2021-03-15 22:58:00,031 P27777 INFO [Metrics] AUC: 0.949927 - logloss: 0.255191
2021-03-15 22:58:00,032 P27777 INFO Save best model: monitor(max): 0.949927
2021-03-15 22:58:00,042 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:58:00,091 P27777 INFO Train loss: 0.227666
2021-03-15 22:58:00,091 P27777 INFO ************ Epoch=10 end ************
2021-03-15 22:58:09,420 P27777 INFO [Metrics] AUC: 0.950683 - logloss: 0.253181
2021-03-15 22:58:09,421 P27777 INFO Save best model: monitor(max): 0.950683
2021-03-15 22:58:09,435 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:58:09,500 P27777 INFO Train loss: 0.222495
2021-03-15 22:58:09,500 P27777 INFO ************ Epoch=11 end ************
2021-03-15 22:58:18,810 P27777 INFO [Metrics] AUC: 0.951403 - logloss: 0.251318
2021-03-15 22:58:18,811 P27777 INFO Save best model: monitor(max): 0.951403
2021-03-15 22:58:18,825 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:58:18,888 P27777 INFO Train loss: 0.217528
2021-03-15 22:58:18,888 P27777 INFO ************ Epoch=12 end ************
2021-03-15 22:58:28,021 P27777 INFO [Metrics] AUC: 0.952044 - logloss: 0.249601
2021-03-15 22:58:28,022 P27777 INFO Save best model: monitor(max): 0.952044
2021-03-15 22:58:28,032 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:58:28,086 P27777 INFO Train loss: 0.212707
2021-03-15 22:58:28,086 P27777 INFO ************ Epoch=13 end ************
2021-03-15 22:58:37,295 P27777 INFO [Metrics] AUC: 0.952629 - logloss: 0.248050
2021-03-15 22:58:37,296 P27777 INFO Save best model: monitor(max): 0.952629
2021-03-15 22:58:37,307 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:58:37,364 P27777 INFO Train loss: 0.208012
2021-03-15 22:58:37,364 P27777 INFO ************ Epoch=14 end ************
2021-03-15 22:58:46,180 P27777 INFO [Metrics] AUC: 0.953150 - logloss: 0.246640
2021-03-15 22:58:46,181 P27777 INFO Save best model: monitor(max): 0.953150
2021-03-15 22:58:46,192 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:58:46,242 P27777 INFO Train loss: 0.203437
2021-03-15 22:58:46,242 P27777 INFO ************ Epoch=15 end ************
2021-03-15 22:58:55,200 P27777 INFO [Metrics] AUC: 0.953630 - logloss: 0.245380
2021-03-15 22:58:55,200 P27777 INFO Save best model: monitor(max): 0.953630
2021-03-15 22:58:55,214 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:58:55,276 P27777 INFO Train loss: 0.198950
2021-03-15 22:58:55,276 P27777 INFO ************ Epoch=16 end ************
2021-03-15 22:59:04,107 P27777 INFO [Metrics] AUC: 0.954031 - logloss: 0.244280
2021-03-15 22:59:04,108 P27777 INFO Save best model: monitor(max): 0.954031
2021-03-15 22:59:04,119 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:59:04,179 P27777 INFO Train loss: 0.194593
2021-03-15 22:59:04,179 P27777 INFO ************ Epoch=17 end ************
2021-03-15 22:59:13,246 P27777 INFO [Metrics] AUC: 0.954410 - logloss: 0.243317
2021-03-15 22:59:13,247 P27777 INFO Save best model: monitor(max): 0.954410
2021-03-15 22:59:13,270 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:59:13,319 P27777 INFO Train loss: 0.190361
2021-03-15 22:59:13,319 P27777 INFO ************ Epoch=18 end ************
2021-03-15 22:59:22,555 P27777 INFO [Metrics] AUC: 0.954735 - logloss: 0.242498
2021-03-15 22:59:22,556 P27777 INFO Save best model: monitor(max): 0.954735
2021-03-15 22:59:22,567 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:59:22,642 P27777 INFO Train loss: 0.186276
2021-03-15 22:59:22,642 P27777 INFO ************ Epoch=19 end ************
2021-03-15 22:59:31,761 P27777 INFO [Metrics] AUC: 0.954998 - logloss: 0.241871
2021-03-15 22:59:31,762 P27777 INFO Save best model: monitor(max): 0.954998
2021-03-15 22:59:31,773 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:59:31,833 P27777 INFO Train loss: 0.182349
2021-03-15 22:59:31,833 P27777 INFO ************ Epoch=20 end ************
2021-03-15 22:59:40,537 P27777 INFO [Metrics] AUC: 0.955212 - logloss: 0.241368
2021-03-15 22:59:40,538 P27777 INFO Save best model: monitor(max): 0.955212
2021-03-15 22:59:40,548 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:59:40,612 P27777 INFO Train loss: 0.178587
2021-03-15 22:59:40,612 P27777 INFO ************ Epoch=21 end ************
2021-03-15 22:59:49,647 P27777 INFO [Metrics] AUC: 0.955375 - logloss: 0.241076
2021-03-15 22:59:49,648 P27777 INFO Save best model: monitor(max): 0.955375
2021-03-15 22:59:49,661 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:59:49,711 P27777 INFO Train loss: 0.175010
2021-03-15 22:59:49,711 P27777 INFO ************ Epoch=22 end ************
2021-03-15 22:59:58,824 P27777 INFO [Metrics] AUC: 0.955521 - logloss: 0.240865
2021-03-15 22:59:58,825 P27777 INFO Save best model: monitor(max): 0.955521
2021-03-15 22:59:58,836 P27777 INFO --- 343/343 batches finished ---
2021-03-15 22:59:58,883 P27777 INFO Train loss: 0.171583
2021-03-15 22:59:58,883 P27777 INFO ************ Epoch=23 end ************
2021-03-15 23:00:07,982 P27777 INFO [Metrics] AUC: 0.955636 - logloss: 0.240735
2021-03-15 23:00:07,983 P27777 INFO Save best model: monitor(max): 0.955636
2021-03-15 23:00:07,997 P27777 INFO --- 343/343 batches finished ---
2021-03-15 23:00:08,037 P27777 INFO Train loss: 0.168333
2021-03-15 23:00:08,038 P27777 INFO ************ Epoch=24 end ************
2021-03-15 23:00:17,086 P27777 INFO [Metrics] AUC: 0.955719 - logloss: 0.240735
2021-03-15 23:00:17,088 P27777 INFO Save best model: monitor(max): 0.955719
2021-03-15 23:00:17,101 P27777 INFO --- 343/343 batches finished ---
2021-03-15 23:00:17,168 P27777 INFO Train loss: 0.165219
2021-03-15 23:00:17,168 P27777 INFO ************ Epoch=25 end ************
2021-03-15 23:00:26,270 P27777 INFO [Metrics] AUC: 0.955780 - logloss: 0.240869
2021-03-15 23:00:26,271 P27777 INFO Save best model: monitor(max): 0.955780
2021-03-15 23:00:26,285 P27777 INFO --- 343/343 batches finished ---
2021-03-15 23:00:26,330 P27777 INFO Train loss: 0.162263
2021-03-15 23:00:26,330 P27777 INFO ************ Epoch=26 end ************
2021-03-15 23:00:35,229 P27777 INFO [Metrics] AUC: 0.955822 - logloss: 0.241060
2021-03-15 23:00:35,230 P27777 INFO Save best model: monitor(max): 0.955822
2021-03-15 23:00:35,241 P27777 INFO --- 343/343 batches finished ---
2021-03-15 23:00:35,280 P27777 INFO Train loss: 0.159436
2021-03-15 23:00:35,280 P27777 INFO ************ Epoch=27 end ************
2021-03-15 23:00:44,203 P27777 INFO [Metrics] AUC: 0.955831 - logloss: 0.241359
2021-03-15 23:00:44,204 P27777 INFO Save best model: monitor(max): 0.955831
2021-03-15 23:00:44,215 P27777 INFO --- 343/343 batches finished ---
2021-03-15 23:00:44,273 P27777 INFO Train loss: 0.156745
2021-03-15 23:00:44,273 P27777 INFO ************ Epoch=28 end ************
2021-03-15 23:00:53,396 P27777 INFO [Metrics] AUC: 0.955836 - logloss: 0.241663
2021-03-15 23:00:53,397 P27777 INFO Save best model: monitor(max): 0.955836
2021-03-15 23:00:53,411 P27777 INFO --- 343/343 batches finished ---
2021-03-15 23:00:53,473 P27777 INFO Train loss: 0.154170
2021-03-15 23:00:53,473 P27777 INFO ************ Epoch=29 end ************
2021-03-15 23:01:02,327 P27777 INFO [Metrics] AUC: 0.955812 - logloss: 0.242165
2021-03-15 23:01:02,328 P27777 INFO Monitor(max) STOP: 0.955812 !
2021-03-15 23:01:02,328 P27777 INFO Reduce learning rate on plateau: 0.000100
2021-03-15 23:01:02,328 P27777 INFO --- 343/343 batches finished ---
2021-03-15 23:01:02,376 P27777 INFO Train loss: 0.151738
2021-03-15 23:01:02,376 P27777 INFO ************ Epoch=30 end ************
2021-03-15 23:01:11,369 P27777 INFO [Metrics] AUC: 0.955817 - logloss: 0.242170
2021-03-15 23:01:11,370 P27777 INFO Monitor(max) STOP: 0.955817 !
2021-03-15 23:01:11,370 P27777 INFO Reduce learning rate on plateau: 0.000010
2021-03-15 23:01:11,370 P27777 INFO Early stopping at epoch=31
2021-03-15 23:01:11,370 P27777 INFO --- 343/343 batches finished ---
2021-03-15 23:01:11,421 P27777 INFO Train loss: 0.146282
2021-03-15 23:01:11,421 P27777 INFO Training finished.
2021-03-15 23:01:11,421 P27777 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/MovielensLatest/FFM_movielenslatest_x0/movielenslatest_x0_bcd26aed/FFM_movielenslatest_x0_001_33a2d50b_model.ckpt
2021-03-15 23:01:11,438 P27777 INFO ****** Train/validation evaluation ******
2021-03-15 23:01:12,665 P27777 INFO [Metrics] AUC: 0.955836 - logloss: 0.241663
2021-03-15 23:01:12,695 P27777 INFO ******** Test evaluation ********
2021-03-15 23:01:12,695 P27777 INFO Loading data...
2021-03-15 23:01:12,695 P27777 INFO Loading data from h5: ../data/MovielensLatest/movielenslatest_x0_bcd26aed/test.h5
2021-03-15 23:01:12,699 P27777 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2021-03-15 23:01:12,699 P27777 INFO Loading test data done.
2021-03-15 23:01:13,351 P27777 INFO [Metrics] AUC: 0.955627 - logloss: 0.242157

```
