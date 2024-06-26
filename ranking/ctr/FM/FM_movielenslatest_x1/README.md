## FM_movielenslatest_x1

A hands-on guide to run the FM model on the Movielenslatest_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

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
Dataset ID: [MovielensLatest_x1](https://github.com/reczoo/Datasets/tree/main/MovieLens/MovielensLatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [FM](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FM_movielenslatest_x1_tuner_config_03](./FM_movielenslatest_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FM_movielenslatest_x1
    nohup python run_expid.py --config ./FM_movielenslatest_x1_tuner_config_03 --expid FM_movielenslatest_x1_003_3e63baf6 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.943435 | 0.272893  |


### Logs
```python
2021-01-07 17:46:57,366 P6900 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/MovielensLatest/",
    "dataset_id": "movielenslatest_x1_bcd26aed",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FM",
    "model_id": "FM_movielenslatest_x1_003_b52fd4c1",
    "model_root": "./MovielensLatest/FM_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "5e-06",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/MovielensLatest/MovielensLatest_x1/test.csv",
    "train_data": "../data/MovielensLatest/MovielensLatest_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/MovielensLatest/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2021-01-07 17:46:57,366 P6900 INFO Set up feature encoder...
2021-01-07 17:46:57,367 P6900 INFO Load feature_encoder from pickle: ../data/MovielensLatest/movielenslatest_x1_bcd26aed/feature_encoder.pkl
2021-01-07 17:46:57,522 P6900 INFO Total number of parameters: 992630.
2021-01-07 17:46:57,522 P6900 INFO Loading data...
2021-01-07 17:46:57,524 P6900 INFO Loading data from h5: ../data/MovielensLatest/movielenslatest_x1_bcd26aed/train.h5
2021-01-07 17:46:57,557 P6900 INFO Loading data from h5: ../data/MovielensLatest/movielenslatest_x1_bcd26aed/valid.h5
2021-01-07 17:46:57,567 P6900 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2021-01-07 17:46:57,567 P6900 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2021-01-07 17:46:57,568 P6900 INFO Loading train data done.
2021-01-07 17:47:00,824 P6900 INFO Start training: 343 batches/epoch
2021-01-07 17:47:00,824 P6900 INFO ************ Epoch=1 start ************
2021-01-07 17:47:08,009 P6900 INFO [Metrics] AUC: 0.894945 - logloss: 0.500827
2021-01-07 17:47:08,009 P6900 INFO Save best model: monitor(max): 0.894945
2021-01-07 17:47:08,014 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:47:08,057 P6900 INFO Train loss: 0.606503
2021-01-07 17:47:08,057 P6900 INFO ************ Epoch=1 end ************
2021-01-07 17:47:15,325 P6900 INFO [Metrics] AUC: 0.916791 - logloss: 0.377244
2021-01-07 17:47:15,326 P6900 INFO Save best model: monitor(max): 0.916791
2021-01-07 17:47:15,331 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:47:15,394 P6900 INFO Train loss: 0.437394
2021-01-07 17:47:15,394 P6900 INFO ************ Epoch=2 end ************
2021-01-07 17:47:22,349 P6900 INFO [Metrics] AUC: 0.925686 - logloss: 0.331897
2021-01-07 17:47:22,349 P6900 INFO Save best model: monitor(max): 0.925686
2021-01-07 17:47:22,359 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:47:22,408 P6900 INFO Train loss: 0.365310
2021-01-07 17:47:22,408 P6900 INFO ************ Epoch=3 end ************
2021-01-07 17:47:29,734 P6900 INFO [Metrics] AUC: 0.930239 - logloss: 0.312579
2021-01-07 17:47:29,735 P6900 INFO Save best model: monitor(max): 0.930239
2021-01-07 17:47:29,740 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:47:29,783 P6900 INFO Train loss: 0.337556
2021-01-07 17:47:29,783 P6900 INFO ************ Epoch=4 end ************
2021-01-07 17:47:36,743 P6900 INFO [Metrics] AUC: 0.933064 - logloss: 0.302511
2021-01-07 17:47:36,744 P6900 INFO Save best model: monitor(max): 0.933064
2021-01-07 17:47:36,750 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:47:36,795 P6900 INFO Train loss: 0.323363
2021-01-07 17:47:36,795 P6900 INFO ************ Epoch=5 end ************
2021-01-07 17:47:44,018 P6900 INFO [Metrics] AUC: 0.935027 - logloss: 0.296305
2021-01-07 17:47:44,019 P6900 INFO Save best model: monitor(max): 0.935027
2021-01-07 17:47:44,027 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:47:44,087 P6900 INFO Train loss: 0.314287
2021-01-07 17:47:44,087 P6900 INFO ************ Epoch=6 end ************
2021-01-07 17:47:51,329 P6900 INFO [Metrics] AUC: 0.936443 - logloss: 0.292202
2021-01-07 17:47:51,330 P6900 INFO Save best model: monitor(max): 0.936443
2021-01-07 17:47:51,336 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:47:51,399 P6900 INFO Train loss: 0.307682
2021-01-07 17:47:51,399 P6900 INFO ************ Epoch=7 end ************
2021-01-07 17:47:58,470 P6900 INFO [Metrics] AUC: 0.937534 - logloss: 0.289120
2021-01-07 17:47:58,471 P6900 INFO Save best model: monitor(max): 0.937534
2021-01-07 17:47:58,477 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:47:58,534 P6900 INFO Train loss: 0.302505
2021-01-07 17:47:58,534 P6900 INFO ************ Epoch=8 end ************
2021-01-07 17:48:05,983 P6900 INFO [Metrics] AUC: 0.938431 - logloss: 0.286716
2021-01-07 17:48:05,983 P6900 INFO Save best model: monitor(max): 0.938431
2021-01-07 17:48:05,989 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:48:06,052 P6900 INFO Train loss: 0.298260
2021-01-07 17:48:06,052 P6900 INFO ************ Epoch=9 end ************
2021-01-07 17:48:13,283 P6900 INFO [Metrics] AUC: 0.939160 - logloss: 0.284789
2021-01-07 17:48:13,284 P6900 INFO Save best model: monitor(max): 0.939160
2021-01-07 17:48:13,290 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:48:13,336 P6900 INFO Train loss: 0.294649
2021-01-07 17:48:13,336 P6900 INFO ************ Epoch=10 end ************
2021-01-07 17:48:20,333 P6900 INFO [Metrics] AUC: 0.939754 - logloss: 0.283201
2021-01-07 17:48:20,333 P6900 INFO Save best model: monitor(max): 0.939754
2021-01-07 17:48:20,339 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:48:20,387 P6900 INFO Train loss: 0.291513
2021-01-07 17:48:20,388 P6900 INFO ************ Epoch=11 end ************
2021-01-07 17:48:27,635 P6900 INFO [Metrics] AUC: 0.940253 - logloss: 0.281821
2021-01-07 17:48:27,636 P6900 INFO Save best model: monitor(max): 0.940253
2021-01-07 17:48:27,642 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:48:27,689 P6900 INFO Train loss: 0.288728
2021-01-07 17:48:27,689 P6900 INFO ************ Epoch=12 end ************
2021-01-07 17:48:34,730 P6900 INFO [Metrics] AUC: 0.940706 - logloss: 0.280605
2021-01-07 17:48:34,731 P6900 INFO Save best model: monitor(max): 0.940706
2021-01-07 17:48:34,739 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:48:34,789 P6900 INFO Train loss: 0.286238
2021-01-07 17:48:34,789 P6900 INFO ************ Epoch=13 end ************
2021-01-07 17:48:41,644 P6900 INFO [Metrics] AUC: 0.941057 - logloss: 0.279608
2021-01-07 17:48:41,645 P6900 INFO Save best model: monitor(max): 0.941057
2021-01-07 17:48:41,651 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:48:41,709 P6900 INFO Train loss: 0.283980
2021-01-07 17:48:41,709 P6900 INFO ************ Epoch=14 end ************
2021-01-07 17:48:48,629 P6900 INFO [Metrics] AUC: 0.941424 - logloss: 0.278661
2021-01-07 17:48:48,630 P6900 INFO Save best model: monitor(max): 0.941424
2021-01-07 17:48:48,637 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:48:48,684 P6900 INFO Train loss: 0.281924
2021-01-07 17:48:48,684 P6900 INFO ************ Epoch=15 end ************
2021-01-07 17:48:55,850 P6900 INFO [Metrics] AUC: 0.941719 - logloss: 0.277839
2021-01-07 17:48:55,851 P6900 INFO Save best model: monitor(max): 0.941719
2021-01-07 17:48:55,857 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:48:55,944 P6900 INFO Train loss: 0.280024
2021-01-07 17:48:55,944 P6900 INFO ************ Epoch=16 end ************
2021-01-07 17:49:02,665 P6900 INFO [Metrics] AUC: 0.941968 - logloss: 0.277138
2021-01-07 17:49:02,665 P6900 INFO Save best model: monitor(max): 0.941968
2021-01-07 17:49:02,671 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:49:02,726 P6900 INFO Train loss: 0.278260
2021-01-07 17:49:02,727 P6900 INFO ************ Epoch=17 end ************
2021-01-07 17:49:09,637 P6900 INFO [Metrics] AUC: 0.942213 - logloss: 0.276480
2021-01-07 17:49:09,637 P6900 INFO Save best model: monitor(max): 0.942213
2021-01-07 17:49:09,643 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:49:09,709 P6900 INFO Train loss: 0.276625
2021-01-07 17:49:09,709 P6900 INFO ************ Epoch=18 end ************
2021-01-07 17:49:16,646 P6900 INFO [Metrics] AUC: 0.942412 - logloss: 0.275912
2021-01-07 17:49:16,646 P6900 INFO Save best model: monitor(max): 0.942412
2021-01-07 17:49:16,652 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:49:16,701 P6900 INFO Train loss: 0.275074
2021-01-07 17:49:16,702 P6900 INFO ************ Epoch=19 end ************
2021-01-07 17:49:23,853 P6900 INFO [Metrics] AUC: 0.942568 - logloss: 0.275462
2021-01-07 17:49:23,853 P6900 INFO Save best model: monitor(max): 0.942568
2021-01-07 17:49:23,859 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:49:23,906 P6900 INFO Train loss: 0.273621
2021-01-07 17:49:23,906 P6900 INFO ************ Epoch=20 end ************
2021-01-07 17:49:31,112 P6900 INFO [Metrics] AUC: 0.942728 - logloss: 0.275043
2021-01-07 17:49:31,113 P6900 INFO Save best model: monitor(max): 0.942728
2021-01-07 17:49:31,119 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:49:31,169 P6900 INFO Train loss: 0.272226
2021-01-07 17:49:31,170 P6900 INFO ************ Epoch=21 end ************
2021-01-07 17:49:38,475 P6900 INFO [Metrics] AUC: 0.942873 - logloss: 0.274649
2021-01-07 17:49:38,475 P6900 INFO Save best model: monitor(max): 0.942873
2021-01-07 17:49:38,481 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:49:38,535 P6900 INFO Train loss: 0.270912
2021-01-07 17:49:38,535 P6900 INFO ************ Epoch=22 end ************
2021-01-07 17:49:45,254 P6900 INFO [Metrics] AUC: 0.942971 - logloss: 0.274347
2021-01-07 17:49:45,254 P6900 INFO Save best model: monitor(max): 0.942971
2021-01-07 17:49:45,261 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:49:45,325 P6900 INFO Train loss: 0.269667
2021-01-07 17:49:45,325 P6900 INFO ************ Epoch=23 end ************
2021-01-07 17:49:52,231 P6900 INFO [Metrics] AUC: 0.943062 - logloss: 0.274051
2021-01-07 17:49:52,232 P6900 INFO Save best model: monitor(max): 0.943062
2021-01-07 17:49:52,238 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:49:52,309 P6900 INFO Train loss: 0.268479
2021-01-07 17:49:52,309 P6900 INFO ************ Epoch=24 end ************
2021-01-07 17:49:58,613 P6900 INFO [Metrics] AUC: 0.943146 - logloss: 0.273793
2021-01-07 17:49:58,614 P6900 INFO Save best model: monitor(max): 0.943146
2021-01-07 17:49:58,620 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:49:58,685 P6900 INFO Train loss: 0.267328
2021-01-07 17:49:58,685 P6900 INFO ************ Epoch=25 end ************
2021-01-07 17:50:04,899 P6900 INFO [Metrics] AUC: 0.943187 - logloss: 0.273585
2021-01-07 17:50:04,900 P6900 INFO Save best model: monitor(max): 0.943187
2021-01-07 17:50:04,905 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:50:04,957 P6900 INFO Train loss: 0.266245
2021-01-07 17:50:04,957 P6900 INFO ************ Epoch=26 end ************
2021-01-07 17:50:10,984 P6900 INFO [Metrics] AUC: 0.943255 - logloss: 0.273411
2021-01-07 17:50:10,985 P6900 INFO Save best model: monitor(max): 0.943255
2021-01-07 17:50:10,991 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:50:11,038 P6900 INFO Train loss: 0.265206
2021-01-07 17:50:11,038 P6900 INFO ************ Epoch=27 end ************
2021-01-07 17:50:17,373 P6900 INFO [Metrics] AUC: 0.943306 - logloss: 0.273279
2021-01-07 17:50:17,373 P6900 INFO Save best model: monitor(max): 0.943306
2021-01-07 17:50:17,379 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:50:17,429 P6900 INFO Train loss: 0.264226
2021-01-07 17:50:17,430 P6900 INFO ************ Epoch=28 end ************
2021-01-07 17:50:23,464 P6900 INFO [Metrics] AUC: 0.943324 - logloss: 0.273150
2021-01-07 17:50:23,464 P6900 INFO Save best model: monitor(max): 0.943324
2021-01-07 17:50:23,470 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:50:23,522 P6900 INFO Train loss: 0.263262
2021-01-07 17:50:23,522 P6900 INFO ************ Epoch=29 end ************
2021-01-07 17:50:29,685 P6900 INFO [Metrics] AUC: 0.943280 - logloss: 0.273163
2021-01-07 17:50:29,685 P6900 INFO Monitor(max) STOP: 0.943280 !
2021-01-07 17:50:29,685 P6900 INFO Reduce learning rate on plateau: 0.000100
2021-01-07 17:50:29,685 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:50:29,741 P6900 INFO Train loss: 0.262348
2021-01-07 17:50:29,742 P6900 INFO ************ Epoch=30 end ************
2021-01-07 17:50:35,931 P6900 INFO [Metrics] AUC: 0.943302 - logloss: 0.273083
2021-01-07 17:50:35,931 P6900 INFO Monitor(max) STOP: 0.943302 !
2021-01-07 17:50:35,932 P6900 INFO Reduce learning rate on plateau: 0.000010
2021-01-07 17:50:35,932 P6900 INFO Early stopping at epoch=31
2021-01-07 17:50:35,932 P6900 INFO --- 343/343 batches finished ---
2021-01-07 17:50:35,996 P6900 INFO Train loss: 0.256959
2021-01-07 17:50:35,996 P6900 INFO Training finished.
2021-01-07 17:50:35,996 P6900 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/MovielensLatest/FM_movielenslatest_x1/movielenslatest_x1_bcd26aed/FM_movielenslatest_x1_003_b52fd4c1_model.ckpt
2021-01-07 17:50:36,006 P6900 INFO ****** Train/validation evaluation ******
2021-01-07 17:50:37,481 P6900 INFO [Metrics] AUC: 0.943324 - logloss: 0.273150
2021-01-07 17:50:37,581 P6900 INFO ******** Test evaluation ********
2021-01-07 17:50:37,581 P6900 INFO Loading data...
2021-01-07 17:50:37,582 P6900 INFO Loading data from h5: ../data/MovielensLatest/movielenslatest_x1_bcd26aed/test.h5
2021-01-07 17:50:37,586 P6900 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2021-01-07 17:50:37,586 P6900 INFO Loading test data done.
2021-01-07 17:50:38,487 P6900 INFO [Metrics] AUC: 0.943435 - logloss: 0.272893

```
