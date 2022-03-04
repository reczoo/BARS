## FmFM_movielenslatest_x1

A hands-on guide to run the FmFM model on the Movielenslatest_x1 dataset.

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
Dataset ID: [Movielenslatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/MovieLens/README.md#Movielenslatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [FmFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FmFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FmFM_movielenslatest_x1_tuner_config_03](./FmFM_movielenslatest_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FmFM_movielenslatest_x1
    nohup python run_expid.py --config ./FmFM_movielenslatest_x1_tuner_config_03 --expid FmFM_movielenslatest_x1_003_28d4fd9f --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.946525 | 0.271427  |


### Logs
```python
2022-01-20 07:59:48,864 P7046 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "embedding_dim": "10",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "field_interaction_type": "vectorized",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FmFM",
    "model_id": "FmFM_movielenslatest_x1_003_28d4fd9f",
    "model_root": "./Movielens/FmFM_movielenslatest_x1/",
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
    "test_data": "../data/Movielens/MovielensLatest_x1/test.csv",
    "train_data": "../data/Movielens/MovielensLatest_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-20 07:59:48,864 P7046 INFO Set up feature encoder...
2022-01-20 07:59:48,864 P7046 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-01-20 07:59:48,865 P7046 INFO Loading data...
2022-01-20 07:59:48,867 P7046 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-01-20 07:59:48,896 P7046 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-01-20 07:59:48,906 P7046 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-01-20 07:59:48,906 P7046 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-01-20 07:59:48,906 P7046 INFO Loading train data done.
2022-01-20 07:59:52,364 P7046 INFO Total number of parameters: 992660.
2022-01-20 07:59:52,364 P7046 INFO Start training: 343 batches/epoch
2022-01-20 07:59:52,364 P7046 INFO ************ Epoch=1 start ************
2022-01-20 08:00:00,618 P7046 INFO [Metrics] AUC: 0.898777 - logloss: 0.428422
2022-01-20 08:00:00,619 P7046 INFO Save best model: monitor(max): 0.898777
2022-01-20 08:00:00,624 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:00:00,709 P7046 INFO Train loss: 0.589549
2022-01-20 08:00:00,710 P7046 INFO ************ Epoch=1 end ************
2022-01-20 08:00:08,997 P7046 INFO [Metrics] AUC: 0.925763 - logloss: 0.318883
2022-01-20 08:00:08,997 P7046 INFO Save best model: monitor(max): 0.925763
2022-01-20 08:00:09,004 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:00:09,062 P7046 INFO Train loss: 0.384453
2022-01-20 08:00:09,062 P7046 INFO ************ Epoch=2 end ************
2022-01-20 08:00:17,250 P7046 INFO [Metrics] AUC: 0.933519 - logloss: 0.298981
2022-01-20 08:00:17,250 P7046 INFO Save best model: monitor(max): 0.933519
2022-01-20 08:00:17,257 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:00:17,347 P7046 INFO Train loss: 0.341959
2022-01-20 08:00:17,347 P7046 INFO ************ Epoch=3 end ************
2022-01-20 08:00:25,306 P7046 INFO [Metrics] AUC: 0.937210 - logloss: 0.290456
2022-01-20 08:00:25,307 P7046 INFO Save best model: monitor(max): 0.937210
2022-01-20 08:00:25,313 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:00:25,357 P7046 INFO Train loss: 0.326528
2022-01-20 08:00:25,358 P7046 INFO ************ Epoch=4 end ************
2022-01-20 08:00:33,565 P7046 INFO [Metrics] AUC: 0.939257 - logloss: 0.285173
2022-01-20 08:00:33,566 P7046 INFO Save best model: monitor(max): 0.939257
2022-01-20 08:00:33,572 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:00:33,623 P7046 INFO Train loss: 0.315068
2022-01-20 08:00:33,623 P7046 INFO ************ Epoch=5 end ************
2022-01-20 08:00:41,694 P7046 INFO [Metrics] AUC: 0.940673 - logloss: 0.281290
2022-01-20 08:00:41,695 P7046 INFO Save best model: monitor(max): 0.940673
2022-01-20 08:00:41,702 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:00:41,748 P7046 INFO Train loss: 0.305241
2022-01-20 08:00:41,748 P7046 INFO ************ Epoch=6 end ************
2022-01-20 08:00:49,885 P7046 INFO [Metrics] AUC: 0.941747 - logloss: 0.278285
2022-01-20 08:00:49,886 P7046 INFO Save best model: monitor(max): 0.941747
2022-01-20 08:00:49,892 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:00:49,949 P7046 INFO Train loss: 0.296460
2022-01-20 08:00:49,949 P7046 INFO ************ Epoch=7 end ************
2022-01-20 08:00:57,868 P7046 INFO [Metrics] AUC: 0.942719 - logloss: 0.275720
2022-01-20 08:00:57,869 P7046 INFO Save best model: monitor(max): 0.942719
2022-01-20 08:00:57,875 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:00:57,929 P7046 INFO Train loss: 0.288514
2022-01-20 08:00:57,929 P7046 INFO ************ Epoch=8 end ************
2022-01-20 08:01:05,821 P7046 INFO [Metrics] AUC: 0.943447 - logloss: 0.273686
2022-01-20 08:01:05,822 P7046 INFO Save best model: monitor(max): 0.943447
2022-01-20 08:01:05,828 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:01:05,880 P7046 INFO Train loss: 0.281310
2022-01-20 08:01:05,880 P7046 INFO ************ Epoch=9 end ************
2022-01-20 08:01:13,805 P7046 INFO [Metrics] AUC: 0.944110 - logloss: 0.272023
2022-01-20 08:01:13,806 P7046 INFO Save best model: monitor(max): 0.944110
2022-01-20 08:01:13,812 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:01:13,909 P7046 INFO Train loss: 0.274726
2022-01-20 08:01:13,910 P7046 INFO ************ Epoch=10 end ************
2022-01-20 08:01:21,814 P7046 INFO [Metrics] AUC: 0.944649 - logloss: 0.270814
2022-01-20 08:01:21,815 P7046 INFO Save best model: monitor(max): 0.944649
2022-01-20 08:01:21,821 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:01:21,873 P7046 INFO Train loss: 0.268648
2022-01-20 08:01:21,873 P7046 INFO ************ Epoch=11 end ************
2022-01-20 08:01:29,956 P7046 INFO [Metrics] AUC: 0.945085 - logloss: 0.269913
2022-01-20 08:01:29,957 P7046 INFO Save best model: monitor(max): 0.945085
2022-01-20 08:01:29,963 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:01:30,018 P7046 INFO Train loss: 0.262845
2022-01-20 08:01:30,018 P7046 INFO ************ Epoch=12 end ************
2022-01-20 08:01:38,073 P7046 INFO [Metrics] AUC: 0.945458 - logloss: 0.269315
2022-01-20 08:01:38,074 P7046 INFO Save best model: monitor(max): 0.945458
2022-01-20 08:01:38,080 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:01:38,143 P7046 INFO Train loss: 0.257165
2022-01-20 08:01:38,143 P7046 INFO ************ Epoch=13 end ************
2022-01-20 08:01:45,837 P7046 INFO [Metrics] AUC: 0.945794 - logloss: 0.268834
2022-01-20 08:01:45,838 P7046 INFO Save best model: monitor(max): 0.945794
2022-01-20 08:01:45,844 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:01:45,902 P7046 INFO Train loss: 0.251490
2022-01-20 08:01:45,903 P7046 INFO ************ Epoch=14 end ************
2022-01-20 08:01:53,864 P7046 INFO [Metrics] AUC: 0.946030 - logloss: 0.268748
2022-01-20 08:01:53,865 P7046 INFO Save best model: monitor(max): 0.946030
2022-01-20 08:01:53,872 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:01:53,930 P7046 INFO Train loss: 0.245718
2022-01-20 08:01:53,931 P7046 INFO ************ Epoch=15 end ************
2022-01-20 08:02:01,845 P7046 INFO [Metrics] AUC: 0.946190 - logloss: 0.268973
2022-01-20 08:02:01,846 P7046 INFO Save best model: monitor(max): 0.946190
2022-01-20 08:02:01,853 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:02:01,906 P7046 INFO Train loss: 0.239824
2022-01-20 08:02:01,906 P7046 INFO ************ Epoch=16 end ************
2022-01-20 08:02:08,029 P7046 INFO [Metrics] AUC: 0.946323 - logloss: 0.269343
2022-01-20 08:02:08,029 P7046 INFO Save best model: monitor(max): 0.946323
2022-01-20 08:02:08,035 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:02:08,092 P7046 INFO Train loss: 0.233813
2022-01-20 08:02:08,092 P7046 INFO ************ Epoch=17 end ************
2022-01-20 08:02:13,731 P7046 INFO [Metrics] AUC: 0.946421 - logloss: 0.269870
2022-01-20 08:02:13,732 P7046 INFO Save best model: monitor(max): 0.946421
2022-01-20 08:02:13,738 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:02:13,788 P7046 INFO Train loss: 0.227708
2022-01-20 08:02:13,788 P7046 INFO ************ Epoch=18 end ************
2022-01-20 08:02:19,566 P7046 INFO [Metrics] AUC: 0.946481 - logloss: 0.270697
2022-01-20 08:02:19,567 P7046 INFO Save best model: monitor(max): 0.946481
2022-01-20 08:02:19,573 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:02:19,642 P7046 INFO Train loss: 0.221513
2022-01-20 08:02:19,642 P7046 INFO ************ Epoch=19 end ************
2022-01-20 08:02:25,391 P7046 INFO [Metrics] AUC: 0.946527 - logloss: 0.271644
2022-01-20 08:02:25,392 P7046 INFO Save best model: monitor(max): 0.946527
2022-01-20 08:02:25,398 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:02:25,450 P7046 INFO Train loss: 0.215262
2022-01-20 08:02:25,450 P7046 INFO ************ Epoch=20 end ************
2022-01-20 08:02:31,307 P7046 INFO [Metrics] AUC: 0.946483 - logloss: 0.273164
2022-01-20 08:02:31,307 P7046 INFO Monitor(max) STOP: 0.946483 !
2022-01-20 08:02:31,307 P7046 INFO Reduce learning rate on plateau: 0.000100
2022-01-20 08:02:31,308 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:02:31,359 P7046 INFO Train loss: 0.209064
2022-01-20 08:02:31,360 P7046 INFO ************ Epoch=21 end ************
2022-01-20 08:02:37,357 P7046 INFO [Metrics] AUC: 0.946508 - logloss: 0.273259
2022-01-20 08:02:37,358 P7046 INFO Monitor(max) STOP: 0.946508 !
2022-01-20 08:02:37,358 P7046 INFO Reduce learning rate on plateau: 0.000010
2022-01-20 08:02:37,358 P7046 INFO Early stopping at epoch=22
2022-01-20 08:02:37,358 P7046 INFO --- 343/343 batches finished ---
2022-01-20 08:02:37,405 P7046 INFO Train loss: 0.196749
2022-01-20 08:02:37,405 P7046 INFO Training finished.
2022-01-20 08:02:37,405 P7046 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Movielens/FmFM_movielenslatest_x1/movielenslatest_x1_cd32d937/FmFM_movielenslatest_x1_003_28d4fd9f.model
2022-01-20 08:02:37,418 P7046 INFO ****** Validation evaluation ******
2022-01-20 08:02:38,760 P7046 INFO [Metrics] AUC: 0.946527 - logloss: 0.271644
2022-01-20 08:02:38,821 P7046 INFO ******** Test evaluation ********
2022-01-20 08:02:38,822 P7046 INFO Loading data...
2022-01-20 08:02:38,822 P7046 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-01-20 08:02:38,827 P7046 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-01-20 08:02:38,827 P7046 INFO Loading test data done.
2022-01-20 08:02:39,578 P7046 INFO [Metrics] AUC: 0.946525 - logloss: 0.271427

```
