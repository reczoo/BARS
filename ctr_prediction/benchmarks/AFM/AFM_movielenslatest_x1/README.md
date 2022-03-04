## AFM_movielenslatest_x1

A hands-on guide to run the AFM model on the Movielenslatest_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [AFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/AFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFM_movielenslatest_x1_tuner_config_04](./AFM_movielenslatest_x1_tuner_config_04). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFM_movielenslatest_x1
    nohup python run_expid.py --config ./AFM_movielenslatest_x1_tuner_config_04 --expid AFM_movielenslatest_x1_002_76325415 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.947193 | 0.265332  |


### Logs
```python
2022-01-28 12:15:35,707 P45055 INFO {
    "attention_dim": "16",
    "attention_dropout": "[0.4, 0.4]",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AFM",
    "model_id": "AFM_movielenslatest_x1_002_76325415",
    "model_root": "./Movielens/AFM_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
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
    "test_data": "../data/Movielens/MovielensLatest_x1/test.csv",
    "train_data": "../data/Movielens/MovielensLatest_x1/train.csv",
    "use_attention": "True",
    "use_hdf5": "True",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-28 12:15:35,708 P45055 INFO Set up feature encoder...
2022-01-28 12:15:35,708 P45055 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-01-28 12:15:35,709 P45055 INFO Loading data...
2022-01-28 12:15:35,712 P45055 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-01-28 12:15:35,742 P45055 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-01-28 12:15:35,751 P45055 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-01-28 12:15:35,752 P45055 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-01-28 12:15:35,752 P45055 INFO Loading train data done.
2022-01-28 12:15:39,193 P45055 INFO Total number of parameters: 992832.
2022-01-28 12:15:39,194 P45055 INFO Start training: 343 batches/epoch
2022-01-28 12:15:39,194 P45055 INFO ************ Epoch=1 start ************
2022-01-28 12:15:45,477 P45055 INFO [Metrics] AUC: 0.890836 - logloss: 0.539265
2022-01-28 12:15:45,478 P45055 INFO Save best model: monitor(max): 0.890836
2022-01-28 12:15:45,483 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:15:45,519 P45055 INFO Train loss: 0.628946
2022-01-28 12:15:45,519 P45055 INFO ************ Epoch=1 end ************
2022-01-28 12:15:51,819 P45055 INFO [Metrics] AUC: 0.913496 - logloss: 0.394555
2022-01-28 12:15:51,820 P45055 INFO Save best model: monitor(max): 0.913496
2022-01-28 12:15:51,826 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:15:51,861 P45055 INFO Train loss: 0.495984
2022-01-28 12:15:51,861 P45055 INFO ************ Epoch=2 end ************
2022-01-28 12:15:58,272 P45055 INFO [Metrics] AUC: 0.926631 - logloss: 0.349000
2022-01-28 12:15:58,273 P45055 INFO Save best model: monitor(max): 0.926631
2022-01-28 12:15:58,279 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:15:58,315 P45055 INFO Train loss: 0.443313
2022-01-28 12:15:58,315 P45055 INFO ************ Epoch=3 end ************
2022-01-28 12:16:04,776 P45055 INFO [Metrics] AUC: 0.932608 - logloss: 0.328083
2022-01-28 12:16:04,776 P45055 INFO Save best model: monitor(max): 0.932608
2022-01-28 12:16:04,782 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:16:04,830 P45055 INFO Train loss: 0.421501
2022-01-28 12:16:04,830 P45055 INFO ************ Epoch=4 end ************
2022-01-28 12:16:10,893 P45055 INFO [Metrics] AUC: 0.936469 - logloss: 0.315550
2022-01-28 12:16:10,893 P45055 INFO Save best model: monitor(max): 0.936469
2022-01-28 12:16:10,899 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:16:10,937 P45055 INFO Train loss: 0.407147
2022-01-28 12:16:10,937 P45055 INFO ************ Epoch=5 end ************
2022-01-28 12:16:17,070 P45055 INFO [Metrics] AUC: 0.939044 - logloss: 0.306961
2022-01-28 12:16:17,070 P45055 INFO Save best model: monitor(max): 0.939044
2022-01-28 12:16:17,076 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:16:17,113 P45055 INFO Train loss: 0.395218
2022-01-28 12:16:17,114 P45055 INFO ************ Epoch=6 end ************
2022-01-28 12:16:23,285 P45055 INFO [Metrics] AUC: 0.941246 - logloss: 0.299797
2022-01-28 12:16:23,285 P45055 INFO Save best model: monitor(max): 0.941246
2022-01-28 12:16:23,292 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:16:23,330 P45055 INFO Train loss: 0.384532
2022-01-28 12:16:23,330 P45055 INFO ************ Epoch=7 end ************
2022-01-28 12:16:29,400 P45055 INFO [Metrics] AUC: 0.942520 - logloss: 0.292888
2022-01-28 12:16:29,400 P45055 INFO Save best model: monitor(max): 0.942520
2022-01-28 12:16:29,406 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:16:29,442 P45055 INFO Train loss: 0.375629
2022-01-28 12:16:29,442 P45055 INFO ************ Epoch=8 end ************
2022-01-28 12:16:35,619 P45055 INFO [Metrics] AUC: 0.943601 - logloss: 0.287936
2022-01-28 12:16:35,619 P45055 INFO Save best model: monitor(max): 0.943601
2022-01-28 12:16:35,625 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:16:35,669 P45055 INFO Train loss: 0.368083
2022-01-28 12:16:35,669 P45055 INFO ************ Epoch=9 end ************
2022-01-28 12:16:41,754 P45055 INFO [Metrics] AUC: 0.944448 - logloss: 0.283287
2022-01-28 12:16:41,755 P45055 INFO Save best model: monitor(max): 0.944448
2022-01-28 12:16:41,760 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:16:41,795 P45055 INFO Train loss: 0.361246
2022-01-28 12:16:41,796 P45055 INFO ************ Epoch=10 end ************
2022-01-28 12:16:47,874 P45055 INFO [Metrics] AUC: 0.945206 - logloss: 0.279353
2022-01-28 12:16:47,874 P45055 INFO Save best model: monitor(max): 0.945206
2022-01-28 12:16:47,881 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:16:47,917 P45055 INFO Train loss: 0.355226
2022-01-28 12:16:47,917 P45055 INFO ************ Epoch=11 end ************
2022-01-28 12:16:53,931 P45055 INFO [Metrics] AUC: 0.945768 - logloss: 0.276587
2022-01-28 12:16:53,932 P45055 INFO Save best model: monitor(max): 0.945768
2022-01-28 12:16:53,938 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:16:53,973 P45055 INFO Train loss: 0.350322
2022-01-28 12:16:53,973 P45055 INFO ************ Epoch=12 end ************
2022-01-28 12:17:00,120 P45055 INFO [Metrics] AUC: 0.946231 - logloss: 0.273834
2022-01-28 12:17:00,121 P45055 INFO Save best model: monitor(max): 0.946231
2022-01-28 12:17:00,127 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:17:00,163 P45055 INFO Train loss: 0.345790
2022-01-28 12:17:00,163 P45055 INFO ************ Epoch=13 end ************
2022-01-28 12:17:06,480 P45055 INFO [Metrics] AUC: 0.946688 - logloss: 0.271581
2022-01-28 12:17:06,481 P45055 INFO Save best model: monitor(max): 0.946688
2022-01-28 12:17:06,489 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:17:06,526 P45055 INFO Train loss: 0.341648
2022-01-28 12:17:06,526 P45055 INFO ************ Epoch=14 end ************
2022-01-28 12:17:13,039 P45055 INFO [Metrics] AUC: 0.946951 - logloss: 0.269724
2022-01-28 12:17:13,040 P45055 INFO Save best model: monitor(max): 0.946951
2022-01-28 12:17:13,046 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:17:13,084 P45055 INFO Train loss: 0.337486
2022-01-28 12:17:13,085 P45055 INFO ************ Epoch=15 end ************
2022-01-28 12:17:19,668 P45055 INFO [Metrics] AUC: 0.947121 - logloss: 0.268290
2022-01-28 12:17:19,669 P45055 INFO Save best model: monitor(max): 0.947121
2022-01-28 12:17:19,675 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:17:19,716 P45055 INFO Train loss: 0.334312
2022-01-28 12:17:19,716 P45055 INFO ************ Epoch=16 end ************
2022-01-28 12:17:26,020 P45055 INFO [Metrics] AUC: 0.947124 - logloss: 0.267507
2022-01-28 12:17:26,021 P45055 INFO Save best model: monitor(max): 0.947124
2022-01-28 12:17:26,027 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:17:26,069 P45055 INFO Train loss: 0.330976
2022-01-28 12:17:26,069 P45055 INFO ************ Epoch=17 end ************
2022-01-28 12:17:32,239 P45055 INFO [Metrics] AUC: 0.947267 - logloss: 0.266114
2022-01-28 12:17:32,240 P45055 INFO Save best model: monitor(max): 0.947267
2022-01-28 12:17:32,247 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:17:32,301 P45055 INFO Train loss: 0.328412
2022-01-28 12:17:32,301 P45055 INFO ************ Epoch=18 end ************
2022-01-28 12:17:38,164 P45055 INFO [Metrics] AUC: 0.947478 - logloss: 0.265064
2022-01-28 12:17:38,164 P45055 INFO Save best model: monitor(max): 0.947478
2022-01-28 12:17:38,171 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:17:38,208 P45055 INFO Train loss: 0.326103
2022-01-28 12:17:38,208 P45055 INFO ************ Epoch=19 end ************
2022-01-28 12:17:44,097 P45055 INFO [Metrics] AUC: 0.947284 - logloss: 0.264781
2022-01-28 12:17:44,098 P45055 INFO Monitor(max) STOP: 0.947284 !
2022-01-28 12:17:44,098 P45055 INFO Reduce learning rate on plateau: 0.000100
2022-01-28 12:17:44,098 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:17:44,133 P45055 INFO Train loss: 0.323068
2022-01-28 12:17:44,133 P45055 INFO ************ Epoch=20 end ************
2022-01-28 12:17:49,910 P45055 INFO [Metrics] AUC: 0.947449 - logloss: 0.264200
2022-01-28 12:17:49,910 P45055 INFO Monitor(max) STOP: 0.947449 !
2022-01-28 12:17:49,911 P45055 INFO Reduce learning rate on plateau: 0.000010
2022-01-28 12:17:49,911 P45055 INFO Early stopping at epoch=21
2022-01-28 12:17:49,911 P45055 INFO --- 343/343 batches finished ---
2022-01-28 12:17:49,951 P45055 INFO Train loss: 0.317658
2022-01-28 12:17:49,952 P45055 INFO Training finished.
2022-01-28 12:17:49,952 P45055 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Movielens/AFM_movielenslatest_x1/movielenslatest_x1_cd32d937/AFM_movielenslatest_x1_002_76325415.model
2022-01-28 12:17:53,422 P45055 INFO ****** Validation evaluation ******
2022-01-28 12:17:54,756 P45055 INFO [Metrics] AUC: 0.947478 - logloss: 0.265064
2022-01-28 12:17:54,809 P45055 INFO ******** Test evaluation ********
2022-01-28 12:17:54,810 P45055 INFO Loading data...
2022-01-28 12:17:54,810 P45055 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-01-28 12:17:54,814 P45055 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-01-28 12:17:54,814 P45055 INFO Loading test data done.
2022-01-28 12:17:55,518 P45055 INFO [Metrics] AUC: 0.947193 - logloss: 0.265332

```
