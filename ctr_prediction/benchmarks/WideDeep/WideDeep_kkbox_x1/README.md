## WideDeep_kkbox_x1

A hands-on guide to run the WideDeep model on the KKBox_x1 dataset.

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
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [KKBox_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/KKBox#KKBox_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [WideDeep](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/WideDeep.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [WideDeep_kkbox_x1_tuner_config_03](./WideDeep_kkbox_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd WideDeep_kkbox_x1
    nohup python run_expid.py --config ./WideDeep_kkbox_x1_tuner_config_03 --expid WideDeep_kkbox_x1_012_62151ded --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.485182 | 0.850420  |


### Logs
```python
2022-03-07 22:57:45,285 P29952 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.0001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "3",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "WideDeep",
    "model_id": "WideDeep_kkbox_x1_012_62151ded",
    "model_root": "./KKBox/WideDeep_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/KKBox/KKBox_x1/test.csv",
    "train_data": "../data/KKBox/KKBox_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/KKBox/KKBox_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2022-03-07 22:57:45,286 P29952 INFO Set up feature encoder...
2022-03-07 22:57:45,286 P29952 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-07 22:57:46,828 P29952 INFO Total number of parameters: 15567864.
2022-03-07 22:57:46,828 P29952 INFO Loading data...
2022-03-07 22:57:46,828 P29952 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-07 22:57:47,282 P29952 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-07 22:57:47,569 P29952 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-07 22:57:47,600 P29952 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-07 22:57:47,601 P29952 INFO Loading train data done.
2022-03-07 22:57:53,525 P29952 INFO Start training: 591 batches/epoch
2022-03-07 22:57:53,525 P29952 INFO ************ Epoch=1 start ************
2022-03-07 23:00:28,803 P29952 INFO [Metrics] logloss: 0.547858 - AUC: 0.793893
2022-03-07 23:00:28,807 P29952 INFO Save best model: monitor(max): 0.246035
2022-03-07 23:00:29,412 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:00:29,445 P29952 INFO Train loss: 0.589426
2022-03-07 23:00:29,445 P29952 INFO ************ Epoch=1 end ************
2022-03-07 23:03:04,376 P29952 INFO [Metrics] logloss: 0.529284 - AUC: 0.810919
2022-03-07 23:03:04,377 P29952 INFO Save best model: monitor(max): 0.281635
2022-03-07 23:03:04,473 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:03:04,509 P29952 INFO Train loss: 0.569581
2022-03-07 23:03:04,509 P29952 INFO ************ Epoch=2 end ************
2022-03-07 23:05:39,451 P29952 INFO [Metrics] logloss: 0.518594 - AUC: 0.819454
2022-03-07 23:05:39,452 P29952 INFO Save best model: monitor(max): 0.300860
2022-03-07 23:05:39,513 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:05:39,547 P29952 INFO Train loss: 0.560866
2022-03-07 23:05:39,547 P29952 INFO ************ Epoch=3 end ************
2022-03-07 23:08:13,936 P29952 INFO [Metrics] logloss: 0.511849 - AUC: 0.825003
2022-03-07 23:08:13,937 P29952 INFO Save best model: monitor(max): 0.313154
2022-03-07 23:08:14,006 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:08:14,044 P29952 INFO Train loss: 0.555288
2022-03-07 23:08:14,044 P29952 INFO ************ Epoch=4 end ************
2022-03-07 23:10:48,834 P29952 INFO [Metrics] logloss: 0.506361 - AUC: 0.829291
2022-03-07 23:10:48,835 P29952 INFO Save best model: monitor(max): 0.322929
2022-03-07 23:10:48,898 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:10:48,941 P29952 INFO Train loss: 0.551036
2022-03-07 23:10:48,941 P29952 INFO ************ Epoch=5 end ************
2022-03-07 23:13:23,185 P29952 INFO [Metrics] logloss: 0.503423 - AUC: 0.831756
2022-03-07 23:13:23,186 P29952 INFO Save best model: monitor(max): 0.328334
2022-03-07 23:13:23,265 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:13:23,300 P29952 INFO Train loss: 0.547738
2022-03-07 23:13:23,300 P29952 INFO ************ Epoch=6 end ************
2022-03-07 23:15:57,284 P29952 INFO [Metrics] logloss: 0.500505 - AUC: 0.833938
2022-03-07 23:15:57,285 P29952 INFO Save best model: monitor(max): 0.333433
2022-03-07 23:15:57,352 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:15:57,387 P29952 INFO Train loss: 0.545065
2022-03-07 23:15:57,387 P29952 INFO ************ Epoch=7 end ************
2022-03-07 23:17:47,852 P29952 INFO [Metrics] logloss: 0.498809 - AUC: 0.835261
2022-03-07 23:17:47,853 P29952 INFO Save best model: monitor(max): 0.336453
2022-03-07 23:17:47,935 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:17:47,993 P29952 INFO Train loss: 0.543019
2022-03-07 23:17:47,993 P29952 INFO ************ Epoch=8 end ************
2022-03-07 23:19:37,881 P29952 INFO [Metrics] logloss: 0.496508 - AUC: 0.836867
2022-03-07 23:19:37,882 P29952 INFO Save best model: monitor(max): 0.340359
2022-03-07 23:19:37,956 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:19:37,997 P29952 INFO Train loss: 0.540965
2022-03-07 23:19:37,997 P29952 INFO ************ Epoch=9 end ************
2022-03-07 23:21:27,463 P29952 INFO [Metrics] logloss: 0.494447 - AUC: 0.838679
2022-03-07 23:21:27,463 P29952 INFO Save best model: monitor(max): 0.344232
2022-03-07 23:21:27,525 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:21:27,565 P29952 INFO Train loss: 0.539226
2022-03-07 23:21:27,565 P29952 INFO ************ Epoch=10 end ************
2022-03-07 23:23:16,979 P29952 INFO [Metrics] logloss: 0.493059 - AUC: 0.839580
2022-03-07 23:23:16,980 P29952 INFO Save best model: monitor(max): 0.346521
2022-03-07 23:23:17,063 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:23:17,099 P29952 INFO Train loss: 0.537671
2022-03-07 23:23:17,099 P29952 INFO ************ Epoch=11 end ************
2022-03-07 23:25:06,771 P29952 INFO [Metrics] logloss: 0.492670 - AUC: 0.839960
2022-03-07 23:25:06,771 P29952 INFO Save best model: monitor(max): 0.347290
2022-03-07 23:25:06,852 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:25:06,889 P29952 INFO Train loss: 0.536484
2022-03-07 23:25:06,889 P29952 INFO ************ Epoch=12 end ************
2022-03-07 23:26:56,660 P29952 INFO [Metrics] logloss: 0.491521 - AUC: 0.840820
2022-03-07 23:26:56,661 P29952 INFO Save best model: monitor(max): 0.349298
2022-03-07 23:26:56,735 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:26:56,772 P29952 INFO Train loss: 0.535382
2022-03-07 23:26:56,772 P29952 INFO ************ Epoch=13 end ************
2022-03-07 23:28:46,307 P29952 INFO [Metrics] logloss: 0.490420 - AUC: 0.841699
2022-03-07 23:28:46,308 P29952 INFO Save best model: monitor(max): 0.351279
2022-03-07 23:28:46,387 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:28:46,424 P29952 INFO Train loss: 0.534451
2022-03-07 23:28:46,424 P29952 INFO ************ Epoch=14 end ************
2022-03-07 23:30:35,812 P29952 INFO [Metrics] logloss: 0.489867 - AUC: 0.841942
2022-03-07 23:30:35,812 P29952 INFO Save best model: monitor(max): 0.352075
2022-03-07 23:30:35,876 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:30:35,924 P29952 INFO Train loss: 0.533636
2022-03-07 23:30:35,924 P29952 INFO ************ Epoch=15 end ************
2022-03-07 23:32:24,520 P29952 INFO [Metrics] logloss: 0.489232 - AUC: 0.842708
2022-03-07 23:32:24,521 P29952 INFO Save best model: monitor(max): 0.353476
2022-03-07 23:32:24,599 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:32:24,633 P29952 INFO Train loss: 0.532719
2022-03-07 23:32:24,633 P29952 INFO ************ Epoch=16 end ************
2022-03-07 23:34:13,068 P29952 INFO [Metrics] logloss: 0.489625 - AUC: 0.842418
2022-03-07 23:34:13,069 P29952 INFO Monitor(max) STOP: 0.352793 !
2022-03-07 23:34:13,069 P29952 INFO Reduce learning rate on plateau: 0.000100
2022-03-07 23:34:13,069 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:34:13,112 P29952 INFO Train loss: 0.531996
2022-03-07 23:34:13,112 P29952 INFO ************ Epoch=17 end ************
2022-03-07 23:36:01,124 P29952 INFO [Metrics] logloss: 0.484815 - AUC: 0.849254
2022-03-07 23:36:01,125 P29952 INFO Save best model: monitor(max): 0.364439
2022-03-07 23:36:01,197 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:36:01,232 P29952 INFO Train loss: 0.480497
2022-03-07 23:36:01,232 P29952 INFO ************ Epoch=18 end ************
2022-03-07 23:37:49,033 P29952 INFO [Metrics] logloss: 0.485038 - AUC: 0.850526
2022-03-07 23:37:49,034 P29952 INFO Save best model: monitor(max): 0.365489
2022-03-07 23:37:49,113 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:37:49,147 P29952 INFO Train loss: 0.454754
2022-03-07 23:37:49,148 P29952 INFO ************ Epoch=19 end ************
2022-03-07 23:39:36,396 P29952 INFO [Metrics] logloss: 0.489586 - AUC: 0.849931
2022-03-07 23:39:36,397 P29952 INFO Monitor(max) STOP: 0.360344 !
2022-03-07 23:39:36,397 P29952 INFO Reduce learning rate on plateau: 0.000010
2022-03-07 23:39:36,397 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:39:36,432 P29952 INFO Train loss: 0.440003
2022-03-07 23:39:36,432 P29952 INFO ************ Epoch=20 end ************
2022-03-07 23:41:23,547 P29952 INFO [Metrics] logloss: 0.500982 - AUC: 0.849019
2022-03-07 23:41:23,548 P29952 INFO Monitor(max) STOP: 0.348037 !
2022-03-07 23:41:23,548 P29952 INFO Reduce learning rate on plateau: 0.000001
2022-03-07 23:41:23,548 P29952 INFO Early stopping at epoch=21
2022-03-07 23:41:23,548 P29952 INFO --- 591/591 batches finished ---
2022-03-07 23:41:23,582 P29952 INFO Train loss: 0.418803
2022-03-07 23:41:23,582 P29952 INFO Training finished.
2022-03-07 23:41:23,582 P29952 INFO Load best model: /cache/FuxiCTR/benchmarks/KKBox/WideDeep_kkbox_x1/kkbox_x1_227d337d/WideDeep_kkbox_x1_012_62151ded_model.ckpt
2022-03-07 23:41:23,709 P29952 INFO ****** Validation evaluation ******
2022-03-07 23:41:27,788 P29952 INFO [Metrics] logloss: 0.485038 - AUC: 0.850526
2022-03-07 23:41:27,858 P29952 INFO ******** Test evaluation ********
2022-03-07 23:41:27,858 P29952 INFO Loading data...
2022-03-07 23:41:27,858 P29952 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-07 23:41:27,928 P29952 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-07 23:41:27,929 P29952 INFO Loading test data done.
2022-03-07 23:41:31,955 P29952 INFO [Metrics] logloss: 0.485182 - AUC: 0.850420

```
