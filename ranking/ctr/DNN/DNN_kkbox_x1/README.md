## DNN_kkbox_x1

A hands-on guide to run the DNN model on the KKBox_x1 dataset.

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
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [KKBox_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/KKBox#KKBox_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [DNN](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DNN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DNN_kkbox_x1_tuner_config_02](./DNN_kkbox_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DNN_kkbox_x1
    nohup python run_expid.py --config ./DNN_kkbox_x1_tuner_config_02 --expid DNN_kkbox_x1_021_6036677c --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.481129 | 0.850076  |


### Logs
```python
2022-03-09 21:13:58,669 P9352 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.0005",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "3",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000, 1000]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "DNN",
    "model_id": "DNN_kkbox_x1_021_6036677c",
    "model_root": "./KKBox/DNN_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.2",
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
2022-03-09 21:13:58,670 P9352 INFO Set up feature encoder...
2022-03-09 21:13:58,670 P9352 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-09 21:13:59,038 P9352 INFO Total number of parameters: 16476617.
2022-03-09 21:13:59,039 P9352 INFO Loading data...
2022-03-09 21:13:59,039 P9352 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-09 21:13:59,383 P9352 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-09 21:13:59,623 P9352 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-09 21:13:59,640 P9352 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-09 21:13:59,640 P9352 INFO Loading train data done.
2022-03-09 21:14:02,622 P9352 INFO Start training: 591 batches/epoch
2022-03-09 21:14:02,623 P9352 INFO ************ Epoch=1 start ************
2022-03-09 21:14:45,647 P9352 INFO [Metrics] logloss: 0.554110 - AUC: 0.787705
2022-03-09 21:14:45,648 P9352 INFO Save best model: monitor(max): 0.233595
2022-03-09 21:14:45,874 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:14:45,909 P9352 INFO Train loss: 0.610670
2022-03-09 21:14:45,910 P9352 INFO ************ Epoch=1 end ************
2022-03-09 21:15:27,817 P9352 INFO [Metrics] logloss: 0.540737 - AUC: 0.800516
2022-03-09 21:15:27,818 P9352 INFO Save best model: monitor(max): 0.259779
2022-03-09 21:15:27,900 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:15:27,939 P9352 INFO Train loss: 0.592543
2022-03-09 21:15:27,939 P9352 INFO ************ Epoch=2 end ************
2022-03-09 21:16:10,247 P9352 INFO [Metrics] logloss: 0.531939 - AUC: 0.808282
2022-03-09 21:16:10,251 P9352 INFO Save best model: monitor(max): 0.276343
2022-03-09 21:16:10,335 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:16:10,390 P9352 INFO Train loss: 0.584989
2022-03-09 21:16:10,390 P9352 INFO ************ Epoch=3 end ************
2022-03-09 21:16:52,551 P9352 INFO [Metrics] logloss: 0.527654 - AUC: 0.812453
2022-03-09 21:16:52,551 P9352 INFO Save best model: monitor(max): 0.284799
2022-03-09 21:16:52,630 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:16:52,667 P9352 INFO Train loss: 0.580681
2022-03-09 21:16:52,667 P9352 INFO ************ Epoch=4 end ************
2022-03-09 21:17:35,205 P9352 INFO [Metrics] logloss: 0.523035 - AUC: 0.816011
2022-03-09 21:17:35,208 P9352 INFO Save best model: monitor(max): 0.292976
2022-03-09 21:17:35,299 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:17:35,339 P9352 INFO Train loss: 0.578239
2022-03-09 21:17:35,339 P9352 INFO ************ Epoch=5 end ************
2022-03-09 21:18:18,277 P9352 INFO [Metrics] logloss: 0.520294 - AUC: 0.818043
2022-03-09 21:18:18,278 P9352 INFO Save best model: monitor(max): 0.297749
2022-03-09 21:18:18,367 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:18:18,410 P9352 INFO Train loss: 0.576257
2022-03-09 21:18:18,411 P9352 INFO ************ Epoch=6 end ************
2022-03-09 21:19:00,319 P9352 INFO [Metrics] logloss: 0.517870 - AUC: 0.820221
2022-03-09 21:19:00,322 P9352 INFO Save best model: monitor(max): 0.302350
2022-03-09 21:19:00,417 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:19:00,455 P9352 INFO Train loss: 0.574479
2022-03-09 21:19:00,456 P9352 INFO ************ Epoch=7 end ************
2022-03-09 21:19:42,769 P9352 INFO [Metrics] logloss: 0.515275 - AUC: 0.822254
2022-03-09 21:19:42,771 P9352 INFO Save best model: monitor(max): 0.306979
2022-03-09 21:19:42,854 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:19:42,900 P9352 INFO Train loss: 0.573473
2022-03-09 21:19:42,900 P9352 INFO ************ Epoch=8 end ************
2022-03-09 21:20:24,795 P9352 INFO [Metrics] logloss: 0.514120 - AUC: 0.823252
2022-03-09 21:20:24,798 P9352 INFO Save best model: monitor(max): 0.309133
2022-03-09 21:20:24,869 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:20:24,910 P9352 INFO Train loss: 0.572140
2022-03-09 21:20:24,910 P9352 INFO ************ Epoch=9 end ************
2022-03-09 21:21:06,767 P9352 INFO [Metrics] logloss: 0.512701 - AUC: 0.824387
2022-03-09 21:21:06,769 P9352 INFO Save best model: monitor(max): 0.311686
2022-03-09 21:21:06,865 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:21:06,902 P9352 INFO Train loss: 0.571720
2022-03-09 21:21:06,902 P9352 INFO ************ Epoch=10 end ************
2022-03-09 21:21:50,266 P9352 INFO [Metrics] logloss: 0.511230 - AUC: 0.825489
2022-03-09 21:21:50,269 P9352 INFO Save best model: monitor(max): 0.314259
2022-03-09 21:21:50,361 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:21:50,405 P9352 INFO Train loss: 0.570561
2022-03-09 21:21:50,405 P9352 INFO ************ Epoch=11 end ************
2022-03-09 21:22:33,126 P9352 INFO [Metrics] logloss: 0.509973 - AUC: 0.826476
2022-03-09 21:22:33,127 P9352 INFO Save best model: monitor(max): 0.316503
2022-03-09 21:22:33,228 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:22:33,271 P9352 INFO Train loss: 0.569834
2022-03-09 21:22:33,271 P9352 INFO ************ Epoch=12 end ************
2022-03-09 21:23:15,769 P9352 INFO [Metrics] logloss: 0.509024 - AUC: 0.827189
2022-03-09 21:23:15,770 P9352 INFO Save best model: monitor(max): 0.318165
2022-03-09 21:23:15,861 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:23:15,906 P9352 INFO Train loss: 0.568976
2022-03-09 21:23:15,906 P9352 INFO ************ Epoch=13 end ************
2022-03-09 21:23:58,656 P9352 INFO [Metrics] logloss: 0.507701 - AUC: 0.828261
2022-03-09 21:23:58,657 P9352 INFO Save best model: monitor(max): 0.320560
2022-03-09 21:23:58,743 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:23:58,785 P9352 INFO Train loss: 0.568350
2022-03-09 21:23:58,785 P9352 INFO ************ Epoch=14 end ************
2022-03-09 21:24:41,501 P9352 INFO [Metrics] logloss: 0.507031 - AUC: 0.829109
2022-03-09 21:24:41,502 P9352 INFO Save best model: monitor(max): 0.322078
2022-03-09 21:24:41,581 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:24:41,622 P9352 INFO Train loss: 0.567613
2022-03-09 21:24:41,622 P9352 INFO ************ Epoch=15 end ************
2022-03-09 21:25:24,414 P9352 INFO [Metrics] logloss: 0.507106 - AUC: 0.829242
2022-03-09 21:25:24,415 P9352 INFO Save best model: monitor(max): 0.322136
2022-03-09 21:25:24,496 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:25:24,537 P9352 INFO Train loss: 0.567043
2022-03-09 21:25:24,538 P9352 INFO ************ Epoch=16 end ************
2022-03-09 21:26:07,035 P9352 INFO [Metrics] logloss: 0.505521 - AUC: 0.830014
2022-03-09 21:26:07,036 P9352 INFO Save best model: monitor(max): 0.324493
2022-03-09 21:26:07,113 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:26:07,156 P9352 INFO Train loss: 0.566601
2022-03-09 21:26:07,156 P9352 INFO ************ Epoch=17 end ************
2022-03-09 21:26:50,174 P9352 INFO [Metrics] logloss: 0.505098 - AUC: 0.830376
2022-03-09 21:26:50,175 P9352 INFO Save best model: monitor(max): 0.325278
2022-03-09 21:26:50,251 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:26:50,309 P9352 INFO Train loss: 0.566382
2022-03-09 21:26:50,309 P9352 INFO ************ Epoch=18 end ************
2022-03-09 21:27:33,434 P9352 INFO [Metrics] logloss: 0.504427 - AUC: 0.830829
2022-03-09 21:27:33,435 P9352 INFO Save best model: monitor(max): 0.326401
2022-03-09 21:27:33,512 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:27:33,554 P9352 INFO Train loss: 0.565928
2022-03-09 21:27:33,554 P9352 INFO ************ Epoch=19 end ************
2022-03-09 21:28:16,249 P9352 INFO [Metrics] logloss: 0.503918 - AUC: 0.831403
2022-03-09 21:28:16,250 P9352 INFO Save best model: monitor(max): 0.327485
2022-03-09 21:28:16,324 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:28:16,364 P9352 INFO Train loss: 0.565174
2022-03-09 21:28:16,364 P9352 INFO ************ Epoch=20 end ************
2022-03-09 21:28:58,431 P9352 INFO [Metrics] logloss: 0.502949 - AUC: 0.831996
2022-03-09 21:28:58,432 P9352 INFO Save best model: monitor(max): 0.329046
2022-03-09 21:28:58,511 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:28:58,551 P9352 INFO Train loss: 0.564924
2022-03-09 21:28:58,551 P9352 INFO ************ Epoch=21 end ************
2022-03-09 21:29:40,427 P9352 INFO [Metrics] logloss: 0.502703 - AUC: 0.832232
2022-03-09 21:29:40,430 P9352 INFO Save best model: monitor(max): 0.329529
2022-03-09 21:29:40,513 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:29:40,552 P9352 INFO Train loss: 0.564572
2022-03-09 21:29:40,552 P9352 INFO ************ Epoch=22 end ************
2022-03-09 21:30:22,680 P9352 INFO [Metrics] logloss: 0.503069 - AUC: 0.832342
2022-03-09 21:30:22,680 P9352 INFO Monitor(max) STOP: 0.329273 !
2022-03-09 21:30:22,680 P9352 INFO Reduce learning rate on plateau: 0.000100
2022-03-09 21:30:22,680 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:30:22,717 P9352 INFO Train loss: 0.564093
2022-03-09 21:30:22,718 P9352 INFO ************ Epoch=23 end ************
2022-03-09 21:31:04,372 P9352 INFO [Metrics] logloss: 0.486397 - AUC: 0.845092
2022-03-09 21:31:04,373 P9352 INFO Save best model: monitor(max): 0.358695
2022-03-09 21:31:04,451 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:31:04,489 P9352 INFO Train loss: 0.511084
2022-03-09 21:31:04,489 P9352 INFO ************ Epoch=24 end ************
2022-03-09 21:31:47,457 P9352 INFO [Metrics] logloss: 0.482885 - AUC: 0.848071
2022-03-09 21:31:47,458 P9352 INFO Save best model: monitor(max): 0.365185
2022-03-09 21:31:47,539 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:31:47,598 P9352 INFO Train loss: 0.485379
2022-03-09 21:31:47,598 P9352 INFO ************ Epoch=25 end ************
2022-03-09 21:32:29,498 P9352 INFO [Metrics] logloss: 0.481931 - AUC: 0.849171
2022-03-09 21:32:29,499 P9352 INFO Save best model: monitor(max): 0.367240
2022-03-09 21:32:29,576 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:32:29,636 P9352 INFO Train loss: 0.474851
2022-03-09 21:32:29,636 P9352 INFO ************ Epoch=26 end ************
2022-03-09 21:33:11,550 P9352 INFO [Metrics] logloss: 0.481624 - AUC: 0.849752
2022-03-09 21:33:11,551 P9352 INFO Save best model: monitor(max): 0.368128
2022-03-09 21:33:11,633 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:33:11,672 P9352 INFO Train loss: 0.467707
2022-03-09 21:33:11,672 P9352 INFO ************ Epoch=27 end ************
2022-03-09 21:33:53,870 P9352 INFO [Metrics] logloss: 0.482753 - AUC: 0.849747
2022-03-09 21:33:53,871 P9352 INFO Monitor(max) STOP: 0.366994 !
2022-03-09 21:33:53,871 P9352 INFO Reduce learning rate on plateau: 0.000010
2022-03-09 21:33:53,871 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:33:53,919 P9352 INFO Train loss: 0.462013
2022-03-09 21:33:53,919 P9352 INFO ************ Epoch=28 end ************
2022-03-09 21:34:36,047 P9352 INFO [Metrics] logloss: 0.489857 - AUC: 0.849128
2022-03-09 21:34:36,048 P9352 INFO Monitor(max) STOP: 0.359271 !
2022-03-09 21:34:36,048 P9352 INFO Reduce learning rate on plateau: 0.000001
2022-03-09 21:34:36,048 P9352 INFO Early stopping at epoch=29
2022-03-09 21:34:36,048 P9352 INFO --- 591/591 batches finished ---
2022-03-09 21:34:36,086 P9352 INFO Train loss: 0.440746
2022-03-09 21:34:36,086 P9352 INFO Training finished.
2022-03-09 21:34:36,086 P9352 INFO Load best model: /cache/FuxiCTR/benchmarks/KKBox/DNN_kkbox_x1/kkbox_x1_227d337d/DNN_kkbox_x1_021_6036677c_model.ckpt
2022-03-09 21:34:36,158 P9352 INFO ****** Validation evaluation ******
2022-03-09 21:34:40,478 P9352 INFO [Metrics] logloss: 0.481624 - AUC: 0.849752
2022-03-09 21:34:40,535 P9352 INFO ******** Test evaluation ********
2022-03-09 21:34:40,536 P9352 INFO Loading data...
2022-03-09 21:34:40,536 P9352 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-09 21:34:40,599 P9352 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-09 21:34:40,599 P9352 INFO Loading test data done.
2022-03-09 21:34:44,839 P9352 INFO [Metrics] logloss: 0.481129 - AUC: 0.850076

```
