## FM_kkbox_x1

A hands-on guide to run the FM model on the KKBox_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
  GPU: Tesla P100 16G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 10.0
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [KKBox_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/KKBox#KKBox_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FM](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FM_kkbox_x1_tuner_config_01](./FM_kkbox_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FM_kkbox_x1
    nohup python run_expid.py --config ./FM_kkbox_x1_tuner_config_01 --expid FM_kkbox_x1_002_c55e9f15 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.506000 | 0.830423  |


### Logs
```python
2022-03-08 10:13:15,506 P39894 INFO {
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "FM",
    "model_id": "FM_kkbox_x1_002_c55e9f15",
    "model_root": "./KKBox/FM_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-05",
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
2022-03-08 10:13:15,507 P39894 INFO Set up feature encoder...
2022-03-08 10:13:15,507 P39894 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-08 10:13:15,817 P39894 INFO Total number of parameters: 11899864.
2022-03-08 10:13:15,817 P39894 INFO Loading data...
2022-03-08 10:13:15,819 P39894 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-08 10:13:16,224 P39894 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-08 10:13:16,426 P39894 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-08 10:13:16,446 P39894 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-08 10:13:16,446 P39894 INFO Loading train data done.
2022-03-08 10:13:19,336 P39894 INFO Start training: 591 batches/epoch
2022-03-08 10:13:19,336 P39894 INFO ************ Epoch=1 start ************
2022-03-08 10:15:43,742 P39894 INFO [Metrics] logloss: 0.561696 - AUC: 0.781554
2022-03-08 10:15:43,743 P39894 INFO Save best model: monitor(max): 0.219858
2022-03-08 10:15:43,788 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:15:43,820 P39894 INFO Train loss: 0.597372
2022-03-08 10:15:43,820 P39894 INFO ************ Epoch=1 end ************
2022-03-08 10:18:07,905 P39894 INFO [Metrics] logloss: 0.547048 - AUC: 0.796007
2022-03-08 10:18:07,906 P39894 INFO Save best model: monitor(max): 0.248959
2022-03-08 10:18:07,981 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:18:08,014 P39894 INFO Train loss: 0.570620
2022-03-08 10:18:08,015 P39894 INFO ************ Epoch=2 end ************
2022-03-08 10:20:31,040 P39894 INFO [Metrics] logloss: 0.537536 - AUC: 0.804478
2022-03-08 10:20:31,041 P39894 INFO Save best model: monitor(max): 0.266942
2022-03-08 10:20:31,116 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:20:31,149 P39894 INFO Train loss: 0.559786
2022-03-08 10:20:31,149 P39894 INFO ************ Epoch=3 end ************
2022-03-08 10:22:53,333 P39894 INFO [Metrics] logloss: 0.532078 - AUC: 0.808786
2022-03-08 10:22:53,334 P39894 INFO Save best model: monitor(max): 0.276708
2022-03-08 10:22:53,398 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:22:53,430 P39894 INFO Train loss: 0.551905
2022-03-08 10:22:53,431 P39894 INFO ************ Epoch=4 end ************
2022-03-08 10:25:15,218 P39894 INFO [Metrics] logloss: 0.527580 - AUC: 0.812968
2022-03-08 10:25:15,219 P39894 INFO Save best model: monitor(max): 0.285388
2022-03-08 10:25:15,281 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:25:15,313 P39894 INFO Train loss: 0.546121
2022-03-08 10:25:15,313 P39894 INFO ************ Epoch=5 end ************
2022-03-08 10:27:36,758 P39894 INFO [Metrics] logloss: 0.524979 - AUC: 0.815094
2022-03-08 10:27:36,759 P39894 INFO Save best model: monitor(max): 0.290115
2022-03-08 10:27:36,844 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:27:36,878 P39894 INFO Train loss: 0.541536
2022-03-08 10:27:36,878 P39894 INFO ************ Epoch=6 end ************
2022-03-08 10:29:58,500 P39894 INFO [Metrics] logloss: 0.523280 - AUC: 0.816354
2022-03-08 10:29:58,501 P39894 INFO Save best model: monitor(max): 0.293074
2022-03-08 10:29:58,573 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:29:58,605 P39894 INFO Train loss: 0.537744
2022-03-08 10:29:58,606 P39894 INFO ************ Epoch=7 end ************
2022-03-08 10:32:19,908 P39894 INFO [Metrics] logloss: 0.521564 - AUC: 0.817689
2022-03-08 10:32:19,909 P39894 INFO Save best model: monitor(max): 0.296125
2022-03-08 10:32:19,978 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:32:20,013 P39894 INFO Train loss: 0.534650
2022-03-08 10:32:20,013 P39894 INFO ************ Epoch=8 end ************
2022-03-08 10:34:41,406 P39894 INFO [Metrics] logloss: 0.519983 - AUC: 0.819009
2022-03-08 10:34:41,407 P39894 INFO Save best model: monitor(max): 0.299026
2022-03-08 10:34:41,476 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:34:41,536 P39894 INFO Train loss: 0.531941
2022-03-08 10:34:41,537 P39894 INFO ************ Epoch=9 end ************
2022-03-08 10:37:03,282 P39894 INFO [Metrics] logloss: 0.519038 - AUC: 0.819851
2022-03-08 10:37:03,283 P39894 INFO Save best model: monitor(max): 0.300813
2022-03-08 10:37:03,354 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:37:03,403 P39894 INFO Train loss: 0.529605
2022-03-08 10:37:03,403 P39894 INFO ************ Epoch=10 end ************
2022-03-08 10:39:23,785 P39894 INFO [Metrics] logloss: 0.518127 - AUC: 0.820498
2022-03-08 10:39:23,786 P39894 INFO Save best model: monitor(max): 0.302371
2022-03-08 10:39:23,856 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:39:23,913 P39894 INFO Train loss: 0.527568
2022-03-08 10:39:23,913 P39894 INFO ************ Epoch=11 end ************
2022-03-08 10:41:45,510 P39894 INFO [Metrics] logloss: 0.517858 - AUC: 0.820840
2022-03-08 10:41:45,511 P39894 INFO Save best model: monitor(max): 0.302981
2022-03-08 10:41:45,581 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:41:45,614 P39894 INFO Train loss: 0.525703
2022-03-08 10:41:45,615 P39894 INFO ************ Epoch=12 end ************
2022-03-08 10:44:06,610 P39894 INFO [Metrics] logloss: 0.517193 - AUC: 0.821357
2022-03-08 10:44:06,612 P39894 INFO Save best model: monitor(max): 0.304164
2022-03-08 10:44:06,693 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:44:06,742 P39894 INFO Train loss: 0.523985
2022-03-08 10:44:06,743 P39894 INFO ************ Epoch=13 end ************
2022-03-08 10:46:27,509 P39894 INFO [Metrics] logloss: 0.516506 - AUC: 0.821982
2022-03-08 10:46:27,510 P39894 INFO Save best model: monitor(max): 0.305476
2022-03-08 10:46:27,580 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:46:27,628 P39894 INFO Train loss: 0.522445
2022-03-08 10:46:27,628 P39894 INFO ************ Epoch=14 end ************
2022-03-08 10:48:49,075 P39894 INFO [Metrics] logloss: 0.515886 - AUC: 0.822487
2022-03-08 10:48:49,076 P39894 INFO Save best model: monitor(max): 0.306601
2022-03-08 10:48:49,146 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:48:49,180 P39894 INFO Train loss: 0.521116
2022-03-08 10:48:49,180 P39894 INFO ************ Epoch=15 end ************
2022-03-08 10:51:10,324 P39894 INFO [Metrics] logloss: 0.515862 - AUC: 0.822549
2022-03-08 10:51:10,325 P39894 INFO Save best model: monitor(max): 0.306687
2022-03-08 10:51:10,403 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:51:10,443 P39894 INFO Train loss: 0.519768
2022-03-08 10:51:10,443 P39894 INFO ************ Epoch=16 end ************
2022-03-08 10:53:31,109 P39894 INFO [Metrics] logloss: 0.515576 - AUC: 0.822939
2022-03-08 10:53:31,111 P39894 INFO Save best model: monitor(max): 0.307363
2022-03-08 10:53:31,181 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:53:31,216 P39894 INFO Train loss: 0.518716
2022-03-08 10:53:31,216 P39894 INFO ************ Epoch=17 end ************
2022-03-08 10:55:51,694 P39894 INFO [Metrics] logloss: 0.515422 - AUC: 0.822929
2022-03-08 10:55:51,695 P39894 INFO Save best model: monitor(max): 0.307507
2022-03-08 10:55:51,764 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:55:51,813 P39894 INFO Train loss: 0.517558
2022-03-08 10:55:51,814 P39894 INFO ************ Epoch=18 end ************
2022-03-08 10:58:12,691 P39894 INFO [Metrics] logloss: 0.515031 - AUC: 0.823471
2022-03-08 10:58:12,692 P39894 INFO Save best model: monitor(max): 0.308440
2022-03-08 10:58:12,776 P39894 INFO --- 591/591 batches finished ---
2022-03-08 10:58:12,825 P39894 INFO Train loss: 0.516637
2022-03-08 10:58:12,825 P39894 INFO ************ Epoch=19 end ************
2022-03-08 11:00:34,131 P39894 INFO [Metrics] logloss: 0.514749 - AUC: 0.823706
2022-03-08 11:00:34,132 P39894 INFO Save best model: monitor(max): 0.308956
2022-03-08 11:00:34,202 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:00:34,251 P39894 INFO Train loss: 0.515542
2022-03-08 11:00:34,252 P39894 INFO ************ Epoch=20 end ************
2022-03-08 11:02:55,492 P39894 INFO [Metrics] logloss: 0.514450 - AUC: 0.823934
2022-03-08 11:02:55,493 P39894 INFO Save best model: monitor(max): 0.309484
2022-03-08 11:02:55,574 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:02:55,622 P39894 INFO Train loss: 0.514821
2022-03-08 11:02:55,623 P39894 INFO ************ Epoch=21 end ************
2022-03-08 11:04:49,999 P39894 INFO [Metrics] logloss: 0.514432 - AUC: 0.824071
2022-03-08 11:04:50,001 P39894 INFO Save best model: monitor(max): 0.309639
2022-03-08 11:04:50,072 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:04:50,121 P39894 INFO Train loss: 0.513919
2022-03-08 11:04:50,121 P39894 INFO ************ Epoch=22 end ************
2022-03-08 11:06:28,367 P39894 INFO [Metrics] logloss: 0.514754 - AUC: 0.824033
2022-03-08 11:06:28,368 P39894 INFO Monitor(max) STOP: 0.309280 !
2022-03-08 11:06:28,369 P39894 INFO Reduce learning rate on plateau: 0.000100
2022-03-08 11:06:28,369 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:06:28,417 P39894 INFO Train loss: 0.513191
2022-03-08 11:06:28,418 P39894 INFO ************ Epoch=23 end ************
2022-03-08 11:08:06,785 P39894 INFO [Metrics] logloss: 0.508845 - AUC: 0.828262
2022-03-08 11:08:06,786 P39894 INFO Save best model: monitor(max): 0.319417
2022-03-08 11:08:06,865 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:08:06,914 P39894 INFO Train loss: 0.485374
2022-03-08 11:08:06,914 P39894 INFO ************ Epoch=24 end ************
2022-03-08 11:09:45,420 P39894 INFO [Metrics] logloss: 0.507655 - AUC: 0.829137
2022-03-08 11:09:45,421 P39894 INFO Save best model: monitor(max): 0.321481
2022-03-08 11:09:45,494 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:09:45,528 P39894 INFO Train loss: 0.481781
2022-03-08 11:09:45,529 P39894 INFO ************ Epoch=25 end ************
2022-03-08 11:11:23,766 P39894 INFO [Metrics] logloss: 0.507157 - AUC: 0.829554
2022-03-08 11:11:23,767 P39894 INFO Save best model: monitor(max): 0.322396
2022-03-08 11:11:23,846 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:11:23,887 P39894 INFO Train loss: 0.480242
2022-03-08 11:11:23,887 P39894 INFO ************ Epoch=26 end ************
2022-03-08 11:13:02,523 P39894 INFO [Metrics] logloss: 0.506891 - AUC: 0.829721
2022-03-08 11:13:02,524 P39894 INFO Save best model: monitor(max): 0.322830
2022-03-08 11:13:02,603 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:13:02,651 P39894 INFO Train loss: 0.479329
2022-03-08 11:13:02,651 P39894 INFO ************ Epoch=27 end ************
2022-03-08 11:14:41,094 P39894 INFO [Metrics] logloss: 0.506761 - AUC: 0.829843
2022-03-08 11:14:41,095 P39894 INFO Save best model: monitor(max): 0.323083
2022-03-08 11:14:41,167 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:14:41,215 P39894 INFO Train loss: 0.478651
2022-03-08 11:14:41,216 P39894 INFO ************ Epoch=28 end ************
2022-03-08 11:16:19,705 P39894 INFO [Metrics] logloss: 0.506711 - AUC: 0.829862
2022-03-08 11:16:19,707 P39894 INFO Save best model: monitor(max): 0.323151
2022-03-08 11:16:19,780 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:16:19,829 P39894 INFO Train loss: 0.478125
2022-03-08 11:16:19,829 P39894 INFO ************ Epoch=29 end ************
2022-03-08 11:17:58,090 P39894 INFO [Metrics] logloss: 0.506693 - AUC: 0.829893
2022-03-08 11:17:58,091 P39894 INFO Save best model: monitor(max): 0.323200
2022-03-08 11:17:58,163 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:17:58,212 P39894 INFO Train loss: 0.477639
2022-03-08 11:17:58,212 P39894 INFO ************ Epoch=30 end ************
2022-03-08 11:19:36,541 P39894 INFO [Metrics] logloss: 0.506645 - AUC: 0.829943
2022-03-08 11:19:36,542 P39894 INFO Save best model: monitor(max): 0.323298
2022-03-08 11:19:36,613 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:19:36,658 P39894 INFO Train loss: 0.477203
2022-03-08 11:19:36,658 P39894 INFO ************ Epoch=31 end ************
2022-03-08 11:21:15,042 P39894 INFO [Metrics] logloss: 0.506657 - AUC: 0.829946
2022-03-08 11:21:15,044 P39894 INFO Monitor(max) STOP: 0.323290 !
2022-03-08 11:21:15,044 P39894 INFO Reduce learning rate on plateau: 0.000010
2022-03-08 11:21:15,044 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:21:15,093 P39894 INFO Train loss: 0.476838
2022-03-08 11:21:15,093 P39894 INFO ************ Epoch=32 end ************
2022-03-08 11:22:53,352 P39894 INFO [Metrics] logloss: 0.506529 - AUC: 0.830041
2022-03-08 11:22:53,353 P39894 INFO Save best model: monitor(max): 0.323512
2022-03-08 11:22:53,424 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:22:53,499 P39894 INFO Train loss: 0.472513
2022-03-08 11:22:53,499 P39894 INFO ************ Epoch=33 end ************
2022-03-08 11:24:31,865 P39894 INFO [Metrics] logloss: 0.506506 - AUC: 0.830056
2022-03-08 11:24:31,866 P39894 INFO Save best model: monitor(max): 0.323549
2022-03-08 11:24:31,938 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:24:31,986 P39894 INFO Train loss: 0.472436
2022-03-08 11:24:31,986 P39894 INFO ************ Epoch=34 end ************
2022-03-08 11:26:10,401 P39894 INFO [Metrics] logloss: 0.506505 - AUC: 0.830063
2022-03-08 11:26:10,403 P39894 INFO Save best model: monitor(max): 0.323559
2022-03-08 11:26:10,474 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:26:10,549 P39894 INFO Train loss: 0.472385
2022-03-08 11:26:10,549 P39894 INFO ************ Epoch=35 end ************
2022-03-08 11:27:48,907 P39894 INFO [Metrics] logloss: 0.506491 - AUC: 0.830076
2022-03-08 11:27:48,908 P39894 INFO Save best model: monitor(max): 0.323585
2022-03-08 11:27:48,984 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:27:49,059 P39894 INFO Train loss: 0.472328
2022-03-08 11:27:49,059 P39894 INFO ************ Epoch=36 end ************
2022-03-08 11:29:27,447 P39894 INFO [Metrics] logloss: 0.506486 - AUC: 0.830079
2022-03-08 11:29:27,448 P39894 INFO Save best model: monitor(max): 0.323593
2022-03-08 11:29:27,518 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:29:27,567 P39894 INFO Train loss: 0.472292
2022-03-08 11:29:27,568 P39894 INFO ************ Epoch=37 end ************
2022-03-08 11:31:06,107 P39894 INFO [Metrics] logloss: 0.506482 - AUC: 0.830084
2022-03-08 11:31:06,108 P39894 INFO Save best model: monitor(max): 0.323602
2022-03-08 11:31:06,187 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:31:06,258 P39894 INFO Train loss: 0.472231
2022-03-08 11:31:06,258 P39894 INFO ************ Epoch=38 end ************
2022-03-08 11:32:44,733 P39894 INFO [Metrics] logloss: 0.506487 - AUC: 0.830079
2022-03-08 11:32:44,735 P39894 INFO Monitor(max) STOP: 0.323592 !
2022-03-08 11:32:44,735 P39894 INFO Reduce learning rate on plateau: 0.000001
2022-03-08 11:32:44,735 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:32:44,808 P39894 INFO Train loss: 0.472205
2022-03-08 11:32:44,808 P39894 INFO ************ Epoch=39 end ************
2022-03-08 11:34:23,172 P39894 INFO [Metrics] logloss: 0.506485 - AUC: 0.830080
2022-03-08 11:34:23,173 P39894 INFO Monitor(max) STOP: 0.323596 !
2022-03-08 11:34:23,173 P39894 INFO Reduce learning rate on plateau: 0.000001
2022-03-08 11:34:23,173 P39894 INFO Early stopping at epoch=40
2022-03-08 11:34:23,173 P39894 INFO --- 591/591 batches finished ---
2022-03-08 11:34:23,226 P39894 INFO Train loss: 0.471743
2022-03-08 11:34:23,227 P39894 INFO Training finished.
2022-03-08 11:34:23,227 P39894 INFO Load best model: /home/XXX/FuxiCTRv1.0/benchmarks_local/KKBox/FM_kkbox_x1/kkbox_x1_227d337d/FM_kkbox_x1_002_c55e9f15_model.ckpt
2022-03-08 11:34:23,300 P39894 INFO ****** Validation evaluation ******
2022-03-08 11:34:27,518 P39894 INFO [Metrics] logloss: 0.506482 - AUC: 0.830084
2022-03-08 11:34:27,554 P39894 INFO ******** Test evaluation ********
2022-03-08 11:34:27,554 P39894 INFO Loading data...
2022-03-08 11:34:27,555 P39894 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-08 11:34:27,625 P39894 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-08 11:34:27,625 P39894 INFO Loading test data done.
2022-03-08 11:34:31,786 P39894 INFO [Metrics] logloss: 0.506000 - AUC: 0.830423

```
