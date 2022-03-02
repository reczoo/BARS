## LR_kkbox_x1

A hands-on guide to run the LR model on the Kkbox_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

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
Dataset ID: [Kkbox_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Kkbox/README.md#Kkbox_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [LR](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/LR.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [LR_kkbox_x1_tuner_config_06](./LR_kkbox_x1_tuner_config_06). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd LR_kkbox_x1
    nohup python run_expid.py --config ./LR_kkbox_x1_tuner_config_06 --expid LR_kkbox_x1_008_aab85bd2 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.574564 | 0.767785  |


### Logs
```python
2020-04-06 05:26:46,819 P31634 INFO {
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x4_002_c5c9c6e3",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "LR",
    "model_id": "LR_kkbox_x4_008_5ea0ce9c",
    "model_root": "./KKBox/LR_kkbox/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-07",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/KKBox/KKBox_x4/test.csv",
    "train_data": "../data/KKBox/KKBox_x4/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/KKBox/KKBox_x4/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-04-06 05:26:46,820 P31634 INFO Set up feature encoder...
2020-04-06 05:26:46,820 P31634 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x4_002_c5c9c6e3/feature_encoder.pkl
2020-04-06 05:26:47,005 P31634 INFO Loading data...
2020-04-06 05:26:47,008 P31634 INFO Loading data from h5: ../data/KKBox/kkbox_x4_002_c5c9c6e3/train.h5
2020-04-06 05:26:47,503 P31634 INFO Loading data from h5: ../data/KKBox/kkbox_x4_002_c5c9c6e3/valid.h5
2020-04-06 05:26:47,788 P31634 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2020-04-06 05:26:47,816 P31634 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-04-06 05:26:47,816 P31634 INFO Loading train data done.
2020-04-06 05:26:52,911 P31634 INFO **** Start training: 591 batches/epoch ****
2020-04-06 05:28:27,838 P31634 INFO [Metrics] logloss: 0.621953 - AUC: 0.716514
2020-04-06 05:28:27,840 P31634 INFO Save best model: monitor(max): 0.094561
2020-04-06 05:28:27,842 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:28:27,910 P31634 INFO Train loss: 0.640387
2020-04-06 05:28:27,910 P31634 INFO ************ Epoch=1 end ************
2020-04-06 05:30:11,153 P31634 INFO [Metrics] logloss: 0.607171 - AUC: 0.736694
2020-04-06 05:30:11,154 P31634 INFO Save best model: monitor(max): 0.129523
2020-04-06 05:30:11,157 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:30:11,262 P31634 INFO Train loss: 0.612505
2020-04-06 05:30:11,262 P31634 INFO ************ Epoch=2 end ************
2020-04-06 05:31:53,977 P31634 INFO [Metrics] logloss: 0.597854 - AUC: 0.747239
2020-04-06 05:31:53,980 P31634 INFO Save best model: monitor(max): 0.149384
2020-04-06 05:31:53,984 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:31:54,089 P31634 INFO Train loss: 0.599715
2020-04-06 05:31:54,090 P31634 INFO ************ Epoch=3 end ************
2020-04-06 05:33:38,351 P31634 INFO [Metrics] logloss: 0.591590 - AUC: 0.753631
2020-04-06 05:33:38,352 P31634 INFO Save best model: monitor(max): 0.162040
2020-04-06 05:33:38,354 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:33:38,409 P31634 INFO Train loss: 0.591175
2020-04-06 05:33:38,409 P31634 INFO ************ Epoch=4 end ************
2020-04-06 05:35:19,899 P31634 INFO [Metrics] logloss: 0.587237 - AUC: 0.757938
2020-04-06 05:35:19,901 P31634 INFO Save best model: monitor(max): 0.170701
2020-04-06 05:35:19,904 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:35:19,990 P31634 INFO Train loss: 0.585229
2020-04-06 05:35:19,990 P31634 INFO ************ Epoch=5 end ************
2020-04-06 05:37:04,643 P31634 INFO [Metrics] logloss: 0.584146 - AUC: 0.760507
2020-04-06 05:37:04,644 P31634 INFO Save best model: monitor(max): 0.176360
2020-04-06 05:37:04,647 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:37:04,703 P31634 INFO Train loss: 0.580913
2020-04-06 05:37:04,703 P31634 INFO ************ Epoch=6 end ************
2020-04-06 05:38:44,725 P31634 INFO [Metrics] logloss: 0.581892 - AUC: 0.762368
2020-04-06 05:38:44,726 P31634 INFO Save best model: monitor(max): 0.180475
2020-04-06 05:38:44,729 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:38:44,786 P31634 INFO Train loss: 0.577731
2020-04-06 05:38:44,786 P31634 INFO ************ Epoch=7 end ************
2020-04-06 05:40:29,149 P31634 INFO [Metrics] logloss: 0.580243 - AUC: 0.763674
2020-04-06 05:40:29,150 P31634 INFO Save best model: monitor(max): 0.183431
2020-04-06 05:40:29,153 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:40:29,305 P31634 INFO Train loss: 0.575320
2020-04-06 05:40:29,305 P31634 INFO ************ Epoch=8 end ************
2020-04-06 05:42:12,779 P31634 INFO [Metrics] logloss: 0.578949 - AUC: 0.764557
2020-04-06 05:42:12,781 P31634 INFO Save best model: monitor(max): 0.185609
2020-04-06 05:42:12,784 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:42:12,861 P31634 INFO Train loss: 0.573462
2020-04-06 05:42:12,861 P31634 INFO ************ Epoch=9 end ************
2020-04-06 05:43:55,693 P31634 INFO [Metrics] logloss: 0.577981 - AUC: 0.765342
2020-04-06 05:43:55,694 P31634 INFO Save best model: monitor(max): 0.187360
2020-04-06 05:43:55,697 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:43:55,797 P31634 INFO Train loss: 0.571978
2020-04-06 05:43:55,797 P31634 INFO ************ Epoch=10 end ************
2020-04-06 05:45:21,921 P31634 INFO [Metrics] logloss: 0.577255 - AUC: 0.765861
2020-04-06 05:45:21,922 P31634 INFO Save best model: monitor(max): 0.188606
2020-04-06 05:45:21,924 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:45:22,010 P31634 INFO Train loss: 0.570795
2020-04-06 05:45:22,010 P31634 INFO ************ Epoch=11 end ************
2020-04-06 05:47:06,227 P31634 INFO [Metrics] logloss: 0.576704 - AUC: 0.766241
2020-04-06 05:47:06,229 P31634 INFO Save best model: monitor(max): 0.189538
2020-04-06 05:47:06,231 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:47:06,290 P31634 INFO Train loss: 0.569863
2020-04-06 05:47:06,290 P31634 INFO ************ Epoch=12 end ************
2020-04-06 05:48:50,642 P31634 INFO [Metrics] logloss: 0.576268 - AUC: 0.766610
2020-04-06 05:48:50,643 P31634 INFO Save best model: monitor(max): 0.190342
2020-04-06 05:48:50,646 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:48:50,823 P31634 INFO Train loss: 0.569065
2020-04-06 05:48:50,823 P31634 INFO ************ Epoch=13 end ************
2020-04-06 05:50:27,187 P31634 INFO [Metrics] logloss: 0.575928 - AUC: 0.766826
2020-04-06 05:50:27,188 P31634 INFO Save best model: monitor(max): 0.190899
2020-04-06 05:50:27,190 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:50:27,281 P31634 INFO Train loss: 0.568407
2020-04-06 05:50:27,282 P31634 INFO ************ Epoch=14 end ************
2020-04-06 05:52:02,996 P31634 INFO [Metrics] logloss: 0.575646 - AUC: 0.767034
2020-04-06 05:52:02,998 P31634 INFO Save best model: monitor(max): 0.191388
2020-04-06 05:52:03,000 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:52:03,083 P31634 INFO Train loss: 0.567924
2020-04-06 05:52:03,083 P31634 INFO ************ Epoch=15 end ************
2020-04-06 05:53:47,300 P31634 INFO [Metrics] logloss: 0.575472 - AUC: 0.767178
2020-04-06 05:53:47,302 P31634 INFO Save best model: monitor(max): 0.191706
2020-04-06 05:53:47,304 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:53:47,379 P31634 INFO Train loss: 0.567463
2020-04-06 05:53:47,380 P31634 INFO ************ Epoch=16 end ************
2020-04-06 05:55:31,475 P31634 INFO [Metrics] logloss: 0.575270 - AUC: 0.767325
2020-04-06 05:55:31,476 P31634 INFO Save best model: monitor(max): 0.192055
2020-04-06 05:55:31,479 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:55:31,567 P31634 INFO Train loss: 0.567075
2020-04-06 05:55:31,567 P31634 INFO ************ Epoch=17 end ************
2020-04-06 05:57:16,036 P31634 INFO [Metrics] logloss: 0.575169 - AUC: 0.767387
2020-04-06 05:57:16,037 P31634 INFO Save best model: monitor(max): 0.192217
2020-04-06 05:57:16,039 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:57:16,140 P31634 INFO Train loss: 0.566766
2020-04-06 05:57:16,140 P31634 INFO ************ Epoch=18 end ************
2020-04-06 05:59:00,305 P31634 INFO [Metrics] logloss: 0.575066 - AUC: 0.767451
2020-04-06 05:59:00,307 P31634 INFO Save best model: monitor(max): 0.192385
2020-04-06 05:59:00,309 P31634 INFO --- 591/591 batches finished ---
2020-04-06 05:59:00,410 P31634 INFO Train loss: 0.566496
2020-04-06 05:59:00,410 P31634 INFO ************ Epoch=19 end ************
2020-04-06 06:00:44,142 P31634 INFO [Metrics] logloss: 0.575059 - AUC: 0.767502
2020-04-06 06:00:44,143 P31634 INFO Save best model: monitor(max): 0.192443
2020-04-06 06:00:44,146 P31634 INFO --- 591/591 batches finished ---
2020-04-06 06:00:44,235 P31634 INFO Train loss: 0.566267
2020-04-06 06:00:44,235 P31634 INFO ************ Epoch=20 end ************
2020-04-06 06:02:30,267 P31634 INFO [Metrics] logloss: 0.574942 - AUC: 0.767543
2020-04-06 06:02:30,268 P31634 INFO Save best model: monitor(max): 0.192600
2020-04-06 06:02:30,271 P31634 INFO --- 591/591 batches finished ---
2020-04-06 06:02:30,371 P31634 INFO Train loss: 0.566092
2020-04-06 06:02:30,371 P31634 INFO ************ Epoch=21 end ************
2020-04-06 06:04:12,038 P31634 INFO [Metrics] logloss: 0.574911 - AUC: 0.767575
2020-04-06 06:04:12,039 P31634 INFO Save best model: monitor(max): 0.192665
2020-04-06 06:04:12,041 P31634 INFO --- 591/591 batches finished ---
2020-04-06 06:04:12,146 P31634 INFO Train loss: 0.565891
2020-04-06 06:04:12,146 P31634 INFO ************ Epoch=22 end ************
2020-04-06 06:05:58,319 P31634 INFO [Metrics] logloss: 0.574879 - AUC: 0.767595
2020-04-06 06:05:58,320 P31634 INFO Save best model: monitor(max): 0.192716
2020-04-06 06:05:58,323 P31634 INFO --- 591/591 batches finished ---
2020-04-06 06:05:58,389 P31634 INFO Train loss: 0.565743
2020-04-06 06:05:58,389 P31634 INFO ************ Epoch=23 end ************
2020-04-06 06:07:41,155 P31634 INFO [Metrics] logloss: 0.574835 - AUC: 0.767637
2020-04-06 06:07:41,157 P31634 INFO Save best model: monitor(max): 0.192801
2020-04-06 06:07:41,159 P31634 INFO --- 591/591 batches finished ---
2020-04-06 06:07:41,234 P31634 INFO Train loss: 0.565600
2020-04-06 06:07:41,235 P31634 INFO ************ Epoch=24 end ************
2020-04-06 06:09:26,349 P31634 INFO [Metrics] logloss: 0.574844 - AUC: 0.767636
2020-04-06 06:09:26,351 P31634 INFO Monitor(max) STOP: 0.192792 !
2020-04-06 06:09:26,351 P31634 INFO Reduce learning rate on plateau: 0.000100
2020-04-06 06:09:26,351 P31634 INFO --- 591/591 batches finished ---
2020-04-06 06:09:26,407 P31634 INFO Train loss: 0.565502
2020-04-06 06:09:26,407 P31634 INFO ************ Epoch=25 end ************
2020-04-06 06:11:07,945 P31634 INFO [Metrics] logloss: 0.574820 - AUC: 0.767659
2020-04-06 06:11:07,946 P31634 INFO Save best model: monitor(max): 0.192839
2020-04-06 06:11:07,949 P31634 INFO --- 591/591 batches finished ---
2020-04-06 06:11:08,036 P31634 INFO Train loss: 0.564766
2020-04-06 06:11:08,036 P31634 INFO ************ Epoch=26 end ************
2020-04-06 06:12:53,027 P31634 INFO [Metrics] logloss: 0.574813 - AUC: 0.767663
2020-04-06 06:12:53,028 P31634 INFO Save best model: monitor(max): 0.192850
2020-04-06 06:12:53,031 P31634 INFO --- 591/591 batches finished ---
2020-04-06 06:12:53,118 P31634 INFO Train loss: 0.564755
2020-04-06 06:12:53,118 P31634 INFO ************ Epoch=27 end ************
2020-04-06 06:14:35,820 P31634 INFO [Metrics] logloss: 0.574813 - AUC: 0.767665
2020-04-06 06:14:35,821 P31634 INFO Save best model: monitor(max): 0.192852
2020-04-06 06:14:35,823 P31634 INFO --- 591/591 batches finished ---
2020-04-06 06:14:35,879 P31634 INFO Train loss: 0.564737
2020-04-06 06:14:35,879 P31634 INFO ************ Epoch=28 end ************
2020-04-06 06:16:19,266 P31634 INFO [Metrics] logloss: 0.574816 - AUC: 0.767666
2020-04-06 06:16:19,268 P31634 INFO Monitor(max) STOP: 0.192850 !
2020-04-06 06:16:19,268 P31634 INFO Reduce learning rate on plateau: 0.000010
2020-04-06 06:16:19,268 P31634 INFO --- 591/591 batches finished ---
2020-04-06 06:16:19,354 P31634 INFO Train loss: 0.564686
2020-04-06 06:16:19,355 P31634 INFO ************ Epoch=29 end ************
2020-04-06 06:18:02,827 P31634 INFO [Metrics] logloss: 0.574811 - AUC: 0.767667
2020-04-06 06:18:02,829 P31634 INFO Save best model: monitor(max): 0.192856
2020-04-06 06:18:02,831 P31634 INFO --- 591/591 batches finished ---
2020-04-06 06:18:02,924 P31634 INFO Train loss: 0.564658
2020-04-06 06:18:02,924 P31634 INFO ************ Epoch=30 end ************
2020-04-06 06:19:46,696 P31634 INFO [Metrics] logloss: 0.574810 - AUC: 0.767667
2020-04-06 06:19:46,697 P31634 INFO Save best model: monitor(max): 0.192857
2020-04-06 06:19:46,700 P31634 INFO --- 591/591 batches finished ---
2020-04-06 06:19:46,785 P31634 INFO Train loss: 0.564659
2020-04-06 06:19:46,785 P31634 INFO ************ Epoch=31 end ************
2020-04-06 06:21:30,164 P31634 INFO [Metrics] logloss: 0.574810 - AUC: 0.767667
2020-04-06 06:21:30,165 P31634 INFO Monitor(max) STOP: 0.192857 !
2020-04-06 06:21:30,166 P31634 INFO Reduce learning rate on plateau: 0.000001
2020-04-06 06:21:30,166 P31634 INFO --- 591/591 batches finished ---
2020-04-06 06:21:30,271 P31634 INFO Train loss: 0.564677
2020-04-06 06:21:30,271 P31634 INFO ************ Epoch=32 end ************
2020-04-06 06:23:05,613 P31634 INFO [Metrics] logloss: 0.574810 - AUC: 0.767667
2020-04-06 06:23:05,615 P31634 INFO Monitor(max) STOP: 0.192857 !
2020-04-06 06:23:05,615 P31634 INFO Reduce learning rate on plateau: 0.000001
2020-04-06 06:23:05,615 P31634 INFO Early stopping at epoch=33
2020-04-06 06:23:05,615 P31634 INFO --- 591/591 batches finished ---
2020-04-06 06:23:05,701 P31634 INFO Train loss: 0.564667
2020-04-06 06:23:05,701 P31634 INFO Training finished.
2020-04-06 06:23:05,701 P31634 INFO Load best model: /home/XXX/benchmarks/KKBox/LR_kkbox/kkbox_x4_002_c5c9c6e3/LR_kkbox_x4_008_5ea0ce9c_model.ckpt
2020-04-06 06:23:05,719 P31634 INFO ****** Train/validation evaluation ******
2020-04-06 06:24:03,038 P31634 INFO [Metrics] logloss: 0.562996 - AUC: 0.779752
2020-04-06 06:24:09,968 P31634 INFO [Metrics] logloss: 0.574810 - AUC: 0.767667
2020-04-06 06:24:10,053 P31634 INFO ******** Test evaluation ********
2020-04-06 06:24:10,053 P31634 INFO Loading data...
2020-04-06 06:24:10,053 P31634 INFO Loading data from h5: ../data/KKBox/kkbox_x4_002_c5c9c6e3/test.h5
2020-04-06 06:24:10,137 P31634 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-04-06 06:24:10,137 P31634 INFO Loading test data done.
2020-04-06 06:24:16,638 P31634 INFO [Metrics] logloss: 0.574564 - AUC: 0.767785

```
