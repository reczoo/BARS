## DeepFM_kkbox_x1

A hands-on guide to run the DeepFM model on the KKBox_x1 dataset.

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
Dataset ID: [KKBox_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/KKBox/README.md#KKBox_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [DeepFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DeepFM_kkbox_x1_tuner_config_03](./DeepFM_kkbox_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DeepFM_kkbox_x1
    nohup python run_expid.py --config ./DeepFM_kkbox_x1_tuner_config_03 --expid DeepFM_kkbox_x1_005_32cb2ca8 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.478468 | 0.853109  |


### Logs
```python
2020-04-11 01:30:00,459 P36740 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x4_001_c5c9c6e3",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.0005",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[5000, 5000]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "DeepFM",
    "model_id": "DeepFM_kkbox_x4_005_4366b458",
    "model_root": "./KKBox/DeepFM_kkbox/",
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
    "test_data": "../data/KKBox/KKBox_x4/test.csv",
    "train_data": "../data/KKBox/KKBox_x4/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/KKBox/KKBox_x4/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-04-11 01:30:00,460 P36740 INFO Set up feature encoder...
2020-04-11 01:30:00,461 P36740 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x4_001_c5c9c6e3/feature_encoder.pkl
2020-04-11 01:30:00,657 P36740 INFO Loading data...
2020-04-11 01:30:00,662 P36740 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/train.h5
2020-04-11 01:30:01,060 P36740 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/valid.h5
2020-04-11 01:30:01,279 P36740 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2020-04-11 01:30:01,298 P36740 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-04-11 01:30:01,298 P36740 INFO Loading train data done.
2020-04-11 01:30:06,211 P36740 INFO **** Start training: 591 batches/epoch ****
2020-04-11 01:35:54,445 P36740 INFO [Metrics] logloss: 0.553069 - AUC: 0.789367
2020-04-11 01:35:54,446 P36740 INFO Save best model: monitor(max): 0.236298
2020-04-11 01:35:54,629 P36740 INFO --- 591/591 batches finished ---
2020-04-11 01:35:54,672 P36740 INFO Train loss: 0.607366
2020-04-11 01:35:54,672 P36740 INFO ************ Epoch=1 end ************
2020-04-11 01:41:44,308 P36740 INFO [Metrics] logloss: 0.537251 - AUC: 0.803791
2020-04-11 01:41:44,309 P36740 INFO Save best model: monitor(max): 0.266540
2020-04-11 01:41:44,627 P36740 INFO --- 591/591 batches finished ---
2020-04-11 01:41:44,674 P36740 INFO Train loss: 0.586910
2020-04-11 01:41:44,675 P36740 INFO ************ Epoch=2 end ************
2020-04-11 01:47:34,336 P36740 INFO [Metrics] logloss: 0.529845 - AUC: 0.809816
2020-04-11 01:47:34,337 P36740 INFO Save best model: monitor(max): 0.279972
2020-04-11 01:47:34,662 P36740 INFO --- 591/591 batches finished ---
2020-04-11 01:47:34,703 P36740 INFO Train loss: 0.579820
2020-04-11 01:47:34,704 P36740 INFO ************ Epoch=3 end ************
2020-04-11 01:53:25,468 P36740 INFO [Metrics] logloss: 0.524599 - AUC: 0.814243
2020-04-11 01:53:25,470 P36740 INFO Save best model: monitor(max): 0.289644
2020-04-11 01:53:25,795 P36740 INFO --- 591/591 batches finished ---
2020-04-11 01:53:25,837 P36740 INFO Train loss: 0.576339
2020-04-11 01:53:25,838 P36740 INFO ************ Epoch=4 end ************
2020-04-11 01:59:16,767 P36740 INFO [Metrics] logloss: 0.521133 - AUC: 0.817315
2020-04-11 01:59:16,768 P36740 INFO Save best model: monitor(max): 0.296182
2020-04-11 01:59:17,091 P36740 INFO --- 591/591 batches finished ---
2020-04-11 01:59:17,138 P36740 INFO Train loss: 0.573385
2020-04-11 01:59:17,138 P36740 INFO ************ Epoch=5 end ************
2020-04-11 02:05:09,603 P36740 INFO [Metrics] logloss: 0.517988 - AUC: 0.819908
2020-04-11 02:05:09,605 P36740 INFO Save best model: monitor(max): 0.301920
2020-04-11 02:05:09,941 P36740 INFO --- 591/591 batches finished ---
2020-04-11 02:05:09,983 P36740 INFO Train loss: 0.571387
2020-04-11 02:05:09,983 P36740 INFO ************ Epoch=6 end ************
2020-04-11 02:11:01,420 P36740 INFO [Metrics] logloss: 0.515047 - AUC: 0.822200
2020-04-11 02:11:01,421 P36740 INFO Save best model: monitor(max): 0.307154
2020-04-11 02:11:01,764 P36740 INFO --- 591/591 batches finished ---
2020-04-11 02:11:01,805 P36740 INFO Train loss: 0.569743
2020-04-11 02:11:01,806 P36740 INFO ************ Epoch=7 end ************
2020-04-11 02:16:56,279 P36740 INFO [Metrics] logloss: 0.512316 - AUC: 0.824445
2020-04-11 02:16:56,279 P36740 INFO Save best model: monitor(max): 0.312129
2020-04-11 02:16:56,593 P36740 INFO --- 591/591 batches finished ---
2020-04-11 02:16:56,634 P36740 INFO Train loss: 0.568383
2020-04-11 02:16:56,634 P36740 INFO ************ Epoch=8 end ************
2020-04-11 02:22:51,485 P36740 INFO [Metrics] logloss: 0.510845 - AUC: 0.826088
2020-04-11 02:22:51,486 P36740 INFO Save best model: monitor(max): 0.315242
2020-04-11 02:22:51,808 P36740 INFO --- 591/591 batches finished ---
2020-04-11 02:22:51,849 P36740 INFO Train loss: 0.566919
2020-04-11 02:22:51,849 P36740 INFO ************ Epoch=9 end ************
2020-04-11 02:28:46,645 P36740 INFO [Metrics] logloss: 0.508298 - AUC: 0.827768
2020-04-11 02:28:46,647 P36740 INFO Save best model: monitor(max): 0.319470
2020-04-11 02:28:46,963 P36740 INFO --- 591/591 batches finished ---
2020-04-11 02:28:47,004 P36740 INFO Train loss: 0.565689
2020-04-11 02:28:47,004 P36740 INFO ************ Epoch=10 end ************
2020-04-11 02:34:42,189 P36740 INFO [Metrics] logloss: 0.507543 - AUC: 0.828453
2020-04-11 02:34:42,190 P36740 INFO Save best model: monitor(max): 0.320910
2020-04-11 02:34:42,496 P36740 INFO --- 591/591 batches finished ---
2020-04-11 02:34:42,537 P36740 INFO Train loss: 0.564678
2020-04-11 02:34:42,538 P36740 INFO ************ Epoch=11 end ************
2020-04-11 02:40:39,385 P36740 INFO [Metrics] logloss: 0.505524 - AUC: 0.830030
2020-04-11 02:40:39,386 P36740 INFO Save best model: monitor(max): 0.324506
2020-04-11 02:40:39,702 P36740 INFO --- 591/591 batches finished ---
2020-04-11 02:40:39,749 P36740 INFO Train loss: 0.563457
2020-04-11 02:40:39,749 P36740 INFO ************ Epoch=12 end ************
2020-04-11 02:46:37,805 P36740 INFO [Metrics] logloss: 0.504108 - AUC: 0.831174
2020-04-11 02:46:37,806 P36740 INFO Save best model: monitor(max): 0.327066
2020-04-11 02:46:38,118 P36740 INFO --- 591/591 batches finished ---
2020-04-11 02:46:38,160 P36740 INFO Train loss: 0.562488
2020-04-11 02:46:38,160 P36740 INFO ************ Epoch=13 end ************
2020-04-11 02:52:35,066 P36740 INFO [Metrics] logloss: 0.502881 - AUC: 0.832354
2020-04-11 02:52:35,067 P36740 INFO Save best model: monitor(max): 0.329474
2020-04-11 02:52:35,402 P36740 INFO --- 591/591 batches finished ---
2020-04-11 02:52:35,445 P36740 INFO Train loss: 0.561633
2020-04-11 02:52:35,445 P36740 INFO ************ Epoch=14 end ************
2020-04-11 02:58:32,899 P36740 INFO [Metrics] logloss: 0.501243 - AUC: 0.833330
2020-04-11 02:58:32,900 P36740 INFO Save best model: monitor(max): 0.332086
2020-04-11 02:58:33,219 P36740 INFO --- 591/591 batches finished ---
2020-04-11 02:58:33,260 P36740 INFO Train loss: 0.560903
2020-04-11 02:58:33,261 P36740 INFO ************ Epoch=15 end ************
2020-04-11 03:04:33,793 P36740 INFO [Metrics] logloss: 0.500089 - AUC: 0.834473
2020-04-11 03:04:33,795 P36740 INFO Save best model: monitor(max): 0.334383
2020-04-11 03:04:34,146 P36740 INFO --- 591/591 batches finished ---
2020-04-11 03:04:34,187 P36740 INFO Train loss: 0.559979
2020-04-11 03:04:34,187 P36740 INFO ************ Epoch=16 end ************
2020-04-11 03:10:35,363 P36740 INFO [Metrics] logloss: 0.499854 - AUC: 0.834536
2020-04-11 03:10:35,364 P36740 INFO Save best model: monitor(max): 0.334682
2020-04-11 03:10:35,686 P36740 INFO --- 591/591 batches finished ---
2020-04-11 03:10:35,727 P36740 INFO Train loss: 0.559008
2020-04-11 03:10:35,727 P36740 INFO ************ Epoch=17 end ************
2020-04-11 03:16:36,304 P36740 INFO [Metrics] logloss: 0.498868 - AUC: 0.835288
2020-04-11 03:16:36,306 P36740 INFO Save best model: monitor(max): 0.336420
2020-04-11 03:16:36,647 P36740 INFO --- 591/591 batches finished ---
2020-04-11 03:16:36,689 P36740 INFO Train loss: 0.558551
2020-04-11 03:16:36,689 P36740 INFO ************ Epoch=18 end ************
2020-04-11 03:22:36,211 P36740 INFO [Metrics] logloss: 0.498406 - AUC: 0.835749
2020-04-11 03:22:36,212 P36740 INFO Save best model: monitor(max): 0.337343
2020-04-11 03:22:36,600 P36740 INFO --- 591/591 batches finished ---
2020-04-11 03:22:36,673 P36740 INFO Train loss: 0.557625
2020-04-11 03:22:36,673 P36740 INFO ************ Epoch=19 end ************
2020-04-11 03:28:38,792 P36740 INFO [Metrics] logloss: 0.497251 - AUC: 0.836602
2020-04-11 03:28:38,793 P36740 INFO Save best model: monitor(max): 0.339351
2020-04-11 03:28:39,116 P36740 INFO --- 591/591 batches finished ---
2020-04-11 03:28:39,157 P36740 INFO Train loss: 0.556984
2020-04-11 03:28:39,157 P36740 INFO ************ Epoch=20 end ************
2020-04-11 03:34:39,157 P36740 INFO [Metrics] logloss: 0.497197 - AUC: 0.836704
2020-04-11 03:34:39,158 P36740 INFO Save best model: monitor(max): 0.339507
2020-04-11 03:34:39,489 P36740 INFO --- 591/591 batches finished ---
2020-04-11 03:34:39,541 P36740 INFO Train loss: 0.556047
2020-04-11 03:34:39,541 P36740 INFO ************ Epoch=21 end ************
2020-04-11 03:40:39,769 P36740 INFO [Metrics] logloss: 0.495856 - AUC: 0.837580
2020-04-11 03:40:39,770 P36740 INFO Save best model: monitor(max): 0.341724
2020-04-11 03:40:40,125 P36740 INFO --- 591/591 batches finished ---
2020-04-11 03:40:40,166 P36740 INFO Train loss: 0.555625
2020-04-11 03:40:40,166 P36740 INFO ************ Epoch=22 end ************
2020-04-11 03:46:40,850 P36740 INFO [Metrics] logloss: 0.495688 - AUC: 0.838069
2020-04-11 03:46:40,851 P36740 INFO Save best model: monitor(max): 0.342381
2020-04-11 03:46:41,206 P36740 INFO --- 591/591 batches finished ---
2020-04-11 03:46:41,247 P36740 INFO Train loss: 0.554776
2020-04-11 03:46:41,247 P36740 INFO ************ Epoch=23 end ************
2020-04-11 03:52:41,292 P36740 INFO [Metrics] logloss: 0.494691 - AUC: 0.838636
2020-04-11 03:52:41,293 P36740 INFO Save best model: monitor(max): 0.343945
2020-04-11 03:52:41,628 P36740 INFO --- 591/591 batches finished ---
2020-04-11 03:52:41,671 P36740 INFO Train loss: 0.554247
2020-04-11 03:52:41,671 P36740 INFO ************ Epoch=24 end ************
2020-04-11 03:58:41,496 P36740 INFO [Metrics] logloss: 0.493865 - AUC: 0.839322
2020-04-11 03:58:41,497 P36740 INFO Save best model: monitor(max): 0.345457
2020-04-11 03:58:41,821 P36740 INFO --- 591/591 batches finished ---
2020-04-11 03:58:41,868 P36740 INFO Train loss: 0.553507
2020-04-11 03:58:41,868 P36740 INFO ************ Epoch=25 end ************
2020-04-11 04:04:41,837 P36740 INFO [Metrics] logloss: 0.494122 - AUC: 0.839153
2020-04-11 04:04:41,838 P36740 INFO Monitor(max) STOP: 0.345032 !
2020-04-11 04:04:41,838 P36740 INFO Reduce learning rate on plateau: 0.000100
2020-04-11 04:04:41,838 P36740 INFO --- 591/591 batches finished ---
2020-04-11 04:04:41,880 P36740 INFO Train loss: 0.552933
2020-04-11 04:04:41,880 P36740 INFO ************ Epoch=26 end ************
2020-04-11 04:10:42,759 P36740 INFO [Metrics] logloss: 0.480871 - AUC: 0.849818
2020-04-11 04:10:42,760 P36740 INFO Save best model: monitor(max): 0.368946
2020-04-11 04:10:43,112 P36740 INFO --- 591/591 batches finished ---
2020-04-11 04:10:43,152 P36740 INFO Train loss: 0.491113
2020-04-11 04:10:43,153 P36740 INFO ************ Epoch=27 end ************
2020-04-11 04:16:44,619 P36740 INFO [Metrics] logloss: 0.478851 - AUC: 0.851732
2020-04-11 04:16:44,620 P36740 INFO Save best model: monitor(max): 0.372881
2020-04-11 04:16:44,964 P36740 INFO --- 591/591 batches finished ---
2020-04-11 04:16:45,006 P36740 INFO Train loss: 0.462090
2020-04-11 04:16:45,006 P36740 INFO ************ Epoch=28 end ************
2020-04-11 04:22:44,255 P36740 INFO [Metrics] logloss: 0.478786 - AUC: 0.852442
2020-04-11 04:22:44,257 P36740 INFO Save best model: monitor(max): 0.373656
2020-04-11 04:22:44,605 P36740 INFO --- 591/591 batches finished ---
2020-04-11 04:22:44,647 P36740 INFO Train loss: 0.451615
2020-04-11 04:22:44,647 P36740 INFO ************ Epoch=29 end ************
2020-04-11 04:28:46,325 P36740 INFO [Metrics] logloss: 0.478818 - AUC: 0.852763
2020-04-11 04:28:46,326 P36740 INFO Save best model: monitor(max): 0.373945
2020-04-11 04:28:46,692 P36740 INFO --- 591/591 batches finished ---
2020-04-11 04:28:46,733 P36740 INFO Train loss: 0.444771
2020-04-11 04:28:46,734 P36740 INFO ************ Epoch=30 end ************
2020-04-11 04:34:48,106 P36740 INFO [Metrics] logloss: 0.479602 - AUC: 0.852723
2020-04-11 04:34:48,107 P36740 INFO Monitor(max) STOP: 0.373121 !
2020-04-11 04:34:48,107 P36740 INFO Reduce learning rate on plateau: 0.000010
2020-04-11 04:34:48,107 P36740 INFO --- 591/591 batches finished ---
2020-04-11 04:34:48,170 P36740 INFO Train loss: 0.439492
2020-04-11 04:34:48,170 P36740 INFO ************ Epoch=31 end ************
2020-04-11 04:40:48,089 P36740 INFO [Metrics] logloss: 0.484425 - AUC: 0.852475
2020-04-11 04:40:48,091 P36740 INFO Monitor(max) STOP: 0.368050 !
2020-04-11 04:40:48,091 P36740 INFO Reduce learning rate on plateau: 0.000001
2020-04-11 04:40:48,091 P36740 INFO Early stopping at epoch=32
2020-04-11 04:40:48,091 P36740 INFO --- 591/591 batches finished ---
2020-04-11 04:40:48,153 P36740 INFO Train loss: 0.421003
2020-04-11 04:40:48,153 P36740 INFO Training finished.
2020-04-11 04:40:48,154 P36740 INFO Load best model: /home/XXX/benchmarks/KKBox/DeepFM_kkbox/kkbox_x4_001_c5c9c6e3/DeepFM_kkbox_x4_005_4366b458_model.ckpt
2020-04-11 04:40:48,376 P36740 INFO ****** Train/validation evaluation ******
2020-04-11 04:42:28,598 P36740 INFO [Metrics] logloss: 0.379490 - AUC: 0.911731
2020-04-11 04:42:41,196 P36740 INFO [Metrics] logloss: 0.478818 - AUC: 0.852763
2020-04-11 04:42:41,283 P36740 INFO ******** Test evaluation ********
2020-04-11 04:42:41,283 P36740 INFO Loading data...
2020-04-11 04:42:41,284 P36740 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/test.h5
2020-04-11 04:42:41,350 P36740 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-04-11 04:42:41,350 P36740 INFO Loading test data done.
2020-04-11 04:42:53,826 P36740 INFO [Metrics] logloss: 0.478468 - AUC: 0.853109

```
