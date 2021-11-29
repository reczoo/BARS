## HFM+_KKBox_x4_001 

A notebook to benchmark HFM+ on KKBox_x4_001 dataset.

Author: [XUEPAI Team](https://github.com/xue-pai)


### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
  RAM: 500G+
  ```
+ Software

  ```python
  python: 3.6.5
  pandas: 1.0.0
  numpy: 1.18.1
  ```

### Dataset
In this setting, For all categorical fields, we replace infrequent features with a default ``<OOV>`` token by setting the threshold min_category_count=10. 

To make a fair comparison, we fix **embedding_dim=128**, which performs well.


### Code
1. Install FuxiCTR
  
    Install FuxiCTR via `pip install fuxictr==1.0` to get all dependencies ready. Then download [the FuxiCTR repository](https://github.com/huawei-noah/benchmark/archive/53e314461c19dbc7f462b42bf0f0bfae020dc398.zip) to your local path.

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/KKBox/KKBox_x4/split_kkbox_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [HFM+_kkbox_x4_tuner_config_05.yaml](./HFM+_kkbox_x4_tuner_config_05.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/HFM+_kkbox_x4_tuner_config_05.yaml --tag 004 --gpu 0
  ```


### Results
```python
[Metrics] logloss: 0.477141 - AUC: 0.852262
```


### Logs
```python
2020-05-10 18:45:16,707 P24594 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "dataset_id": "kkbox_x4_001_c5c9c6e3",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.0005",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000, 1000]",
    "interaction_type": "circular_correlation",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "HFM",
    "model_id": "HFM_kkbox_x4_004_40337e9e",
    "model_root": "./KKBox/HFM_kkbox/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.5",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_dnn": "True",
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/KKBox/",
    "test_data": "../data/KKBox/kkbox_x4_001_c5c9c6e3/test.h5",
    "train_data": "../data/KKBox/kkbox_x4_001_c5c9c6e3/train.h5",
    "valid_data": "../data/KKBox/kkbox_x4_001_c5c9c6e3/valid.h5",
    "version": "pytorch",
    "gpu": "1"
}
2020-05-10 18:45:16,708 P24594 INFO Set up feature encoder...
2020-05-10 18:45:16,708 P24594 INFO Load feature_map from json: ../data/KKBox/kkbox_x4_001_c5c9c6e3/feature_map.json
2020-05-10 18:45:16,709 P24594 INFO Loading data...
2020-05-10 18:45:16,711 P24594 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/train.h5
2020-05-10 18:45:26,974 P24594 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/valid.h5
2020-05-10 18:45:27,239 P24594 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2020-05-10 18:45:27,258 P24594 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-05-10 18:45:27,258 P24594 INFO Loading train data done.
2020-05-10 18:45:36,776 P24594 INFO **** Start training: 591 batches/epoch ****
2020-05-10 18:50:13,225 P24594 INFO [Metrics] logloss: 0.554825 - AUC: 0.789221
2020-05-10 18:50:13,238 P24594 INFO Save best model: monitor(max): 0.234396
2020-05-10 18:50:13,323 P24594 INFO --- 591/591 batches finished ---
2020-05-10 18:50:13,390 P24594 INFO Train loss: 0.618438
2020-05-10 18:50:13,391 P24594 INFO ************ Epoch=1 end ************
2020-05-10 18:54:48,534 P24594 INFO [Metrics] logloss: 0.542574 - AUC: 0.799331
2020-05-10 18:54:48,546 P24594 INFO Save best model: monitor(max): 0.256757
2020-05-10 18:54:48,669 P24594 INFO --- 591/591 batches finished ---
2020-05-10 18:54:48,738 P24594 INFO Train loss: 0.608572
2020-05-10 18:54:48,739 P24594 INFO ************ Epoch=2 end ************
2020-05-10 18:59:23,810 P24594 INFO [Metrics] logloss: 0.535181 - AUC: 0.806589
2020-05-10 18:59:23,823 P24594 INFO Save best model: monitor(max): 0.271408
2020-05-10 18:59:23,952 P24594 INFO --- 591/591 batches finished ---
2020-05-10 18:59:24,015 P24594 INFO Train loss: 0.605923
2020-05-10 18:59:24,015 P24594 INFO ************ Epoch=3 end ************
2020-05-10 19:03:58,529 P24594 INFO [Metrics] logloss: 0.527912 - AUC: 0.812088
2020-05-10 19:03:58,541 P24594 INFO Save best model: monitor(max): 0.284176
2020-05-10 19:03:58,671 P24594 INFO --- 591/591 batches finished ---
2020-05-10 19:03:58,730 P24594 INFO Train loss: 0.603512
2020-05-10 19:03:58,731 P24594 INFO ************ Epoch=4 end ************
2020-05-10 19:08:37,526 P24594 INFO [Metrics] logloss: 0.522025 - AUC: 0.816916
2020-05-10 19:08:37,540 P24594 INFO Save best model: monitor(max): 0.294892
2020-05-10 19:08:37,676 P24594 INFO --- 591/591 batches finished ---
2020-05-10 19:08:37,754 P24594 INFO Train loss: 0.601023
2020-05-10 19:08:37,754 P24594 INFO ************ Epoch=5 end ************
2020-05-10 19:13:14,149 P24594 INFO [Metrics] logloss: 0.519263 - AUC: 0.819301
2020-05-10 19:13:14,162 P24594 INFO Save best model: monitor(max): 0.300037
2020-05-10 19:13:14,313 P24594 INFO --- 591/591 batches finished ---
2020-05-10 19:13:14,377 P24594 INFO Train loss: 0.598609
2020-05-10 19:13:14,377 P24594 INFO ************ Epoch=6 end ************
2020-05-10 19:17:49,892 P24594 INFO [Metrics] logloss: 0.517402 - AUC: 0.821124
2020-05-10 19:17:49,904 P24594 INFO Save best model: monitor(max): 0.303722
2020-05-10 19:17:50,020 P24594 INFO --- 591/591 batches finished ---
2020-05-10 19:17:50,100 P24594 INFO Train loss: 0.597310
2020-05-10 19:17:50,100 P24594 INFO ************ Epoch=7 end ************
2020-05-10 19:22:25,681 P24594 INFO [Metrics] logloss: 0.514706 - AUC: 0.822899
2020-05-10 19:22:25,695 P24594 INFO Save best model: monitor(max): 0.308192
2020-05-10 19:22:25,824 P24594 INFO --- 591/591 batches finished ---
2020-05-10 19:22:25,885 P24594 INFO Train loss: 0.596466
2020-05-10 19:22:25,885 P24594 INFO ************ Epoch=8 end ************
2020-05-10 19:27:00,234 P24594 INFO [Metrics] logloss: 0.512542 - AUC: 0.824600
2020-05-10 19:27:00,248 P24594 INFO Save best model: monitor(max): 0.312059
2020-05-10 19:27:00,372 P24594 INFO --- 591/591 batches finished ---
2020-05-10 19:27:00,445 P24594 INFO Train loss: 0.595938
2020-05-10 19:27:00,446 P24594 INFO ************ Epoch=9 end ************
2020-05-10 19:31:34,980 P24594 INFO [Metrics] logloss: 0.511540 - AUC: 0.825825
2020-05-10 19:31:34,992 P24594 INFO Save best model: monitor(max): 0.314285
2020-05-10 19:31:35,165 P24594 INFO --- 591/591 batches finished ---
2020-05-10 19:31:35,230 P24594 INFO Train loss: 0.595637
2020-05-10 19:31:35,231 P24594 INFO ************ Epoch=10 end ************
2020-05-10 19:36:10,943 P24594 INFO [Metrics] logloss: 0.511151 - AUC: 0.826420
2020-05-10 19:36:10,955 P24594 INFO Save best model: monitor(max): 0.315268
2020-05-10 19:36:11,075 P24594 INFO --- 591/591 batches finished ---
2020-05-10 19:36:11,144 P24594 INFO Train loss: 0.595583
2020-05-10 19:36:11,145 P24594 INFO ************ Epoch=11 end ************
2020-05-10 19:40:45,822 P24594 INFO [Metrics] logloss: 0.508915 - AUC: 0.827638
2020-05-10 19:40:45,834 P24594 INFO Save best model: monitor(max): 0.318723
2020-05-10 19:40:45,951 P24594 INFO --- 591/591 batches finished ---
2020-05-10 19:40:46,020 P24594 INFO Train loss: 0.595357
2020-05-10 19:40:46,020 P24594 INFO ************ Epoch=12 end ************
2020-05-10 19:45:21,376 P24594 INFO [Metrics] logloss: 0.507607 - AUC: 0.828547
2020-05-10 19:45:21,388 P24594 INFO Save best model: monitor(max): 0.320940
2020-05-10 19:45:21,513 P24594 INFO --- 591/591 batches finished ---
2020-05-10 19:45:21,579 P24594 INFO Train loss: 0.595318
2020-05-10 19:45:21,579 P24594 INFO ************ Epoch=13 end ************
2020-05-10 19:49:57,175 P24594 INFO [Metrics] logloss: 0.506603 - AUC: 0.829513
2020-05-10 19:49:57,187 P24594 INFO Save best model: monitor(max): 0.322909
2020-05-10 19:49:57,308 P24594 INFO --- 591/591 batches finished ---
2020-05-10 19:49:57,380 P24594 INFO Train loss: 0.595094
2020-05-10 19:49:57,381 P24594 INFO ************ Epoch=14 end ************
2020-05-10 19:54:31,983 P24594 INFO [Metrics] logloss: 0.505791 - AUC: 0.829924
2020-05-10 19:54:31,998 P24594 INFO Save best model: monitor(max): 0.324134
2020-05-10 19:54:32,132 P24594 INFO --- 591/591 batches finished ---
2020-05-10 19:54:32,194 P24594 INFO Train loss: 0.595011
2020-05-10 19:54:32,194 P24594 INFO ************ Epoch=15 end ************
2020-05-10 19:59:07,703 P24594 INFO [Metrics] logloss: 0.505077 - AUC: 0.830463
2020-05-10 19:59:07,715 P24594 INFO Save best model: monitor(max): 0.325386
2020-05-10 19:59:07,835 P24594 INFO --- 591/591 batches finished ---
2020-05-10 19:59:07,905 P24594 INFO Train loss: 0.595128
2020-05-10 19:59:07,905 P24594 INFO ************ Epoch=16 end ************
2020-05-10 20:03:42,747 P24594 INFO [Metrics] logloss: 0.504157 - AUC: 0.831394
2020-05-10 20:03:42,764 P24594 INFO Save best model: monitor(max): 0.327238
2020-05-10 20:03:42,916 P24594 INFO --- 591/591 batches finished ---
2020-05-10 20:03:42,979 P24594 INFO Train loss: 0.594834
2020-05-10 20:03:42,979 P24594 INFO ************ Epoch=17 end ************
2020-05-10 20:08:18,183 P24594 INFO [Metrics] logloss: 0.504035 - AUC: 0.831305
2020-05-10 20:08:18,203 P24594 INFO Save best model: monitor(max): 0.327270
2020-05-10 20:08:18,358 P24594 INFO --- 591/591 batches finished ---
2020-05-10 20:08:18,419 P24594 INFO Train loss: 0.595147
2020-05-10 20:08:18,419 P24594 INFO ************ Epoch=18 end ************
2020-05-10 20:12:52,927 P24594 INFO [Metrics] logloss: 0.503151 - AUC: 0.831995
2020-05-10 20:12:52,939 P24594 INFO Save best model: monitor(max): 0.328844
2020-05-10 20:12:53,066 P24594 INFO --- 591/591 batches finished ---
2020-05-10 20:12:53,136 P24594 INFO Train loss: 0.594801
2020-05-10 20:12:53,136 P24594 INFO ************ Epoch=19 end ************
2020-05-10 20:17:27,714 P24594 INFO [Metrics] logloss: 0.502842 - AUC: 0.832256
2020-05-10 20:17:27,726 P24594 INFO Save best model: monitor(max): 0.329414
2020-05-10 20:17:27,844 P24594 INFO --- 591/591 batches finished ---
2020-05-10 20:17:27,931 P24594 INFO Train loss: 0.594632
2020-05-10 20:17:27,932 P24594 INFO ************ Epoch=20 end ************
2020-05-10 20:22:02,385 P24594 INFO [Metrics] logloss: 0.502223 - AUC: 0.832691
2020-05-10 20:22:02,397 P24594 INFO Save best model: monitor(max): 0.330468
2020-05-10 20:22:02,511 P24594 INFO --- 591/591 batches finished ---
2020-05-10 20:22:02,590 P24594 INFO Train loss: 0.594588
2020-05-10 20:22:02,591 P24594 INFO ************ Epoch=21 end ************
2020-05-10 20:26:38,065 P24594 INFO [Metrics] logloss: 0.502012 - AUC: 0.833130
2020-05-10 20:26:38,079 P24594 INFO Save best model: monitor(max): 0.331118
2020-05-10 20:26:38,209 P24594 INFO --- 591/591 batches finished ---
2020-05-10 20:26:38,286 P24594 INFO Train loss: 0.595141
2020-05-10 20:26:38,286 P24594 INFO ************ Epoch=22 end ************
2020-05-10 20:31:13,586 P24594 INFO [Metrics] logloss: 0.501512 - AUC: 0.833435
2020-05-10 20:31:13,597 P24594 INFO Save best model: monitor(max): 0.331923
2020-05-10 20:31:13,722 P24594 INFO --- 591/591 batches finished ---
2020-05-10 20:31:13,804 P24594 INFO Train loss: 0.594543
2020-05-10 20:31:13,804 P24594 INFO ************ Epoch=23 end ************
2020-05-10 20:35:48,894 P24594 INFO [Metrics] logloss: 0.501492 - AUC: 0.833823
2020-05-10 20:35:48,909 P24594 INFO Save best model: monitor(max): 0.332330
2020-05-10 20:35:49,029 P24594 INFO --- 591/591 batches finished ---
2020-05-10 20:35:49,111 P24594 INFO Train loss: 0.594566
2020-05-10 20:35:49,111 P24594 INFO ************ Epoch=24 end ************
2020-05-10 20:40:24,220 P24594 INFO [Metrics] logloss: 0.500238 - AUC: 0.834841
2020-05-10 20:40:24,235 P24594 INFO Save best model: monitor(max): 0.334603
2020-05-10 20:40:24,378 P24594 INFO --- 591/591 batches finished ---
2020-05-10 20:40:24,450 P24594 INFO Train loss: 0.594607
2020-05-10 20:40:24,451 P24594 INFO ************ Epoch=25 end ************
2020-05-10 20:44:58,684 P24594 INFO [Metrics] logloss: 0.499675 - AUC: 0.834688
2020-05-10 20:44:58,700 P24594 INFO Save best model: monitor(max): 0.335013
2020-05-10 20:44:58,821 P24594 INFO --- 591/591 batches finished ---
2020-05-10 20:44:58,890 P24594 INFO Train loss: 0.594467
2020-05-10 20:44:58,891 P24594 INFO ************ Epoch=26 end ************
2020-05-10 20:49:33,052 P24594 INFO [Metrics] logloss: 0.499588 - AUC: 0.834788
2020-05-10 20:49:33,064 P24594 INFO Save best model: monitor(max): 0.335200
2020-05-10 20:49:33,185 P24594 INFO --- 591/591 batches finished ---
2020-05-10 20:49:33,250 P24594 INFO Train loss: 0.594880
2020-05-10 20:49:33,250 P24594 INFO ************ Epoch=27 end ************
2020-05-10 20:54:07,448 P24594 INFO [Metrics] logloss: 0.500349 - AUC: 0.834830
2020-05-10 20:54:07,460 P24594 INFO Monitor(max) STOP: 0.334482 !
2020-05-10 20:54:07,460 P24594 INFO Reduce learning rate on plateau: 0.000100
2020-05-10 20:54:07,460 P24594 INFO --- 591/591 batches finished ---
2020-05-10 20:54:07,556 P24594 INFO Train loss: 0.594856
2020-05-10 20:54:07,556 P24594 INFO ************ Epoch=28 end ************
2020-05-10 20:58:41,855 P24594 INFO [Metrics] logloss: 0.480593 - AUC: 0.849353
2020-05-10 20:58:41,871 P24594 INFO Save best model: monitor(max): 0.368760
2020-05-10 20:58:41,995 P24594 INFO --- 591/591 batches finished ---
2020-05-10 20:58:42,061 P24594 INFO Train loss: 0.520046
2020-05-10 20:58:42,061 P24594 INFO ************ Epoch=29 end ************
2020-05-10 21:03:17,118 P24594 INFO [Metrics] logloss: 0.477422 - AUC: 0.852201
2020-05-10 21:03:17,136 P24594 INFO Save best model: monitor(max): 0.374778
2020-05-10 21:03:17,275 P24594 INFO --- 591/591 batches finished ---
2020-05-10 21:03:17,336 P24594 INFO Train loss: 0.476371
2020-05-10 21:03:17,336 P24594 INFO ************ Epoch=30 end ************
2020-05-10 21:07:51,866 P24594 INFO [Metrics] logloss: 0.480093 - AUC: 0.852234
2020-05-10 21:07:51,879 P24594 INFO Monitor(max) STOP: 0.372141 !
2020-05-10 21:07:51,879 P24594 INFO Reduce learning rate on plateau: 0.000010
2020-05-10 21:07:51,880 P24594 INFO --- 591/591 batches finished ---
2020-05-10 21:07:51,950 P24594 INFO Train loss: 0.456513
2020-05-10 21:07:51,950 P24594 INFO ************ Epoch=31 end ************
2020-05-10 21:12:26,035 P24594 INFO [Metrics] logloss: 0.495797 - AUC: 0.850553
2020-05-10 21:12:26,050 P24594 INFO Monitor(max) STOP: 0.354756 !
2020-05-10 21:12:26,050 P24594 INFO Reduce learning rate on plateau: 0.000001
2020-05-10 21:12:26,050 P24594 INFO Early stopping at epoch=32
2020-05-10 21:12:26,050 P24594 INFO --- 591/591 batches finished ---
2020-05-10 21:12:26,145 P24594 INFO Train loss: 0.420009
2020-05-10 21:12:26,145 P24594 INFO Training finished.
2020-05-10 21:12:26,145 P24594 INFO Load best model: /home/hispace/container/data/xxx/FuxiCTR/benchmarks/KKBox/HFM_kkbox/kkbox_x4_001_c5c9c6e3/HFM_kkbox_x4_004_40337e9e_kkbox_x4_001_c5c9c6e3_model.ckpt
2020-05-10 21:12:26,346 P24594 INFO ****** Train/validation evaluation ******
2020-05-10 21:14:02,135 P24594 INFO [Metrics] logloss: 0.384181 - AUC: 0.911535
2020-05-10 21:14:14,017 P24594 INFO [Metrics] logloss: 0.477422 - AUC: 0.852201
2020-05-10 21:14:14,134 P24594 INFO ******** Test evaluation ********
2020-05-10 21:14:14,134 P24594 INFO Loading data...
2020-05-10 21:14:14,134 P24594 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/test.h5
2020-05-10 21:14:14,215 P24594 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-05-10 21:14:14,215 P24594 INFO Loading test data done.
2020-05-10 21:14:25,590 P24594 INFO [Metrics] logloss: 0.477141 - AUC: 0.852262
```
