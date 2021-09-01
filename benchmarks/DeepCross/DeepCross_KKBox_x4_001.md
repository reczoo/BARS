## DeepCross_KKBox_x4_001

A notebook to benchmark DeepCross on KKBox_x4_001 dataset.

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
In this setting, For all categorical fields, we replace infrequent features with a default <OOV> token by setting the threshold min_category_count=10.

To make a fair comparison, we fix embedding_dim=128, which performs well.


### Code




### Results
```python
[Metrics] logloss: 0.479906 - AUC: 0.849628
```


### Logs
```python
2020-04-20 06:38:21,057 P12536 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "dataset_id": "kkbox_x4_001_c5c9c6e3",
    "dnn_activations": "relu",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.0005",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DeepCrossing",
    "model_id": "DeepCrossing_kkbox_x4_020_d6a25617",
    "model_root": "./KKBox/DeepCrossing_kkbox/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.4",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "residual_blocks": "[1000, 1000, 1000]",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "use_residual": "True",
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
2020-04-20 06:38:21,057 P12536 INFO Set up feature encoder...
2020-04-20 06:38:21,058 P12536 INFO Load feature_map from json: ../data/KKBox/kkbox_x4_001_c5c9c6e3/feature_map.json
2020-04-20 06:38:21,058 P12536 INFO Loading data...
2020-04-20 06:38:21,061 P12536 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/train.h5
2020-04-20 06:38:21,407 P12536 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/valid.h5
2020-04-20 06:38:21,614 P12536 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2020-04-20 06:38:21,634 P12536 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-04-20 06:38:21,634 P12536 INFO Loading train data done.
2020-04-20 06:38:25,752 P12536 INFO **** Start training: 591 batches/epoch ****
2020-04-20 06:39:55,741 P12536 INFO [Metrics] logloss: 0.553170 - AUC: 0.790024
2020-04-20 06:39:55,752 P12536 INFO Save best model: monitor(max): 0.236853
2020-04-20 06:39:55,820 P12536 INFO --- 591/591 batches finished ---
2020-04-20 06:39:55,878 P12536 INFO Train loss: 0.609594
2020-04-20 06:39:55,879 P12536 INFO ************ Epoch=1 end ************
2020-04-20 06:41:26,050 P12536 INFO [Metrics] logloss: 0.540293 - AUC: 0.802961
2020-04-20 06:41:26,062 P12536 INFO Save best model: monitor(max): 0.262668
2020-04-20 06:41:26,164 P12536 INFO --- 591/591 batches finished ---
2020-04-20 06:41:26,224 P12536 INFO Train loss: 0.591221
2020-04-20 06:41:26,224 P12536 INFO ************ Epoch=2 end ************
2020-04-20 06:42:56,037 P12536 INFO [Metrics] logloss: 0.532356 - AUC: 0.809201
2020-04-20 06:42:56,050 P12536 INFO Save best model: monitor(max): 0.276845
2020-04-20 06:42:56,167 P12536 INFO --- 591/591 batches finished ---
2020-04-20 06:42:56,223 P12536 INFO Train loss: 0.583883
2020-04-20 06:42:56,223 P12536 INFO ************ Epoch=3 end ************
2020-04-20 06:44:26,771 P12536 INFO [Metrics] logloss: 0.528871 - AUC: 0.813545
2020-04-20 06:44:26,782 P12536 INFO Save best model: monitor(max): 0.284674
2020-04-20 06:44:26,917 P12536 INFO --- 591/591 batches finished ---
2020-04-20 06:44:26,960 P12536 INFO Train loss: 0.579448
2020-04-20 06:44:26,960 P12536 INFO ************ Epoch=4 end ************
2020-04-20 06:45:57,482 P12536 INFO [Metrics] logloss: 0.524173 - AUC: 0.817522
2020-04-20 06:45:57,499 P12536 INFO Save best model: monitor(max): 0.293348
2020-04-20 06:45:57,632 P12536 INFO --- 591/591 batches finished ---
2020-04-20 06:45:57,675 P12536 INFO Train loss: 0.575932
2020-04-20 06:45:57,675 P12536 INFO ************ Epoch=5 end ************
2020-04-20 06:47:27,789 P12536 INFO [Metrics] logloss: 0.520344 - AUC: 0.820639
2020-04-20 06:47:27,808 P12536 INFO Save best model: monitor(max): 0.300295
2020-04-20 06:47:27,940 P12536 INFO --- 591/591 batches finished ---
2020-04-20 06:47:27,983 P12536 INFO Train loss: 0.573869
2020-04-20 06:47:27,983 P12536 INFO ************ Epoch=6 end ************
2020-04-20 06:48:58,726 P12536 INFO [Metrics] logloss: 0.517843 - AUC: 0.822311
2020-04-20 06:48:58,744 P12536 INFO Save best model: monitor(max): 0.304468
2020-04-20 06:48:58,878 P12536 INFO --- 591/591 batches finished ---
2020-04-20 06:48:58,921 P12536 INFO Train loss: 0.571461
2020-04-20 06:48:58,921 P12536 INFO ************ Epoch=7 end ************
2020-04-20 06:50:28,946 P12536 INFO [Metrics] logloss: 0.515620 - AUC: 0.824027
2020-04-20 06:50:28,960 P12536 INFO Save best model: monitor(max): 0.308406
2020-04-20 06:50:29,079 P12536 INFO --- 591/591 batches finished ---
2020-04-20 06:50:29,137 P12536 INFO Train loss: 0.569839
2020-04-20 06:50:29,137 P12536 INFO ************ Epoch=8 end ************
2020-04-20 06:51:59,549 P12536 INFO [Metrics] logloss: 0.513242 - AUC: 0.826456
2020-04-20 06:51:59,564 P12536 INFO Save best model: monitor(max): 0.313214
2020-04-20 06:51:59,679 P12536 INFO --- 591/591 batches finished ---
2020-04-20 06:51:59,731 P12536 INFO Train loss: 0.568119
2020-04-20 06:51:59,731 P12536 INFO ************ Epoch=9 end ************
2020-04-20 06:53:30,050 P12536 INFO [Metrics] logloss: 0.514223 - AUC: 0.827500
2020-04-20 06:53:30,066 P12536 INFO Save best model: monitor(max): 0.313277
2020-04-20 06:53:30,181 P12536 INFO --- 591/591 batches finished ---
2020-04-20 06:53:30,236 P12536 INFO Train loss: 0.566637
2020-04-20 06:53:30,236 P12536 INFO ************ Epoch=10 end ************
2020-04-20 06:55:01,517 P12536 INFO [Metrics] logloss: 0.513310 - AUC: 0.828940
2020-04-20 06:55:01,537 P12536 INFO Save best model: monitor(max): 0.315630
2020-04-20 06:55:01,667 P12536 INFO --- 591/591 batches finished ---
2020-04-20 06:55:01,710 P12536 INFO Train loss: 0.565640
2020-04-20 06:55:01,710 P12536 INFO ************ Epoch=11 end ************
2020-04-20 06:56:31,761 P12536 INFO [Metrics] logloss: 0.509535 - AUC: 0.829345
2020-04-20 06:56:31,777 P12536 INFO Save best model: monitor(max): 0.319810
2020-04-20 06:56:31,899 P12536 INFO --- 591/591 batches finished ---
2020-04-20 06:56:31,943 P12536 INFO Train loss: 0.564497
2020-04-20 06:56:31,943 P12536 INFO ************ Epoch=12 end ************
2020-04-20 06:58:02,387 P12536 INFO [Metrics] logloss: 0.509238 - AUC: 0.830240
2020-04-20 06:58:02,400 P12536 INFO Save best model: monitor(max): 0.321002
2020-04-20 06:58:02,517 P12536 INFO --- 591/591 batches finished ---
2020-04-20 06:58:02,570 P12536 INFO Train loss: 0.563463
2020-04-20 06:58:02,571 P12536 INFO ************ Epoch=13 end ************
2020-04-20 06:59:32,924 P12536 INFO [Metrics] logloss: 0.505804 - AUC: 0.831218
2020-04-20 06:59:32,935 P12536 INFO Save best model: monitor(max): 0.325414
2020-04-20 06:59:33,045 P12536 INFO --- 591/591 batches finished ---
2020-04-20 06:59:33,106 P12536 INFO Train loss: 0.562584
2020-04-20 06:59:33,106 P12536 INFO ************ Epoch=14 end ************
2020-04-20 07:01:03,874 P12536 INFO [Metrics] logloss: 0.507898 - AUC: 0.832107
2020-04-20 07:01:03,892 P12536 INFO Monitor(max) STOP: 0.324208 !
2020-04-20 07:01:03,892 P12536 INFO Reduce learning rate on plateau: 0.000100
2020-04-20 07:01:03,892 P12536 INFO --- 591/591 batches finished ---
2020-04-20 07:01:03,936 P12536 INFO Train loss: 0.561348
2020-04-20 07:01:03,937 P12536 INFO ************ Epoch=15 end ************
2020-04-20 07:02:34,580 P12536 INFO [Metrics] logloss: 0.484900 - AUC: 0.845799
2020-04-20 07:02:34,599 P12536 INFO Save best model: monitor(max): 0.360899
2020-04-20 07:02:34,712 P12536 INFO --- 591/591 batches finished ---
2020-04-20 07:02:34,760 P12536 INFO Train loss: 0.504040
2020-04-20 07:02:34,760 P12536 INFO ************ Epoch=16 end ************
2020-04-20 07:04:04,949 P12536 INFO [Metrics] logloss: 0.481202 - AUC: 0.848488
2020-04-20 07:04:04,974 P12536 INFO Save best model: monitor(max): 0.367286
2020-04-20 07:04:05,089 P12536 INFO --- 591/591 batches finished ---
2020-04-20 07:04:05,153 P12536 INFO Train loss: 0.475680
2020-04-20 07:04:05,154 P12536 INFO ************ Epoch=17 end ************
2020-04-20 07:05:36,441 P12536 INFO [Metrics] logloss: 0.480315 - AUC: 0.849426
2020-04-20 07:05:36,455 P12536 INFO Save best model: monitor(max): 0.369111
2020-04-20 07:05:36,566 P12536 INFO --- 591/591 batches finished ---
2020-04-20 07:05:36,616 P12536 INFO Train loss: 0.463227
2020-04-20 07:05:36,616 P12536 INFO ************ Epoch=18 end ************
2020-04-20 07:07:07,695 P12536 INFO [Metrics] logloss: 0.481756 - AUC: 0.849268
2020-04-20 07:07:07,714 P12536 INFO Monitor(max) STOP: 0.367512 !
2020-04-20 07:07:07,714 P12536 INFO Reduce learning rate on plateau: 0.000010
2020-04-20 07:07:07,714 P12536 INFO --- 591/591 batches finished ---
2020-04-20 07:07:07,787 P12536 INFO Train loss: 0.453737
2020-04-20 07:07:07,787 P12536 INFO ************ Epoch=19 end ************
2020-04-20 07:08:38,311 P12536 INFO [Metrics] logloss: 0.493066 - AUC: 0.847947
2020-04-20 07:08:38,324 P12536 INFO Monitor(max) STOP: 0.354881 !
2020-04-20 07:08:38,324 P12536 INFO Reduce learning rate on plateau: 0.000001
2020-04-20 07:08:38,324 P12536 INFO Early stopping at epoch=20
2020-04-20 07:08:38,324 P12536 INFO --- 591/591 batches finished ---
2020-04-20 07:08:38,394 P12536 INFO Train loss: 0.425427
2020-04-20 07:08:38,394 P12536 INFO Training finished.
2020-04-20 07:08:38,394 P12536 INFO Load best model: /home/hispace/container/data/xxx/FuxiCTR/benchmarks/KKBox/DeepCrossing_kkbox/kkbox_x4_001_c5c9c6e3/DeepCrossing_kkbox_x4_020_d6a25617_kkbox_x4_001_c5c9c6e3_model.ckpt
2020-04-20 07:08:38,500 P12536 INFO ****** Train/validation evaluation ******
2020-04-20 07:09:30,130 P12536 INFO [Metrics] logloss: 0.406248 - AUC: 0.899586
2020-04-20 07:09:36,852 P12536 INFO [Metrics] logloss: 0.480315 - AUC: 0.849426
2020-04-20 07:09:36,953 P12536 INFO ******** Test evaluation ********
2020-04-20 07:09:36,953 P12536 INFO Loading data...
2020-04-20 07:09:36,953 P12536 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/test.h5
2020-04-20 07:09:37,091 P12536 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-04-20 07:09:37,091 P12536 INFO Loading test data done.
2020-04-20 07:09:43,872 P12536 INFO [Metrics] logloss: 0.479906 - AUC: 0.849628


```
