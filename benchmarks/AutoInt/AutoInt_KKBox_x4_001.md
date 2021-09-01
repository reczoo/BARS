## AutoInt_KKBox_x4_001

A notebook to benchmark AutoInt on KKBox_x4_001 dataset.

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
[Metrics] logloss: 0.491948 - AUC: 0.843641
```


### Logs
```python
2020-05-08 12:05:35,558 P1143 INFO {
    "attention_dim": "256",
    "attention_layers": "5",
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x4_001_c5c9c6e3",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.0005",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AutoInt",
    "model_id": "AutoInt_kkbox_x4_004_b4c26cf3",
    "model_root": "./KKBox/AutoInt_kkbox/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "2",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/KKBox/kkbox_x4_001_c5c9c6e3/test.h5",
    "train_data": "../data/KKBox/kkbox_x4_001_c5c9c6e3/train.h5",
    "use_hdf5": "True",
    "use_residual": "True",
    "use_scale": "False",
    "use_wide": "False",
    "valid_data": "../data/KKBox/kkbox_x4_001_c5c9c6e3/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-05-08 12:05:35,562 P1143 INFO Set up feature encoder...
2020-05-08 12:05:35,562 P1143 INFO Load feature_map from json: ../data/KKBox/kkbox_x4_001_c5c9c6e3/feature_map.json
2020-05-08 12:05:35,563 P1143 INFO Loading data...
2020-05-08 12:05:35,568 P1143 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/train.h5
2020-05-08 12:05:36,100 P1143 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/valid.h5
2020-05-08 12:05:36,378 P1143 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2020-05-08 12:05:36,396 P1143 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-05-08 12:05:36,397 P1143 INFO Loading train data done.
2020-05-08 12:05:41,218 P1143 INFO **** Start training: 591 batches/epoch ****
2020-05-08 12:09:23,614 P1143 INFO [Metrics] logloss: 0.558937 - AUC: 0.783630
2020-05-08 12:09:23,615 P1143 INFO Save best model: monitor(max): 0.224693
2020-05-08 12:09:23,676 P1143 INFO --- 591/591 batches finished ---
2020-05-08 12:09:23,742 P1143 INFO Train loss: 0.615862
2020-05-08 12:09:23,742 P1143 INFO ************ Epoch=1 end ************
2020-05-08 12:13:04,398 P1143 INFO [Metrics] logloss: 0.547042 - AUC: 0.794349
2020-05-08 12:13:04,401 P1143 INFO Save best model: monitor(max): 0.247307
2020-05-08 12:13:04,473 P1143 INFO --- 591/591 batches finished ---
2020-05-08 12:13:04,542 P1143 INFO Train loss: 0.592660
2020-05-08 12:13:04,543 P1143 INFO ************ Epoch=2 end ************
2020-05-08 12:16:44,696 P1143 INFO [Metrics] logloss: 0.538364 - AUC: 0.801984
2020-05-08 12:16:44,699 P1143 INFO Save best model: monitor(max): 0.263620
2020-05-08 12:16:44,793 P1143 INFO --- 591/591 batches finished ---
2020-05-08 12:16:44,858 P1143 INFO Train loss: 0.586217
2020-05-08 12:16:44,858 P1143 INFO ************ Epoch=3 end ************
2020-05-08 12:20:25,275 P1143 INFO [Metrics] logloss: 0.532996 - AUC: 0.807182
2020-05-08 12:20:25,277 P1143 INFO Save best model: monitor(max): 0.274187
2020-05-08 12:20:25,369 P1143 INFO --- 591/591 batches finished ---
2020-05-08 12:20:25,464 P1143 INFO Train loss: 0.582106
2020-05-08 12:20:25,464 P1143 INFO ************ Epoch=4 end ************
2020-05-08 12:24:06,315 P1143 INFO [Metrics] logloss: 0.527692 - AUC: 0.812027
2020-05-08 12:24:06,318 P1143 INFO Save best model: monitor(max): 0.284334
2020-05-08 12:24:06,392 P1143 INFO --- 591/591 batches finished ---
2020-05-08 12:24:06,461 P1143 INFO Train loss: 0.579212
2020-05-08 12:24:06,461 P1143 INFO ************ Epoch=5 end ************
2020-05-08 12:27:47,861 P1143 INFO [Metrics] logloss: 0.524193 - AUC: 0.814756
2020-05-08 12:27:47,867 P1143 INFO Save best model: monitor(max): 0.290563
2020-05-08 12:27:47,979 P1143 INFO --- 591/591 batches finished ---
2020-05-08 12:27:48,032 P1143 INFO Train loss: 0.576024
2020-05-08 12:27:48,033 P1143 INFO ************ Epoch=6 end ************
2020-05-08 12:31:29,223 P1143 INFO [Metrics] logloss: 0.522272 - AUC: 0.816466
2020-05-08 12:31:29,240 P1143 INFO Save best model: monitor(max): 0.294195
2020-05-08 12:31:29,353 P1143 INFO --- 591/591 batches finished ---
2020-05-08 12:31:29,411 P1143 INFO Train loss: 0.573326
2020-05-08 12:31:29,411 P1143 INFO ************ Epoch=7 end ************
2020-05-08 12:35:09,999 P1143 INFO [Metrics] logloss: 0.519071 - AUC: 0.818976
2020-05-08 12:35:10,005 P1143 INFO Save best model: monitor(max): 0.299905
2020-05-08 12:35:10,124 P1143 INFO --- 591/591 batches finished ---
2020-05-08 12:35:10,189 P1143 INFO Train loss: 0.571319
2020-05-08 12:35:10,189 P1143 INFO ************ Epoch=8 end ************
2020-05-08 12:38:50,972 P1143 INFO [Metrics] logloss: 0.516839 - AUC: 0.820684
2020-05-08 12:38:50,984 P1143 INFO Save best model: monitor(max): 0.303845
2020-05-08 12:38:51,092 P1143 INFO --- 591/591 batches finished ---
2020-05-08 12:38:51,162 P1143 INFO Train loss: 0.569412
2020-05-08 12:38:51,163 P1143 INFO ************ Epoch=9 end ************
2020-05-08 12:42:31,628 P1143 INFO [Metrics] logloss: 0.515002 - AUC: 0.822172
2020-05-08 12:42:31,630 P1143 INFO Save best model: monitor(max): 0.307170
2020-05-08 12:42:31,716 P1143 INFO --- 591/591 batches finished ---
2020-05-08 12:42:31,784 P1143 INFO Train loss: 0.567751
2020-05-08 12:42:31,784 P1143 INFO ************ Epoch=10 end ************
2020-05-08 12:46:11,639 P1143 INFO [Metrics] logloss: 0.513697 - AUC: 0.823322
2020-05-08 12:46:11,642 P1143 INFO Save best model: monitor(max): 0.309625
2020-05-08 12:46:11,729 P1143 INFO --- 591/591 batches finished ---
2020-05-08 12:46:11,800 P1143 INFO Train loss: 0.566308
2020-05-08 12:46:11,800 P1143 INFO ************ Epoch=11 end ************
2020-05-08 12:49:52,102 P1143 INFO [Metrics] logloss: 0.511810 - AUC: 0.824877
2020-05-08 12:49:52,105 P1143 INFO Save best model: monitor(max): 0.313066
2020-05-08 12:49:52,181 P1143 INFO --- 591/591 batches finished ---
2020-05-08 12:49:52,246 P1143 INFO Train loss: 0.565079
2020-05-08 12:49:52,246 P1143 INFO ************ Epoch=12 end ************
2020-05-08 12:53:32,011 P1143 INFO [Metrics] logloss: 0.511717 - AUC: 0.825359
2020-05-08 12:53:32,013 P1143 INFO Save best model: monitor(max): 0.313641
2020-05-08 12:53:32,089 P1143 INFO --- 591/591 batches finished ---
2020-05-08 12:53:32,158 P1143 INFO Train loss: 0.563369
2020-05-08 12:53:32,158 P1143 INFO ************ Epoch=13 end ************
2020-05-08 12:57:11,747 P1143 INFO [Metrics] logloss: 0.510415 - AUC: 0.826172
2020-05-08 12:57:11,749 P1143 INFO Save best model: monitor(max): 0.315757
2020-05-08 12:57:11,824 P1143 INFO --- 591/591 batches finished ---
2020-05-08 12:57:11,886 P1143 INFO Train loss: 0.562240
2020-05-08 12:57:11,886 P1143 INFO ************ Epoch=14 end ************
2020-05-08 13:00:51,724 P1143 INFO [Metrics] logloss: 0.508758 - AUC: 0.827289
2020-05-08 13:00:51,726 P1143 INFO Save best model: monitor(max): 0.318531
2020-05-08 13:00:51,802 P1143 INFO --- 591/591 batches finished ---
2020-05-08 13:00:51,871 P1143 INFO Train loss: 0.561082
2020-05-08 13:00:51,871 P1143 INFO ************ Epoch=15 end ************
2020-05-08 13:04:32,333 P1143 INFO [Metrics] logloss: 0.507794 - AUC: 0.828138
2020-05-08 13:04:32,336 P1143 INFO Save best model: monitor(max): 0.320345
2020-05-08 13:04:32,430 P1143 INFO --- 591/591 batches finished ---
2020-05-08 13:04:32,491 P1143 INFO Train loss: 0.559770
2020-05-08 13:04:32,492 P1143 INFO ************ Epoch=16 end ************
2020-05-08 13:08:12,306 P1143 INFO [Metrics] logloss: 0.507010 - AUC: 0.828703
2020-05-08 13:08:12,313 P1143 INFO Save best model: monitor(max): 0.321693
2020-05-08 13:08:12,416 P1143 INFO --- 591/591 batches finished ---
2020-05-08 13:08:12,480 P1143 INFO Train loss: 0.558677
2020-05-08 13:08:12,480 P1143 INFO ************ Epoch=17 end ************
2020-05-08 13:11:52,239 P1143 INFO [Metrics] logloss: 0.506101 - AUC: 0.829928
2020-05-08 13:11:52,244 P1143 INFO Save best model: monitor(max): 0.323827
2020-05-08 13:11:52,345 P1143 INFO --- 591/591 batches finished ---
2020-05-08 13:11:52,401 P1143 INFO Train loss: 0.557749
2020-05-08 13:11:52,401 P1143 INFO ************ Epoch=18 end ************
2020-05-08 13:15:32,614 P1143 INFO [Metrics] logloss: 0.505928 - AUC: 0.829598
2020-05-08 13:15:32,619 P1143 INFO Monitor(max) STOP: 0.323670 !
2020-05-08 13:15:32,619 P1143 INFO Reduce learning rate on plateau: 0.000100
2020-05-08 13:15:32,619 P1143 INFO --- 591/591 batches finished ---
2020-05-08 13:15:32,682 P1143 INFO Train loss: 0.556634
2020-05-08 13:15:32,683 P1143 INFO ************ Epoch=19 end ************
2020-05-08 13:19:13,123 P1143 INFO [Metrics] logloss: 0.492161 - AUC: 0.841986
2020-05-08 13:19:13,124 P1143 INFO Save best model: monitor(max): 0.349825
2020-05-08 13:19:13,228 P1143 INFO --- 591/591 batches finished ---
2020-05-08 13:19:13,296 P1143 INFO Train loss: 0.502724
2020-05-08 13:19:13,296 P1143 INFO ************ Epoch=20 end ************
2020-05-08 13:22:53,591 P1143 INFO [Metrics] logloss: 0.492257 - AUC: 0.843571
2020-05-08 13:22:53,597 P1143 INFO Save best model: monitor(max): 0.351315
2020-05-08 13:22:53,692 P1143 INFO --- 591/591 batches finished ---
2020-05-08 13:22:53,746 P1143 INFO Train loss: 0.475451
2020-05-08 13:22:53,747 P1143 INFO ************ Epoch=21 end ************
2020-05-08 13:26:33,720 P1143 INFO [Metrics] logloss: 0.496608 - AUC: 0.843058
2020-05-08 13:26:33,734 P1143 INFO Monitor(max) STOP: 0.346449 !
2020-05-08 13:26:33,734 P1143 INFO Reduce learning rate on plateau: 0.000010
2020-05-08 13:26:33,734 P1143 INFO --- 591/591 batches finished ---
2020-05-08 13:26:33,802 P1143 INFO Train loss: 0.461290
2020-05-08 13:26:33,802 P1143 INFO ************ Epoch=22 end ************
2020-05-08 13:30:13,941 P1143 INFO [Metrics] logloss: 0.515784 - AUC: 0.839828
2020-05-08 13:30:13,942 P1143 INFO Monitor(max) STOP: 0.324044 !
2020-05-08 13:30:13,942 P1143 INFO Reduce learning rate on plateau: 0.000001
2020-05-08 13:30:13,942 P1143 INFO Early stopping at epoch=23
2020-05-08 13:30:13,942 P1143 INFO --- 591/591 batches finished ---
2020-05-08 13:30:14,008 P1143 INFO Train loss: 0.430168
2020-05-08 13:30:14,008 P1143 INFO Training finished.
2020-05-08 13:30:14,009 P1143 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/KKBox/AutoInt_kkbox/kkbox_x4_001_c5c9c6e3/AutoInt_kkbox_x4_004_b4c26cf3_model.ckpt
2020-05-08 13:30:14,119 P1143 INFO ****** Train/validation evaluation ******
2020-05-08 13:31:28,831 P1143 INFO [Metrics] logloss: 0.414526 - AUC: 0.891056
2020-05-08 13:31:38,184 P1143 INFO [Metrics] logloss: 0.492257 - AUC: 0.843571
2020-05-08 13:31:38,301 P1143 INFO ******** Test evaluation ********
2020-05-08 13:31:38,301 P1143 INFO Loading data...
2020-05-08 13:31:38,301 P1143 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/test.h5
2020-05-08 13:31:38,383 P1143 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-05-08 13:31:38,383 P1143 INFO Loading test data done.
2020-05-08 13:31:47,861 P1143 INFO [Metrics] logloss: 0.491948 - AUC: 0.843641



```
