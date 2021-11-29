## LR_Taobao_x0_001

A notebook to benchmark LR on Taobao_x0_001 dataset.

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


### Code




### Results
```python
[Metrics] logloss: 0.194103 - AUC: 0.641402
```


### Logs
```python
2020-06-26 00:25:52,357 P55150 INFO {
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobao_x0_87391c5c",
    "debug": "False",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'adgroup_id', 'pid', 'cate_id', 'campaign_id', 'customer', 'brand', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation'], 'type': 'categorical'}, {'active': False, 'dtype': 'str', 'name': 'new_user_class_level', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 64, 'name': 'click_history', 'padding': 'pre', 'share_embedding': 'adgroup_id', 'type': 'sequence'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "LR",
    "model_id": "LR_taobao_x0_004_cf07e736",
    "model_root": "./Taobao/LR_taobao/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "l2(1.e-7)",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Taobao/Taobao_x0/test.csv",
    "train_data": "../data/Taobao/Taobao_x0/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Taobao/Taobao_x0/test.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-06-26 00:25:52,358 P55150 INFO Set up feature encoder...
2020-06-26 00:25:52,358 P55150 INFO Load feature_encoder from pickle: ../data/Taobao/taobao_x0_87391c5c/feature_encoder.pkl
2020-06-26 00:25:52,796 P55150 INFO Total number of parameters: 1352373.
2020-06-26 00:25:52,796 P55150 INFO Loading data...
2020-06-26 00:25:52,798 P55150 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/train.h5
2020-06-26 00:26:00,094 P55150 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-26 00:26:02,037 P55150 INFO Train samples: total/23709456, pos/1222521, neg/22486935, ratio/5.16%
2020-06-26 00:26:02,168 P55150 INFO Validation samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-26 00:26:02,168 P55150 INFO Loading train data done.
2020-06-26 00:26:05,294 P55150 INFO Start training: 2371 batches/epoch
2020-06-26 00:26:05,294 P55150 INFO ************ Epoch=1 start ************
2020-06-26 00:30:32,172 P55150 INFO [Metrics] logloss: 0.195878 - AUC: 0.615599
2020-06-26 00:30:32,176 P55150 INFO Save best model: monitor(max): 0.419721
2020-06-26 00:30:32,183 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 00:30:32,239 P55150 INFO Train loss: 0.219458
2020-06-26 00:30:32,239 P55150 INFO ************ Epoch=1 end ************
2020-06-26 00:37:55,708 P55150 INFO [Metrics] logloss: 0.194793 - AUC: 0.629078
2020-06-26 00:37:55,712 P55150 INFO Save best model: monitor(max): 0.434285
2020-06-26 00:37:55,721 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 00:37:55,775 P55150 INFO Train loss: 0.195667
2020-06-26 00:37:55,775 P55150 INFO ************ Epoch=2 end ************
2020-06-26 00:45:00,134 P55150 INFO [Metrics] logloss: 0.194433 - AUC: 0.633583
2020-06-26 00:45:00,139 P55150 INFO Save best model: monitor(max): 0.439150
2020-06-26 00:45:00,148 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 00:45:00,198 P55150 INFO Train loss: 0.193225
2020-06-26 00:45:00,198 P55150 INFO ************ Epoch=3 end ************
2020-06-26 00:52:24,284 P55150 INFO [Metrics] logloss: 0.194201 - AUC: 0.636038
2020-06-26 00:52:24,288 P55150 INFO Save best model: monitor(max): 0.441837
2020-06-26 00:52:24,297 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 00:52:24,346 P55150 INFO Train loss: 0.191953
2020-06-26 00:52:24,346 P55150 INFO ************ Epoch=4 end ************
2020-06-26 00:59:31,808 P55150 INFO [Metrics] logloss: 0.194031 - AUC: 0.637621
2020-06-26 00:59:31,812 P55150 INFO Save best model: monitor(max): 0.443590
2020-06-26 00:59:31,820 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 00:59:31,870 P55150 INFO Train loss: 0.191206
2020-06-26 00:59:31,870 P55150 INFO ************ Epoch=5 end ************
2020-06-26 01:06:57,956 P55150 INFO [Metrics] logloss: 0.194071 - AUC: 0.638392
2020-06-26 01:06:57,960 P55150 INFO Save best model: monitor(max): 0.444320
2020-06-26 01:06:57,968 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 01:06:58,019 P55150 INFO Train loss: 0.190728
2020-06-26 01:06:58,019 P55150 INFO ************ Epoch=6 end ************
2020-06-26 01:12:52,195 P55150 INFO [Metrics] logloss: 0.194094 - AUC: 0.639341
2020-06-26 01:12:52,199 P55150 INFO Save best model: monitor(max): 0.445247
2020-06-26 01:12:52,206 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 01:12:52,262 P55150 INFO Train loss: 0.190408
2020-06-26 01:12:52,262 P55150 INFO ************ Epoch=7 end ************
2020-06-26 01:18:18,259 P55150 INFO [Metrics] logloss: 0.194038 - AUC: 0.639638
2020-06-26 01:18:18,264 P55150 INFO Save best model: monitor(max): 0.445600
2020-06-26 01:18:18,272 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 01:18:18,322 P55150 INFO Train loss: 0.190184
2020-06-26 01:18:18,323 P55150 INFO ************ Epoch=8 end ************
2020-06-26 01:22:03,080 P55150 INFO [Metrics] logloss: 0.194027 - AUC: 0.640024
2020-06-26 01:22:03,085 P55150 INFO Save best model: monitor(max): 0.445997
2020-06-26 01:22:03,093 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 01:22:03,150 P55150 INFO Train loss: 0.190025
2020-06-26 01:22:03,150 P55150 INFO ************ Epoch=9 end ************
2020-06-26 01:25:45,810 P55150 INFO [Metrics] logloss: 0.194061 - AUC: 0.640410
2020-06-26 01:25:45,815 P55150 INFO Save best model: monitor(max): 0.446349
2020-06-26 01:25:45,823 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 01:25:45,880 P55150 INFO Train loss: 0.189912
2020-06-26 01:25:45,880 P55150 INFO ************ Epoch=10 end ************
2020-06-26 01:29:26,030 P55150 INFO [Metrics] logloss: 0.194213 - AUC: 0.640593
2020-06-26 01:29:26,034 P55150 INFO Save best model: monitor(max): 0.446380
2020-06-26 01:29:26,042 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 01:29:26,098 P55150 INFO Train loss: 0.189830
2020-06-26 01:29:26,098 P55150 INFO ************ Epoch=11 end ************
2020-06-26 01:33:08,079 P55150 INFO [Metrics] logloss: 0.194061 - AUC: 0.640760
2020-06-26 01:33:08,083 P55150 INFO Save best model: monitor(max): 0.446699
2020-06-26 01:33:08,091 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 01:33:08,147 P55150 INFO Train loss: 0.189770
2020-06-26 01:33:08,147 P55150 INFO ************ Epoch=12 end ************
2020-06-26 01:36:49,720 P55150 INFO [Metrics] logloss: 0.194114 - AUC: 0.641026
2020-06-26 01:36:49,725 P55150 INFO Save best model: monitor(max): 0.446912
2020-06-26 01:36:49,733 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 01:36:49,788 P55150 INFO Train loss: 0.189726
2020-06-26 01:36:49,789 P55150 INFO ************ Epoch=13 end ************
2020-06-26 01:40:31,984 P55150 INFO [Metrics] logloss: 0.194146 - AUC: 0.641212
2020-06-26 01:40:31,988 P55150 INFO Save best model: monitor(max): 0.447066
2020-06-26 01:40:31,996 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 01:40:32,053 P55150 INFO Train loss: 0.189694
2020-06-26 01:40:32,054 P55150 INFO ************ Epoch=14 end ************
2020-06-26 01:44:12,740 P55150 INFO [Metrics] logloss: 0.194114 - AUC: 0.641180
2020-06-26 01:44:12,744 P55150 INFO Monitor(max) STOP: 0.447066 !
2020-06-26 01:44:12,744 P55150 INFO Reduce learning rate on plateau: 0.000100
2020-06-26 01:44:12,744 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 01:44:12,812 P55150 INFO Train loss: 0.189669
2020-06-26 01:44:12,812 P55150 INFO ************ Epoch=15 end ************
2020-06-26 01:47:53,243 P55150 INFO [Metrics] logloss: 0.194141 - AUC: 0.641278
2020-06-26 01:47:53,248 P55150 INFO Save best model: monitor(max): 0.447137
2020-06-26 01:47:53,258 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 01:47:53,327 P55150 INFO Train loss: 0.188414
2020-06-26 01:47:53,327 P55150 INFO ************ Epoch=16 end ************
2020-06-26 01:51:33,284 P55150 INFO [Metrics] logloss: 0.194116 - AUC: 0.641325
2020-06-26 01:51:33,289 P55150 INFO Save best model: monitor(max): 0.447209
2020-06-26 01:51:33,297 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 01:51:33,354 P55150 INFO Train loss: 0.188405
2020-06-26 01:51:33,354 P55150 INFO ************ Epoch=17 end ************
2020-06-26 01:55:13,558 P55150 INFO [Metrics] logloss: 0.194111 - AUC: 0.641363
2020-06-26 01:55:13,563 P55150 INFO Save best model: monitor(max): 0.447251
2020-06-26 01:55:13,571 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 01:55:13,628 P55150 INFO Train loss: 0.188400
2020-06-26 01:55:13,628 P55150 INFO ************ Epoch=18 end ************
2020-06-26 01:58:52,751 P55150 INFO [Metrics] logloss: 0.194087 - AUC: 0.641377
2020-06-26 01:58:52,755 P55150 INFO Save best model: monitor(max): 0.447290
2020-06-26 01:58:52,764 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 01:58:52,820 P55150 INFO Train loss: 0.188397
2020-06-26 01:58:52,821 P55150 INFO ************ Epoch=19 end ************
2020-06-26 02:02:31,419 P55150 INFO [Metrics] logloss: 0.194103 - AUC: 0.641402
2020-06-26 02:02:31,424 P55150 INFO Save best model: monitor(max): 0.447299
2020-06-26 02:02:31,432 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 02:02:31,489 P55150 INFO Train loss: 0.188395
2020-06-26 02:02:31,489 P55150 INFO ************ Epoch=20 end ************
2020-06-26 02:06:10,851 P55150 INFO [Metrics] logloss: 0.194120 - AUC: 0.641396
2020-06-26 02:06:10,855 P55150 INFO Monitor(max) STOP: 0.447276 !
2020-06-26 02:06:10,855 P55150 INFO Reduce learning rate on plateau: 0.000010
2020-06-26 02:06:10,855 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 02:06:10,905 P55150 INFO Train loss: 0.188394
2020-06-26 02:06:10,905 P55150 INFO ************ Epoch=21 end ************
2020-06-26 02:09:48,516 P55150 INFO [Metrics] logloss: 0.194114 - AUC: 0.641396
2020-06-26 02:09:48,520 P55150 INFO Monitor(max) STOP: 0.447282 !
2020-06-26 02:09:48,520 P55150 INFO Reduce learning rate on plateau: 0.000001
2020-06-26 02:09:48,520 P55150 INFO Early stopping at epoch=22
2020-06-26 02:09:48,520 P55150 INFO --- 2371/2371 batches finished ---
2020-06-26 02:09:48,576 P55150 INFO Train loss: 0.188256
2020-06-26 02:09:48,576 P55150 INFO Training finished.
2020-06-26 02:09:48,576 P55150 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Taobao/LR_taobao/taobao_x0_87391c5c/LR_taobao_x0_004_cf07e736_model.ckpt
2020-06-26 02:09:48,591 P55150 INFO ****** Train/validation evaluation ******
2020-06-26 02:10:07,263 P55150 INFO [Metrics] logloss: 0.194103 - AUC: 0.641402
2020-06-26 02:10:07,337 P55150 INFO ******** Test evaluation ********
2020-06-26 02:10:07,337 P55150 INFO Loading data...
2020-06-26 02:10:07,338 P55150 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-26 02:10:08,385 P55150 INFO Test samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-26 02:10:08,385 P55150 INFO Loading test data done.
2020-06-26 02:10:26,869 P55150 INFO [Metrics] logloss: 0.194103 - AUC: 0.641402


```
