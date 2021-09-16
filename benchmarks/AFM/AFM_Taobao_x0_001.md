## AFM_Taobao_x0_001

A notebook to benchmark AFM on Taobao_x0_001 dataset.

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
[Metrics] logloss: 0.194126 - AUC: 0.641498
```


### Logs
```python
2020-07-03 15:12:26,246 P17264 INFO {
    "attention_dim": "32",
    "attention_dropout": "[0, 0]",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobao_x0_87391c5c",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-07",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'adgroup_id', 'pid', 'cate_id', 'campaign_id', 'customer', 'brand', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation'], 'type': 'categorical'}, {'active': False, 'dtype': 'str', 'name': 'new_user_class_level', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 64, 'name': 'click_history', 'padding': 'pre', 'share_embedding': 'adgroup_id', 'type': 'sequence'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "AFM",
    "model_id": "AFM_taobao_x0_004_551ec96c",
    "model_root": "./Taobao/AFM_taobao/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Taobao/Taobao_x0/test.csv",
    "train_data": "../data/Taobao/Taobao_x0/train.csv",
    "use_attention": "True",
    "use_hdf5": "True",
    "valid_data": "../data/Taobao/Taobao_x0/test.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-03 15:12:26,247 P17264 INFO Set up feature encoder...
2020-07-03 15:12:26,247 P17264 INFO Load feature_encoder from pickle: ../data/Taobao/taobao_x0_87391c5c/feature_encoder.pkl
2020-07-03 15:12:27,332 P17264 INFO Total number of parameters: 22990917.
2020-07-03 15:12:27,333 P17264 INFO Loading data...
2020-07-03 15:12:27,335 P17264 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/train.h5
2020-07-03 15:12:39,627 P17264 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-07-03 15:12:42,628 P17264 INFO Train samples: total/23709456, pos/1222521, neg/22486935, ratio/5.16%
2020-07-03 15:12:42,771 P17264 INFO Validation samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-07-03 15:12:42,771 P17264 INFO Loading train data done.
2020-07-03 15:12:46,332 P17264 INFO Start training: 2371 batches/epoch
2020-07-03 15:12:46,332 P17264 INFO ************ Epoch=1 start ************
2020-07-03 15:24:55,993 P17264 INFO [Metrics] logloss: 0.195773 - AUC: 0.618685
2020-07-03 15:24:55,997 P17264 INFO Save best model: monitor(max): 0.422912
2020-07-03 15:24:56,084 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 15:24:56,165 P17264 INFO Train loss: 0.215687
2020-07-03 15:24:56,165 P17264 INFO ************ Epoch=1 end ************
2020-07-03 15:36:33,883 P17264 INFO [Metrics] logloss: 0.194855 - AUC: 0.629899
2020-07-03 15:36:33,886 P17264 INFO Save best model: monitor(max): 0.435045
2020-07-03 15:36:34,025 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 15:36:34,099 P17264 INFO Train loss: 0.195455
2020-07-03 15:36:34,099 P17264 INFO ************ Epoch=2 end ************
2020-07-03 15:48:46,699 P17264 INFO [Metrics] logloss: 0.194269 - AUC: 0.633792
2020-07-03 15:48:46,702 P17264 INFO Save best model: monitor(max): 0.439523
2020-07-03 15:48:46,848 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 15:48:46,930 P17264 INFO Train loss: 0.193058
2020-07-03 15:48:46,930 P17264 INFO ************ Epoch=3 end ************
2020-07-03 16:01:01,556 P17264 INFO [Metrics] logloss: 0.194096 - AUC: 0.635937
2020-07-03 16:01:01,560 P17264 INFO Save best model: monitor(max): 0.441840
2020-07-03 16:01:01,729 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 16:01:01,812 P17264 INFO Train loss: 0.191801
2020-07-03 16:01:01,812 P17264 INFO ************ Epoch=4 end ************
2020-07-03 16:13:16,413 P17264 INFO [Metrics] logloss: 0.194119 - AUC: 0.637269
2020-07-03 16:13:16,416 P17264 INFO Save best model: monitor(max): 0.443149
2020-07-03 16:13:16,580 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 16:13:16,665 P17264 INFO Train loss: 0.191059
2020-07-03 16:13:16,665 P17264 INFO ************ Epoch=5 end ************
2020-07-03 16:24:44,751 P17264 INFO [Metrics] logloss: 0.194055 - AUC: 0.638293
2020-07-03 16:24:44,754 P17264 INFO Save best model: monitor(max): 0.444238
2020-07-03 16:24:44,903 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 16:24:44,980 P17264 INFO Train loss: 0.190576
2020-07-03 16:24:44,980 P17264 INFO ************ Epoch=6 end ************
2020-07-03 16:36:59,615 P17264 INFO [Metrics] logloss: 0.194187 - AUC: 0.639133
2020-07-03 16:36:59,619 P17264 INFO Save best model: monitor(max): 0.444946
2020-07-03 16:36:59,769 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 16:36:59,859 P17264 INFO Train loss: 0.190255
2020-07-03 16:36:59,859 P17264 INFO ************ Epoch=7 end ************
2020-07-03 16:49:10,876 P17264 INFO [Metrics] logloss: 0.194121 - AUC: 0.639835
2020-07-03 16:49:10,879 P17264 INFO Save best model: monitor(max): 0.445714
2020-07-03 16:49:11,020 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 16:49:11,101 P17264 INFO Train loss: 0.190032
2020-07-03 16:49:11,101 P17264 INFO ************ Epoch=8 end ************
2020-07-03 17:01:19,419 P17264 INFO [Metrics] logloss: 0.194102 - AUC: 0.640006
2020-07-03 17:01:19,422 P17264 INFO Save best model: monitor(max): 0.445904
2020-07-03 17:01:19,583 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 17:01:19,659 P17264 INFO Train loss: 0.189877
2020-07-03 17:01:19,659 P17264 INFO ************ Epoch=9 end ************
2020-07-03 17:13:12,911 P17264 INFO [Metrics] logloss: 0.194033 - AUC: 0.640447
2020-07-03 17:13:12,914 P17264 INFO Save best model: monitor(max): 0.446413
2020-07-03 17:13:13,056 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 17:13:13,137 P17264 INFO Train loss: 0.189766
2020-07-03 17:13:13,138 P17264 INFO ************ Epoch=10 end ************
2020-07-03 17:25:06,482 P17264 INFO [Metrics] logloss: 0.194111 - AUC: 0.640577
2020-07-03 17:25:06,485 P17264 INFO Save best model: monitor(max): 0.446467
2020-07-03 17:25:06,636 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 17:25:06,758 P17264 INFO Train loss: 0.189687
2020-07-03 17:25:06,759 P17264 INFO ************ Epoch=11 end ************
2020-07-03 17:37:19,232 P17264 INFO [Metrics] logloss: 0.194117 - AUC: 0.640755
2020-07-03 17:37:19,235 P17264 INFO Save best model: monitor(max): 0.446638
2020-07-03 17:37:19,385 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 17:37:19,467 P17264 INFO Train loss: 0.189630
2020-07-03 17:37:19,467 P17264 INFO ************ Epoch=12 end ************
2020-07-03 17:49:13,012 P17264 INFO [Metrics] logloss: 0.194148 - AUC: 0.640989
2020-07-03 17:49:13,014 P17264 INFO Save best model: monitor(max): 0.446841
2020-07-03 17:49:13,144 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 17:49:13,213 P17264 INFO Train loss: 0.189587
2020-07-03 17:49:13,213 P17264 INFO ************ Epoch=13 end ************
2020-07-03 17:55:12,328 P17264 INFO [Metrics] logloss: 0.194089 - AUC: 0.641179
2020-07-03 17:55:12,332 P17264 INFO Save best model: monitor(max): 0.447090
2020-07-03 17:55:12,495 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 17:55:12,576 P17264 INFO Train loss: 0.189559
2020-07-03 17:55:12,576 P17264 INFO ************ Epoch=14 end ************
2020-07-03 18:03:06,475 P17264 INFO [Metrics] logloss: 0.194254 - AUC: 0.641346
2020-07-03 18:03:06,479 P17264 INFO Save best model: monitor(max): 0.447092
2020-07-03 18:03:06,635 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 18:03:06,728 P17264 INFO Train loss: 0.189539
2020-07-03 18:03:06,729 P17264 INFO ************ Epoch=15 end ************
2020-07-03 18:09:43,453 P17264 INFO [Metrics] logloss: 0.194171 - AUC: 0.641219
2020-07-03 18:09:43,457 P17264 INFO Monitor(max) STOP: 0.447048 !
2020-07-03 18:09:43,457 P17264 INFO Reduce learning rate on plateau: 0.000100
2020-07-03 18:09:43,457 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 18:09:43,534 P17264 INFO Train loss: 0.189523
2020-07-03 18:09:43,534 P17264 INFO ************ Epoch=16 end ************
2020-07-03 18:17:23,709 P17264 INFO [Metrics] logloss: 0.194124 - AUC: 0.641344
2020-07-03 18:17:23,713 P17264 INFO Save best model: monitor(max): 0.447220
2020-07-03 18:17:23,879 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 18:17:23,960 P17264 INFO Train loss: 0.188259
2020-07-03 18:17:23,960 P17264 INFO ************ Epoch=17 end ************
2020-07-03 18:25:08,995 P17264 INFO [Metrics] logloss: 0.194144 - AUC: 0.641372
2020-07-03 18:25:08,998 P17264 INFO Save best model: monitor(max): 0.447229
2020-07-03 18:25:09,172 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 18:25:09,262 P17264 INFO Train loss: 0.188248
2020-07-03 18:25:09,262 P17264 INFO ************ Epoch=18 end ************
2020-07-03 18:32:54,123 P17264 INFO [Metrics] logloss: 0.194149 - AUC: 0.641398
2020-07-03 18:32:54,128 P17264 INFO Save best model: monitor(max): 0.447249
2020-07-03 18:32:54,294 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 18:32:54,374 P17264 INFO Train loss: 0.188243
2020-07-03 18:32:54,374 P17264 INFO ************ Epoch=19 end ************
2020-07-03 18:40:42,020 P17264 INFO [Metrics] logloss: 0.194135 - AUC: 0.641431
2020-07-03 18:40:42,023 P17264 INFO Save best model: monitor(max): 0.447296
2020-07-03 18:40:42,191 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 18:40:42,265 P17264 INFO Train loss: 0.188240
2020-07-03 18:40:42,265 P17264 INFO ************ Epoch=20 end ************
2020-07-03 18:47:16,580 P17264 INFO [Metrics] logloss: 0.194128 - AUC: 0.641464
2020-07-03 18:47:16,583 P17264 INFO Save best model: monitor(max): 0.447336
2020-07-03 18:47:16,727 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 18:47:16,806 P17264 INFO Train loss: 0.188238
2020-07-03 18:47:16,806 P17264 INFO ************ Epoch=21 end ************
2020-07-03 18:55:10,424 P17264 INFO [Metrics] logloss: 0.194126 - AUC: 0.641498
2020-07-03 18:55:10,427 P17264 INFO Save best model: monitor(max): 0.447371
2020-07-03 18:55:10,577 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 18:55:10,677 P17264 INFO Train loss: 0.188237
2020-07-03 18:55:10,678 P17264 INFO ************ Epoch=22 end ************
2020-07-03 19:02:53,471 P17264 INFO [Metrics] logloss: 0.194171 - AUC: 0.641477
2020-07-03 19:02:53,475 P17264 INFO Monitor(max) STOP: 0.447306 !
2020-07-03 19:02:53,475 P17264 INFO Reduce learning rate on plateau: 0.000010
2020-07-03 19:02:53,475 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 19:02:53,551 P17264 INFO Train loss: 0.188236
2020-07-03 19:02:53,551 P17264 INFO ************ Epoch=23 end ************
2020-07-03 19:10:37,479 P17264 INFO [Metrics] logloss: 0.194134 - AUC: 0.641472
2020-07-03 19:10:37,483 P17264 INFO Monitor(max) STOP: 0.447339 !
2020-07-03 19:10:37,483 P17264 INFO Reduce learning rate on plateau: 0.000001
2020-07-03 19:10:37,483 P17264 INFO Early stopping at epoch=24
2020-07-03 19:10:37,483 P17264 INFO --- 2371/2371 batches finished ---
2020-07-03 19:10:37,567 P17264 INFO Train loss: 0.188097
2020-07-03 19:10:37,567 P17264 INFO Training finished.
2020-07-03 19:10:37,567 P17264 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Taobao/AFM_taobao/taobao_x0_87391c5c/AFM_taobao_x0_004_551ec96c_model.ckpt
2020-07-03 19:10:37,721 P17264 INFO ****** Train/validation evaluation ******
2020-07-03 19:11:01,000 P17264 INFO [Metrics] logloss: 0.194126 - AUC: 0.641498
2020-07-03 19:11:01,051 P17264 INFO ******** Test evaluation ********
2020-07-03 19:11:01,051 P17264 INFO Loading data...
2020-07-03 19:11:01,051 P17264 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-07-03 19:11:02,033 P17264 INFO Test samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-07-03 19:11:02,033 P17264 INFO Loading test data done.
2020-07-03 19:11:25,872 P17264 INFO [Metrics] logloss: 0.194126 - AUC: 0.641498

```
