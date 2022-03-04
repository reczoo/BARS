## AFN_frappe_x1

A hands-on guide to run the AFN model on the Frappe_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
  GPU: Tesla P100 16G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 11.4
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  fuxictr: 1.1.0
  ```

### Dataset
Dataset ID: [Frappe_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Frappe/README.md#Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [AFN](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/AFN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFN_frappe_x1_tuner_config_02](./AFN_frappe_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFN_frappe_x1
    nohup python run_expid.py --config ./AFN_frappe_x1_tuner_config_02 --expid AFN_frappe_x1_008_f15b0bf0 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.981099 | 0.233090  |


### Logs
```python
2022-01-29 21:48:38,196 P35593 INFO {
    "afn_activations": "relu",
    "afn_dropout": "0.4",
    "afn_hidden_units": "[400]",
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_dropout": "0",
    "dnn_hidden_units": "[]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.001",
    "ensemble_dnn": "False",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "logarithmic_neurons": "1000",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AFN",
    "model_id": "AFN_frappe_x1_008_f15b0bf0",
    "model_root": "./Frappe/AFN_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Frappe/Frappe_x1/test.csv",
    "train_data": "../data/Frappe/Frappe_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-29 21:48:38,197 P35593 INFO Set up feature encoder...
2022-01-29 21:48:38,197 P35593 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-01-29 21:48:38,197 P35593 INFO Loading data...
2022-01-29 21:48:38,199 P35593 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-01-29 21:48:38,211 P35593 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-01-29 21:48:38,215 P35593 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-01-29 21:48:38,215 P35593 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-01-29 21:48:38,215 P35593 INFO Loading train data done.
2022-01-29 21:48:42,171 P35593 INFO Total number of parameters: 4066711.
2022-01-29 21:48:42,171 P35593 INFO Start training: 50 batches/epoch
2022-01-29 21:48:42,172 P35593 INFO ************ Epoch=1 start ************
2022-01-29 21:48:53,361 P35593 INFO [Metrics] AUC: 0.933218 - logloss: 0.301576
2022-01-29 21:48:53,361 P35593 INFO Save best model: monitor(max): 0.933218
2022-01-29 21:48:53,379 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:48:53,421 P35593 INFO Train loss: 0.486886
2022-01-29 21:48:53,421 P35593 INFO ************ Epoch=1 end ************
2022-01-29 21:49:04,462 P35593 INFO [Metrics] AUC: 0.942921 - logloss: 0.272532
2022-01-29 21:49:04,463 P35593 INFO Save best model: monitor(max): 0.942921
2022-01-29 21:49:04,487 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:49:04,527 P35593 INFO Train loss: 0.287110
2022-01-29 21:49:04,527 P35593 INFO ************ Epoch=2 end ************
2022-01-29 21:49:15,492 P35593 INFO [Metrics] AUC: 0.949266 - logloss: 0.260289
2022-01-29 21:49:15,492 P35593 INFO Save best model: monitor(max): 0.949266
2022-01-29 21:49:15,517 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:49:15,555 P35593 INFO Train loss: 0.266011
2022-01-29 21:49:15,556 P35593 INFO ************ Epoch=3 end ************
2022-01-29 21:49:26,504 P35593 INFO [Metrics] AUC: 0.939641 - logloss: 0.314289
2022-01-29 21:49:26,504 P35593 INFO Monitor(max) STOP: 0.939641 !
2022-01-29 21:49:26,504 P35593 INFO Reduce learning rate on plateau: 0.000100
2022-01-29 21:49:26,504 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:49:26,536 P35593 INFO Train loss: 0.251805
2022-01-29 21:49:26,537 P35593 INFO ************ Epoch=4 end ************
2022-01-29 21:49:37,572 P35593 INFO [Metrics] AUC: 0.956276 - logloss: 0.253624
2022-01-29 21:49:37,573 P35593 INFO Save best model: monitor(max): 0.956276
2022-01-29 21:49:37,596 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:49:37,633 P35593 INFO Train loss: 0.239013
2022-01-29 21:49:37,633 P35593 INFO ************ Epoch=5 end ************
2022-01-29 21:49:48,492 P35593 INFO [Metrics] AUC: 0.960907 - logloss: 0.241979
2022-01-29 21:49:48,492 P35593 INFO Save best model: monitor(max): 0.960907
2022-01-29 21:49:48,515 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:49:48,561 P35593 INFO Train loss: 0.212483
2022-01-29 21:49:48,561 P35593 INFO ************ Epoch=6 end ************
2022-01-29 21:49:59,444 P35593 INFO [Metrics] AUC: 0.963784 - logloss: 0.236456
2022-01-29 21:49:59,445 P35593 INFO Save best model: monitor(max): 0.963784
2022-01-29 21:49:59,469 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:49:59,504 P35593 INFO Train loss: 0.196617
2022-01-29 21:49:59,504 P35593 INFO ************ Epoch=7 end ************
2022-01-29 21:50:10,367 P35593 INFO [Metrics] AUC: 0.966294 - logloss: 0.227085
2022-01-29 21:50:10,367 P35593 INFO Save best model: monitor(max): 0.966294
2022-01-29 21:50:10,391 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:50:10,442 P35593 INFO Train loss: 0.184285
2022-01-29 21:50:10,442 P35593 INFO ************ Epoch=8 end ************
2022-01-29 21:50:21,303 P35593 INFO [Metrics] AUC: 0.968026 - logloss: 0.224754
2022-01-29 21:50:21,303 P35593 INFO Save best model: monitor(max): 0.968026
2022-01-29 21:50:21,327 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:50:21,360 P35593 INFO Train loss: 0.173205
2022-01-29 21:50:21,360 P35593 INFO ************ Epoch=9 end ************
2022-01-29 21:50:32,234 P35593 INFO [Metrics] AUC: 0.969849 - logloss: 0.219214
2022-01-29 21:50:32,235 P35593 INFO Save best model: monitor(max): 0.969849
2022-01-29 21:50:32,260 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:50:32,293 P35593 INFO Train loss: 0.163873
2022-01-29 21:50:32,294 P35593 INFO ************ Epoch=10 end ************
2022-01-29 21:50:43,142 P35593 INFO [Metrics] AUC: 0.971127 - logloss: 0.215938
2022-01-29 21:50:43,142 P35593 INFO Save best model: monitor(max): 0.971127
2022-01-29 21:50:43,167 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:50:43,203 P35593 INFO Train loss: 0.153753
2022-01-29 21:50:43,204 P35593 INFO ************ Epoch=11 end ************
2022-01-29 21:50:54,068 P35593 INFO [Metrics] AUC: 0.972227 - logloss: 0.215049
2022-01-29 21:50:54,068 P35593 INFO Save best model: monitor(max): 0.972227
2022-01-29 21:50:54,092 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:50:54,153 P35593 INFO Train loss: 0.146080
2022-01-29 21:50:54,153 P35593 INFO ************ Epoch=12 end ************
2022-01-29 21:51:05,019 P35593 INFO [Metrics] AUC: 0.973152 - logloss: 0.215426
2022-01-29 21:51:05,019 P35593 INFO Save best model: monitor(max): 0.973152
2022-01-29 21:51:05,043 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:51:05,076 P35593 INFO Train loss: 0.139435
2022-01-29 21:51:05,076 P35593 INFO ************ Epoch=13 end ************
2022-01-29 21:51:15,937 P35593 INFO [Metrics] AUC: 0.973998 - logloss: 0.212984
2022-01-29 21:51:15,937 P35593 INFO Save best model: monitor(max): 0.973998
2022-01-29 21:51:15,967 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:51:16,008 P35593 INFO Train loss: 0.132574
2022-01-29 21:51:16,008 P35593 INFO ************ Epoch=14 end ************
2022-01-29 21:51:26,878 P35593 INFO [Metrics] AUC: 0.974636 - logloss: 0.214447
2022-01-29 21:51:26,879 P35593 INFO Save best model: monitor(max): 0.974636
2022-01-29 21:51:26,903 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:51:26,939 P35593 INFO Train loss: 0.126508
2022-01-29 21:51:26,939 P35593 INFO ************ Epoch=15 end ************
2022-01-29 21:51:37,799 P35593 INFO [Metrics] AUC: 0.975418 - logloss: 0.213245
2022-01-29 21:51:37,800 P35593 INFO Save best model: monitor(max): 0.975418
2022-01-29 21:51:37,830 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:51:37,869 P35593 INFO Train loss: 0.121768
2022-01-29 21:51:37,869 P35593 INFO ************ Epoch=16 end ************
2022-01-29 21:51:48,735 P35593 INFO [Metrics] AUC: 0.976074 - logloss: 0.211660
2022-01-29 21:51:48,735 P35593 INFO Save best model: monitor(max): 0.976074
2022-01-29 21:51:48,759 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:51:48,796 P35593 INFO Train loss: 0.115737
2022-01-29 21:51:48,796 P35593 INFO ************ Epoch=17 end ************
2022-01-29 21:51:59,690 P35593 INFO [Metrics] AUC: 0.976628 - logloss: 0.211576
2022-01-29 21:51:59,690 P35593 INFO Save best model: monitor(max): 0.976628
2022-01-29 21:51:59,715 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:51:59,748 P35593 INFO Train loss: 0.111549
2022-01-29 21:51:59,748 P35593 INFO ************ Epoch=18 end ************
2022-01-29 21:52:10,629 P35593 INFO [Metrics] AUC: 0.977144 - logloss: 0.213072
2022-01-29 21:52:10,629 P35593 INFO Save best model: monitor(max): 0.977144
2022-01-29 21:52:10,654 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:52:10,704 P35593 INFO Train loss: 0.106031
2022-01-29 21:52:10,704 P35593 INFO ************ Epoch=19 end ************
2022-01-29 21:52:21,562 P35593 INFO [Metrics] AUC: 0.977552 - logloss: 0.211107
2022-01-29 21:52:21,562 P35593 INFO Save best model: monitor(max): 0.977552
2022-01-29 21:52:21,586 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:52:21,619 P35593 INFO Train loss: 0.102756
2022-01-29 21:52:21,619 P35593 INFO ************ Epoch=20 end ************
2022-01-29 21:52:32,505 P35593 INFO [Metrics] AUC: 0.978060 - logloss: 0.212438
2022-01-29 21:52:32,506 P35593 INFO Save best model: monitor(max): 0.978060
2022-01-29 21:52:32,530 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:52:32,562 P35593 INFO Train loss: 0.098605
2022-01-29 21:52:32,562 P35593 INFO ************ Epoch=21 end ************
2022-01-29 21:52:43,497 P35593 INFO [Metrics] AUC: 0.978271 - logloss: 0.214480
2022-01-29 21:52:43,498 P35593 INFO Save best model: monitor(max): 0.978271
2022-01-29 21:52:43,524 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:52:43,561 P35593 INFO Train loss: 0.095231
2022-01-29 21:52:43,561 P35593 INFO ************ Epoch=22 end ************
2022-01-29 21:52:54,505 P35593 INFO [Metrics] AUC: 0.978702 - logloss: 0.213503
2022-01-29 21:52:54,505 P35593 INFO Save best model: monitor(max): 0.978702
2022-01-29 21:52:54,531 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:52:54,574 P35593 INFO Train loss: 0.092222
2022-01-29 21:52:54,574 P35593 INFO ************ Epoch=23 end ************
2022-01-29 21:53:05,501 P35593 INFO [Metrics] AUC: 0.978916 - logloss: 0.213343
2022-01-29 21:53:05,502 P35593 INFO Save best model: monitor(max): 0.978916
2022-01-29 21:53:05,526 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:53:05,561 P35593 INFO Train loss: 0.090099
2022-01-29 21:53:05,561 P35593 INFO ************ Epoch=24 end ************
2022-01-29 21:53:16,516 P35593 INFO [Metrics] AUC: 0.979113 - logloss: 0.214605
2022-01-29 21:53:16,516 P35593 INFO Save best model: monitor(max): 0.979113
2022-01-29 21:53:16,540 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:53:16,582 P35593 INFO Train loss: 0.086707
2022-01-29 21:53:16,582 P35593 INFO ************ Epoch=25 end ************
2022-01-29 21:53:27,564 P35593 INFO [Metrics] AUC: 0.979581 - logloss: 0.214334
2022-01-29 21:53:27,565 P35593 INFO Save best model: monitor(max): 0.979581
2022-01-29 21:53:27,588 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:53:27,622 P35593 INFO Train loss: 0.084400
2022-01-29 21:53:27,622 P35593 INFO ************ Epoch=26 end ************
2022-01-29 21:53:38,562 P35593 INFO [Metrics] AUC: 0.979722 - logloss: 0.217277
2022-01-29 21:53:38,562 P35593 INFO Save best model: monitor(max): 0.979722
2022-01-29 21:53:38,586 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:53:38,619 P35593 INFO Train loss: 0.081473
2022-01-29 21:53:38,620 P35593 INFO ************ Epoch=27 end ************
2022-01-29 21:53:49,579 P35593 INFO [Metrics] AUC: 0.979770 - logloss: 0.217570
2022-01-29 21:53:49,579 P35593 INFO Save best model: monitor(max): 0.979770
2022-01-29 21:53:49,604 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:53:49,639 P35593 INFO Train loss: 0.079451
2022-01-29 21:53:49,639 P35593 INFO ************ Epoch=28 end ************
2022-01-29 21:54:00,554 P35593 INFO [Metrics] AUC: 0.980192 - logloss: 0.219640
2022-01-29 21:54:00,554 P35593 INFO Save best model: monitor(max): 0.980192
2022-01-29 21:54:00,578 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:54:00,615 P35593 INFO Train loss: 0.077719
2022-01-29 21:54:00,615 P35593 INFO ************ Epoch=29 end ************
2022-01-29 21:54:11,483 P35593 INFO [Metrics] AUC: 0.980242 - logloss: 0.218266
2022-01-29 21:54:11,483 P35593 INFO Save best model: monitor(max): 0.980242
2022-01-29 21:54:11,507 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:54:11,546 P35593 INFO Train loss: 0.075314
2022-01-29 21:54:11,547 P35593 INFO ************ Epoch=30 end ************
2022-01-29 21:54:22,416 P35593 INFO [Metrics] AUC: 0.980527 - logloss: 0.217972
2022-01-29 21:54:22,417 P35593 INFO Save best model: monitor(max): 0.980527
2022-01-29 21:54:22,440 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:54:22,474 P35593 INFO Train loss: 0.073746
2022-01-29 21:54:22,474 P35593 INFO ************ Epoch=31 end ************
2022-01-29 21:54:33,318 P35593 INFO [Metrics] AUC: 0.980627 - logloss: 0.222095
2022-01-29 21:54:33,319 P35593 INFO Save best model: monitor(max): 0.980627
2022-01-29 21:54:33,343 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:54:33,376 P35593 INFO Train loss: 0.072575
2022-01-29 21:54:33,376 P35593 INFO ************ Epoch=32 end ************
2022-01-29 21:54:44,240 P35593 INFO [Metrics] AUC: 0.980797 - logloss: 0.222708
2022-01-29 21:54:44,240 P35593 INFO Save best model: monitor(max): 0.980797
2022-01-29 21:54:44,264 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:54:44,298 P35593 INFO Train loss: 0.071191
2022-01-29 21:54:44,299 P35593 INFO ************ Epoch=33 end ************
2022-01-29 21:54:55,177 P35593 INFO [Metrics] AUC: 0.980809 - logloss: 0.222963
2022-01-29 21:54:55,177 P35593 INFO Save best model: monitor(max): 0.980809
2022-01-29 21:54:55,202 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:54:55,235 P35593 INFO Train loss: 0.069169
2022-01-29 21:54:55,235 P35593 INFO ************ Epoch=34 end ************
2022-01-29 21:55:06,065 P35593 INFO [Metrics] AUC: 0.980827 - logloss: 0.227730
2022-01-29 21:55:06,066 P35593 INFO Save best model: monitor(max): 0.980827
2022-01-29 21:55:06,093 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:55:06,129 P35593 INFO Train loss: 0.067336
2022-01-29 21:55:06,129 P35593 INFO ************ Epoch=35 end ************
2022-01-29 21:55:16,978 P35593 INFO [Metrics] AUC: 0.981033 - logloss: 0.225469
2022-01-29 21:55:16,978 P35593 INFO Save best model: monitor(max): 0.981033
2022-01-29 21:55:17,007 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:55:17,052 P35593 INFO Train loss: 0.066382
2022-01-29 21:55:17,052 P35593 INFO ************ Epoch=36 end ************
2022-01-29 21:55:27,948 P35593 INFO [Metrics] AUC: 0.981025 - logloss: 0.229107
2022-01-29 21:55:27,948 P35593 INFO Monitor(max) STOP: 0.981025 !
2022-01-29 21:55:27,949 P35593 INFO Reduce learning rate on plateau: 0.000010
2022-01-29 21:55:27,949 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:55:27,982 P35593 INFO Train loss: 0.065404
2022-01-29 21:55:27,982 P35593 INFO ************ Epoch=37 end ************
2022-01-29 21:55:38,847 P35593 INFO [Metrics] AUC: 0.981176 - logloss: 0.228427
2022-01-29 21:55:38,847 P35593 INFO Save best model: monitor(max): 0.981176
2022-01-29 21:55:38,872 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:55:38,911 P35593 INFO Train loss: 0.060206
2022-01-29 21:55:38,911 P35593 INFO ************ Epoch=38 end ************
2022-01-29 21:55:49,737 P35593 INFO [Metrics] AUC: 0.981237 - logloss: 0.228083
2022-01-29 21:55:49,737 P35593 INFO Save best model: monitor(max): 0.981237
2022-01-29 21:55:49,763 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:55:49,797 P35593 INFO Train loss: 0.059114
2022-01-29 21:55:49,797 P35593 INFO ************ Epoch=39 end ************
2022-01-29 21:56:00,654 P35593 INFO [Metrics] AUC: 0.981275 - logloss: 0.228650
2022-01-29 21:56:00,654 P35593 INFO Save best model: monitor(max): 0.981275
2022-01-29 21:56:00,677 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:56:00,711 P35593 INFO Train loss: 0.058711
2022-01-29 21:56:00,711 P35593 INFO ************ Epoch=40 end ************
2022-01-29 21:56:11,579 P35593 INFO [Metrics] AUC: 0.981328 - logloss: 0.228853
2022-01-29 21:56:11,580 P35593 INFO Save best model: monitor(max): 0.981328
2022-01-29 21:56:11,604 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:56:11,640 P35593 INFO Train loss: 0.057532
2022-01-29 21:56:11,640 P35593 INFO ************ Epoch=41 end ************
2022-01-29 21:56:22,511 P35593 INFO [Metrics] AUC: 0.981343 - logloss: 0.228503
2022-01-29 21:56:22,512 P35593 INFO Save best model: monitor(max): 0.981343
2022-01-29 21:56:22,535 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:56:22,569 P35593 INFO Train loss: 0.057152
2022-01-29 21:56:22,569 P35593 INFO ************ Epoch=42 end ************
2022-01-29 21:56:33,410 P35593 INFO [Metrics] AUC: 0.981382 - logloss: 0.229592
2022-01-29 21:56:33,411 P35593 INFO Save best model: monitor(max): 0.981382
2022-01-29 21:56:33,435 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:56:33,469 P35593 INFO Train loss: 0.055987
2022-01-29 21:56:33,469 P35593 INFO ************ Epoch=43 end ************
2022-01-29 21:56:44,365 P35593 INFO [Metrics] AUC: 0.981380 - logloss: 0.228657
2022-01-29 21:56:44,365 P35593 INFO Monitor(max) STOP: 0.981380 !
2022-01-29 21:56:44,365 P35593 INFO Reduce learning rate on plateau: 0.000001
2022-01-29 21:56:44,365 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:56:44,422 P35593 INFO Train loss: 0.055962
2022-01-29 21:56:44,423 P35593 INFO ************ Epoch=44 end ************
2022-01-29 21:56:55,267 P35593 INFO [Metrics] AUC: 0.981403 - logloss: 0.229595
2022-01-29 21:56:55,267 P35593 INFO Save best model: monitor(max): 0.981403
2022-01-29 21:56:55,291 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:56:55,327 P35593 INFO Train loss: 0.055739
2022-01-29 21:56:55,327 P35593 INFO ************ Epoch=45 end ************
2022-01-29 21:57:06,199 P35593 INFO [Metrics] AUC: 0.981407 - logloss: 0.229170
2022-01-29 21:57:06,199 P35593 INFO Save best model: monitor(max): 0.981407
2022-01-29 21:57:06,232 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:57:06,270 P35593 INFO Train loss: 0.055488
2022-01-29 21:57:06,270 P35593 INFO ************ Epoch=46 end ************
2022-01-29 21:57:17,109 P35593 INFO [Metrics] AUC: 0.981419 - logloss: 0.229615
2022-01-29 21:57:17,110 P35593 INFO Save best model: monitor(max): 0.981419
2022-01-29 21:57:17,135 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:57:17,168 P35593 INFO Train loss: 0.055308
2022-01-29 21:57:17,168 P35593 INFO ************ Epoch=47 end ************
2022-01-29 21:57:25,147 P35593 INFO [Metrics] AUC: 0.981422 - logloss: 0.230077
2022-01-29 21:57:25,148 P35593 INFO Save best model: monitor(max): 0.981422
2022-01-29 21:57:25,172 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:57:25,209 P35593 INFO Train loss: 0.055333
2022-01-29 21:57:25,209 P35593 INFO ************ Epoch=48 end ************
2022-01-29 21:57:32,905 P35593 INFO [Metrics] AUC: 0.981421 - logloss: 0.229756
2022-01-29 21:57:32,905 P35593 INFO Monitor(max) STOP: 0.981421 !
2022-01-29 21:57:32,905 P35593 INFO Reduce learning rate on plateau: 0.000001
2022-01-29 21:57:32,905 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:57:32,938 P35593 INFO Train loss: 0.055166
2022-01-29 21:57:32,938 P35593 INFO ************ Epoch=49 end ************
2022-01-29 21:57:42,729 P35593 INFO [Metrics] AUC: 0.981422 - logloss: 0.229436
2022-01-29 21:57:42,729 P35593 INFO Monitor(max) STOP: 0.981422 !
2022-01-29 21:57:42,729 P35593 INFO Reduce learning rate on plateau: 0.000001
2022-01-29 21:57:42,729 P35593 INFO Early stopping at epoch=50
2022-01-29 21:57:42,729 P35593 INFO --- 50/50 batches finished ---
2022-01-29 21:57:42,764 P35593 INFO Train loss: 0.055272
2022-01-29 21:57:42,764 P35593 INFO Training finished.
2022-01-29 21:57:42,764 P35593 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Frappe/AFN_frappe_x1/frappe_x1_04e961e9/AFN_frappe_x1_008_f15b0bf0.model
2022-01-29 21:57:47,448 P35593 INFO ****** Validation evaluation ******
2022-01-29 21:57:48,251 P35593 INFO [Metrics] AUC: 0.981422 - logloss: 0.230077
2022-01-29 21:57:48,289 P35593 INFO ******** Test evaluation ********
2022-01-29 21:57:48,289 P35593 INFO Loading data...
2022-01-29 21:57:48,290 P35593 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-01-29 21:57:48,293 P35593 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-01-29 21:57:48,293 P35593 INFO Loading test data done.
2022-01-29 21:57:48,744 P35593 INFO [Metrics] AUC: 0.981099 - logloss: 0.233090

```
