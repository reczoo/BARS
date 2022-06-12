## EDCN_frappe_x1

A hands-on guide to run the EDCN model on the Frappe_x1 dataset.

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
  fuxictr: 1.2.1
  ```

### Dataset
Dataset ID: [Frappe_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Frappe#Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [EDCN](https://github.com/xue-pai/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/EDCN.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [EDCN_frappe_x1_tuner_config_01](./EDCN_frappe_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd EDCN_frappe_x1
    nohup python run_expid.py --config ./EDCN_frappe_x1_tuner_config_01 --expid EDCN_frappe_x1_015_4cce8ad8 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.984130 | 0.152086  |


### Logs
```python
2022-05-28 15:33:05,780 P56710 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "bridge_type": "hadamard_product",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "ReLU",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "EDCN",
    "model_id": "EDCN_frappe_x1_015_4cce8ad8",
    "model_root": "./Frappe/EDCN_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.5",
    "net_regularizer": "0",
    "num_cross_layers": "3",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "temperature": "1",
    "test_data": "../data/Frappe/Frappe_x1/test.csv",
    "train_data": "../data/Frappe/Frappe_x1/train.csv",
    "use_hdf5": "True",
    "use_regulation_module": "True",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-05-28 15:33:05,781 P56710 INFO Set up feature encoder...
2022-05-28 15:33:05,781 P56710 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-05-28 15:33:05,781 P56710 INFO Loading data...
2022-05-28 15:33:05,783 P56710 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-05-28 15:33:05,793 P56710 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-05-28 15:33:05,797 P56710 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-05-28 15:33:05,797 P56710 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-05-28 15:33:05,797 P56710 INFO Loading train data done.
2022-05-28 15:33:08,554 P56710 INFO Total number of parameters: 86351.
2022-05-28 15:33:08,554 P56710 INFO Start training: 50 batches/epoch
2022-05-28 15:33:08,555 P56710 INFO ************ Epoch=1 start ************
2022-05-28 15:33:10,521 P56710 INFO [Metrics] AUC: 0.904760 - logloss: 0.632861
2022-05-28 15:33:10,521 P56710 INFO Save best model: monitor(max): 0.904760
2022-05-28 15:33:10,525 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:10,560 P56710 INFO Train loss: 0.543212
2022-05-28 15:33:10,561 P56710 INFO ************ Epoch=1 end ************
2022-05-28 15:33:12,559 P56710 INFO [Metrics] AUC: 0.936480 - logloss: 0.403660
2022-05-28 15:33:12,560 P56710 INFO Save best model: monitor(max): 0.936480
2022-05-28 15:33:12,566 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:12,598 P56710 INFO Train loss: 0.330887
2022-05-28 15:33:12,599 P56710 INFO ************ Epoch=2 end ************
2022-05-28 15:33:14,545 P56710 INFO [Metrics] AUC: 0.946800 - logloss: 0.261522
2022-05-28 15:33:14,546 P56710 INFO Save best model: monitor(max): 0.946800
2022-05-28 15:33:14,551 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:14,595 P56710 INFO Train loss: 0.307200
2022-05-28 15:33:14,596 P56710 INFO ************ Epoch=3 end ************
2022-05-28 15:33:16,534 P56710 INFO [Metrics] AUC: 0.953777 - logloss: 0.244476
2022-05-28 15:33:16,534 P56710 INFO Save best model: monitor(max): 0.953777
2022-05-28 15:33:16,540 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:16,587 P56710 INFO Train loss: 0.293452
2022-05-28 15:33:16,587 P56710 INFO ************ Epoch=4 end ************
2022-05-28 15:33:18,586 P56710 INFO [Metrics] AUC: 0.958578 - logloss: 0.233492
2022-05-28 15:33:18,587 P56710 INFO Save best model: monitor(max): 0.958578
2022-05-28 15:33:18,592 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:18,625 P56710 INFO Train loss: 0.282770
2022-05-28 15:33:18,625 P56710 INFO ************ Epoch=5 end ************
2022-05-28 15:33:20,628 P56710 INFO [Metrics] AUC: 0.962441 - logloss: 0.220996
2022-05-28 15:33:20,629 P56710 INFO Save best model: monitor(max): 0.962441
2022-05-28 15:33:20,632 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:20,675 P56710 INFO Train loss: 0.275407
2022-05-28 15:33:20,675 P56710 INFO ************ Epoch=6 end ************
2022-05-28 15:33:22,886 P56710 INFO [Metrics] AUC: 0.966566 - logloss: 0.210555
2022-05-28 15:33:22,886 P56710 INFO Save best model: monitor(max): 0.966566
2022-05-28 15:33:22,890 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:22,937 P56710 INFO Train loss: 0.267233
2022-05-28 15:33:22,938 P56710 INFO ************ Epoch=7 end ************
2022-05-28 15:33:24,900 P56710 INFO [Metrics] AUC: 0.967735 - logloss: 0.204167
2022-05-28 15:33:24,900 P56710 INFO Save best model: monitor(max): 0.967735
2022-05-28 15:33:24,904 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:24,949 P56710 INFO Train loss: 0.259791
2022-05-28 15:33:24,949 P56710 INFO ************ Epoch=8 end ************
2022-05-28 15:33:27,078 P56710 INFO [Metrics] AUC: 0.969862 - logloss: 0.197246
2022-05-28 15:33:27,078 P56710 INFO Save best model: monitor(max): 0.969862
2022-05-28 15:33:27,082 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:27,113 P56710 INFO Train loss: 0.255548
2022-05-28 15:33:27,113 P56710 INFO ************ Epoch=9 end ************
2022-05-28 15:33:29,023 P56710 INFO [Metrics] AUC: 0.971835 - logloss: 0.190012
2022-05-28 15:33:29,024 P56710 INFO Save best model: monitor(max): 0.971835
2022-05-28 15:33:29,030 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:29,061 P56710 INFO Train loss: 0.250697
2022-05-28 15:33:29,061 P56710 INFO ************ Epoch=10 end ************
2022-05-28 15:33:30,994 P56710 INFO [Metrics] AUC: 0.972803 - logloss: 0.184941
2022-05-28 15:33:30,994 P56710 INFO Save best model: monitor(max): 0.972803
2022-05-28 15:33:31,000 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:31,032 P56710 INFO Train loss: 0.245455
2022-05-28 15:33:31,032 P56710 INFO ************ Epoch=11 end ************
2022-05-28 15:33:32,984 P56710 INFO [Metrics] AUC: 0.972998 - logloss: 0.184675
2022-05-28 15:33:32,985 P56710 INFO Save best model: monitor(max): 0.972998
2022-05-28 15:33:32,988 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:33,021 P56710 INFO Train loss: 0.241785
2022-05-28 15:33:33,021 P56710 INFO ************ Epoch=12 end ************
2022-05-28 15:33:34,917 P56710 INFO [Metrics] AUC: 0.974092 - logloss: 0.179771
2022-05-28 15:33:34,917 P56710 INFO Save best model: monitor(max): 0.974092
2022-05-28 15:33:34,921 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:34,963 P56710 INFO Train loss: 0.239772
2022-05-28 15:33:34,963 P56710 INFO ************ Epoch=13 end ************
2022-05-28 15:33:36,876 P56710 INFO [Metrics] AUC: 0.975125 - logloss: 0.176392
2022-05-28 15:33:36,877 P56710 INFO Save best model: monitor(max): 0.975125
2022-05-28 15:33:36,881 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:36,926 P56710 INFO Train loss: 0.234879
2022-05-28 15:33:36,926 P56710 INFO ************ Epoch=14 end ************
2022-05-28 15:33:38,837 P56710 INFO [Metrics] AUC: 0.976198 - logloss: 0.173818
2022-05-28 15:33:38,838 P56710 INFO Save best model: monitor(max): 0.976198
2022-05-28 15:33:38,843 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:38,875 P56710 INFO Train loss: 0.233047
2022-05-28 15:33:38,875 P56710 INFO ************ Epoch=15 end ************
2022-05-28 15:33:40,824 P56710 INFO [Metrics] AUC: 0.976824 - logloss: 0.170106
2022-05-28 15:33:40,824 P56710 INFO Save best model: monitor(max): 0.976824
2022-05-28 15:33:40,830 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:40,877 P56710 INFO Train loss: 0.228059
2022-05-28 15:33:40,877 P56710 INFO ************ Epoch=16 end ************
2022-05-28 15:33:42,792 P56710 INFO [Metrics] AUC: 0.977309 - logloss: 0.168272
2022-05-28 15:33:42,793 P56710 INFO Save best model: monitor(max): 0.977309
2022-05-28 15:33:42,799 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:42,843 P56710 INFO Train loss: 0.225271
2022-05-28 15:33:42,843 P56710 INFO ************ Epoch=17 end ************
2022-05-28 15:33:44,799 P56710 INFO [Metrics] AUC: 0.977173 - logloss: 0.168816
2022-05-28 15:33:44,800 P56710 INFO Monitor(max) STOP: 0.977173 !
2022-05-28 15:33:44,800 P56710 INFO Reduce learning rate on plateau: 0.000100
2022-05-28 15:33:44,800 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:44,832 P56710 INFO Train loss: 0.226694
2022-05-28 15:33:44,833 P56710 INFO ************ Epoch=18 end ************
2022-05-28 15:33:46,776 P56710 INFO [Metrics] AUC: 0.979966 - logloss: 0.159673
2022-05-28 15:33:46,777 P56710 INFO Save best model: monitor(max): 0.979966
2022-05-28 15:33:46,781 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:46,812 P56710 INFO Train loss: 0.194881
2022-05-28 15:33:46,812 P56710 INFO ************ Epoch=19 end ************
2022-05-28 15:33:48,725 P56710 INFO [Metrics] AUC: 0.981467 - logloss: 0.154084
2022-05-28 15:33:48,726 P56710 INFO Save best model: monitor(max): 0.981467
2022-05-28 15:33:48,730 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:48,762 P56710 INFO Train loss: 0.172140
2022-05-28 15:33:48,762 P56710 INFO ************ Epoch=20 end ************
2022-05-28 15:33:50,679 P56710 INFO [Metrics] AUC: 0.982262 - logloss: 0.151565
2022-05-28 15:33:50,679 P56710 INFO Save best model: monitor(max): 0.982262
2022-05-28 15:33:50,685 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:50,717 P56710 INFO Train loss: 0.157371
2022-05-28 15:33:50,717 P56710 INFO ************ Epoch=21 end ************
2022-05-28 15:33:52,860 P56710 INFO [Metrics] AUC: 0.982862 - logloss: 0.150446
2022-05-28 15:33:52,861 P56710 INFO Save best model: monitor(max): 0.982862
2022-05-28 15:33:52,864 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:52,895 P56710 INFO Train loss: 0.146815
2022-05-28 15:33:52,895 P56710 INFO ************ Epoch=22 end ************
2022-05-28 15:33:54,832 P56710 INFO [Metrics] AUC: 0.983261 - logloss: 0.149337
2022-05-28 15:33:54,833 P56710 INFO Save best model: monitor(max): 0.983261
2022-05-28 15:33:54,837 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:54,868 P56710 INFO Train loss: 0.137696
2022-05-28 15:33:54,868 P56710 INFO ************ Epoch=23 end ************
2022-05-28 15:33:56,791 P56710 INFO [Metrics] AUC: 0.983619 - logloss: 0.148233
2022-05-28 15:33:56,792 P56710 INFO Save best model: monitor(max): 0.983619
2022-05-28 15:33:56,796 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:56,827 P56710 INFO Train loss: 0.131599
2022-05-28 15:33:56,827 P56710 INFO ************ Epoch=24 end ************
2022-05-28 15:33:58,722 P56710 INFO [Metrics] AUC: 0.983888 - logloss: 0.148508
2022-05-28 15:33:58,723 P56710 INFO Save best model: monitor(max): 0.983888
2022-05-28 15:33:58,726 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:33:58,757 P56710 INFO Train loss: 0.125753
2022-05-28 15:33:58,757 P56710 INFO ************ Epoch=25 end ************
2022-05-28 15:34:00,696 P56710 INFO [Metrics] AUC: 0.984052 - logloss: 0.148518
2022-05-28 15:34:00,697 P56710 INFO Save best model: monitor(max): 0.984052
2022-05-28 15:34:00,703 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:00,735 P56710 INFO Train loss: 0.120365
2022-05-28 15:34:00,735 P56710 INFO ************ Epoch=26 end ************
2022-05-28 15:34:02,657 P56710 INFO [Metrics] AUC: 0.984240 - logloss: 0.147304
2022-05-28 15:34:02,658 P56710 INFO Save best model: monitor(max): 0.984240
2022-05-28 15:34:02,662 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:02,693 P56710 INFO Train loss: 0.116479
2022-05-28 15:34:02,693 P56710 INFO ************ Epoch=27 end ************
2022-05-28 15:34:04,614 P56710 INFO [Metrics] AUC: 0.984360 - logloss: 0.147064
2022-05-28 15:34:04,615 P56710 INFO Save best model: monitor(max): 0.984360
2022-05-28 15:34:04,619 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:04,657 P56710 INFO Train loss: 0.113814
2022-05-28 15:34:04,657 P56710 INFO ************ Epoch=28 end ************
2022-05-28 15:34:06,548 P56710 INFO [Metrics] AUC: 0.984332 - logloss: 0.145941
2022-05-28 15:34:06,548 P56710 INFO Monitor(max) STOP: 0.984332 !
2022-05-28 15:34:06,549 P56710 INFO Reduce learning rate on plateau: 0.000010
2022-05-28 15:34:06,549 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:06,579 P56710 INFO Train loss: 0.111339
2022-05-28 15:34:06,579 P56710 INFO ************ Epoch=29 end ************
2022-05-28 15:34:08,544 P56710 INFO [Metrics] AUC: 0.984403 - logloss: 0.146439
2022-05-28 15:34:08,545 P56710 INFO Save best model: monitor(max): 0.984403
2022-05-28 15:34:08,551 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:08,595 P56710 INFO Train loss: 0.105538
2022-05-28 15:34:08,595 P56710 INFO ************ Epoch=30 end ************
2022-05-28 15:34:10,516 P56710 INFO [Metrics] AUC: 0.984466 - logloss: 0.147602
2022-05-28 15:34:10,517 P56710 INFO Save best model: monitor(max): 0.984466
2022-05-28 15:34:10,521 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:10,566 P56710 INFO Train loss: 0.104971
2022-05-28 15:34:10,566 P56710 INFO ************ Epoch=31 end ************
2022-05-28 15:34:12,538 P56710 INFO [Metrics] AUC: 0.984518 - logloss: 0.147192
2022-05-28 15:34:12,539 P56710 INFO Save best model: monitor(max): 0.984518
2022-05-28 15:34:12,543 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:12,588 P56710 INFO Train loss: 0.103738
2022-05-28 15:34:12,588 P56710 INFO ************ Epoch=32 end ************
2022-05-28 15:34:14,496 P56710 INFO [Metrics] AUC: 0.984567 - logloss: 0.147828
2022-05-28 15:34:14,497 P56710 INFO Save best model: monitor(max): 0.984567
2022-05-28 15:34:14,500 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:14,531 P56710 INFO Train loss: 0.102359
2022-05-28 15:34:14,531 P56710 INFO ************ Epoch=33 end ************
2022-05-28 15:34:16,453 P56710 INFO [Metrics] AUC: 0.984574 - logloss: 0.148338
2022-05-28 15:34:16,454 P56710 INFO Save best model: monitor(max): 0.984574
2022-05-28 15:34:16,460 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:16,491 P56710 INFO Train loss: 0.102097
2022-05-28 15:34:16,491 P56710 INFO ************ Epoch=34 end ************
2022-05-28 15:34:18,462 P56710 INFO [Metrics] AUC: 0.984650 - logloss: 0.147299
2022-05-28 15:34:18,463 P56710 INFO Save best model: monitor(max): 0.984650
2022-05-28 15:34:18,466 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:18,514 P56710 INFO Train loss: 0.101741
2022-05-28 15:34:18,514 P56710 INFO ************ Epoch=35 end ************
2022-05-28 15:34:20,465 P56710 INFO [Metrics] AUC: 0.984677 - logloss: 0.148013
2022-05-28 15:34:20,466 P56710 INFO Save best model: monitor(max): 0.984677
2022-05-28 15:34:20,470 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:20,501 P56710 INFO Train loss: 0.101163
2022-05-28 15:34:20,501 P56710 INFO ************ Epoch=36 end ************
2022-05-28 15:34:22,402 P56710 INFO [Metrics] AUC: 0.984693 - logloss: 0.147274
2022-05-28 15:34:22,403 P56710 INFO Save best model: monitor(max): 0.984693
2022-05-28 15:34:22,407 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:22,437 P56710 INFO Train loss: 0.101197
2022-05-28 15:34:22,437 P56710 INFO ************ Epoch=37 end ************
2022-05-28 15:34:24,356 P56710 INFO [Metrics] AUC: 0.984721 - logloss: 0.147725
2022-05-28 15:34:24,357 P56710 INFO Save best model: monitor(max): 0.984721
2022-05-28 15:34:24,363 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:24,395 P56710 INFO Train loss: 0.100341
2022-05-28 15:34:24,395 P56710 INFO ************ Epoch=38 end ************
2022-05-28 15:34:26,315 P56710 INFO [Metrics] AUC: 0.984736 - logloss: 0.148285
2022-05-28 15:34:26,316 P56710 INFO Save best model: monitor(max): 0.984736
2022-05-28 15:34:26,321 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:26,353 P56710 INFO Train loss: 0.099384
2022-05-28 15:34:26,353 P56710 INFO ************ Epoch=39 end ************
2022-05-28 15:34:28,331 P56710 INFO [Metrics] AUC: 0.984755 - logloss: 0.147686
2022-05-28 15:34:28,332 P56710 INFO Save best model: monitor(max): 0.984755
2022-05-28 15:34:28,336 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:28,368 P56710 INFO Train loss: 0.098724
2022-05-28 15:34:28,368 P56710 INFO ************ Epoch=40 end ************
2022-05-28 15:34:30,297 P56710 INFO [Metrics] AUC: 0.984779 - logloss: 0.147279
2022-05-28 15:34:30,297 P56710 INFO Save best model: monitor(max): 0.984779
2022-05-28 15:34:30,303 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:30,334 P56710 INFO Train loss: 0.097760
2022-05-28 15:34:30,334 P56710 INFO ************ Epoch=41 end ************
2022-05-28 15:34:32,259 P56710 INFO [Metrics] AUC: 0.984798 - logloss: 0.147725
2022-05-28 15:34:32,260 P56710 INFO Save best model: monitor(max): 0.984798
2022-05-28 15:34:32,264 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:32,295 P56710 INFO Train loss: 0.098834
2022-05-28 15:34:32,295 P56710 INFO ************ Epoch=42 end ************
2022-05-28 15:34:34,182 P56710 INFO [Metrics] AUC: 0.984819 - logloss: 0.148081
2022-05-28 15:34:34,183 P56710 INFO Save best model: monitor(max): 0.984819
2022-05-28 15:34:34,187 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:34,217 P56710 INFO Train loss: 0.097667
2022-05-28 15:34:34,217 P56710 INFO ************ Epoch=43 end ************
2022-05-28 15:34:36,128 P56710 INFO [Metrics] AUC: 0.984848 - logloss: 0.147282
2022-05-28 15:34:36,129 P56710 INFO Save best model: monitor(max): 0.984848
2022-05-28 15:34:36,134 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:36,165 P56710 INFO Train loss: 0.097482
2022-05-28 15:34:36,165 P56710 INFO ************ Epoch=44 end ************
2022-05-28 15:34:38,111 P56710 INFO [Metrics] AUC: 0.984826 - logloss: 0.148176
2022-05-28 15:34:38,112 P56710 INFO Monitor(max) STOP: 0.984826 !
2022-05-28 15:34:38,112 P56710 INFO Reduce learning rate on plateau: 0.000001
2022-05-28 15:34:38,112 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:38,144 P56710 INFO Train loss: 0.096324
2022-05-28 15:34:38,144 P56710 INFO ************ Epoch=45 end ************
2022-05-28 15:34:40,076 P56710 INFO [Metrics] AUC: 0.984846 - logloss: 0.147289
2022-05-28 15:34:40,077 P56710 INFO Monitor(max) STOP: 0.984846 !
2022-05-28 15:34:40,077 P56710 INFO Reduce learning rate on plateau: 0.000001
2022-05-28 15:34:40,077 P56710 INFO Early stopping at epoch=46
2022-05-28 15:34:40,078 P56710 INFO --- 50/50 batches finished ---
2022-05-28 15:34:40,108 P56710 INFO Train loss: 0.095979
2022-05-28 15:34:40,108 P56710 INFO Training finished.
2022-05-28 15:34:40,108 P56710 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Frappe/EDCN_frappe_x1/frappe_x1_04e961e9/EDCN_frappe_x1_015_4cce8ad8.model
2022-05-28 15:34:40,122 P56710 INFO ****** Validation evaluation ******
2022-05-28 15:34:40,457 P56710 INFO [Metrics] AUC: 0.984848 - logloss: 0.147282
2022-05-28 15:34:40,491 P56710 INFO ******** Test evaluation ********
2022-05-28 15:34:40,492 P56710 INFO Loading data...
2022-05-28 15:34:40,492 P56710 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-05-28 15:34:40,495 P56710 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-05-28 15:34:40,495 P56710 INFO Loading test data done.
2022-05-28 15:34:40,701 P56710 INFO [Metrics] AUC: 0.984130 - logloss: 0.152086

```
