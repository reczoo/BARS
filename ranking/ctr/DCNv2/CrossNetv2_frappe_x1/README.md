## CrossNetv2_frappe_x1

A hands-on guide to run the DCNv2 model on the Frappe_x1 dataset.

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
Dataset ID: [Frappe_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Frappe#Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DCNv2](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DCNv2.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CrossNetv2_frappe_x1_tuner_config_01](./CrossNetv2_frappe_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CrossNetv2_frappe_x1
    nohup python run_expid.py --config ./CrossNetv2_frappe_x1_tuner_config_01 --expid DCNv2_frappe_x1_001_881fe47a --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.972914 | 0.221563  |


### Logs
```python
2022-01-23 13:19:22,184 P8678 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "dnn_activations": "relu",
    "embedding_dim": "10",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "low_rank": "32",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCNv2",
    "model_id": "DCNv2_frappe_x1_001_881fe47a",
    "model_root": "./Frappe/DCN_frappe_x1/",
    "model_structure": "crossnet_only",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_cross_layers": "4",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_dnn_hidden_units": "[500, 500, 500]",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "stacked_dnn_hidden_units": "[500, 500, 500]",
    "task": "binary_classification",
    "test_data": "../data/Frappe/Frappe_x1/test.csv",
    "train_data": "../data/Frappe/Frappe_x1/train.csv",
    "use_hdf5": "True",
    "use_low_rank_mixture": "False",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-23 13:19:22,185 P8678 INFO Set up feature encoder...
2022-01-23 13:19:22,185 P8678 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-01-23 13:19:22,185 P8678 INFO Loading data...
2022-01-23 13:19:22,189 P8678 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-01-23 13:19:22,202 P8678 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-01-23 13:19:22,211 P8678 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-01-23 13:19:22,211 P8678 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-01-23 13:19:22,214 P8678 INFO Loading train data done.
2022-01-23 13:19:26,999 P8678 INFO Total number of parameters: 94391.
2022-01-23 13:19:27,000 P8678 INFO Start training: 50 batches/epoch
2022-01-23 13:19:27,000 P8678 INFO ************ Epoch=1 start ************
2022-01-23 13:19:35,819 P8678 INFO [Metrics] AUC: 0.910902 - logloss: 0.580078
2022-01-23 13:19:35,819 P8678 INFO Save best model: monitor(max): 0.910902
2022-01-23 13:19:35,822 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:19:35,862 P8678 INFO Train loss: 0.643431
2022-01-23 13:19:35,863 P8678 INFO ************ Epoch=1 end ************
2022-01-23 13:19:44,689 P8678 INFO [Metrics] AUC: 0.935373 - logloss: 0.293460
2022-01-23 13:19:44,689 P8678 INFO Save best model: monitor(max): 0.935373
2022-01-23 13:19:44,691 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:19:44,734 P8678 INFO Train loss: 0.456181
2022-01-23 13:19:44,734 P8678 INFO ************ Epoch=2 end ************
2022-01-23 13:19:53,503 P8678 INFO [Metrics] AUC: 0.941552 - logloss: 0.277782
2022-01-23 13:19:53,503 P8678 INFO Save best model: monitor(max): 0.941552
2022-01-23 13:19:53,505 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:19:53,549 P8678 INFO Train loss: 0.303605
2022-01-23 13:19:53,549 P8678 INFO ************ Epoch=3 end ************
2022-01-23 13:20:02,092 P8678 INFO [Metrics] AUC: 0.942362 - logloss: 0.275842
2022-01-23 13:20:02,093 P8678 INFO Save best model: monitor(max): 0.942362
2022-01-23 13:20:02,096 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:20:02,142 P8678 INFO Train loss: 0.290690
2022-01-23 13:20:02,142 P8678 INFO ************ Epoch=4 end ************
2022-01-23 13:20:10,696 P8678 INFO [Metrics] AUC: 0.942442 - logloss: 0.275535
2022-01-23 13:20:10,697 P8678 INFO Save best model: monitor(max): 0.942442
2022-01-23 13:20:10,699 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:20:10,739 P8678 INFO Train loss: 0.285734
2022-01-23 13:20:10,739 P8678 INFO ************ Epoch=5 end ************
2022-01-23 13:20:19,353 P8678 INFO [Metrics] AUC: 0.942545 - logloss: 0.275352
2022-01-23 13:20:19,354 P8678 INFO Save best model: monitor(max): 0.942545
2022-01-23 13:20:19,356 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:20:19,398 P8678 INFO Train loss: 0.282796
2022-01-23 13:20:19,398 P8678 INFO ************ Epoch=6 end ************
2022-01-23 13:20:27,249 P8678 INFO [Metrics] AUC: 0.942597 - logloss: 0.275241
2022-01-23 13:20:27,250 P8678 INFO Save best model: monitor(max): 0.942597
2022-01-23 13:20:27,252 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:20:27,304 P8678 INFO Train loss: 0.281243
2022-01-23 13:20:27,304 P8678 INFO ************ Epoch=7 end ************
2022-01-23 13:20:35,903 P8678 INFO [Metrics] AUC: 0.942633 - logloss: 0.274992
2022-01-23 13:20:35,903 P8678 INFO Save best model: monitor(max): 0.942633
2022-01-23 13:20:35,906 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:20:35,948 P8678 INFO Train loss: 0.279661
2022-01-23 13:20:35,948 P8678 INFO ************ Epoch=8 end ************
2022-01-23 13:20:44,625 P8678 INFO [Metrics] AUC: 0.942707 - logloss: 0.275076
2022-01-23 13:20:44,626 P8678 INFO Save best model: monitor(max): 0.942707
2022-01-23 13:20:44,628 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:20:44,671 P8678 INFO Train loss: 0.278146
2022-01-23 13:20:44,672 P8678 INFO ************ Epoch=9 end ************
2022-01-23 13:20:53,227 P8678 INFO [Metrics] AUC: 0.942910 - logloss: 0.274590
2022-01-23 13:20:53,227 P8678 INFO Save best model: monitor(max): 0.942910
2022-01-23 13:20:53,230 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:20:53,271 P8678 INFO Train loss: 0.277274
2022-01-23 13:20:53,271 P8678 INFO ************ Epoch=10 end ************
2022-01-23 13:21:01,930 P8678 INFO [Metrics] AUC: 0.943264 - logloss: 0.273870
2022-01-23 13:21:01,930 P8678 INFO Save best model: monitor(max): 0.943264
2022-01-23 13:21:01,932 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:21:01,972 P8678 INFO Train loss: 0.275808
2022-01-23 13:21:01,973 P8678 INFO ************ Epoch=11 end ************
2022-01-23 13:21:10,703 P8678 INFO [Metrics] AUC: 0.943466 - logloss: 0.273368
2022-01-23 13:21:10,704 P8678 INFO Save best model: monitor(max): 0.943466
2022-01-23 13:21:10,706 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:21:10,752 P8678 INFO Train loss: 0.275251
2022-01-23 13:21:10,752 P8678 INFO ************ Epoch=12 end ************
2022-01-23 13:21:19,323 P8678 INFO [Metrics] AUC: 0.943678 - logloss: 0.273141
2022-01-23 13:21:19,323 P8678 INFO Save best model: monitor(max): 0.943678
2022-01-23 13:21:19,326 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:21:19,367 P8678 INFO Train loss: 0.273837
2022-01-23 13:21:19,367 P8678 INFO ************ Epoch=13 end ************
2022-01-23 13:21:27,977 P8678 INFO [Metrics] AUC: 0.943941 - logloss: 0.273005
2022-01-23 13:21:27,977 P8678 INFO Save best model: monitor(max): 0.943941
2022-01-23 13:21:27,979 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:21:28,022 P8678 INFO Train loss: 0.272797
2022-01-23 13:21:28,022 P8678 INFO ************ Epoch=14 end ************
2022-01-23 13:21:37,001 P8678 INFO [Metrics] AUC: 0.944051 - logloss: 0.273526
2022-01-23 13:21:37,001 P8678 INFO Save best model: monitor(max): 0.944051
2022-01-23 13:21:37,003 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:21:37,051 P8678 INFO Train loss: 0.272302
2022-01-23 13:21:37,051 P8678 INFO ************ Epoch=15 end ************
2022-01-23 13:21:45,789 P8678 INFO [Metrics] AUC: 0.944449 - logloss: 0.272320
2022-01-23 13:21:45,789 P8678 INFO Save best model: monitor(max): 0.944449
2022-01-23 13:21:45,791 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:21:45,835 P8678 INFO Train loss: 0.271684
2022-01-23 13:21:45,836 P8678 INFO ************ Epoch=16 end ************
2022-01-23 13:21:54,471 P8678 INFO [Metrics] AUC: 0.944505 - logloss: 0.272419
2022-01-23 13:21:54,472 P8678 INFO Save best model: monitor(max): 0.944505
2022-01-23 13:21:54,474 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:21:54,515 P8678 INFO Train loss: 0.270771
2022-01-23 13:21:54,515 P8678 INFO ************ Epoch=17 end ************
2022-01-23 13:22:03,148 P8678 INFO [Metrics] AUC: 0.944715 - logloss: 0.271987
2022-01-23 13:22:03,149 P8678 INFO Save best model: monitor(max): 0.944715
2022-01-23 13:22:03,151 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:22:03,188 P8678 INFO Train loss: 0.269926
2022-01-23 13:22:03,189 P8678 INFO ************ Epoch=18 end ************
2022-01-23 13:22:11,851 P8678 INFO [Metrics] AUC: 0.945263 - logloss: 0.270840
2022-01-23 13:22:11,851 P8678 INFO Save best model: monitor(max): 0.945263
2022-01-23 13:22:11,853 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:22:11,896 P8678 INFO Train loss: 0.268580
2022-01-23 13:22:11,896 P8678 INFO ************ Epoch=19 end ************
2022-01-23 13:22:19,976 P8678 INFO [Metrics] AUC: 0.945617 - logloss: 0.269775
2022-01-23 13:22:19,976 P8678 INFO Save best model: monitor(max): 0.945617
2022-01-23 13:22:19,978 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:22:20,020 P8678 INFO Train loss: 0.266955
2022-01-23 13:22:20,020 P8678 INFO ************ Epoch=20 end ************
2022-01-23 13:22:28,408 P8678 INFO [Metrics] AUC: 0.946799 - logloss: 0.267475
2022-01-23 13:22:28,409 P8678 INFO Save best model: monitor(max): 0.946799
2022-01-23 13:22:28,411 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:22:28,457 P8678 INFO Train loss: 0.265011
2022-01-23 13:22:28,457 P8678 INFO ************ Epoch=21 end ************
2022-01-23 13:22:37,075 P8678 INFO [Metrics] AUC: 0.948013 - logloss: 0.264647
2022-01-23 13:22:37,075 P8678 INFO Save best model: monitor(max): 0.948013
2022-01-23 13:22:37,077 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:22:37,119 P8678 INFO Train loss: 0.261987
2022-01-23 13:22:37,119 P8678 INFO ************ Epoch=22 end ************
2022-01-23 13:22:45,680 P8678 INFO [Metrics] AUC: 0.950066 - logloss: 0.259121
2022-01-23 13:22:45,680 P8678 INFO Save best model: monitor(max): 0.950066
2022-01-23 13:22:45,683 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:22:45,725 P8678 INFO Train loss: 0.256648
2022-01-23 13:22:45,725 P8678 INFO ************ Epoch=23 end ************
2022-01-23 13:22:54,465 P8678 INFO [Metrics] AUC: 0.952018 - logloss: 0.254717
2022-01-23 13:22:54,465 P8678 INFO Save best model: monitor(max): 0.952018
2022-01-23 13:22:54,468 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:22:54,520 P8678 INFO Train loss: 0.249451
2022-01-23 13:22:54,521 P8678 INFO ************ Epoch=24 end ************
2022-01-23 13:23:03,229 P8678 INFO [Metrics] AUC: 0.953701 - logloss: 0.249319
2022-01-23 13:23:03,230 P8678 INFO Save best model: monitor(max): 0.953701
2022-01-23 13:23:03,232 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:23:03,272 P8678 INFO Train loss: 0.242641
2022-01-23 13:23:03,273 P8678 INFO ************ Epoch=25 end ************
2022-01-23 13:23:12,002 P8678 INFO [Metrics] AUC: 0.955953 - logloss: 0.243181
2022-01-23 13:23:12,003 P8678 INFO Save best model: monitor(max): 0.955953
2022-01-23 13:23:12,005 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:23:12,047 P8678 INFO Train loss: 0.235340
2022-01-23 13:23:12,047 P8678 INFO ************ Epoch=26 end ************
2022-01-23 13:23:20,834 P8678 INFO [Metrics] AUC: 0.958437 - logloss: 0.237520
2022-01-23 13:23:20,834 P8678 INFO Save best model: monitor(max): 0.958437
2022-01-23 13:23:20,836 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:23:20,882 P8678 INFO Train loss: 0.226483
2022-01-23 13:23:20,882 P8678 INFO ************ Epoch=27 end ************
2022-01-23 13:23:29,475 P8678 INFO [Metrics] AUC: 0.959698 - logloss: 0.234134
2022-01-23 13:23:29,475 P8678 INFO Save best model: monitor(max): 0.959698
2022-01-23 13:23:29,477 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:23:29,524 P8678 INFO Train loss: 0.219893
2022-01-23 13:23:29,524 P8678 INFO ************ Epoch=28 end ************
2022-01-23 13:23:38,141 P8678 INFO [Metrics] AUC: 0.960520 - logloss: 0.233627
2022-01-23 13:23:38,142 P8678 INFO Save best model: monitor(max): 0.960520
2022-01-23 13:23:38,144 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:23:38,191 P8678 INFO Train loss: 0.215770
2022-01-23 13:23:38,191 P8678 INFO ************ Epoch=29 end ************
2022-01-23 13:23:46,781 P8678 INFO [Metrics] AUC: 0.961112 - logloss: 0.232027
2022-01-23 13:23:46,781 P8678 INFO Save best model: monitor(max): 0.961112
2022-01-23 13:23:46,784 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:23:46,826 P8678 INFO Train loss: 0.212484
2022-01-23 13:23:46,826 P8678 INFO ************ Epoch=30 end ************
2022-01-23 13:23:55,400 P8678 INFO [Metrics] AUC: 0.961619 - logloss: 0.231297
2022-01-23 13:23:55,400 P8678 INFO Save best model: monitor(max): 0.961619
2022-01-23 13:23:55,402 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:23:55,443 P8678 INFO Train loss: 0.209636
2022-01-23 13:23:55,443 P8678 INFO ************ Epoch=31 end ************
2022-01-23 13:24:04,064 P8678 INFO [Metrics] AUC: 0.962707 - logloss: 0.227940
2022-01-23 13:24:04,064 P8678 INFO Save best model: monitor(max): 0.962707
2022-01-23 13:24:04,066 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:24:04,108 P8678 INFO Train loss: 0.205696
2022-01-23 13:24:04,108 P8678 INFO ************ Epoch=32 end ************
2022-01-23 13:24:12,946 P8678 INFO [Metrics] AUC: 0.964619 - logloss: 0.223376
2022-01-23 13:24:12,946 P8678 INFO Save best model: monitor(max): 0.964619
2022-01-23 13:24:12,948 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:24:12,989 P8678 INFO Train loss: 0.199004
2022-01-23 13:24:12,990 P8678 INFO ************ Epoch=33 end ************
2022-01-23 13:24:21,529 P8678 INFO [Metrics] AUC: 0.966698 - logloss: 0.216419
2022-01-23 13:24:21,530 P8678 INFO Save best model: monitor(max): 0.966698
2022-01-23 13:24:21,532 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:24:21,572 P8678 INFO Train loss: 0.188986
2022-01-23 13:24:21,574 P8678 INFO ************ Epoch=34 end ************
2022-01-23 13:24:29,924 P8678 INFO [Metrics] AUC: 0.968299 - logloss: 0.212921
2022-01-23 13:24:29,924 P8678 INFO Save best model: monitor(max): 0.968299
2022-01-23 13:24:29,926 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:24:29,968 P8678 INFO Train loss: 0.179676
2022-01-23 13:24:29,969 P8678 INFO ************ Epoch=35 end ************
2022-01-23 13:24:37,584 P8678 INFO [Metrics] AUC: 0.969302 - logloss: 0.211691
2022-01-23 13:24:37,585 P8678 INFO Save best model: monitor(max): 0.969302
2022-01-23 13:24:37,587 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:24:37,631 P8678 INFO Train loss: 0.171686
2022-01-23 13:24:37,631 P8678 INFO ************ Epoch=36 end ************
2022-01-23 13:24:45,313 P8678 INFO [Metrics] AUC: 0.970297 - logloss: 0.208445
2022-01-23 13:24:45,314 P8678 INFO Save best model: monitor(max): 0.970297
2022-01-23 13:24:45,316 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:24:45,358 P8678 INFO Train loss: 0.165703
2022-01-23 13:24:45,358 P8678 INFO ************ Epoch=37 end ************
2022-01-23 13:24:53,003 P8678 INFO [Metrics] AUC: 0.971446 - logloss: 0.206009
2022-01-23 13:24:53,004 P8678 INFO Save best model: monitor(max): 0.971446
2022-01-23 13:24:53,006 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:24:53,046 P8678 INFO Train loss: 0.159737
2022-01-23 13:24:53,047 P8678 INFO ************ Epoch=38 end ************
2022-01-23 13:25:00,870 P8678 INFO [Metrics] AUC: 0.971682 - logloss: 0.207063
2022-01-23 13:25:00,871 P8678 INFO Save best model: monitor(max): 0.971682
2022-01-23 13:25:00,873 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:25:00,918 P8678 INFO Train loss: 0.154849
2022-01-23 13:25:00,919 P8678 INFO ************ Epoch=39 end ************
2022-01-23 13:25:08,586 P8678 INFO [Metrics] AUC: 0.972221 - logloss: 0.208214
2022-01-23 13:25:08,586 P8678 INFO Save best model: monitor(max): 0.972221
2022-01-23 13:25:08,588 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:25:08,632 P8678 INFO Train loss: 0.149439
2022-01-23 13:25:08,632 P8678 INFO ************ Epoch=40 end ************
2022-01-23 13:25:16,461 P8678 INFO [Metrics] AUC: 0.972362 - logloss: 0.207262
2022-01-23 13:25:16,461 P8678 INFO Save best model: monitor(max): 0.972362
2022-01-23 13:25:16,464 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:25:16,505 P8678 INFO Train loss: 0.145370
2022-01-23 13:25:16,505 P8678 INFO ************ Epoch=41 end ************
2022-01-23 13:25:24,318 P8678 INFO [Metrics] AUC: 0.972966 - logloss: 0.208922
2022-01-23 13:25:24,318 P8678 INFO Save best model: monitor(max): 0.972966
2022-01-23 13:25:24,320 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:25:24,361 P8678 INFO Train loss: 0.141250
2022-01-23 13:25:24,362 P8678 INFO ************ Epoch=42 end ************
2022-01-23 13:25:32,167 P8678 INFO [Metrics] AUC: 0.973035 - logloss: 0.209804
2022-01-23 13:25:32,167 P8678 INFO Save best model: monitor(max): 0.973035
2022-01-23 13:25:32,169 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:25:32,205 P8678 INFO Train loss: 0.138384
2022-01-23 13:25:32,206 P8678 INFO ************ Epoch=43 end ************
2022-01-23 13:25:40,032 P8678 INFO [Metrics] AUC: 0.973162 - logloss: 0.211314
2022-01-23 13:25:40,033 P8678 INFO Save best model: monitor(max): 0.973162
2022-01-23 13:25:40,035 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:25:40,077 P8678 INFO Train loss: 0.135973
2022-01-23 13:25:40,077 P8678 INFO ************ Epoch=44 end ************
2022-01-23 13:25:47,686 P8678 INFO [Metrics] AUC: 0.973071 - logloss: 0.213931
2022-01-23 13:25:47,687 P8678 INFO Monitor(max) STOP: 0.973071 !
2022-01-23 13:25:47,687 P8678 INFO Reduce learning rate on plateau: 0.000100
2022-01-23 13:25:47,687 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:25:47,730 P8678 INFO Train loss: 0.133771
2022-01-23 13:25:47,730 P8678 INFO ************ Epoch=45 end ************
2022-01-23 13:25:55,460 P8678 INFO [Metrics] AUC: 0.973546 - logloss: 0.216095
2022-01-23 13:25:55,461 P8678 INFO Save best model: monitor(max): 0.973546
2022-01-23 13:25:55,463 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:25:55,506 P8678 INFO Train loss: 0.116788
2022-01-23 13:25:55,506 P8678 INFO ************ Epoch=46 end ************
2022-01-23 13:26:03,078 P8678 INFO [Metrics] AUC: 0.973658 - logloss: 0.217769
2022-01-23 13:26:03,079 P8678 INFO Save best model: monitor(max): 0.973658
2022-01-23 13:26:03,081 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:26:03,124 P8678 INFO Train loss: 0.113894
2022-01-23 13:26:03,124 P8678 INFO ************ Epoch=47 end ************
2022-01-23 13:26:10,807 P8678 INFO [Metrics] AUC: 0.973630 - logloss: 0.220884
2022-01-23 13:26:10,807 P8678 INFO Monitor(max) STOP: 0.973630 !
2022-01-23 13:26:10,807 P8678 INFO Reduce learning rate on plateau: 0.000010
2022-01-23 13:26:10,807 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:26:10,849 P8678 INFO Train loss: 0.112187
2022-01-23 13:26:10,849 P8678 INFO ************ Epoch=48 end ************
2022-01-23 13:26:18,387 P8678 INFO [Metrics] AUC: 0.973633 - logloss: 0.221116
2022-01-23 13:26:18,388 P8678 INFO Monitor(max) STOP: 0.973633 !
2022-01-23 13:26:18,388 P8678 INFO Reduce learning rate on plateau: 0.000001
2022-01-23 13:26:18,388 P8678 INFO Early stopping at epoch=49
2022-01-23 13:26:18,388 P8678 INFO --- 50/50 batches finished ---
2022-01-23 13:26:18,438 P8678 INFO Train loss: 0.109833
2022-01-23 13:26:18,438 P8678 INFO Training finished.
2022-01-23 13:26:18,438 P8678 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Frappe/DCN_frappe_x1/frappe_x1_04e961e9/DCNv2_frappe_x1_001_881fe47a.model
2022-01-23 13:26:18,485 P8678 INFO ****** Validation evaluation ******
2022-01-23 13:26:18,875 P8678 INFO [Metrics] AUC: 0.973658 - logloss: 0.217769
2022-01-23 13:26:18,912 P8678 INFO ******** Test evaluation ********
2022-01-23 13:26:18,912 P8678 INFO Loading data...
2022-01-23 13:26:18,912 P8678 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-01-23 13:26:18,915 P8678 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-01-23 13:26:18,915 P8678 INFO Loading test data done.
2022-01-23 13:26:19,164 P8678 INFO [Metrics] AUC: 0.972914 - logloss: 0.221563

```
