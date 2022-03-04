## FM_frappe_x1

A hands-on guide to run the FM model on the Frappe_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [FM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FM_frappe_x1_tuner_config_03](./FM_frappe_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FM_frappe_x1
    nohup python run_expid.py --config ./FM_frappe_x1_tuner_config_03 --expid FM_frappe_x1_001_6234f1e3 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.967108 | 0.206515  |


### Logs
```python
2021-01-07 16:51:31,761 P2689 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_7f91d67a",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FM",
    "model_id": "FM_frappe_x1_001_b16e1650",
    "model_root": "./Frappe/FM_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "5e-06",
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
2021-01-07 16:51:31,762 P2689 INFO Set up feature encoder...
2021-01-07 16:51:31,763 P2689 INFO Load feature_encoder from pickle: ../data/Frappe/frappe_x1_7f91d67a/feature_encoder.pkl
2021-01-07 16:51:31,789 P2689 INFO Total number of parameters: 59280.
2021-01-07 16:51:31,789 P2689 INFO Loading data...
2021-01-07 16:51:31,792 P2689 INFO Loading data from h5: ../data/Frappe/frappe_x1_7f91d67a/train.h5
2021-01-07 16:51:31,806 P2689 INFO Loading data from h5: ../data/Frappe/frappe_x1_7f91d67a/valid.h5
2021-01-07 16:51:31,810 P2689 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2021-01-07 16:51:31,810 P2689 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2021-01-07 16:51:31,811 P2689 INFO Loading train data done.
2021-01-07 16:51:34,846 P2689 INFO Start training: 50 batches/epoch
2021-01-07 16:51:34,847 P2689 INFO ************ Epoch=1 start ************
2021-01-07 16:51:38,466 P2689 INFO [Metrics] AUC: 0.888030 - logloss: 0.620988
2021-01-07 16:51:38,467 P2689 INFO Save best model: monitor(max): 0.888030
2021-01-07 16:51:38,470 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:51:38,514 P2689 INFO Train loss: 0.656666
2021-01-07 16:51:38,514 P2689 INFO ************ Epoch=1 end ************
2021-01-07 16:51:42,093 P2689 INFO [Metrics] AUC: 0.902473 - logloss: 0.532536
2021-01-07 16:51:42,093 P2689 INFO Save best model: monitor(max): 0.902473
2021-01-07 16:51:42,096 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:51:42,138 P2689 INFO Train loss: 0.580623
2021-01-07 16:51:42,138 P2689 INFO ************ Epoch=2 end ************
2021-01-07 16:51:45,944 P2689 INFO [Metrics] AUC: 0.918769 - logloss: 0.436947
2021-01-07 16:51:45,945 P2689 INFO Save best model: monitor(max): 0.918769
2021-01-07 16:51:45,948 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:51:45,993 P2689 INFO Train loss: 0.482765
2021-01-07 16:51:45,994 P2689 INFO ************ Epoch=3 end ************
2021-01-07 16:51:49,767 P2689 INFO [Metrics] AUC: 0.927271 - logloss: 0.367722
2021-01-07 16:51:49,767 P2689 INFO Save best model: monitor(max): 0.927271
2021-01-07 16:51:49,770 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:51:49,819 P2689 INFO Train loss: 0.397320
2021-01-07 16:51:49,819 P2689 INFO ************ Epoch=4 end ************
2021-01-07 16:51:53,500 P2689 INFO [Metrics] AUC: 0.931416 - logloss: 0.328548
2021-01-07 16:51:53,500 P2689 INFO Save best model: monitor(max): 0.931416
2021-01-07 16:51:53,503 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:51:53,555 P2689 INFO Train loss: 0.342278
2021-01-07 16:51:53,555 P2689 INFO ************ Epoch=5 end ************
2021-01-07 16:51:57,251 P2689 INFO [Metrics] AUC: 0.933437 - logloss: 0.307933
2021-01-07 16:51:57,251 P2689 INFO Save best model: monitor(max): 0.933437
2021-01-07 16:51:57,255 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:51:57,303 P2689 INFO Train loss: 0.312211
2021-01-07 16:51:57,303 P2689 INFO ************ Epoch=6 end ************
2021-01-07 16:52:01,059 P2689 INFO [Metrics] AUC: 0.934320 - logloss: 0.297369
2021-01-07 16:52:01,060 P2689 INFO Save best model: monitor(max): 0.934320
2021-01-07 16:52:01,062 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:52:01,115 P2689 INFO Train loss: 0.295920
2021-01-07 16:52:01,116 P2689 INFO ************ Epoch=7 end ************
2021-01-07 16:52:04,943 P2689 INFO [Metrics] AUC: 0.935027 - logloss: 0.291662
2021-01-07 16:52:04,943 P2689 INFO Save best model: monitor(max): 0.935027
2021-01-07 16:52:04,949 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:52:04,994 P2689 INFO Train loss: 0.286826
2021-01-07 16:52:04,994 P2689 INFO ************ Epoch=8 end ************
2021-01-07 16:52:08,684 P2689 INFO [Metrics] AUC: 0.935731 - logloss: 0.288183
2021-01-07 16:52:08,685 P2689 INFO Save best model: monitor(max): 0.935731
2021-01-07 16:52:08,687 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:52:08,735 P2689 INFO Train loss: 0.281429
2021-01-07 16:52:08,735 P2689 INFO ************ Epoch=9 end ************
2021-01-07 16:52:12,406 P2689 INFO [Metrics] AUC: 0.935902 - logloss: 0.286174
2021-01-07 16:52:12,407 P2689 INFO Save best model: monitor(max): 0.935902
2021-01-07 16:52:12,409 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:52:12,458 P2689 INFO Train loss: 0.277606
2021-01-07 16:52:12,458 P2689 INFO ************ Epoch=10 end ************
2021-01-07 16:52:16,352 P2689 INFO [Metrics] AUC: 0.936096 - logloss: 0.284998
2021-01-07 16:52:16,353 P2689 INFO Save best model: monitor(max): 0.936096
2021-01-07 16:52:16,355 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:52:16,404 P2689 INFO Train loss: 0.275437
2021-01-07 16:52:16,404 P2689 INFO ************ Epoch=11 end ************
2021-01-07 16:52:20,087 P2689 INFO [Metrics] AUC: 0.936263 - logloss: 0.284348
2021-01-07 16:52:20,087 P2689 INFO Save best model: monitor(max): 0.936263
2021-01-07 16:52:20,089 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:52:20,137 P2689 INFO Train loss: 0.273303
2021-01-07 16:52:20,137 P2689 INFO ************ Epoch=12 end ************
2021-01-07 16:52:23,971 P2689 INFO [Metrics] AUC: 0.936328 - logloss: 0.283541
2021-01-07 16:52:23,972 P2689 INFO Save best model: monitor(max): 0.936328
2021-01-07 16:52:23,974 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:52:24,026 P2689 INFO Train loss: 0.271585
2021-01-07 16:52:24,026 P2689 INFO ************ Epoch=13 end ************
2021-01-07 16:52:27,910 P2689 INFO [Metrics] AUC: 0.936486 - logloss: 0.283132
2021-01-07 16:52:27,910 P2689 INFO Save best model: monitor(max): 0.936486
2021-01-07 16:52:27,912 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:52:27,960 P2689 INFO Train loss: 0.270647
2021-01-07 16:52:27,960 P2689 INFO ************ Epoch=14 end ************
2021-01-07 16:52:31,728 P2689 INFO [Metrics] AUC: 0.936515 - logloss: 0.283058
2021-01-07 16:52:31,729 P2689 INFO Save best model: monitor(max): 0.936515
2021-01-07 16:52:31,731 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:52:31,782 P2689 INFO Train loss: 0.269325
2021-01-07 16:52:31,782 P2689 INFO ************ Epoch=15 end ************
2021-01-07 16:52:35,525 P2689 INFO [Metrics] AUC: 0.936731 - logloss: 0.282327
2021-01-07 16:52:35,526 P2689 INFO Save best model: monitor(max): 0.936731
2021-01-07 16:52:35,531 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:52:35,579 P2689 INFO Train loss: 0.268314
2021-01-07 16:52:35,579 P2689 INFO ************ Epoch=16 end ************
2021-01-07 16:52:39,309 P2689 INFO [Metrics] AUC: 0.936869 - logloss: 0.281839
2021-01-07 16:52:39,310 P2689 INFO Save best model: monitor(max): 0.936869
2021-01-07 16:52:39,312 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:52:39,354 P2689 INFO Train loss: 0.266944
2021-01-07 16:52:39,354 P2689 INFO ************ Epoch=17 end ************
2021-01-07 16:52:42,954 P2689 INFO [Metrics] AUC: 0.937127 - logloss: 0.281368
2021-01-07 16:52:42,955 P2689 INFO Save best model: monitor(max): 0.937127
2021-01-07 16:52:42,957 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:52:43,002 P2689 INFO Train loss: 0.266168
2021-01-07 16:52:43,002 P2689 INFO ************ Epoch=18 end ************
2021-01-07 16:52:46,564 P2689 INFO [Metrics] AUC: 0.937369 - logloss: 0.281079
2021-01-07 16:52:46,565 P2689 INFO Save best model: monitor(max): 0.937369
2021-01-07 16:52:46,567 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:52:46,613 P2689 INFO Train loss: 0.265122
2021-01-07 16:52:46,613 P2689 INFO ************ Epoch=19 end ************
2021-01-07 16:52:50,084 P2689 INFO [Metrics] AUC: 0.937589 - logloss: 0.280054
2021-01-07 16:52:50,085 P2689 INFO Save best model: monitor(max): 0.937589
2021-01-07 16:52:50,087 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:52:50,142 P2689 INFO Train loss: 0.263379
2021-01-07 16:52:50,142 P2689 INFO ************ Epoch=20 end ************
2021-01-07 16:52:53,681 P2689 INFO [Metrics] AUC: 0.937867 - logloss: 0.279539
2021-01-07 16:52:53,681 P2689 INFO Save best model: monitor(max): 0.937867
2021-01-07 16:52:53,684 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:52:53,737 P2689 INFO Train loss: 0.262501
2021-01-07 16:52:53,737 P2689 INFO ************ Epoch=21 end ************
2021-01-07 16:52:57,415 P2689 INFO [Metrics] AUC: 0.938228 - logloss: 0.278623
2021-01-07 16:52:57,416 P2689 INFO Save best model: monitor(max): 0.938228
2021-01-07 16:52:57,418 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:52:57,469 P2689 INFO Train loss: 0.261277
2021-01-07 16:52:57,469 P2689 INFO ************ Epoch=22 end ************
2021-01-07 16:53:01,120 P2689 INFO [Metrics] AUC: 0.938592 - logloss: 0.277740
2021-01-07 16:53:01,120 P2689 INFO Save best model: monitor(max): 0.938592
2021-01-07 16:53:01,123 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:53:01,168 P2689 INFO Train loss: 0.260026
2021-01-07 16:53:01,169 P2689 INFO ************ Epoch=23 end ************
2021-01-07 16:53:04,739 P2689 INFO [Metrics] AUC: 0.939011 - logloss: 0.276623
2021-01-07 16:53:04,740 P2689 INFO Save best model: monitor(max): 0.939011
2021-01-07 16:53:04,742 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:53:04,791 P2689 INFO Train loss: 0.258418
2021-01-07 16:53:04,791 P2689 INFO ************ Epoch=24 end ************
2021-01-07 16:53:08,456 P2689 INFO [Metrics] AUC: 0.939395 - logloss: 0.275832
2021-01-07 16:53:08,456 P2689 INFO Save best model: monitor(max): 0.939395
2021-01-07 16:53:08,459 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:53:08,509 P2689 INFO Train loss: 0.257200
2021-01-07 16:53:08,509 P2689 INFO ************ Epoch=25 end ************
2021-01-07 16:53:12,131 P2689 INFO [Metrics] AUC: 0.939926 - logloss: 0.274588
2021-01-07 16:53:12,131 P2689 INFO Save best model: monitor(max): 0.939926
2021-01-07 16:53:12,134 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:53:12,175 P2689 INFO Train loss: 0.255449
2021-01-07 16:53:12,175 P2689 INFO ************ Epoch=26 end ************
2021-01-07 16:53:15,771 P2689 INFO [Metrics] AUC: 0.940507 - logloss: 0.273272
2021-01-07 16:53:15,772 P2689 INFO Save best model: monitor(max): 0.940507
2021-01-07 16:53:15,774 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:53:15,825 P2689 INFO Train loss: 0.253471
2021-01-07 16:53:15,825 P2689 INFO ************ Epoch=27 end ************
2021-01-07 16:53:19,408 P2689 INFO [Metrics] AUC: 0.940995 - logloss: 0.272321
2021-01-07 16:53:19,409 P2689 INFO Save best model: monitor(max): 0.940995
2021-01-07 16:53:19,412 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:53:19,463 P2689 INFO Train loss: 0.252055
2021-01-07 16:53:19,463 P2689 INFO ************ Epoch=28 end ************
2021-01-07 16:53:22,979 P2689 INFO [Metrics] AUC: 0.941573 - logloss: 0.270773
2021-01-07 16:53:22,981 P2689 INFO Save best model: monitor(max): 0.941573
2021-01-07 16:53:22,985 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:53:23,030 P2689 INFO Train loss: 0.250190
2021-01-07 16:53:23,031 P2689 INFO ************ Epoch=29 end ************
2021-01-07 16:53:26,425 P2689 INFO [Metrics] AUC: 0.942242 - logloss: 0.269026
2021-01-07 16:53:26,426 P2689 INFO Save best model: monitor(max): 0.942242
2021-01-07 16:53:26,428 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:53:26,469 P2689 INFO Train loss: 0.248388
2021-01-07 16:53:26,469 P2689 INFO ************ Epoch=30 end ************
2021-01-07 16:53:29,820 P2689 INFO [Metrics] AUC: 0.942761 - logloss: 0.267676
2021-01-07 16:53:29,820 P2689 INFO Save best model: monitor(max): 0.942761
2021-01-07 16:53:29,822 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:53:29,863 P2689 INFO Train loss: 0.246419
2021-01-07 16:53:29,863 P2689 INFO ************ Epoch=31 end ************
2021-01-07 16:53:33,222 P2689 INFO [Metrics] AUC: 0.943442 - logloss: 0.266039
2021-01-07 16:53:33,223 P2689 INFO Save best model: monitor(max): 0.943442
2021-01-07 16:53:33,225 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:53:33,274 P2689 INFO Train loss: 0.244498
2021-01-07 16:53:33,274 P2689 INFO ************ Epoch=32 end ************
2021-01-07 16:53:36,480 P2689 INFO [Metrics] AUC: 0.944019 - logloss: 0.264580
2021-01-07 16:53:36,480 P2689 INFO Save best model: monitor(max): 0.944019
2021-01-07 16:53:36,483 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:53:36,532 P2689 INFO Train loss: 0.242510
2021-01-07 16:53:36,532 P2689 INFO ************ Epoch=33 end ************
2021-01-07 16:53:39,769 P2689 INFO [Metrics] AUC: 0.944718 - logloss: 0.263053
2021-01-07 16:53:39,770 P2689 INFO Save best model: monitor(max): 0.944718
2021-01-07 16:53:39,772 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:53:39,823 P2689 INFO Train loss: 0.240337
2021-01-07 16:53:39,823 P2689 INFO ************ Epoch=34 end ************
2021-01-07 16:53:43,463 P2689 INFO [Metrics] AUC: 0.945449 - logloss: 0.261557
2021-01-07 16:53:43,464 P2689 INFO Save best model: monitor(max): 0.945449
2021-01-07 16:53:43,466 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:53:43,516 P2689 INFO Train loss: 0.238248
2021-01-07 16:53:43,516 P2689 INFO ************ Epoch=35 end ************
2021-01-07 16:53:47,044 P2689 INFO [Metrics] AUC: 0.946115 - logloss: 0.259833
2021-01-07 16:53:47,045 P2689 INFO Save best model: monitor(max): 0.946115
2021-01-07 16:53:47,047 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:53:47,095 P2689 INFO Train loss: 0.236044
2021-01-07 16:53:47,095 P2689 INFO ************ Epoch=36 end ************
2021-01-07 16:53:50,642 P2689 INFO [Metrics] AUC: 0.946773 - logloss: 0.258188
2021-01-07 16:53:50,643 P2689 INFO Save best model: monitor(max): 0.946773
2021-01-07 16:53:50,646 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:53:50,692 P2689 INFO Train loss: 0.234175
2021-01-07 16:53:50,692 P2689 INFO ************ Epoch=37 end ************
2021-01-07 16:53:54,201 P2689 INFO [Metrics] AUC: 0.947454 - logloss: 0.256565
2021-01-07 16:53:54,202 P2689 INFO Save best model: monitor(max): 0.947454
2021-01-07 16:53:54,205 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:53:54,262 P2689 INFO Train loss: 0.232121
2021-01-07 16:53:54,262 P2689 INFO ************ Epoch=38 end ************
2021-01-07 16:53:57,832 P2689 INFO [Metrics] AUC: 0.947976 - logloss: 0.255229
2021-01-07 16:53:57,832 P2689 INFO Save best model: monitor(max): 0.947976
2021-01-07 16:53:57,835 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:53:57,893 P2689 INFO Train loss: 0.229768
2021-01-07 16:53:57,894 P2689 INFO ************ Epoch=39 end ************
2021-01-07 16:54:01,390 P2689 INFO [Metrics] AUC: 0.948850 - logloss: 0.253290
2021-01-07 16:54:01,390 P2689 INFO Save best model: monitor(max): 0.948850
2021-01-07 16:54:01,393 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:54:01,446 P2689 INFO Train loss: 0.227833
2021-01-07 16:54:01,446 P2689 INFO ************ Epoch=40 end ************
2021-01-07 16:54:05,041 P2689 INFO [Metrics] AUC: 0.949433 - logloss: 0.251821
2021-01-07 16:54:05,042 P2689 INFO Save best model: monitor(max): 0.949433
2021-01-07 16:54:05,044 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:54:05,092 P2689 INFO Train loss: 0.225538
2021-01-07 16:54:05,093 P2689 INFO ************ Epoch=41 end ************
2021-01-07 16:54:08,787 P2689 INFO [Metrics] AUC: 0.950057 - logloss: 0.250200
2021-01-07 16:54:08,788 P2689 INFO Save best model: monitor(max): 0.950057
2021-01-07 16:54:08,792 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:54:08,843 P2689 INFO Train loss: 0.223468
2021-01-07 16:54:08,844 P2689 INFO ************ Epoch=42 end ************
2021-01-07 16:54:12,537 P2689 INFO [Metrics] AUC: 0.950863 - logloss: 0.248472
2021-01-07 16:54:12,537 P2689 INFO Save best model: monitor(max): 0.950863
2021-01-07 16:54:12,540 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:54:12,587 P2689 INFO Train loss: 0.221371
2021-01-07 16:54:12,587 P2689 INFO ************ Epoch=43 end ************
2021-01-07 16:54:16,267 P2689 INFO [Metrics] AUC: 0.951416 - logloss: 0.247077
2021-01-07 16:54:16,268 P2689 INFO Save best model: monitor(max): 0.951416
2021-01-07 16:54:16,270 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:54:16,317 P2689 INFO Train loss: 0.218938
2021-01-07 16:54:16,317 P2689 INFO ************ Epoch=44 end ************
2021-01-07 16:54:20,022 P2689 INFO [Metrics] AUC: 0.951978 - logloss: 0.245641
2021-01-07 16:54:20,022 P2689 INFO Save best model: monitor(max): 0.951978
2021-01-07 16:54:20,025 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:54:20,083 P2689 INFO Train loss: 0.216796
2021-01-07 16:54:20,083 P2689 INFO ************ Epoch=45 end ************
2021-01-07 16:54:23,847 P2689 INFO [Metrics] AUC: 0.952621 - logloss: 0.244170
2021-01-07 16:54:23,848 P2689 INFO Save best model: monitor(max): 0.952621
2021-01-07 16:54:23,850 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:54:23,901 P2689 INFO Train loss: 0.215166
2021-01-07 16:54:23,901 P2689 INFO ************ Epoch=46 end ************
2021-01-07 16:54:27,814 P2689 INFO [Metrics] AUC: 0.953225 - logloss: 0.242763
2021-01-07 16:54:27,815 P2689 INFO Save best model: monitor(max): 0.953225
2021-01-07 16:54:27,819 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:54:27,871 P2689 INFO Train loss: 0.212981
2021-01-07 16:54:27,871 P2689 INFO ************ Epoch=47 end ************
2021-01-07 16:54:31,467 P2689 INFO [Metrics] AUC: 0.953812 - logloss: 0.241393
2021-01-07 16:54:31,468 P2689 INFO Save best model: monitor(max): 0.953812
2021-01-07 16:54:31,470 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:54:31,531 P2689 INFO Train loss: 0.211029
2021-01-07 16:54:31,531 P2689 INFO ************ Epoch=48 end ************
2021-01-07 16:54:35,277 P2689 INFO [Metrics] AUC: 0.954421 - logloss: 0.239708
2021-01-07 16:54:35,278 P2689 INFO Save best model: monitor(max): 0.954421
2021-01-07 16:54:35,280 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:54:35,341 P2689 INFO Train loss: 0.209167
2021-01-07 16:54:35,342 P2689 INFO ************ Epoch=49 end ************
2021-01-07 16:54:38,752 P2689 INFO [Metrics] AUC: 0.954972 - logloss: 0.238437
2021-01-07 16:54:38,753 P2689 INFO Save best model: monitor(max): 0.954972
2021-01-07 16:54:38,755 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:54:38,805 P2689 INFO Train loss: 0.206886
2021-01-07 16:54:38,805 P2689 INFO ************ Epoch=50 end ************
2021-01-07 16:54:42,181 P2689 INFO [Metrics] AUC: 0.955451 - logloss: 0.237190
2021-01-07 16:54:42,181 P2689 INFO Save best model: monitor(max): 0.955451
2021-01-07 16:54:42,184 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:54:42,241 P2689 INFO Train loss: 0.204812
2021-01-07 16:54:42,241 P2689 INFO ************ Epoch=51 end ************
2021-01-07 16:54:45,552 P2689 INFO [Metrics] AUC: 0.955952 - logloss: 0.236073
2021-01-07 16:54:45,552 P2689 INFO Save best model: monitor(max): 0.955952
2021-01-07 16:54:45,555 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:54:45,607 P2689 INFO Train loss: 0.202778
2021-01-07 16:54:45,607 P2689 INFO ************ Epoch=52 end ************
2021-01-07 16:54:48,824 P2689 INFO [Metrics] AUC: 0.956469 - logloss: 0.234777
2021-01-07 16:54:48,825 P2689 INFO Save best model: monitor(max): 0.956469
2021-01-07 16:54:48,828 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:54:48,880 P2689 INFO Train loss: 0.201185
2021-01-07 16:54:48,880 P2689 INFO ************ Epoch=53 end ************
2021-01-07 16:54:52,313 P2689 INFO [Metrics] AUC: 0.956898 - logloss: 0.233553
2021-01-07 16:54:52,314 P2689 INFO Save best model: monitor(max): 0.956898
2021-01-07 16:54:52,316 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:54:52,366 P2689 INFO Train loss: 0.199046
2021-01-07 16:54:52,366 P2689 INFO ************ Epoch=54 end ************
2021-01-07 16:54:55,590 P2689 INFO [Metrics] AUC: 0.957424 - logloss: 0.232143
2021-01-07 16:54:55,590 P2689 INFO Save best model: monitor(max): 0.957424
2021-01-07 16:54:55,592 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:54:55,645 P2689 INFO Train loss: 0.197100
2021-01-07 16:54:55,645 P2689 INFO ************ Epoch=55 end ************
2021-01-07 16:54:58,981 P2689 INFO [Metrics] AUC: 0.957866 - logloss: 0.231121
2021-01-07 16:54:58,982 P2689 INFO Save best model: monitor(max): 0.957866
2021-01-07 16:54:58,984 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:54:59,030 P2689 INFO Train loss: 0.195481
2021-01-07 16:54:59,030 P2689 INFO ************ Epoch=56 end ************
2021-01-07 16:55:02,259 P2689 INFO [Metrics] AUC: 0.958349 - logloss: 0.229868
2021-01-07 16:55:02,260 P2689 INFO Save best model: monitor(max): 0.958349
2021-01-07 16:55:02,263 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:55:02,307 P2689 INFO Train loss: 0.193471
2021-01-07 16:55:02,307 P2689 INFO ************ Epoch=57 end ************
2021-01-07 16:55:05,491 P2689 INFO [Metrics] AUC: 0.958707 - logloss: 0.228858
2021-01-07 16:55:05,491 P2689 INFO Save best model: monitor(max): 0.958707
2021-01-07 16:55:05,494 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:55:05,539 P2689 INFO Train loss: 0.191900
2021-01-07 16:55:05,539 P2689 INFO ************ Epoch=58 end ************
2021-01-07 16:55:08,752 P2689 INFO [Metrics] AUC: 0.959124 - logloss: 0.228219
2021-01-07 16:55:08,752 P2689 INFO Save best model: monitor(max): 0.959124
2021-01-07 16:55:08,754 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:55:08,807 P2689 INFO Train loss: 0.190167
2021-01-07 16:55:08,807 P2689 INFO ************ Epoch=59 end ************
2021-01-07 16:55:12,056 P2689 INFO [Metrics] AUC: 0.959565 - logloss: 0.226891
2021-01-07 16:55:12,057 P2689 INFO Save best model: monitor(max): 0.959565
2021-01-07 16:55:12,060 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:55:12,116 P2689 INFO Train loss: 0.188461
2021-01-07 16:55:12,116 P2689 INFO ************ Epoch=60 end ************
2021-01-07 16:55:15,382 P2689 INFO [Metrics] AUC: 0.959981 - logloss: 0.225852
2021-01-07 16:55:15,384 P2689 INFO Save best model: monitor(max): 0.959981
2021-01-07 16:55:15,387 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:55:15,435 P2689 INFO Train loss: 0.186819
2021-01-07 16:55:15,435 P2689 INFO ************ Epoch=61 end ************
2021-01-07 16:55:18,662 P2689 INFO [Metrics] AUC: 0.960397 - logloss: 0.224756
2021-01-07 16:55:18,663 P2689 INFO Save best model: monitor(max): 0.960397
2021-01-07 16:55:18,665 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:55:18,718 P2689 INFO Train loss: 0.185111
2021-01-07 16:55:18,718 P2689 INFO ************ Epoch=62 end ************
2021-01-07 16:55:22,006 P2689 INFO [Metrics] AUC: 0.960729 - logloss: 0.223851
2021-01-07 16:55:22,007 P2689 INFO Save best model: monitor(max): 0.960729
2021-01-07 16:55:22,010 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:55:22,053 P2689 INFO Train loss: 0.183518
2021-01-07 16:55:22,053 P2689 INFO ************ Epoch=63 end ************
2021-01-07 16:55:25,439 P2689 INFO [Metrics] AUC: 0.961079 - logloss: 0.222887
2021-01-07 16:55:25,439 P2689 INFO Save best model: monitor(max): 0.961079
2021-01-07 16:55:25,441 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:55:25,494 P2689 INFO Train loss: 0.181794
2021-01-07 16:55:25,494 P2689 INFO ************ Epoch=64 end ************
2021-01-07 16:55:28,671 P2689 INFO [Metrics] AUC: 0.961419 - logloss: 0.221984
2021-01-07 16:55:28,672 P2689 INFO Save best model: monitor(max): 0.961419
2021-01-07 16:55:28,675 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:55:28,727 P2689 INFO Train loss: 0.180471
2021-01-07 16:55:28,727 P2689 INFO ************ Epoch=65 end ************
2021-01-07 16:55:32,152 P2689 INFO [Metrics] AUC: 0.961699 - logloss: 0.221333
2021-01-07 16:55:32,153 P2689 INFO Save best model: monitor(max): 0.961699
2021-01-07 16:55:32,156 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:55:32,205 P2689 INFO Train loss: 0.178787
2021-01-07 16:55:32,205 P2689 INFO ************ Epoch=66 end ************
2021-01-07 16:55:35,651 P2689 INFO [Metrics] AUC: 0.962076 - logloss: 0.220289
2021-01-07 16:55:35,652 P2689 INFO Save best model: monitor(max): 0.962076
2021-01-07 16:55:35,655 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:55:35,703 P2689 INFO Train loss: 0.177525
2021-01-07 16:55:35,703 P2689 INFO ************ Epoch=67 end ************
2021-01-07 16:55:39,139 P2689 INFO [Metrics] AUC: 0.962375 - logloss: 0.219521
2021-01-07 16:55:39,140 P2689 INFO Save best model: monitor(max): 0.962375
2021-01-07 16:55:39,145 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:55:39,189 P2689 INFO Train loss: 0.176321
2021-01-07 16:55:39,189 P2689 INFO ************ Epoch=68 end ************
2021-01-07 16:55:42,662 P2689 INFO [Metrics] AUC: 0.962611 - logloss: 0.218895
2021-01-07 16:55:42,663 P2689 INFO Save best model: monitor(max): 0.962611
2021-01-07 16:55:42,666 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:55:42,717 P2689 INFO Train loss: 0.174438
2021-01-07 16:55:42,717 P2689 INFO ************ Epoch=69 end ************
2021-01-07 16:55:46,206 P2689 INFO [Metrics] AUC: 0.962966 - logloss: 0.218012
2021-01-07 16:55:46,206 P2689 INFO Save best model: monitor(max): 0.962966
2021-01-07 16:55:46,208 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:55:46,260 P2689 INFO Train loss: 0.173427
2021-01-07 16:55:46,260 P2689 INFO ************ Epoch=70 end ************
2021-01-07 16:55:49,676 P2689 INFO [Metrics] AUC: 0.963255 - logloss: 0.217246
2021-01-07 16:55:49,676 P2689 INFO Save best model: monitor(max): 0.963255
2021-01-07 16:55:49,678 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:55:49,723 P2689 INFO Train loss: 0.171989
2021-01-07 16:55:49,723 P2689 INFO ************ Epoch=71 end ************
2021-01-07 16:55:53,258 P2689 INFO [Metrics] AUC: 0.963474 - logloss: 0.216664
2021-01-07 16:55:53,259 P2689 INFO Save best model: monitor(max): 0.963474
2021-01-07 16:55:53,262 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:55:53,313 P2689 INFO Train loss: 0.170621
2021-01-07 16:55:53,313 P2689 INFO ************ Epoch=72 end ************
2021-01-07 16:55:56,941 P2689 INFO [Metrics] AUC: 0.963721 - logloss: 0.216042
2021-01-07 16:55:56,942 P2689 INFO Save best model: monitor(max): 0.963721
2021-01-07 16:55:56,945 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:55:56,993 P2689 INFO Train loss: 0.169663
2021-01-07 16:55:56,993 P2689 INFO ************ Epoch=73 end ************
2021-01-07 16:56:00,600 P2689 INFO [Metrics] AUC: 0.963962 - logloss: 0.215301
2021-01-07 16:56:00,601 P2689 INFO Save best model: monitor(max): 0.963962
2021-01-07 16:56:00,604 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:56:00,651 P2689 INFO Train loss: 0.168207
2021-01-07 16:56:00,652 P2689 INFO ************ Epoch=74 end ************
2021-01-07 16:56:04,407 P2689 INFO [Metrics] AUC: 0.964223 - logloss: 0.214631
2021-01-07 16:56:04,408 P2689 INFO Save best model: monitor(max): 0.964223
2021-01-07 16:56:04,410 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:56:04,456 P2689 INFO Train loss: 0.167242
2021-01-07 16:56:04,457 P2689 INFO ************ Epoch=75 end ************
2021-01-07 16:56:08,225 P2689 INFO [Metrics] AUC: 0.964528 - logloss: 0.213881
2021-01-07 16:56:08,226 P2689 INFO Save best model: monitor(max): 0.964528
2021-01-07 16:56:08,229 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:56:08,282 P2689 INFO Train loss: 0.165844
2021-01-07 16:56:08,282 P2689 INFO ************ Epoch=76 end ************
2021-01-07 16:56:12,028 P2689 INFO [Metrics] AUC: 0.964742 - logloss: 0.213430
2021-01-07 16:56:12,028 P2689 INFO Save best model: monitor(max): 0.964742
2021-01-07 16:56:12,030 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:56:12,080 P2689 INFO Train loss: 0.164755
2021-01-07 16:56:12,080 P2689 INFO ************ Epoch=77 end ************
2021-01-07 16:56:15,864 P2689 INFO [Metrics] AUC: 0.964948 - logloss: 0.212787
2021-01-07 16:56:15,865 P2689 INFO Save best model: monitor(max): 0.964948
2021-01-07 16:56:15,868 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:56:15,918 P2689 INFO Train loss: 0.163408
2021-01-07 16:56:15,919 P2689 INFO ************ Epoch=78 end ************
2021-01-07 16:56:19,676 P2689 INFO [Metrics] AUC: 0.965189 - logloss: 0.212174
2021-01-07 16:56:19,677 P2689 INFO Save best model: monitor(max): 0.965189
2021-01-07 16:56:19,680 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:56:19,735 P2689 INFO Train loss: 0.162514
2021-01-07 16:56:19,735 P2689 INFO ************ Epoch=79 end ************
2021-01-07 16:56:23,550 P2689 INFO [Metrics] AUC: 0.965394 - logloss: 0.211504
2021-01-07 16:56:23,550 P2689 INFO Save best model: monitor(max): 0.965394
2021-01-07 16:56:23,552 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:56:23,608 P2689 INFO Train loss: 0.161178
2021-01-07 16:56:23,608 P2689 INFO ************ Epoch=80 end ************
2021-01-07 16:56:27,471 P2689 INFO [Metrics] AUC: 0.965515 - logloss: 0.211105
2021-01-07 16:56:27,471 P2689 INFO Save best model: monitor(max): 0.965515
2021-01-07 16:56:27,473 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:56:27,526 P2689 INFO Train loss: 0.160001
2021-01-07 16:56:27,526 P2689 INFO ************ Epoch=81 end ************
2021-01-07 16:56:31,271 P2689 INFO [Metrics] AUC: 0.965810 - logloss: 0.210498
2021-01-07 16:56:31,272 P2689 INFO Save best model: monitor(max): 0.965810
2021-01-07 16:56:31,274 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:56:31,319 P2689 INFO Train loss: 0.158974
2021-01-07 16:56:31,319 P2689 INFO ************ Epoch=82 end ************
2021-01-07 16:56:34,896 P2689 INFO [Metrics] AUC: 0.966047 - logloss: 0.209961
2021-01-07 16:56:34,896 P2689 INFO Save best model: monitor(max): 0.966047
2021-01-07 16:56:34,899 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:56:34,951 P2689 INFO Train loss: 0.157787
2021-01-07 16:56:34,951 P2689 INFO ************ Epoch=83 end ************
2021-01-07 16:56:38,326 P2689 INFO [Metrics] AUC: 0.966188 - logloss: 0.209366
2021-01-07 16:56:38,327 P2689 INFO Save best model: monitor(max): 0.966188
2021-01-07 16:56:38,329 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:56:38,371 P2689 INFO Train loss: 0.156833
2021-01-07 16:56:38,371 P2689 INFO ************ Epoch=84 end ************
2021-01-07 16:56:41,757 P2689 INFO [Metrics] AUC: 0.966409 - logloss: 0.208820
2021-01-07 16:56:41,757 P2689 INFO Save best model: monitor(max): 0.966409
2021-01-07 16:56:41,760 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:56:41,815 P2689 INFO Train loss: 0.155792
2021-01-07 16:56:41,816 P2689 INFO ************ Epoch=85 end ************
2021-01-07 16:56:45,292 P2689 INFO [Metrics] AUC: 0.966473 - logloss: 0.208818
2021-01-07 16:56:45,293 P2689 INFO Save best model: monitor(max): 0.966473
2021-01-07 16:56:45,295 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:56:45,348 P2689 INFO Train loss: 0.154569
2021-01-07 16:56:45,348 P2689 INFO ************ Epoch=86 end ************
2021-01-07 16:56:48,709 P2689 INFO [Metrics] AUC: 0.966821 - logloss: 0.207797
2021-01-07 16:56:48,709 P2689 INFO Save best model: monitor(max): 0.966821
2021-01-07 16:56:48,712 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:56:48,756 P2689 INFO Train loss: 0.153858
2021-01-07 16:56:48,756 P2689 INFO ************ Epoch=87 end ************
2021-01-07 16:56:51,942 P2689 INFO [Metrics] AUC: 0.966939 - logloss: 0.207240
2021-01-07 16:56:51,942 P2689 INFO Save best model: monitor(max): 0.966939
2021-01-07 16:56:51,945 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:56:51,987 P2689 INFO Train loss: 0.152950
2021-01-07 16:56:51,988 P2689 INFO ************ Epoch=88 end ************
2021-01-07 16:56:55,085 P2689 INFO [Metrics] AUC: 0.967094 - logloss: 0.206836
2021-01-07 16:56:55,085 P2689 INFO Save best model: monitor(max): 0.967094
2021-01-07 16:56:55,088 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:56:55,135 P2689 INFO Train loss: 0.151890
2021-01-07 16:56:55,135 P2689 INFO ************ Epoch=89 end ************
2021-01-07 16:56:58,426 P2689 INFO [Metrics] AUC: 0.967272 - logloss: 0.206345
2021-01-07 16:56:58,426 P2689 INFO Save best model: monitor(max): 0.967272
2021-01-07 16:56:58,429 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:56:58,478 P2689 INFO Train loss: 0.150629
2021-01-07 16:56:58,478 P2689 INFO ************ Epoch=90 end ************
2021-01-07 16:57:01,653 P2689 INFO [Metrics] AUC: 0.967424 - logloss: 0.205885
2021-01-07 16:57:01,654 P2689 INFO Save best model: monitor(max): 0.967424
2021-01-07 16:57:01,656 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:57:01,703 P2689 INFO Train loss: 0.149940
2021-01-07 16:57:01,704 P2689 INFO ************ Epoch=91 end ************
2021-01-07 16:57:05,075 P2689 INFO [Metrics] AUC: 0.967472 - logloss: 0.205838
2021-01-07 16:57:05,075 P2689 INFO Save best model: monitor(max): 0.967472
2021-01-07 16:57:05,078 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:57:05,124 P2689 INFO Train loss: 0.148863
2021-01-07 16:57:05,124 P2689 INFO ************ Epoch=92 end ************
2021-01-07 16:57:08,505 P2689 INFO [Metrics] AUC: 0.967794 - logloss: 0.204971
2021-01-07 16:57:08,505 P2689 INFO Save best model: monitor(max): 0.967794
2021-01-07 16:57:08,508 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:57:08,560 P2689 INFO Train loss: 0.147887
2021-01-07 16:57:08,561 P2689 INFO ************ Epoch=93 end ************
2021-01-07 16:57:11,830 P2689 INFO [Metrics] AUC: 0.967871 - logloss: 0.204652
2021-01-07 16:57:11,830 P2689 INFO Save best model: monitor(max): 0.967871
2021-01-07 16:57:11,833 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:57:11,876 P2689 INFO Train loss: 0.146925
2021-01-07 16:57:11,876 P2689 INFO ************ Epoch=94 end ************
2021-01-07 16:57:15,117 P2689 INFO [Metrics] AUC: 0.968058 - logloss: 0.204332
2021-01-07 16:57:15,118 P2689 INFO Save best model: monitor(max): 0.968058
2021-01-07 16:57:15,120 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:57:15,170 P2689 INFO Train loss: 0.146365
2021-01-07 16:57:15,170 P2689 INFO ************ Epoch=95 end ************
2021-01-07 16:57:18,617 P2689 INFO [Metrics] AUC: 0.968261 - logloss: 0.203870
2021-01-07 16:57:18,618 P2689 INFO Save best model: monitor(max): 0.968261
2021-01-07 16:57:18,621 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:57:18,675 P2689 INFO Train loss: 0.145451
2021-01-07 16:57:18,675 P2689 INFO ************ Epoch=96 end ************
2021-01-07 16:57:22,088 P2689 INFO [Metrics] AUC: 0.968366 - logloss: 0.203432
2021-01-07 16:57:22,089 P2689 INFO Save best model: monitor(max): 0.968366
2021-01-07 16:57:22,091 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:57:22,142 P2689 INFO Train loss: 0.144642
2021-01-07 16:57:22,142 P2689 INFO ************ Epoch=97 end ************
2021-01-07 16:57:24,917 P2689 INFO [Metrics] AUC: 0.968401 - logloss: 0.203249
2021-01-07 16:57:24,918 P2689 INFO Save best model: monitor(max): 0.968401
2021-01-07 16:57:24,921 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:57:24,970 P2689 INFO Train loss: 0.143359
2021-01-07 16:57:24,970 P2689 INFO ************ Epoch=98 end ************
2021-01-07 16:57:27,559 P2689 INFO [Metrics] AUC: 0.968634 - logloss: 0.202919
2021-01-07 16:57:27,559 P2689 INFO Save best model: monitor(max): 0.968634
2021-01-07 16:57:27,562 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:57:27,609 P2689 INFO Train loss: 0.142682
2021-01-07 16:57:27,610 P2689 INFO ************ Epoch=99 end ************
2021-01-07 16:57:30,226 P2689 INFO [Metrics] AUC: 0.968650 - logloss: 0.202600
2021-01-07 16:57:30,227 P2689 INFO Save best model: monitor(max): 0.968650
2021-01-07 16:57:30,229 P2689 INFO --- 50/50 batches finished ---
2021-01-07 16:57:30,271 P2689 INFO Train loss: 0.142052
2021-01-07 16:57:30,271 P2689 INFO ************ Epoch=100 end ************
2021-01-07 16:57:30,271 P2689 INFO Training finished.
2021-01-07 16:57:30,272 P2689 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Frappe/FM_frappe_x1/frappe_x1_7f91d67a/FM_frappe_x1_001_b16e1650_model.ckpt
2021-01-07 16:57:30,280 P2689 INFO ****** Train/validation evaluation ******
2021-01-07 16:57:30,755 P2689 INFO [Metrics] AUC: 0.968650 - logloss: 0.202600
2021-01-07 16:57:30,878 P2689 INFO ******** Test evaluation ********
2021-01-07 16:57:30,879 P2689 INFO Loading data...
2021-01-07 16:57:30,879 P2689 INFO Loading data from h5: ../data/Frappe/frappe_x1_7f91d67a/test.h5
2021-01-07 16:57:30,882 P2689 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2021-01-07 16:57:30,882 P2689 INFO Loading test data done.
2021-01-07 16:57:31,140 P2689 INFO [Metrics] AUC: 0.967108 - logloss: 0.206515

```
