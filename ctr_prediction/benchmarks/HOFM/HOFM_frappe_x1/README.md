## HOFM_frappe_x1

A hands-on guide to run the HOFM model on the Frappe_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
  GPU: Tesla P100 16G
  RAM: 503G

  ```

+ Software

  ```python
  CUDA: 10.2
  python: 3.6.4
  pytorch: 1.0.0
  pandas: 0.22.0
  numpy: 1.19.1
  scipy: 1.2.2
  sklearn: 0.19.1
  pyyaml: 6.0
  h5py: 2.8.0
  tqdm: 4.28.1
  fuxictr: 1.1.0
  ```

### Dataset
Dataset ID: [Frappe_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Frappe/README.md#Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [HOFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/HOFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HOFM_frappe_x1_tuner_config_02](./HOFM_frappe_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HOFM_frappe_x1
    nohup python run_expid.py --config ./HOFM_frappe_x1_tuner_config_02 --expid HOFM_frappe_x1_003_64d2d04e --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.974228 | 0.207269  |


### Logs
```python
2022-01-26 11:46:31,159 P22435 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "embedding_dim": "10",
    "epochs": "200",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "HOFM",
    "model_id": "HOFM_frappe_x1_003_64d2d04e",
    "model_root": "./Frappe/HOFM_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "order": "3",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-06",
    "reuse_embedding": "False",
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
2022-01-26 11:46:31,160 P22435 INFO Set up feature encoder...
2022-01-26 11:46:31,160 P22435 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-01-26 11:46:31,160 P22435 INFO Loading data...
2022-01-26 11:46:31,163 P22435 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-01-26 11:46:31,175 P22435 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-01-26 11:46:31,179 P22435 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-01-26 11:46:31,179 P22435 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-01-26 11:46:31,179 P22435 INFO Loading train data done.
2022-01-26 11:46:35,184 P22435 INFO Total number of parameters: 113170.
2022-01-26 11:46:35,184 P22435 INFO Start training: 50 batches/epoch
2022-01-26 11:46:35,185 P22435 INFO ************ Epoch=1 start ************
2022-01-26 11:46:45,198 P22435 INFO [Metrics] AUC: 0.886874 - logloss: 0.621561
2022-01-26 11:46:45,198 P22435 INFO Save best model: monitor(max): 0.886874
2022-01-26 11:46:45,202 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:46:45,241 P22435 INFO Train loss: 0.656697
2022-01-26 11:46:45,241 P22435 INFO ************ Epoch=1 end ************
2022-01-26 11:46:55,038 P22435 INFO [Metrics] AUC: 0.908873 - logloss: 0.508016
2022-01-26 11:46:55,038 P22435 INFO Save best model: monitor(max): 0.908873
2022-01-26 11:46:55,042 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:46:55,085 P22435 INFO Train loss: 0.574308
2022-01-26 11:46:55,085 P22435 INFO ************ Epoch=2 end ************
2022-01-26 11:47:04,898 P22435 INFO [Metrics] AUC: 0.930146 - logloss: 0.350207
2022-01-26 11:47:04,898 P22435 INFO Save best model: monitor(max): 0.930146
2022-01-26 11:47:04,902 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:47:04,948 P22435 INFO Train loss: 0.423325
2022-01-26 11:47:04,948 P22435 INFO ************ Epoch=3 end ************
2022-01-26 11:47:14,557 P22435 INFO [Metrics] AUC: 0.935636 - logloss: 0.293265
2022-01-26 11:47:14,558 P22435 INFO Save best model: monitor(max): 0.935636
2022-01-26 11:47:14,561 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:47:14,600 P22435 INFO Train loss: 0.308345
2022-01-26 11:47:14,600 P22435 INFO ************ Epoch=4 end ************
2022-01-26 11:47:24,377 P22435 INFO [Metrics] AUC: 0.937523 - logloss: 0.283180
2022-01-26 11:47:24,377 P22435 INFO Save best model: monitor(max): 0.937523
2022-01-26 11:47:24,380 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:47:24,421 P22435 INFO Train loss: 0.277209
2022-01-26 11:47:24,421 P22435 INFO ************ Epoch=5 end ************
2022-01-26 11:47:34,081 P22435 INFO [Metrics] AUC: 0.938274 - logloss: 0.280719
2022-01-26 11:47:34,081 P22435 INFO Save best model: monitor(max): 0.938274
2022-01-26 11:47:34,086 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:47:34,131 P22435 INFO Train loss: 0.268866
2022-01-26 11:47:34,131 P22435 INFO ************ Epoch=6 end ************
2022-01-26 11:47:43,397 P22435 INFO [Metrics] AUC: 0.938727 - logloss: 0.279426
2022-01-26 11:47:43,398 P22435 INFO Save best model: monitor(max): 0.938727
2022-01-26 11:47:43,401 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:47:43,441 P22435 INFO Train loss: 0.264266
2022-01-26 11:47:43,442 P22435 INFO ************ Epoch=7 end ************
2022-01-26 11:47:52,710 P22435 INFO [Metrics] AUC: 0.939142 - logloss: 0.277973
2022-01-26 11:47:52,711 P22435 INFO Save best model: monitor(max): 0.939142
2022-01-26 11:47:52,714 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:47:52,755 P22435 INFO Train loss: 0.261287
2022-01-26 11:47:52,755 P22435 INFO ************ Epoch=8 end ************
2022-01-26 11:48:01,816 P22435 INFO [Metrics] AUC: 0.939727 - logloss: 0.276870
2022-01-26 11:48:01,816 P22435 INFO Save best model: monitor(max): 0.939727
2022-01-26 11:48:01,820 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:48:01,866 P22435 INFO Train loss: 0.258584
2022-01-26 11:48:01,866 P22435 INFO ************ Epoch=9 end ************
2022-01-26 11:48:10,741 P22435 INFO [Metrics] AUC: 0.940325 - logloss: 0.275280
2022-01-26 11:48:10,741 P22435 INFO Save best model: monitor(max): 0.940325
2022-01-26 11:48:10,746 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:48:10,786 P22435 INFO Train loss: 0.255743
2022-01-26 11:48:10,786 P22435 INFO ************ Epoch=10 end ************
2022-01-26 11:48:19,597 P22435 INFO [Metrics] AUC: 0.940985 - logloss: 0.273668
2022-01-26 11:48:19,597 P22435 INFO Save best model: monitor(max): 0.940985
2022-01-26 11:48:19,601 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:48:19,650 P22435 INFO Train loss: 0.252409
2022-01-26 11:48:19,650 P22435 INFO ************ Epoch=11 end ************
2022-01-26 11:48:28,562 P22435 INFO [Metrics] AUC: 0.941686 - logloss: 0.272095
2022-01-26 11:48:28,562 P22435 INFO Save best model: monitor(max): 0.941686
2022-01-26 11:48:28,568 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:48:28,615 P22435 INFO Train loss: 0.249493
2022-01-26 11:48:28,616 P22435 INFO ************ Epoch=12 end ************
2022-01-26 11:48:37,448 P22435 INFO [Metrics] AUC: 0.942522 - logloss: 0.270105
2022-01-26 11:48:37,449 P22435 INFO Save best model: monitor(max): 0.942522
2022-01-26 11:48:37,452 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:48:37,494 P22435 INFO Train loss: 0.245960
2022-01-26 11:48:37,494 P22435 INFO ************ Epoch=13 end ************
2022-01-26 11:48:46,490 P22435 INFO [Metrics] AUC: 0.943341 - logloss: 0.268328
2022-01-26 11:48:46,490 P22435 INFO Save best model: monitor(max): 0.943341
2022-01-26 11:48:46,494 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:48:46,537 P22435 INFO Train loss: 0.243241
2022-01-26 11:48:46,537 P22435 INFO ************ Epoch=14 end ************
2022-01-26 11:48:55,952 P22435 INFO [Metrics] AUC: 0.944233 - logloss: 0.266370
2022-01-26 11:48:55,952 P22435 INFO Save best model: monitor(max): 0.944233
2022-01-26 11:48:55,956 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:48:55,999 P22435 INFO Train loss: 0.239981
2022-01-26 11:48:55,999 P22435 INFO ************ Epoch=15 end ************
2022-01-26 11:49:05,889 P22435 INFO [Metrics] AUC: 0.945063 - logloss: 0.264603
2022-01-26 11:49:05,890 P22435 INFO Save best model: monitor(max): 0.945063
2022-01-26 11:49:05,893 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:49:05,943 P22435 INFO Train loss: 0.236802
2022-01-26 11:49:05,943 P22435 INFO ************ Epoch=16 end ************
2022-01-26 11:49:16,184 P22435 INFO [Metrics] AUC: 0.945987 - logloss: 0.262558
2022-01-26 11:49:16,185 P22435 INFO Save best model: monitor(max): 0.945987
2022-01-26 11:49:16,188 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:49:16,234 P22435 INFO Train loss: 0.233316
2022-01-26 11:49:16,234 P22435 INFO ************ Epoch=17 end ************
2022-01-26 11:49:26,397 P22435 INFO [Metrics] AUC: 0.946903 - logloss: 0.260524
2022-01-26 11:49:26,398 P22435 INFO Save best model: monitor(max): 0.946903
2022-01-26 11:49:26,401 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:49:26,442 P22435 INFO Train loss: 0.230532
2022-01-26 11:49:26,442 P22435 INFO ************ Epoch=18 end ************
2022-01-26 11:49:36,575 P22435 INFO [Metrics] AUC: 0.947938 - logloss: 0.258179
2022-01-26 11:49:36,576 P22435 INFO Save best model: monitor(max): 0.947938
2022-01-26 11:49:36,579 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:49:36,624 P22435 INFO Train loss: 0.226963
2022-01-26 11:49:36,625 P22435 INFO ************ Epoch=19 end ************
2022-01-26 11:49:46,790 P22435 INFO [Metrics] AUC: 0.948894 - logloss: 0.255949
2022-01-26 11:49:46,790 P22435 INFO Save best model: monitor(max): 0.948894
2022-01-26 11:49:46,794 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:49:46,839 P22435 INFO Train loss: 0.223192
2022-01-26 11:49:46,840 P22435 INFO ************ Epoch=20 end ************
2022-01-26 11:49:56,828 P22435 INFO [Metrics] AUC: 0.949792 - logloss: 0.254115
2022-01-26 11:49:56,828 P22435 INFO Save best model: monitor(max): 0.949792
2022-01-26 11:49:56,832 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:49:56,887 P22435 INFO Train loss: 0.219539
2022-01-26 11:49:56,887 P22435 INFO ************ Epoch=21 end ************
2022-01-26 11:50:06,559 P22435 INFO [Metrics] AUC: 0.950908 - logloss: 0.251255
2022-01-26 11:50:06,559 P22435 INFO Save best model: monitor(max): 0.950908
2022-01-26 11:50:06,562 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:50:06,606 P22435 INFO Train loss: 0.215623
2022-01-26 11:50:06,606 P22435 INFO ************ Epoch=22 end ************
2022-01-26 11:50:15,785 P22435 INFO [Metrics] AUC: 0.951838 - logloss: 0.249134
2022-01-26 11:50:15,785 P22435 INFO Save best model: monitor(max): 0.951838
2022-01-26 11:50:15,791 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:50:15,834 P22435 INFO Train loss: 0.212135
2022-01-26 11:50:15,834 P22435 INFO ************ Epoch=23 end ************
2022-01-26 11:50:24,716 P22435 INFO [Metrics] AUC: 0.952946 - logloss: 0.246409
2022-01-26 11:50:24,717 P22435 INFO Save best model: monitor(max): 0.952946
2022-01-26 11:50:24,720 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:50:24,764 P22435 INFO Train loss: 0.207728
2022-01-26 11:50:24,764 P22435 INFO ************ Epoch=24 end ************
2022-01-26 11:50:33,726 P22435 INFO [Metrics] AUC: 0.953994 - logloss: 0.243863
2022-01-26 11:50:33,727 P22435 INFO Save best model: monitor(max): 0.953994
2022-01-26 11:50:33,730 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:50:33,776 P22435 INFO Train loss: 0.203507
2022-01-26 11:50:33,776 P22435 INFO ************ Epoch=25 end ************
2022-01-26 11:50:42,761 P22435 INFO [Metrics] AUC: 0.954916 - logloss: 0.241503
2022-01-26 11:50:42,761 P22435 INFO Save best model: monitor(max): 0.954916
2022-01-26 11:50:42,765 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:50:42,810 P22435 INFO Train loss: 0.199034
2022-01-26 11:50:42,811 P22435 INFO ************ Epoch=26 end ************
2022-01-26 11:50:51,657 P22435 INFO [Metrics] AUC: 0.955896 - logloss: 0.238962
2022-01-26 11:50:51,658 P22435 INFO Save best model: monitor(max): 0.955896
2022-01-26 11:50:51,661 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:50:51,705 P22435 INFO Train loss: 0.194943
2022-01-26 11:50:51,706 P22435 INFO ************ Epoch=27 end ************
2022-01-26 11:51:00,684 P22435 INFO [Metrics] AUC: 0.956827 - logloss: 0.236640
2022-01-26 11:51:00,684 P22435 INFO Save best model: monitor(max): 0.956827
2022-01-26 11:51:00,687 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:51:00,736 P22435 INFO Train loss: 0.190522
2022-01-26 11:51:00,736 P22435 INFO ************ Epoch=28 end ************
2022-01-26 11:51:09,692 P22435 INFO [Metrics] AUC: 0.957745 - logloss: 0.234205
2022-01-26 11:51:09,693 P22435 INFO Save best model: monitor(max): 0.957745
2022-01-26 11:51:09,696 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:51:09,740 P22435 INFO Train loss: 0.185977
2022-01-26 11:51:09,740 P22435 INFO ************ Epoch=29 end ************
2022-01-26 11:51:18,660 P22435 INFO [Metrics] AUC: 0.958573 - logloss: 0.231835
2022-01-26 11:51:18,661 P22435 INFO Save best model: monitor(max): 0.958573
2022-01-26 11:51:18,665 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:51:18,710 P22435 INFO Train loss: 0.181900
2022-01-26 11:51:18,710 P22435 INFO ************ Epoch=30 end ************
2022-01-26 11:51:28,439 P22435 INFO [Metrics] AUC: 0.959465 - logloss: 0.229431
2022-01-26 11:51:28,440 P22435 INFO Save best model: monitor(max): 0.959465
2022-01-26 11:51:28,444 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:51:28,487 P22435 INFO Train loss: 0.177504
2022-01-26 11:51:28,487 P22435 INFO ************ Epoch=31 end ************
2022-01-26 11:51:38,526 P22435 INFO [Metrics] AUC: 0.960233 - logloss: 0.227522
2022-01-26 11:51:38,526 P22435 INFO Save best model: monitor(max): 0.960233
2022-01-26 11:51:38,530 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:51:38,578 P22435 INFO Train loss: 0.173712
2022-01-26 11:51:38,578 P22435 INFO ************ Epoch=32 end ************
2022-01-26 11:51:48,786 P22435 INFO [Metrics] AUC: 0.960956 - logloss: 0.225549
2022-01-26 11:51:48,787 P22435 INFO Save best model: monitor(max): 0.960956
2022-01-26 11:51:48,790 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:51:48,835 P22435 INFO Train loss: 0.169591
2022-01-26 11:51:48,835 P22435 INFO ************ Epoch=33 end ************
2022-01-26 11:51:58,628 P22435 INFO [Metrics] AUC: 0.961566 - logloss: 0.223999
2022-01-26 11:51:58,628 P22435 INFO Save best model: monitor(max): 0.961566
2022-01-26 11:51:58,632 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:51:58,676 P22435 INFO Train loss: 0.165780
2022-01-26 11:51:58,676 P22435 INFO ************ Epoch=34 end ************
2022-01-26 11:52:08,939 P22435 INFO [Metrics] AUC: 0.962356 - logloss: 0.221775
2022-01-26 11:52:08,939 P22435 INFO Save best model: monitor(max): 0.962356
2022-01-26 11:52:08,943 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:52:08,987 P22435 INFO Train loss: 0.161786
2022-01-26 11:52:08,987 P22435 INFO ************ Epoch=35 end ************
2022-01-26 11:52:19,210 P22435 INFO [Metrics] AUC: 0.962955 - logloss: 0.220345
2022-01-26 11:52:19,211 P22435 INFO Save best model: monitor(max): 0.962955
2022-01-26 11:52:19,214 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:52:19,268 P22435 INFO Train loss: 0.158458
2022-01-26 11:52:19,269 P22435 INFO ************ Epoch=36 end ************
2022-01-26 11:52:29,519 P22435 INFO [Metrics] AUC: 0.963428 - logloss: 0.218952
2022-01-26 11:52:29,519 P22435 INFO Save best model: monitor(max): 0.963428
2022-01-26 11:52:29,522 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:52:29,573 P22435 INFO Train loss: 0.155140
2022-01-26 11:52:29,573 P22435 INFO ************ Epoch=37 end ************
2022-01-26 11:52:39,846 P22435 INFO [Metrics] AUC: 0.964008 - logloss: 0.217403
2022-01-26 11:52:39,846 P22435 INFO Save best model: monitor(max): 0.964008
2022-01-26 11:52:39,850 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:52:39,892 P22435 INFO Train loss: 0.151476
2022-01-26 11:52:39,892 P22435 INFO ************ Epoch=38 end ************
2022-01-26 11:52:50,148 P22435 INFO [Metrics] AUC: 0.964490 - logloss: 0.216209
2022-01-26 11:52:50,148 P22435 INFO Save best model: monitor(max): 0.964490
2022-01-26 11:52:50,151 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:52:50,191 P22435 INFO Train loss: 0.148378
2022-01-26 11:52:50,192 P22435 INFO ************ Epoch=39 end ************
2022-01-26 11:53:00,302 P22435 INFO [Metrics] AUC: 0.964970 - logloss: 0.215173
2022-01-26 11:53:00,303 P22435 INFO Save best model: monitor(max): 0.964970
2022-01-26 11:53:00,306 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:53:00,357 P22435 INFO Train loss: 0.145531
2022-01-26 11:53:00,357 P22435 INFO ************ Epoch=40 end ************
2022-01-26 11:53:10,304 P22435 INFO [Metrics] AUC: 0.965442 - logloss: 0.213942
2022-01-26 11:53:10,305 P22435 INFO Save best model: monitor(max): 0.965442
2022-01-26 11:53:10,310 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:53:10,354 P22435 INFO Train loss: 0.142868
2022-01-26 11:53:10,355 P22435 INFO ************ Epoch=41 end ************
2022-01-26 11:53:19,879 P22435 INFO [Metrics] AUC: 0.965662 - logloss: 0.213380
2022-01-26 11:53:19,880 P22435 INFO Save best model: monitor(max): 0.965662
2022-01-26 11:53:19,883 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:53:19,928 P22435 INFO Train loss: 0.139981
2022-01-26 11:53:19,928 P22435 INFO ************ Epoch=42 end ************
2022-01-26 11:53:28,912 P22435 INFO [Metrics] AUC: 0.966238 - logloss: 0.211741
2022-01-26 11:53:28,913 P22435 INFO Save best model: monitor(max): 0.966238
2022-01-26 11:53:28,916 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:53:28,957 P22435 INFO Train loss: 0.137491
2022-01-26 11:53:28,957 P22435 INFO ************ Epoch=43 end ************
2022-01-26 11:53:37,868 P22435 INFO [Metrics] AUC: 0.966575 - logloss: 0.210884
2022-01-26 11:53:37,869 P22435 INFO Save best model: monitor(max): 0.966575
2022-01-26 11:53:37,872 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:53:37,917 P22435 INFO Train loss: 0.134325
2022-01-26 11:53:37,917 P22435 INFO ************ Epoch=44 end ************
2022-01-26 11:53:46,920 P22435 INFO [Metrics] AUC: 0.966928 - logloss: 0.210148
2022-01-26 11:53:46,920 P22435 INFO Save best model: monitor(max): 0.966928
2022-01-26 11:53:46,924 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:53:46,970 P22435 INFO Train loss: 0.132210
2022-01-26 11:53:46,970 P22435 INFO ************ Epoch=45 end ************
2022-01-26 11:53:55,796 P22435 INFO [Metrics] AUC: 0.967224 - logloss: 0.209428
2022-01-26 11:53:55,797 P22435 INFO Save best model: monitor(max): 0.967224
2022-01-26 11:53:55,800 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:53:55,842 P22435 INFO Train loss: 0.129427
2022-01-26 11:53:55,843 P22435 INFO ************ Epoch=46 end ************
2022-01-26 11:54:04,746 P22435 INFO [Metrics] AUC: 0.967585 - logloss: 0.208574
2022-01-26 11:54:04,747 P22435 INFO Save best model: monitor(max): 0.967585
2022-01-26 11:54:04,750 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:54:04,794 P22435 INFO Train loss: 0.127401
2022-01-26 11:54:04,794 P22435 INFO ************ Epoch=47 end ************
2022-01-26 11:54:13,715 P22435 INFO [Metrics] AUC: 0.967824 - logloss: 0.208188
2022-01-26 11:54:13,716 P22435 INFO Save best model: monitor(max): 0.967824
2022-01-26 11:54:13,719 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:54:13,764 P22435 INFO Train loss: 0.124866
2022-01-26 11:54:13,765 P22435 INFO ************ Epoch=48 end ************
2022-01-26 11:54:22,665 P22435 INFO [Metrics] AUC: 0.968129 - logloss: 0.207566
2022-01-26 11:54:22,665 P22435 INFO Save best model: monitor(max): 0.968129
2022-01-26 11:54:22,669 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:54:22,710 P22435 INFO Train loss: 0.122890
2022-01-26 11:54:22,710 P22435 INFO ************ Epoch=49 end ************
2022-01-26 11:54:32,115 P22435 INFO [Metrics] AUC: 0.968491 - logloss: 0.206639
2022-01-26 11:54:32,116 P22435 INFO Save best model: monitor(max): 0.968491
2022-01-26 11:54:32,119 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:54:32,158 P22435 INFO Train loss: 0.120470
2022-01-26 11:54:32,158 P22435 INFO ************ Epoch=50 end ************
2022-01-26 11:54:41,975 P22435 INFO [Metrics] AUC: 0.968803 - logloss: 0.206158
2022-01-26 11:54:41,976 P22435 INFO Save best model: monitor(max): 0.968803
2022-01-26 11:54:41,979 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:54:42,027 P22435 INFO Train loss: 0.118479
2022-01-26 11:54:42,027 P22435 INFO ************ Epoch=51 end ************
2022-01-26 11:54:51,454 P22435 INFO [Metrics] AUC: 0.968964 - logloss: 0.205778
2022-01-26 11:54:51,455 P22435 INFO Save best model: monitor(max): 0.968964
2022-01-26 11:54:51,458 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:54:51,504 P22435 INFO Train loss: 0.116547
2022-01-26 11:54:51,504 P22435 INFO ************ Epoch=52 end ************
2022-01-26 11:55:01,723 P22435 INFO [Metrics] AUC: 0.969305 - logloss: 0.205233
2022-01-26 11:55:01,724 P22435 INFO Save best model: monitor(max): 0.969305
2022-01-26 11:55:01,727 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:55:01,770 P22435 INFO Train loss: 0.114792
2022-01-26 11:55:01,771 P22435 INFO ************ Epoch=53 end ************
2022-01-26 11:55:11,904 P22435 INFO [Metrics] AUC: 0.969481 - logloss: 0.204847
2022-01-26 11:55:11,905 P22435 INFO Save best model: monitor(max): 0.969481
2022-01-26 11:55:11,908 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:55:11,952 P22435 INFO Train loss: 0.112811
2022-01-26 11:55:11,952 P22435 INFO ************ Epoch=54 end ************
2022-01-26 11:55:22,028 P22435 INFO [Metrics] AUC: 0.969813 - logloss: 0.204174
2022-01-26 11:55:22,028 P22435 INFO Save best model: monitor(max): 0.969813
2022-01-26 11:55:22,032 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:55:22,076 P22435 INFO Train loss: 0.111017
2022-01-26 11:55:22,076 P22435 INFO ************ Epoch=55 end ************
2022-01-26 11:55:32,147 P22435 INFO [Metrics] AUC: 0.970055 - logloss: 0.203700
2022-01-26 11:55:32,148 P22435 INFO Save best model: monitor(max): 0.970055
2022-01-26 11:55:32,151 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:55:32,199 P22435 INFO Train loss: 0.109216
2022-01-26 11:55:32,199 P22435 INFO ************ Epoch=56 end ************
2022-01-26 11:55:42,310 P22435 INFO [Metrics] AUC: 0.970243 - logloss: 0.203617
2022-01-26 11:55:42,310 P22435 INFO Save best model: monitor(max): 0.970243
2022-01-26 11:55:42,314 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:55:42,361 P22435 INFO Train loss: 0.107330
2022-01-26 11:55:42,361 P22435 INFO ************ Epoch=57 end ************
2022-01-26 11:55:52,329 P22435 INFO [Metrics] AUC: 0.970458 - logloss: 0.203002
2022-01-26 11:55:52,329 P22435 INFO Save best model: monitor(max): 0.970458
2022-01-26 11:55:52,332 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:55:52,377 P22435 INFO Train loss: 0.105597
2022-01-26 11:55:52,377 P22435 INFO ************ Epoch=58 end ************
2022-01-26 11:56:02,458 P22435 INFO [Metrics] AUC: 0.970639 - logloss: 0.202960
2022-01-26 11:56:02,458 P22435 INFO Save best model: monitor(max): 0.970639
2022-01-26 11:56:02,461 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:56:02,506 P22435 INFO Train loss: 0.104065
2022-01-26 11:56:02,506 P22435 INFO ************ Epoch=59 end ************
2022-01-26 11:56:12,580 P22435 INFO [Metrics] AUC: 0.970836 - logloss: 0.202735
2022-01-26 11:56:12,581 P22435 INFO Save best model: monitor(max): 0.970836
2022-01-26 11:56:12,585 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:56:12,625 P22435 INFO Train loss: 0.102449
2022-01-26 11:56:12,625 P22435 INFO ************ Epoch=60 end ************
2022-01-26 11:56:22,663 P22435 INFO [Metrics] AUC: 0.971045 - logloss: 0.202462
2022-01-26 11:56:22,663 P22435 INFO Save best model: monitor(max): 0.971045
2022-01-26 11:56:22,667 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:56:22,708 P22435 INFO Train loss: 0.100893
2022-01-26 11:56:22,708 P22435 INFO ************ Epoch=61 end ************
2022-01-26 11:56:32,781 P22435 INFO [Metrics] AUC: 0.971273 - logloss: 0.201918
2022-01-26 11:56:32,782 P22435 INFO Save best model: monitor(max): 0.971273
2022-01-26 11:56:32,785 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:56:32,826 P22435 INFO Train loss: 0.099317
2022-01-26 11:56:32,826 P22435 INFO ************ Epoch=62 end ************
2022-01-26 11:56:42,924 P22435 INFO [Metrics] AUC: 0.971466 - logloss: 0.202240
2022-01-26 11:56:42,925 P22435 INFO Save best model: monitor(max): 0.971466
2022-01-26 11:56:42,928 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:56:42,972 P22435 INFO Train loss: 0.097745
2022-01-26 11:56:42,972 P22435 INFO ************ Epoch=63 end ************
2022-01-26 11:56:53,048 P22435 INFO [Metrics] AUC: 0.971640 - logloss: 0.201765
2022-01-26 11:56:53,049 P22435 INFO Save best model: monitor(max): 0.971640
2022-01-26 11:56:53,052 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:56:53,099 P22435 INFO Train loss: 0.096309
2022-01-26 11:56:53,099 P22435 INFO ************ Epoch=64 end ************
2022-01-26 11:57:03,072 P22435 INFO [Metrics] AUC: 0.971771 - logloss: 0.201741
2022-01-26 11:57:03,073 P22435 INFO Save best model: monitor(max): 0.971771
2022-01-26 11:57:03,076 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:57:03,123 P22435 INFO Train loss: 0.095070
2022-01-26 11:57:03,123 P22435 INFO ************ Epoch=65 end ************
2022-01-26 11:57:12,727 P22435 INFO [Metrics] AUC: 0.971947 - logloss: 0.201451
2022-01-26 11:57:12,728 P22435 INFO Save best model: monitor(max): 0.971947
2022-01-26 11:57:12,731 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:57:12,775 P22435 INFO Train loss: 0.093634
2022-01-26 11:57:12,775 P22435 INFO ************ Epoch=66 end ************
2022-01-26 11:57:21,492 P22435 INFO [Metrics] AUC: 0.972138 - logloss: 0.201283
2022-01-26 11:57:21,493 P22435 INFO Save best model: monitor(max): 0.972138
2022-01-26 11:57:21,496 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:57:21,542 P22435 INFO Train loss: 0.092291
2022-01-26 11:57:21,542 P22435 INFO ************ Epoch=67 end ************
2022-01-26 11:57:30,386 P22435 INFO [Metrics] AUC: 0.972324 - logloss: 0.201213
2022-01-26 11:57:30,387 P22435 INFO Save best model: monitor(max): 0.972324
2022-01-26 11:57:30,390 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:57:30,432 P22435 INFO Train loss: 0.090725
2022-01-26 11:57:30,432 P22435 INFO ************ Epoch=68 end ************
2022-01-26 11:57:39,378 P22435 INFO [Metrics] AUC: 0.972489 - logloss: 0.201137
2022-01-26 11:57:39,378 P22435 INFO Save best model: monitor(max): 0.972489
2022-01-26 11:57:39,382 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:57:39,427 P22435 INFO Train loss: 0.089711
2022-01-26 11:57:39,427 P22435 INFO ************ Epoch=69 end ************
2022-01-26 11:57:48,480 P22435 INFO [Metrics] AUC: 0.972599 - logloss: 0.200985
2022-01-26 11:57:48,481 P22435 INFO Save best model: monitor(max): 0.972599
2022-01-26 11:57:48,484 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:57:48,528 P22435 INFO Train loss: 0.088447
2022-01-26 11:57:48,528 P22435 INFO ************ Epoch=70 end ************
2022-01-26 11:57:57,450 P22435 INFO [Metrics] AUC: 0.972736 - logloss: 0.200973
2022-01-26 11:57:57,450 P22435 INFO Save best model: monitor(max): 0.972736
2022-01-26 11:57:57,454 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:57:57,493 P22435 INFO Train loss: 0.087184
2022-01-26 11:57:57,493 P22435 INFO ************ Epoch=71 end ************
2022-01-26 11:58:06,436 P22435 INFO [Metrics] AUC: 0.972848 - logloss: 0.201374
2022-01-26 11:58:06,437 P22435 INFO Save best model: monitor(max): 0.972848
2022-01-26 11:58:06,440 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:58:06,484 P22435 INFO Train loss: 0.085840
2022-01-26 11:58:06,484 P22435 INFO ************ Epoch=72 end ************
2022-01-26 11:58:15,295 P22435 INFO [Metrics] AUC: 0.973017 - logloss: 0.201137
2022-01-26 11:58:15,295 P22435 INFO Save best model: monitor(max): 0.973017
2022-01-26 11:58:15,299 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:58:15,339 P22435 INFO Train loss: 0.084739
2022-01-26 11:58:15,339 P22435 INFO ************ Epoch=73 end ************
2022-01-26 11:58:24,544 P22435 INFO [Metrics] AUC: 0.973146 - logloss: 0.201057
2022-01-26 11:58:24,544 P22435 INFO Save best model: monitor(max): 0.973146
2022-01-26 11:58:24,548 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:58:24,588 P22435 INFO Train loss: 0.083731
2022-01-26 11:58:24,589 P22435 INFO ************ Epoch=74 end ************
2022-01-26 11:58:34,307 P22435 INFO [Metrics] AUC: 0.973256 - logloss: 0.201552
2022-01-26 11:58:34,307 P22435 INFO Save best model: monitor(max): 0.973256
2022-01-26 11:58:34,311 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:58:34,354 P22435 INFO Train loss: 0.082309
2022-01-26 11:58:34,355 P22435 INFO ************ Epoch=75 end ************
2022-01-26 11:58:42,033 P22435 INFO [Metrics] AUC: 0.973352 - logloss: 0.201195
2022-01-26 11:58:42,033 P22435 INFO Save best model: monitor(max): 0.973352
2022-01-26 11:58:42,036 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:58:42,082 P22435 INFO Train loss: 0.081339
2022-01-26 11:58:42,082 P22435 INFO ************ Epoch=76 end ************
2022-01-26 11:58:49,772 P22435 INFO [Metrics] AUC: 0.973524 - logloss: 0.201158
2022-01-26 11:58:49,772 P22435 INFO Save best model: monitor(max): 0.973524
2022-01-26 11:58:49,776 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:58:49,830 P22435 INFO Train loss: 0.080410
2022-01-26 11:58:49,830 P22435 INFO ************ Epoch=77 end ************
2022-01-26 11:58:57,483 P22435 INFO [Metrics] AUC: 0.973582 - logloss: 0.201577
2022-01-26 11:58:57,484 P22435 INFO Save best model: monitor(max): 0.973582
2022-01-26 11:58:57,487 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:58:57,528 P22435 INFO Train loss: 0.079275
2022-01-26 11:58:57,528 P22435 INFO ************ Epoch=78 end ************
2022-01-26 11:59:05,088 P22435 INFO [Metrics] AUC: 0.973734 - logloss: 0.201291
2022-01-26 11:59:05,088 P22435 INFO Save best model: monitor(max): 0.973734
2022-01-26 11:59:05,091 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:59:05,135 P22435 INFO Train loss: 0.078072
2022-01-26 11:59:05,135 P22435 INFO ************ Epoch=79 end ************
2022-01-26 11:59:12,514 P22435 INFO [Metrics] AUC: 0.973837 - logloss: 0.201458
2022-01-26 11:59:12,514 P22435 INFO Save best model: monitor(max): 0.973837
2022-01-26 11:59:12,518 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:59:12,558 P22435 INFO Train loss: 0.076984
2022-01-26 11:59:12,558 P22435 INFO ************ Epoch=80 end ************
2022-01-26 11:59:20,036 P22435 INFO [Metrics] AUC: 0.973935 - logloss: 0.201447
2022-01-26 11:59:20,037 P22435 INFO Save best model: monitor(max): 0.973935
2022-01-26 11:59:20,042 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:59:20,086 P22435 INFO Train loss: 0.076051
2022-01-26 11:59:20,087 P22435 INFO ************ Epoch=81 end ************
2022-01-26 11:59:27,571 P22435 INFO [Metrics] AUC: 0.974033 - logloss: 0.201834
2022-01-26 11:59:27,572 P22435 INFO Save best model: monitor(max): 0.974033
2022-01-26 11:59:27,575 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:59:27,623 P22435 INFO Train loss: 0.075132
2022-01-26 11:59:27,624 P22435 INFO ************ Epoch=82 end ************
2022-01-26 11:59:35,091 P22435 INFO [Metrics] AUC: 0.974075 - logloss: 0.202113
2022-01-26 11:59:35,092 P22435 INFO Save best model: monitor(max): 0.974075
2022-01-26 11:59:35,095 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:59:35,139 P22435 INFO Train loss: 0.074265
2022-01-26 11:59:35,139 P22435 INFO ************ Epoch=83 end ************
2022-01-26 11:59:42,609 P22435 INFO [Metrics] AUC: 0.974114 - logloss: 0.202510
2022-01-26 11:59:42,609 P22435 INFO Save best model: monitor(max): 0.974114
2022-01-26 11:59:42,613 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:59:42,660 P22435 INFO Train loss: 0.073227
2022-01-26 11:59:42,660 P22435 INFO ************ Epoch=84 end ************
2022-01-26 11:59:50,166 P22435 INFO [Metrics] AUC: 0.974302 - logloss: 0.202121
2022-01-26 11:59:50,167 P22435 INFO Save best model: monitor(max): 0.974302
2022-01-26 11:59:50,170 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:59:50,210 P22435 INFO Train loss: 0.072477
2022-01-26 11:59:50,210 P22435 INFO ************ Epoch=85 end ************
2022-01-26 11:59:57,662 P22435 INFO [Metrics] AUC: 0.974320 - logloss: 0.202868
2022-01-26 11:59:57,662 P22435 INFO Save best model: monitor(max): 0.974320
2022-01-26 11:59:57,665 P22435 INFO --- 50/50 batches finished ---
2022-01-26 11:59:57,705 P22435 INFO Train loss: 0.071681
2022-01-26 11:59:57,705 P22435 INFO ************ Epoch=86 end ************
2022-01-26 12:00:05,211 P22435 INFO [Metrics] AUC: 0.974372 - logloss: 0.203339
2022-01-26 12:00:05,211 P22435 INFO Save best model: monitor(max): 0.974372
2022-01-26 12:00:05,215 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:00:05,256 P22435 INFO Train loss: 0.070550
2022-01-26 12:00:05,256 P22435 INFO ************ Epoch=87 end ************
2022-01-26 12:00:12,732 P22435 INFO [Metrics] AUC: 0.974445 - logloss: 0.202987
2022-01-26 12:00:12,733 P22435 INFO Save best model: monitor(max): 0.974445
2022-01-26 12:00:12,736 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:00:12,781 P22435 INFO Train loss: 0.069915
2022-01-26 12:00:12,781 P22435 INFO ************ Epoch=88 end ************
2022-01-26 12:00:20,272 P22435 INFO [Metrics] AUC: 0.974522 - logloss: 0.203234
2022-01-26 12:00:20,272 P22435 INFO Save best model: monitor(max): 0.974522
2022-01-26 12:00:20,275 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:00:20,320 P22435 INFO Train loss: 0.068895
2022-01-26 12:00:20,320 P22435 INFO ************ Epoch=89 end ************
2022-01-26 12:00:27,808 P22435 INFO [Metrics] AUC: 0.974646 - logloss: 0.203512
2022-01-26 12:00:27,809 P22435 INFO Save best model: monitor(max): 0.974646
2022-01-26 12:00:27,812 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:00:27,854 P22435 INFO Train loss: 0.067913
2022-01-26 12:00:27,855 P22435 INFO ************ Epoch=90 end ************
2022-01-26 12:00:35,308 P22435 INFO [Metrics] AUC: 0.974641 - logloss: 0.204139
2022-01-26 12:00:35,309 P22435 INFO Monitor(max) STOP: 0.974641 !
2022-01-26 12:00:35,309 P22435 INFO Reduce learning rate on plateau: 0.000100
2022-01-26 12:00:35,309 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:00:35,351 P22435 INFO Train loss: 0.067078
2022-01-26 12:00:35,351 P22435 INFO ************ Epoch=91 end ************
2022-01-26 12:00:42,837 P22435 INFO [Metrics] AUC: 0.974692 - logloss: 0.203716
2022-01-26 12:00:42,837 P22435 INFO Save best model: monitor(max): 0.974692
2022-01-26 12:00:42,841 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:00:42,885 P22435 INFO Train loss: 0.064632
2022-01-26 12:00:42,885 P22435 INFO ************ Epoch=92 end ************
2022-01-26 12:00:50,466 P22435 INFO [Metrics] AUC: 0.974719 - logloss: 0.203641
2022-01-26 12:00:50,467 P22435 INFO Save best model: monitor(max): 0.974719
2022-01-26 12:00:50,470 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:00:50,516 P22435 INFO Train loss: 0.064437
2022-01-26 12:00:50,516 P22435 INFO ************ Epoch=93 end ************
2022-01-26 12:00:58,091 P22435 INFO [Metrics] AUC: 0.974737 - logloss: 0.203644
2022-01-26 12:00:58,091 P22435 INFO Save best model: monitor(max): 0.974737
2022-01-26 12:00:58,094 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:00:58,139 P22435 INFO Train loss: 0.064489
2022-01-26 12:00:58,140 P22435 INFO ************ Epoch=94 end ************
2022-01-26 12:01:05,706 P22435 INFO [Metrics] AUC: 0.974748 - logloss: 0.203608
2022-01-26 12:01:05,707 P22435 INFO Save best model: monitor(max): 0.974748
2022-01-26 12:01:05,711 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:01:05,751 P22435 INFO Train loss: 0.064298
2022-01-26 12:01:05,752 P22435 INFO ************ Epoch=95 end ************
2022-01-26 12:01:13,245 P22435 INFO [Metrics] AUC: 0.974760 - logloss: 0.203596
2022-01-26 12:01:13,246 P22435 INFO Save best model: monitor(max): 0.974760
2022-01-26 12:01:13,249 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:01:13,303 P22435 INFO Train loss: 0.064367
2022-01-26 12:01:13,304 P22435 INFO ************ Epoch=96 end ************
2022-01-26 12:01:20,695 P22435 INFO [Metrics] AUC: 0.974763 - logloss: 0.203662
2022-01-26 12:01:20,696 P22435 INFO Save best model: monitor(max): 0.974763
2022-01-26 12:01:20,699 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:01:20,744 P22435 INFO Train loss: 0.064087
2022-01-26 12:01:20,745 P22435 INFO ************ Epoch=97 end ************
2022-01-26 12:01:27,979 P22435 INFO [Metrics] AUC: 0.974780 - logloss: 0.203679
2022-01-26 12:01:27,980 P22435 INFO Save best model: monitor(max): 0.974780
2022-01-26 12:01:27,983 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:01:28,023 P22435 INFO Train loss: 0.064062
2022-01-26 12:01:28,023 P22435 INFO ************ Epoch=98 end ************
2022-01-26 12:01:34,805 P22435 INFO [Metrics] AUC: 0.974781 - logloss: 0.203686
2022-01-26 12:01:34,805 P22435 INFO Save best model: monitor(max): 0.974781
2022-01-26 12:01:34,809 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:01:34,853 P22435 INFO Train loss: 0.063970
2022-01-26 12:01:34,853 P22435 INFO ************ Epoch=99 end ************
2022-01-26 12:01:41,599 P22435 INFO [Metrics] AUC: 0.974790 - logloss: 0.203764
2022-01-26 12:01:41,600 P22435 INFO Save best model: monitor(max): 0.974790
2022-01-26 12:01:41,603 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:01:41,643 P22435 INFO Train loss: 0.063882
2022-01-26 12:01:41,643 P22435 INFO ************ Epoch=100 end ************
2022-01-26 12:01:48,596 P22435 INFO [Metrics] AUC: 0.974802 - logloss: 0.203715
2022-01-26 12:01:48,597 P22435 INFO Save best model: monitor(max): 0.974802
2022-01-26 12:01:48,600 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:01:48,641 P22435 INFO Train loss: 0.063896
2022-01-26 12:01:48,641 P22435 INFO ************ Epoch=101 end ************
2022-01-26 12:01:55,578 P22435 INFO [Metrics] AUC: 0.974803 - logloss: 0.203772
2022-01-26 12:01:55,578 P22435 INFO Save best model: monitor(max): 0.974803
2022-01-26 12:01:55,581 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:01:55,625 P22435 INFO Train loss: 0.063868
2022-01-26 12:01:55,626 P22435 INFO ************ Epoch=102 end ************
2022-01-26 12:02:02,556 P22435 INFO [Metrics] AUC: 0.974807 - logloss: 0.203806
2022-01-26 12:02:02,556 P22435 INFO Save best model: monitor(max): 0.974807
2022-01-26 12:02:02,559 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:02:02,603 P22435 INFO Train loss: 0.063697
2022-01-26 12:02:02,603 P22435 INFO ************ Epoch=103 end ************
2022-01-26 12:02:09,497 P22435 INFO [Metrics] AUC: 0.974822 - logloss: 0.203798
2022-01-26 12:02:09,498 P22435 INFO Save best model: monitor(max): 0.974822
2022-01-26 12:02:09,501 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:02:09,546 P22435 INFO Train loss: 0.063747
2022-01-26 12:02:09,546 P22435 INFO ************ Epoch=104 end ************
2022-01-26 12:02:16,493 P22435 INFO [Metrics] AUC: 0.974826 - logloss: 0.203885
2022-01-26 12:02:16,493 P22435 INFO Save best model: monitor(max): 0.974826
2022-01-26 12:02:16,497 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:02:16,541 P22435 INFO Train loss: 0.063544
2022-01-26 12:02:16,541 P22435 INFO ************ Epoch=105 end ************
2022-01-26 12:02:23,378 P22435 INFO [Metrics] AUC: 0.974827 - logloss: 0.203904
2022-01-26 12:02:23,378 P22435 INFO Save best model: monitor(max): 0.974827
2022-01-26 12:02:23,381 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:02:23,426 P22435 INFO Train loss: 0.063521
2022-01-26 12:02:23,426 P22435 INFO ************ Epoch=106 end ************
2022-01-26 12:02:30,337 P22435 INFO [Metrics] AUC: 0.974839 - logloss: 0.204031
2022-01-26 12:02:30,337 P22435 INFO Save best model: monitor(max): 0.974839
2022-01-26 12:02:30,341 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:02:30,386 P22435 INFO Train loss: 0.063185
2022-01-26 12:02:30,387 P22435 INFO ************ Epoch=107 end ************
2022-01-26 12:02:37,198 P22435 INFO [Metrics] AUC: 0.974846 - logloss: 0.203951
2022-01-26 12:02:37,198 P22435 INFO Save best model: monitor(max): 0.974846
2022-01-26 12:02:37,202 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:02:37,247 P22435 INFO Train loss: 0.063147
2022-01-26 12:02:37,247 P22435 INFO ************ Epoch=108 end ************
2022-01-26 12:02:44,398 P22435 INFO [Metrics] AUC: 0.974856 - logloss: 0.203999
2022-01-26 12:02:44,399 P22435 INFO Save best model: monitor(max): 0.974856
2022-01-26 12:02:44,402 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:02:44,447 P22435 INFO Train loss: 0.063044
2022-01-26 12:02:44,447 P22435 INFO ************ Epoch=109 end ************
2022-01-26 12:02:49,650 P22435 INFO [Metrics] AUC: 0.974858 - logloss: 0.204082
2022-01-26 12:02:49,650 P22435 INFO Save best model: monitor(max): 0.974858
2022-01-26 12:02:49,654 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:02:49,697 P22435 INFO Train loss: 0.063272
2022-01-26 12:02:49,697 P22435 INFO ************ Epoch=110 end ************
2022-01-26 12:02:54,829 P22435 INFO [Metrics] AUC: 0.974872 - logloss: 0.203997
2022-01-26 12:02:54,830 P22435 INFO Save best model: monitor(max): 0.974872
2022-01-26 12:02:54,833 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:02:54,880 P22435 INFO Train loss: 0.062914
2022-01-26 12:02:54,880 P22435 INFO ************ Epoch=111 end ************
2022-01-26 12:03:00,014 P22435 INFO [Metrics] AUC: 0.974877 - logloss: 0.204124
2022-01-26 12:03:00,015 P22435 INFO Save best model: monitor(max): 0.974877
2022-01-26 12:03:00,020 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:03:00,063 P22435 INFO Train loss: 0.062850
2022-01-26 12:03:00,063 P22435 INFO ************ Epoch=112 end ************
2022-01-26 12:03:05,072 P22435 INFO [Metrics] AUC: 0.974885 - logloss: 0.204108
2022-01-26 12:03:05,072 P22435 INFO Save best model: monitor(max): 0.974885
2022-01-26 12:03:05,076 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:03:05,122 P22435 INFO Train loss: 0.062798
2022-01-26 12:03:05,122 P22435 INFO ************ Epoch=113 end ************
2022-01-26 12:03:10,196 P22435 INFO [Metrics] AUC: 0.974882 - logloss: 0.204226
2022-01-26 12:03:10,196 P22435 INFO Monitor(max) STOP: 0.974882 !
2022-01-26 12:03:10,196 P22435 INFO Reduce learning rate on plateau: 0.000010
2022-01-26 12:03:10,196 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:03:10,235 P22435 INFO Train loss: 0.062810
2022-01-26 12:03:10,235 P22435 INFO ************ Epoch=114 end ************
2022-01-26 12:03:15,484 P22435 INFO [Metrics] AUC: 0.974885 - logloss: 0.204208
2022-01-26 12:03:15,484 P22435 INFO Monitor(max) STOP: 0.974885 !
2022-01-26 12:03:15,484 P22435 INFO Reduce learning rate on plateau: 0.000001
2022-01-26 12:03:15,484 P22435 INFO Early stopping at epoch=115
2022-01-26 12:03:15,484 P22435 INFO --- 50/50 batches finished ---
2022-01-26 12:03:15,527 P22435 INFO Train loss: 0.062453
2022-01-26 12:03:15,527 P22435 INFO Training finished.
2022-01-26 12:03:15,527 P22435 INFO Load best model: /home/ma-user/work/FuxiCTRv1.1/benchmarks/Frappe/HOFM_frappe_x1/frappe_x1_04e961e9/HOFM_frappe_x1_003_64d2d04e.model
2022-01-26 12:03:15,552 P22435 INFO ****** Validation evaluation ******
2022-01-26 12:03:15,954 P22435 INFO [Metrics] AUC: 0.974885 - logloss: 0.204108
2022-01-26 12:03:15,996 P22435 INFO ******** Test evaluation ********
2022-01-26 12:03:15,996 P22435 INFO Loading data...
2022-01-26 12:03:15,997 P22435 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-01-26 12:03:15,999 P22435 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-01-26 12:03:16,000 P22435 INFO Loading test data done.
2022-01-26 12:03:16,271 P22435 INFO [Metrics] AUC: 0.974228 - logloss: 0.207269

```
