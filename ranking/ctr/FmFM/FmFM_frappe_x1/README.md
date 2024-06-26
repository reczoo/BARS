## FmFM_frappe_x1

A hands-on guide to run the FmFM model on the Frappe_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

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
Dataset ID: [Frappe_x1](https://github.com/reczoo/Datasets/tree/main/Frappe/Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [FmFM](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FmFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FmFM_frappe_x1_tuner_config_02](./FmFM_frappe_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FmFM_frappe_x1
    nohup python run_expid.py --config ./FmFM_frappe_x1_tuner_config_02 --expid FmFM_frappe_x1_004_2445bd6f --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.974947 | 0.200439  |


### Logs
```python
2022-01-20 08:15:48,659 P11986 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "embedding_dim": "10",
    "epochs": "200",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "field_interaction_type": "vectorized",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FmFM",
    "model_id": "FmFM_frappe_x1_004_2445bd6f",
    "model_root": "./Frappe/FmFM_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-05",
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
2022-01-20 08:15:48,660 P11986 INFO Set up feature encoder...
2022-01-20 08:15:48,660 P11986 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-01-20 08:15:48,667 P11986 INFO Loading data...
2022-01-20 08:15:48,669 P11986 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-01-20 08:15:48,727 P11986 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-01-20 08:15:48,732 P11986 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-01-20 08:15:48,732 P11986 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-01-20 08:15:48,732 P11986 INFO Loading train data done.
2022-01-20 08:15:51,560 P11986 INFO Total number of parameters: 59730.
2022-01-20 08:15:51,560 P11986 INFO Start training: 50 batches/epoch
2022-01-20 08:15:51,560 P11986 INFO ************ Epoch=1 start ************
2022-01-20 08:15:53,735 P11986 INFO [Metrics] AUC: 0.904094 - logloss: 0.618887
2022-01-20 08:15:53,736 P11986 INFO Save best model: monitor(max): 0.904094
2022-01-20 08:15:53,738 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:15:53,789 P11986 INFO Train loss: 0.655370
2022-01-20 08:15:53,789 P11986 INFO ************ Epoch=1 end ************
2022-01-20 08:15:55,940 P11986 INFO [Metrics] AUC: 0.927094 - logloss: 0.550056
2022-01-20 08:15:55,940 P11986 INFO Save best model: monitor(max): 0.927094
2022-01-20 08:15:55,942 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:15:55,989 P11986 INFO Train loss: 0.591712
2022-01-20 08:15:55,989 P11986 INFO ************ Epoch=2 end ************
2022-01-20 08:15:58,137 P11986 INFO [Metrics] AUC: 0.930897 - logloss: 0.419149
2022-01-20 08:15:58,137 P11986 INFO Save best model: monitor(max): 0.930897
2022-01-20 08:15:58,139 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:15:58,185 P11986 INFO Train loss: 0.490991
2022-01-20 08:15:58,186 P11986 INFO ************ Epoch=3 end ************
2022-01-20 08:16:00,367 P11986 INFO [Metrics] AUC: 0.934061 - logloss: 0.320036
2022-01-20 08:16:00,367 P11986 INFO Save best model: monitor(max): 0.934061
2022-01-20 08:16:00,370 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:00,416 P11986 INFO Train loss: 0.362704
2022-01-20 08:16:00,416 P11986 INFO ************ Epoch=4 end ************
2022-01-20 08:16:02,562 P11986 INFO [Metrics] AUC: 0.936737 - logloss: 0.291636
2022-01-20 08:16:02,562 P11986 INFO Save best model: monitor(max): 0.936737
2022-01-20 08:16:02,564 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:02,611 P11986 INFO Train loss: 0.300310
2022-01-20 08:16:02,612 P11986 INFO ************ Epoch=5 end ************
2022-01-20 08:16:04,788 P11986 INFO [Metrics] AUC: 0.938073 - logloss: 0.284535
2022-01-20 08:16:04,789 P11986 INFO Save best model: monitor(max): 0.938073
2022-01-20 08:16:04,791 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:04,838 P11986 INFO Train loss: 0.282784
2022-01-20 08:16:04,838 P11986 INFO ************ Epoch=6 end ************
2022-01-20 08:16:07,031 P11986 INFO [Metrics] AUC: 0.938738 - logloss: 0.282221
2022-01-20 08:16:07,031 P11986 INFO Save best model: monitor(max): 0.938738
2022-01-20 08:16:07,033 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:07,080 P11986 INFO Train loss: 0.276606
2022-01-20 08:16:07,080 P11986 INFO ************ Epoch=7 end ************
2022-01-20 08:16:09,281 P11986 INFO [Metrics] AUC: 0.938918 - logloss: 0.281305
2022-01-20 08:16:09,282 P11986 INFO Save best model: monitor(max): 0.938918
2022-01-20 08:16:09,284 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:09,331 P11986 INFO Train loss: 0.273791
2022-01-20 08:16:09,331 P11986 INFO ************ Epoch=8 end ************
2022-01-20 08:16:11,471 P11986 INFO [Metrics] AUC: 0.939030 - logloss: 0.281024
2022-01-20 08:16:11,472 P11986 INFO Save best model: monitor(max): 0.939030
2022-01-20 08:16:11,474 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:11,520 P11986 INFO Train loss: 0.271881
2022-01-20 08:16:11,520 P11986 INFO ************ Epoch=9 end ************
2022-01-20 08:16:13,662 P11986 INFO [Metrics] AUC: 0.939094 - logloss: 0.280727
2022-01-20 08:16:13,662 P11986 INFO Save best model: monitor(max): 0.939094
2022-01-20 08:16:13,664 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:13,713 P11986 INFO Train loss: 0.271102
2022-01-20 08:16:13,713 P11986 INFO ************ Epoch=10 end ************
2022-01-20 08:16:15,853 P11986 INFO [Metrics] AUC: 0.939168 - logloss: 0.280616
2022-01-20 08:16:15,853 P11986 INFO Save best model: monitor(max): 0.939168
2022-01-20 08:16:15,855 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:15,902 P11986 INFO Train loss: 0.270065
2022-01-20 08:16:15,902 P11986 INFO ************ Epoch=11 end ************
2022-01-20 08:16:18,049 P11986 INFO [Metrics] AUC: 0.939188 - logloss: 0.280626
2022-01-20 08:16:18,049 P11986 INFO Save best model: monitor(max): 0.939188
2022-01-20 08:16:18,051 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:18,099 P11986 INFO Train loss: 0.269268
2022-01-20 08:16:18,099 P11986 INFO ************ Epoch=12 end ************
2022-01-20 08:16:20,238 P11986 INFO [Metrics] AUC: 0.939250 - logloss: 0.280667
2022-01-20 08:16:20,238 P11986 INFO Save best model: monitor(max): 0.939250
2022-01-20 08:16:20,240 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:20,285 P11986 INFO Train loss: 0.268633
2022-01-20 08:16:20,285 P11986 INFO ************ Epoch=13 end ************
2022-01-20 08:16:22,422 P11986 INFO [Metrics] AUC: 0.939388 - logloss: 0.280392
2022-01-20 08:16:22,422 P11986 INFO Save best model: monitor(max): 0.939388
2022-01-20 08:16:22,424 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:22,470 P11986 INFO Train loss: 0.268517
2022-01-20 08:16:22,470 P11986 INFO ************ Epoch=14 end ************
2022-01-20 08:16:24,627 P11986 INFO [Metrics] AUC: 0.939433 - logloss: 0.280228
2022-01-20 08:16:24,628 P11986 INFO Save best model: monitor(max): 0.939433
2022-01-20 08:16:24,630 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:24,684 P11986 INFO Train loss: 0.267670
2022-01-20 08:16:24,684 P11986 INFO ************ Epoch=15 end ************
2022-01-20 08:16:26,887 P11986 INFO [Metrics] AUC: 0.939504 - logloss: 0.280131
2022-01-20 08:16:26,887 P11986 INFO Save best model: monitor(max): 0.939504
2022-01-20 08:16:26,889 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:26,936 P11986 INFO Train loss: 0.267402
2022-01-20 08:16:26,936 P11986 INFO ************ Epoch=16 end ************
2022-01-20 08:16:29,230 P11986 INFO [Metrics] AUC: 0.939593 - logloss: 0.280013
2022-01-20 08:16:29,231 P11986 INFO Save best model: monitor(max): 0.939593
2022-01-20 08:16:29,233 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:29,280 P11986 INFO Train loss: 0.267037
2022-01-20 08:16:29,281 P11986 INFO ************ Epoch=17 end ************
2022-01-20 08:16:31,541 P11986 INFO [Metrics] AUC: 0.939724 - logloss: 0.279821
2022-01-20 08:16:31,541 P11986 INFO Save best model: monitor(max): 0.939724
2022-01-20 08:16:31,543 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:31,591 P11986 INFO Train loss: 0.266802
2022-01-20 08:16:31,591 P11986 INFO ************ Epoch=18 end ************
2022-01-20 08:16:33,753 P11986 INFO [Metrics] AUC: 0.939748 - logloss: 0.279795
2022-01-20 08:16:33,753 P11986 INFO Save best model: monitor(max): 0.939748
2022-01-20 08:16:33,755 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:33,804 P11986 INFO Train loss: 0.266102
2022-01-20 08:16:33,804 P11986 INFO ************ Epoch=19 end ************
2022-01-20 08:16:36,019 P11986 INFO [Metrics] AUC: 0.939964 - logloss: 0.279420
2022-01-20 08:16:36,020 P11986 INFO Save best model: monitor(max): 0.939964
2022-01-20 08:16:36,022 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:36,069 P11986 INFO Train loss: 0.265990
2022-01-20 08:16:36,070 P11986 INFO ************ Epoch=20 end ************
2022-01-20 08:16:38,237 P11986 INFO [Metrics] AUC: 0.940064 - logloss: 0.279181
2022-01-20 08:16:38,238 P11986 INFO Save best model: monitor(max): 0.940064
2022-01-20 08:16:38,240 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:38,310 P11986 INFO Train loss: 0.265531
2022-01-20 08:16:38,310 P11986 INFO ************ Epoch=21 end ************
2022-01-20 08:16:40,488 P11986 INFO [Metrics] AUC: 0.940267 - logloss: 0.278876
2022-01-20 08:16:40,488 P11986 INFO Save best model: monitor(max): 0.940267
2022-01-20 08:16:40,490 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:40,536 P11986 INFO Train loss: 0.264866
2022-01-20 08:16:40,536 P11986 INFO ************ Epoch=22 end ************
2022-01-20 08:16:42,711 P11986 INFO [Metrics] AUC: 0.940480 - logloss: 0.278431
2022-01-20 08:16:42,711 P11986 INFO Save best model: monitor(max): 0.940480
2022-01-20 08:16:42,713 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:42,761 P11986 INFO Train loss: 0.264474
2022-01-20 08:16:42,761 P11986 INFO ************ Epoch=23 end ************
2022-01-20 08:16:44,961 P11986 INFO [Metrics] AUC: 0.940591 - logloss: 0.278198
2022-01-20 08:16:44,961 P11986 INFO Save best model: monitor(max): 0.940591
2022-01-20 08:16:44,963 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:45,010 P11986 INFO Train loss: 0.264150
2022-01-20 08:16:45,010 P11986 INFO ************ Epoch=24 end ************
2022-01-20 08:16:47,210 P11986 INFO [Metrics] AUC: 0.940816 - logloss: 0.277760
2022-01-20 08:16:47,210 P11986 INFO Save best model: monitor(max): 0.940816
2022-01-20 08:16:47,212 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:47,259 P11986 INFO Train loss: 0.263814
2022-01-20 08:16:47,259 P11986 INFO ************ Epoch=25 end ************
2022-01-20 08:16:49,482 P11986 INFO [Metrics] AUC: 0.941060 - logloss: 0.277318
2022-01-20 08:16:49,483 P11986 INFO Save best model: monitor(max): 0.941060
2022-01-20 08:16:49,485 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:49,532 P11986 INFO Train loss: 0.262702
2022-01-20 08:16:49,532 P11986 INFO ************ Epoch=26 end ************
2022-01-20 08:16:51,744 P11986 INFO [Metrics] AUC: 0.941322 - logloss: 0.276849
2022-01-20 08:16:51,745 P11986 INFO Save best model: monitor(max): 0.941322
2022-01-20 08:16:51,747 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:51,794 P11986 INFO Train loss: 0.262422
2022-01-20 08:16:51,794 P11986 INFO ************ Epoch=27 end ************
2022-01-20 08:16:53,966 P11986 INFO [Metrics] AUC: 0.941632 - logloss: 0.276194
2022-01-20 08:16:53,967 P11986 INFO Save best model: monitor(max): 0.941632
2022-01-20 08:16:53,969 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:54,016 P11986 INFO Train loss: 0.261519
2022-01-20 08:16:54,016 P11986 INFO ************ Epoch=28 end ************
2022-01-20 08:16:56,176 P11986 INFO [Metrics] AUC: 0.941943 - logloss: 0.275501
2022-01-20 08:16:56,176 P11986 INFO Save best model: monitor(max): 0.941943
2022-01-20 08:16:56,178 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:56,225 P11986 INFO Train loss: 0.260848
2022-01-20 08:16:56,225 P11986 INFO ************ Epoch=29 end ************
2022-01-20 08:16:58,401 P11986 INFO [Metrics] AUC: 0.942290 - logloss: 0.274665
2022-01-20 08:16:58,401 P11986 INFO Save best model: monitor(max): 0.942290
2022-01-20 08:16:58,403 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:16:58,451 P11986 INFO Train loss: 0.260061
2022-01-20 08:16:58,451 P11986 INFO ************ Epoch=30 end ************
2022-01-20 08:17:00,627 P11986 INFO [Metrics] AUC: 0.942650 - logloss: 0.273793
2022-01-20 08:17:00,628 P11986 INFO Save best model: monitor(max): 0.942650
2022-01-20 08:17:00,630 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:00,677 P11986 INFO Train loss: 0.258666
2022-01-20 08:17:00,677 P11986 INFO ************ Epoch=31 end ************
2022-01-20 08:17:02,874 P11986 INFO [Metrics] AUC: 0.943142 - logloss: 0.272731
2022-01-20 08:17:02,874 P11986 INFO Save best model: monitor(max): 0.943142
2022-01-20 08:17:02,876 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:02,924 P11986 INFO Train loss: 0.257486
2022-01-20 08:17:02,924 P11986 INFO ************ Epoch=32 end ************
2022-01-20 08:17:05,099 P11986 INFO [Metrics] AUC: 0.943652 - logloss: 0.271561
2022-01-20 08:17:05,100 P11986 INFO Save best model: monitor(max): 0.943652
2022-01-20 08:17:05,102 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:05,148 P11986 INFO Train loss: 0.255935
2022-01-20 08:17:05,148 P11986 INFO ************ Epoch=33 end ************
2022-01-20 08:17:07,330 P11986 INFO [Metrics] AUC: 0.944226 - logloss: 0.270127
2022-01-20 08:17:07,331 P11986 INFO Save best model: monitor(max): 0.944226
2022-01-20 08:17:07,333 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:07,380 P11986 INFO Train loss: 0.254632
2022-01-20 08:17:07,380 P11986 INFO ************ Epoch=34 end ************
2022-01-20 08:17:09,542 P11986 INFO [Metrics] AUC: 0.944983 - logloss: 0.268403
2022-01-20 08:17:09,542 P11986 INFO Save best model: monitor(max): 0.944983
2022-01-20 08:17:09,544 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:09,591 P11986 INFO Train loss: 0.252818
2022-01-20 08:17:09,592 P11986 INFO ************ Epoch=35 end ************
2022-01-20 08:17:11,751 P11986 INFO [Metrics] AUC: 0.945677 - logloss: 0.266670
2022-01-20 08:17:11,752 P11986 INFO Save best model: monitor(max): 0.945677
2022-01-20 08:17:11,754 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:11,801 P11986 INFO Train loss: 0.250607
2022-01-20 08:17:11,801 P11986 INFO ************ Epoch=36 end ************
2022-01-20 08:17:13,962 P11986 INFO [Metrics] AUC: 0.946445 - logloss: 0.264669
2022-01-20 08:17:13,963 P11986 INFO Save best model: monitor(max): 0.946445
2022-01-20 08:17:13,965 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:14,012 P11986 INFO Train loss: 0.247913
2022-01-20 08:17:14,012 P11986 INFO ************ Epoch=37 end ************
2022-01-20 08:17:16,202 P11986 INFO [Metrics] AUC: 0.947401 - logloss: 0.262302
2022-01-20 08:17:16,202 P11986 INFO Save best model: monitor(max): 0.947401
2022-01-20 08:17:16,204 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:16,252 P11986 INFO Train loss: 0.245385
2022-01-20 08:17:16,252 P11986 INFO ************ Epoch=38 end ************
2022-01-20 08:17:18,445 P11986 INFO [Metrics] AUC: 0.948421 - logloss: 0.259679
2022-01-20 08:17:18,446 P11986 INFO Save best model: monitor(max): 0.948421
2022-01-20 08:17:18,448 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:18,493 P11986 INFO Train loss: 0.241884
2022-01-20 08:17:18,494 P11986 INFO ************ Epoch=39 end ************
2022-01-20 08:17:20,704 P11986 INFO [Metrics] AUC: 0.949552 - logloss: 0.256898
2022-01-20 08:17:20,704 P11986 INFO Save best model: monitor(max): 0.949552
2022-01-20 08:17:20,706 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:20,754 P11986 INFO Train loss: 0.238753
2022-01-20 08:17:20,754 P11986 INFO ************ Epoch=40 end ************
2022-01-20 08:17:22,938 P11986 INFO [Metrics] AUC: 0.950722 - logloss: 0.253741
2022-01-20 08:17:22,938 P11986 INFO Save best model: monitor(max): 0.950722
2022-01-20 08:17:22,940 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:22,987 P11986 INFO Train loss: 0.235340
2022-01-20 08:17:22,987 P11986 INFO ************ Epoch=41 end ************
2022-01-20 08:17:25,163 P11986 INFO [Metrics] AUC: 0.951910 - logloss: 0.250532
2022-01-20 08:17:25,163 P11986 INFO Save best model: monitor(max): 0.951910
2022-01-20 08:17:25,166 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:25,214 P11986 INFO Train loss: 0.231027
2022-01-20 08:17:25,214 P11986 INFO ************ Epoch=42 end ************
2022-01-20 08:17:27,388 P11986 INFO [Metrics] AUC: 0.953143 - logloss: 0.247124
2022-01-20 08:17:27,388 P11986 INFO Save best model: monitor(max): 0.953143
2022-01-20 08:17:27,390 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:27,440 P11986 INFO Train loss: 0.226204
2022-01-20 08:17:27,440 P11986 INFO ************ Epoch=43 end ************
2022-01-20 08:17:29,604 P11986 INFO [Metrics] AUC: 0.954480 - logloss: 0.243577
2022-01-20 08:17:29,604 P11986 INFO Save best model: monitor(max): 0.954480
2022-01-20 08:17:29,606 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:29,653 P11986 INFO Train loss: 0.221588
2022-01-20 08:17:29,653 P11986 INFO ************ Epoch=44 end ************
2022-01-20 08:17:31,827 P11986 INFO [Metrics] AUC: 0.955799 - logloss: 0.239962
2022-01-20 08:17:31,827 P11986 INFO Save best model: monitor(max): 0.955799
2022-01-20 08:17:31,829 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:31,877 P11986 INFO Train loss: 0.216704
2022-01-20 08:17:31,877 P11986 INFO ************ Epoch=45 end ************
2022-01-20 08:17:34,026 P11986 INFO [Metrics] AUC: 0.957044 - logloss: 0.236565
2022-01-20 08:17:34,026 P11986 INFO Save best model: monitor(max): 0.957044
2022-01-20 08:17:34,028 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:34,075 P11986 INFO Train loss: 0.212256
2022-01-20 08:17:34,076 P11986 INFO ************ Epoch=46 end ************
2022-01-20 08:17:36,255 P11986 INFO [Metrics] AUC: 0.958290 - logloss: 0.232977
2022-01-20 08:17:36,255 P11986 INFO Save best model: monitor(max): 0.958290
2022-01-20 08:17:36,257 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:36,304 P11986 INFO Train loss: 0.206856
2022-01-20 08:17:36,304 P11986 INFO ************ Epoch=47 end ************
2022-01-20 08:17:38,497 P11986 INFO [Metrics] AUC: 0.959514 - logloss: 0.229415
2022-01-20 08:17:38,498 P11986 INFO Save best model: monitor(max): 0.959514
2022-01-20 08:17:38,500 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:38,546 P11986 INFO Train loss: 0.202164
2022-01-20 08:17:38,546 P11986 INFO ************ Epoch=48 end ************
2022-01-20 08:17:40,693 P11986 INFO [Metrics] AUC: 0.960733 - logloss: 0.226081
2022-01-20 08:17:40,694 P11986 INFO Save best model: monitor(max): 0.960733
2022-01-20 08:17:40,696 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:40,742 P11986 INFO Train loss: 0.197263
2022-01-20 08:17:40,743 P11986 INFO ************ Epoch=49 end ************
2022-01-20 08:17:42,900 P11986 INFO [Metrics] AUC: 0.961886 - logloss: 0.222794
2022-01-20 08:17:42,900 P11986 INFO Save best model: monitor(max): 0.961886
2022-01-20 08:17:42,902 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:42,949 P11986 INFO Train loss: 0.192341
2022-01-20 08:17:42,949 P11986 INFO ************ Epoch=50 end ************
2022-01-20 08:17:45,121 P11986 INFO [Metrics] AUC: 0.962986 - logloss: 0.219563
2022-01-20 08:17:45,121 P11986 INFO Save best model: monitor(max): 0.962986
2022-01-20 08:17:45,123 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:45,171 P11986 INFO Train loss: 0.187282
2022-01-20 08:17:45,171 P11986 INFO ************ Epoch=51 end ************
2022-01-20 08:17:47,342 P11986 INFO [Metrics] AUC: 0.963975 - logloss: 0.216718
2022-01-20 08:17:47,343 P11986 INFO Save best model: monitor(max): 0.963975
2022-01-20 08:17:47,345 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:47,392 P11986 INFO Train loss: 0.182807
2022-01-20 08:17:47,392 P11986 INFO ************ Epoch=52 end ************
2022-01-20 08:17:49,544 P11986 INFO [Metrics] AUC: 0.964911 - logloss: 0.213936
2022-01-20 08:17:49,545 P11986 INFO Save best model: monitor(max): 0.964911
2022-01-20 08:17:49,547 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:49,594 P11986 INFO Train loss: 0.178270
2022-01-20 08:17:49,594 P11986 INFO ************ Epoch=53 end ************
2022-01-20 08:17:51,804 P11986 INFO [Metrics] AUC: 0.965784 - logloss: 0.211406
2022-01-20 08:17:51,805 P11986 INFO Save best model: monitor(max): 0.965784
2022-01-20 08:17:51,807 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:51,854 P11986 INFO Train loss: 0.173901
2022-01-20 08:17:51,854 P11986 INFO ************ Epoch=54 end ************
2022-01-20 08:17:54,036 P11986 INFO [Metrics] AUC: 0.966567 - logloss: 0.209223
2022-01-20 08:17:54,037 P11986 INFO Save best model: monitor(max): 0.966567
2022-01-20 08:17:54,039 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:54,086 P11986 INFO Train loss: 0.170023
2022-01-20 08:17:54,086 P11986 INFO ************ Epoch=55 end ************
2022-01-20 08:17:56,242 P11986 INFO [Metrics] AUC: 0.967256 - logloss: 0.207170
2022-01-20 08:17:56,243 P11986 INFO Save best model: monitor(max): 0.967256
2022-01-20 08:17:56,245 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:56,292 P11986 INFO Train loss: 0.166106
2022-01-20 08:17:56,292 P11986 INFO ************ Epoch=56 end ************
2022-01-20 08:17:58,444 P11986 INFO [Metrics] AUC: 0.967903 - logloss: 0.205452
2022-01-20 08:17:58,444 P11986 INFO Save best model: monitor(max): 0.967903
2022-01-20 08:17:58,446 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:17:58,493 P11986 INFO Train loss: 0.162832
2022-01-20 08:17:58,494 P11986 INFO ************ Epoch=57 end ************
2022-01-20 08:18:00,665 P11986 INFO [Metrics] AUC: 0.968438 - logloss: 0.203906
2022-01-20 08:18:00,666 P11986 INFO Save best model: monitor(max): 0.968438
2022-01-20 08:18:00,668 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:00,714 P11986 INFO Train loss: 0.159553
2022-01-20 08:18:00,714 P11986 INFO ************ Epoch=58 end ************
2022-01-20 08:18:02,875 P11986 INFO [Metrics] AUC: 0.968928 - logloss: 0.202626
2022-01-20 08:18:02,875 P11986 INFO Save best model: monitor(max): 0.968928
2022-01-20 08:18:02,877 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:02,924 P11986 INFO Train loss: 0.156546
2022-01-20 08:18:02,924 P11986 INFO ************ Epoch=59 end ************
2022-01-20 08:18:05,092 P11986 INFO [Metrics] AUC: 0.969417 - logloss: 0.201270
2022-01-20 08:18:05,092 P11986 INFO Save best model: monitor(max): 0.969417
2022-01-20 08:18:05,094 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:05,141 P11986 INFO Train loss: 0.153423
2022-01-20 08:18:05,141 P11986 INFO ************ Epoch=60 end ************
2022-01-20 08:18:07,346 P11986 INFO [Metrics] AUC: 0.969839 - logloss: 0.200411
2022-01-20 08:18:07,346 P11986 INFO Save best model: monitor(max): 0.969839
2022-01-20 08:18:07,348 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:07,397 P11986 INFO Train loss: 0.151064
2022-01-20 08:18:07,397 P11986 INFO ************ Epoch=61 end ************
2022-01-20 08:18:09,543 P11986 INFO [Metrics] AUC: 0.970247 - logloss: 0.199289
2022-01-20 08:18:09,544 P11986 INFO Save best model: monitor(max): 0.970247
2022-01-20 08:18:09,546 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:09,593 P11986 INFO Train loss: 0.148640
2022-01-20 08:18:09,593 P11986 INFO ************ Epoch=62 end ************
2022-01-20 08:18:11,759 P11986 INFO [Metrics] AUC: 0.970524 - logloss: 0.198651
2022-01-20 08:18:11,759 P11986 INFO Save best model: monitor(max): 0.970524
2022-01-20 08:18:11,761 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:11,807 P11986 INFO Train loss: 0.146312
2022-01-20 08:18:11,808 P11986 INFO ************ Epoch=63 end ************
2022-01-20 08:18:13,955 P11986 INFO [Metrics] AUC: 0.970863 - logloss: 0.197813
2022-01-20 08:18:13,955 P11986 INFO Save best model: monitor(max): 0.970863
2022-01-20 08:18:13,957 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:14,004 P11986 INFO Train loss: 0.143991
2022-01-20 08:18:14,004 P11986 INFO ************ Epoch=64 end ************
2022-01-20 08:18:16,155 P11986 INFO [Metrics] AUC: 0.971125 - logloss: 0.197374
2022-01-20 08:18:16,156 P11986 INFO Save best model: monitor(max): 0.971125
2022-01-20 08:18:16,158 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:16,204 P11986 INFO Train loss: 0.141897
2022-01-20 08:18:16,204 P11986 INFO ************ Epoch=65 end ************
2022-01-20 08:18:18,391 P11986 INFO [Metrics] AUC: 0.971406 - logloss: 0.196940
2022-01-20 08:18:18,391 P11986 INFO Save best model: monitor(max): 0.971406
2022-01-20 08:18:18,393 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:18,440 P11986 INFO Train loss: 0.139987
2022-01-20 08:18:18,441 P11986 INFO ************ Epoch=66 end ************
2022-01-20 08:18:20,641 P11986 INFO [Metrics] AUC: 0.971658 - logloss: 0.196361
2022-01-20 08:18:20,641 P11986 INFO Save best model: monitor(max): 0.971658
2022-01-20 08:18:20,643 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:20,691 P11986 INFO Train loss: 0.138293
2022-01-20 08:18:20,691 P11986 INFO ************ Epoch=67 end ************
2022-01-20 08:18:22,876 P11986 INFO [Metrics] AUC: 0.971821 - logloss: 0.196094
2022-01-20 08:18:22,876 P11986 INFO Save best model: monitor(max): 0.971821
2022-01-20 08:18:22,878 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:22,925 P11986 INFO Train loss: 0.136646
2022-01-20 08:18:22,925 P11986 INFO ************ Epoch=68 end ************
2022-01-20 08:18:25,123 P11986 INFO [Metrics] AUC: 0.972102 - logloss: 0.195543
2022-01-20 08:18:25,123 P11986 INFO Save best model: monitor(max): 0.972102
2022-01-20 08:18:25,125 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:25,172 P11986 INFO Train loss: 0.134752
2022-01-20 08:18:25,172 P11986 INFO ************ Epoch=69 end ************
2022-01-20 08:18:27,363 P11986 INFO [Metrics] AUC: 0.972275 - logloss: 0.195390
2022-01-20 08:18:27,363 P11986 INFO Save best model: monitor(max): 0.972275
2022-01-20 08:18:27,365 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:27,412 P11986 INFO Train loss: 0.133398
2022-01-20 08:18:27,413 P11986 INFO ************ Epoch=70 end ************
2022-01-20 08:18:29,606 P11986 INFO [Metrics] AUC: 0.972478 - logloss: 0.195017
2022-01-20 08:18:29,607 P11986 INFO Save best model: monitor(max): 0.972478
2022-01-20 08:18:29,609 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:29,656 P11986 INFO Train loss: 0.131647
2022-01-20 08:18:29,656 P11986 INFO ************ Epoch=71 end ************
2022-01-20 08:18:31,856 P11986 INFO [Metrics] AUC: 0.972636 - logloss: 0.194990
2022-01-20 08:18:31,856 P11986 INFO Save best model: monitor(max): 0.972636
2022-01-20 08:18:31,858 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:31,906 P11986 INFO Train loss: 0.130170
2022-01-20 08:18:31,906 P11986 INFO ************ Epoch=72 end ************
2022-01-20 08:18:34,083 P11986 INFO [Metrics] AUC: 0.972797 - logloss: 0.195153
2022-01-20 08:18:34,084 P11986 INFO Save best model: monitor(max): 0.972797
2022-01-20 08:18:34,086 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:34,132 P11986 INFO Train loss: 0.128821
2022-01-20 08:18:34,132 P11986 INFO ************ Epoch=73 end ************
2022-01-20 08:18:36,335 P11986 INFO [Metrics] AUC: 0.972969 - logloss: 0.194794
2022-01-20 08:18:36,335 P11986 INFO Save best model: monitor(max): 0.972969
2022-01-20 08:18:36,338 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:36,383 P11986 INFO Train loss: 0.126995
2022-01-20 08:18:36,383 P11986 INFO ************ Epoch=74 end ************
2022-01-20 08:18:38,594 P11986 INFO [Metrics] AUC: 0.973172 - logloss: 0.194245
2022-01-20 08:18:38,595 P11986 INFO Save best model: monitor(max): 0.973172
2022-01-20 08:18:38,597 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:38,644 P11986 INFO Train loss: 0.125873
2022-01-20 08:18:38,644 P11986 INFO ************ Epoch=75 end ************
2022-01-20 08:18:40,849 P11986 INFO [Metrics] AUC: 0.973271 - logloss: 0.194313
2022-01-20 08:18:40,849 P11986 INFO Save best model: monitor(max): 0.973271
2022-01-20 08:18:40,851 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:40,898 P11986 INFO Train loss: 0.124306
2022-01-20 08:18:40,898 P11986 INFO ************ Epoch=76 end ************
2022-01-20 08:18:43,087 P11986 INFO [Metrics] AUC: 0.973392 - logloss: 0.194221
2022-01-20 08:18:43,088 P11986 INFO Save best model: monitor(max): 0.973392
2022-01-20 08:18:43,090 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:43,137 P11986 INFO Train loss: 0.123167
2022-01-20 08:18:43,137 P11986 INFO ************ Epoch=77 end ************
2022-01-20 08:18:45,313 P11986 INFO [Metrics] AUC: 0.973587 - logloss: 0.194163
2022-01-20 08:18:45,314 P11986 INFO Save best model: monitor(max): 0.973587
2022-01-20 08:18:45,316 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:45,363 P11986 INFO Train loss: 0.121556
2022-01-20 08:18:45,363 P11986 INFO ************ Epoch=78 end ************
2022-01-20 08:18:47,570 P11986 INFO [Metrics] AUC: 0.973711 - logloss: 0.194293
2022-01-20 08:18:47,571 P11986 INFO Save best model: monitor(max): 0.973711
2022-01-20 08:18:47,573 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:47,620 P11986 INFO Train loss: 0.120197
2022-01-20 08:18:47,620 P11986 INFO ************ Epoch=79 end ************
2022-01-20 08:18:49,805 P11986 INFO [Metrics] AUC: 0.973857 - logloss: 0.193729
2022-01-20 08:18:49,806 P11986 INFO Save best model: monitor(max): 0.973857
2022-01-20 08:18:49,808 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:49,856 P11986 INFO Train loss: 0.118769
2022-01-20 08:18:49,856 P11986 INFO ************ Epoch=80 end ************
2022-01-20 08:18:52,037 P11986 INFO [Metrics] AUC: 0.973930 - logloss: 0.193915
2022-01-20 08:18:52,037 P11986 INFO Save best model: monitor(max): 0.973930
2022-01-20 08:18:52,039 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:52,086 P11986 INFO Train loss: 0.117526
2022-01-20 08:18:52,086 P11986 INFO ************ Epoch=81 end ************
2022-01-20 08:18:54,263 P11986 INFO [Metrics] AUC: 0.974081 - logloss: 0.193791
2022-01-20 08:18:54,263 P11986 INFO Save best model: monitor(max): 0.974081
2022-01-20 08:18:54,265 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:54,312 P11986 INFO Train loss: 0.116253
2022-01-20 08:18:54,313 P11986 INFO ************ Epoch=82 end ************
2022-01-20 08:18:56,496 P11986 INFO [Metrics] AUC: 0.974175 - logloss: 0.194124
2022-01-20 08:18:56,497 P11986 INFO Save best model: monitor(max): 0.974175
2022-01-20 08:18:56,499 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:56,546 P11986 INFO Train loss: 0.115341
2022-01-20 08:18:56,546 P11986 INFO ************ Epoch=83 end ************
2022-01-20 08:18:58,718 P11986 INFO [Metrics] AUC: 0.974322 - logloss: 0.193612
2022-01-20 08:18:58,719 P11986 INFO Save best model: monitor(max): 0.974322
2022-01-20 08:18:58,721 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:18:58,767 P11986 INFO Train loss: 0.113850
2022-01-20 08:18:58,767 P11986 INFO ************ Epoch=84 end ************
2022-01-20 08:19:00,934 P11986 INFO [Metrics] AUC: 0.974412 - logloss: 0.193856
2022-01-20 08:19:00,935 P11986 INFO Save best model: monitor(max): 0.974412
2022-01-20 08:19:00,937 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:00,984 P11986 INFO Train loss: 0.112580
2022-01-20 08:19:00,984 P11986 INFO ************ Epoch=85 end ************
2022-01-20 08:19:03,207 P11986 INFO [Metrics] AUC: 0.974526 - logloss: 0.193735
2022-01-20 08:19:03,207 P11986 INFO Save best model: monitor(max): 0.974526
2022-01-20 08:19:03,209 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:03,256 P11986 INFO Train loss: 0.111552
2022-01-20 08:19:03,256 P11986 INFO ************ Epoch=86 end ************
2022-01-20 08:19:05,421 P11986 INFO [Metrics] AUC: 0.974658 - logloss: 0.193747
2022-01-20 08:19:05,422 P11986 INFO Save best model: monitor(max): 0.974658
2022-01-20 08:19:05,424 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:05,469 P11986 INFO Train loss: 0.110214
2022-01-20 08:19:05,469 P11986 INFO ************ Epoch=87 end ************
2022-01-20 08:19:07,637 P11986 INFO [Metrics] AUC: 0.974748 - logloss: 0.194257
2022-01-20 08:19:07,638 P11986 INFO Save best model: monitor(max): 0.974748
2022-01-20 08:19:07,640 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:07,686 P11986 INFO Train loss: 0.109249
2022-01-20 08:19:07,686 P11986 INFO ************ Epoch=88 end ************
2022-01-20 08:19:09,861 P11986 INFO [Metrics] AUC: 0.974793 - logloss: 0.193930
2022-01-20 08:19:09,862 P11986 INFO Save best model: monitor(max): 0.974793
2022-01-20 08:19:09,864 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:09,911 P11986 INFO Train loss: 0.107748
2022-01-20 08:19:09,911 P11986 INFO ************ Epoch=89 end ************
2022-01-20 08:19:12,100 P11986 INFO [Metrics] AUC: 0.974934 - logloss: 0.194195
2022-01-20 08:19:12,101 P11986 INFO Save best model: monitor(max): 0.974934
2022-01-20 08:19:12,103 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:12,150 P11986 INFO Train loss: 0.106683
2022-01-20 08:19:12,150 P11986 INFO ************ Epoch=90 end ************
2022-01-20 08:19:14,330 P11986 INFO [Metrics] AUC: 0.975027 - logloss: 0.194296
2022-01-20 08:19:14,331 P11986 INFO Save best model: monitor(max): 0.975027
2022-01-20 08:19:14,333 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:14,379 P11986 INFO Train loss: 0.105461
2022-01-20 08:19:14,379 P11986 INFO ************ Epoch=91 end ************
2022-01-20 08:19:16,520 P11986 INFO [Metrics] AUC: 0.975084 - logloss: 0.194254
2022-01-20 08:19:16,520 P11986 INFO Save best model: monitor(max): 0.975084
2022-01-20 08:19:16,522 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:16,568 P11986 INFO Train loss: 0.104305
2022-01-20 08:19:16,568 P11986 INFO ************ Epoch=92 end ************
2022-01-20 08:19:18,702 P11986 INFO [Metrics] AUC: 0.975175 - logloss: 0.194401
2022-01-20 08:19:18,703 P11986 INFO Save best model: monitor(max): 0.975175
2022-01-20 08:19:18,705 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:18,751 P11986 INFO Train loss: 0.103353
2022-01-20 08:19:18,751 P11986 INFO ************ Epoch=93 end ************
2022-01-20 08:19:20,905 P11986 INFO [Metrics] AUC: 0.975238 - logloss: 0.194581
2022-01-20 08:19:20,905 P11986 INFO Save best model: monitor(max): 0.975238
2022-01-20 08:19:20,907 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:20,955 P11986 INFO Train loss: 0.102120
2022-01-20 08:19:20,955 P11986 INFO ************ Epoch=94 end ************
2022-01-20 08:19:23,115 P11986 INFO [Metrics] AUC: 0.975271 - logloss: 0.194724
2022-01-20 08:19:23,116 P11986 INFO Save best model: monitor(max): 0.975271
2022-01-20 08:19:23,118 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:23,163 P11986 INFO Train loss: 0.101229
2022-01-20 08:19:23,164 P11986 INFO ************ Epoch=95 end ************
2022-01-20 08:19:25,312 P11986 INFO [Metrics] AUC: 0.975412 - logloss: 0.195075
2022-01-20 08:19:25,313 P11986 INFO Save best model: monitor(max): 0.975412
2022-01-20 08:19:25,315 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:25,364 P11986 INFO Train loss: 0.099982
2022-01-20 08:19:25,365 P11986 INFO ************ Epoch=96 end ************
2022-01-20 08:19:27,543 P11986 INFO [Metrics] AUC: 0.975486 - logloss: 0.195287
2022-01-20 08:19:27,544 P11986 INFO Save best model: monitor(max): 0.975486
2022-01-20 08:19:27,546 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:27,593 P11986 INFO Train loss: 0.099224
2022-01-20 08:19:27,593 P11986 INFO ************ Epoch=97 end ************
2022-01-20 08:19:29,749 P11986 INFO [Metrics] AUC: 0.975568 - logloss: 0.195726
2022-01-20 08:19:29,749 P11986 INFO Save best model: monitor(max): 0.975568
2022-01-20 08:19:29,751 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:29,798 P11986 INFO Train loss: 0.098004
2022-01-20 08:19:29,798 P11986 INFO ************ Epoch=98 end ************
2022-01-20 08:19:31,964 P11986 INFO [Metrics] AUC: 0.975595 - logloss: 0.195434
2022-01-20 08:19:31,965 P11986 INFO Save best model: monitor(max): 0.975595
2022-01-20 08:19:31,967 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:32,013 P11986 INFO Train loss: 0.096832
2022-01-20 08:19:32,013 P11986 INFO ************ Epoch=99 end ************
2022-01-20 08:19:34,206 P11986 INFO [Metrics] AUC: 0.975673 - logloss: 0.195885
2022-01-20 08:19:34,206 P11986 INFO Save best model: monitor(max): 0.975673
2022-01-20 08:19:34,208 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:34,255 P11986 INFO Train loss: 0.096158
2022-01-20 08:19:34,255 P11986 INFO ************ Epoch=100 end ************
2022-01-20 08:19:36,381 P11986 INFO [Metrics] AUC: 0.975669 - logloss: 0.196345
2022-01-20 08:19:36,382 P11986 INFO Monitor(max) STOP: 0.975669 !
2022-01-20 08:19:36,382 P11986 INFO Reduce learning rate on plateau: 0.000100
2022-01-20 08:19:36,382 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:36,428 P11986 INFO Train loss: 0.095099
2022-01-20 08:19:36,429 P11986 INFO ************ Epoch=101 end ************
2022-01-20 08:19:38,612 P11986 INFO [Metrics] AUC: 0.975687 - logloss: 0.196207
2022-01-20 08:19:38,612 P11986 INFO Save best model: monitor(max): 0.975687
2022-01-20 08:19:38,614 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:38,660 P11986 INFO Train loss: 0.092571
2022-01-20 08:19:38,661 P11986 INFO ************ Epoch=102 end ************
2022-01-20 08:19:40,796 P11986 INFO [Metrics] AUC: 0.975700 - logloss: 0.196202
2022-01-20 08:19:40,797 P11986 INFO Save best model: monitor(max): 0.975700
2022-01-20 08:19:40,799 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:40,864 P11986 INFO Train loss: 0.092329
2022-01-20 08:19:40,864 P11986 INFO ************ Epoch=103 end ************
2022-01-20 08:19:43,035 P11986 INFO [Metrics] AUC: 0.975710 - logloss: 0.196218
2022-01-20 08:19:43,036 P11986 INFO Save best model: monitor(max): 0.975710
2022-01-20 08:19:43,038 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:43,103 P11986 INFO Train loss: 0.092061
2022-01-20 08:19:43,103 P11986 INFO ************ Epoch=104 end ************
2022-01-20 08:19:45,257 P11986 INFO [Metrics] AUC: 0.975723 - logloss: 0.196248
2022-01-20 08:19:45,258 P11986 INFO Save best model: monitor(max): 0.975723
2022-01-20 08:19:45,260 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:45,306 P11986 INFO Train loss: 0.092329
2022-01-20 08:19:45,306 P11986 INFO ************ Epoch=105 end ************
2022-01-20 08:19:47,472 P11986 INFO [Metrics] AUC: 0.975729 - logloss: 0.196278
2022-01-20 08:19:47,472 P11986 INFO Save best model: monitor(max): 0.975729
2022-01-20 08:19:47,474 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:47,522 P11986 INFO Train loss: 0.092083
2022-01-20 08:19:47,522 P11986 INFO ************ Epoch=106 end ************
2022-01-20 08:19:49,670 P11986 INFO [Metrics] AUC: 0.975739 - logloss: 0.196340
2022-01-20 08:19:49,670 P11986 INFO Save best model: monitor(max): 0.975739
2022-01-20 08:19:49,672 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:49,721 P11986 INFO Train loss: 0.091756
2022-01-20 08:19:49,721 P11986 INFO ************ Epoch=107 end ************
2022-01-20 08:19:51,870 P11986 INFO [Metrics] AUC: 0.975747 - logloss: 0.196372
2022-01-20 08:19:51,870 P11986 INFO Save best model: monitor(max): 0.975747
2022-01-20 08:19:51,872 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:51,919 P11986 INFO Train loss: 0.091996
2022-01-20 08:19:51,919 P11986 INFO ************ Epoch=108 end ************
2022-01-20 08:19:54,077 P11986 INFO [Metrics] AUC: 0.975755 - logloss: 0.196369
2022-01-20 08:19:54,077 P11986 INFO Save best model: monitor(max): 0.975755
2022-01-20 08:19:54,079 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:54,125 P11986 INFO Train loss: 0.091599
2022-01-20 08:19:54,125 P11986 INFO ************ Epoch=109 end ************
2022-01-20 08:19:56,332 P11986 INFO [Metrics] AUC: 0.975763 - logloss: 0.196423
2022-01-20 08:19:56,332 P11986 INFO Save best model: monitor(max): 0.975763
2022-01-20 08:19:56,334 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:56,381 P11986 INFO Train loss: 0.091539
2022-01-20 08:19:56,381 P11986 INFO ************ Epoch=110 end ************
2022-01-20 08:19:58,556 P11986 INFO [Metrics] AUC: 0.975768 - logloss: 0.196415
2022-01-20 08:19:58,556 P11986 INFO Save best model: monitor(max): 0.975768
2022-01-20 08:19:58,559 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:19:58,607 P11986 INFO Train loss: 0.091603
2022-01-20 08:19:58,607 P11986 INFO ************ Epoch=111 end ************
2022-01-20 08:20:00,784 P11986 INFO [Metrics] AUC: 0.975774 - logloss: 0.196480
2022-01-20 08:20:00,784 P11986 INFO Save best model: monitor(max): 0.975774
2022-01-20 08:20:00,786 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:20:00,832 P11986 INFO Train loss: 0.091465
2022-01-20 08:20:00,833 P11986 INFO ************ Epoch=112 end ************
2022-01-20 08:20:03,054 P11986 INFO [Metrics] AUC: 0.975780 - logloss: 0.196496
2022-01-20 08:20:03,055 P11986 INFO Save best model: monitor(max): 0.975780
2022-01-20 08:20:03,057 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:20:03,104 P11986 INFO Train loss: 0.091311
2022-01-20 08:20:03,104 P11986 INFO ************ Epoch=113 end ************
2022-01-20 08:20:05,306 P11986 INFO [Metrics] AUC: 0.975792 - logloss: 0.196554
2022-01-20 08:20:05,307 P11986 INFO Save best model: monitor(max): 0.975792
2022-01-20 08:20:05,309 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:20:05,356 P11986 INFO Train loss: 0.091496
2022-01-20 08:20:05,356 P11986 INFO ************ Epoch=114 end ************
2022-01-20 08:20:07,529 P11986 INFO [Metrics] AUC: 0.975789 - logloss: 0.196568
2022-01-20 08:20:07,530 P11986 INFO Monitor(max) STOP: 0.975789 !
2022-01-20 08:20:07,530 P11986 INFO Reduce learning rate on plateau: 0.000010
2022-01-20 08:20:07,530 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:20:07,577 P11986 INFO Train loss: 0.091202
2022-01-20 08:20:07,578 P11986 INFO ************ Epoch=115 end ************
2022-01-20 08:20:09,757 P11986 INFO [Metrics] AUC: 0.975790 - logloss: 0.196578
2022-01-20 08:20:09,758 P11986 INFO Monitor(max) STOP: 0.975790 !
2022-01-20 08:20:09,758 P11986 INFO Reduce learning rate on plateau: 0.000001
2022-01-20 08:20:09,758 P11986 INFO Early stopping at epoch=116
2022-01-20 08:20:09,758 P11986 INFO --- 50/50 batches finished ---
2022-01-20 08:20:09,804 P11986 INFO Train loss: 0.090901
2022-01-20 08:20:09,805 P11986 INFO Training finished.
2022-01-20 08:20:09,805 P11986 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Frappe/FmFM_frappe_x1/frappe_x1_04e961e9/FmFM_frappe_x1_004_2445bd6f.model
2022-01-20 08:20:09,811 P11986 INFO ****** Validation evaluation ******
2022-01-20 08:20:10,213 P11986 INFO [Metrics] AUC: 0.975792 - logloss: 0.196554
2022-01-20 08:20:10,249 P11986 INFO ******** Test evaluation ********
2022-01-20 08:20:10,249 P11986 INFO Loading data...
2022-01-20 08:20:10,249 P11986 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-01-20 08:20:10,252 P11986 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-01-20 08:20:10,252 P11986 INFO Loading test data done.
2022-01-20 08:20:10,505 P11986 INFO [Metrics] AUC: 0.974947 - logloss: 0.200439

```
