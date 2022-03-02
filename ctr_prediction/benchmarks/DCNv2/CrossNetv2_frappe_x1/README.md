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
Dataset ID: [Frappe_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Frappe/README.md#Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [DCNv2](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DCNv2.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CrossNetv2_frappe_x1_tuner_config_01](./CrossNetv2_frappe_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CrossNetv2_frappe_x1
    nohup python run_expid.py --config ./CrossNetv2_frappe_x1_tuner_config_01 --expid DCNv2_frappe_x1_003_77d06b51 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.971605 | 0.232684  |


### Logs
```python
2022-02-04 23:46:47,020 P28284 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "dnn_activations": "relu",
    "embedding_dim": "10",
    "embedding_regularizer": "0.0001",
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
    "model_id": "DCNv2_frappe_x1_003_77d06b51",
    "model_root": "./Frappe/DCNv2_frappe_x1/",
    "model_structure": "crossnet_only",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_cross_layers": "2",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_dnn_hidden_units": "[500, 500, 500]",
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
2022-02-04 23:46:47,021 P28284 INFO Set up feature encoder...
2022-02-04 23:46:47,021 P28284 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-02-04 23:46:47,022 P28284 INFO Loading data...
2022-02-04 23:46:47,034 P28284 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-02-04 23:46:47,093 P28284 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-02-04 23:46:47,117 P28284 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-02-04 23:46:47,117 P28284 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-02-04 23:46:47,117 P28284 INFO Loading train data done.
2022-02-04 23:46:56,560 P28284 INFO Total number of parameters: 74191.
2022-02-04 23:46:56,571 P28284 INFO Start training: 50 batches/epoch
2022-02-04 23:46:56,571 P28284 INFO ************ Epoch=1 start ************
2022-02-04 23:47:01,395 P28284 INFO [Metrics] AUC: 0.907543 - logloss: 0.600979
2022-02-04 23:47:01,396 P28284 INFO Save best model: monitor(max): 0.907543
2022-02-04 23:47:01,399 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:47:01,539 P28284 INFO Train loss: 0.647469
2022-02-04 23:47:01,539 P28284 INFO ************ Epoch=1 end ************
2022-02-04 23:47:06,647 P28284 INFO [Metrics] AUC: 0.930488 - logloss: 0.393063
2022-02-04 23:47:06,647 P28284 INFO Save best model: monitor(max): 0.930488
2022-02-04 23:47:06,650 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:47:06,778 P28284 INFO Train loss: 0.530034
2022-02-04 23:47:06,778 P28284 INFO ************ Epoch=2 end ************
2022-02-04 23:47:11,446 P28284 INFO [Metrics] AUC: 0.937770 - logloss: 0.285439
2022-02-04 23:47:11,447 P28284 INFO Save best model: monitor(max): 0.937770
2022-02-04 23:47:11,465 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:47:11,604 P28284 INFO Train loss: 0.315819
2022-02-04 23:47:11,604 P28284 INFO ************ Epoch=3 end ************
2022-02-04 23:47:15,832 P28284 INFO [Metrics] AUC: 0.940672 - logloss: 0.278652
2022-02-04 23:47:15,833 P28284 INFO Save best model: monitor(max): 0.940672
2022-02-04 23:47:15,843 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:47:15,949 P28284 INFO Train loss: 0.282905
2022-02-04 23:47:15,949 P28284 INFO ************ Epoch=4 end ************
2022-02-04 23:47:21,396 P28284 INFO [Metrics] AUC: 0.941499 - logloss: 0.277041
2022-02-04 23:47:21,397 P28284 INFO Save best model: monitor(max): 0.941499
2022-02-04 23:47:21,400 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:47:21,517 P28284 INFO Train loss: 0.277413
2022-02-04 23:47:21,517 P28284 INFO ************ Epoch=5 end ************
2022-02-04 23:47:26,453 P28284 INFO [Metrics] AUC: 0.941918 - logloss: 0.276165
2022-02-04 23:47:26,455 P28284 INFO Save best model: monitor(max): 0.941918
2022-02-04 23:47:26,458 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:47:26,592 P28284 INFO Train loss: 0.274909
2022-02-04 23:47:26,592 P28284 INFO ************ Epoch=6 end ************
2022-02-04 23:47:30,591 P28284 INFO [Metrics] AUC: 0.942321 - logloss: 0.275611
2022-02-04 23:47:30,592 P28284 INFO Save best model: monitor(max): 0.942321
2022-02-04 23:47:30,594 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:47:30,655 P28284 INFO Train loss: 0.272693
2022-02-04 23:47:30,655 P28284 INFO ************ Epoch=7 end ************
2022-02-04 23:47:34,654 P28284 INFO [Metrics] AUC: 0.942503 - logloss: 0.275432
2022-02-04 23:47:34,655 P28284 INFO Save best model: monitor(max): 0.942503
2022-02-04 23:47:34,671 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:47:34,786 P28284 INFO Train loss: 0.271728
2022-02-04 23:47:34,787 P28284 INFO ************ Epoch=8 end ************
2022-02-04 23:47:39,279 P28284 INFO [Metrics] AUC: 0.942789 - logloss: 0.275001
2022-02-04 23:47:39,279 P28284 INFO Save best model: monitor(max): 0.942789
2022-02-04 23:47:39,290 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:47:39,424 P28284 INFO Train loss: 0.270870
2022-02-04 23:47:39,424 P28284 INFO ************ Epoch=9 end ************
2022-02-04 23:47:43,259 P28284 INFO [Metrics] AUC: 0.942799 - logloss: 0.275080
2022-02-04 23:47:43,268 P28284 INFO Save best model: monitor(max): 0.942799
2022-02-04 23:47:43,271 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:47:43,381 P28284 INFO Train loss: 0.269629
2022-02-04 23:47:43,382 P28284 INFO ************ Epoch=10 end ************
2022-02-04 23:47:47,511 P28284 INFO [Metrics] AUC: 0.942922 - logloss: 0.274900
2022-02-04 23:47:47,511 P28284 INFO Save best model: monitor(max): 0.942922
2022-02-04 23:47:47,514 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:47:47,609 P28284 INFO Train loss: 0.268648
2022-02-04 23:47:47,610 P28284 INFO ************ Epoch=11 end ************
2022-02-04 23:47:50,940 P28284 INFO [Metrics] AUC: 0.943069 - logloss: 0.274932
2022-02-04 23:47:50,941 P28284 INFO Save best model: monitor(max): 0.943069
2022-02-04 23:47:50,943 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:47:51,012 P28284 INFO Train loss: 0.267764
2022-02-04 23:47:51,013 P28284 INFO ************ Epoch=12 end ************
2022-02-04 23:47:54,624 P28284 INFO [Metrics] AUC: 0.943285 - logloss: 0.274457
2022-02-04 23:47:54,624 P28284 INFO Save best model: monitor(max): 0.943285
2022-02-04 23:47:54,636 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:47:54,823 P28284 INFO Train loss: 0.267215
2022-02-04 23:47:54,824 P28284 INFO ************ Epoch=13 end ************
2022-02-04 23:47:58,883 P28284 INFO [Metrics] AUC: 0.943476 - logloss: 0.274136
2022-02-04 23:47:58,883 P28284 INFO Save best model: monitor(max): 0.943476
2022-02-04 23:47:58,886 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:47:59,019 P28284 INFO Train loss: 0.266297
2022-02-04 23:47:59,020 P28284 INFO ************ Epoch=14 end ************
2022-02-04 23:48:02,981 P28284 INFO [Metrics] AUC: 0.943634 - logloss: 0.273949
2022-02-04 23:48:02,981 P28284 INFO Save best model: monitor(max): 0.943634
2022-02-04 23:48:02,984 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:03,094 P28284 INFO Train loss: 0.265109
2022-02-04 23:48:03,094 P28284 INFO ************ Epoch=15 end ************
2022-02-04 23:48:06,556 P28284 INFO [Metrics] AUC: 0.943880 - logloss: 0.273768
2022-02-04 23:48:06,557 P28284 INFO Save best model: monitor(max): 0.943880
2022-02-04 23:48:06,559 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:06,636 P28284 INFO Train loss: 0.264789
2022-02-04 23:48:06,637 P28284 INFO ************ Epoch=16 end ************
2022-02-04 23:48:11,034 P28284 INFO [Metrics] AUC: 0.943973 - logloss: 0.273597
2022-02-04 23:48:11,039 P28284 INFO Save best model: monitor(max): 0.943973
2022-02-04 23:48:11,044 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:11,246 P28284 INFO Train loss: 0.263922
2022-02-04 23:48:11,247 P28284 INFO ************ Epoch=17 end ************
2022-02-04 23:48:15,246 P28284 INFO [Metrics] AUC: 0.944203 - logloss: 0.273385
2022-02-04 23:48:15,246 P28284 INFO Save best model: monitor(max): 0.944203
2022-02-04 23:48:15,257 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:15,355 P28284 INFO Train loss: 0.263175
2022-02-04 23:48:15,355 P28284 INFO ************ Epoch=18 end ************
2022-02-04 23:48:18,965 P28284 INFO [Metrics] AUC: 0.944308 - logloss: 0.273588
2022-02-04 23:48:18,965 P28284 INFO Save best model: monitor(max): 0.944308
2022-02-04 23:48:18,968 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:19,083 P28284 INFO Train loss: 0.262822
2022-02-04 23:48:19,084 P28284 INFO ************ Epoch=19 end ************
2022-02-04 23:48:21,870 P28284 INFO [Metrics] AUC: 0.944473 - logloss: 0.273468
2022-02-04 23:48:21,871 P28284 INFO Save best model: monitor(max): 0.944473
2022-02-04 23:48:21,873 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:22,040 P28284 INFO Train loss: 0.262104
2022-02-04 23:48:22,040 P28284 INFO ************ Epoch=20 end ************
2022-02-04 23:48:25,766 P28284 INFO [Metrics] AUC: 0.944575 - logloss: 0.273448
2022-02-04 23:48:25,767 P28284 INFO Save best model: monitor(max): 0.944575
2022-02-04 23:48:25,771 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:25,876 P28284 INFO Train loss: 0.261420
2022-02-04 23:48:25,876 P28284 INFO ************ Epoch=21 end ************
2022-02-04 23:48:30,041 P28284 INFO [Metrics] AUC: 0.944654 - logloss: 0.273304
2022-02-04 23:48:30,042 P28284 INFO Save best model: monitor(max): 0.944654
2022-02-04 23:48:30,045 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:30,158 P28284 INFO Train loss: 0.260913
2022-02-04 23:48:30,158 P28284 INFO ************ Epoch=22 end ************
2022-02-04 23:48:34,325 P28284 INFO [Metrics] AUC: 0.944861 - logloss: 0.273090
2022-02-04 23:48:34,325 P28284 INFO Save best model: monitor(max): 0.944861
2022-02-04 23:48:34,328 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:34,467 P28284 INFO Train loss: 0.260543
2022-02-04 23:48:34,467 P28284 INFO ************ Epoch=23 end ************
2022-02-04 23:48:36,746 P28284 INFO [Metrics] AUC: 0.944992 - logloss: 0.273044
2022-02-04 23:48:36,746 P28284 INFO Save best model: monitor(max): 0.944992
2022-02-04 23:48:36,749 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:36,852 P28284 INFO Train loss: 0.259428
2022-02-04 23:48:36,852 P28284 INFO ************ Epoch=24 end ************
2022-02-04 23:48:38,948 P28284 INFO [Metrics] AUC: 0.945253 - logloss: 0.272428
2022-02-04 23:48:38,948 P28284 INFO Save best model: monitor(max): 0.945253
2022-02-04 23:48:38,950 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:38,995 P28284 INFO Train loss: 0.258483
2022-02-04 23:48:38,996 P28284 INFO ************ Epoch=25 end ************
2022-02-04 23:48:41,204 P28284 INFO [Metrics] AUC: 0.945690 - logloss: 0.271525
2022-02-04 23:48:41,204 P28284 INFO Save best model: monitor(max): 0.945690
2022-02-04 23:48:41,207 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:41,301 P28284 INFO Train loss: 0.258057
2022-02-04 23:48:41,301 P28284 INFO ************ Epoch=26 end ************
2022-02-04 23:48:43,615 P28284 INFO [Metrics] AUC: 0.946090 - logloss: 0.270601
2022-02-04 23:48:43,616 P28284 INFO Save best model: monitor(max): 0.946090
2022-02-04 23:48:43,618 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:43,681 P28284 INFO Train loss: 0.256639
2022-02-04 23:48:43,681 P28284 INFO ************ Epoch=27 end ************
2022-02-04 23:48:46,037 P28284 INFO [Metrics] AUC: 0.946882 - logloss: 0.268872
2022-02-04 23:48:46,038 P28284 INFO Save best model: monitor(max): 0.946882
2022-02-04 23:48:46,040 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:46,116 P28284 INFO Train loss: 0.254799
2022-02-04 23:48:46,117 P28284 INFO ************ Epoch=28 end ************
2022-02-04 23:48:48,475 P28284 INFO [Metrics] AUC: 0.948049 - logloss: 0.265667
2022-02-04 23:48:48,476 P28284 INFO Save best model: monitor(max): 0.948049
2022-02-04 23:48:48,478 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:48,542 P28284 INFO Train loss: 0.251678
2022-02-04 23:48:48,543 P28284 INFO ************ Epoch=29 end ************
2022-02-04 23:48:51,018 P28284 INFO [Metrics] AUC: 0.950164 - logloss: 0.259972
2022-02-04 23:48:51,018 P28284 INFO Save best model: monitor(max): 0.950164
2022-02-04 23:48:51,021 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:51,103 P28284 INFO Train loss: 0.245346
2022-02-04 23:48:51,103 P28284 INFO ************ Epoch=30 end ************
2022-02-04 23:48:53,534 P28284 INFO [Metrics] AUC: 0.953079 - logloss: 0.252400
2022-02-04 23:48:53,534 P28284 INFO Save best model: monitor(max): 0.953079
2022-02-04 23:48:53,537 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:53,630 P28284 INFO Train loss: 0.235943
2022-02-04 23:48:53,630 P28284 INFO ************ Epoch=31 end ************
2022-02-04 23:48:55,691 P28284 INFO [Metrics] AUC: 0.955367 - logloss: 0.246690
2022-02-04 23:48:55,691 P28284 INFO Save best model: monitor(max): 0.955367
2022-02-04 23:48:55,694 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:55,776 P28284 INFO Train loss: 0.226158
2022-02-04 23:48:55,777 P28284 INFO ************ Epoch=32 end ************
2022-02-04 23:48:58,254 P28284 INFO [Metrics] AUC: 0.956769 - logloss: 0.243664
2022-02-04 23:48:58,255 P28284 INFO Save best model: monitor(max): 0.956769
2022-02-04 23:48:58,257 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:48:58,331 P28284 INFO Train loss: 0.218262
2022-02-04 23:48:58,332 P28284 INFO ************ Epoch=33 end ************
2022-02-04 23:49:00,839 P28284 INFO [Metrics] AUC: 0.957818 - logloss: 0.242087
2022-02-04 23:49:00,839 P28284 INFO Save best model: monitor(max): 0.957818
2022-02-04 23:49:00,842 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:00,924 P28284 INFO Train loss: 0.213144
2022-02-04 23:49:00,924 P28284 INFO ************ Epoch=34 end ************
2022-02-04 23:49:03,433 P28284 INFO [Metrics] AUC: 0.958590 - logloss: 0.240676
2022-02-04 23:49:03,433 P28284 INFO Save best model: monitor(max): 0.958590
2022-02-04 23:49:03,436 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:03,525 P28284 INFO Train loss: 0.208867
2022-02-04 23:49:03,525 P28284 INFO ************ Epoch=35 end ************
2022-02-04 23:49:05,713 P28284 INFO [Metrics] AUC: 0.959495 - logloss: 0.238756
2022-02-04 23:49:05,713 P28284 INFO Save best model: monitor(max): 0.959495
2022-02-04 23:49:05,716 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:05,777 P28284 INFO Train loss: 0.204636
2022-02-04 23:49:05,778 P28284 INFO ************ Epoch=36 end ************
2022-02-04 23:49:08,216 P28284 INFO [Metrics] AUC: 0.960218 - logloss: 0.238054
2022-02-04 23:49:08,217 P28284 INFO Save best model: monitor(max): 0.960218
2022-02-04 23:49:08,219 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:08,286 P28284 INFO Train loss: 0.200772
2022-02-04 23:49:08,286 P28284 INFO ************ Epoch=37 end ************
2022-02-04 23:49:10,561 P28284 INFO [Metrics] AUC: 0.961127 - logloss: 0.236420
2022-02-04 23:49:10,561 P28284 INFO Save best model: monitor(max): 0.961127
2022-02-04 23:49:10,563 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:10,613 P28284 INFO Train loss: 0.196429
2022-02-04 23:49:10,613 P28284 INFO ************ Epoch=38 end ************
2022-02-04 23:49:12,720 P28284 INFO [Metrics] AUC: 0.962437 - logloss: 0.233958
2022-02-04 23:49:12,721 P28284 INFO Save best model: monitor(max): 0.962437
2022-02-04 23:49:12,723 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:12,800 P28284 INFO Train loss: 0.190837
2022-02-04 23:49:12,800 P28284 INFO ************ Epoch=39 end ************
2022-02-04 23:49:15,224 P28284 INFO [Metrics] AUC: 0.963758 - logloss: 0.230705
2022-02-04 23:49:15,224 P28284 INFO Save best model: monitor(max): 0.963758
2022-02-04 23:49:15,227 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:15,298 P28284 INFO Train loss: 0.182751
2022-02-04 23:49:15,298 P28284 INFO ************ Epoch=40 end ************
2022-02-04 23:49:17,704 P28284 INFO [Metrics] AUC: 0.964643 - logloss: 0.230390
2022-02-04 23:49:17,705 P28284 INFO Save best model: monitor(max): 0.964643
2022-02-04 23:49:17,707 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:17,791 P28284 INFO Train loss: 0.176210
2022-02-04 23:49:17,791 P28284 INFO ************ Epoch=41 end ************
2022-02-04 23:49:20,271 P28284 INFO [Metrics] AUC: 0.965164 - logloss: 0.230189
2022-02-04 23:49:20,272 P28284 INFO Save best model: monitor(max): 0.965164
2022-02-04 23:49:20,274 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:20,356 P28284 INFO Train loss: 0.171436
2022-02-04 23:49:20,356 P28284 INFO ************ Epoch=42 end ************
2022-02-04 23:49:22,611 P28284 INFO [Metrics] AUC: 0.965803 - logloss: 0.229418
2022-02-04 23:49:22,611 P28284 INFO Save best model: monitor(max): 0.965803
2022-02-04 23:49:22,614 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:22,694 P28284 INFO Train loss: 0.167906
2022-02-04 23:49:22,694 P28284 INFO ************ Epoch=43 end ************
2022-02-04 23:49:25,132 P28284 INFO [Metrics] AUC: 0.966349 - logloss: 0.229632
2022-02-04 23:49:25,133 P28284 INFO Save best model: monitor(max): 0.966349
2022-02-04 23:49:25,136 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:25,212 P28284 INFO Train loss: 0.163599
2022-02-04 23:49:25,212 P28284 INFO ************ Epoch=44 end ************
2022-02-04 23:49:27,579 P28284 INFO [Metrics] AUC: 0.967198 - logloss: 0.227957
2022-02-04 23:49:27,579 P28284 INFO Save best model: monitor(max): 0.967198
2022-02-04 23:49:27,581 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:27,624 P28284 INFO Train loss: 0.159471
2022-02-04 23:49:27,624 P28284 INFO ************ Epoch=45 end ************
2022-02-04 23:49:29,569 P28284 INFO [Metrics] AUC: 0.967597 - logloss: 0.228168
2022-02-04 23:49:29,570 P28284 INFO Save best model: monitor(max): 0.967597
2022-02-04 23:49:29,572 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:29,644 P28284 INFO Train loss: 0.155321
2022-02-04 23:49:29,644 P28284 INFO ************ Epoch=46 end ************
2022-02-04 23:49:31,992 P28284 INFO [Metrics] AUC: 0.968151 - logloss: 0.227788
2022-02-04 23:49:31,992 P28284 INFO Save best model: monitor(max): 0.968151
2022-02-04 23:49:31,995 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:32,103 P28284 INFO Train loss: 0.151960
2022-02-04 23:49:32,103 P28284 INFO ************ Epoch=47 end ************
2022-02-04 23:49:34,540 P28284 INFO [Metrics] AUC: 0.968426 - logloss: 0.230473
2022-02-04 23:49:34,541 P28284 INFO Save best model: monitor(max): 0.968426
2022-02-04 23:49:34,543 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:34,604 P28284 INFO Train loss: 0.148741
2022-02-04 23:49:34,605 P28284 INFO ************ Epoch=48 end ************
2022-02-04 23:49:36,938 P28284 INFO [Metrics] AUC: 0.968567 - logloss: 0.229408
2022-02-04 23:49:36,939 P28284 INFO Save best model: monitor(max): 0.968567
2022-02-04 23:49:36,942 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:37,007 P28284 INFO Train loss: 0.145942
2022-02-04 23:49:37,008 P28284 INFO ************ Epoch=49 end ************
2022-02-04 23:49:39,528 P28284 INFO [Metrics] AUC: 0.968909 - logloss: 0.229863
2022-02-04 23:49:39,528 P28284 INFO Save best model: monitor(max): 0.968909
2022-02-04 23:49:39,531 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:39,620 P28284 INFO Train loss: 0.143086
2022-02-04 23:49:39,621 P28284 INFO ************ Epoch=50 end ************
2022-02-04 23:49:41,966 P28284 INFO [Metrics] AUC: 0.969117 - logloss: 0.230757
2022-02-04 23:49:41,966 P28284 INFO Save best model: monitor(max): 0.969117
2022-02-04 23:49:41,968 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:42,041 P28284 INFO Train loss: 0.140215
2022-02-04 23:49:42,041 P28284 INFO ************ Epoch=51 end ************
2022-02-04 23:49:44,474 P28284 INFO [Metrics] AUC: 0.969506 - logloss: 0.231256
2022-02-04 23:49:44,475 P28284 INFO Save best model: monitor(max): 0.969506
2022-02-04 23:49:44,477 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:44,564 P28284 INFO Train loss: 0.137413
2022-02-04 23:49:44,565 P28284 INFO ************ Epoch=52 end ************
2022-02-04 23:49:46,776 P28284 INFO [Metrics] AUC: 0.969639 - logloss: 0.232535
2022-02-04 23:49:46,776 P28284 INFO Save best model: monitor(max): 0.969639
2022-02-04 23:49:46,778 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:46,823 P28284 INFO Train loss: 0.134328
2022-02-04 23:49:46,823 P28284 INFO ************ Epoch=53 end ************
2022-02-04 23:49:48,941 P28284 INFO [Metrics] AUC: 0.970022 - logloss: 0.232901
2022-02-04 23:49:48,941 P28284 INFO Save best model: monitor(max): 0.970022
2022-02-04 23:49:48,944 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:49,013 P28284 INFO Train loss: 0.131024
2022-02-04 23:49:49,013 P28284 INFO ************ Epoch=54 end ************
2022-02-04 23:49:51,444 P28284 INFO [Metrics] AUC: 0.970539 - logloss: 0.232071
2022-02-04 23:49:51,447 P28284 INFO Save best model: monitor(max): 0.970539
2022-02-04 23:49:51,450 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:51,542 P28284 INFO Train loss: 0.127877
2022-02-04 23:49:51,543 P28284 INFO ************ Epoch=55 end ************
2022-02-04 23:49:53,985 P28284 INFO [Metrics] AUC: 0.970461 - logloss: 0.233581
2022-02-04 23:49:53,985 P28284 INFO Monitor(max) STOP: 0.970461 !
2022-02-04 23:49:53,986 P28284 INFO Reduce learning rate on plateau: 0.000100
2022-02-04 23:49:53,986 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:54,078 P28284 INFO Train loss: 0.124901
2022-02-04 23:49:54,079 P28284 INFO ************ Epoch=56 end ************
2022-02-04 23:49:56,612 P28284 INFO [Metrics] AUC: 0.970740 - logloss: 0.234046
2022-02-04 23:49:56,612 P28284 INFO Save best model: monitor(max): 0.970740
2022-02-04 23:49:56,615 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:56,697 P28284 INFO Train loss: 0.113700
2022-02-04 23:49:56,697 P28284 INFO ************ Epoch=57 end ************
2022-02-04 23:49:59,212 P28284 INFO [Metrics] AUC: 0.970834 - logloss: 0.234654
2022-02-04 23:49:59,212 P28284 INFO Save best model: monitor(max): 0.970834
2022-02-04 23:49:59,215 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:49:59,320 P28284 INFO Train loss: 0.112703
2022-02-04 23:49:59,321 P28284 INFO ************ Epoch=58 end ************
2022-02-04 23:50:01,724 P28284 INFO [Metrics] AUC: 0.970850 - logloss: 0.235091
2022-02-04 23:50:01,724 P28284 INFO Save best model: monitor(max): 0.970850
2022-02-04 23:50:01,727 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:50:01,805 P28284 INFO Train loss: 0.112190
2022-02-04 23:50:01,805 P28284 INFO ************ Epoch=59 end ************
2022-02-04 23:50:03,833 P28284 INFO [Metrics] AUC: 0.970855 - logloss: 0.235971
2022-02-04 23:50:03,833 P28284 INFO Save best model: monitor(max): 0.970855
2022-02-04 23:50:03,835 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:50:03,877 P28284 INFO Train loss: 0.111519
2022-02-04 23:50:03,877 P28284 INFO ************ Epoch=60 end ************
2022-02-04 23:50:06,012 P28284 INFO [Metrics] AUC: 0.970871 - logloss: 0.237107
2022-02-04 23:50:06,012 P28284 INFO Save best model: monitor(max): 0.970871
2022-02-04 23:50:06,015 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:50:06,100 P28284 INFO Train loss: 0.110983
2022-02-04 23:50:06,100 P28284 INFO ************ Epoch=61 end ************
2022-02-04 23:50:08,305 P28284 INFO [Metrics] AUC: 0.970849 - logloss: 0.237239
2022-02-04 23:50:08,306 P28284 INFO Monitor(max) STOP: 0.970849 !
2022-02-04 23:50:08,306 P28284 INFO Reduce learning rate on plateau: 0.000010
2022-02-04 23:50:08,306 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:50:08,393 P28284 INFO Train loss: 0.110772
2022-02-04 23:50:08,393 P28284 INFO ************ Epoch=62 end ************
2022-02-04 23:50:10,760 P28284 INFO [Metrics] AUC: 0.970865 - logloss: 0.237496
2022-02-04 23:50:10,761 P28284 INFO Monitor(max) STOP: 0.970865 !
2022-02-04 23:50:10,761 P28284 INFO Reduce learning rate on plateau: 0.000001
2022-02-04 23:50:10,761 P28284 INFO Early stopping at epoch=63
2022-02-04 23:50:10,761 P28284 INFO --- 50/50 batches finished ---
2022-02-04 23:50:10,829 P28284 INFO Train loss: 0.109603
2022-02-04 23:50:10,829 P28284 INFO Training finished.
2022-02-04 23:50:10,829 P28284 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Frappe/DCNv2_frappe_x1/frappe_x1_04e961e9/DCNv2_frappe_x1_003_77d06b51.model
2022-02-04 23:50:10,837 P28284 INFO ****** Validation evaluation ******
2022-02-04 23:50:11,570 P28284 INFO [Metrics] AUC: 0.970871 - logloss: 0.237107
2022-02-04 23:50:11,621 P28284 INFO ******** Test evaluation ********
2022-02-04 23:50:11,622 P28284 INFO Loading data...
2022-02-04 23:50:11,622 P28284 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-02-04 23:50:11,626 P28284 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-02-04 23:50:11,626 P28284 INFO Loading test data done.
2022-02-04 23:50:12,098 P28284 INFO [Metrics] AUC: 0.971605 - logloss: 0.232684

```
