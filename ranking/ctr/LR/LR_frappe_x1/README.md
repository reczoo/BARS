## LR_frappe_x1

A hands-on guide to run the LR model on the Frappe_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [LR](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/LR.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [LR_frappe_x1_tuner_config_01](./LR_frappe_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd LR_frappe_x1
    nohup python run_expid.py --config ./LR_frappe_x1_tuner_config_01 --expid LR_frappe_x1_001_ff730f43 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.935584 | 0.307643  |


### Logs
```python
2022-01-25 14:34:42,266 P47737 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "LR",
    "model_id": "LR_frappe_x1_001_ff730f43",
    "model_root": "./Frappe/LR_frappe_x1/",
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
2022-01-25 14:34:42,267 P47737 INFO Set up feature encoder...
2022-01-25 14:34:42,267 P47737 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-01-25 14:34:42,267 P47737 INFO Loading data...
2022-01-25 14:34:42,269 P47737 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-01-25 14:34:42,284 P47737 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-01-25 14:34:42,289 P47737 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-01-25 14:34:42,289 P47737 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-01-25 14:34:42,289 P47737 INFO Loading train data done.
2022-01-25 14:34:49,830 P47737 INFO Total number of parameters: 5390.
2022-01-25 14:34:49,832 P47737 INFO Start training: 50 batches/epoch
2022-01-25 14:34:49,833 P47737 INFO ************ Epoch=1 start ************
2022-01-25 14:35:10,365 P47737 INFO [Metrics] AUC: 0.902462 - logloss: 0.632142
2022-01-25 14:35:10,366 P47737 INFO Save best model: monitor(max): 0.902462
2022-01-25 14:35:10,367 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:35:10,424 P47737 INFO Train loss: 0.659054
2022-01-25 14:35:10,424 P47737 INFO ************ Epoch=1 end ************
2022-01-25 14:35:30,821 P47737 INFO [Metrics] AUC: 0.910819 - logloss: 0.612575
2022-01-25 14:35:30,822 P47737 INFO Save best model: monitor(max): 0.910819
2022-01-25 14:35:30,824 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:35:30,876 P47737 INFO Train loss: 0.623044
2022-01-25 14:35:30,876 P47737 INFO ************ Epoch=2 end ************
2022-01-25 14:35:50,800 P47737 INFO [Metrics] AUC: 0.916198 - logloss: 0.601349
2022-01-25 14:35:50,801 P47737 INFO Save best model: monitor(max): 0.916198
2022-01-25 14:35:50,802 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:35:50,855 P47737 INFO Train loss: 0.609086
2022-01-25 14:35:50,855 P47737 INFO ************ Epoch=3 end ************
2022-01-25 14:36:11,108 P47737 INFO [Metrics] AUC: 0.918841 - logloss: 0.591127
2022-01-25 14:36:11,109 P47737 INFO Save best model: monitor(max): 0.918841
2022-01-25 14:36:11,110 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:36:11,143 P47737 INFO Train loss: 0.598188
2022-01-25 14:36:11,143 P47737 INFO ************ Epoch=4 end ************
2022-01-25 14:36:31,635 P47737 INFO [Metrics] AUC: 0.920958 - logloss: 0.581277
2022-01-25 14:36:31,636 P47737 INFO Save best model: monitor(max): 0.920958
2022-01-25 14:36:31,638 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:36:31,680 P47737 INFO Train loss: 0.587877
2022-01-25 14:36:31,680 P47737 INFO ************ Epoch=5 end ************
2022-01-25 14:36:52,011 P47737 INFO [Metrics] AUC: 0.921779 - logloss: 0.571779
2022-01-25 14:36:52,011 P47737 INFO Save best model: monitor(max): 0.921779
2022-01-25 14:36:52,013 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:36:52,047 P47737 INFO Train loss: 0.578117
2022-01-25 14:36:52,047 P47737 INFO ************ Epoch=6 end ************
2022-01-25 14:37:12,465 P47737 INFO [Metrics] AUC: 0.923017 - logloss: 0.562560
2022-01-25 14:37:12,465 P47737 INFO Save best model: monitor(max): 0.923017
2022-01-25 14:37:12,467 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:37:12,501 P47737 INFO Train loss: 0.568378
2022-01-25 14:37:12,502 P47737 INFO ************ Epoch=7 end ************
2022-01-25 14:37:32,706 P47737 INFO [Metrics] AUC: 0.923809 - logloss: 0.553691
2022-01-25 14:37:32,707 P47737 INFO Save best model: monitor(max): 0.923809
2022-01-25 14:37:32,708 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:37:32,743 P47737 INFO Train loss: 0.559020
2022-01-25 14:37:32,744 P47737 INFO ************ Epoch=8 end ************
2022-01-25 14:37:53,050 P47737 INFO [Metrics] AUC: 0.924597 - logloss: 0.545076
2022-01-25 14:37:53,050 P47737 INFO Save best model: monitor(max): 0.924597
2022-01-25 14:37:53,052 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:37:53,089 P47737 INFO Train loss: 0.550260
2022-01-25 14:37:53,090 P47737 INFO ************ Epoch=9 end ************
2022-01-25 14:38:13,582 P47737 INFO [Metrics] AUC: 0.925293 - logloss: 0.536755
2022-01-25 14:38:13,582 P47737 INFO Save best model: monitor(max): 0.925293
2022-01-25 14:38:13,584 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:38:13,637 P47737 INFO Train loss: 0.541906
2022-01-25 14:38:13,637 P47737 INFO ************ Epoch=10 end ************
2022-01-25 14:38:34,066 P47737 INFO [Metrics] AUC: 0.926134 - logloss: 0.528691
2022-01-25 14:38:34,066 P47737 INFO Save best model: monitor(max): 0.926134
2022-01-25 14:38:34,067 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:38:34,103 P47737 INFO Train loss: 0.533360
2022-01-25 14:38:34,103 P47737 INFO ************ Epoch=11 end ************
2022-01-25 14:38:54,470 P47737 INFO [Metrics] AUC: 0.926647 - logloss: 0.520888
2022-01-25 14:38:54,471 P47737 INFO Save best model: monitor(max): 0.926647
2022-01-25 14:38:54,472 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:38:54,510 P47737 INFO Train loss: 0.525406
2022-01-25 14:38:54,510 P47737 INFO ************ Epoch=12 end ************
2022-01-25 14:39:14,738 P47737 INFO [Metrics] AUC: 0.927147 - logloss: 0.513340
2022-01-25 14:39:14,738 P47737 INFO Save best model: monitor(max): 0.927147
2022-01-25 14:39:14,740 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:39:14,774 P47737 INFO Train loss: 0.517789
2022-01-25 14:39:14,774 P47737 INFO ************ Epoch=13 end ************
2022-01-25 14:39:34,417 P47737 INFO [Metrics] AUC: 0.927442 - logloss: 0.506105
2022-01-25 14:39:34,417 P47737 INFO Save best model: monitor(max): 0.927442
2022-01-25 14:39:34,419 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:39:34,461 P47737 INFO Train loss: 0.510610
2022-01-25 14:39:34,461 P47737 INFO ************ Epoch=14 end ************
2022-01-25 14:39:51,233 P47737 INFO [Metrics] AUC: 0.928006 - logloss: 0.499062
2022-01-25 14:39:51,234 P47737 INFO Save best model: monitor(max): 0.928006
2022-01-25 14:39:51,235 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:39:51,269 P47737 INFO Train loss: 0.503493
2022-01-25 14:39:51,269 P47737 INFO ************ Epoch=15 end ************
2022-01-25 14:40:07,167 P47737 INFO [Metrics] AUC: 0.928362 - logloss: 0.492296
2022-01-25 14:40:07,167 P47737 INFO Save best model: monitor(max): 0.928362
2022-01-25 14:40:07,169 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:40:07,204 P47737 INFO Train loss: 0.496555
2022-01-25 14:40:07,204 P47737 INFO ************ Epoch=16 end ************
2022-01-25 14:40:21,253 P47737 INFO [Metrics] AUC: 0.928848 - logloss: 0.485701
2022-01-25 14:40:21,254 P47737 INFO Save best model: monitor(max): 0.928848
2022-01-25 14:40:21,255 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:40:21,296 P47737 INFO Train loss: 0.489877
2022-01-25 14:40:21,297 P47737 INFO ************ Epoch=17 end ************
2022-01-25 14:40:34,138 P47737 INFO [Metrics] AUC: 0.929298 - logloss: 0.479334
2022-01-25 14:40:34,138 P47737 INFO Save best model: monitor(max): 0.929298
2022-01-25 14:40:34,140 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:40:34,175 P47737 INFO Train loss: 0.483393
2022-01-25 14:40:34,175 P47737 INFO ************ Epoch=18 end ************
2022-01-25 14:40:46,504 P47737 INFO [Metrics] AUC: 0.929631 - logloss: 0.473268
2022-01-25 14:40:46,505 P47737 INFO Save best model: monitor(max): 0.929631
2022-01-25 14:40:46,506 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:40:46,561 P47737 INFO Train loss: 0.477393
2022-01-25 14:40:46,561 P47737 INFO ************ Epoch=19 end ************
2022-01-25 14:40:58,916 P47737 INFO [Metrics] AUC: 0.929917 - logloss: 0.467339
2022-01-25 14:40:58,917 P47737 INFO Save best model: monitor(max): 0.929917
2022-01-25 14:40:58,918 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:40:58,952 P47737 INFO Train loss: 0.471632
2022-01-25 14:40:58,952 P47737 INFO ************ Epoch=20 end ************
2022-01-25 14:41:11,292 P47737 INFO [Metrics] AUC: 0.930262 - logloss: 0.461586
2022-01-25 14:41:11,292 P47737 INFO Save best model: monitor(max): 0.930262
2022-01-25 14:41:11,294 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:41:11,327 P47737 INFO Train loss: 0.465705
2022-01-25 14:41:11,327 P47737 INFO ************ Epoch=21 end ************
2022-01-25 14:41:23,686 P47737 INFO [Metrics] AUC: 0.930454 - logloss: 0.456085
2022-01-25 14:41:23,686 P47737 INFO Save best model: monitor(max): 0.930454
2022-01-25 14:41:23,688 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:41:23,720 P47737 INFO Train loss: 0.460451
2022-01-25 14:41:23,720 P47737 INFO ************ Epoch=22 end ************
2022-01-25 14:41:36,037 P47737 INFO [Metrics] AUC: 0.930712 - logloss: 0.450763
2022-01-25 14:41:36,037 P47737 INFO Save best model: monitor(max): 0.930712
2022-01-25 14:41:36,038 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:41:36,072 P47737 INFO Train loss: 0.455047
2022-01-25 14:41:36,072 P47737 INFO ************ Epoch=23 end ************
2022-01-25 14:41:47,723 P47737 INFO [Metrics] AUC: 0.930953 - logloss: 0.445633
2022-01-25 14:41:47,724 P47737 INFO Save best model: monitor(max): 0.930953
2022-01-25 14:41:47,725 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:41:47,759 P47737 INFO Train loss: 0.450150
2022-01-25 14:41:47,759 P47737 INFO ************ Epoch=24 end ************
2022-01-25 14:41:59,383 P47737 INFO [Metrics] AUC: 0.931217 - logloss: 0.440665
2022-01-25 14:41:59,384 P47737 INFO Save best model: monitor(max): 0.931217
2022-01-25 14:41:59,385 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:41:59,418 P47737 INFO Train loss: 0.445125
2022-01-25 14:41:59,418 P47737 INFO ************ Epoch=25 end ************
2022-01-25 14:42:11,084 P47737 INFO [Metrics] AUC: 0.931426 - logloss: 0.435850
2022-01-25 14:42:11,084 P47737 INFO Save best model: monitor(max): 0.931426
2022-01-25 14:42:11,086 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:42:11,122 P47737 INFO Train loss: 0.440393
2022-01-25 14:42:11,122 P47737 INFO ************ Epoch=26 end ************
2022-01-25 14:42:22,686 P47737 INFO [Metrics] AUC: 0.931650 - logloss: 0.431273
2022-01-25 14:42:22,687 P47737 INFO Save best model: monitor(max): 0.931650
2022-01-25 14:42:22,688 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:42:22,721 P47737 INFO Train loss: 0.436019
2022-01-25 14:42:22,721 P47737 INFO ************ Epoch=27 end ************
2022-01-25 14:42:34,366 P47737 INFO [Metrics] AUC: 0.931892 - logloss: 0.426744
2022-01-25 14:42:34,367 P47737 INFO Save best model: monitor(max): 0.931892
2022-01-25 14:42:34,368 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:42:34,399 P47737 INFO Train loss: 0.431592
2022-01-25 14:42:34,399 P47737 INFO ************ Epoch=28 end ************
2022-01-25 14:42:46,038 P47737 INFO [Metrics] AUC: 0.932037 - logloss: 0.422431
2022-01-25 14:42:46,039 P47737 INFO Save best model: monitor(max): 0.932037
2022-01-25 14:42:46,040 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:42:46,071 P47737 INFO Train loss: 0.427560
2022-01-25 14:42:46,072 P47737 INFO ************ Epoch=29 end ************
2022-01-25 14:42:57,690 P47737 INFO [Metrics] AUC: 0.932144 - logloss: 0.418251
2022-01-25 14:42:57,691 P47737 INFO Save best model: monitor(max): 0.932144
2022-01-25 14:42:57,692 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:42:57,722 P47737 INFO Train loss: 0.423521
2022-01-25 14:42:57,723 P47737 INFO ************ Epoch=30 end ************
2022-01-25 14:43:09,382 P47737 INFO [Metrics] AUC: 0.932364 - logloss: 0.414255
2022-01-25 14:43:09,383 P47737 INFO Save best model: monitor(max): 0.932364
2022-01-25 14:43:09,385 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:43:09,418 P47737 INFO Train loss: 0.419604
2022-01-25 14:43:09,418 P47737 INFO ************ Epoch=31 end ************
2022-01-25 14:43:21,027 P47737 INFO [Metrics] AUC: 0.932498 - logloss: 0.410374
2022-01-25 14:43:21,027 P47737 INFO Save best model: monitor(max): 0.932498
2022-01-25 14:43:21,029 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:43:21,064 P47737 INFO Train loss: 0.415852
2022-01-25 14:43:21,065 P47737 INFO ************ Epoch=32 end ************
2022-01-25 14:43:32,709 P47737 INFO [Metrics] AUC: 0.932719 - logloss: 0.406583
2022-01-25 14:43:32,710 P47737 INFO Save best model: monitor(max): 0.932719
2022-01-25 14:43:32,711 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:43:32,744 P47737 INFO Train loss: 0.412301
2022-01-25 14:43:32,744 P47737 INFO ************ Epoch=33 end ************
2022-01-25 14:43:43,602 P47737 INFO [Metrics] AUC: 0.932868 - logloss: 0.402970
2022-01-25 14:43:43,602 P47737 INFO Save best model: monitor(max): 0.932868
2022-01-25 14:43:43,604 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:43:43,636 P47737 INFO Train loss: 0.408826
2022-01-25 14:43:43,636 P47737 INFO ************ Epoch=34 end ************
2022-01-25 14:43:55,707 P47737 INFO [Metrics] AUC: 0.933018 - logloss: 0.399443
2022-01-25 14:43:55,707 P47737 INFO Save best model: monitor(max): 0.933018
2022-01-25 14:43:55,709 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:43:55,769 P47737 INFO Train loss: 0.405334
2022-01-25 14:43:55,769 P47737 INFO ************ Epoch=35 end ************
2022-01-25 14:44:07,802 P47737 INFO [Metrics] AUC: 0.933179 - logloss: 0.396050
2022-01-25 14:44:07,803 P47737 INFO Save best model: monitor(max): 0.933179
2022-01-25 14:44:07,804 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:44:07,839 P47737 INFO Train loss: 0.402375
2022-01-25 14:44:07,840 P47737 INFO ************ Epoch=36 end ************
2022-01-25 14:44:19,095 P47737 INFO [Metrics] AUC: 0.933295 - logloss: 0.392788
2022-01-25 14:44:19,096 P47737 INFO Save best model: monitor(max): 0.933295
2022-01-25 14:44:19,097 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:44:19,133 P47737 INFO Train loss: 0.399182
2022-01-25 14:44:19,133 P47737 INFO ************ Epoch=37 end ************
2022-01-25 14:44:30,535 P47737 INFO [Metrics] AUC: 0.933407 - logloss: 0.389614
2022-01-25 14:44:30,535 P47737 INFO Save best model: monitor(max): 0.933407
2022-01-25 14:44:30,537 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:44:30,570 P47737 INFO Train loss: 0.396329
2022-01-25 14:44:30,571 P47737 INFO ************ Epoch=38 end ************
2022-01-25 14:44:38,728 P47737 INFO [Metrics] AUC: 0.933566 - logloss: 0.386528
2022-01-25 14:44:38,729 P47737 INFO Save best model: monitor(max): 0.933566
2022-01-25 14:44:38,730 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:44:38,763 P47737 INFO Train loss: 0.393185
2022-01-25 14:44:38,763 P47737 INFO ************ Epoch=39 end ************
2022-01-25 14:44:43,024 P47737 INFO [Metrics] AUC: 0.933656 - logloss: 0.383579
2022-01-25 14:44:43,025 P47737 INFO Save best model: monitor(max): 0.933656
2022-01-25 14:44:43,027 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:44:43,076 P47737 INFO Train loss: 0.390631
2022-01-25 14:44:43,076 P47737 INFO ************ Epoch=40 end ************
2022-01-25 14:44:47,912 P47737 INFO [Metrics] AUC: 0.933733 - logloss: 0.380746
2022-01-25 14:44:47,912 P47737 INFO Save best model: monitor(max): 0.933733
2022-01-25 14:44:47,914 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:44:47,952 P47737 INFO Train loss: 0.388081
2022-01-25 14:44:47,953 P47737 INFO ************ Epoch=41 end ************
2022-01-25 14:44:53,826 P47737 INFO [Metrics] AUC: 0.933868 - logloss: 0.377965
2022-01-25 14:44:53,827 P47737 INFO Save best model: monitor(max): 0.933868
2022-01-25 14:44:53,829 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:44:53,883 P47737 INFO Train loss: 0.385459
2022-01-25 14:44:53,884 P47737 INFO ************ Epoch=42 end ************
2022-01-25 14:44:57,842 P47737 INFO [Metrics] AUC: 0.933946 - logloss: 0.375288
2022-01-25 14:44:57,842 P47737 INFO Save best model: monitor(max): 0.933946
2022-01-25 14:44:57,844 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:44:57,882 P47737 INFO Train loss: 0.382840
2022-01-25 14:44:57,882 P47737 INFO ************ Epoch=43 end ************
2022-01-25 14:45:01,285 P47737 INFO [Metrics] AUC: 0.934061 - logloss: 0.372734
2022-01-25 14:45:01,285 P47737 INFO Save best model: monitor(max): 0.934061
2022-01-25 14:45:01,288 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:01,330 P47737 INFO Train loss: 0.380466
2022-01-25 14:45:01,330 P47737 INFO ************ Epoch=44 end ************
2022-01-25 14:45:03,993 P47737 INFO [Metrics] AUC: 0.934170 - logloss: 0.370213
2022-01-25 14:45:03,993 P47737 INFO Save best model: monitor(max): 0.934170
2022-01-25 14:45:03,995 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:04,031 P47737 INFO Train loss: 0.378134
2022-01-25 14:45:04,031 P47737 INFO ************ Epoch=45 end ************
2022-01-25 14:45:06,453 P47737 INFO [Metrics] AUC: 0.934260 - logloss: 0.367802
2022-01-25 14:45:06,454 P47737 INFO Save best model: monitor(max): 0.934260
2022-01-25 14:45:06,455 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:06,494 P47737 INFO Train loss: 0.375969
2022-01-25 14:45:06,494 P47737 INFO ************ Epoch=46 end ************
2022-01-25 14:45:08,890 P47737 INFO [Metrics] AUC: 0.934375 - logloss: 0.365456
2022-01-25 14:45:08,891 P47737 INFO Save best model: monitor(max): 0.934375
2022-01-25 14:45:08,892 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:08,927 P47737 INFO Train loss: 0.374035
2022-01-25 14:45:08,927 P47737 INFO ************ Epoch=47 end ************
2022-01-25 14:45:11,007 P47737 INFO [Metrics] AUC: 0.934477 - logloss: 0.363184
2022-01-25 14:45:11,008 P47737 INFO Save best model: monitor(max): 0.934477
2022-01-25 14:45:11,009 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:11,045 P47737 INFO Train loss: 0.371853
2022-01-25 14:45:11,045 P47737 INFO ************ Epoch=48 end ************
2022-01-25 14:45:13,675 P47737 INFO [Metrics] AUC: 0.934550 - logloss: 0.360993
2022-01-25 14:45:13,676 P47737 INFO Save best model: monitor(max): 0.934550
2022-01-25 14:45:13,678 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:13,718 P47737 INFO Train loss: 0.370130
2022-01-25 14:45:13,718 P47737 INFO ************ Epoch=49 end ************
2022-01-25 14:45:16,799 P47737 INFO [Metrics] AUC: 0.934699 - logloss: 0.358987
2022-01-25 14:45:16,799 P47737 INFO Save best model: monitor(max): 0.934699
2022-01-25 14:45:16,801 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:16,843 P47737 INFO Train loss: 0.367785
2022-01-25 14:45:16,843 P47737 INFO ************ Epoch=50 end ************
2022-01-25 14:45:18,958 P47737 INFO [Metrics] AUC: 0.934754 - logloss: 0.356840
2022-01-25 14:45:18,959 P47737 INFO Save best model: monitor(max): 0.934754
2022-01-25 14:45:18,960 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:18,998 P47737 INFO Train loss: 0.366240
2022-01-25 14:45:18,998 P47737 INFO ************ Epoch=51 end ************
2022-01-25 14:45:21,460 P47737 INFO [Metrics] AUC: 0.934853 - logloss: 0.354852
2022-01-25 14:45:21,460 P47737 INFO Save best model: monitor(max): 0.934853
2022-01-25 14:45:21,462 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:21,504 P47737 INFO Train loss: 0.364275
2022-01-25 14:45:21,504 P47737 INFO ************ Epoch=52 end ************
2022-01-25 14:45:24,633 P47737 INFO [Metrics] AUC: 0.934911 - logloss: 0.352947
2022-01-25 14:45:24,633 P47737 INFO Save best model: monitor(max): 0.934911
2022-01-25 14:45:24,635 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:24,674 P47737 INFO Train loss: 0.362775
2022-01-25 14:45:24,674 P47737 INFO ************ Epoch=53 end ************
2022-01-25 14:45:27,379 P47737 INFO [Metrics] AUC: 0.934994 - logloss: 0.351073
2022-01-25 14:45:27,380 P47737 INFO Save best model: monitor(max): 0.934994
2022-01-25 14:45:27,381 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:27,420 P47737 INFO Train loss: 0.360949
2022-01-25 14:45:27,420 P47737 INFO ************ Epoch=54 end ************
2022-01-25 14:45:29,933 P47737 INFO [Metrics] AUC: 0.935057 - logloss: 0.349291
2022-01-25 14:45:29,933 P47737 INFO Save best model: monitor(max): 0.935057
2022-01-25 14:45:29,935 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:29,972 P47737 INFO Train loss: 0.359739
2022-01-25 14:45:29,972 P47737 INFO ************ Epoch=55 end ************
2022-01-25 14:45:32,993 P47737 INFO [Metrics] AUC: 0.935163 - logloss: 0.347519
2022-01-25 14:45:32,993 P47737 INFO Save best model: monitor(max): 0.935163
2022-01-25 14:45:32,995 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:33,246 P47737 INFO Train loss: 0.357720
2022-01-25 14:45:33,246 P47737 INFO ************ Epoch=56 end ************
2022-01-25 14:45:35,868 P47737 INFO [Metrics] AUC: 0.935208 - logloss: 0.345855
2022-01-25 14:45:35,869 P47737 INFO Save best model: monitor(max): 0.935208
2022-01-25 14:45:35,870 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:35,914 P47737 INFO Train loss: 0.356352
2022-01-25 14:45:35,914 P47737 INFO ************ Epoch=57 end ************
2022-01-25 14:45:39,604 P47737 INFO [Metrics] AUC: 0.935308 - logloss: 0.344208
2022-01-25 14:45:39,605 P47737 INFO Save best model: monitor(max): 0.935308
2022-01-25 14:45:39,606 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:39,645 P47737 INFO Train loss: 0.354946
2022-01-25 14:45:39,645 P47737 INFO ************ Epoch=58 end ************
2022-01-25 14:45:42,281 P47737 INFO [Metrics] AUC: 0.935378 - logloss: 0.342639
2022-01-25 14:45:42,282 P47737 INFO Save best model: monitor(max): 0.935378
2022-01-25 14:45:42,284 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:42,327 P47737 INFO Train loss: 0.353478
2022-01-25 14:45:42,327 P47737 INFO ************ Epoch=59 end ************
2022-01-25 14:45:46,255 P47737 INFO [Metrics] AUC: 0.935457 - logloss: 0.341094
2022-01-25 14:45:46,255 P47737 INFO Save best model: monitor(max): 0.935457
2022-01-25 14:45:46,257 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:46,300 P47737 INFO Train loss: 0.352031
2022-01-25 14:45:46,301 P47737 INFO ************ Epoch=60 end ************
2022-01-25 14:45:48,999 P47737 INFO [Metrics] AUC: 0.935528 - logloss: 0.339630
2022-01-25 14:45:49,000 P47737 INFO Save best model: monitor(max): 0.935528
2022-01-25 14:45:49,001 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:49,039 P47737 INFO Train loss: 0.350778
2022-01-25 14:45:49,040 P47737 INFO ************ Epoch=61 end ************
2022-01-25 14:45:52,752 P47737 INFO [Metrics] AUC: 0.935578 - logloss: 0.338174
2022-01-25 14:45:52,753 P47737 INFO Save best model: monitor(max): 0.935578
2022-01-25 14:45:52,754 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:52,790 P47737 INFO Train loss: 0.349333
2022-01-25 14:45:52,790 P47737 INFO ************ Epoch=62 end ************
2022-01-25 14:45:56,885 P47737 INFO [Metrics] AUC: 0.935656 - logloss: 0.336777
2022-01-25 14:45:56,886 P47737 INFO Save best model: monitor(max): 0.935656
2022-01-25 14:45:56,887 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:45:56,925 P47737 INFO Train loss: 0.347926
2022-01-25 14:45:56,925 P47737 INFO ************ Epoch=63 end ************
2022-01-25 14:46:01,723 P47737 INFO [Metrics] AUC: 0.935703 - logloss: 0.335459
2022-01-25 14:46:01,724 P47737 INFO Save best model: monitor(max): 0.935703
2022-01-25 14:46:01,725 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:46:01,764 P47737 INFO Train loss: 0.346745
2022-01-25 14:46:01,764 P47737 INFO ************ Epoch=64 end ************
2022-01-25 14:46:04,729 P47737 INFO [Metrics] AUC: 0.935786 - logloss: 0.334109
2022-01-25 14:46:04,730 P47737 INFO Save best model: monitor(max): 0.935786
2022-01-25 14:46:04,731 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:46:04,795 P47737 INFO Train loss: 0.346003
2022-01-25 14:46:04,796 P47737 INFO ************ Epoch=65 end ************
2022-01-25 14:46:09,299 P47737 INFO [Metrics] AUC: 0.935866 - logloss: 0.332835
2022-01-25 14:46:09,299 P47737 INFO Save best model: monitor(max): 0.935866
2022-01-25 14:46:09,300 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:46:09,342 P47737 INFO Train loss: 0.345050
2022-01-25 14:46:09,342 P47737 INFO ************ Epoch=66 end ************
2022-01-25 14:46:13,987 P47737 INFO [Metrics] AUC: 0.935920 - logloss: 0.331600
2022-01-25 14:46:13,988 P47737 INFO Save best model: monitor(max): 0.935920
2022-01-25 14:46:13,989 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:46:14,025 P47737 INFO Train loss: 0.343479
2022-01-25 14:46:14,025 P47737 INFO ************ Epoch=67 end ************
2022-01-25 14:46:18,440 P47737 INFO [Metrics] AUC: 0.935985 - logloss: 0.330417
2022-01-25 14:46:18,440 P47737 INFO Save best model: monitor(max): 0.935985
2022-01-25 14:46:18,442 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:46:18,480 P47737 INFO Train loss: 0.342359
2022-01-25 14:46:18,480 P47737 INFO ************ Epoch=68 end ************
2022-01-25 14:46:22,619 P47737 INFO [Metrics] AUC: 0.936029 - logloss: 0.329220
2022-01-25 14:46:22,619 P47737 INFO Save best model: monitor(max): 0.936029
2022-01-25 14:46:22,621 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:46:22,660 P47737 INFO Train loss: 0.341449
2022-01-25 14:46:22,660 P47737 INFO ************ Epoch=69 end ************
2022-01-25 14:46:27,886 P47737 INFO [Metrics] AUC: 0.936118 - logloss: 0.328134
2022-01-25 14:46:27,887 P47737 INFO Save best model: monitor(max): 0.936118
2022-01-25 14:46:27,888 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:46:27,923 P47737 INFO Train loss: 0.340625
2022-01-25 14:46:27,924 P47737 INFO ************ Epoch=70 end ************
2022-01-25 14:46:31,262 P47737 INFO [Metrics] AUC: 0.936176 - logloss: 0.327016
2022-01-25 14:46:31,263 P47737 INFO Save best model: monitor(max): 0.936176
2022-01-25 14:46:31,264 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:46:31,303 P47737 INFO Train loss: 0.339353
2022-01-25 14:46:31,303 P47737 INFO ************ Epoch=71 end ************
2022-01-25 14:46:34,004 P47737 INFO [Metrics] AUC: 0.936232 - logloss: 0.325937
2022-01-25 14:46:34,005 P47737 INFO Save best model: monitor(max): 0.936232
2022-01-25 14:46:34,006 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:46:34,043 P47737 INFO Train loss: 0.338691
2022-01-25 14:46:34,043 P47737 INFO ************ Epoch=72 end ************
2022-01-25 14:46:37,525 P47737 INFO [Metrics] AUC: 0.936294 - logloss: 0.324912
2022-01-25 14:46:37,526 P47737 INFO Save best model: monitor(max): 0.936294
2022-01-25 14:46:37,527 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:46:37,563 P47737 INFO Train loss: 0.337619
2022-01-25 14:46:37,564 P47737 INFO ************ Epoch=73 end ************
2022-01-25 14:46:40,128 P47737 INFO [Metrics] AUC: 0.936357 - logloss: 0.323927
2022-01-25 14:46:40,129 P47737 INFO Save best model: monitor(max): 0.936357
2022-01-25 14:46:40,130 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:46:40,167 P47737 INFO Train loss: 0.336715
2022-01-25 14:46:40,167 P47737 INFO ************ Epoch=74 end ************
2022-01-25 14:46:42,601 P47737 INFO [Metrics] AUC: 0.936418 - logloss: 0.322980
2022-01-25 14:46:42,601 P47737 INFO Save best model: monitor(max): 0.936418
2022-01-25 14:46:42,603 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:46:43,232 P47737 INFO Train loss: 0.335922
2022-01-25 14:46:43,232 P47737 INFO ************ Epoch=75 end ************
2022-01-25 14:46:48,164 P47737 INFO [Metrics] AUC: 0.936460 - logloss: 0.321988
2022-01-25 14:46:48,165 P47737 INFO Save best model: monitor(max): 0.936460
2022-01-25 14:46:48,166 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:46:48,205 P47737 INFO Train loss: 0.335099
2022-01-25 14:46:48,205 P47737 INFO ************ Epoch=76 end ************
2022-01-25 14:46:50,872 P47737 INFO [Metrics] AUC: 0.936523 - logloss: 0.321064
2022-01-25 14:46:50,873 P47737 INFO Save best model: monitor(max): 0.936523
2022-01-25 14:46:50,874 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:46:50,931 P47737 INFO Train loss: 0.334094
2022-01-25 14:46:50,931 P47737 INFO ************ Epoch=77 end ************
2022-01-25 14:46:54,125 P47737 INFO [Metrics] AUC: 0.936607 - logloss: 0.320193
2022-01-25 14:46:54,125 P47737 INFO Save best model: monitor(max): 0.936607
2022-01-25 14:46:54,127 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:46:54,166 P47737 INFO Train loss: 0.333446
2022-01-25 14:46:54,166 P47737 INFO ************ Epoch=78 end ************
2022-01-25 14:46:57,207 P47737 INFO [Metrics] AUC: 0.936654 - logloss: 0.319306
2022-01-25 14:46:57,207 P47737 INFO Save best model: monitor(max): 0.936654
2022-01-25 14:46:57,208 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:46:57,249 P47737 INFO Train loss: 0.332650
2022-01-25 14:46:57,249 P47737 INFO ************ Epoch=79 end ************
2022-01-25 14:47:00,056 P47737 INFO [Metrics] AUC: 0.936715 - logloss: 0.318509
2022-01-25 14:47:00,056 P47737 INFO Save best model: monitor(max): 0.936715
2022-01-25 14:47:00,057 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:00,095 P47737 INFO Train loss: 0.331889
2022-01-25 14:47:00,095 P47737 INFO ************ Epoch=80 end ************
2022-01-25 14:47:03,432 P47737 INFO [Metrics] AUC: 0.936776 - logloss: 0.317651
2022-01-25 14:47:03,433 P47737 INFO Save best model: monitor(max): 0.936776
2022-01-25 14:47:03,434 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:03,472 P47737 INFO Train loss: 0.331219
2022-01-25 14:47:03,472 P47737 INFO ************ Epoch=81 end ************
2022-01-25 14:47:07,601 P47737 INFO [Metrics] AUC: 0.936822 - logloss: 0.316865
2022-01-25 14:47:07,602 P47737 INFO Save best model: monitor(max): 0.936822
2022-01-25 14:47:07,604 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:07,644 P47737 INFO Train loss: 0.330543
2022-01-25 14:47:07,644 P47737 INFO ************ Epoch=82 end ************
2022-01-25 14:47:10,302 P47737 INFO [Metrics] AUC: 0.936909 - logloss: 0.316109
2022-01-25 14:47:10,302 P47737 INFO Save best model: monitor(max): 0.936909
2022-01-25 14:47:10,304 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:10,366 P47737 INFO Train loss: 0.329647
2022-01-25 14:47:10,366 P47737 INFO ************ Epoch=83 end ************
2022-01-25 14:47:13,024 P47737 INFO [Metrics] AUC: 0.936958 - logloss: 0.315314
2022-01-25 14:47:13,024 P47737 INFO Save best model: monitor(max): 0.936958
2022-01-25 14:47:13,025 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:13,066 P47737 INFO Train loss: 0.329190
2022-01-25 14:47:13,066 P47737 INFO ************ Epoch=84 end ************
2022-01-25 14:47:16,640 P47737 INFO [Metrics] AUC: 0.937000 - logloss: 0.314581
2022-01-25 14:47:16,641 P47737 INFO Save best model: monitor(max): 0.937000
2022-01-25 14:47:16,642 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:16,683 P47737 INFO Train loss: 0.328315
2022-01-25 14:47:16,684 P47737 INFO ************ Epoch=85 end ************
2022-01-25 14:47:20,250 P47737 INFO [Metrics] AUC: 0.937047 - logloss: 0.313872
2022-01-25 14:47:20,250 P47737 INFO Save best model: monitor(max): 0.937047
2022-01-25 14:47:20,251 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:20,289 P47737 INFO Train loss: 0.327908
2022-01-25 14:47:20,290 P47737 INFO ************ Epoch=86 end ************
2022-01-25 14:47:23,041 P47737 INFO [Metrics] AUC: 0.937100 - logloss: 0.313210
2022-01-25 14:47:23,041 P47737 INFO Save best model: monitor(max): 0.937100
2022-01-25 14:47:23,043 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:23,080 P47737 INFO Train loss: 0.327253
2022-01-25 14:47:23,080 P47737 INFO ************ Epoch=87 end ************
2022-01-25 14:47:25,922 P47737 INFO [Metrics] AUC: 0.937162 - logloss: 0.312510
2022-01-25 14:47:25,922 P47737 INFO Save best model: monitor(max): 0.937162
2022-01-25 14:47:25,923 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:25,963 P47737 INFO Train loss: 0.326612
2022-01-25 14:47:25,963 P47737 INFO ************ Epoch=88 end ************
2022-01-25 14:47:29,985 P47737 INFO [Metrics] AUC: 0.937197 - logloss: 0.311853
2022-01-25 14:47:29,985 P47737 INFO Save best model: monitor(max): 0.937197
2022-01-25 14:47:29,986 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:30,023 P47737 INFO Train loss: 0.326250
2022-01-25 14:47:30,023 P47737 INFO ************ Epoch=89 end ************
2022-01-25 14:47:32,791 P47737 INFO [Metrics] AUC: 0.937251 - logloss: 0.311236
2022-01-25 14:47:32,792 P47737 INFO Save best model: monitor(max): 0.937251
2022-01-25 14:47:32,793 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:32,835 P47737 INFO Train loss: 0.325575
2022-01-25 14:47:32,835 P47737 INFO ************ Epoch=90 end ************
2022-01-25 14:47:35,280 P47737 INFO [Metrics] AUC: 0.937310 - logloss: 0.310596
2022-01-25 14:47:35,280 P47737 INFO Save best model: monitor(max): 0.937310
2022-01-25 14:47:35,282 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:35,319 P47737 INFO Train loss: 0.324768
2022-01-25 14:47:35,319 P47737 INFO ************ Epoch=91 end ************
2022-01-25 14:47:37,489 P47737 INFO [Metrics] AUC: 0.937361 - logloss: 0.309999
2022-01-25 14:47:37,489 P47737 INFO Save best model: monitor(max): 0.937361
2022-01-25 14:47:37,491 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:37,529 P47737 INFO Train loss: 0.324268
2022-01-25 14:47:37,529 P47737 INFO ************ Epoch=92 end ************
2022-01-25 14:47:41,372 P47737 INFO [Metrics] AUC: 0.937394 - logloss: 0.309416
2022-01-25 14:47:41,373 P47737 INFO Save best model: monitor(max): 0.937394
2022-01-25 14:47:41,374 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:41,414 P47737 INFO Train loss: 0.323626
2022-01-25 14:47:41,414 P47737 INFO ************ Epoch=93 end ************
2022-01-25 14:47:43,604 P47737 INFO [Metrics] AUC: 0.937457 - logloss: 0.308837
2022-01-25 14:47:43,604 P47737 INFO Save best model: monitor(max): 0.937457
2022-01-25 14:47:43,606 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:43,643 P47737 INFO Train loss: 0.322918
2022-01-25 14:47:43,643 P47737 INFO ************ Epoch=94 end ************
2022-01-25 14:47:46,125 P47737 INFO [Metrics] AUC: 0.937491 - logloss: 0.308278
2022-01-25 14:47:46,126 P47737 INFO Save best model: monitor(max): 0.937491
2022-01-25 14:47:46,127 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:46,168 P47737 INFO Train loss: 0.322741
2022-01-25 14:47:46,168 P47737 INFO ************ Epoch=95 end ************
2022-01-25 14:47:48,661 P47737 INFO [Metrics] AUC: 0.937532 - logloss: 0.307740
2022-01-25 14:47:48,661 P47737 INFO Save best model: monitor(max): 0.937532
2022-01-25 14:47:48,663 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:48,700 P47737 INFO Train loss: 0.322171
2022-01-25 14:47:48,700 P47737 INFO ************ Epoch=96 end ************
2022-01-25 14:47:51,059 P47737 INFO [Metrics] AUC: 0.937588 - logloss: 0.307224
2022-01-25 14:47:51,059 P47737 INFO Save best model: monitor(max): 0.937588
2022-01-25 14:47:51,061 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:51,099 P47737 INFO Train loss: 0.321593
2022-01-25 14:47:51,099 P47737 INFO ************ Epoch=97 end ************
2022-01-25 14:47:53,759 P47737 INFO [Metrics] AUC: 0.937654 - logloss: 0.306676
2022-01-25 14:47:53,759 P47737 INFO Save best model: monitor(max): 0.937654
2022-01-25 14:47:53,761 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:53,799 P47737 INFO Train loss: 0.321075
2022-01-25 14:47:53,799 P47737 INFO ************ Epoch=98 end ************
2022-01-25 14:47:56,141 P47737 INFO [Metrics] AUC: 0.937704 - logloss: 0.306204
2022-01-25 14:47:56,141 P47737 INFO Save best model: monitor(max): 0.937704
2022-01-25 14:47:56,142 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:56,178 P47737 INFO Train loss: 0.320783
2022-01-25 14:47:56,178 P47737 INFO ************ Epoch=99 end ************
2022-01-25 14:47:58,269 P47737 INFO [Metrics] AUC: 0.937734 - logloss: 0.305725
2022-01-25 14:47:58,270 P47737 INFO Save best model: monitor(max): 0.937734
2022-01-25 14:47:58,271 P47737 INFO --- 50/50 batches finished ---
2022-01-25 14:47:58,316 P47737 INFO Train loss: 0.320487
2022-01-25 14:47:58,316 P47737 INFO ************ Epoch=100 end ************
2022-01-25 14:47:58,316 P47737 INFO Training finished.
2022-01-25 14:47:58,316 P47737 INFO Load best model: /home/XXX/benchmarks/Frappe/LR_frappe_x1/frappe_x1_04e961e9/LR_frappe_x1_001_ff730f43.model
2022-01-25 14:47:58,328 P47737 INFO ****** Validation evaluation ******
2022-01-25 14:47:58,754 P47737 INFO [Metrics] AUC: 0.937734 - logloss: 0.305725
2022-01-25 14:47:58,817 P47737 INFO ******** Test evaluation ********
2022-01-25 14:47:58,818 P47737 INFO Loading data...
2022-01-25 14:47:58,818 P47737 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-01-25 14:47:58,821 P47737 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-01-25 14:47:58,821 P47737 INFO Loading test data done.
2022-01-25 14:47:59,045 P47737 INFO [Metrics] AUC: 0.935584 - logloss: 0.307643

```
