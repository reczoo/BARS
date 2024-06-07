## FM_amazonelectronics_x1

A hands-on guide to run the FM model on the AmazonElectronics_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)


| [Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs) |
|:-----------------------------:|:-----------:|:--------:|:--------:|-------|
### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) Gold 6278C CPU @ 2.60GHz
  GPU: Tesla V100 32G
  RAM: 755G

  ```

+ Software

  ```python
  cuda: 10.2
  python: 3.7.10
  pytorch: 1.11.0
  pandas: 1.1.5
  numpy: 1.19.5
  scipy: 1.5.2
  sklearn: 0.22.1
  pyyaml: 6.0
  h5py: 2.8.0
  tqdm: 4.64.0
  fuxictr: 2.0.1

  ```

### Dataset
Please refer to [AmazonElectronics_x1](https://github.com/reczoo/Datasets/tree/main/Amazon/AmazonElectronics_x1) to get the dataset details.

### Code

We use the [FM](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/FM) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Amazon/AmazonElectronics_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FM_amazonelectronics_x1_tuner_config_02](./FM_amazonelectronics_x1_tuner_config_02). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/FM
    nohup python run_expid.py --config XXX/benchmarks/FM/FM_amazonelectronics_x1_tuner_config_02 --expid FM_amazonelectronics_x1_009_a87f4871 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.849405 | 0.848518 | 0.506075  |


### Logs
```python
2022-08-11 14:26:37,636 P52963 INFO Params: {
    "batch_size": "1024",
    "data_format": "csv",
    "data_root": "../data/Amazon/",
    "dataset_id": "amazonelectronics_x1_b7a43f49",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'user_id', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'item_history', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'cate_history', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "None",
    "gpu": "0",
    "group_id": "user_id",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FM",
    "model_id": "FM_amazonelectronics_x1_009_a87f4871",
    "model_root": "./checkpoints/FM_amazonelectronics_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "regularizer": "5e-07",
    "save_best_only": "True",
    "seed": "20222023",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "train_data": "../data/Amazon/AmazonElectronics_x1/train.csv",
    "valid_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "verbose": "1"
}
2022-08-11 14:26:37,637 P52963 INFO Set up feature processor...
2022-08-11 14:26:37,637 P52963 INFO Load feature_map from json: ../data/Amazon/amazonelectronics_x1_b7a43f49/feature_map.json
2022-08-11 14:26:37,638 P52963 INFO Set column index...
2022-08-11 14:26:37,638 P52963 INFO Feature specs: {
    "cate_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803, 'max_len': 100}",
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803}",
    "item_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003, 'max_len': 100}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003}",
    "user_id": "{'type': 'meta'}"
}
2022-08-11 14:26:43,694 P52963 INFO Total number of parameters: 4211197.
2022-08-11 14:26:43,694 P52963 INFO Loading data...
2022-08-11 14:26:43,694 P52963 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/train.h5
2022-08-11 14:26:47,535 P52963 INFO Train samples: total/2608764, blocks/1
2022-08-11 14:26:47,535 P52963 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/valid.h5
2022-08-11 14:26:48,095 P52963 INFO Validation samples: total/384806, blocks/1
2022-08-11 14:26:48,095 P52963 INFO Loading train and validation data done.
2022-08-11 14:26:48,095 P52963 INFO Start training: 2548 batches/epoch
2022-08-11 14:26:48,095 P52963 INFO ************ Epoch=1 start ************
2022-08-11 14:35:47,451 P52963 INFO [Metrics] AUC: 0.816742 - gAUC: 0.822716
2022-08-11 14:35:47,612 P52963 INFO Save best model: monitor(max): 1.639457
2022-08-11 14:35:47,640 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 14:35:47,741 P52963 INFO Train loss: 0.565701
2022-08-11 14:35:47,741 P52963 INFO ************ Epoch=1 end ************
2022-08-11 14:44:43,014 P52963 INFO [Metrics] AUC: 0.822295 - gAUC: 0.825081
2022-08-11 14:44:43,177 P52963 INFO Save best model: monitor(max): 1.647376
2022-08-11 14:44:43,207 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 14:44:43,333 P52963 INFO Train loss: 0.494651
2022-08-11 14:44:43,333 P52963 INFO ************ Epoch=2 end ************
2022-08-11 14:53:39,831 P52963 INFO [Metrics] AUC: 0.826478 - gAUC: 0.827976
2022-08-11 14:53:39,973 P52963 INFO Save best model: monitor(max): 1.654454
2022-08-11 14:53:40,021 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 14:53:40,115 P52963 INFO Train loss: 0.471983
2022-08-11 14:53:40,115 P52963 INFO ************ Epoch=3 end ************
2022-08-11 15:02:12,214 P52963 INFO [Metrics] AUC: 0.830167 - gAUC: 0.830959
2022-08-11 15:02:12,364 P52963 INFO Save best model: monitor(max): 1.661126
2022-08-11 15:02:12,392 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 15:02:12,507 P52963 INFO Train loss: 0.455408
2022-08-11 15:02:12,507 P52963 INFO ************ Epoch=4 end ************
2022-08-11 15:10:28,017 P52963 INFO [Metrics] AUC: 0.833673 - gAUC: 0.834140
2022-08-11 15:10:28,205 P52963 INFO Save best model: monitor(max): 1.667813
2022-08-11 15:10:28,277 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 15:10:28,435 P52963 INFO Train loss: 0.441037
2022-08-11 15:10:28,436 P52963 INFO ************ Epoch=5 end ************
2022-08-11 15:18:41,596 P52963 INFO [Metrics] AUC: 0.836441 - gAUC: 0.836697
2022-08-11 15:18:41,807 P52963 INFO Save best model: monitor(max): 1.673138
2022-08-11 15:18:41,897 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 15:18:42,023 P52963 INFO Train loss: 0.428023
2022-08-11 15:18:42,024 P52963 INFO ************ Epoch=6 end ************
2022-08-11 15:26:54,587 P52963 INFO [Metrics] AUC: 0.838800 - gAUC: 0.839540
2022-08-11 15:26:54,889 P52963 INFO Save best model: monitor(max): 1.678340
2022-08-11 15:26:54,937 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 15:26:55,095 P52963 INFO Train loss: 0.416043
2022-08-11 15:26:55,095 P52963 INFO ************ Epoch=7 end ************
2022-08-11 15:34:49,114 P52963 INFO [Metrics] AUC: 0.840816 - gAUC: 0.841110
2022-08-11 15:34:49,325 P52963 INFO Save best model: monitor(max): 1.681926
2022-08-11 15:34:49,390 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 15:34:49,525 P52963 INFO Train loss: 0.404847
2022-08-11 15:34:49,525 P52963 INFO ************ Epoch=8 end ************
2022-08-11 15:42:31,549 P52963 INFO [Metrics] AUC: 0.842340 - gAUC: 0.842809
2022-08-11 15:42:31,758 P52963 INFO Save best model: monitor(max): 1.685149
2022-08-11 15:42:31,807 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 15:42:32,074 P52963 INFO Train loss: 0.394446
2022-08-11 15:42:32,074 P52963 INFO ************ Epoch=9 end ************
2022-08-11 15:50:11,311 P52963 INFO [Metrics] AUC: 0.843834 - gAUC: 0.844566
2022-08-11 15:50:11,496 P52963 INFO Save best model: monitor(max): 1.688400
2022-08-11 15:50:11,534 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 15:50:11,633 P52963 INFO Train loss: 0.384703
2022-08-11 15:50:11,633 P52963 INFO ************ Epoch=10 end ************
2022-08-11 15:57:48,312 P52963 INFO [Metrics] AUC: 0.844721 - gAUC: 0.845881
2022-08-11 15:57:48,516 P52963 INFO Save best model: monitor(max): 1.690602
2022-08-11 15:57:48,555 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 15:57:48,668 P52963 INFO Train loss: 0.375602
2022-08-11 15:57:48,668 P52963 INFO ************ Epoch=11 end ************
2022-08-11 16:05:29,961 P52963 INFO [Metrics] AUC: 0.845731 - gAUC: 0.846510
2022-08-11 16:05:30,170 P52963 INFO Save best model: monitor(max): 1.692240
2022-08-11 16:05:30,219 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 16:05:30,338 P52963 INFO Train loss: 0.367065
2022-08-11 16:05:30,338 P52963 INFO ************ Epoch=12 end ************
2022-08-11 16:13:04,073 P52963 INFO [Metrics] AUC: 0.846387 - gAUC: 0.847310
2022-08-11 16:13:04,271 P52963 INFO Save best model: monitor(max): 1.693698
2022-08-11 16:13:04,343 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 16:13:04,477 P52963 INFO Train loss: 0.359051
2022-08-11 16:13:04,477 P52963 INFO ************ Epoch=13 end ************
2022-08-11 16:20:48,923 P52963 INFO [Metrics] AUC: 0.846931 - gAUC: 0.847861
2022-08-11 16:20:49,171 P52963 INFO Save best model: monitor(max): 1.694792
2022-08-11 16:20:49,221 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 16:20:49,356 P52963 INFO Train loss: 0.351549
2022-08-11 16:20:49,357 P52963 INFO ************ Epoch=14 end ************
2022-08-11 16:28:25,103 P52963 INFO [Metrics] AUC: 0.847262 - gAUC: 0.848500
2022-08-11 16:28:25,316 P52963 INFO Save best model: monitor(max): 1.695762
2022-08-11 16:28:25,353 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 16:28:25,486 P52963 INFO Train loss: 0.344571
2022-08-11 16:28:25,486 P52963 INFO ************ Epoch=15 end ************
2022-08-11 16:35:58,881 P52963 INFO [Metrics] AUC: 0.847647 - gAUC: 0.848781
2022-08-11 16:35:59,072 P52963 INFO Save best model: monitor(max): 1.696428
2022-08-11 16:35:59,123 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 16:35:59,256 P52963 INFO Train loss: 0.338148
2022-08-11 16:35:59,256 P52963 INFO ************ Epoch=16 end ************
2022-08-11 16:43:32,709 P52963 INFO [Metrics] AUC: 0.848002 - gAUC: 0.849202
2022-08-11 16:43:32,925 P52963 INFO Save best model: monitor(max): 1.697204
2022-08-11 16:43:32,970 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 16:43:33,147 P52963 INFO Train loss: 0.331919
2022-08-11 16:43:33,148 P52963 INFO ************ Epoch=17 end ************
2022-08-11 16:51:07,932 P52963 INFO [Metrics] AUC: 0.848199 - gAUC: 0.849420
2022-08-11 16:51:08,130 P52963 INFO Save best model: monitor(max): 1.697619
2022-08-11 16:51:08,173 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 16:51:08,320 P52963 INFO Train loss: 0.326154
2022-08-11 16:51:08,320 P52963 INFO ************ Epoch=18 end ************
2022-08-11 16:58:46,554 P52963 INFO [Metrics] AUC: 0.848130 - gAUC: 0.849212
2022-08-11 16:58:46,733 P52963 INFO Monitor(max) STOP: 1.697342 !
2022-08-11 16:58:46,733 P52963 INFO Reduce learning rate on plateau: 0.000050
2022-08-11 16:58:46,734 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 16:58:46,856 P52963 INFO Train loss: 0.320874
2022-08-11 16:58:46,856 P52963 INFO ************ Epoch=19 end ************
2022-08-11 17:06:14,145 P52963 INFO [Metrics] AUC: 0.848382 - gAUC: 0.849431
2022-08-11 17:06:14,380 P52963 INFO Save best model: monitor(max): 1.697813
2022-08-11 17:06:14,426 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 17:06:14,539 P52963 INFO Train loss: 0.303372
2022-08-11 17:06:14,540 P52963 INFO ************ Epoch=20 end ************
2022-08-11 17:12:08,110 P52963 INFO [Metrics] AUC: 0.848503 - gAUC: 0.849415
2022-08-11 17:12:08,278 P52963 INFO Save best model: monitor(max): 1.697918
2022-08-11 17:12:08,316 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 17:12:08,408 P52963 INFO Train loss: 0.302647
2022-08-11 17:12:08,408 P52963 INFO ************ Epoch=21 end ************
2022-08-11 17:17:05,637 P52963 INFO [Metrics] AUC: 0.848510 - gAUC: 0.849394
2022-08-11 17:17:05,837 P52963 INFO Monitor(max) STOP: 1.697904 !
2022-08-11 17:17:05,838 P52963 INFO Reduce learning rate on plateau: 0.000005
2022-08-11 17:17:05,838 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 17:17:05,927 P52963 INFO Train loss: 0.301992
2022-08-11 17:17:05,927 P52963 INFO ************ Epoch=22 end ************
2022-08-11 17:21:01,326 P52963 INFO [Metrics] AUC: 0.848518 - gAUC: 0.849405
2022-08-11 17:21:01,463 P52963 INFO Save best model: monitor(max): 1.697922
2022-08-11 17:21:01,492 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 17:21:01,583 P52963 INFO Train loss: 0.300003
2022-08-11 17:21:01,583 P52963 INFO ************ Epoch=23 end ************
2022-08-11 17:24:24,699 P52963 INFO [Metrics] AUC: 0.848525 - gAUC: 0.849394
2022-08-11 17:24:24,898 P52963 INFO Monitor(max) STOP: 1.697919 !
2022-08-11 17:24:24,898 P52963 INFO Reduce learning rate on plateau: 0.000001
2022-08-11 17:24:24,898 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 17:24:25,006 P52963 INFO Train loss: 0.299946
2022-08-11 17:24:25,006 P52963 INFO ************ Epoch=24 end ************
2022-08-11 17:27:14,036 P52963 INFO [Metrics] AUC: 0.848526 - gAUC: 0.849394
2022-08-11 17:27:14,177 P52963 INFO Monitor(max) STOP: 1.697920 !
2022-08-11 17:27:14,177 P52963 INFO Reduce learning rate on plateau: 0.000001
2022-08-11 17:27:14,177 P52963 INFO ********* Epoch==25 early stop *********
2022-08-11 17:27:14,177 P52963 INFO --- 2548/2548 batches finished ---
2022-08-11 17:27:14,277 P52963 INFO Train loss: 0.299754
2022-08-11 17:27:14,277 P52963 INFO Training finished.
2022-08-11 17:27:14,277 P52963 INFO Load best model: /cache/FuxiCTRv2.0/benchmark/checkpoints/FM_amazonelectronics_x1/amazonelectronics_x1_b7a43f49/FM_amazonelectronics_x1_009_a87f4871.model
2022-08-11 17:27:14,308 P52963 INFO ****** Validation evaluation ******
2022-08-11 17:27:59,316 P52963 INFO [Metrics] gAUC: 0.849405 - AUC: 0.848518 - logloss: 0.506075
2022-08-11 17:27:59,540 P52963 INFO ******** Test evaluation ********
2022-08-11 17:27:59,540 P52963 INFO Loading data...
2022-08-11 17:27:59,540 P52963 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/test.h5
2022-08-11 17:27:59,947 P52963 INFO Test samples: total/384806, blocks/1
2022-08-11 17:27:59,947 P52963 INFO Loading test data done.
2022-08-11 17:28:45,259 P52963 INFO [Metrics] gAUC: 0.849405 - AUC: 0.848518 - logloss: 0.506075

```
