## xDeepFM_amazonelectronics_x1

A hands-on guide to run the xDeepFM model on the AmazonElectronics_x1 dataset.

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

We use the [xDeepFM](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/xDeepFM) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Amazon/AmazonElectronics_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [xDeepFM_amazonelectronics_x1_tuner_config_04](./xDeepFM_amazonelectronics_x1_tuner_config_04). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/xDeepFM
    nohup python run_expid.py --config XXX/benchmarks/xDeepFM/xDeepFM_amazonelectronics_x1_tuner_config_04 --expid xDeepFM_amazonelectronics_x1_024_abaa0691 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.878973 | 0.881289 | 0.438688  |


### Logs
```python
2022-08-16 10:27:39,781 P19982 INFO Params: {
    "batch_norm": "True",
    "batch_size": "1024",
    "cin_hidden_units": "[256, 256, 256]",
    "data_format": "csv",
    "data_root": "../data/Amazon/",
    "dataset_id": "amazonelectronics_x1_b7a43f49",
    "debug_mode": "False",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.005",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'user_id', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'item_history', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'cate_history', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "None",
    "gpu": "1",
    "group_id": "user_id",
    "hidden_activations": "relu",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "xDeepFM",
    "model_id": "xDeepFM_amazonelectronics_x1_024_abaa0691",
    "model_root": "./checkpoints/xDeepFM_amazonelectronics_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "20222023",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "train_data": "../data/Amazon/AmazonElectronics_x1/train.csv",
    "valid_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "verbose": "1"
}
2022-08-16 10:27:39,782 P19982 INFO Set up feature processor...
2022-08-16 10:27:39,782 P19982 INFO Load feature_map from json: ../data/Amazon/amazonelectronics_x1_b7a43f49/feature_map.json
2022-08-16 10:27:39,783 P19982 INFO Set column index...
2022-08-16 10:27:39,783 P19982 INFO Feature specs: {
    "cate_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803, 'max_len': 100}",
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803}",
    "item_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003, 'max_len': 100}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003}",
    "user_id": "{'type': 'meta'}"
}
2022-08-16 10:27:44,966 P19982 INFO Total number of parameters: 5664254.
2022-08-16 10:27:44,966 P19982 INFO Loading data...
2022-08-16 10:27:44,966 P19982 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/train.h5
2022-08-16 10:27:48,210 P19982 INFO Train samples: total/2608764, blocks/1
2022-08-16 10:27:48,210 P19982 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/valid.h5
2022-08-16 10:27:48,673 P19982 INFO Validation samples: total/384806, blocks/1
2022-08-16 10:27:48,673 P19982 INFO Loading train and validation data done.
2022-08-16 10:27:48,673 P19982 INFO Start training: 2548 batches/epoch
2022-08-16 10:27:48,673 P19982 INFO ************ Epoch=1 start ************
2022-08-16 10:39:08,213 P19982 INFO [Metrics] AUC: 0.830311 - gAUC: 0.828417
2022-08-16 10:39:08,369 P19982 INFO Save best model: monitor(max): 1.658728
2022-08-16 10:39:08,407 P19982 INFO --- 2548/2548 batches finished ---
2022-08-16 10:39:08,533 P19982 INFO Train loss: 0.650891
2022-08-16 10:39:08,533 P19982 INFO ************ Epoch=1 end ************
2022-08-16 10:50:22,418 P19982 INFO [Metrics] AUC: 0.845086 - gAUC: 0.843807
2022-08-16 10:50:22,590 P19982 INFO Save best model: monitor(max): 1.688893
2022-08-16 10:50:22,731 P19982 INFO --- 2548/2548 batches finished ---
2022-08-16 10:50:22,865 P19982 INFO Train loss: 0.599383
2022-08-16 10:50:22,866 P19982 INFO ************ Epoch=2 end ************
2022-08-16 11:01:28,451 P19982 INFO [Metrics] AUC: 0.849279 - gAUC: 0.846229
2022-08-16 11:01:28,705 P19982 INFO Save best model: monitor(max): 1.695508
2022-08-16 11:01:28,781 P19982 INFO --- 2548/2548 batches finished ---
2022-08-16 11:01:28,891 P19982 INFO Train loss: 0.581513
2022-08-16 11:01:28,892 P19982 INFO ************ Epoch=3 end ************
2022-08-16 11:12:22,851 P19982 INFO [Metrics] AUC: 0.850985 - gAUC: 0.848557
2022-08-16 11:12:23,025 P19982 INFO Save best model: monitor(max): 1.699543
2022-08-16 11:12:23,080 P19982 INFO --- 2548/2548 batches finished ---
2022-08-16 11:12:23,185 P19982 INFO Train loss: 0.575012
2022-08-16 11:12:23,186 P19982 INFO ************ Epoch=4 end ************
2022-08-16 11:22:32,369 P19982 INFO [Metrics] AUC: 0.853699 - gAUC: 0.851229
2022-08-16 11:22:32,578 P19982 INFO Save best model: monitor(max): 1.704928
2022-08-16 11:22:32,645 P19982 INFO --- 2548/2548 batches finished ---
2022-08-16 11:22:32,739 P19982 INFO Train loss: 0.570716
2022-08-16 11:22:32,739 P19982 INFO ************ Epoch=5 end ************
2022-08-16 11:32:28,620 P19982 INFO [Metrics] AUC: 0.855342 - gAUC: 0.852851
2022-08-16 11:32:28,793 P19982 INFO Save best model: monitor(max): 1.708192
2022-08-16 11:32:28,841 P19982 INFO --- 2548/2548 batches finished ---
2022-08-16 11:32:28,964 P19982 INFO Train loss: 0.568558
2022-08-16 11:32:28,965 P19982 INFO ************ Epoch=6 end ************
2022-08-16 11:42:19,507 P19982 INFO [Metrics] AUC: 0.855530 - gAUC: 0.852939
2022-08-16 11:42:19,689 P19982 INFO Save best model: monitor(max): 1.708468
2022-08-16 11:42:19,740 P19982 INFO --- 2548/2548 batches finished ---
2022-08-16 11:42:19,862 P19982 INFO Train loss: 0.567427
2022-08-16 11:42:19,863 P19982 INFO ************ Epoch=7 end ************
2022-08-16 11:51:32,960 P19982 INFO [Metrics] AUC: 0.855768 - gAUC: 0.853791
2022-08-16 11:51:33,113 P19982 INFO Save best model: monitor(max): 1.709559
2022-08-16 11:51:33,162 P19982 INFO --- 2548/2548 batches finished ---
2022-08-16 11:51:33,255 P19982 INFO Train loss: 0.567174
2022-08-16 11:51:33,256 P19982 INFO ************ Epoch=8 end ************
2022-08-16 12:00:21,031 P19982 INFO [Metrics] AUC: 0.857003 - gAUC: 0.854399
2022-08-16 12:00:21,204 P19982 INFO Save best model: monitor(max): 1.711402
2022-08-16 12:00:21,262 P19982 INFO --- 2548/2548 batches finished ---
2022-08-16 12:00:21,367 P19982 INFO Train loss: 0.566651
2022-08-16 12:00:21,367 P19982 INFO ************ Epoch=9 end ************
2022-08-16 12:09:05,446 P19982 INFO [Metrics] AUC: 0.857532 - gAUC: 0.855761
2022-08-16 12:09:05,591 P19982 INFO Save best model: monitor(max): 1.713293
2022-08-16 12:09:05,641 P19982 INFO --- 2548/2548 batches finished ---
2022-08-16 12:09:05,731 P19982 INFO Train loss: 0.567079
2022-08-16 12:09:05,731 P19982 INFO ************ Epoch=10 end ************
2022-08-16 12:17:22,086 P19982 INFO [Metrics] AUC: 0.856968 - gAUC: 0.855283
2022-08-16 12:17:22,256 P19982 INFO Monitor(max) STOP: 1.712251 !
2022-08-16 12:17:22,256 P19982 INFO Reduce learning rate on plateau: 0.000050
2022-08-16 12:17:22,257 P19982 INFO --- 2548/2548 batches finished ---
2022-08-16 12:17:22,354 P19982 INFO Train loss: 0.567014
2022-08-16 12:17:22,354 P19982 INFO ************ Epoch=11 end ************
2022-08-16 12:24:06,864 P19982 INFO [Metrics] AUC: 0.875023 - gAUC: 0.871883
2022-08-16 12:24:07,020 P19982 INFO Save best model: monitor(max): 1.746907
2022-08-16 12:24:07,071 P19982 INFO --- 2548/2548 batches finished ---
2022-08-16 12:24:07,170 P19982 INFO Train loss: 0.470997
2022-08-16 12:24:07,170 P19982 INFO ************ Epoch=12 end ************
2022-08-16 12:30:24,708 P19982 INFO [Metrics] AUC: 0.880036 - gAUC: 0.876941
2022-08-16 12:30:24,874 P19982 INFO Save best model: monitor(max): 1.756977
2022-08-16 12:30:24,923 P19982 INFO --- 2548/2548 batches finished ---
2022-08-16 12:30:25,023 P19982 INFO Train loss: 0.427599
2022-08-16 12:30:25,023 P19982 INFO ************ Epoch=13 end ************
2022-08-16 12:36:37,105 P19982 INFO [Metrics] AUC: 0.881289 - gAUC: 0.878973
2022-08-16 12:36:37,254 P19982 INFO Save best model: monitor(max): 1.760262
2022-08-16 12:36:37,304 P19982 INFO --- 2548/2548 batches finished ---
2022-08-16 12:36:37,396 P19982 INFO Train loss: 0.409502
2022-08-16 12:36:37,396 P19982 INFO ************ Epoch=14 end ************
2022-08-16 12:42:26,251 P19982 INFO [Metrics] AUC: 0.880933 - gAUC: 0.878536
2022-08-16 12:42:26,379 P19982 INFO Monitor(max) STOP: 1.759469 !
2022-08-16 12:42:26,379 P19982 INFO Reduce learning rate on plateau: 0.000005
2022-08-16 12:42:26,380 P19982 INFO --- 2548/2548 batches finished ---
2022-08-16 12:42:26,451 P19982 INFO Train loss: 0.398272
2022-08-16 12:42:26,451 P19982 INFO ************ Epoch=15 end ************
2022-08-16 12:48:08,173 P19982 INFO [Metrics] AUC: 0.878967 - gAUC: 0.876582
2022-08-16 12:48:08,306 P19982 INFO Monitor(max) STOP: 1.755549 !
2022-08-16 12:48:08,306 P19982 INFO Reduce learning rate on plateau: 0.000001
2022-08-16 12:48:08,306 P19982 INFO ********* Epoch==16 early stop *********
2022-08-16 12:48:08,307 P19982 INFO --- 2548/2548 batches finished ---
2022-08-16 12:48:08,375 P19982 INFO Train loss: 0.348962
2022-08-16 12:48:08,375 P19982 INFO Training finished.
2022-08-16 12:48:08,375 P19982 INFO Load best model: /cache/FuxiCTRv2.0/benchmark/checkpoints/xDeepFM_amazonelectronics_x1/amazonelectronics_x1_b7a43f49/xDeepFM_amazonelectronics_x1_024_abaa0691.model
2022-08-16 12:48:12,127 P19982 INFO ****** Validation evaluation ******
2022-08-16 12:48:57,610 P19982 INFO [Metrics] gAUC: 0.878973 - AUC: 0.881289 - logloss: 0.438688
2022-08-16 12:48:57,812 P19982 INFO ******** Test evaluation ********
2022-08-16 12:48:57,812 P19982 INFO Loading data...
2022-08-16 12:48:57,812 P19982 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/test.h5
2022-08-16 12:48:58,252 P19982 INFO Test samples: total/384806, blocks/1
2022-08-16 12:48:58,252 P19982 INFO Loading test data done.
2022-08-16 12:49:44,862 P19982 INFO [Metrics] gAUC: 0.878973 - AUC: 0.881289 - logloss: 0.438688

```
