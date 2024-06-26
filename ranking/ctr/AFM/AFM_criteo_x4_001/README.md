## AFM_criteo_x4_001

A hands-on guide to run the AFM model on the Criteo_x4_001 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) Gold 6278C CPU @ 2.60GHz
  GPU: Tesla V100 32G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 10.2
  python: 3.6.4
  pytorch: 1.0.0
  pandas: 0.22.0
  numpy: 1.19.2
  scipy: 1.5.4
  sklearn: 0.22.1
  pyyaml: 5.4.1
  h5py: 2.8.0
  tqdm: 4.60.0
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [Criteo_x4](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x4). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [AFM](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/AFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFM_criteo_x4_tuner_config_01](./AFM_criteo_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFM_criteo_x4_001
    nohup python run_expid.py --config ./AFM_criteo_x4_tuner_config_01 --expid AFM_criteo_x4_5c863b0f_010_f040edb0 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.445523 | 0.805961  |


### Logs
```python
2020-07-08 13:08:14,660 P2785 INFO {
    "attention_dim": "16",
    "attention_dropout": "[0, 0]",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-06",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AFM",
    "model_id": "AFM_criteo_x4_5c863b0f_010_f040edb0",
    "model_root": "./Criteo/AFM_criteo/min10/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/criteo_x4_5c863b0f/test.h5",
    "train_data": "../data/Criteo/criteo_x4_5c863b0f/train.h5",
    "use_attention": "True",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-07-08 13:08:14,662 P2785 INFO Set up feature encoder...
2020-07-08 13:08:14,663 P2785 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-07-08 13:08:14,663 P2785 INFO Loading data...
2020-07-08 13:08:14,668 P2785 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-07-08 13:08:21,441 P2785 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-07-08 13:08:23,518 P2785 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-07-08 13:08:23,648 P2785 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-08 13:08:23,648 P2785 INFO Loading train data done.
2020-07-08 13:08:30,810 P2785 INFO Start training: 3668 batches/epoch
2020-07-08 13:08:30,810 P2785 INFO ************ Epoch=1 start ************
2020-07-08 13:20:10,710 P2785 INFO [Metrics] logloss: 0.456584 - AUC: 0.793220
2020-07-08 13:20:10,713 P2785 INFO Save best model: monitor(max): 0.336636
2020-07-08 13:20:11,238 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 13:20:11,297 P2785 INFO Train loss: 0.466129
2020-07-08 13:20:11,297 P2785 INFO ************ Epoch=1 end ************
2020-07-08 13:31:42,948 P2785 INFO [Metrics] logloss: 0.454645 - AUC: 0.795486
2020-07-08 13:31:42,949 P2785 INFO Save best model: monitor(max): 0.340841
2020-07-08 13:31:43,025 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 13:31:43,099 P2785 INFO Train loss: 0.457495
2020-07-08 13:31:43,099 P2785 INFO ************ Epoch=2 end ************
2020-07-08 13:43:14,322 P2785 INFO [Metrics] logloss: 0.453432 - AUC: 0.796853
2020-07-08 13:43:14,323 P2785 INFO Save best model: monitor(max): 0.343421
2020-07-08 13:43:14,396 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 13:43:14,463 P2785 INFO Train loss: 0.456014
2020-07-08 13:43:14,463 P2785 INFO ************ Epoch=3 end ************
2020-07-08 13:54:46,451 P2785 INFO [Metrics] logloss: 0.452259 - AUC: 0.798234
2020-07-08 13:54:46,452 P2785 INFO Save best model: monitor(max): 0.345975
2020-07-08 13:54:46,534 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 13:54:46,601 P2785 INFO Train loss: 0.454830
2020-07-08 13:54:46,601 P2785 INFO ************ Epoch=4 end ************
2020-07-08 14:06:19,462 P2785 INFO [Metrics] logloss: 0.451300 - AUC: 0.799340
2020-07-08 14:06:19,464 P2785 INFO Save best model: monitor(max): 0.348040
2020-07-08 14:06:19,539 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 14:06:19,605 P2785 INFO Train loss: 0.453705
2020-07-08 14:06:19,605 P2785 INFO ************ Epoch=5 end ************
2020-07-08 14:17:52,563 P2785 INFO [Metrics] logloss: 0.450380 - AUC: 0.800430
2020-07-08 14:17:52,565 P2785 INFO Save best model: monitor(max): 0.350050
2020-07-08 14:17:52,676 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 14:17:52,730 P2785 INFO Train loss: 0.452707
2020-07-08 14:17:52,730 P2785 INFO ************ Epoch=6 end ************
2020-07-08 14:29:24,914 P2785 INFO [Metrics] logloss: 0.449644 - AUC: 0.801287
2020-07-08 14:29:24,916 P2785 INFO Save best model: monitor(max): 0.351643
2020-07-08 14:29:24,988 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 14:29:25,065 P2785 INFO Train loss: 0.451826
2020-07-08 14:29:25,065 P2785 INFO ************ Epoch=7 end ************
2020-07-08 14:40:59,887 P2785 INFO [Metrics] logloss: 0.449199 - AUC: 0.801771
2020-07-08 14:40:59,889 P2785 INFO Save best model: monitor(max): 0.352572
2020-07-08 14:41:00,019 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 14:41:00,069 P2785 INFO Train loss: 0.451166
2020-07-08 14:41:00,070 P2785 INFO ************ Epoch=8 end ************
2020-07-08 14:52:28,757 P2785 INFO [Metrics] logloss: 0.448805 - AUC: 0.802201
2020-07-08 14:52:28,758 P2785 INFO Save best model: monitor(max): 0.353396
2020-07-08 14:52:28,878 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 14:52:28,928 P2785 INFO Train loss: 0.450642
2020-07-08 14:52:28,928 P2785 INFO ************ Epoch=9 end ************
2020-07-08 15:03:58,377 P2785 INFO [Metrics] logloss: 0.448451 - AUC: 0.802598
2020-07-08 15:03:58,379 P2785 INFO Save best model: monitor(max): 0.354147
2020-07-08 15:03:58,471 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 15:03:58,552 P2785 INFO Train loss: 0.450200
2020-07-08 15:03:58,552 P2785 INFO ************ Epoch=10 end ************
2020-07-08 15:15:26,209 P2785 INFO [Metrics] logloss: 0.448276 - AUC: 0.802887
2020-07-08 15:15:26,210 P2785 INFO Save best model: monitor(max): 0.354611
2020-07-08 15:15:26,281 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 15:15:26,331 P2785 INFO Train loss: 0.449806
2020-07-08 15:15:26,331 P2785 INFO ************ Epoch=11 end ************
2020-07-08 15:27:03,358 P2785 INFO [Metrics] logloss: 0.447957 - AUC: 0.803169
2020-07-08 15:27:03,360 P2785 INFO Save best model: monitor(max): 0.355211
2020-07-08 15:27:03,443 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 15:27:03,511 P2785 INFO Train loss: 0.449460
2020-07-08 15:27:03,511 P2785 INFO ************ Epoch=12 end ************
2020-07-08 15:38:35,868 P2785 INFO [Metrics] logloss: 0.447856 - AUC: 0.803400
2020-07-08 15:38:35,869 P2785 INFO Save best model: monitor(max): 0.355544
2020-07-08 15:38:35,941 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 15:38:35,999 P2785 INFO Train loss: 0.449132
2020-07-08 15:38:35,999 P2785 INFO ************ Epoch=13 end ************
2020-07-08 15:50:12,495 P2785 INFO [Metrics] logloss: 0.447494 - AUC: 0.803673
2020-07-08 15:50:12,496 P2785 INFO Save best model: monitor(max): 0.356178
2020-07-08 15:50:12,620 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 15:50:12,686 P2785 INFO Train loss: 0.448819
2020-07-08 15:50:12,686 P2785 INFO ************ Epoch=14 end ************
2020-07-08 16:01:44,264 P2785 INFO [Metrics] logloss: 0.447374 - AUC: 0.803836
2020-07-08 16:01:44,265 P2785 INFO Save best model: monitor(max): 0.356462
2020-07-08 16:01:44,342 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 16:01:44,396 P2785 INFO Train loss: 0.448565
2020-07-08 16:01:44,396 P2785 INFO ************ Epoch=15 end ************
2020-07-08 16:13:12,030 P2785 INFO [Metrics] logloss: 0.447228 - AUC: 0.803972
2020-07-08 16:13:12,031 P2785 INFO Save best model: monitor(max): 0.356743
2020-07-08 16:13:12,101 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 16:13:12,156 P2785 INFO Train loss: 0.448321
2020-07-08 16:13:12,156 P2785 INFO ************ Epoch=16 end ************
2020-07-08 16:24:41,837 P2785 INFO [Metrics] logloss: 0.447130 - AUC: 0.804102
2020-07-08 16:24:41,838 P2785 INFO Save best model: monitor(max): 0.356972
2020-07-08 16:24:41,918 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 16:24:42,016 P2785 INFO Train loss: 0.448105
2020-07-08 16:24:42,017 P2785 INFO ************ Epoch=17 end ************
2020-07-08 16:36:10,285 P2785 INFO [Metrics] logloss: 0.447063 - AUC: 0.804170
2020-07-08 16:36:10,287 P2785 INFO Save best model: monitor(max): 0.357107
2020-07-08 16:36:10,359 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 16:36:10,422 P2785 INFO Train loss: 0.447912
2020-07-08 16:36:10,422 P2785 INFO ************ Epoch=18 end ************
2020-07-08 16:47:44,339 P2785 INFO [Metrics] logloss: 0.447020 - AUC: 0.804254
2020-07-08 16:47:44,341 P2785 INFO Save best model: monitor(max): 0.357234
2020-07-08 16:47:44,414 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 16:47:44,496 P2785 INFO Train loss: 0.447718
2020-07-08 16:47:44,496 P2785 INFO ************ Epoch=19 end ************
2020-07-08 16:59:14,449 P2785 INFO [Metrics] logloss: 0.446895 - AUC: 0.804385
2020-07-08 16:59:14,450 P2785 INFO Save best model: monitor(max): 0.357490
2020-07-08 16:59:14,519 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 16:59:14,589 P2785 INFO Train loss: 0.447561
2020-07-08 16:59:14,589 P2785 INFO ************ Epoch=20 end ************
2020-07-08 17:10:49,149 P2785 INFO [Metrics] logloss: 0.446875 - AUC: 0.804370
2020-07-08 17:10:49,151 P2785 INFO Save best model: monitor(max): 0.357495
2020-07-08 17:10:49,247 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 17:10:49,308 P2785 INFO Train loss: 0.447399
2020-07-08 17:10:49,308 P2785 INFO ************ Epoch=21 end ************
2020-07-08 17:22:27,622 P2785 INFO [Metrics] logloss: 0.446747 - AUC: 0.804495
2020-07-08 17:22:27,623 P2785 INFO Save best model: monitor(max): 0.357749
2020-07-08 17:22:27,691 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 17:22:27,755 P2785 INFO Train loss: 0.447243
2020-07-08 17:22:27,755 P2785 INFO ************ Epoch=22 end ************
2020-07-08 17:34:05,502 P2785 INFO [Metrics] logloss: 0.446680 - AUC: 0.804565
2020-07-08 17:34:05,504 P2785 INFO Save best model: monitor(max): 0.357885
2020-07-08 17:34:05,576 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 17:34:05,665 P2785 INFO Train loss: 0.447102
2020-07-08 17:34:05,665 P2785 INFO ************ Epoch=23 end ************
2020-07-08 17:45:37,045 P2785 INFO [Metrics] logloss: 0.446713 - AUC: 0.804625
2020-07-08 17:45:37,046 P2785 INFO Save best model: monitor(max): 0.357911
2020-07-08 17:45:37,118 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 17:45:37,184 P2785 INFO Train loss: 0.446976
2020-07-08 17:45:37,184 P2785 INFO ************ Epoch=24 end ************
2020-07-08 17:57:03,861 P2785 INFO [Metrics] logloss: 0.446697 - AUC: 0.804648
2020-07-08 17:57:03,862 P2785 INFO Save best model: monitor(max): 0.357950
2020-07-08 17:57:03,940 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 17:57:04,011 P2785 INFO Train loss: 0.446854
2020-07-08 17:57:04,011 P2785 INFO ************ Epoch=25 end ************
2020-07-08 18:08:32,213 P2785 INFO [Metrics] logloss: 0.446715 - AUC: 0.804650
2020-07-08 18:08:32,214 P2785 INFO Monitor(max) STOP: 0.357935 !
2020-07-08 18:08:32,214 P2785 INFO Reduce learning rate on plateau: 0.000100
2020-07-08 18:08:32,214 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 18:08:32,282 P2785 INFO Train loss: 0.446751
2020-07-08 18:08:32,282 P2785 INFO ************ Epoch=26 end ************
2020-07-08 18:20:00,961 P2785 INFO [Metrics] logloss: 0.445848 - AUC: 0.805587
2020-07-08 18:20:00,962 P2785 INFO Save best model: monitor(max): 0.359739
2020-07-08 18:20:01,033 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 18:20:01,090 P2785 INFO Train loss: 0.442689
2020-07-08 18:20:01,090 P2785 INFO ************ Epoch=27 end ************
2020-07-08 18:31:29,381 P2785 INFO [Metrics] logloss: 0.445833 - AUC: 0.805609
2020-07-08 18:31:29,383 P2785 INFO Save best model: monitor(max): 0.359775
2020-07-08 18:31:29,457 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 18:31:29,529 P2785 INFO Train loss: 0.441667
2020-07-08 18:31:29,530 P2785 INFO ************ Epoch=28 end ************
2020-07-08 18:43:02,529 P2785 INFO [Metrics] logloss: 0.445908 - AUC: 0.805580
2020-07-08 18:43:02,531 P2785 INFO Monitor(max) STOP: 0.359672 !
2020-07-08 18:43:02,531 P2785 INFO Reduce learning rate on plateau: 0.000010
2020-07-08 18:43:02,531 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 18:43:02,601 P2785 INFO Train loss: 0.441148
2020-07-08 18:43:02,602 P2785 INFO ************ Epoch=29 end ************
2020-07-08 18:54:33,943 P2785 INFO [Metrics] logloss: 0.445962 - AUC: 0.805519
2020-07-08 18:54:33,944 P2785 INFO Monitor(max) STOP: 0.359557 !
2020-07-08 18:54:33,945 P2785 INFO Reduce learning rate on plateau: 0.000001
2020-07-08 18:54:33,945 P2785 INFO Early stopping at epoch=30
2020-07-08 18:54:33,945 P2785 INFO --- 3668/3668 batches finished ---
2020-07-08 18:54:34,006 P2785 INFO Train loss: 0.440233
2020-07-08 18:54:34,007 P2785 INFO Training finished.
2020-07-08 18:54:34,007 P2785 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/AFM_criteo/min10/criteo_x4_5c863b0f/AFM_criteo_x4_5c863b0f_010_f040edb0_model.ckpt
2020-07-08 18:54:34,302 P2785 INFO ****** Train/validation evaluation ******
2020-07-08 18:58:20,393 P2785 INFO [Metrics] logloss: 0.435982 - AUC: 0.816322
2020-07-08 18:58:48,779 P2785 INFO [Metrics] logloss: 0.445833 - AUC: 0.805609
2020-07-08 18:58:48,859 P2785 INFO ******** Test evaluation ********
2020-07-08 18:58:48,859 P2785 INFO Loading data...
2020-07-08 18:58:48,859 P2785 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-07-08 18:58:49,921 P2785 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-08 18:58:49,921 P2785 INFO Loading test data done.
2020-07-08 18:59:16,339 P2785 INFO [Metrics] logloss: 0.445523 - AUC: 0.805961

```
