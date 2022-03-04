## LR_avazu_x4_002

A hands-on guide to run the LR model on the Avazu_x4_002 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
  GPU: Tesla P100 16G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 10.0
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [Avazu_x4_002](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x4_002). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [LR](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/LR.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [LR_avazu_x4_tuner_config_04](./LR_avazu_x4_tuner_config_04). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd LR_avazu_x4_002
    nohup python run_expid.py --config ./LR_avazu_x4_tuner_config_04 --expid LR_avazu_x4_003_509815a1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.379905 | 0.780441  |


### Logs
```python
2020-02-24 03:46:23,067 P30759 INFO {
    "batch_size": "20000",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "LR",
    "model_id": "LR_avazu_x4_003_ff0c15bc",
    "model_root": "./Avazu/LR_avazu/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "l2(1.e-8)",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "verbose": "1",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "version": "pytorch",
    "gpu": "0"
}
2020-02-24 03:46:23,067 P30759 INFO Set up feature encoder...
2020-02-24 03:46:23,067 P30759 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-02-24 03:46:23,068 P30759 INFO Loading data...
2020-02-24 03:46:23,071 P30759 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-02-24 03:46:26,000 P30759 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-02-24 03:46:27,550 P30759 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-02-24 03:46:27,663 P30759 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-02-24 03:46:27,663 P30759 INFO Loading train data done.
2020-02-24 03:46:30,945 P30759 INFO **** Start training: 1618 batches/epoch ****
2020-02-24 03:51:36,928 P30759 INFO [Metrics] logloss: 0.395695 - AUC: 0.752601
2020-02-24 03:51:36,984 P30759 INFO Save best model: monitor(max): 0.356906
2020-02-24 03:51:37,014 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 03:51:37,098 P30759 INFO Train loss: 0.410460
2020-02-24 03:51:37,098 P30759 INFO ************ Epoch=1 end ************
2020-02-24 03:56:42,476 P30759 INFO [Metrics] logloss: 0.391273 - AUC: 0.760644
2020-02-24 03:56:42,529 P30759 INFO Save best model: monitor(max): 0.369370
2020-02-24 03:56:42,567 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 03:56:42,671 P30759 INFO Train loss: 0.390049
2020-02-24 03:56:42,671 P30759 INFO ************ Epoch=2 end ************
2020-02-24 04:01:46,147 P30759 INFO [Metrics] logloss: 0.388718 - AUC: 0.765397
2020-02-24 04:01:46,262 P30759 INFO Save best model: monitor(max): 0.376678
2020-02-24 04:01:46,298 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 04:01:46,367 P30759 INFO Train loss: 0.383501
2020-02-24 04:01:46,367 P30759 INFO ************ Epoch=3 end ************
2020-02-24 04:06:54,024 P30759 INFO [Metrics] logloss: 0.386827 - AUC: 0.768756
2020-02-24 04:06:54,111 P30759 INFO Save best model: monitor(max): 0.381929
2020-02-24 04:06:54,160 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 04:06:54,246 P30759 INFO Train loss: 0.378352
2020-02-24 04:06:54,246 P30759 INFO ************ Epoch=4 end ************
2020-02-24 04:12:00,707 P30759 INFO [Metrics] logloss: 0.385413 - AUC: 0.771300
2020-02-24 04:12:00,817 P30759 INFO Save best model: monitor(max): 0.385887
2020-02-24 04:12:00,868 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 04:12:00,944 P30759 INFO Train loss: 0.374026
2020-02-24 04:12:00,945 P30759 INFO ************ Epoch=5 end ************
2020-02-24 04:17:03,020 P30759 INFO [Metrics] logloss: 0.384296 - AUC: 0.773238
2020-02-24 04:17:03,089 P30759 INFO Save best model: monitor(max): 0.388942
2020-02-24 04:17:03,130 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 04:17:03,228 P30759 INFO Train loss: 0.370329
2020-02-24 04:17:03,228 P30759 INFO ************ Epoch=6 end ************
2020-02-24 04:22:10,468 P30759 INFO [Metrics] logloss: 0.383409 - AUC: 0.774823
2020-02-24 04:22:10,539 P30759 INFO Save best model: monitor(max): 0.391414
2020-02-24 04:22:10,575 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 04:22:10,672 P30759 INFO Train loss: 0.367102
2020-02-24 04:22:10,673 P30759 INFO ************ Epoch=7 end ************
2020-02-24 04:27:16,884 P30759 INFO [Metrics] logloss: 0.382668 - AUC: 0.776023
2020-02-24 04:27:16,941 P30759 INFO Save best model: monitor(max): 0.393355
2020-02-24 04:27:16,977 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 04:27:17,048 P30759 INFO Train loss: 0.364262
2020-02-24 04:27:17,048 P30759 INFO ************ Epoch=8 end ************
2020-02-24 04:32:19,164 P30759 INFO [Metrics] logloss: 0.382091 - AUC: 0.776938
2020-02-24 04:32:19,271 P30759 INFO Save best model: monitor(max): 0.394847
2020-02-24 04:32:19,320 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 04:32:19,398 P30759 INFO Train loss: 0.361758
2020-02-24 04:32:19,399 P30759 INFO ************ Epoch=9 end ************
2020-02-24 04:37:26,410 P30759 INFO [Metrics] logloss: 0.381623 - AUC: 0.777671
2020-02-24 04:37:26,474 P30759 INFO Save best model: monitor(max): 0.396048
2020-02-24 04:37:26,514 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 04:37:26,583 P30759 INFO Train loss: 0.359540
2020-02-24 04:37:26,583 P30759 INFO ************ Epoch=10 end ************
2020-02-24 04:42:31,693 P30759 INFO [Metrics] logloss: 0.381232 - AUC: 0.778269
2020-02-24 04:42:31,752 P30759 INFO Save best model: monitor(max): 0.397037
2020-02-24 04:42:31,792 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 04:42:31,887 P30759 INFO Train loss: 0.357560
2020-02-24 04:42:31,887 P30759 INFO ************ Epoch=11 end ************
2020-02-24 04:47:39,362 P30759 INFO [Metrics] logloss: 0.380905 - AUC: 0.778747
2020-02-24 04:47:39,435 P30759 INFO Save best model: monitor(max): 0.397842
2020-02-24 04:47:39,475 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 04:47:39,550 P30759 INFO Train loss: 0.355787
2020-02-24 04:47:39,550 P30759 INFO ************ Epoch=12 end ************
2020-02-24 04:52:43,329 P30759 INFO [Metrics] logloss: 0.380677 - AUC: 0.779111
2020-02-24 04:52:43,387 P30759 INFO Save best model: monitor(max): 0.398434
2020-02-24 04:52:43,423 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 04:52:43,522 P30759 INFO Train loss: 0.354193
2020-02-24 04:52:43,523 P30759 INFO ************ Epoch=13 end ************
2020-02-24 04:57:48,548 P30759 INFO [Metrics] logloss: 0.380484 - AUC: 0.779375
2020-02-24 04:57:48,605 P30759 INFO Save best model: monitor(max): 0.398891
2020-02-24 04:57:48,642 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 04:57:48,709 P30759 INFO Train loss: 0.352760
2020-02-24 04:57:48,709 P30759 INFO ************ Epoch=14 end ************
2020-02-24 05:02:54,165 P30759 INFO [Metrics] logloss: 0.380331 - AUC: 0.779640
2020-02-24 05:02:54,231 P30759 INFO Save best model: monitor(max): 0.399309
2020-02-24 05:02:54,268 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 05:02:54,341 P30759 INFO Train loss: 0.351469
2020-02-24 05:02:54,341 P30759 INFO ************ Epoch=15 end ************
2020-02-24 05:07:57,712 P30759 INFO [Metrics] logloss: 0.380240 - AUC: 0.779783
2020-02-24 05:07:57,774 P30759 INFO Save best model: monitor(max): 0.399543
2020-02-24 05:07:57,815 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 05:07:57,918 P30759 INFO Train loss: 0.350293
2020-02-24 05:07:57,918 P30759 INFO ************ Epoch=16 end ************
2020-02-24 05:13:06,310 P30759 INFO [Metrics] logloss: 0.380108 - AUC: 0.779969
2020-02-24 05:13:06,367 P30759 INFO Save best model: monitor(max): 0.399861
2020-02-24 05:13:06,406 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 05:13:06,507 P30759 INFO Train loss: 0.349223
2020-02-24 05:13:06,507 P30759 INFO ************ Epoch=17 end ************
2020-02-24 05:18:15,855 P30759 INFO [Metrics] logloss: 0.380031 - AUC: 0.780088
2020-02-24 05:18:15,938 P30759 INFO Save best model: monitor(max): 0.400057
2020-02-24 05:18:15,988 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 05:18:16,057 P30759 INFO Train loss: 0.348252
2020-02-24 05:18:16,057 P30759 INFO ************ Epoch=18 end ************
2020-02-24 05:23:22,876 P30759 INFO [Metrics] logloss: 0.379990 - AUC: 0.780199
2020-02-24 05:23:22,962 P30759 INFO Save best model: monitor(max): 0.400209
2020-02-24 05:23:23,014 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 05:23:23,084 P30759 INFO Train loss: 0.347361
2020-02-24 05:23:23,084 P30759 INFO ************ Epoch=19 end ************
2020-02-24 05:28:32,782 P30759 INFO [Metrics] logloss: 0.379954 - AUC: 0.780254
2020-02-24 05:28:32,845 P30759 INFO Save best model: monitor(max): 0.400301
2020-02-24 05:28:32,885 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 05:28:32,954 P30759 INFO Train loss: 0.346547
2020-02-24 05:28:32,954 P30759 INFO ************ Epoch=20 end ************
2020-02-24 05:33:40,185 P30759 INFO [Metrics] logloss: 0.379940 - AUC: 0.780335
2020-02-24 05:33:40,242 P30759 INFO Save best model: monitor(max): 0.400395
2020-02-24 05:33:40,278 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 05:33:40,346 P30759 INFO Train loss: 0.345800
2020-02-24 05:33:40,347 P30759 INFO ************ Epoch=21 end ************
2020-02-24 05:38:45,238 P30759 INFO [Metrics] logloss: 0.380024 - AUC: 0.780319
2020-02-24 05:38:45,296 P30759 INFO Monitor(max) STOP: 0.400295 !
2020-02-24 05:38:45,297 P30759 INFO Reduce learning rate on plateau: 0.000100
2020-02-24 05:38:45,297 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 05:38:45,399 P30759 INFO Train loss: 0.345113
2020-02-24 05:38:45,399 P30759 INFO ************ Epoch=22 end ************
2020-02-24 05:43:50,474 P30759 INFO [Metrics] logloss: 0.379915 - AUC: 0.780399
2020-02-24 05:43:50,560 P30759 INFO Save best model: monitor(max): 0.400484
2020-02-24 05:43:50,609 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 05:43:50,692 P30759 INFO Train loss: 0.343254
2020-02-24 05:43:50,693 P30759 INFO ************ Epoch=23 end ************
2020-02-24 05:48:53,237 P30759 INFO [Metrics] logloss: 0.379910 - AUC: 0.780404
2020-02-24 05:48:53,324 P30759 INFO Save best model: monitor(max): 0.400494
2020-02-24 05:48:53,373 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 05:48:53,459 P30759 INFO Train loss: 0.343184
2020-02-24 05:48:53,460 P30759 INFO ************ Epoch=24 end ************
2020-02-24 05:53:55,190 P30759 INFO [Metrics] logloss: 0.379902 - AUC: 0.780406
2020-02-24 05:53:55,248 P30759 INFO Save best model: monitor(max): 0.400503
2020-02-24 05:53:55,283 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 05:53:55,350 P30759 INFO Train loss: 0.343118
2020-02-24 05:53:55,350 P30759 INFO ************ Epoch=25 end ************
2020-02-24 05:59:01,150 P30759 INFO [Metrics] logloss: 0.379895 - AUC: 0.780423
2020-02-24 05:59:01,208 P30759 INFO Save best model: monitor(max): 0.400528
2020-02-24 05:59:01,244 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 05:59:01,306 P30759 INFO Train loss: 0.343050
2020-02-24 05:59:01,306 P30759 INFO ************ Epoch=26 end ************
2020-02-24 06:04:08,246 P30759 INFO [Metrics] logloss: 0.379900 - AUC: 0.780420
2020-02-24 06:04:08,303 P30759 INFO Monitor(max) STOP: 0.400520 !
2020-02-24 06:04:08,303 P30759 INFO Reduce learning rate on plateau: 0.000010
2020-02-24 06:04:08,303 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 06:04:08,371 P30759 INFO Train loss: 0.342997
2020-02-24 06:04:08,371 P30759 INFO ************ Epoch=27 end ************
2020-02-24 06:09:17,473 P30759 INFO [Metrics] logloss: 0.379902 - AUC: 0.780423
2020-02-24 06:09:17,529 P30759 INFO Monitor(max) STOP: 0.400521 !
2020-02-24 06:09:17,529 P30759 INFO Reduce learning rate on plateau: 0.000001
2020-02-24 06:09:17,529 P30759 INFO Early stopping at epoch=28
2020-02-24 06:09:17,529 P30759 INFO --- 1618/1618 batches finished ---
2020-02-24 06:09:17,604 P30759 INFO Train loss: 0.342816
2020-02-24 06:09:17,604 P30759 INFO Training finished.
2020-02-24 06:09:17,604 P30759 INFO Load best model: /home/XXX/benchmarks/Avazu/LR_avazu/avazu_x4_001_d45ad60e/LR_avazu_x4_003_ff0c15bc_avazu_x4_001_d45ad60e_model.ckpt
2020-02-24 06:09:17,641 P30759 INFO ****** Train/validation evaluation ******
2020-02-24 06:14:29,185 P30759 INFO [Metrics] logloss: 0.333669 - AUC: 0.853672
2020-02-24 06:15:06,748 P30759 INFO [Metrics] logloss: 0.379895 - AUC: 0.780423
2020-02-24 06:15:06,897 P30759 INFO ******** Test evaluation ********
2020-02-24 06:15:06,897 P30759 INFO Loading data...
2020-02-24 06:15:06,898 P30759 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-02-24 06:15:07,598 P30759 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-02-24 06:15:07,598 P30759 INFO Loading test data done.
2020-02-24 06:15:45,469 P30759 INFO [Metrics] logloss: 0.379905 - AUC: 0.780441

```
