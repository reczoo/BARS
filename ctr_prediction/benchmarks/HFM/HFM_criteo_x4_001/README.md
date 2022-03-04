## HFM_criteo_x4_001

A hands-on guide to run the HFM model on the Criteo_x4_001 dataset.

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
Dataset ID: [Criteo_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [HFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/HFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HFM_criteo_x4_tuner_config_01](./HFM_criteo_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HFM_criteo_x4_001
    nohup python run_expid.py --config ./HFM_criteo_x4_tuner_config_01 --expid HFM_criteo_x4_008_c1d1ba8a --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.442403 | 0.809473  |


### Logs
```python
2020-07-25 14:27:12,547 P47693 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "l2(1.e-5)",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[64, 64, 64]",
    "interaction_type": "circular_correlation",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "HFM",
    "model_id": "HFM_criteo_x4_5c863b0f_008_1187de69",
    "model_root": "./Criteo/HFM_criteo/min10/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
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
    "use_dnn": "False",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-25 14:27:12,548 P47693 INFO Set up feature encoder...
2020-07-25 14:27:12,548 P47693 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-07-25 14:27:12,549 P47693 INFO Loading data...
2020-07-25 14:27:12,550 P47693 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-07-25 14:27:18,253 P47693 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-07-25 14:27:20,029 P47693 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-07-25 14:27:20,164 P47693 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-25 14:27:20,164 P47693 INFO Loading train data done.
2020-07-25 14:27:23,234 P47693 INFO **** Start training: 3668 batches/epoch ****
2020-07-25 15:01:41,017 P47693 INFO [Metrics] logloss: 0.450886 - AUC: 0.799920
2020-07-25 15:01:41,020 P47693 INFO Save best model: monitor(max): 0.349034
2020-07-25 15:01:41,079 P47693 INFO --- 3668/3668 batches finished ---
2020-07-25 15:01:41,119 P47693 INFO Train loss: 0.464370
2020-07-25 15:01:41,120 P47693 INFO ************ Epoch=1 end ************
2020-07-25 15:35:56,757 P47693 INFO [Metrics] logloss: 0.449606 - AUC: 0.801240
2020-07-25 15:35:56,758 P47693 INFO Save best model: monitor(max): 0.351633
2020-07-25 15:35:56,871 P47693 INFO --- 3668/3668 batches finished ---
2020-07-25 15:35:56,912 P47693 INFO Train loss: 0.459891
2020-07-25 15:35:56,912 P47693 INFO ************ Epoch=2 end ************
2020-07-25 16:10:13,449 P47693 INFO [Metrics] logloss: 0.448806 - AUC: 0.802114
2020-07-25 16:10:13,450 P47693 INFO Save best model: monitor(max): 0.353308
2020-07-25 16:10:13,557 P47693 INFO --- 3668/3668 batches finished ---
2020-07-25 16:10:13,601 P47693 INFO Train loss: 0.459269
2020-07-25 16:10:13,601 P47693 INFO ************ Epoch=3 end ************
2020-07-25 16:44:27,431 P47693 INFO [Metrics] logloss: 0.448584 - AUC: 0.802378
2020-07-25 16:44:27,433 P47693 INFO Save best model: monitor(max): 0.353793
2020-07-25 16:44:27,544 P47693 INFO --- 3668/3668 batches finished ---
2020-07-25 16:44:27,588 P47693 INFO Train loss: 0.458973
2020-07-25 16:44:27,588 P47693 INFO ************ Epoch=4 end ************
2020-07-25 17:18:40,478 P47693 INFO [Metrics] logloss: 0.448391 - AUC: 0.802585
2020-07-25 17:18:40,479 P47693 INFO Save best model: monitor(max): 0.354194
2020-07-25 17:18:40,579 P47693 INFO --- 3668/3668 batches finished ---
2020-07-25 17:18:40,624 P47693 INFO Train loss: 0.458794
2020-07-25 17:18:40,624 P47693 INFO ************ Epoch=5 end ************
2020-07-25 17:52:54,686 P47693 INFO [Metrics] logloss: 0.448490 - AUC: 0.802573
2020-07-25 17:52:54,687 P47693 INFO Monitor(max) STOP: 0.354082 !
2020-07-25 17:52:54,687 P47693 INFO Reduce learning rate on plateau: 0.000100
2020-07-25 17:52:54,688 P47693 INFO --- 3668/3668 batches finished ---
2020-07-25 17:52:54,733 P47693 INFO Train loss: 0.458683
2020-07-25 17:52:54,733 P47693 INFO ************ Epoch=6 end ************
2020-07-25 18:27:08,731 P47693 INFO [Metrics] logloss: 0.444576 - AUC: 0.806848
2020-07-25 18:27:08,732 P47693 INFO Save best model: monitor(max): 0.362272
2020-07-25 18:27:08,812 P47693 INFO --- 3668/3668 batches finished ---
2020-07-25 18:27:08,859 P47693 INFO Train loss: 0.449447
2020-07-25 18:27:08,859 P47693 INFO ************ Epoch=7 end ************
2020-07-25 19:01:21,928 P47693 INFO [Metrics] logloss: 0.443899 - AUC: 0.807627
2020-07-25 19:01:21,929 P47693 INFO Save best model: monitor(max): 0.363728
2020-07-25 19:01:22,031 P47693 INFO --- 3668/3668 batches finished ---
2020-07-25 19:01:22,076 P47693 INFO Train loss: 0.445889
2020-07-25 19:01:22,076 P47693 INFO ************ Epoch=8 end ************
2020-07-25 19:35:36,572 P47693 INFO [Metrics] logloss: 0.443516 - AUC: 0.808049
2020-07-25 19:35:36,573 P47693 INFO Save best model: monitor(max): 0.364533
2020-07-25 19:35:36,662 P47693 INFO --- 3668/3668 batches finished ---
2020-07-25 19:35:36,706 P47693 INFO Train loss: 0.444573
2020-07-25 19:35:36,706 P47693 INFO ************ Epoch=9 end ************
2020-07-25 20:09:55,231 P47693 INFO [Metrics] logloss: 0.443307 - AUC: 0.808293
2020-07-25 20:09:55,232 P47693 INFO Save best model: monitor(max): 0.364986
2020-07-25 20:09:55,319 P47693 INFO --- 3668/3668 batches finished ---
2020-07-25 20:09:55,365 P47693 INFO Train loss: 0.443645
2020-07-25 20:09:55,365 P47693 INFO ************ Epoch=10 end ************
2020-07-25 20:44:11,820 P47693 INFO [Metrics] logloss: 0.443155 - AUC: 0.808472
2020-07-25 20:44:11,821 P47693 INFO Save best model: monitor(max): 0.365317
2020-07-25 20:44:11,909 P47693 INFO --- 3668/3668 batches finished ---
2020-07-25 20:44:11,953 P47693 INFO Train loss: 0.442894
2020-07-25 20:44:11,953 P47693 INFO ************ Epoch=11 end ************
2020-07-25 21:18:28,664 P47693 INFO [Metrics] logloss: 0.443083 - AUC: 0.808570
2020-07-25 21:18:28,665 P47693 INFO Save best model: monitor(max): 0.365487
2020-07-25 21:18:28,761 P47693 INFO --- 3668/3668 batches finished ---
2020-07-25 21:18:28,805 P47693 INFO Train loss: 0.442250
2020-07-25 21:18:28,806 P47693 INFO ************ Epoch=12 end ************
2020-07-25 21:52:41,207 P47693 INFO [Metrics] logloss: 0.443054 - AUC: 0.808615
2020-07-25 21:52:41,209 P47693 INFO Save best model: monitor(max): 0.365561
2020-07-25 21:52:41,290 P47693 INFO --- 3668/3668 batches finished ---
2020-07-25 21:52:41,334 P47693 INFO Train loss: 0.441695
2020-07-25 21:52:41,334 P47693 INFO ************ Epoch=13 end ************
2020-07-25 22:26:55,662 P47693 INFO [Metrics] logloss: 0.443054 - AUC: 0.808657
2020-07-25 22:26:55,663 P47693 INFO Save best model: monitor(max): 0.365603
2020-07-25 22:26:55,732 P47693 INFO --- 3668/3668 batches finished ---
2020-07-25 22:26:55,778 P47693 INFO Train loss: 0.441201
2020-07-25 22:26:55,778 P47693 INFO ************ Epoch=14 end ************
2020-07-25 23:01:06,529 P47693 INFO [Metrics] logloss: 0.443011 - AUC: 0.808724
2020-07-25 23:01:06,531 P47693 INFO Save best model: monitor(max): 0.365713
2020-07-25 23:01:06,622 P47693 INFO --- 3668/3668 batches finished ---
2020-07-25 23:01:06,667 P47693 INFO Train loss: 0.440759
2020-07-25 23:01:06,667 P47693 INFO ************ Epoch=15 end ************
2020-07-25 23:35:26,465 P47693 INFO [Metrics] logloss: 0.443020 - AUC: 0.808710
2020-07-25 23:35:26,467 P47693 INFO Monitor(max) STOP: 0.365689 !
2020-07-25 23:35:26,467 P47693 INFO Reduce learning rate on plateau: 0.000010
2020-07-25 23:35:26,467 P47693 INFO --- 3668/3668 batches finished ---
2020-07-25 23:35:26,511 P47693 INFO Train loss: 0.440357
2020-07-25 23:35:26,511 P47693 INFO ************ Epoch=16 end ************
2020-07-26 00:09:36,377 P47693 INFO [Metrics] logloss: 0.442696 - AUC: 0.809075
2020-07-26 00:09:36,378 P47693 INFO Save best model: monitor(max): 0.366379
2020-07-26 00:09:36,468 P47693 INFO --- 3668/3668 batches finished ---
2020-07-26 00:09:36,514 P47693 INFO Train loss: 0.436629
2020-07-26 00:09:36,515 P47693 INFO ************ Epoch=17 end ************
2020-07-26 00:43:50,290 P47693 INFO [Metrics] logloss: 0.442686 - AUC: 0.809113
2020-07-26 00:43:50,291 P47693 INFO Save best model: monitor(max): 0.366426
2020-07-26 00:43:50,386 P47693 INFO --- 3668/3668 batches finished ---
2020-07-26 00:43:50,433 P47693 INFO Train loss: 0.436320
2020-07-26 00:43:50,433 P47693 INFO ************ Epoch=18 end ************
2020-07-26 01:18:02,242 P47693 INFO [Metrics] logloss: 0.442701 - AUC: 0.809111
2020-07-26 01:18:02,243 P47693 INFO Monitor(max) STOP: 0.366410 !
2020-07-26 01:18:02,243 P47693 INFO Reduce learning rate on plateau: 0.000001
2020-07-26 01:18:02,243 P47693 INFO --- 3668/3668 batches finished ---
2020-07-26 01:18:02,289 P47693 INFO Train loss: 0.436135
2020-07-26 01:18:02,289 P47693 INFO ************ Epoch=19 end ************
2020-07-26 01:52:11,548 P47693 INFO [Metrics] logloss: 0.442701 - AUC: 0.809115
2020-07-26 01:52:11,550 P47693 INFO Monitor(max) STOP: 0.366415 !
2020-07-26 01:52:11,550 P47693 INFO Reduce learning rate on plateau: 0.000001
2020-07-26 01:52:11,550 P47693 INFO Early stopping at epoch=20
2020-07-26 01:52:11,550 P47693 INFO --- 3668/3668 batches finished ---
2020-07-26 01:52:11,595 P47693 INFO Train loss: 0.435572
2020-07-26 01:52:11,595 P47693 INFO Training finished.
2020-07-26 01:52:11,595 P47693 INFO Load best model: /home/XXX/benchmarks/Criteo/HFM_criteo/min10/criteo_x4_5c863b0f/HFM_criteo_x4_5c863b0f_008_1187de69_model.ckpt
2020-07-26 01:52:11,681 P47693 INFO ****** Train/validation evaluation ******
2020-07-26 01:53:26,430 P47693 INFO [Metrics] logloss: 0.442686 - AUC: 0.809113
2020-07-26 01:53:26,517 P47693 INFO ******** Test evaluation ********
2020-07-26 01:53:26,517 P47693 INFO Loading data...
2020-07-26 01:53:26,517 P47693 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-07-26 01:53:27,255 P47693 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-26 01:53:27,255 P47693 INFO Loading test data done.
2020-07-26 01:54:42,431 P47693 INFO [Metrics] logloss: 0.442403 - AUC: 0.809473

```
