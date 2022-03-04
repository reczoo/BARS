## CCPM_criteo_x4_001

A hands-on guide to run the CCPM model on the Criteo_x4_001 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

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
Dataset ID: [Criteo_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [CCPM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/CCPM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CCPM_criteo_x4_tuner_config_01](./CCPM_criteo_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CCPM_criteo_x4_001
    nohup python run_expid.py --config ./CCPM_criteo_x4_tuner_config_01 --expid CCPM_criteo_x4_008_9b98f944 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.441482 | 0.810411  |


### Logs
```python
2020-07-13 04:57:16,823 P2447 INFO {
    "activation": "Tanh",
    "batch_size": "10000",
    "channels": "[64, 128, 256]",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "kernel_heights": "[7, 5, 3]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "CCPM",
    "model_id": "CCPM_criteo_x4_5c863b0f_008_4e4b673c",
    "model_root": "./Criteo/CCPM_criteo/min10/",
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
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-07-13 04:57:16,824 P2447 INFO Set up feature encoder...
2020-07-13 04:57:16,824 P2447 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-07-13 04:57:16,824 P2447 INFO Loading data...
2020-07-13 04:57:16,826 P2447 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-07-13 04:57:22,378 P2447 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-07-13 04:57:25,967 P2447 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-07-13 04:57:26,095 P2447 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-13 04:57:26,095 P2447 INFO Loading train data done.
2020-07-13 04:58:35,923 P2447 INFO Start training: 3668 batches/epoch
2020-07-13 04:58:35,923 P2447 INFO ************ Epoch=1 start ************
2020-07-13 06:59:00,484 P2447 INFO [Metrics] logloss: 0.449744 - AUC: 0.801113
2020-07-13 06:59:00,495 P2447 INFO Save best model: monitor(max): 0.351370
2020-07-13 06:59:00,562 P2447 INFO --- 3668/3668 batches finished ---
2020-07-13 06:59:00,660 P2447 INFO Train loss: 0.465242
2020-07-13 06:59:00,660 P2447 INFO ************ Epoch=1 end ************
2020-07-13 08:56:59,689 P2447 INFO [Metrics] logloss: 0.448452 - AUC: 0.803245
2020-07-13 08:56:59,690 P2447 INFO Save best model: monitor(max): 0.354793
2020-07-13 08:56:59,777 P2447 INFO --- 3668/3668 batches finished ---
2020-07-13 08:56:59,860 P2447 INFO Train loss: 0.458433
2020-07-13 08:56:59,860 P2447 INFO ************ Epoch=2 end ************
2020-07-13 10:52:53,976 P2447 INFO [Metrics] logloss: 0.447318 - AUC: 0.804350
2020-07-13 10:52:53,977 P2447 INFO Save best model: monitor(max): 0.357033
2020-07-13 10:52:54,059 P2447 INFO --- 3668/3668 batches finished ---
2020-07-13 10:52:54,131 P2447 INFO Train loss: 0.457179
2020-07-13 10:52:54,131 P2447 INFO ************ Epoch=3 end ************
2020-07-13 12:47:41,698 P2447 INFO [Metrics] logloss: 0.446216 - AUC: 0.805005
2020-07-13 12:47:41,699 P2447 INFO Save best model: monitor(max): 0.358789
2020-07-13 12:47:41,780 P2447 INFO --- 3668/3668 batches finished ---
2020-07-13 12:47:41,882 P2447 INFO Train loss: 0.456535
2020-07-13 12:47:41,882 P2447 INFO ************ Epoch=4 end ************
2020-07-13 14:42:22,156 P2447 INFO [Metrics] logloss: 0.445934 - AUC: 0.805397
2020-07-13 14:42:22,158 P2447 INFO Save best model: monitor(max): 0.359462
2020-07-13 14:42:22,237 P2447 INFO --- 3668/3668 batches finished ---
2020-07-13 14:42:22,320 P2447 INFO Train loss: 0.456116
2020-07-13 14:42:22,320 P2447 INFO ************ Epoch=5 end ************
2020-07-13 16:36:57,352 P2447 INFO [Metrics] logloss: 0.445908 - AUC: 0.805542
2020-07-13 16:36:57,353 P2447 INFO Save best model: monitor(max): 0.359633
2020-07-13 16:36:57,445 P2447 INFO --- 3668/3668 batches finished ---
2020-07-13 16:36:57,543 P2447 INFO Train loss: 0.455795
2020-07-13 16:36:57,543 P2447 INFO ************ Epoch=6 end ************
2020-07-13 18:31:27,388 P2447 INFO [Metrics] logloss: 0.446010 - AUC: 0.805645
2020-07-13 18:31:27,390 P2447 INFO Save best model: monitor(max): 0.359635
2020-07-13 18:31:27,472 P2447 INFO --- 3668/3668 batches finished ---
2020-07-13 18:31:27,564 P2447 INFO Train loss: 0.455551
2020-07-13 18:31:27,564 P2447 INFO ************ Epoch=7 end ************
2020-07-13 20:25:55,215 P2447 INFO [Metrics] logloss: 0.445635 - AUC: 0.805711
2020-07-13 20:25:55,216 P2447 INFO Save best model: monitor(max): 0.360076
2020-07-13 20:25:55,300 P2447 INFO --- 3668/3668 batches finished ---
2020-07-13 20:25:55,383 P2447 INFO Train loss: 0.455370
2020-07-13 20:25:55,383 P2447 INFO ************ Epoch=8 end ************
2020-07-13 22:20:25,169 P2447 INFO [Metrics] logloss: 0.445289 - AUC: 0.806037
2020-07-13 22:20:25,171 P2447 INFO Save best model: monitor(max): 0.360748
2020-07-13 22:20:25,252 P2447 INFO --- 3668/3668 batches finished ---
2020-07-13 22:20:25,348 P2447 INFO Train loss: 0.455205
2020-07-13 22:20:25,349 P2447 INFO ************ Epoch=9 end ************
2020-07-14 00:14:53,419 P2447 INFO [Metrics] logloss: 0.445252 - AUC: 0.806060
2020-07-14 00:14:53,420 P2447 INFO Save best model: monitor(max): 0.360808
2020-07-14 00:14:53,503 P2447 INFO --- 3668/3668 batches finished ---
2020-07-14 00:14:53,590 P2447 INFO Train loss: 0.454994
2020-07-14 00:14:53,590 P2447 INFO ************ Epoch=10 end ************
2020-07-14 02:09:18,359 P2447 INFO [Metrics] logloss: 0.445207 - AUC: 0.806199
2020-07-14 02:09:18,360 P2447 INFO Save best model: monitor(max): 0.360991
2020-07-14 02:09:18,442 P2447 INFO --- 3668/3668 batches finished ---
2020-07-14 02:09:18,523 P2447 INFO Train loss: 0.454970
2020-07-14 02:09:18,523 P2447 INFO ************ Epoch=11 end ************
2020-07-14 04:03:45,378 P2447 INFO [Metrics] logloss: 0.445140 - AUC: 0.806513
2020-07-14 04:03:45,380 P2447 INFO Save best model: monitor(max): 0.361372
2020-07-14 04:03:45,464 P2447 INFO --- 3668/3668 batches finished ---
2020-07-14 04:03:45,571 P2447 INFO Train loss: 0.454864
2020-07-14 04:03:45,571 P2447 INFO ************ Epoch=12 end ************
2020-07-14 05:58:12,957 P2447 INFO [Metrics] logloss: 0.444984 - AUC: 0.806416
2020-07-14 05:58:12,959 P2447 INFO Save best model: monitor(max): 0.361432
2020-07-14 05:58:13,055 P2447 INFO --- 3668/3668 batches finished ---
2020-07-14 05:58:13,143 P2447 INFO Train loss: 0.454792
2020-07-14 05:58:13,143 P2447 INFO ************ Epoch=13 end ************
2020-07-14 07:52:36,521 P2447 INFO [Metrics] logloss: 0.445010 - AUC: 0.806298
2020-07-14 07:52:36,522 P2447 INFO Monitor(max) STOP: 0.361288 !
2020-07-14 07:52:36,523 P2447 INFO Reduce learning rate on plateau: 0.000100
2020-07-14 07:52:36,523 P2447 INFO --- 3668/3668 batches finished ---
2020-07-14 07:52:36,614 P2447 INFO Train loss: 0.454668
2020-07-14 07:52:36,614 P2447 INFO ************ Epoch=14 end ************
2020-07-14 09:46:58,411 P2447 INFO [Metrics] logloss: 0.442114 - AUC: 0.809586
2020-07-14 09:46:58,412 P2447 INFO Save best model: monitor(max): 0.367472
2020-07-14 09:46:58,493 P2447 INFO --- 3668/3668 batches finished ---
2020-07-14 09:46:58,577 P2447 INFO Train loss: 0.444674
2020-07-14 09:46:58,578 P2447 INFO ************ Epoch=15 end ************
2020-07-14 11:41:28,459 P2447 INFO [Metrics] logloss: 0.441811 - AUC: 0.810005
2020-07-14 11:41:28,461 P2447 INFO Save best model: monitor(max): 0.368194
2020-07-14 11:41:28,578 P2447 INFO --- 3668/3668 batches finished ---
2020-07-14 11:41:28,722 P2447 INFO Train loss: 0.440599
2020-07-14 11:41:28,722 P2447 INFO ************ Epoch=16 end ************
2020-07-14 13:36:02,402 P2447 INFO [Metrics] logloss: 0.442075 - AUC: 0.809803
2020-07-14 13:36:02,403 P2447 INFO Monitor(max) STOP: 0.367728 !
2020-07-14 13:36:02,403 P2447 INFO Reduce learning rate on plateau: 0.000010
2020-07-14 13:36:02,403 P2447 INFO --- 3668/3668 batches finished ---
2020-07-14 13:36:02,488 P2447 INFO Train loss: 0.438481
2020-07-14 13:36:02,488 P2447 INFO ************ Epoch=17 end ************
2020-07-14 15:30:34,563 P2447 INFO [Metrics] logloss: 0.442757 - AUC: 0.809356
2020-07-14 15:30:34,565 P2447 INFO Monitor(max) STOP: 0.366600 !
2020-07-14 15:30:34,565 P2447 INFO Reduce learning rate on plateau: 0.000001
2020-07-14 15:30:34,565 P2447 INFO Early stopping at epoch=18
2020-07-14 15:30:34,565 P2447 INFO --- 3668/3668 batches finished ---
2020-07-14 15:30:34,703 P2447 INFO Train loss: 0.434043
2020-07-14 15:30:34,703 P2447 INFO Training finished.
2020-07-14 15:30:34,703 P2447 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/CCPM_criteo/min10/criteo_x4_5c863b0f/CCPM_criteo_x4_5c863b0f_008_4e4b673c_model.ckpt
2020-07-14 15:30:34,843 P2447 INFO ****** Train/validation evaluation ******
2020-07-14 16:50:21,879 P2447 INFO [Metrics] logloss: 0.430022 - AUC: 0.822622
2020-07-14 17:00:19,044 P2447 INFO [Metrics] logloss: 0.441811 - AUC: 0.810005
2020-07-14 17:00:19,150 P2447 INFO ******** Test evaluation ********
2020-07-14 17:00:19,150 P2447 INFO Loading data...
2020-07-14 17:00:19,150 P2447 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-07-14 17:00:20,023 P2447 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-14 17:00:20,023 P2447 INFO Loading test data done.
2020-07-14 17:10:17,254 P2447 INFO [Metrics] logloss: 0.441482 - AUC: 0.810411

```
