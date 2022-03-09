## LorentzFM_avazu_x4_002

A hands-on guide to run the LorentzFM model on the Avazu_x4_002 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [LorentzFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/LorentzFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [LorentzFM_avazu_x4_tuner_config_01](./LorentzFM_avazu_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd LorentzFM_avazu_x4_002
    nohup python run_expid.py --config ./LorentzFM_avazu_x4_tuner_config_01 --expid LorentzFM_avazu_x4_001_c77e5dd3 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.374173 | 0.791223  |


### Logs
```python
2020-02-06 07:07:07,080 P4561 INFO {
    "batch_size": "10000",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "LorentzFM",
    "model_id": "LorentzFM_avazu_x4_001_1246f5a7",
    "model_root": "./Avazu/LorentzFM_avazu/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "l2(1.e-6)",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "version": "pytorch",
    "gpu": "1"
}
2020-02-06 07:07:07,081 P4561 INFO Set up feature encoder...
2020-02-06 07:07:07,081 P4561 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-02-06 07:07:07,082 P4561 INFO Loading data...
2020-02-06 07:07:07,083 P4561 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-02-06 07:07:10,141 P4561 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-02-06 07:07:12,861 P4561 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-02-06 07:07:12,983 P4561 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-02-06 07:07:12,983 P4561 INFO Loading train data done.
2020-02-06 07:07:46,216 P4561 INFO **** Start training: 3235 batches/epoch ****
2020-02-06 07:54:01,853 P4561 INFO [Metrics] logloss: 0.385560 - AUC: 0.770192
2020-02-06 07:54:01,951 P4561 INFO Save best model: monitor(max): 0.384631
2020-02-06 07:54:15,563 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 07:54:15,610 P4561 INFO Train loss: 0.400754
2020-02-06 07:54:15,610 P4561 INFO ************ Epoch=1 end ************
2020-02-06 08:40:35,248 P4561 INFO [Metrics] logloss: 0.381175 - AUC: 0.777955
2020-02-06 08:40:35,351 P4561 INFO Save best model: monitor(max): 0.396780
2020-02-06 08:40:47,979 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 08:40:48,028 P4561 INFO Train loss: 0.395365
2020-02-06 08:40:48,028 P4561 INFO ************ Epoch=2 end ************
2020-02-06 09:27:03,206 P4561 INFO [Metrics] logloss: 0.378590 - AUC: 0.782487
2020-02-06 09:27:03,320 P4561 INFO Save best model: monitor(max): 0.403897
2020-02-06 09:27:16,672 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 09:27:16,722 P4561 INFO Train loss: 0.391286
2020-02-06 09:27:16,723 P4561 INFO ************ Epoch=3 end ************
2020-02-06 10:13:33,059 P4561 INFO [Metrics] logloss: 0.377673 - AUC: 0.784496
2020-02-06 10:13:33,140 P4561 INFO Save best model: monitor(max): 0.406823
2020-02-06 10:13:43,929 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 10:13:43,977 P4561 INFO Train loss: 0.386889
2020-02-06 10:13:43,978 P4561 INFO ************ Epoch=4 end ************
2020-02-06 10:59:57,892 P4561 INFO [Metrics] logloss: 0.377101 - AUC: 0.785668
2020-02-06 10:59:57,999 P4561 INFO Save best model: monitor(max): 0.408567
2020-02-06 11:00:19,303 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 11:00:19,353 P4561 INFO Train loss: 0.382872
2020-02-06 11:00:19,353 P4561 INFO ************ Epoch=5 end ************
2020-02-06 11:46:41,766 P4561 INFO [Metrics] logloss: 0.377499 - AUC: 0.785861
2020-02-06 11:46:41,855 P4561 INFO Monitor(max) STOP: 0.408362 !
2020-02-06 11:46:41,856 P4561 INFO Reduce learning rate on plateau: 0.000100
2020-02-06 11:46:41,856 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 11:46:41,928 P4561 INFO Train loss: 0.379510
2020-02-06 11:46:41,928 P4561 INFO ************ Epoch=6 end ************
2020-02-06 12:32:58,989 P4561 INFO [Metrics] logloss: 0.374553 - AUC: 0.790108
2020-02-06 12:32:59,120 P4561 INFO Save best model: monitor(max): 0.415554
2020-02-06 12:33:18,606 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 12:33:18,656 P4561 INFO Train loss: 0.349914
2020-02-06 12:33:18,657 P4561 INFO ************ Epoch=7 end ************
2020-02-06 13:19:42,336 P4561 INFO [Metrics] logloss: 0.374328 - AUC: 0.790591
2020-02-06 13:19:42,420 P4561 INFO Save best model: monitor(max): 0.416263
2020-02-06 13:20:00,194 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 13:20:00,248 P4561 INFO Train loss: 0.345496
2020-02-06 13:20:00,248 P4561 INFO ************ Epoch=8 end ************
2020-02-06 14:06:15,599 P4561 INFO [Metrics] logloss: 0.374264 - AUC: 0.790730
2020-02-06 14:06:15,735 P4561 INFO Save best model: monitor(max): 0.416466
2020-02-06 14:06:33,050 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 14:06:33,100 P4561 INFO Train loss: 0.343414
2020-02-06 14:06:33,101 P4561 INFO ************ Epoch=9 end ************
2020-02-06 14:52:50,204 P4561 INFO [Metrics] logloss: 0.374311 - AUC: 0.790953
2020-02-06 14:52:50,293 P4561 INFO Save best model: monitor(max): 0.416643
2020-02-06 14:53:04,580 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 14:53:04,628 P4561 INFO Train loss: 0.341959
2020-02-06 14:53:04,629 P4561 INFO ************ Epoch=10 end ************
2020-02-06 15:39:23,838 P4561 INFO [Metrics] logloss: 0.374378 - AUC: 0.790846
2020-02-06 15:39:23,917 P4561 INFO Monitor(max) STOP: 0.416468 !
2020-02-06 15:39:23,917 P4561 INFO Reduce learning rate on plateau: 0.000010
2020-02-06 15:39:23,918 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 15:39:23,991 P4561 INFO Train loss: 0.340768
2020-02-06 15:39:23,991 P4561 INFO ************ Epoch=11 end ************
2020-02-06 16:25:27,483 P4561 INFO [Metrics] logloss: 0.374190 - AUC: 0.791119
2020-02-06 16:25:27,554 P4561 INFO Save best model: monitor(max): 0.416929
2020-02-06 16:25:29,956 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 16:25:30,004 P4561 INFO Train loss: 0.335031
2020-02-06 16:25:30,005 P4561 INFO ************ Epoch=12 end ************
2020-02-06 17:11:30,708 P4561 INFO [Metrics] logloss: 0.374180 - AUC: 0.791160
2020-02-06 17:11:30,930 P4561 INFO Save best model: monitor(max): 0.416980
2020-02-06 17:11:33,318 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 17:11:33,369 P4561 INFO Train loss: 0.334867
2020-02-06 17:11:33,369 P4561 INFO ************ Epoch=13 end ************
2020-02-06 17:57:33,816 P4561 INFO [Metrics] logloss: 0.374184 - AUC: 0.791147
2020-02-06 17:57:33,902 P4561 INFO Monitor(max) STOP: 0.416963 !
2020-02-06 17:57:33,902 P4561 INFO Reduce learning rate on plateau: 0.000001
2020-02-06 17:57:33,902 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 17:57:33,974 P4561 INFO Train loss: 0.334738
2020-02-06 17:57:33,974 P4561 INFO ************ Epoch=14 end ************
2020-02-06 18:43:34,569 P4561 INFO [Metrics] logloss: 0.374180 - AUC: 0.791165
2020-02-06 18:43:34,648 P4561 INFO Save best model: monitor(max): 0.416985
2020-02-06 18:43:37,130 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 18:43:37,180 P4561 INFO Train loss: 0.334081
2020-02-06 18:43:37,180 P4561 INFO ************ Epoch=15 end ************
2020-02-06 19:29:36,916 P4561 INFO [Metrics] logloss: 0.374181 - AUC: 0.791168
2020-02-06 19:29:36,990 P4561 INFO Save best model: monitor(max): 0.416987
2020-02-06 19:29:39,392 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 19:29:39,440 P4561 INFO Train loss: 0.334066
2020-02-06 19:29:39,440 P4561 INFO ************ Epoch=16 end ************
2020-02-06 20:15:39,032 P4561 INFO [Metrics] logloss: 0.374183 - AUC: 0.791171
2020-02-06 20:15:39,141 P4561 INFO Save best model: monitor(max): 0.416988
2020-02-06 20:15:41,658 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 20:15:41,707 P4561 INFO Train loss: 0.334059
2020-02-06 20:15:41,708 P4561 INFO ************ Epoch=17 end ************
2020-02-06 21:01:41,310 P4561 INFO [Metrics] logloss: 0.374184 - AUC: 0.791166
2020-02-06 21:01:41,375 P4561 INFO Monitor(max) STOP: 0.416982 !
2020-02-06 21:01:41,375 P4561 INFO Reduce learning rate on plateau: 0.000001
2020-02-06 21:01:41,375 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 21:01:41,458 P4561 INFO Train loss: 0.334045
2020-02-06 21:01:41,458 P4561 INFO ************ Epoch=18 end ************
2020-02-06 21:47:41,939 P4561 INFO [Metrics] logloss: 0.374184 - AUC: 0.791169
2020-02-06 21:47:42,003 P4561 INFO Monitor(max) STOP: 0.416985 !
2020-02-06 21:47:42,003 P4561 INFO Reduce learning rate on plateau: 0.000001
2020-02-06 21:47:42,003 P4561 INFO Early stopping at epoch=19
2020-02-06 21:47:42,003 P4561 INFO --- 3235/3235 batches finished ---
2020-02-06 21:47:42,081 P4561 INFO Train loss: 0.334032
2020-02-06 21:47:42,081 P4561 INFO Training finished.
2020-02-06 21:47:42,081 P4561 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Avazu/LorentzFM_avazu/avazu_x4_001_d45ad60e/LorentzFM_avazu_x4_001_1246f5a7_avazu_x4_001_d45ad60e_model.ckpt
2020-02-06 21:47:44,037 P4561 INFO ****** Train/validation evaluation ******
2020-02-06 22:01:32,488 P4561 INFO [Metrics] logloss: 0.319675 - AUC: 0.867070
2020-02-06 22:03:15,986 P4561 INFO [Metrics] logloss: 0.374183 - AUC: 0.791171
2020-02-06 22:03:16,152 P4561 INFO ******** Test evaluation ********
2020-02-06 22:03:16,152 P4561 INFO Loading data...
2020-02-06 22:03:16,152 P4561 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-02-06 22:03:16,574 P4561 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-02-06 22:03:16,574 P4561 INFO Loading test data done.
2020-02-06 22:04:59,480 P4561 INFO [Metrics] logloss: 0.374173 - AUC: 0.791223

```
