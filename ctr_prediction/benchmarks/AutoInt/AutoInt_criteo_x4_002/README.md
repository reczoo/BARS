## AutoInt_criteo_x4_002

A hands-on guide to run the AutoInt model on the Criteo_x4_002 dataset.

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
Dataset ID: [Criteo_x4_002](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x4_002). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [AutoInt](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt_criteo_x4_tuner_config_11](./AutoInt_criteo_x4_tuner_config_11). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AutoInt_criteo_x4_002
    nohup python run_expid.py --config ./AutoInt_criteo_x4_tuner_config_11 --expid AutoInt_criteo_x4_015_f1bb9713 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.439038 | 0.812877  |


### Logs
```python
2020-06-04 10:25:03,050 P5109 INFO {
    "attention_dim": "80",
    "attention_layers": "3",
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_001_be98441d",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AutoInt",
    "model_id": "AutoInt_criteo_x4_015_9c21dae1",
    "model_root": "./Criteo/AutoInt_criteo/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "1",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/criteo_x4_001_be98441d/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_be98441d/train.h5",
    "use_hdf5": "True",
    "use_residual": "True",
    "use_scale": "False",
    "use_wide": "True",
    "valid_data": "../data/Criteo/criteo_x4_001_be98441d/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-04 10:25:03,050 P5109 INFO Set up feature encoder...
2020-06-04 10:25:03,051 P5109 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-06-04 10:25:03,051 P5109 INFO Loading data...
2020-06-04 10:25:03,052 P5109 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-06-04 10:25:08,137 P5109 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-06-04 10:25:10,033 P5109 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-06-04 10:25:10,156 P5109 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-04 10:25:10,157 P5109 INFO Loading train data done.
2020-06-04 10:25:18,076 P5109 INFO **** Start training: 3668 batches/epoch ****
2020-06-04 10:39:04,422 P5109 INFO [Metrics] logloss: 0.451164 - AUC: 0.799584
2020-06-04 10:39:04,426 P5109 INFO Save best model: monitor(max): 0.348421
2020-06-04 10:39:05,845 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 10:39:05,898 P5109 INFO Train loss: 0.470655
2020-06-04 10:39:05,899 P5109 INFO ************ Epoch=1 end ************
2020-06-04 10:52:49,536 P5109 INFO [Metrics] logloss: 0.448226 - AUC: 0.802922
2020-06-04 10:52:49,537 P5109 INFO Save best model: monitor(max): 0.354696
2020-06-04 10:52:51,528 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 10:52:51,645 P5109 INFO Train loss: 0.464472
2020-06-04 10:52:51,645 P5109 INFO ************ Epoch=2 end ************
2020-06-04 11:06:36,703 P5109 INFO [Metrics] logloss: 0.446524 - AUC: 0.804605
2020-06-04 11:06:36,704 P5109 INFO Save best model: monitor(max): 0.358081
2020-06-04 11:06:38,455 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 11:06:38,512 P5109 INFO Train loss: 0.462313
2020-06-04 11:06:38,512 P5109 INFO ************ Epoch=3 end ************
2020-06-04 11:20:23,453 P5109 INFO [Metrics] logloss: 0.445637 - AUC: 0.805617
2020-06-04 11:20:23,455 P5109 INFO Save best model: monitor(max): 0.359980
2020-06-04 11:20:25,362 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 11:20:25,417 P5109 INFO Train loss: 0.461016
2020-06-04 11:20:25,417 P5109 INFO ************ Epoch=4 end ************
2020-06-04 11:34:08,037 P5109 INFO [Metrics] logloss: 0.444837 - AUC: 0.806485
2020-06-04 11:34:08,038 P5109 INFO Save best model: monitor(max): 0.361648
2020-06-04 11:34:09,595 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 11:34:09,649 P5109 INFO Train loss: 0.460120
2020-06-04 11:34:09,649 P5109 INFO ************ Epoch=5 end ************
2020-06-04 11:47:52,056 P5109 INFO [Metrics] logloss: 0.444536 - AUC: 0.806994
2020-06-04 11:47:52,057 P5109 INFO Save best model: monitor(max): 0.362459
2020-06-04 11:47:54,020 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 11:47:54,076 P5109 INFO Train loss: 0.459520
2020-06-04 11:47:54,076 P5109 INFO ************ Epoch=6 end ************
2020-06-04 12:01:37,655 P5109 INFO [Metrics] logloss: 0.444086 - AUC: 0.807267
2020-06-04 12:01:37,656 P5109 INFO Save best model: monitor(max): 0.363180
2020-06-04 12:01:38,925 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 12:01:38,980 P5109 INFO Train loss: 0.459089
2020-06-04 12:01:38,980 P5109 INFO ************ Epoch=7 end ************
2020-06-04 12:15:28,860 P5109 INFO [Metrics] logloss: 0.443719 - AUC: 0.807691
2020-06-04 12:15:28,861 P5109 INFO Save best model: monitor(max): 0.363972
2020-06-04 12:15:30,943 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 12:15:31,014 P5109 INFO Train loss: 0.458775
2020-06-04 12:15:31,014 P5109 INFO ************ Epoch=8 end ************
2020-06-04 12:29:16,745 P5109 INFO [Metrics] logloss: 0.443434 - AUC: 0.807955
2020-06-04 12:29:16,746 P5109 INFO Save best model: monitor(max): 0.364521
2020-06-04 12:29:18,084 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 12:29:18,158 P5109 INFO Train loss: 0.458560
2020-06-04 12:29:18,159 P5109 INFO ************ Epoch=9 end ************
2020-06-04 12:43:03,697 P5109 INFO [Metrics] logloss: 0.443375 - AUC: 0.808058
2020-06-04 12:43:03,698 P5109 INFO Save best model: monitor(max): 0.364683
2020-06-04 12:43:05,869 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 12:43:05,936 P5109 INFO Train loss: 0.458417
2020-06-04 12:43:05,937 P5109 INFO ************ Epoch=10 end ************
2020-06-04 12:56:50,082 P5109 INFO [Metrics] logloss: 0.443215 - AUC: 0.808181
2020-06-04 12:56:50,083 P5109 INFO Save best model: monitor(max): 0.364967
2020-06-04 12:56:51,446 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 12:56:51,513 P5109 INFO Train loss: 0.458307
2020-06-04 12:56:51,513 P5109 INFO ************ Epoch=11 end ************
2020-06-04 13:10:33,493 P5109 INFO [Metrics] logloss: 0.443108 - AUC: 0.808370
2020-06-04 13:10:33,495 P5109 INFO Save best model: monitor(max): 0.365262
2020-06-04 13:10:34,972 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 13:10:35,039 P5109 INFO Train loss: 0.458175
2020-06-04 13:10:35,040 P5109 INFO ************ Epoch=12 end ************
2020-06-04 13:24:16,900 P5109 INFO [Metrics] logloss: 0.443001 - AUC: 0.808446
2020-06-04 13:24:16,901 P5109 INFO Save best model: monitor(max): 0.365446
2020-06-04 13:24:18,482 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 13:24:18,555 P5109 INFO Train loss: 0.458133
2020-06-04 13:24:18,556 P5109 INFO ************ Epoch=13 end ************
2020-06-04 13:38:02,815 P5109 INFO [Metrics] logloss: 0.442944 - AUC: 0.808535
2020-06-04 13:38:02,817 P5109 INFO Save best model: monitor(max): 0.365592
2020-06-04 13:38:04,249 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 13:38:04,317 P5109 INFO Train loss: 0.458044
2020-06-04 13:38:04,318 P5109 INFO ************ Epoch=14 end ************
2020-06-04 13:51:46,641 P5109 INFO [Metrics] logloss: 0.442856 - AUC: 0.808666
2020-06-04 13:51:46,642 P5109 INFO Save best model: monitor(max): 0.365810
2020-06-04 13:51:48,339 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 13:51:48,406 P5109 INFO Train loss: 0.457977
2020-06-04 13:51:48,406 P5109 INFO ************ Epoch=15 end ************
2020-06-04 14:05:31,422 P5109 INFO [Metrics] logloss: 0.442608 - AUC: 0.808887
2020-06-04 14:05:31,423 P5109 INFO Save best model: monitor(max): 0.366279
2020-06-04 14:05:32,799 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 14:05:32,867 P5109 INFO Train loss: 0.457970
2020-06-04 14:05:32,867 P5109 INFO ************ Epoch=16 end ************
2020-06-04 14:19:16,165 P5109 INFO [Metrics] logloss: 0.442608 - AUC: 0.808889
2020-06-04 14:19:16,166 P5109 INFO Save best model: monitor(max): 0.366281
2020-06-04 14:19:17,886 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 14:19:17,952 P5109 INFO Train loss: 0.457947
2020-06-04 14:19:17,952 P5109 INFO ************ Epoch=17 end ************
2020-06-04 14:33:01,086 P5109 INFO [Metrics] logloss: 0.442675 - AUC: 0.808931
2020-06-04 14:33:01,087 P5109 INFO Monitor(max) STOP: 0.366255 !
2020-06-04 14:33:01,087 P5109 INFO Reduce learning rate on plateau: 0.000100
2020-06-04 14:33:01,087 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 14:33:01,154 P5109 INFO Train loss: 0.457904
2020-06-04 14:33:01,155 P5109 INFO ************ Epoch=18 end ************
2020-06-04 14:46:46,056 P5109 INFO [Metrics] logloss: 0.439397 - AUC: 0.812387
2020-06-04 14:46:46,058 P5109 INFO Save best model: monitor(max): 0.372989
2020-06-04 14:46:47,324 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 14:46:47,391 P5109 INFO Train loss: 0.442974
2020-06-04 14:46:47,391 P5109 INFO ************ Epoch=19 end ************
2020-06-04 15:00:28,286 P5109 INFO [Metrics] logloss: 0.439321 - AUC: 0.812559
2020-06-04 15:00:28,287 P5109 INFO Save best model: monitor(max): 0.373237
2020-06-04 15:00:29,571 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 15:00:29,636 P5109 INFO Train loss: 0.438189
2020-06-04 15:00:29,636 P5109 INFO ************ Epoch=20 end ************
2020-06-04 15:14:11,548 P5109 INFO [Metrics] logloss: 0.439889 - AUC: 0.812120
2020-06-04 15:14:11,549 P5109 INFO Monitor(max) STOP: 0.372231 !
2020-06-04 15:14:11,550 P5109 INFO Reduce learning rate on plateau: 0.000010
2020-06-04 15:14:11,550 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 15:14:11,616 P5109 INFO Train loss: 0.435616
2020-06-04 15:14:11,616 P5109 INFO ************ Epoch=21 end ************
2020-06-04 15:27:52,652 P5109 INFO [Metrics] logloss: 0.443747 - AUC: 0.809263
2020-06-04 15:27:52,653 P5109 INFO Monitor(max) STOP: 0.365516 !
2020-06-04 15:27:52,653 P5109 INFO Reduce learning rate on plateau: 0.000001
2020-06-04 15:27:52,653 P5109 INFO Early stopping at epoch=22
2020-06-04 15:27:52,653 P5109 INFO --- 3668/3668 batches finished ---
2020-06-04 15:27:52,718 P5109 INFO Train loss: 0.426606
2020-06-04 15:27:52,719 P5109 INFO Training finished.
2020-06-04 15:27:52,719 P5109 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/AutoInt_criteo/criteo_x4_001_be98441d/AutoInt_criteo_x4_015_9c21dae1_model.ckpt
2020-06-04 15:27:53,938 P5109 INFO ****** Train/validation evaluation ******
2020-06-04 15:31:49,102 P5109 INFO [Metrics] logloss: 0.424682 - AUC: 0.827847
2020-06-04 15:32:16,270 P5109 INFO [Metrics] logloss: 0.439321 - AUC: 0.812559
2020-06-04 15:32:16,385 P5109 INFO ******** Test evaluation ********
2020-06-04 15:32:16,385 P5109 INFO Loading data...
2020-06-04 15:32:16,385 P5109 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-06-04 15:32:17,125 P5109 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-04 15:32:17,125 P5109 INFO Loading test data done.
2020-06-04 15:32:44,742 P5109 INFO [Metrics] logloss: 0.439038 - AUC: 0.812877

```
