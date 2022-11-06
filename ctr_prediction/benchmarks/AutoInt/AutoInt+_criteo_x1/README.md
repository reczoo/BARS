## AutoInt+_criteo_x1

A hands-on guide to run the AutoInt model on the Criteo_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

### Index

[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs) | [Revision History](#Revision-History)

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
  fuxictr: 1.2.2
  ```

### Dataset

Dataset ID: [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo#Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.2](https://github.com/xue-pai/FuxiCTR/tree/v1.2.2) for this experiment. See the model code: [AutoInt](https://github.com/xue-pai/FuxiCTR/blob/v1.2.2/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.2.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.2.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [fuxictr_version.py](./fuxictr_version.py#L3) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt+_criteo_x1_tuner_config_08](./AutoInt+_criteo_x1_tuner_config_08). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd AutoInt+_criteo_x1
   nohup python run_expid.py --config ./AutoInt+_criteo_x1_tuner_config_08 --expid AutoInt_criteo_x1_005_a4b5787e --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

| AUC      | logloss  |
|:--------:|:--------:|
| 0.813939 | 0.437892 |

### Logs

```python
2022-07-01 10:34:23,159 P45054 INFO {
    "attention_dim": "256",
    "attention_layers": "3",
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "4",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "layer_norm": "True",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AutoInt",
    "model_id": "AutoInt_criteo_x1_005_a4b5787e",
    "model_root": "./Criteo/AutoInt_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_heads": "2",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_hdf5": "True",
    "use_residual": "True",
    "use_scale": "False",
    "use_wide": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-07-01 10:34:23,159 P45054 INFO Set up feature encoder...
2022-07-01 10:34:23,160 P45054 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-07-01 10:34:23,160 P45054 INFO Loading data...
2022-07-01 10:34:23,161 P45054 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-07-01 10:34:28,067 P45054 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-07-01 10:34:29,265 P45054 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-07-01 10:34:29,266 P45054 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-07-01 10:34:29,266 P45054 INFO Loading train data done.
2022-07-01 10:34:33,998 P45054 INFO Total number of parameters: 23844454.
2022-07-01 10:34:33,998 P45054 INFO Start training: 8058 batches/epoch
2022-07-01 10:34:33,998 P45054 INFO ************ Epoch=1 start ************
2022-07-01 11:00:40,646 P45054 INFO [Metrics] AUC: 0.802593 - logloss: 0.448432
2022-07-01 11:00:40,648 P45054 INFO Save best model: monitor(max): 0.802593
2022-07-01 11:00:40,915 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 11:00:40,954 P45054 INFO Train loss: 0.465839
2022-07-01 11:00:40,954 P45054 INFO ************ Epoch=1 end ************
2022-07-01 11:26:43,364 P45054 INFO [Metrics] AUC: 0.805218 - logloss: 0.446432
2022-07-01 11:26:43,365 P45054 INFO Save best model: monitor(max): 0.805218
2022-07-01 11:26:43,480 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 11:26:43,524 P45054 INFO Train loss: 0.458519
2022-07-01 11:26:43,524 P45054 INFO ************ Epoch=2 end ************
2022-07-01 11:52:49,800 P45054 INFO [Metrics] AUC: 0.806444 - logloss: 0.444995
2022-07-01 11:52:49,801 P45054 INFO Save best model: monitor(max): 0.806444
2022-07-01 11:52:49,915 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 11:52:49,963 P45054 INFO Train loss: 0.456570
2022-07-01 11:52:49,963 P45054 INFO ************ Epoch=3 end ************
2022-07-01 12:18:51,840 P45054 INFO [Metrics] AUC: 0.807135 - logloss: 0.444310
2022-07-01 12:18:51,842 P45054 INFO Save best model: monitor(max): 0.807135
2022-07-01 12:18:51,957 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 12:18:51,998 P45054 INFO Train loss: 0.455652
2022-07-01 12:18:51,998 P45054 INFO ************ Epoch=4 end ************
2022-07-01 12:44:54,770 P45054 INFO [Metrics] AUC: 0.807669 - logloss: 0.443929
2022-07-01 12:44:54,772 P45054 INFO Save best model: monitor(max): 0.807669
2022-07-01 12:44:54,884 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 12:44:54,926 P45054 INFO Train loss: 0.455046
2022-07-01 12:44:54,926 P45054 INFO ************ Epoch=5 end ************
2022-07-01 13:11:01,067 P45054 INFO [Metrics] AUC: 0.808001 - logloss: 0.443488
2022-07-01 13:11:01,068 P45054 INFO Save best model: monitor(max): 0.808001
2022-07-01 13:11:01,202 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 13:11:01,253 P45054 INFO Train loss: 0.454674
2022-07-01 13:11:01,253 P45054 INFO ************ Epoch=6 end ************
2022-07-01 13:37:17,020 P45054 INFO [Metrics] AUC: 0.808240 - logloss: 0.443349
2022-07-01 13:37:17,021 P45054 INFO Save best model: monitor(max): 0.808240
2022-07-01 13:37:17,126 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 13:37:17,173 P45054 INFO Train loss: 0.454345
2022-07-01 13:37:17,174 P45054 INFO ************ Epoch=7 end ************
2022-07-01 14:03:29,172 P45054 INFO [Metrics] AUC: 0.808510 - logloss: 0.442981
2022-07-01 14:03:29,173 P45054 INFO Save best model: monitor(max): 0.808510
2022-07-01 14:03:29,310 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 14:03:29,371 P45054 INFO Train loss: 0.454120
2022-07-01 14:03:29,371 P45054 INFO ************ Epoch=8 end ************
2022-07-01 14:29:47,364 P45054 INFO [Metrics] AUC: 0.808778 - logloss: 0.442801
2022-07-01 14:29:47,365 P45054 INFO Save best model: monitor(max): 0.808778
2022-07-01 14:29:47,499 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 14:29:47,540 P45054 INFO Train loss: 0.453915
2022-07-01 14:29:47,540 P45054 INFO ************ Epoch=9 end ************
2022-07-01 14:55:55,881 P45054 INFO [Metrics] AUC: 0.808916 - logloss: 0.442864
2022-07-01 14:55:55,883 P45054 INFO Save best model: monitor(max): 0.808916
2022-07-01 14:55:56,006 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 14:55:56,059 P45054 INFO Train loss: 0.453772
2022-07-01 14:55:56,060 P45054 INFO ************ Epoch=10 end ************
2022-07-01 15:21:58,839 P45054 INFO [Metrics] AUC: 0.809035 - logloss: 0.442503
2022-07-01 15:21:58,841 P45054 INFO Save best model: monitor(max): 0.809035
2022-07-01 15:21:58,957 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 15:21:59,001 P45054 INFO Train loss: 0.453589
2022-07-01 15:21:59,001 P45054 INFO ************ Epoch=11 end ************
2022-07-01 15:48:17,060 P45054 INFO [Metrics] AUC: 0.809244 - logloss: 0.442330
2022-07-01 15:48:17,062 P45054 INFO Save best model: monitor(max): 0.809244
2022-07-01 15:48:17,200 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 15:48:17,252 P45054 INFO Train loss: 0.453494
2022-07-01 15:48:17,252 P45054 INFO ************ Epoch=12 end ************
2022-07-01 16:14:25,875 P45054 INFO [Metrics] AUC: 0.809261 - logloss: 0.442302
2022-07-01 16:14:25,877 P45054 INFO Save best model: monitor(max): 0.809261
2022-07-01 16:14:26,001 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 16:14:26,042 P45054 INFO Train loss: 0.453341
2022-07-01 16:14:26,043 P45054 INFO ************ Epoch=13 end ************
2022-07-01 16:40:35,100 P45054 INFO [Metrics] AUC: 0.809367 - logloss: 0.442394
2022-07-01 16:40:35,101 P45054 INFO Save best model: monitor(max): 0.809367
2022-07-01 16:40:35,216 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 16:40:35,263 P45054 INFO Train loss: 0.453243
2022-07-01 16:40:35,263 P45054 INFO ************ Epoch=14 end ************
2022-07-01 17:06:33,947 P45054 INFO [Metrics] AUC: 0.809445 - logloss: 0.442184
2022-07-01 17:06:33,948 P45054 INFO Save best model: monitor(max): 0.809445
2022-07-01 17:06:34,071 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 17:06:34,121 P45054 INFO Train loss: 0.453138
2022-07-01 17:06:34,121 P45054 INFO ************ Epoch=15 end ************
2022-07-01 17:32:30,388 P45054 INFO [Metrics] AUC: 0.809539 - logloss: 0.442247
2022-07-01 17:32:30,390 P45054 INFO Save best model: monitor(max): 0.809539
2022-07-01 17:32:30,504 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 17:32:30,546 P45054 INFO Train loss: 0.453102
2022-07-01 17:32:30,546 P45054 INFO ************ Epoch=16 end ************
2022-07-01 17:58:25,833 P45054 INFO [Metrics] AUC: 0.809583 - logloss: 0.442018
2022-07-01 17:58:25,834 P45054 INFO Save best model: monitor(max): 0.809583
2022-07-01 17:58:25,962 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 17:58:26,002 P45054 INFO Train loss: 0.453036
2022-07-01 17:58:26,002 P45054 INFO ************ Epoch=17 end ************
2022-07-01 18:24:20,420 P45054 INFO [Metrics] AUC: 0.809603 - logloss: 0.441999
2022-07-01 18:24:20,421 P45054 INFO Save best model: monitor(max): 0.809603
2022-07-01 18:24:20,538 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 18:24:20,582 P45054 INFO Train loss: 0.452950
2022-07-01 18:24:20,582 P45054 INFO ************ Epoch=18 end ************
2022-07-01 18:50:13,777 P45054 INFO [Metrics] AUC: 0.809686 - logloss: 0.441934
2022-07-01 18:50:13,779 P45054 INFO Save best model: monitor(max): 0.809686
2022-07-01 18:50:13,882 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 18:50:13,923 P45054 INFO Train loss: 0.452879
2022-07-01 18:50:13,924 P45054 INFO ************ Epoch=19 end ************
2022-07-01 19:16:06,810 P45054 INFO [Metrics] AUC: 0.809732 - logloss: 0.441919
2022-07-01 19:16:06,811 P45054 INFO Save best model: monitor(max): 0.809732
2022-07-01 19:16:06,925 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 19:16:06,972 P45054 INFO Train loss: 0.452795
2022-07-01 19:16:06,973 P45054 INFO ************ Epoch=20 end ************
2022-07-01 19:41:57,002 P45054 INFO [Metrics] AUC: 0.809772 - logloss: 0.441899
2022-07-01 19:41:57,003 P45054 INFO Save best model: monitor(max): 0.809772
2022-07-01 19:41:57,140 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 19:41:57,180 P45054 INFO Train loss: 0.452725
2022-07-01 19:41:57,181 P45054 INFO ************ Epoch=21 end ************
2022-07-01 20:07:46,660 P45054 INFO [Metrics] AUC: 0.809762 - logloss: 0.441840
2022-07-01 20:07:46,661 P45054 INFO Monitor(max) STOP: 0.809762 !
2022-07-01 20:07:46,661 P45054 INFO Reduce learning rate on plateau: 0.000100
2022-07-01 20:07:46,661 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 20:07:46,703 P45054 INFO Train loss: 0.452685
2022-07-01 20:07:46,703 P45054 INFO ************ Epoch=22 end ************
2022-07-01 20:33:35,860 P45054 INFO [Metrics] AUC: 0.813045 - logloss: 0.438863
2022-07-01 20:33:35,862 P45054 INFO Save best model: monitor(max): 0.813045
2022-07-01 20:33:35,977 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 20:33:36,017 P45054 INFO Train loss: 0.441532
2022-07-01 20:33:36,017 P45054 INFO ************ Epoch=23 end ************
2022-07-01 20:59:29,726 P45054 INFO [Metrics] AUC: 0.813527 - logloss: 0.438384
2022-07-01 20:59:29,728 P45054 INFO Save best model: monitor(max): 0.813527
2022-07-01 20:59:29,851 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 20:59:29,891 P45054 INFO Train loss: 0.437178
2022-07-01 20:59:29,891 P45054 INFO ************ Epoch=24 end ************
2022-07-01 21:25:19,265 P45054 INFO [Metrics] AUC: 0.813597 - logloss: 0.438378
2022-07-01 21:25:19,266 P45054 INFO Save best model: monitor(max): 0.813597
2022-07-01 21:25:19,385 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 21:25:19,422 P45054 INFO Train loss: 0.435387
2022-07-01 21:25:19,423 P45054 INFO ************ Epoch=25 end ************
2022-07-01 21:51:11,049 P45054 INFO [Metrics] AUC: 0.813477 - logloss: 0.438494
2022-07-01 21:51:11,051 P45054 INFO Monitor(max) STOP: 0.813477 !
2022-07-01 21:51:11,051 P45054 INFO Reduce learning rate on plateau: 0.000010
2022-07-01 21:51:11,051 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 21:51:11,091 P45054 INFO Train loss: 0.434035
2022-07-01 21:51:11,091 P45054 INFO ************ Epoch=26 end ************
2022-07-01 22:17:03,939 P45054 INFO [Metrics] AUC: 0.813151 - logloss: 0.439088
2022-07-01 22:17:03,941 P45054 INFO Monitor(max) STOP: 0.813151 !
2022-07-01 22:17:03,941 P45054 INFO Reduce learning rate on plateau: 0.000001
2022-07-01 22:17:03,941 P45054 INFO Early stopping at epoch=27
2022-07-01 22:17:03,941 P45054 INFO --- 8058/8058 batches finished ---
2022-07-01 22:17:03,993 P45054 INFO Train loss: 0.429952
2022-07-01 22:17:03,994 P45054 INFO Training finished.
2022-07-01 22:17:03,994 P45054 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/AutoInt_criteo_x1/criteo_x1_7b681156/AutoInt_criteo_x1_005_a4b5787e.model
2022-07-01 22:17:06,687 P45054 INFO ****** Validation evaluation ******
2022-07-01 22:18:50,329 P45054 INFO [Metrics] AUC: 0.813597 - logloss: 0.438378
2022-07-01 22:18:50,404 P45054 INFO ******** Test evaluation ********
2022-07-01 22:18:50,405 P45054 INFO Loading data...
2022-07-01 22:18:50,405 P45054 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-07-01 22:18:51,198 P45054 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-07-01 22:18:51,198 P45054 INFO Loading test data done.
2022-07-01 22:19:48,411 P45054 INFO [Metrics] AUC: 0.813939 - logloss: 0.437892
```

### Revision History

- [Version 1](https://github.com/openbenchmark/BARS/tree/88d3a0faa4565e975141ae89a52d35d3a8b56eda/ctr_prediction/benchmarks/AutoInt/AutoInt%2B_criteo_x1#autoint_criteo_x1): deprecated due to bug fix [#30](https://github.com/xue-pai/FuxiCTR/issues/30) of FuxiCTR.
