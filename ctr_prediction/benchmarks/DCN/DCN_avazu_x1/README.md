## DCN_avazu_x1

A guide to benchmark DCN on [Avazu_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x1).

Author: [zhujiem](https://github.com/zhujiem)

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
  CUDA: 10.0.130
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  ```

### Dataset

To reproduce the dataset splitting, please follow the details of [Avazu_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x1) to get data ready.

### Code

We use [FuxiCTR v1.1](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment.

1. Install FuxiCTR and all the dependencies.
   ```bash
   pip install fuxictr==1.1.*
   ```
   
2. Put the downloaded dataset in `../data/Avazu/Avazu_x1`. 

3. The dataset_config and model_config files are available in the sub-folder `DCN_avazu_x1_tuner_config_02`.

   Note that in this setting, we follow the AFN work to fix embedding_dim=10, batch_size=4096, and MLP_hidden_units=[400, 400, 400] to make fair comparisons. Other hyper-parameters are tuned via grid search.

4. Run the following script to start.

  ```bash
  cd BARS/ctr_prediction/benchmarks/DCN/DCN_avazu_x1
  nohup python run_expid.py --version pytorch --config DCN_avazu_x1_tuner_config_02 --expid DCN_avazu_x1_010_6afb45f5 --gpu 0 > run.log & 
  tail -f run.log
  ```


### Results
```python
[Metrics] logloss: 0.366885 - AUC: 0.764671
```

### Logs
```python
2021-12-30 13:48:20,969 P30627 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "crossing_layers": "5",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "1",
    "model": "DCN",
    "model_id": "DCN_avazu_x1_010_6afb45f5",
    "model_root": "./Avazu/DCN_avazu_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2021-12-30 13:48:20,969 P30627 INFO Set up feature encoder...
2021-12-30 13:48:20,970 P30627 INFO Reading file: ../data/Avazu/Avazu_x1/train.csv
2021-12-30 13:49:38,387 P30627 INFO Reading file: ../data/Avazu/Avazu_x1/valid.csv
2021-12-30 13:49:50,345 P30627 INFO Reading file: ../data/Avazu/Avazu_x1/test.csv
2021-12-30 13:50:12,989 P30627 INFO Preprocess feature columns...
2021-12-30 13:50:16,035 P30627 INFO Fit feature encoder...
2021-12-30 13:50:16,036 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_1', 'type': 'categorical'}
2021-12-30 13:50:24,103 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_2', 'type': 'categorical'}
2021-12-30 13:50:32,402 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_3', 'type': 'categorical'}
2021-12-30 13:50:40,913 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_4', 'type': 'categorical'}
2021-12-30 13:50:48,867 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_5', 'type': 'categorical'}
2021-12-30 13:50:56,815 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_6', 'type': 'categorical'}
2021-12-30 13:51:04,970 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_7', 'type': 'categorical'}
2021-12-30 13:51:12,610 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_8', 'type': 'categorical'}
2021-12-30 13:51:19,772 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_9', 'type': 'categorical'}
2021-12-30 13:51:30,737 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_10', 'type': 'categorical'}
2021-12-30 13:51:56,607 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_11', 'type': 'categorical'}
2021-12-30 13:52:04,112 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_12', 'type': 'categorical'}
2021-12-30 13:52:10,569 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_13', 'type': 'categorical'}
2021-12-30 13:52:17,024 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_14', 'type': 'categorical'}
2021-12-30 13:52:23,904 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_15', 'type': 'categorical'}
2021-12-30 13:52:30,817 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_16', 'type': 'categorical'}
2021-12-30 13:52:37,357 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_17', 'type': 'categorical'}
2021-12-30 13:52:44,197 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_18', 'type': 'categorical'}
2021-12-30 13:52:51,638 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_19', 'type': 'categorical'}
2021-12-30 13:52:59,569 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_20', 'type': 'categorical'}
2021-12-30 13:53:07,559 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_21', 'type': 'categorical'}
2021-12-30 13:53:15,457 P30627 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_22', 'type': 'categorical'}
2021-12-30 13:53:23,157 P30627 INFO Set feature index...
2021-12-30 13:53:23,158 P30627 INFO Pickle feature_encode: ../data/Avazu/avazu_x1_3fb65689/feature_encoder.pkl
2021-12-30 13:53:30,081 P30627 INFO Save feature_map to json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2021-12-30 13:53:30,083 P30627 INFO Set feature encoder done.
2021-12-30 13:53:30,083 P30627 INFO Transform feature columns...
2021-12-30 13:59:00,459 P30627 INFO Saving data to h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2021-12-30 13:59:04,109 P30627 INFO Preprocess feature columns...
2021-12-30 13:59:05,164 P30627 INFO Transform feature columns...
2021-12-30 13:59:51,903 P30627 INFO Saving data to h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2021-12-30 13:59:52,419 P30627 INFO Preprocess feature columns...
2021-12-30 13:59:53,197 P30627 INFO Transform feature columns...
2021-12-30 14:01:24,620 P30627 INFO Saving data to h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2021-12-30 14:01:25,737 P30627 INFO Transform csv data to h5 done.
2021-12-30 14:01:25,737 P30627 INFO Loading data...
2021-12-30 14:01:25,742 P30627 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2021-12-30 14:01:28,948 P30627 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2021-12-30 14:01:29,368 P30627 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2021-12-30 14:01:29,368 P30627 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2021-12-30 14:01:29,368 P30627 INFO Loading train data done.
2021-12-30 14:01:32,611 P30627 INFO Total number of parameters: 13398011.
2021-12-30 14:01:32,611 P30627 INFO Start training: 6910 batches/epoch
2021-12-30 14:01:32,611 P30627 INFO ************ Epoch=1 start ************
2021-12-30 14:12:23,690 P30627 INFO [Metrics] logloss: 0.397372 - AUC: 0.743992
2021-12-30 14:12:23,691 P30627 INFO Save best model: monitor(max): 0.346619
2021-12-30 14:12:23,738 P30627 INFO --- 6910/6910 batches finished ---
2021-12-30 14:12:23,954 P30627 INFO Train loss: 0.420003
2021-12-30 14:12:23,954 P30627 INFO ************ Epoch=1 end ************
2021-12-30 14:22:53,456 P30627 INFO [Metrics] logloss: 0.396507 - AUC: 0.745616
2021-12-30 14:22:53,456 P30627 INFO Save best model: monitor(max): 0.349109
2021-12-30 14:22:53,532 P30627 INFO --- 6910/6910 batches finished ---
2021-12-30 14:22:53,731 P30627 INFO Train loss: 0.420978
2021-12-30 14:22:53,731 P30627 INFO ************ Epoch=2 end ************
2021-12-30 14:34:09,667 P30627 INFO [Metrics] logloss: 0.396295 - AUC: 0.745970
2021-12-30 14:34:09,668 P30627 INFO Save best model: monitor(max): 0.349675
2021-12-30 14:34:09,747 P30627 INFO --- 6910/6910 batches finished ---
2021-12-30 14:34:09,955 P30627 INFO Train loss: 0.422253
2021-12-30 14:34:09,955 P30627 INFO ************ Epoch=3 end ************
2021-12-30 14:45:30,520 P30627 INFO [Metrics] logloss: 0.399075 - AUC: 0.742307
2021-12-30 14:45:30,521 P30627 INFO Monitor(max) STOP: 0.343232 !
2021-12-30 14:45:30,521 P30627 INFO Reduce learning rate on plateau: 0.000100
2021-12-30 14:45:30,521 P30627 INFO --- 6910/6910 batches finished ---
2021-12-30 14:45:30,748 P30627 INFO Train loss: 0.422529
2021-12-30 14:45:30,748 P30627 INFO ************ Epoch=4 end ************
2021-12-30 14:56:48,432 P30627 INFO [Metrics] logloss: 0.396215 - AUC: 0.747274
2021-12-30 14:56:48,432 P30627 INFO Save best model: monitor(max): 0.351059
2021-12-30 14:56:48,530 P30627 INFO --- 6910/6910 batches finished ---
2021-12-30 14:56:48,788 P30627 INFO Train loss: 0.399464
2021-12-30 14:56:48,788 P30627 INFO ************ Epoch=5 end ************
2021-12-30 15:08:13,549 P30627 INFO [Metrics] logloss: 0.396993 - AUC: 0.744173
2021-12-30 15:08:13,550 P30627 INFO Monitor(max) STOP: 0.347180 !
2021-12-30 15:08:13,550 P30627 INFO Reduce learning rate on plateau: 0.000010
2021-12-30 15:08:13,550 P30627 INFO --- 6910/6910 batches finished ---
2021-12-30 15:08:13,781 P30627 INFO Train loss: 0.395828
2021-12-30 15:08:13,781 P30627 INFO ************ Epoch=6 end ************
2021-12-30 15:19:37,669 P30627 INFO [Metrics] logloss: 0.401493 - AUC: 0.737824
2021-12-30 15:19:37,670 P30627 INFO Monitor(max) STOP: 0.336331 !
2021-12-30 15:19:37,670 P30627 INFO Reduce learning rate on plateau: 0.000001
2021-12-30 15:19:37,670 P30627 INFO Early stopping at epoch=7
2021-12-30 15:19:37,670 P30627 INFO --- 6910/6910 batches finished ---
2021-12-30 15:19:37,920 P30627 INFO Train loss: 0.386000
2021-12-30 15:19:37,920 P30627 INFO Training finished.
2021-12-30 15:19:37,920 P30627 INFO Load best model: /home/xx/FuxiCTR/benchmarks/Avazu/DCN_avazu_x1/avazu_x1_3fb65689/DCN_avazu_x1_010_6afb45f5.model
2021-12-30 15:19:37,966 P30627 INFO ****** Validation evaluation ******
2021-12-30 15:19:51,774 P30627 INFO [Metrics] logloss: 0.396215 - AUC: 0.747274
2021-12-30 15:19:51,855 P30627 INFO ******** Test evaluation ********
2021-12-30 15:19:51,855 P30627 INFO Loading data...
2021-12-30 15:19:51,856 P30627 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2021-12-30 15:19:52,937 P30627 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2021-12-30 15:19:52,938 P30627 INFO Loading test data done.
2021-12-30 15:20:21,279 P30627 INFO [Metrics] logloss: 0.366885 - AUC: 0.764671

```
