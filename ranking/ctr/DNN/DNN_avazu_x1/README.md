## DNN_avazu_x1

A hands-on guide to run the DNN model on the Avazu_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index

[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments

+ Hardware
  
  ```python
  CPU: Intel(R) Xeon(R) Gold 6151 CPU @ 3.00GHz
  GPU: Tesla V100 32G
  RAM: 512G
  ```

+ Software
  
  ```python
  CUDA: 10.2
  python: 3.7.10
  pytorch: 1.0.0
  pandas: 1.1.5
  numpy: 1.19.2
  scipy: 1.5.4
  sklearn: 0.22.1
  pyyaml: 5.4.1
  h5py: 2.8.0
  tqdm: 4.60.0
  fuxictr: 1.1.0
  ```

### Dataset

Dataset ID: [Avazu_x1](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DNN](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DNN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then edit [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DNN_avazu_x1_tuner_config_01](./DNN_avazu_x1_tuner_config_01). Make sure that the data paths in `dataset_config.yaml` are correctly set to the directory we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd .
   nohup python run_expid.py --config ./DNN_avazu_x1_tuner_config_01 --expid DNN_avazu_x1_001_3da2d674 --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

| AUC      | logloss  |
|:--------:|:--------:|
| 0.763827 | 0.367394 |

### Logs

```python
2024-06-11 15:24:47,922 P169311 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DNN",
    "model_id": "DNN_avazu_x1_001_3da2d674",
    "model_root": "./Avazu/DNN_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
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
    "verbose": "0",
    "version": "pytorch"
}
2024-06-11 15:24:47,924 P169311 INFO Set up feature encoder...
2024-06-11 15:24:47,924 P169311 INFO Reading file: ../data/Avazu/Avazu_x1/train.csv
2024-06-11 15:25:48,605 P169311 INFO Reading file: ../data/Avazu/Avazu_x1/valid.csv
2024-06-11 15:25:56,754 P169311 INFO Reading file: ../data/Avazu/Avazu_x1/test.csv
2024-06-11 15:26:13,225 P169311 INFO Preprocess feature columns...
2024-06-11 15:26:15,317 P169311 INFO Fit feature encoder...
2024-06-11 15:26:15,317 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_1', 'type': 'categorical'}
2024-06-11 15:26:25,500 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_2', 'type': 'categorical'}
2024-06-11 15:26:35,766 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_3', 'type': 'categorical'}
2024-06-11 15:26:46,412 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_4', 'type': 'categorical'}
2024-06-11 15:26:56,793 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_5', 'type': 'categorical'}
2024-06-11 15:27:06,938 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_6', 'type': 'categorical'}
2024-06-11 15:27:17,414 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_7', 'type': 'categorical'}
2024-06-11 15:27:27,648 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_8', 'type': 'categorical'}
2024-06-11 15:27:37,620 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_9', 'type': 'categorical'}
2024-06-11 15:27:50,724 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_10', 'type': 'categorical'}
2024-06-11 15:28:14,989 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_11', 'type': 'categorical'}
2024-06-11 15:28:25,269 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_12', 'type': 'categorical'}
2024-06-11 15:28:35,297 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_13', 'type': 'categorical'}
2024-06-11 15:28:45,089 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_14', 'type': 'categorical'}
2024-06-11 15:28:55,233 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_15', 'type': 'categorical'}
2024-06-11 15:29:05,185 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_16', 'type': 'categorical'}
2024-06-11 15:29:15,109 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_17', 'type': 'categorical'}
2024-06-11 15:29:25,233 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_18', 'type': 'categorical'}
2024-06-11 15:29:34,955 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_19', 'type': 'categorical'}
2024-06-11 15:29:44,869 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_20', 'type': 'categorical'}
2024-06-11 15:29:54,723 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_21', 'type': 'categorical'}
2024-06-11 15:30:04,695 P169311 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'feat_22', 'type': 'categorical'}
2024-06-11 15:30:14,458 P169311 INFO Set feature index...
2024-06-11 15:30:14,458 P169311 INFO Pickle feature_encoder: ../data/Avazu/avazu_x1_3fb65689/feature_encoder.pkl
2024-06-11 15:30:20,672 P169311 INFO Save feature_map to json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2024-06-11 15:30:20,673 P169311 INFO Set feature encoder done.
2024-06-11 15:30:20,673 P169311 INFO Transform feature columns...
2024-06-11 15:36:56,128 P169311 INFO Saving data to h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2024-06-11 15:36:59,477 P169311 INFO Preprocess feature columns...
2024-06-11 15:36:59,817 P169311 INFO Transform feature columns...
2024-06-11 15:37:57,112 P169311 INFO Saving data to h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2024-06-11 15:37:57,612 P169311 INFO Preprocess feature columns...
2024-06-11 15:37:58,172 P169311 INFO Transform feature columns...
2024-06-11 15:39:52,553 P169311 INFO Saving data to h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2024-06-11 15:39:53,540 P169311 INFO Transform csv data to h5 done.
2024-06-11 15:39:53,540 P169311 INFO Loading data...
2024-06-11 15:39:53,546 P169311 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2024-06-11 15:39:55,407 P169311 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2024-06-11 15:39:55,744 P169311 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2024-06-11 15:39:55,744 P169311 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2024-06-11 15:39:55,744 P169311 INFO Loading train data done.
2024-06-11 15:40:01,389 P169311 INFO Total number of parameters: 13395591.
2024-06-11 15:40:01,389 P169311 INFO Start training: 6910 batches/epoch
2024-06-11 15:40:01,389 P169311 INFO ************ Epoch=1 start ************
2024-06-11 15:44:00,317 P169311 INFO [Metrics] AUC: 0.738388 - logloss: 0.400051
2024-06-11 15:44:00,317 P169311 INFO Save best model: monitor(max): 0.738388
2024-06-11 15:44:00,375 P169311 INFO --- 6910/6910 batches finished ---
2024-06-11 15:44:00,526 P169311 INFO Train loss: 0.428714
2024-06-11 15:44:00,526 P169311 INFO ************ Epoch=1 end ************
2024-06-11 15:47:58,946 P169311 INFO [Metrics] AUC: 0.740958 - logloss: 0.398840
2024-06-11 15:47:58,948 P169311 INFO Save best model: monitor(max): 0.740958
2024-06-11 15:47:59,078 P169311 INFO --- 6910/6910 batches finished ---
2024-06-11 15:47:59,240 P169311 INFO Train loss: 0.428697
2024-06-11 15:47:59,240 P169311 INFO ************ Epoch=2 end ************
2024-06-11 15:51:58,178 P169311 INFO [Metrics] AUC: 0.741541 - logloss: 0.399047
2024-06-11 15:51:58,185 P169311 INFO Save best model: monitor(max): 0.741541
2024-06-11 15:51:58,314 P169311 INFO --- 6910/6910 batches finished ---
2024-06-11 15:51:58,477 P169311 INFO Train loss: 0.428326
2024-06-11 15:51:58,478 P169311 INFO ************ Epoch=3 end ************
2024-06-11 15:56:00,843 P169311 INFO [Metrics] AUC: 0.739486 - logloss: 0.399609
2024-06-11 15:56:00,845 P169311 INFO Monitor(max) STOP: 0.739486 !
2024-06-11 15:56:00,845 P169311 INFO Reduce learning rate on plateau: 0.000100
2024-06-11 15:56:00,846 P169311 INFO --- 6910/6910 batches finished ---
2024-06-11 15:56:01,004 P169311 INFO Train loss: 0.428737
2024-06-11 15:56:01,004 P169311 INFO ************ Epoch=4 end ************
2024-06-11 15:59:59,972 P169311 INFO [Metrics] AUC: 0.746336 - logloss: 0.396527
2024-06-11 15:59:59,972 P169311 INFO Save best model: monitor(max): 0.746336
2024-06-11 16:00:00,104 P169311 INFO --- 6910/6910 batches finished ---
2024-06-11 16:00:00,264 P169311 INFO Train loss: 0.403978
2024-06-11 16:00:00,265 P169311 INFO ************ Epoch=5 end ************
2024-06-11 16:03:58,553 P169311 INFO [Metrics] AUC: 0.745096 - logloss: 0.396898
2024-06-11 16:03:58,554 P169311 INFO Monitor(max) STOP: 0.745096 !
2024-06-11 16:03:58,554 P169311 INFO Reduce learning rate on plateau: 0.000010
2024-06-11 16:03:58,554 P169311 INFO --- 6910/6910 batches finished ---
2024-06-11 16:03:58,720 P169311 INFO Train loss: 0.404432
2024-06-11 16:03:58,720 P169311 INFO ************ Epoch=6 end ************
2024-06-11 16:07:58,646 P169311 INFO [Metrics] AUC: 0.744883 - logloss: 0.397435
2024-06-11 16:07:58,648 P169311 INFO Monitor(max) STOP: 0.744883 !
2024-06-11 16:07:58,648 P169311 INFO Reduce learning rate on plateau: 0.000001
2024-06-11 16:07:58,648 P169311 INFO Early stopping at epoch=7
2024-06-11 16:07:58,648 P169311 INFO --- 6910/6910 batches finished ---
2024-06-11 16:07:58,839 P169311 INFO Train loss: 0.394071
2024-06-11 16:07:58,839 P169311 INFO Training finished.
2024-06-11 16:07:58,839 P169311 INFO Load best model: /home/ma-user/work/DNN_avazu_x1/Avazu/DNN_avazu_x1/avazu_x1_3fb65689/DNN_avazu_x1_001_3da2d674.model
2024-06-11 16:07:58,987 P169311 INFO ****** Validation evaluation ******
2024-06-11 16:08:13,177 P169311 INFO [Metrics] AUC: 0.746336 - logloss: 0.396527
2024-06-11 16:08:13,250 P169311 INFO ******** Test evaluation ********
2024-06-11 16:08:13,250 P169311 INFO Loading data...
2024-06-11 16:08:13,251 P169311 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2024-06-11 16:08:16,926 P169311 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2024-06-11 16:08:16,927 P169311 INFO Loading test data done.
2024-06-11 16:08:45,943 P169311 INFO [Metrics] AUC: 0.763827 - logloss: 0.367394
```
