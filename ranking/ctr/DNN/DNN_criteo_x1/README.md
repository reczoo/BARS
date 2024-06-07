## DNN_criteo_x1

A hands-on guide to run the DNN model on the Criteo_x1 dataset.

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
  fuxictr: 1.1.0
  ```

### Dataset

Dataset ID: [Criteo_x1](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DNN](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DNN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DNN_criteo_x1_tuner_config_seeds](./DNN_criteo_x1_tuner_config_seeds). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd DNN_criteo_x1
   nohup python run_expid.py --config ./DNN_criteo_x1_tuner_config_01 --expid DNN_criteo_x1_001_be50edb0 --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

Total 5 runs:

| Runs | AUC              | logloss          |
|:----:|:----------------:|:----------------:|
| 1    | 0.813668         | 0.438218         |
| 2    | 0.813701         | 0.438166         |
| 3    | 0.813791         | 0.438045         |
| 4    | 0.813601         | 0.438197         |
| 5    | 0.813630         | 0.438230         |
| Avg  | 0.813678         | 0.438171         |
| Std  | &#177;0.00006577 | &#177;0.00006673 |

### Logs

```python
2022-02-08 10:00:08,437 P789 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DNN",
    "model_id": "DNN_criteo_x1_001_be50edb0",
    "model_root": "./Criteo/DNN_criteo_x1/",
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
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-02-08 10:00:08,438 P789 INFO Set up feature encoder...
2022-02-08 10:00:08,438 P789 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-02-08 10:00:08,438 P789 INFO Loading data...
2022-02-08 10:00:08,440 P789 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-02-08 10:00:18,126 P789 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-02-08 10:00:19,240 P789 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-02-08 10:00:19,240 P789 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-02-08 10:00:19,240 P789 INFO Loading train data done.
2022-02-08 10:00:23,756 P789 INFO Total number of parameters: 21818362.
2022-02-08 10:00:23,756 P789 INFO Start training: 8058 batches/epoch
2022-02-08 10:00:23,756 P789 INFO ************ Epoch=1 start ************
2022-02-08 10:06:29,863 P789 INFO [Metrics] AUC: 0.804278 - logloss: 0.446923
2022-02-08 10:06:29,865 P789 INFO Save best model: monitor(max): 0.804278
2022-02-08 10:06:29,946 P789 INFO --- 8058/8058 batches finished ---
2022-02-08 10:06:29,999 P789 INFO Train loss: 0.461620
2022-02-08 10:06:29,999 P789 INFO ************ Epoch=1 end ************
2022-02-08 10:12:38,683 P789 INFO [Metrics] AUC: 0.806863 - logloss: 0.444837
2022-02-08 10:12:38,684 P789 INFO Save best model: monitor(max): 0.806863
2022-02-08 10:12:38,805 P789 INFO --- 8058/8058 batches finished ---
2022-02-08 10:12:38,860 P789 INFO Train loss: 0.456145
2022-02-08 10:12:38,860 P789 INFO ************ Epoch=2 end ************
2022-02-08 10:18:42,089 P789 INFO [Metrics] AUC: 0.807814 - logloss: 0.444021
2022-02-08 10:18:42,091 P789 INFO Save best model: monitor(max): 0.807814
2022-02-08 10:18:42,193 P789 INFO --- 8058/8058 batches finished ---
2022-02-08 10:18:42,251 P789 INFO Train loss: 0.454652
2022-02-08 10:18:42,252 P789 INFO ************ Epoch=3 end ************
2022-02-08 10:24:44,811 P789 INFO [Metrics] AUC: 0.808508 - logloss: 0.443226
2022-02-08 10:24:44,812 P789 INFO Save best model: monitor(max): 0.808508
2022-02-08 10:24:44,913 P789 INFO --- 8058/8058 batches finished ---
2022-02-08 10:24:44,964 P789 INFO Train loss: 0.453850
2022-02-08 10:24:44,964 P789 INFO ************ Epoch=4 end ************
2022-02-08 10:30:53,245 P789 INFO [Metrics] AUC: 0.808958 - logloss: 0.442642
2022-02-08 10:30:53,247 P789 INFO Save best model: monitor(max): 0.808958
2022-02-08 10:30:53,359 P789 INFO --- 8058/8058 batches finished ---
2022-02-08 10:30:53,408 P789 INFO Train loss: 0.453333
2022-02-08 10:30:53,408 P789 INFO ************ Epoch=5 end ************
2022-02-08 10:36:56,399 P789 INFO [Metrics] AUC: 0.809055 - logloss: 0.442935
2022-02-08 10:36:56,401 P789 INFO Save best model: monitor(max): 0.809055
2022-02-08 10:36:56,508 P789 INFO --- 8058/8058 batches finished ---
2022-02-08 10:36:56,549 P789 INFO Train loss: 0.452989
2022-02-08 10:36:56,549 P789 INFO ************ Epoch=6 end ************
2022-02-08 10:43:02,139 P789 INFO [Metrics] AUC: 0.809364 - logloss: 0.442379
2022-02-08 10:43:02,140 P789 INFO Save best model: monitor(max): 0.809364
2022-02-08 10:43:02,251 P789 INFO --- 8058/8058 batches finished ---
2022-02-08 10:43:02,290 P789 INFO Train loss: 0.452715
2022-02-08 10:43:02,290 P789 INFO ************ Epoch=7 end ************
2022-02-08 10:49:05,421 P789 INFO [Metrics] AUC: 0.809563 - logloss: 0.442125
2022-02-08 10:49:05,423 P789 INFO Save best model: monitor(max): 0.809563
2022-02-08 10:49:05,528 P789 INFO --- 8058/8058 batches finished ---
2022-02-08 10:49:05,567 P789 INFO Train loss: 0.452485
2022-02-08 10:49:05,567 P789 INFO ************ Epoch=8 end ************
2022-02-08 10:55:13,890 P789 INFO [Metrics] AUC: 0.809614 - logloss: 0.442066
2022-02-08 10:55:13,892 P789 INFO Save best model: monitor(max): 0.809614
2022-02-08 10:55:13,997 P789 INFO --- 8058/8058 batches finished ---
2022-02-08 10:55:14,036 P789 INFO Train loss: 0.452291
2022-02-08 10:55:14,036 P789 INFO ************ Epoch=9 end ************
2022-02-08 11:01:22,197 P789 INFO [Metrics] AUC: 0.809701 - logloss: 0.441935
2022-02-08 11:01:22,199 P789 INFO Save best model: monitor(max): 0.809701
2022-02-08 11:01:22,306 P789 INFO --- 8058/8058 batches finished ---
2022-02-08 11:01:22,347 P789 INFO Train loss: 0.452138
2022-02-08 11:01:22,347 P789 INFO ************ Epoch=10 end ************
2022-02-08 11:07:26,284 P789 INFO [Metrics] AUC: 0.809874 - logloss: 0.441828
2022-02-08 11:07:26,286 P789 INFO Save best model: monitor(max): 0.809874
2022-02-08 11:07:26,392 P789 INFO --- 8058/8058 batches finished ---
2022-02-08 11:07:26,431 P789 INFO Train loss: 0.452030
2022-02-08 11:07:26,431 P789 INFO ************ Epoch=11 end ************
2022-02-08 11:13:29,569 P789 INFO [Metrics] AUC: 0.810043 - logloss: 0.441682
2022-02-08 11:13:29,571 P789 INFO Save best model: monitor(max): 0.810043
2022-02-08 11:13:29,693 P789 INFO --- 8058/8058 batches finished ---
2022-02-08 11:13:29,733 P789 INFO Train loss: 0.451874
2022-02-08 11:13:29,734 P789 INFO ************ Epoch=12 end ************
2022-02-08 11:19:33,918 P789 INFO [Metrics] AUC: 0.810009 - logloss: 0.441946
2022-02-08 11:19:33,919 P789 INFO Monitor(max) STOP: 0.810009 !
2022-02-08 11:19:33,919 P789 INFO Reduce learning rate on plateau: 0.000100
2022-02-08 11:19:33,919 P789 INFO --- 8058/8058 batches finished ---
2022-02-08 11:19:33,958 P789 INFO Train loss: 0.451745
2022-02-08 11:19:33,958 P789 INFO ************ Epoch=13 end ************
2022-02-08 11:25:34,333 P789 INFO [Metrics] AUC: 0.812964 - logloss: 0.438972
2022-02-08 11:25:34,334 P789 INFO Save best model: monitor(max): 0.812964
2022-02-08 11:25:34,428 P789 INFO --- 8058/8058 batches finished ---
2022-02-08 11:25:34,467 P789 INFO Train loss: 0.440349
2022-02-08 11:25:34,467 P789 INFO ************ Epoch=14 end ************
2022-02-08 11:31:31,969 P789 INFO [Metrics] AUC: 0.813314 - logloss: 0.438640
2022-02-08 11:31:31,970 P789 INFO Save best model: monitor(max): 0.813314
2022-02-08 11:31:32,075 P789 INFO --- 8058/8058 batches finished ---
2022-02-08 11:31:32,113 P789 INFO Train loss: 0.435879
2022-02-08 11:31:32,114 P789 INFO ************ Epoch=15 end ************
2022-02-08 11:37:26,936 P789 INFO [Metrics] AUC: 0.813374 - logloss: 0.438688
2022-02-08 11:37:26,937 P789 INFO Save best model: monitor(max): 0.813374
2022-02-08 11:37:27,044 P789 INFO --- 8058/8058 batches finished ---
2022-02-08 11:37:27,085 P789 INFO Train loss: 0.433937
2022-02-08 11:37:27,086 P789 INFO ************ Epoch=16 end ************
2022-02-08 11:43:25,004 P789 INFO [Metrics] AUC: 0.813237 - logloss: 0.438824
2022-02-08 11:43:25,005 P789 INFO Monitor(max) STOP: 0.813237 !
2022-02-08 11:43:25,005 P789 INFO Reduce learning rate on plateau: 0.000010
2022-02-08 11:43:25,005 P789 INFO --- 8058/8058 batches finished ---
2022-02-08 11:43:25,044 P789 INFO Train loss: 0.432473
2022-02-08 11:43:25,045 P789 INFO ************ Epoch=17 end ************
2022-02-08 11:49:28,466 P789 INFO [Metrics] AUC: 0.812925 - logloss: 0.439370
2022-02-08 11:49:28,468 P789 INFO Monitor(max) STOP: 0.812925 !
2022-02-08 11:49:28,468 P789 INFO Reduce learning rate on plateau: 0.000001
2022-02-08 11:49:28,468 P789 INFO Early stopping at epoch=18
2022-02-08 11:49:28,468 P789 INFO --- 8058/8058 batches finished ---
2022-02-08 11:49:28,526 P789 INFO Train loss: 0.428365
2022-02-08 11:49:28,526 P789 INFO Training finished.
2022-02-08 11:49:28,526 P789 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/DNN_criteo_x1/criteo_x1_7b681156/DNN_criteo_x1_001_be50edb0.model
2022-02-08 11:49:28,601 P789 INFO ****** Validation evaluation ******
2022-02-08 11:49:54,593 P789 INFO [Metrics] AUC: 0.813374 - logloss: 0.438688
2022-02-08 11:49:54,697 P789 INFO ******** Test evaluation ********
2022-02-08 11:49:54,697 P789 INFO Loading data...
2022-02-08 11:49:54,700 P789 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-02-08 11:49:55,571 P789 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-02-08 11:49:55,571 P789 INFO Loading test data done.
2022-02-08 11:50:10,731 P789 INFO [Metrics] AUC: 0.813668 - logloss: 0.438218
```
