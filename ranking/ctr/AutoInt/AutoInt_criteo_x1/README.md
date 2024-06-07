## AutoInt_criteo_x1

A hands-on guide to run the AutoInt model on the Criteo_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

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

We use [FuxiCTR-v1.2.2](https://github.com/reczoo/FuxiCTR/tree/v1.2.2) for this experiment. See the model code: [AutoInt](https://github.com/reczoo/FuxiCTR/blob/v1.2.2/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.2.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.2.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [fuxictr_version.py](./fuxictr_version.py#L3) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt_criteo_x1_tuner_config_04](./AutoInt_criteo_x1_tuner_config_04). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd AutoInt_criteo_x1
   nohup python run_expid.py --config ./AutoInt_criteo_x1_tuner_config_04 --expid AutoInt_criteo_x1_012_e71394ec --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

| AUC      | logloss  |
|:--------:|:--------:|
| 0.812593 | 0.439238 |

### Logs

```python
2022-06-27 15:19:40,988 P31042 INFO {
    "attention_dim": "256",
    "attention_layers": "3",
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "3",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "layer_norm": "True",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AutoInt",
    "model_id": "AutoInt_criteo_x1_012_e71394ec",
    "model_root": "./Criteo/AutoInt_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "4",
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
    "use_scale": "True",
    "use_wide": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-06-27 15:19:40,989 P31042 INFO Set up feature encoder...
2022-06-27 15:19:40,989 P31042 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-06-27 15:19:40,989 P31042 INFO Loading data...
2022-06-27 15:19:40,991 P31042 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-06-27 15:19:45,339 P31042 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-06-27 15:19:46,468 P31042 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-06-27 15:19:46,469 P31042 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-06-27 15:19:46,469 P31042 INFO Loading train data done.
2022-06-27 15:19:53,069 P31042 INFO Total number of parameters: 23364453.
2022-06-27 15:19:53,069 P31042 INFO Start training: 8058 batches/epoch
2022-06-27 15:19:53,069 P31042 INFO ************ Epoch=1 start ************
2022-06-27 16:30:52,622 P31042 INFO [Metrics] AUC: 0.800618 - logloss: 0.450328
2022-06-27 16:30:52,624 P31042 INFO Save best model: monitor(max): 0.800618
2022-06-27 16:30:52,911 P31042 INFO --- 8058/8058 batches finished ---
2022-06-27 16:30:52,950 P31042 INFO Train loss: 0.468593
2022-06-27 16:30:52,950 P31042 INFO ************ Epoch=1 end ************
2022-06-27 17:41:28,653 P31042 INFO [Metrics] AUC: 0.803271 - logloss: 0.448144
2022-06-27 17:41:28,654 P31042 INFO Save best model: monitor(max): 0.803271
2022-06-27 17:41:28,803 P31042 INFO --- 8058/8058 batches finished ---
2022-06-27 17:41:28,853 P31042 INFO Train loss: 0.463523
2022-06-27 17:41:28,853 P31042 INFO ************ Epoch=2 end ************
2022-06-27 18:52:33,102 P31042 INFO [Metrics] AUC: 0.804691 - logloss: 0.446762
2022-06-27 18:52:33,103 P31042 INFO Save best model: monitor(max): 0.804691
2022-06-27 18:52:33,217 P31042 INFO --- 8058/8058 batches finished ---
2022-06-27 18:52:33,258 P31042 INFO Train loss: 0.461980
2022-06-27 18:52:33,259 P31042 INFO ************ Epoch=3 end ************
2022-06-27 20:03:39,556 P31042 INFO [Metrics] AUC: 0.805641 - logloss: 0.445644
2022-06-27 20:03:39,557 P31042 INFO Save best model: monitor(max): 0.805641
2022-06-27 20:03:39,669 P31042 INFO --- 8058/8058 batches finished ---
2022-06-27 20:03:39,721 P31042 INFO Train loss: 0.461438
2022-06-27 20:03:39,721 P31042 INFO ************ Epoch=4 end ************
2022-06-27 21:14:48,058 P31042 INFO [Metrics] AUC: 0.805922 - logloss: 0.445489
2022-06-27 21:14:48,059 P31042 INFO Save best model: monitor(max): 0.805922
2022-06-27 21:14:48,171 P31042 INFO --- 8058/8058 batches finished ---
2022-06-27 21:14:48,211 P31042 INFO Train loss: 0.461560
2022-06-27 21:14:48,211 P31042 INFO ************ Epoch=5 end ************
2022-06-27 22:25:45,397 P31042 INFO [Metrics] AUC: 0.806306 - logloss: 0.445055
2022-06-27 22:25:45,399 P31042 INFO Save best model: monitor(max): 0.806306
2022-06-27 22:25:45,508 P31042 INFO --- 8058/8058 batches finished ---
2022-06-27 22:25:45,552 P31042 INFO Train loss: 0.462317
2022-06-27 22:25:45,552 P31042 INFO ************ Epoch=6 end ************
2022-06-27 23:36:49,378 P31042 INFO [Metrics] AUC: 0.806968 - logloss: 0.444379
2022-06-27 23:36:49,379 P31042 INFO Save best model: monitor(max): 0.806968
2022-06-27 23:36:49,488 P31042 INFO --- 8058/8058 batches finished ---
2022-06-27 23:36:49,539 P31042 INFO Train loss: 0.463473
2022-06-27 23:36:49,539 P31042 INFO ************ Epoch=7 end ************
2022-06-28 00:47:48,101 P31042 INFO [Metrics] AUC: 0.807107 - logloss: 0.444366
2022-06-28 00:47:48,102 P31042 INFO Save best model: monitor(max): 0.807107
2022-06-28 00:47:48,234 P31042 INFO --- 8058/8058 batches finished ---
2022-06-28 00:47:48,282 P31042 INFO Train loss: 0.464622
2022-06-28 00:47:48,282 P31042 INFO ************ Epoch=8 end ************
2022-06-28 01:58:50,287 P31042 INFO [Metrics] AUC: 0.807415 - logloss: 0.443980
2022-06-28 01:58:50,289 P31042 INFO Save best model: monitor(max): 0.807415
2022-06-28 01:58:50,399 P31042 INFO --- 8058/8058 batches finished ---
2022-06-28 01:58:50,455 P31042 INFO Train loss: 0.465313
2022-06-28 01:58:50,455 P31042 INFO ************ Epoch=9 end ************
2022-06-28 03:09:52,083 P31042 INFO [Metrics] AUC: 0.807325 - logloss: 0.444099
2022-06-28 03:09:52,084 P31042 INFO Monitor(max) STOP: 0.807325 !
2022-06-28 03:09:52,084 P31042 INFO Reduce learning rate on plateau: 0.000100
2022-06-28 03:09:52,084 P31042 INFO --- 8058/8058 batches finished ---
2022-06-28 03:09:52,134 P31042 INFO Train loss: 0.465634
2022-06-28 03:09:52,134 P31042 INFO ************ Epoch=10 end ************
2022-06-28 04:20:52,710 P31042 INFO [Metrics] AUC: 0.812125 - logloss: 0.439700
2022-06-28 04:20:52,711 P31042 INFO Save best model: monitor(max): 0.812125
2022-06-28 04:20:52,809 P31042 INFO --- 8058/8058 batches finished ---
2022-06-28 04:20:52,851 P31042 INFO Train loss: 0.446201
2022-06-28 04:20:52,851 P31042 INFO ************ Epoch=11 end ************
2022-06-28 05:31:48,055 P31042 INFO [Metrics] AUC: 0.812279 - logloss: 0.439670
2022-06-28 05:31:48,056 P31042 INFO Save best model: monitor(max): 0.812279
2022-06-28 05:31:48,156 P31042 INFO --- 8058/8058 batches finished ---
2022-06-28 05:31:48,211 P31042 INFO Train loss: 0.436354
2022-06-28 05:31:48,211 P31042 INFO ************ Epoch=12 end ************
2022-06-28 06:42:36,333 P31042 INFO [Metrics] AUC: 0.811213 - logloss: 0.440935
2022-06-28 06:42:36,334 P31042 INFO Monitor(max) STOP: 0.811213 !
2022-06-28 06:42:36,334 P31042 INFO Reduce learning rate on plateau: 0.000010
2022-06-28 06:42:36,334 P31042 INFO --- 8058/8058 batches finished ---
2022-06-28 06:42:36,388 P31042 INFO Train loss: 0.431064
2022-06-28 06:42:36,389 P31042 INFO ************ Epoch=13 end ************
2022-06-28 07:53:36,846 P31042 INFO [Metrics] AUC: 0.802108 - logloss: 0.455672
2022-06-28 07:53:36,847 P31042 INFO Monitor(max) STOP: 0.802108 !
2022-06-28 07:53:36,847 P31042 INFO Reduce learning rate on plateau: 0.000001
2022-06-28 07:53:36,847 P31042 INFO Early stopping at epoch=14
2022-06-28 07:53:36,848 P31042 INFO --- 8058/8058 batches finished ---
2022-06-28 07:53:36,899 P31042 INFO Train loss: 0.413014
2022-06-28 07:53:36,899 P31042 INFO Training finished.
2022-06-28 07:53:36,899 P31042 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/AutoInt_criteo_x1/criteo_x1_7b681156/AutoInt_criteo_x1_012_e71394ec.model
2022-06-28 07:53:41,894 P31042 INFO ****** Validation evaluation ******
2022-06-28 07:56:45,439 P31042 INFO [Metrics] AUC: 0.812279 - logloss: 0.439670
2022-06-28 07:56:45,523 P31042 INFO ******** Test evaluation ********
2022-06-28 07:56:45,524 P31042 INFO Loading data...
2022-06-28 07:56:45,524 P31042 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-06-28 07:56:46,334 P31042 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-06-28 07:56:46,334 P31042 INFO Loading test data done.
2022-06-28 07:58:28,092 P31042 INFO [Metrics] AUC: 0.812593 - logloss: 0.439238
```

### Revision History

- [Version 1](https://github.com/openbenchmark/BARS/tree/88d3a0faa4565e975141ae89a52d35d3a8b56eda/ctr_prediction/benchmarks/AutoInt/AutoInt_criteo_x1#autoint_criteo_x1): deprecated due to bug fix [#30](https://github.com/reczoo/FuxiCTR/issues/30) of FuxiCTR.
