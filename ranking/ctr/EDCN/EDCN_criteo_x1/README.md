## EDCN_criteo_x1

A hands-on guide to run the EDCN model on the Criteo_x1 dataset.

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

We use [FuxiCTR-v1.2.2](https://github.com/reczoo/FuxiCTR/tree/v1.2.2) for this experiment. See the model code: [EDCN](https://github.com/reczoo/FuxiCTR/blob/v1.2.2/fuxictr/pytorch/models/EDCN.py).

Running steps:

1. Download [FuxiCTR-v1.2.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.2.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [fuxictr_version.py](./fuxictr_version.py#L3) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [EDCN_criteo_x1_tuner_config_02](./EDCN_criteo_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd EDCN_criteo_x1
   nohup python run_expid.py --config ./EDCN_criteo_x1_tuner_config_02 --expid EDCN_criteo_x1_004_4023a363 --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

| AUC      | logloss  |
|:--------:|:--------:|
| 0.814651 | 0.437262 |

### Logs

```python
2022-06-17 09:50:11,487 P53612 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "bridge_type": "hadamard_product",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "3",
    "hidden_activations": "ReLU",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "EDCN",
    "model_id": "EDCN_criteo_x1_004_4023a363",
    "model_root": "./Criteo/EDCN_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_cross_layers": "3",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "temperature": "20",
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_hdf5": "True",
    "use_regulation_module": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-06-17 09:50:11,488 P53612 INFO Set up feature encoder...
2022-06-17 09:50:11,488 P53612 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-06-17 09:50:11,488 P53612 INFO Loading data...
2022-06-17 09:50:11,490 P53612 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-06-17 09:50:17,399 P53612 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-06-17 09:50:18,866 P53612 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-06-17 09:50:18,866 P53612 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-06-17 09:50:18,866 P53612 INFO Loading train data done.
2022-06-17 09:50:24,730 P53612 INFO Total number of parameters: 21329055.
2022-06-17 09:50:24,730 P53612 INFO Start training: 8058 batches/epoch
2022-06-17 09:50:24,730 P53612 INFO ************ Epoch=1 start ************
2022-06-17 10:08:59,248 P53612 INFO [Metrics] AUC: 0.805956 - logloss: 0.445507
2022-06-17 10:08:59,250 P53612 INFO Save best model: monitor(max): 0.805956
2022-06-17 10:08:59,663 P53612 INFO --- 8058/8058 batches finished ---
2022-06-17 10:08:59,705 P53612 INFO Train loss: 0.460109
2022-06-17 10:08:59,705 P53612 INFO ************ Epoch=1 end ************
2022-06-17 10:27:34,345 P53612 INFO [Metrics] AUC: 0.807718 - logloss: 0.443811
2022-06-17 10:27:34,346 P53612 INFO Save best model: monitor(max): 0.807718
2022-06-17 10:27:34,453 P53612 INFO --- 8058/8058 batches finished ---
2022-06-17 10:27:34,498 P53612 INFO Train loss: 0.455034
2022-06-17 10:27:34,498 P53612 INFO ************ Epoch=2 end ************
2022-06-17 10:46:10,169 P53612 INFO [Metrics] AUC: 0.808682 - logloss: 0.442924
2022-06-17 10:46:10,170 P53612 INFO Save best model: monitor(max): 0.808682
2022-06-17 10:46:10,276 P53612 INFO --- 8058/8058 batches finished ---
2022-06-17 10:46:10,329 P53612 INFO Train loss: 0.453480
2022-06-17 10:46:10,329 P53612 INFO ************ Epoch=3 end ************
2022-06-17 11:04:43,926 P53612 INFO [Metrics] AUC: 0.809409 - logloss: 0.442175
2022-06-17 11:04:43,927 P53612 INFO Save best model: monitor(max): 0.809409
2022-06-17 11:04:44,035 P53612 INFO --- 8058/8058 batches finished ---
2022-06-17 11:04:44,078 P53612 INFO Train loss: 0.452680
2022-06-17 11:04:44,078 P53612 INFO ************ Epoch=4 end ************
2022-06-17 11:23:14,548 P53612 INFO [Metrics] AUC: 0.809658 - logloss: 0.441999
2022-06-17 11:23:14,549 P53612 INFO Save best model: monitor(max): 0.809658
2022-06-17 11:23:14,653 P53612 INFO --- 8058/8058 batches finished ---
2022-06-17 11:23:14,698 P53612 INFO Train loss: 0.452085
2022-06-17 11:23:14,698 P53612 INFO ************ Epoch=5 end ************
2022-06-17 11:41:43,266 P53612 INFO [Metrics] AUC: 0.809892 - logloss: 0.441795
2022-06-17 11:41:43,267 P53612 INFO Save best model: monitor(max): 0.809892
2022-06-17 11:41:43,375 P53612 INFO --- 8058/8058 batches finished ---
2022-06-17 11:41:43,429 P53612 INFO Train loss: 0.451689
2022-06-17 11:41:43,430 P53612 INFO ************ Epoch=6 end ************
2022-06-17 12:00:08,324 P53612 INFO [Metrics] AUC: 0.810233 - logloss: 0.441657
2022-06-17 12:00:08,325 P53612 INFO Save best model: monitor(max): 0.810233
2022-06-17 12:00:08,420 P53612 INFO --- 8058/8058 batches finished ---
2022-06-17 12:00:08,469 P53612 INFO Train loss: 0.451364
2022-06-17 12:00:08,469 P53612 INFO ************ Epoch=7 end ************
2022-06-17 12:18:30,234 P53612 INFO [Metrics] AUC: 0.810241 - logloss: 0.441401
2022-06-17 12:18:30,236 P53612 INFO Save best model: monitor(max): 0.810241
2022-06-17 12:18:30,343 P53612 INFO --- 8058/8058 batches finished ---
2022-06-17 12:18:30,386 P53612 INFO Train loss: 0.451130
2022-06-17 12:18:30,387 P53612 INFO ************ Epoch=8 end ************
2022-06-17 12:36:51,117 P53612 INFO [Metrics] AUC: 0.810450 - logloss: 0.441719
2022-06-17 12:36:51,118 P53612 INFO Save best model: monitor(max): 0.810450
2022-06-17 12:36:51,224 P53612 INFO --- 8058/8058 batches finished ---
2022-06-17 12:36:51,276 P53612 INFO Train loss: 0.450945
2022-06-17 12:36:51,277 P53612 INFO ************ Epoch=9 end ************
2022-06-17 12:55:08,526 P53612 INFO [Metrics] AUC: 0.810490 - logloss: 0.441231
2022-06-17 12:55:08,528 P53612 INFO Save best model: monitor(max): 0.810490
2022-06-17 12:55:08,623 P53612 INFO --- 8058/8058 batches finished ---
2022-06-17 12:55:08,679 P53612 INFO Train loss: 0.450770
2022-06-17 12:55:08,679 P53612 INFO ************ Epoch=10 end ************
2022-06-17 13:13:25,125 P53612 INFO [Metrics] AUC: 0.810576 - logloss: 0.441234
2022-06-17 13:13:25,127 P53612 INFO Save best model: monitor(max): 0.810576
2022-06-17 13:13:25,229 P53612 INFO --- 8058/8058 batches finished ---
2022-06-17 13:13:25,285 P53612 INFO Train loss: 0.450664
2022-06-17 13:13:25,285 P53612 INFO ************ Epoch=11 end ************
2022-06-17 13:31:37,130 P53612 INFO [Metrics] AUC: 0.810577 - logloss: 0.441988
2022-06-17 13:31:37,132 P53612 INFO Monitor(max) STOP: 0.810577 !
2022-06-17 13:31:37,132 P53612 INFO Reduce learning rate on plateau: 0.000100
2022-06-17 13:31:37,132 P53612 INFO --- 8058/8058 batches finished ---
2022-06-17 13:31:37,188 P53612 INFO Train loss: 0.450542
2022-06-17 13:31:37,188 P53612 INFO ************ Epoch=12 end ************
2022-06-17 13:49:48,605 P53612 INFO [Metrics] AUC: 0.813770 - logloss: 0.438249
2022-06-17 13:49:48,606 P53612 INFO Save best model: monitor(max): 0.813770
2022-06-17 13:49:48,706 P53612 INFO --- 8058/8058 batches finished ---
2022-06-17 13:49:48,757 P53612 INFO Train loss: 0.440475
2022-06-17 13:49:48,758 P53612 INFO ************ Epoch=13 end ************
2022-06-17 14:07:58,967 P53612 INFO [Metrics] AUC: 0.814210 - logloss: 0.437845
2022-06-17 14:07:58,969 P53612 INFO Save best model: monitor(max): 0.814210
2022-06-17 14:07:59,068 P53612 INFO --- 8058/8058 batches finished ---
2022-06-17 14:07:59,115 P53612 INFO Train loss: 0.436322
2022-06-17 14:07:59,115 P53612 INFO ************ Epoch=14 end ************
2022-06-17 14:26:06,190 P53612 INFO [Metrics] AUC: 0.814187 - logloss: 0.437984
2022-06-17 14:26:06,191 P53612 INFO Monitor(max) STOP: 0.814187 !
2022-06-17 14:26:06,192 P53612 INFO Reduce learning rate on plateau: 0.000010
2022-06-17 14:26:06,192 P53612 INFO --- 8058/8058 batches finished ---
2022-06-17 14:26:06,236 P53612 INFO Train loss: 0.434243
2022-06-17 14:26:06,237 P53612 INFO ************ Epoch=15 end ************
2022-06-17 14:44:08,508 P53612 INFO [Metrics] AUC: 0.813601 - logloss: 0.438781
2022-06-17 14:44:08,510 P53612 INFO Monitor(max) STOP: 0.813601 !
2022-06-17 14:44:08,510 P53612 INFO Reduce learning rate on plateau: 0.000001
2022-06-17 14:44:08,510 P53612 INFO Early stopping at epoch=16
2022-06-17 14:44:08,510 P53612 INFO --- 8058/8058 batches finished ---
2022-06-17 14:44:08,556 P53612 INFO Train loss: 0.429768
2022-06-17 14:44:08,556 P53612 INFO Training finished.
2022-06-17 14:44:08,556 P53612 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/EDCN_criteo_x1/criteo_x1_7b681156/EDCN_criteo_x1_004_4023a363.model
2022-06-17 14:44:11,462 P53612 INFO ****** Validation evaluation ******
2022-06-17 14:44:39,728 P53612 INFO [Metrics] AUC: 0.814210 - logloss: 0.437845
2022-06-17 14:44:39,809 P53612 INFO ******** Test evaluation ********
2022-06-17 14:44:39,810 P53612 INFO Loading data...
2022-06-17 14:44:39,810 P53612 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-06-17 14:44:40,639 P53612 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-06-17 14:44:40,640 P53612 INFO Loading test data done.
2022-06-17 14:44:57,176 P53612 INFO [Metrics] AUC: 0.814651 - logloss: 0.437262
```

### Revision History

- [Version 1](https://github.com/openbenchmark/BARS/tree/88d3a0faa4565e975141ae89a52d35d3a8b56eda/ctr_prediction/benchmarks/EDCN/EDCN_criteo_x1): deprecated due to bug fix [#29](https://github.com/reczoo/FuxiCTR/issues/29) of FuxiCTR.
