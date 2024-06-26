## AFN_criteo_x1

A hands-on guide to run the AFN model on the Criteo_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [AFN](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/AFN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFN_criteo_x1_tuner_config_03](./AFN_criteo_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFN_criteo_x1
    nohup python run_expid.py --config ./AFN_criteo_x1_tuner_config_03 --expid AFN_criteo_x1_002_d1191676 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.810686 | 0.441350  |


### Logs
```python
2022-01-22 09:17:05,398 P811 INFO {
    "afn_activations": "relu",
    "afn_dropout": "0.5",
    "afn_hidden_units": "[800]",
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_dropout": "0",
    "dnn_hidden_units": "[]",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "ensemble_dnn": "False",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "logarithmic_neurons": "400",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AFN",
    "model_id": "AFN_criteo_x1_002_d1191676",
    "model_root": "./Criteo/AFN_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
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
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-22 09:17:05,398 P811 INFO Set up feature encoder...
2022-01-22 09:17:05,398 P811 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-01-22 09:17:05,399 P811 INFO Loading data...
2022-01-22 09:17:05,400 P811 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-01-22 09:17:10,076 P811 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-01-22 09:17:11,324 P811 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-01-22 09:17:11,325 P811 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-01-22 09:17:11,325 P811 INFO Loading train data done.
2022-01-22 09:17:15,714 P811 INFO Total number of parameters: 24081239.
2022-01-22 09:17:15,714 P811 INFO Start training: 8058 batches/epoch
2022-01-22 09:17:15,714 P811 INFO ************ Epoch=1 start ************
2022-01-22 09:34:05,795 P811 INFO [Metrics] AUC: 0.796862 - logloss: 0.453591
2022-01-22 09:34:05,797 P811 INFO Save best model: monitor(max): 0.796862
2022-01-22 09:34:06,066 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 09:34:06,107 P811 INFO Train loss: 0.468207
2022-01-22 09:34:06,107 P811 INFO ************ Epoch=1 end ************
2022-01-22 09:50:35,927 P811 INFO [Metrics] AUC: 0.801121 - logloss: 0.449932
2022-01-22 09:50:35,928 P811 INFO Save best model: monitor(max): 0.801121
2022-01-22 09:50:36,041 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 09:50:36,083 P811 INFO Train loss: 0.459908
2022-01-22 09:50:36,084 P811 INFO ************ Epoch=2 end ************
2022-01-22 10:07:03,764 P811 INFO [Metrics] AUC: 0.802910 - logloss: 0.448275
2022-01-22 10:07:03,765 P811 INFO Save best model: monitor(max): 0.802910
2022-01-22 10:07:03,871 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 10:07:03,913 P811 INFO Train loss: 0.457386
2022-01-22 10:07:03,914 P811 INFO ************ Epoch=3 end ************
2022-01-22 10:23:28,038 P811 INFO [Metrics] AUC: 0.803810 - logloss: 0.447373
2022-01-22 10:23:28,039 P811 INFO Save best model: monitor(max): 0.803810
2022-01-22 10:23:28,146 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 10:23:28,189 P811 INFO Train loss: 0.455956
2022-01-22 10:23:28,189 P811 INFO ************ Epoch=4 end ************
2022-01-22 10:39:50,742 P811 INFO [Metrics] AUC: 0.804581 - logloss: 0.446991
2022-01-22 10:39:50,743 P811 INFO Save best model: monitor(max): 0.804581
2022-01-22 10:39:50,853 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 10:39:50,897 P811 INFO Train loss: 0.455028
2022-01-22 10:39:50,897 P811 INFO ************ Epoch=5 end ************
2022-01-22 10:56:16,646 P811 INFO [Metrics] AUC: 0.804998 - logloss: 0.446365
2022-01-22 10:56:16,647 P811 INFO Save best model: monitor(max): 0.804998
2022-01-22 10:56:16,746 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 10:56:16,791 P811 INFO Train loss: 0.454403
2022-01-22 10:56:16,792 P811 INFO ************ Epoch=6 end ************
2022-01-22 11:12:42,991 P811 INFO [Metrics] AUC: 0.805517 - logloss: 0.445833
2022-01-22 11:12:42,992 P811 INFO Save best model: monitor(max): 0.805517
2022-01-22 11:12:43,105 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 11:12:43,148 P811 INFO Train loss: 0.453871
2022-01-22 11:12:43,149 P811 INFO ************ Epoch=7 end ************
2022-01-22 11:29:09,805 P811 INFO [Metrics] AUC: 0.805772 - logloss: 0.445847
2022-01-22 11:29:09,806 P811 INFO Save best model: monitor(max): 0.805772
2022-01-22 11:29:09,914 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 11:29:09,957 P811 INFO Train loss: 0.453441
2022-01-22 11:29:09,958 P811 INFO ************ Epoch=8 end ************
2022-01-22 11:45:35,545 P811 INFO [Metrics] AUC: 0.806054 - logloss: 0.445382
2022-01-22 11:45:35,546 P811 INFO Save best model: monitor(max): 0.806054
2022-01-22 11:45:35,645 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 11:45:35,692 P811 INFO Train loss: 0.453063
2022-01-22 11:45:35,692 P811 INFO ************ Epoch=9 end ************
2022-01-22 12:02:00,961 P811 INFO [Metrics] AUC: 0.806225 - logloss: 0.445281
2022-01-22 12:02:00,962 P811 INFO Save best model: monitor(max): 0.806225
2022-01-22 12:02:01,062 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 12:02:01,108 P811 INFO Train loss: 0.452767
2022-01-22 12:02:01,109 P811 INFO ************ Epoch=10 end ************
2022-01-22 12:18:25,011 P811 INFO [Metrics] AUC: 0.806389 - logloss: 0.445027
2022-01-22 12:18:25,013 P811 INFO Save best model: monitor(max): 0.806389
2022-01-22 12:18:25,112 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 12:18:25,159 P811 INFO Train loss: 0.452488
2022-01-22 12:18:25,159 P811 INFO ************ Epoch=11 end ************
2022-01-22 12:34:47,748 P811 INFO [Metrics] AUC: 0.806472 - logloss: 0.444921
2022-01-22 12:34:47,749 P811 INFO Save best model: monitor(max): 0.806472
2022-01-22 12:34:47,878 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 12:34:47,929 P811 INFO Train loss: 0.452191
2022-01-22 12:34:47,929 P811 INFO ************ Epoch=12 end ************
2022-01-22 12:51:09,927 P811 INFO [Metrics] AUC: 0.806649 - logloss: 0.444718
2022-01-22 12:51:09,929 P811 INFO Save best model: monitor(max): 0.806649
2022-01-22 12:51:10,028 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 12:51:10,072 P811 INFO Train loss: 0.451917
2022-01-22 12:51:10,072 P811 INFO ************ Epoch=13 end ************
2022-01-22 13:07:32,606 P811 INFO [Metrics] AUC: 0.806722 - logloss: 0.445880
2022-01-22 13:07:32,608 P811 INFO Save best model: monitor(max): 0.806722
2022-01-22 13:07:32,705 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 13:07:32,751 P811 INFO Train loss: 0.451640
2022-01-22 13:07:32,751 P811 INFO ************ Epoch=14 end ************
2022-01-22 13:23:53,959 P811 INFO [Metrics] AUC: 0.806753 - logloss: 0.444920
2022-01-22 13:23:53,960 P811 INFO Save best model: monitor(max): 0.806753
2022-01-22 13:23:54,072 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 13:23:54,114 P811 INFO Train loss: 0.451406
2022-01-22 13:23:54,115 P811 INFO ************ Epoch=15 end ************
2022-01-22 13:40:16,634 P811 INFO [Metrics] AUC: 0.806806 - logloss: 0.444770
2022-01-22 13:40:16,636 P811 INFO Save best model: monitor(max): 0.806806
2022-01-22 13:40:16,753 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 13:40:16,800 P811 INFO Train loss: 0.451145
2022-01-22 13:40:16,800 P811 INFO ************ Epoch=16 end ************
2022-01-22 13:56:45,425 P811 INFO [Metrics] AUC: 0.806884 - logloss: 0.444709
2022-01-22 13:56:45,426 P811 INFO Save best model: monitor(max): 0.806884
2022-01-22 13:56:45,533 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 13:56:45,578 P811 INFO Train loss: 0.450932
2022-01-22 13:56:45,579 P811 INFO ************ Epoch=17 end ************
2022-01-22 14:13:06,899 P811 INFO [Metrics] AUC: 0.806873 - logloss: 0.445105
2022-01-22 14:13:06,901 P811 INFO Monitor(max) STOP: 0.806873 !
2022-01-22 14:13:06,901 P811 INFO Reduce learning rate on plateau: 0.000100
2022-01-22 14:13:06,901 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 14:13:06,946 P811 INFO Train loss: 0.450687
2022-01-22 14:13:06,947 P811 INFO ************ Epoch=18 end ************
2022-01-22 14:29:28,424 P811 INFO [Metrics] AUC: 0.809712 - logloss: 0.442240
2022-01-22 14:29:28,426 P811 INFO Save best model: monitor(max): 0.809712
2022-01-22 14:29:28,545 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 14:29:28,590 P811 INFO Train loss: 0.444379
2022-01-22 14:29:28,590 P811 INFO ************ Epoch=19 end ************
2022-01-22 14:45:50,316 P811 INFO [Metrics] AUC: 0.810028 - logloss: 0.441791
2022-01-22 14:45:50,317 P811 INFO Save best model: monitor(max): 0.810028
2022-01-22 14:45:50,414 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 14:45:50,459 P811 INFO Train loss: 0.441334
2022-01-22 14:45:50,459 P811 INFO ************ Epoch=20 end ************
2022-01-22 15:02:10,484 P811 INFO [Metrics] AUC: 0.810169 - logloss: 0.441738
2022-01-22 15:02:10,485 P811 INFO Save best model: monitor(max): 0.810169
2022-01-22 15:02:10,592 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 15:02:10,641 P811 INFO Train loss: 0.440012
2022-01-22 15:02:10,641 P811 INFO ************ Epoch=21 end ************
2022-01-22 15:18:28,823 P811 INFO [Metrics] AUC: 0.810172 - logloss: 0.441741
2022-01-22 15:18:28,825 P811 INFO Save best model: monitor(max): 0.810172
2022-01-22 15:18:28,931 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 15:18:28,975 P811 INFO Train loss: 0.439129
2022-01-22 15:18:28,975 P811 INFO ************ Epoch=22 end ************
2022-01-22 15:34:47,725 P811 INFO [Metrics] AUC: 0.810204 - logloss: 0.441807
2022-01-22 15:34:47,726 P811 INFO Save best model: monitor(max): 0.810204
2022-01-22 15:34:47,836 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 15:34:47,885 P811 INFO Train loss: 0.438501
2022-01-22 15:34:47,885 P811 INFO ************ Epoch=23 end ************
2022-01-22 15:46:42,823 P811 INFO [Metrics] AUC: 0.810222 - logloss: 0.441756
2022-01-22 15:46:42,825 P811 INFO Save best model: monitor(max): 0.810222
2022-01-22 15:46:42,927 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 15:46:42,974 P811 INFO Train loss: 0.437933
2022-01-22 15:46:42,974 P811 INFO ************ Epoch=24 end ************
2022-01-22 15:53:34,260 P811 INFO [Metrics] AUC: 0.810220 - logloss: 0.441708
2022-01-22 15:53:34,262 P811 INFO Monitor(max) STOP: 0.810220 !
2022-01-22 15:53:34,262 P811 INFO Reduce learning rate on plateau: 0.000010
2022-01-22 15:53:34,262 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 15:53:34,311 P811 INFO Train loss: 0.437463
2022-01-22 15:53:34,311 P811 INFO ************ Epoch=25 end ************
2022-01-22 16:00:25,540 P811 INFO [Metrics] AUC: 0.810314 - logloss: 0.441821
2022-01-22 16:00:25,541 P811 INFO Save best model: monitor(max): 0.810314
2022-01-22 16:00:25,652 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 16:00:25,702 P811 INFO Train loss: 0.435617
2022-01-22 16:00:25,702 P811 INFO ************ Epoch=26 end ************
2022-01-22 16:07:16,392 P811 INFO [Metrics] AUC: 0.810355 - logloss: 0.441789
2022-01-22 16:07:16,394 P811 INFO Save best model: monitor(max): 0.810355
2022-01-22 16:07:16,489 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 16:07:16,532 P811 INFO Train loss: 0.435178
2022-01-22 16:07:16,533 P811 INFO ************ Epoch=27 end ************
2022-01-22 16:14:09,058 P811 INFO [Metrics] AUC: 0.810330 - logloss: 0.441817
2022-01-22 16:14:09,060 P811 INFO Monitor(max) STOP: 0.810330 !
2022-01-22 16:14:09,060 P811 INFO Reduce learning rate on plateau: 0.000001
2022-01-22 16:14:09,060 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 16:14:09,107 P811 INFO Train loss: 0.434813
2022-01-22 16:14:09,107 P811 INFO ************ Epoch=28 end ************
2022-01-22 16:21:00,054 P811 INFO [Metrics] AUC: 0.810320 - logloss: 0.441873
2022-01-22 16:21:00,055 P811 INFO Monitor(max) STOP: 0.810320 !
2022-01-22 16:21:00,056 P811 INFO Reduce learning rate on plateau: 0.000001
2022-01-22 16:21:00,056 P811 INFO Early stopping at epoch=29
2022-01-22 16:21:00,056 P811 INFO --- 8058/8058 batches finished ---
2022-01-22 16:21:00,100 P811 INFO Train loss: 0.434368
2022-01-22 16:21:00,100 P811 INFO Training finished.
2022-01-22 16:21:00,100 P811 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/AFN_criteo_x1/criteo_x1_7b681156/AFN_criteo_x1_002_d1191676.model
2022-01-22 16:21:03,131 P811 INFO ****** Validation evaluation ******
2022-01-22 16:21:34,170 P811 INFO [Metrics] AUC: 0.810355 - logloss: 0.441789
2022-01-22 16:21:34,247 P811 INFO ******** Test evaluation ********
2022-01-22 16:21:34,247 P811 INFO Loading data...
2022-01-22 16:21:34,248 P811 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-01-22 16:21:35,049 P811 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-01-22 16:21:35,049 P811 INFO Loading test data done.
2022-01-22 16:21:52,688 P811 INFO [Metrics] AUC: 0.810686 - logloss: 0.441350

```
