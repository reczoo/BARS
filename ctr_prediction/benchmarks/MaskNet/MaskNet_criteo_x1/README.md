## MaskNet_criteo_x1

A hands-on guide to run the MaskNet model on the Criteo_x1 dataset.

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
  fuxictr: 1.1.1

  ```

### Dataset
Dataset ID: [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.1](fuxictr_url) for this experiment. See model code: [MaskNet](https://github.com/xue-pai/FuxiCTR/blob/v1.1.1/fuxictr/pytorch/models/MaskNet.py).

Running steps:

1. Download [FuxiCTR-v1.1.1](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [MaskNet_criteo_x1_tuner_config_03](./MaskNet_criteo_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd MaskNet_criteo_x1
    nohup python run_expid.py --config ./MaskNet_criteo_x1_tuner_config_03 --expid MaskNet_criteo_x1_002_5b7c4825 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.813913 | 0.437966  |
| 2 | 0.813825 | 0.438112  |
| 3 | 0.813747 | 0.438105  |
| 4 | 0.813554 | 0.438206  |
| 5 | 0.813860 | 0.438186  |
| | | | 
| Avg | 0.813780 | 0.438115 |
| Std | &#177;0.00012512 | &#177;0.00008442 |


### Logs
```python
2022-01-30 10:04:25,386 P53604 INFO {
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "dnn_hidden_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "emb_layernorm": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "MaskNet",
    "model_id": "MaskNet_criteo_x1_002_5b7c4825",
    "model_root": "./Criteo/MaskNet_criteo_x1/",
    "model_type": "SerialMaskNet",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_layernorm": "True",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_block_dim": "64",
    "parallel_num_blocks": "1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "0.1",
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
2022-01-30 10:04:25,387 P53604 INFO Set up feature encoder...
2022-01-30 10:04:25,387 P53604 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-01-30 10:04:25,387 P53604 INFO Loading data...
2022-01-30 10:04:25,388 P53604 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-01-30 10:04:29,718 P53604 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-01-30 10:04:30,845 P53604 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-01-30 10:04:30,845 P53604 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-01-30 10:04:30,845 P53604 INFO Loading train data done.
2022-01-30 10:04:36,273 P53604 INFO Total number of parameters: 21436890.
2022-01-30 10:04:36,274 P53604 INFO Start training: 8058 batches/epoch
2022-01-30 10:04:36,274 P53604 INFO ************ Epoch=1 start ************
2022-01-30 10:10:59,963 P53604 INFO [Metrics] AUC: 0.803839 - logloss: 0.447499
2022-01-30 10:10:59,964 P53604 INFO Save best model: monitor(max): 0.803839
2022-01-30 10:11:00,180 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 10:11:00,222 P53604 INFO Train loss: 0.462620
2022-01-30 10:11:00,222 P53604 INFO ************ Epoch=1 end ************
2022-01-30 10:17:25,255 P53604 INFO [Metrics] AUC: 0.806456 - logloss: 0.445613
2022-01-30 10:17:25,256 P53604 INFO Save best model: monitor(max): 0.806456
2022-01-30 10:17:25,352 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 10:17:25,396 P53604 INFO Train loss: 0.456211
2022-01-30 10:17:25,396 P53604 INFO ************ Epoch=2 end ************
2022-01-30 10:23:49,518 P53604 INFO [Metrics] AUC: 0.807762 - logloss: 0.443785
2022-01-30 10:23:49,520 P53604 INFO Save best model: monitor(max): 0.807762
2022-01-30 10:23:49,611 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 10:23:49,657 P53604 INFO Train loss: 0.454430
2022-01-30 10:23:49,658 P53604 INFO ************ Epoch=3 end ************
2022-01-30 10:30:12,585 P53604 INFO [Metrics] AUC: 0.808352 - logloss: 0.443414
2022-01-30 10:30:12,587 P53604 INFO Save best model: monitor(max): 0.808352
2022-01-30 10:30:12,689 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 10:30:12,736 P53604 INFO Train loss: 0.453625
2022-01-30 10:30:12,736 P53604 INFO ************ Epoch=4 end ************
2022-01-30 10:36:32,956 P53604 INFO [Metrics] AUC: 0.808693 - logloss: 0.443041
2022-01-30 10:36:32,957 P53604 INFO Save best model: monitor(max): 0.808693
2022-01-30 10:36:33,054 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 10:36:33,102 P53604 INFO Train loss: 0.453107
2022-01-30 10:36:33,103 P53604 INFO ************ Epoch=5 end ************
2022-01-30 10:42:56,993 P53604 INFO [Metrics] AUC: 0.809081 - logloss: 0.442502
2022-01-30 10:42:56,995 P53604 INFO Save best model: monitor(max): 0.809081
2022-01-30 10:42:57,087 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 10:42:57,144 P53604 INFO Train loss: 0.452748
2022-01-30 10:42:57,144 P53604 INFO ************ Epoch=6 end ************
2022-01-30 10:49:20,546 P53604 INFO [Metrics] AUC: 0.809340 - logloss: 0.442310
2022-01-30 10:49:20,547 P53604 INFO Save best model: monitor(max): 0.809340
2022-01-30 10:49:20,641 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 10:49:20,679 P53604 INFO Train loss: 0.452529
2022-01-30 10:49:20,679 P53604 INFO ************ Epoch=7 end ************
2022-01-30 10:55:29,776 P53604 INFO [Metrics] AUC: 0.809359 - logloss: 0.442508
2022-01-30 10:55:29,778 P53604 INFO Save best model: monitor(max): 0.809359
2022-01-30 10:55:29,863 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 10:55:29,899 P53604 INFO Train loss: 0.452285
2022-01-30 10:55:29,900 P53604 INFO ************ Epoch=8 end ************
2022-01-30 11:01:38,929 P53604 INFO [Metrics] AUC: 0.809652 - logloss: 0.442048
2022-01-30 11:01:38,930 P53604 INFO Save best model: monitor(max): 0.809652
2022-01-30 11:01:39,026 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 11:01:39,064 P53604 INFO Train loss: 0.452148
2022-01-30 11:01:39,064 P53604 INFO ************ Epoch=9 end ************
2022-01-30 11:07:48,157 P53604 INFO [Metrics] AUC: 0.809733 - logloss: 0.442060
2022-01-30 11:07:48,158 P53604 INFO Save best model: monitor(max): 0.809733
2022-01-30 11:07:48,254 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 11:07:48,298 P53604 INFO Train loss: 0.451988
2022-01-30 11:07:48,298 P53604 INFO ************ Epoch=10 end ************
2022-01-30 11:13:57,565 P53604 INFO [Metrics] AUC: 0.809892 - logloss: 0.441811
2022-01-30 11:13:57,567 P53604 INFO Save best model: monitor(max): 0.809892
2022-01-30 11:13:57,670 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 11:13:57,706 P53604 INFO Train loss: 0.451884
2022-01-30 11:13:57,707 P53604 INFO ************ Epoch=11 end ************
2022-01-30 11:20:06,846 P53604 INFO [Metrics] AUC: 0.809964 - logloss: 0.441645
2022-01-30 11:20:06,848 P53604 INFO Save best model: monitor(max): 0.809964
2022-01-30 11:20:06,940 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 11:20:06,977 P53604 INFO Train loss: 0.451735
2022-01-30 11:20:06,977 P53604 INFO ************ Epoch=12 end ************
2022-01-30 11:26:18,871 P53604 INFO [Metrics] AUC: 0.810038 - logloss: 0.441772
2022-01-30 11:26:18,872 P53604 INFO Save best model: monitor(max): 0.810038
2022-01-30 11:26:18,961 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 11:26:19,000 P53604 INFO Train loss: 0.451645
2022-01-30 11:26:19,000 P53604 INFO ************ Epoch=13 end ************
2022-01-30 11:32:35,849 P53604 INFO [Metrics] AUC: 0.810114 - logloss: 0.441712
2022-01-30 11:32:35,851 P53604 INFO Save best model: monitor(max): 0.810114
2022-01-30 11:32:35,948 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 11:32:35,983 P53604 INFO Train loss: 0.451532
2022-01-30 11:32:35,984 P53604 INFO ************ Epoch=14 end ************
2022-01-30 11:38:40,936 P53604 INFO [Metrics] AUC: 0.810206 - logloss: 0.441440
2022-01-30 11:38:40,938 P53604 INFO Save best model: monitor(max): 0.810206
2022-01-30 11:38:41,032 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 11:38:41,076 P53604 INFO Train loss: 0.451455
2022-01-30 11:38:41,076 P53604 INFO ************ Epoch=15 end ************
2022-01-30 11:44:46,375 P53604 INFO [Metrics] AUC: 0.810152 - logloss: 0.441534
2022-01-30 11:44:46,376 P53604 INFO Monitor(max) STOP: 0.810152 !
2022-01-30 11:44:46,377 P53604 INFO Reduce learning rate on plateau: 0.000100
2022-01-30 11:44:46,377 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 11:44:46,413 P53604 INFO Train loss: 0.451365
2022-01-30 11:44:46,413 P53604 INFO ************ Epoch=16 end ************
2022-01-30 11:50:48,110 P53604 INFO [Metrics] AUC: 0.813148 - logloss: 0.438872
2022-01-30 11:50:48,112 P53604 INFO Save best model: monitor(max): 0.813148
2022-01-30 11:50:48,199 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 11:50:48,235 P53604 INFO Train loss: 0.440674
2022-01-30 11:50:48,235 P53604 INFO ************ Epoch=17 end ************
2022-01-30 11:56:52,371 P53604 INFO [Metrics] AUC: 0.813605 - logloss: 0.438420
2022-01-30 11:56:52,373 P53604 INFO Save best model: monitor(max): 0.813605
2022-01-30 11:56:52,477 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 11:56:52,522 P53604 INFO Train loss: 0.436501
2022-01-30 11:56:52,522 P53604 INFO ************ Epoch=18 end ************
2022-01-30 12:02:57,897 P53604 INFO [Metrics] AUC: 0.813599 - logloss: 0.438398
2022-01-30 12:02:57,898 P53604 INFO Monitor(max) STOP: 0.813599 !
2022-01-30 12:02:57,899 P53604 INFO Reduce learning rate on plateau: 0.000010
2022-01-30 12:02:57,899 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 12:02:57,934 P53604 INFO Train loss: 0.434540
2022-01-30 12:02:57,935 P53604 INFO ************ Epoch=19 end ************
2022-01-30 12:09:01,341 P53604 INFO [Metrics] AUC: 0.813166 - logloss: 0.439077
2022-01-30 12:09:01,343 P53604 INFO Monitor(max) STOP: 0.813166 !
2022-01-30 12:09:01,343 P53604 INFO Reduce learning rate on plateau: 0.000001
2022-01-30 12:09:01,343 P53604 INFO Early stopping at epoch=20
2022-01-30 12:09:01,343 P53604 INFO --- 8058/8058 batches finished ---
2022-01-30 12:09:01,387 P53604 INFO Train loss: 0.430342
2022-01-30 12:09:01,387 P53604 INFO Training finished.
2022-01-30 12:09:01,387 P53604 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/MaskNet_criteo_x1/criteo_x1_7b681156/MaskNet_criteo_x1_002_5b7c4825.model
2022-01-30 12:09:04,350 P53604 INFO ****** Validation evaluation ******
2022-01-30 12:09:28,189 P53604 INFO [Metrics] AUC: 0.813605 - logloss: 0.438420
2022-01-30 12:09:28,284 P53604 INFO ******** Test evaluation ********
2022-01-30 12:09:28,285 P53604 INFO Loading data...
2022-01-30 12:09:28,285 P53604 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-01-30 12:09:29,053 P53604 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-01-30 12:09:29,054 P53604 INFO Loading test data done.
2022-01-30 12:09:42,951 P53604 INFO [Metrics] AUC: 0.813913 - logloss: 0.437966

```
