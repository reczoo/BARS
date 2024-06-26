## CrossNet_criteo_x4_001

A hands-on guide to run the DCN model on the Criteo_x4_001 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

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
Dataset ID: [Criteo_x4](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x4). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [DCN](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DCN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CrossNet_criteo_x4_tuner_config_12](./CrossNet_criteo_x4_tuner_config_12). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CrossNet_criteo_x4_001
    nohup python run_expid.py --config ./CrossNet_criteo_x4_tuner_config_12 --expid DCN_criteo_x4_001_964751c8 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.445639 | 0.805964  |


### Logs
```python
2022-03-01 20:39:38,719 P40465 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "crossing_layers": "8",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_9ea3bdfc",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'na_value': 0, 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'preprocess': 'convert_to_bucket', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'na_value': '', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'Label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "DCN",
    "model_id": "DCN_criteo_x4_001_964751c8",
    "model_root": "./Criteo/CrossNet_criteo_x4_001/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x4/test.csv",
    "train_data": "../data/Criteo/Criteo_x4/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x4/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2022-03-01 20:39:38,719 P40465 INFO Set up feature encoder...
2022-03-01 20:39:38,720 P40465 INFO Reading file: ../data/Criteo/Criteo_x4/train.csv
2022-03-01 20:43:32,897 P40465 INFO Preprocess feature columns...
2022-03-01 21:03:36,949 P40465 INFO Fit feature encoder...
2022-03-01 21:03:36,949 P40465 INFO Processing column: {'active': True, 'dtype': 'float', 'na_value': 0, 'name': 'I1', 'preprocess': 'convert_to_bucket', 'type': 'categorical'}
2022-03-01 21:03:52,925 P40465 INFO Processing column: {'active': True, 'dtype': 'float', 'na_value': 0, 'name': 'I2', 'preprocess': 'convert_to_bucket', 'type': 'categorical'}
2022-03-01 21:04:11,194 P40465 INFO Processing column: {'active': True, 'dtype': 'float', 'na_value': 0, 'name': 'I3', 'preprocess': 'convert_to_bucket', 'type': 'categorical'}
2022-03-01 21:04:29,823 P40465 INFO Processing column: {'active': True, 'dtype': 'float', 'na_value': 0, 'name': 'I4', 'preprocess': 'convert_to_bucket', 'type': 'categorical'}
2022-03-01 21:04:48,005 P40465 INFO Processing column: {'active': True, 'dtype': 'float', 'na_value': 0, 'name': 'I5', 'preprocess': 'convert_to_bucket', 'type': 'categorical'}
2022-03-01 21:05:07,274 P40465 INFO Processing column: {'active': True, 'dtype': 'float', 'na_value': 0, 'name': 'I6', 'preprocess': 'convert_to_bucket', 'type': 'categorical'}
2022-03-01 21:05:26,158 P40465 INFO Processing column: {'active': True, 'dtype': 'float', 'na_value': 0, 'name': 'I7', 'preprocess': 'convert_to_bucket', 'type': 'categorical'}
2022-03-01 21:05:44,665 P40465 INFO Processing column: {'active': True, 'dtype': 'float', 'na_value': 0, 'name': 'I8', 'preprocess': 'convert_to_bucket', 'type': 'categorical'}
2022-03-01 21:06:03,395 P40465 INFO Processing column: {'active': True, 'dtype': 'float', 'na_value': 0, 'name': 'I9', 'preprocess': 'convert_to_bucket', 'type': 'categorical'}
2022-03-01 21:06:22,767 P40465 INFO Processing column: {'active': True, 'dtype': 'float', 'na_value': 0, 'name': 'I10', 'preprocess': 'convert_to_bucket', 'type': 'categorical'}
2022-03-01 21:06:39,136 P40465 INFO Processing column: {'active': True, 'dtype': 'float', 'na_value': 0, 'name': 'I11', 'preprocess': 'convert_to_bucket', 'type': 'categorical'}
2022-03-01 21:06:57,721 P40465 INFO Processing column: {'active': True, 'dtype': 'float', 'na_value': 0, 'name': 'I12', 'preprocess': 'convert_to_bucket', 'type': 'categorical'}
2022-03-01 21:07:12,899 P40465 INFO Processing column: {'active': True, 'dtype': 'float', 'na_value': 0, 'name': 'I13', 'preprocess': 'convert_to_bucket', 'type': 'categorical'}
2022-03-01 21:07:31,207 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C1', 'type': 'categorical'}
2022-03-01 21:07:37,029 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C2', 'type': 'categorical'}
2022-03-01 21:07:42,795 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C3', 'type': 'categorical'}
2022-03-01 21:07:59,425 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C4', 'type': 'categorical'}
2022-03-01 21:08:11,861 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C5', 'type': 'categorical'}
2022-03-01 21:08:17,853 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C6', 'type': 'categorical'}
2022-03-01 21:08:23,182 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C7', 'type': 'categorical'}
2022-03-01 21:08:32,593 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C8', 'type': 'categorical'}
2022-03-01 21:08:38,335 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C9', 'type': 'categorical'}
2022-03-01 21:08:43,673 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C10', 'type': 'categorical'}
2022-03-01 21:08:52,860 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C11', 'type': 'categorical'}
2022-03-01 21:09:00,280 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C12', 'type': 'categorical'}
2022-03-01 21:09:14,456 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C13', 'type': 'categorical'}
2022-03-01 21:09:21,752 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C14', 'type': 'categorical'}
2022-03-01 21:09:27,116 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C15', 'type': 'categorical'}
2022-03-01 21:09:34,770 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C16', 'type': 'categorical'}
2022-03-01 21:09:48,267 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C17', 'type': 'categorical'}
2022-03-01 21:09:53,810 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C18', 'type': 'categorical'}
2022-03-01 21:10:00,329 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C19', 'type': 'categorical'}
2022-03-01 21:10:05,098 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C20', 'type': 'categorical'}
2022-03-01 21:10:09,539 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C21', 'type': 'categorical'}
2022-03-01 21:10:23,463 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C22', 'type': 'categorical'}
2022-03-01 21:10:26,792 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C23', 'type': 'categorical'}
2022-03-01 21:10:32,079 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C24', 'type': 'categorical'}
2022-03-01 21:10:40,373 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C25', 'type': 'categorical'}
2022-03-01 21:10:44,918 P40465 INFO Processing column: {'active': True, 'dtype': 'str', 'na_value': '', 'name': 'C26', 'type': 'categorical'}
2022-03-01 21:10:51,308 P40465 INFO Set feature index...
2022-03-01 21:10:51,309 P40465 INFO Pickle feature_encode: ../data/Criteo/criteo_x4_9ea3bdfc/feature_encoder.pkl
2022-03-01 21:10:51,817 P40465 INFO Save feature_map to json: ../data/Criteo/criteo_x4_9ea3bdfc/feature_map.json
2022-03-01 21:10:51,818 P40465 INFO Set feature encoder done.
2022-03-01 21:11:04,646 P40465 INFO Total number of parameters: 14581937.
2022-03-01 21:11:04,646 P40465 INFO Loading data...
2022-03-01 21:11:04,649 P40465 INFO Reading file: ../data/Criteo/Criteo_x4/train.csv
2022-03-01 21:15:00,632 P40465 INFO Preprocess feature columns...
2022-03-01 21:35:20,846 P40465 INFO Transform feature columns...
2022-03-01 21:44:38,755 P40465 INFO Saving data to h5: ../data/Criteo/criteo_x4_9ea3bdfc/train.h5
2022-03-01 21:45:00,359 P40465 INFO Reading file: ../data/Criteo/Criteo_x4/valid.csv
2022-03-01 21:45:30,701 P40465 INFO Preprocess feature columns...
2022-03-01 21:47:58,452 P40465 INFO Transform feature columns...
2022-03-01 21:49:08,611 P40465 INFO Saving data to h5: ../data/Criteo/criteo_x4_9ea3bdfc/valid.h5
2022-03-01 21:49:16,125 P40465 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2022-03-01 21:49:16,466 P40465 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2022-03-01 21:49:16,466 P40465 INFO Loading train data done.
2022-03-01 21:49:19,011 P40465 INFO Start training: 3668 batches/epoch
2022-03-01 21:49:19,011 P40465 INFO ************ Epoch=1 start ************
2022-03-01 21:56:40,357 P40465 INFO [Metrics] logloss: 0.458623 - AUC: 0.791963
2022-03-01 21:56:40,358 P40465 INFO Save best model: monitor(max): 0.333339
2022-03-01 21:56:40,427 P40465 INFO --- 3668/3668 batches finished ---
2022-03-01 21:56:41,036 P40465 INFO Train loss: 0.466079
2022-03-01 21:56:41,037 P40465 INFO ************ Epoch=1 end ************
2022-03-01 22:04:05,724 P40465 INFO [Metrics] logloss: 0.452875 - AUC: 0.797554
2022-03-01 22:04:05,725 P40465 INFO Save best model: monitor(max): 0.344678
2022-03-01 22:04:05,827 P40465 INFO --- 3668/3668 batches finished ---
2022-03-01 22:04:06,447 P40465 INFO Train loss: 0.460098
2022-03-01 22:04:06,447 P40465 INFO ************ Epoch=2 end ************
2022-03-01 22:11:30,150 P40465 INFO [Metrics] logloss: 0.452030 - AUC: 0.798567
2022-03-01 22:11:30,152 P40465 INFO Save best model: monitor(max): 0.346537
2022-03-01 22:11:30,244 P40465 INFO --- 3668/3668 batches finished ---
2022-03-01 22:11:30,873 P40465 INFO Train loss: 0.457809
2022-03-01 22:11:30,873 P40465 INFO ************ Epoch=3 end ************
2022-03-01 22:18:55,243 P40465 INFO [Metrics] logloss: 0.451565 - AUC: 0.799338
2022-03-01 22:18:55,244 P40465 INFO Save best model: monitor(max): 0.347773
2022-03-01 22:18:55,339 P40465 INFO --- 3668/3668 batches finished ---
2022-03-01 22:18:55,954 P40465 INFO Train loss: 0.457191
2022-03-01 22:18:55,954 P40465 INFO ************ Epoch=4 end ************
2022-03-01 22:26:19,682 P40465 INFO [Metrics] logloss: 0.450932 - AUC: 0.799831
2022-03-01 22:26:19,683 P40465 INFO Save best model: monitor(max): 0.348899
2022-03-01 22:26:19,770 P40465 INFO --- 3668/3668 batches finished ---
2022-03-01 22:26:20,399 P40465 INFO Train loss: 0.456804
2022-03-01 22:26:20,399 P40465 INFO ************ Epoch=5 end ************
2022-03-01 22:33:43,145 P40465 INFO [Metrics] logloss: 0.450819 - AUC: 0.800032
2022-03-01 22:33:43,146 P40465 INFO Save best model: monitor(max): 0.349214
2022-03-01 22:33:43,236 P40465 INFO --- 3668/3668 batches finished ---
2022-03-01 22:33:43,932 P40465 INFO Train loss: 0.456499
2022-03-01 22:33:43,932 P40465 INFO ************ Epoch=6 end ************
2022-03-01 22:41:08,072 P40465 INFO [Metrics] logloss: 0.450549 - AUC: 0.800375
2022-03-01 22:41:08,073 P40465 INFO Save best model: monitor(max): 0.349826
2022-03-01 22:41:08,164 P40465 INFO --- 3668/3668 batches finished ---
2022-03-01 22:41:08,762 P40465 INFO Train loss: 0.456237
2022-03-01 22:41:08,763 P40465 INFO ************ Epoch=7 end ************
2022-03-01 22:48:31,640 P40465 INFO [Metrics] logloss: 0.450139 - AUC: 0.800702
2022-03-01 22:48:31,641 P40465 INFO Save best model: monitor(max): 0.350563
2022-03-01 22:48:31,733 P40465 INFO --- 3668/3668 batches finished ---
2022-03-01 22:48:32,286 P40465 INFO Train loss: 0.456039
2022-03-01 22:48:32,286 P40465 INFO ************ Epoch=8 end ************
2022-03-01 22:55:56,821 P40465 INFO [Metrics] logloss: 0.449953 - AUC: 0.800851
2022-03-01 22:55:56,822 P40465 INFO Save best model: monitor(max): 0.350898
2022-03-01 22:55:56,917 P40465 INFO --- 3668/3668 batches finished ---
2022-03-01 22:55:57,491 P40465 INFO Train loss: 0.455842
2022-03-01 22:55:57,491 P40465 INFO ************ Epoch=9 end ************
2022-03-01 23:03:22,260 P40465 INFO [Metrics] logloss: 0.449780 - AUC: 0.801153
2022-03-01 23:03:22,261 P40465 INFO Save best model: monitor(max): 0.351373
2022-03-01 23:03:22,362 P40465 INFO --- 3668/3668 batches finished ---
2022-03-01 23:03:22,936 P40465 INFO Train loss: 0.455671
2022-03-01 23:03:22,936 P40465 INFO ************ Epoch=10 end ************
2022-03-01 23:10:46,979 P40465 INFO [Metrics] logloss: 0.449574 - AUC: 0.801342
2022-03-01 23:10:46,980 P40465 INFO Save best model: monitor(max): 0.351768
2022-03-01 23:10:47,080 P40465 INFO --- 3668/3668 batches finished ---
2022-03-01 23:10:47,677 P40465 INFO Train loss: 0.455535
2022-03-01 23:10:47,677 P40465 INFO ************ Epoch=11 end ************
2022-03-01 23:18:11,638 P40465 INFO [Metrics] logloss: 0.449536 - AUC: 0.801534
2022-03-01 23:18:11,639 P40465 INFO Save best model: monitor(max): 0.351999
2022-03-01 23:18:11,738 P40465 INFO --- 3668/3668 batches finished ---
2022-03-01 23:18:12,308 P40465 INFO Train loss: 0.455417
2022-03-01 23:18:12,309 P40465 INFO ************ Epoch=12 end ************
2022-03-01 23:25:37,241 P40465 INFO [Metrics] logloss: 0.449451 - AUC: 0.801513
2022-03-01 23:25:37,242 P40465 INFO Save best model: monitor(max): 0.352062
2022-03-01 23:25:37,336 P40465 INFO --- 3668/3668 batches finished ---
2022-03-01 23:25:38,018 P40465 INFO Train loss: 0.455298
2022-03-01 23:25:38,018 P40465 INFO ************ Epoch=13 end ************
2022-03-01 23:33:02,008 P40465 INFO [Metrics] logloss: 0.449587 - AUC: 0.801680
2022-03-01 23:33:02,009 P40465 INFO Save best model: monitor(max): 0.352094
2022-03-01 23:33:02,109 P40465 INFO --- 3668/3668 batches finished ---
2022-03-01 23:33:02,797 P40465 INFO Train loss: 0.455194
2022-03-01 23:33:02,797 P40465 INFO ************ Epoch=14 end ************
2022-03-01 23:40:28,401 P40465 INFO [Metrics] logloss: 0.449055 - AUC: 0.801893
2022-03-01 23:40:28,403 P40465 INFO Save best model: monitor(max): 0.352838
2022-03-01 23:40:28,489 P40465 INFO --- 3668/3668 batches finished ---
2022-03-01 23:40:29,080 P40465 INFO Train loss: 0.455121
2022-03-01 23:40:29,080 P40465 INFO ************ Epoch=15 end ************
2022-03-01 23:47:52,042 P40465 INFO [Metrics] logloss: 0.449052 - AUC: 0.801876
2022-03-01 23:47:52,043 P40465 INFO Monitor(max) STOP: 0.352824 !
2022-03-01 23:47:52,043 P40465 INFO Reduce learning rate on plateau: 0.000100
2022-03-01 23:47:52,043 P40465 INFO --- 3668/3668 batches finished ---
2022-03-01 23:47:52,736 P40465 INFO Train loss: 0.455009
2022-03-01 23:47:52,736 P40465 INFO ************ Epoch=16 end ************
2022-03-01 23:55:17,820 P40465 INFO [Metrics] logloss: 0.446556 - AUC: 0.804710
2022-03-01 23:55:17,821 P40465 INFO Save best model: monitor(max): 0.358155
2022-03-01 23:55:17,942 P40465 INFO --- 3668/3668 batches finished ---
2022-03-01 23:55:18,557 P40465 INFO Train loss: 0.448190
2022-03-01 23:55:18,557 P40465 INFO ************ Epoch=17 end ************
2022-03-02 00:02:48,199 P40465 INFO [Metrics] logloss: 0.446166 - AUC: 0.805159
2022-03-02 00:02:48,200 P40465 INFO Save best model: monitor(max): 0.358993
2022-03-02 00:02:48,292 P40465 INFO --- 3668/3668 batches finished ---
2022-03-02 00:02:48,909 P40465 INFO Train loss: 0.445933
2022-03-02 00:02:48,909 P40465 INFO ************ Epoch=18 end ************
2022-03-02 00:10:16,836 P40465 INFO [Metrics] logloss: 0.446007 - AUC: 0.805366
2022-03-02 00:10:16,837 P40465 INFO Save best model: monitor(max): 0.359359
2022-03-02 00:10:16,926 P40465 INFO --- 3668/3668 batches finished ---
2022-03-02 00:10:17,527 P40465 INFO Train loss: 0.444899
2022-03-02 00:10:17,527 P40465 INFO ************ Epoch=19 end ************
2022-03-02 00:17:43,571 P40465 INFO [Metrics] logloss: 0.445961 - AUC: 0.805465
2022-03-02 00:17:43,572 P40465 INFO Save best model: monitor(max): 0.359504
2022-03-02 00:17:43,665 P40465 INFO --- 3668/3668 batches finished ---
2022-03-02 00:17:44,266 P40465 INFO Train loss: 0.444168
2022-03-02 00:17:44,266 P40465 INFO ************ Epoch=20 end ************
2022-03-02 00:25:10,100 P40465 INFO [Metrics] logloss: 0.445870 - AUC: 0.805548
2022-03-02 00:25:10,101 P40465 INFO Save best model: monitor(max): 0.359678
2022-03-02 00:25:10,252 P40465 INFO --- 3668/3668 batches finished ---
2022-03-02 00:25:10,836 P40465 INFO Train loss: 0.443557
2022-03-02 00:25:10,837 P40465 INFO ************ Epoch=21 end ************
2022-03-02 00:32:38,581 P40465 INFO [Metrics] logloss: 0.445975 - AUC: 0.805497
2022-03-02 00:32:38,583 P40465 INFO Monitor(max) STOP: 0.359521 !
2022-03-02 00:32:38,583 P40465 INFO Reduce learning rate on plateau: 0.000010
2022-03-02 00:32:38,583 P40465 INFO --- 3668/3668 batches finished ---
2022-03-02 00:32:39,286 P40465 INFO Train loss: 0.443035
2022-03-02 00:32:39,286 P40465 INFO ************ Epoch=22 end ************
2022-03-02 00:40:06,503 P40465 INFO [Metrics] logloss: 0.445916 - AUC: 0.805605
2022-03-02 00:40:06,505 P40465 INFO Save best model: monitor(max): 0.359689
2022-03-02 00:40:06,608 P40465 INFO --- 3668/3668 batches finished ---
2022-03-02 00:40:07,196 P40465 INFO Train loss: 0.440715
2022-03-02 00:40:07,196 P40465 INFO ************ Epoch=23 end ************
2022-03-02 00:47:34,076 P40465 INFO [Metrics] logloss: 0.446018 - AUC: 0.805537
2022-03-02 00:47:34,077 P40465 INFO Monitor(max) STOP: 0.359520 !
2022-03-02 00:47:34,078 P40465 INFO Reduce learning rate on plateau: 0.000001
2022-03-02 00:47:34,078 P40465 INFO --- 3668/3668 batches finished ---
2022-03-02 00:47:34,681 P40465 INFO Train loss: 0.440458
2022-03-02 00:47:34,681 P40465 INFO ************ Epoch=24 end ************
2022-03-02 00:54:59,485 P40465 INFO [Metrics] logloss: 0.446014 - AUC: 0.805527
2022-03-02 00:54:59,486 P40465 INFO Monitor(max) STOP: 0.359513 !
2022-03-02 00:54:59,486 P40465 INFO Reduce learning rate on plateau: 0.000001
2022-03-02 00:54:59,486 P40465 INFO Early stopping at epoch=25
2022-03-02 00:54:59,486 P40465 INFO --- 3668/3668 batches finished ---
2022-03-02 00:55:00,097 P40465 INFO Train loss: 0.440093
2022-03-02 00:55:00,098 P40465 INFO Training finished.
2022-03-02 00:55:00,098 P40465 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Criteo/CrossNet_criteo_x4_001/criteo_x4_9ea3bdfc/DCN_criteo_x4_001_964751c8_model.ckpt
2022-03-02 00:55:00,195 P40465 INFO ****** Validation evaluation ******
2022-03-02 00:55:31,196 P40465 INFO [Metrics] logloss: 0.445916 - AUC: 0.805605
2022-03-02 00:55:32,383 P40465 INFO ******** Test evaluation ********
2022-03-02 00:55:32,383 P40465 INFO Loading data...
2022-03-02 00:55:32,384 P40465 INFO Reading file: ../data/Criteo/Criteo_x4/test.csv
2022-03-02 00:56:05,472 P40465 INFO Preprocess feature columns...
2022-03-02 00:58:33,543 P40465 INFO Transform feature columns...
2022-03-02 00:59:43,007 P40465 INFO Saving data to h5: ../data/Criteo/criteo_x4_9ea3bdfc/test.h5
2022-03-02 00:59:44,936 P40465 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2022-03-02 00:59:44,936 P40465 INFO Loading test data done.
2022-03-02 01:00:14,336 P40465 INFO [Metrics] logloss: 0.445639 - AUC: 0.805964

```
