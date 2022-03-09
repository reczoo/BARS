## FiGNN_frappe_x1

A hands-on guide to run the FiGNN model on the Frappe_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

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
  CUDA: 11.4
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  fuxictr: 1.1.0
  ```

### Dataset
Dataset ID: [Frappe_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Frappe/README.md#Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [FiGNN](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FiGNN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FiGNN_frappe_x1_tuner_config_04](./FiGNN_frappe_x1_tuner_config_04). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FiGNN_frappe_x1
    nohup python run_expid.py --config ./FiGNN_frappe_x1_tuner_config_04 --expid FiGNN_frappe_x1_003_57243619 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.976115 | 0.203715  |


### Logs
```python
2022-01-30 14:40:33,065 P33832 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gnn_layers": "17",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FiGNN",
    "model_id": "FiGNN_frappe_x1_003_57243619",
    "model_root": "./Frappe/FiGNN_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reuse_graph_layer": "False",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Frappe/Frappe_x1/test.csv",
    "train_data": "../data/Frappe/Frappe_x1/train.csv",
    "use_gru": "False",
    "use_hdf5": "True",
    "use_residual": "True",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-30 14:40:33,066 P33832 INFO Set up feature encoder...
2022-01-30 14:40:33,066 P33832 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-01-30 14:40:33,066 P33832 INFO Loading data...
2022-01-30 14:40:33,068 P33832 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-01-30 14:40:33,079 P33832 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-01-30 14:40:33,084 P33832 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-01-30 14:40:33,084 P33832 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-01-30 14:40:33,084 P33832 INFO Loading train data done.
2022-01-30 14:40:36,798 P33832 INFO Total number of parameters: 89090.
2022-01-30 14:40:36,799 P33832 INFO Start training: 50 batches/epoch
2022-01-30 14:40:36,799 P33832 INFO ************ Epoch=1 start ************
2022-01-30 14:41:02,516 P33832 INFO [Metrics] AUC: 0.935025 - logloss: 0.288721
2022-01-30 14:41:02,517 P33832 INFO Save best model: monitor(max): 0.935025
2022-01-30 14:41:02,521 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:41:02,561 P33832 INFO Train loss: 0.407789
2022-01-30 14:41:02,562 P33832 INFO ************ Epoch=1 end ************
2022-01-30 14:41:27,968 P33832 INFO [Metrics] AUC: 0.938091 - logloss: 0.282474
2022-01-30 14:41:27,969 P33832 INFO Save best model: monitor(max): 0.938091
2022-01-30 14:41:27,973 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:41:28,012 P33832 INFO Train loss: 0.283334
2022-01-30 14:41:28,012 P33832 INFO ************ Epoch=2 end ************
2022-01-30 14:41:53,276 P33832 INFO [Metrics] AUC: 0.940343 - logloss: 0.278627
2022-01-30 14:41:53,277 P33832 INFO Save best model: monitor(max): 0.940343
2022-01-30 14:41:53,281 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:41:53,323 P33832 INFO Train loss: 0.277152
2022-01-30 14:41:53,323 P33832 INFO ************ Epoch=3 end ************
2022-01-30 14:42:18,413 P33832 INFO [Metrics] AUC: 0.942567 - logloss: 0.275351
2022-01-30 14:42:18,414 P33832 INFO Save best model: monitor(max): 0.942567
2022-01-30 14:42:18,417 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:42:18,456 P33832 INFO Train loss: 0.273230
2022-01-30 14:42:18,456 P33832 INFO ************ Epoch=4 end ************
2022-01-30 14:42:43,786 P33832 INFO [Metrics] AUC: 0.944609 - logloss: 0.270734
2022-01-30 14:42:43,786 P33832 INFO Save best model: monitor(max): 0.944609
2022-01-30 14:42:43,790 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:42:43,834 P33832 INFO Train loss: 0.268710
2022-01-30 14:42:43,835 P33832 INFO ************ Epoch=5 end ************
2022-01-30 14:43:09,139 P33832 INFO [Metrics] AUC: 0.951021 - logloss: 0.255503
2022-01-30 14:43:09,140 P33832 INFO Save best model: monitor(max): 0.951021
2022-01-30 14:43:09,143 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:43:09,186 P33832 INFO Train loss: 0.257577
2022-01-30 14:43:09,186 P33832 INFO ************ Epoch=6 end ************
2022-01-30 14:43:34,520 P33832 INFO [Metrics] AUC: 0.958755 - logloss: 0.235267
2022-01-30 14:43:34,520 P33832 INFO Save best model: monitor(max): 0.958755
2022-01-30 14:43:34,524 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:43:34,560 P33832 INFO Train loss: 0.233225
2022-01-30 14:43:34,560 P33832 INFO ************ Epoch=7 end ************
2022-01-30 14:43:59,887 P33832 INFO [Metrics] AUC: 0.962401 - logloss: 0.227945
2022-01-30 14:43:59,887 P33832 INFO Save best model: monitor(max): 0.962401
2022-01-30 14:43:59,891 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:43:59,930 P33832 INFO Train loss: 0.214106
2022-01-30 14:43:59,930 P33832 INFO ************ Epoch=8 end ************
2022-01-30 14:44:25,249 P33832 INFO [Metrics] AUC: 0.963655 - logloss: 0.224759
2022-01-30 14:44:25,250 P33832 INFO Save best model: monitor(max): 0.963655
2022-01-30 14:44:25,254 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:44:25,293 P33832 INFO Train loss: 0.202558
2022-01-30 14:44:25,293 P33832 INFO ************ Epoch=9 end ************
2022-01-30 14:44:50,539 P33832 INFO [Metrics] AUC: 0.965210 - logloss: 0.220842
2022-01-30 14:44:50,539 P33832 INFO Save best model: monitor(max): 0.965210
2022-01-30 14:44:50,543 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:44:50,579 P33832 INFO Train loss: 0.193668
2022-01-30 14:44:50,579 P33832 INFO ************ Epoch=10 end ************
2022-01-30 14:45:15,629 P33832 INFO [Metrics] AUC: 0.966960 - logloss: 0.216779
2022-01-30 14:45:15,630 P33832 INFO Save best model: monitor(max): 0.966960
2022-01-30 14:45:15,634 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:45:15,673 P33832 INFO Train loss: 0.185044
2022-01-30 14:45:15,673 P33832 INFO ************ Epoch=11 end ************
2022-01-30 14:45:40,728 P33832 INFO [Metrics] AUC: 0.969533 - logloss: 0.209161
2022-01-30 14:45:40,729 P33832 INFO Save best model: monitor(max): 0.969533
2022-01-30 14:45:40,732 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:45:40,778 P33832 INFO Train loss: 0.173330
2022-01-30 14:45:40,778 P33832 INFO ************ Epoch=12 end ************
2022-01-30 14:46:05,979 P33832 INFO [Metrics] AUC: 0.971467 - logloss: 0.206305
2022-01-30 14:46:05,979 P33832 INFO Save best model: monitor(max): 0.971467
2022-01-30 14:46:05,983 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:46:06,030 P33832 INFO Train loss: 0.159345
2022-01-30 14:46:06,030 P33832 INFO ************ Epoch=13 end ************
2022-01-30 14:46:31,148 P33832 INFO [Metrics] AUC: 0.972379 - logloss: 0.203507
2022-01-30 14:46:31,149 P33832 INFO Save best model: monitor(max): 0.972379
2022-01-30 14:46:31,152 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:46:31,191 P33832 INFO Train loss: 0.149075
2022-01-30 14:46:31,191 P33832 INFO ************ Epoch=14 end ************
2022-01-30 14:46:56,317 P33832 INFO [Metrics] AUC: 0.973403 - logloss: 0.201242
2022-01-30 14:46:56,318 P33832 INFO Save best model: monitor(max): 0.973403
2022-01-30 14:46:56,321 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:46:56,364 P33832 INFO Train loss: 0.142260
2022-01-30 14:46:56,364 P33832 INFO ************ Epoch=15 end ************
2022-01-30 14:47:21,486 P33832 INFO [Metrics] AUC: 0.973849 - logloss: 0.203248
2022-01-30 14:47:21,487 P33832 INFO Save best model: monitor(max): 0.973849
2022-01-30 14:47:21,490 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:47:21,538 P33832 INFO Train loss: 0.135955
2022-01-30 14:47:21,538 P33832 INFO ************ Epoch=16 end ************
2022-01-30 14:47:46,540 P33832 INFO [Metrics] AUC: 0.974104 - logloss: 0.204109
2022-01-30 14:47:46,540 P33832 INFO Save best model: monitor(max): 0.974104
2022-01-30 14:47:46,544 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:47:46,584 P33832 INFO Train loss: 0.131322
2022-01-30 14:47:46,584 P33832 INFO ************ Epoch=17 end ************
2022-01-30 14:48:11,623 P33832 INFO [Metrics] AUC: 0.973843 - logloss: 0.207278
2022-01-30 14:48:11,624 P33832 INFO Monitor(max) STOP: 0.973843 !
2022-01-30 14:48:11,624 P33832 INFO Reduce learning rate on plateau: 0.000100
2022-01-30 14:48:11,624 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:48:11,672 P33832 INFO Train loss: 0.126547
2022-01-30 14:48:11,672 P33832 INFO ************ Epoch=18 end ************
2022-01-30 14:48:36,782 P33832 INFO [Metrics] AUC: 0.974908 - logloss: 0.203145
2022-01-30 14:48:36,782 P33832 INFO Save best model: monitor(max): 0.974908
2022-01-30 14:48:36,786 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:48:36,833 P33832 INFO Train loss: 0.110765
2022-01-30 14:48:36,834 P33832 INFO ************ Epoch=19 end ************
2022-01-30 14:49:01,845 P33832 INFO [Metrics] AUC: 0.975183 - logloss: 0.202638
2022-01-30 14:49:01,846 P33832 INFO Save best model: monitor(max): 0.975183
2022-01-30 14:49:01,850 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:49:01,887 P33832 INFO Train loss: 0.105827
2022-01-30 14:49:01,887 P33832 INFO ************ Epoch=20 end ************
2022-01-30 14:49:26,942 P33832 INFO [Metrics] AUC: 0.975352 - logloss: 0.203147
2022-01-30 14:49:26,942 P33832 INFO Save best model: monitor(max): 0.975352
2022-01-30 14:49:26,946 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:49:26,983 P33832 INFO Train loss: 0.103001
2022-01-30 14:49:26,983 P33832 INFO ************ Epoch=21 end ************
2022-01-30 14:49:52,096 P33832 INFO [Metrics] AUC: 0.975385 - logloss: 0.204288
2022-01-30 14:49:52,096 P33832 INFO Save best model: monitor(max): 0.975385
2022-01-30 14:49:52,100 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:49:52,151 P33832 INFO Train loss: 0.100893
2022-01-30 14:49:52,151 P33832 INFO ************ Epoch=22 end ************
2022-01-30 14:50:17,260 P33832 INFO [Metrics] AUC: 0.975541 - logloss: 0.205237
2022-01-30 14:50:17,260 P33832 INFO Save best model: monitor(max): 0.975541
2022-01-30 14:50:17,264 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:50:17,302 P33832 INFO Train loss: 0.099433
2022-01-30 14:50:17,302 P33832 INFO ************ Epoch=23 end ************
2022-01-30 14:50:42,414 P33832 INFO [Metrics] AUC: 0.975631 - logloss: 0.206262
2022-01-30 14:50:42,414 P33832 INFO Save best model: monitor(max): 0.975631
2022-01-30 14:50:42,418 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:50:42,458 P33832 INFO Train loss: 0.098177
2022-01-30 14:50:42,458 P33832 INFO ************ Epoch=24 end ************
2022-01-30 14:51:03,235 P33832 INFO [Metrics] AUC: 0.975505 - logloss: 0.207314
2022-01-30 14:51:03,235 P33832 INFO Monitor(max) STOP: 0.975505 !
2022-01-30 14:51:03,235 P33832 INFO Reduce learning rate on plateau: 0.000010
2022-01-30 14:51:03,235 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:51:03,276 P33832 INFO Train loss: 0.097041
2022-01-30 14:51:03,276 P33832 INFO ************ Epoch=25 end ************
2022-01-30 14:51:19,857 P33832 INFO [Metrics] AUC: 0.975503 - logloss: 0.207545
2022-01-30 14:51:19,857 P33832 INFO Monitor(max) STOP: 0.975503 !
2022-01-30 14:51:19,857 P33832 INFO Reduce learning rate on plateau: 0.000001
2022-01-30 14:51:19,857 P33832 INFO Early stopping at epoch=26
2022-01-30 14:51:19,857 P33832 INFO --- 50/50 batches finished ---
2022-01-30 14:51:19,896 P33832 INFO Train loss: 0.094844
2022-01-30 14:51:19,896 P33832 INFO Training finished.
2022-01-30 14:51:19,896 P33832 INFO Load best model: /home/XXX/benchmarks/Frappe/FiGNN_frappe_x1/frappe_x1_04e961e9/FiGNN_frappe_x1_003_57243619.model
2022-01-30 14:51:19,940 P33832 INFO ****** Validation evaluation ******
2022-01-30 14:51:20,818 P33832 INFO [Metrics] AUC: 0.975631 - logloss: 0.206262
2022-01-30 14:51:20,858 P33832 INFO ******** Test evaluation ********
2022-01-30 14:51:20,858 P33832 INFO Loading data...
2022-01-30 14:51:20,859 P33832 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-01-30 14:51:20,861 P33832 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-01-30 14:51:20,861 P33832 INFO Loading test data done.
2022-01-30 14:51:21,372 P33832 INFO [Metrics] AUC: 0.976115 - logloss: 0.203715

```
