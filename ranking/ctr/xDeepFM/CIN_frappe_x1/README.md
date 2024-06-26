## CIN_frappe_x1

A hands-on guide to run the xDeepFM model on the Frappe_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

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
Dataset ID: [Frappe_x1](https://github.com/reczoo/Datasets/tree/main/Frappe/Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [xDeepFM](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/xDeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CIN_frappe_x1_tuner_config_03](./CIN_frappe_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CIN_frappe_x1
    nohup python run_expid.py --config ./CIN_frappe_x1_tuner_config_03 --expid xDeepFM_frappe_x1_008_9e8759c4 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.978241 | 0.235220  |


### Logs
```python
2022-02-04 11:11:01,093 P46436 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "cin_layer_units": "[64, 64, 64]",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "dnn_hidden_units": "None",
    "embedding_dim": "10",
    "embedding_regularizer": "0.0005",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "xDeepFM",
    "model_id": "xDeepFM_frappe_x1_008_9e8759c4",
    "model_root": "./Frappe/xDeepFM_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
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
    "test_data": "../data/Frappe/Frappe_x1/test.csv",
    "train_data": "../data/Frappe/Frappe_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-02-04 11:11:01,094 P46436 INFO Set up feature encoder...
2022-02-04 11:11:01,094 P46436 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-02-04 11:11:01,094 P46436 INFO Loading data...
2022-02-04 11:11:01,096 P46436 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-02-04 11:11:01,108 P46436 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-02-04 11:11:01,113 P46436 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-02-04 11:11:01,113 P46436 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-02-04 11:11:01,113 P46436 INFO Loading train data done.
2022-02-04 11:11:05,065 P46436 INFO Total number of parameters: 147984.
2022-02-04 11:11:05,066 P46436 INFO Start training: 50 batches/epoch
2022-02-04 11:11:05,066 P46436 INFO ************ Epoch=1 start ************
2022-02-04 11:11:17,384 P46436 INFO [Metrics] AUC: 0.925244 - logloss: 0.354707
2022-02-04 11:11:17,384 P46436 INFO Save best model: monitor(max): 0.925244
2022-02-04 11:11:17,387 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:11:17,422 P46436 INFO Train loss: 0.573485
2022-02-04 11:11:17,422 P46436 INFO ************ Epoch=1 end ************
2022-02-04 11:11:29,782 P46436 INFO [Metrics] AUC: 0.936158 - logloss: 0.283976
2022-02-04 11:11:29,783 P46436 INFO Save best model: monitor(max): 0.936158
2022-02-04 11:11:29,785 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:11:29,844 P46436 INFO Train loss: 0.304019
2022-02-04 11:11:29,844 P46436 INFO ************ Epoch=2 end ************
2022-02-04 11:11:42,098 P46436 INFO [Metrics] AUC: 0.938544 - logloss: 0.280556
2022-02-04 11:11:42,098 P46436 INFO Save best model: monitor(max): 0.938544
2022-02-04 11:11:42,101 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:11:42,139 P46436 INFO Train loss: 0.283794
2022-02-04 11:11:42,140 P46436 INFO ************ Epoch=3 end ************
2022-02-04 11:11:54,415 P46436 INFO [Metrics] AUC: 0.939633 - logloss: 0.280233
2022-02-04 11:11:54,416 P46436 INFO Save best model: monitor(max): 0.939633
2022-02-04 11:11:54,419 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:11:54,457 P46436 INFO Train loss: 0.279838
2022-02-04 11:11:54,457 P46436 INFO ************ Epoch=4 end ************
2022-02-04 11:12:06,809 P46436 INFO [Metrics] AUC: 0.940251 - logloss: 0.278679
2022-02-04 11:12:06,810 P46436 INFO Save best model: monitor(max): 0.940251
2022-02-04 11:12:06,813 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:12:06,857 P46436 INFO Train loss: 0.278425
2022-02-04 11:12:06,857 P46436 INFO ************ Epoch=5 end ************
2022-02-04 11:12:19,116 P46436 INFO [Metrics] AUC: 0.940763 - logloss: 0.277701
2022-02-04 11:12:19,116 P46436 INFO Save best model: monitor(max): 0.940763
2022-02-04 11:12:19,119 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:12:19,153 P46436 INFO Train loss: 0.275414
2022-02-04 11:12:19,153 P46436 INFO ************ Epoch=6 end ************
2022-02-04 11:12:31,532 P46436 INFO [Metrics] AUC: 0.941331 - logloss: 0.277286
2022-02-04 11:12:31,532 P46436 INFO Save best model: monitor(max): 0.941331
2022-02-04 11:12:31,535 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:12:31,569 P46436 INFO Train loss: 0.274926
2022-02-04 11:12:31,569 P46436 INFO ************ Epoch=7 end ************
2022-02-04 11:12:43,760 P46436 INFO [Metrics] AUC: 0.941852 - logloss: 0.277647
2022-02-04 11:12:43,761 P46436 INFO Save best model: monitor(max): 0.941852
2022-02-04 11:12:43,763 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:12:43,810 P46436 INFO Train loss: 0.273314
2022-02-04 11:12:43,810 P46436 INFO ************ Epoch=8 end ************
2022-02-04 11:12:56,057 P46436 INFO [Metrics] AUC: 0.942000 - logloss: 0.276246
2022-02-04 11:12:56,058 P46436 INFO Save best model: monitor(max): 0.942000
2022-02-04 11:12:56,062 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:12:56,135 P46436 INFO Train loss: 0.272168
2022-02-04 11:12:56,135 P46436 INFO ************ Epoch=9 end ************
2022-02-04 11:13:08,463 P46436 INFO [Metrics] AUC: 0.942546 - logloss: 0.276250
2022-02-04 11:13:08,463 P46436 INFO Save best model: monitor(max): 0.942546
2022-02-04 11:13:08,466 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:13:08,506 P46436 INFO Train loss: 0.271052
2022-02-04 11:13:08,506 P46436 INFO ************ Epoch=10 end ************
2022-02-04 11:13:20,768 P46436 INFO [Metrics] AUC: 0.943205 - logloss: 0.274699
2022-02-04 11:13:20,769 P46436 INFO Save best model: monitor(max): 0.943205
2022-02-04 11:13:20,772 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:13:20,809 P46436 INFO Train loss: 0.269695
2022-02-04 11:13:20,809 P46436 INFO ************ Epoch=11 end ************
2022-02-04 11:13:33,088 P46436 INFO [Metrics] AUC: 0.945284 - logloss: 0.274018
2022-02-04 11:13:33,089 P46436 INFO Save best model: monitor(max): 0.945284
2022-02-04 11:13:33,091 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:13:33,129 P46436 INFO Train loss: 0.266709
2022-02-04 11:13:33,129 P46436 INFO ************ Epoch=12 end ************
2022-02-04 11:13:45,341 P46436 INFO [Metrics] AUC: 0.953930 - logloss: 0.249899
2022-02-04 11:13:45,342 P46436 INFO Save best model: monitor(max): 0.953930
2022-02-04 11:13:45,345 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:13:45,384 P46436 INFO Train loss: 0.252862
2022-02-04 11:13:45,384 P46436 INFO ************ Epoch=13 end ************
2022-02-04 11:13:57,572 P46436 INFO [Metrics] AUC: 0.963628 - logloss: 0.222292
2022-02-04 11:13:57,572 P46436 INFO Save best model: monitor(max): 0.963628
2022-02-04 11:13:57,575 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:13:57,612 P46436 INFO Train loss: 0.220386
2022-02-04 11:13:57,612 P46436 INFO ************ Epoch=14 end ************
2022-02-04 11:14:09,782 P46436 INFO [Metrics] AUC: 0.968741 - logloss: 0.209030
2022-02-04 11:14:09,783 P46436 INFO Save best model: monitor(max): 0.968741
2022-02-04 11:14:09,785 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:14:09,835 P46436 INFO Train loss: 0.188808
2022-02-04 11:14:09,835 P46436 INFO ************ Epoch=15 end ************
2022-02-04 11:14:21,950 P46436 INFO [Metrics] AUC: 0.972639 - logloss: 0.197474
2022-02-04 11:14:21,950 P46436 INFO Save best model: monitor(max): 0.972639
2022-02-04 11:14:21,954 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:14:21,991 P46436 INFO Train loss: 0.163833
2022-02-04 11:14:21,991 P46436 INFO ************ Epoch=16 end ************
2022-02-04 11:14:34,117 P46436 INFO [Metrics] AUC: 0.975233 - logloss: 0.190254
2022-02-04 11:14:34,118 P46436 INFO Save best model: monitor(max): 0.975233
2022-02-04 11:14:34,120 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:14:34,158 P46436 INFO Train loss: 0.140164
2022-02-04 11:14:34,159 P46436 INFO ************ Epoch=17 end ************
2022-02-04 11:14:46,332 P46436 INFO [Metrics] AUC: 0.976902 - logloss: 0.191125
2022-02-04 11:14:46,333 P46436 INFO Save best model: monitor(max): 0.976902
2022-02-04 11:14:46,336 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:14:46,377 P46436 INFO Train loss: 0.116964
2022-02-04 11:14:46,377 P46436 INFO ************ Epoch=18 end ************
2022-02-04 11:14:58,538 P46436 INFO [Metrics] AUC: 0.978038 - logloss: 0.196747
2022-02-04 11:14:58,538 P46436 INFO Save best model: monitor(max): 0.978038
2022-02-04 11:14:58,541 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:14:58,577 P46436 INFO Train loss: 0.098095
2022-02-04 11:14:58,577 P46436 INFO ************ Epoch=19 end ************
2022-02-04 11:15:10,851 P46436 INFO [Metrics] AUC: 0.978652 - logloss: 0.206045
2022-02-04 11:15:10,851 P46436 INFO Save best model: monitor(max): 0.978652
2022-02-04 11:15:10,854 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:15:10,890 P46436 INFO Train loss: 0.082835
2022-02-04 11:15:10,890 P46436 INFO ************ Epoch=20 end ************
2022-02-04 11:15:22,982 P46436 INFO [Metrics] AUC: 0.978831 - logloss: 0.220364
2022-02-04 11:15:22,983 P46436 INFO Save best model: monitor(max): 0.978831
2022-02-04 11:15:22,986 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:15:23,031 P46436 INFO Train loss: 0.071678
2022-02-04 11:15:23,032 P46436 INFO ************ Epoch=21 end ************
2022-02-04 11:15:35,099 P46436 INFO [Metrics] AUC: 0.979186 - logloss: 0.230473
2022-02-04 11:15:35,099 P46436 INFO Save best model: monitor(max): 0.979186
2022-02-04 11:15:35,102 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:15:35,146 P46436 INFO Train loss: 0.063483
2022-02-04 11:15:35,146 P46436 INFO ************ Epoch=22 end ************
2022-02-04 11:15:47,221 P46436 INFO [Metrics] AUC: 0.978613 - logloss: 0.243707
2022-02-04 11:15:47,222 P46436 INFO Monitor(max) STOP: 0.978613 !
2022-02-04 11:15:47,222 P46436 INFO Reduce learning rate on plateau: 0.000100
2022-02-04 11:15:47,222 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:15:47,268 P46436 INFO Train loss: 0.058160
2022-02-04 11:15:47,268 P46436 INFO ************ Epoch=23 end ************
2022-02-04 11:15:59,444 P46436 INFO [Metrics] AUC: 0.978951 - logloss: 0.244441
2022-02-04 11:15:59,445 P46436 INFO Monitor(max) STOP: 0.978951 !
2022-02-04 11:15:59,445 P46436 INFO Reduce learning rate on plateau: 0.000010
2022-02-04 11:15:59,445 P46436 INFO Early stopping at epoch=24
2022-02-04 11:15:59,445 P46436 INFO --- 50/50 batches finished ---
2022-02-04 11:15:59,482 P46436 INFO Train loss: 0.044448
2022-02-04 11:15:59,482 P46436 INFO Training finished.
2022-02-04 11:15:59,482 P46436 INFO Load best model: /home/XXX/benchmarks/Frappe/xDeepFM_frappe_x1/frappe_x1_04e961e9/xDeepFM_frappe_x1_008_9e8759c4.model
2022-02-04 11:16:03,916 P46436 INFO ****** Validation evaluation ******
2022-02-04 11:16:04,504 P46436 INFO [Metrics] AUC: 0.979186 - logloss: 0.230473
2022-02-04 11:16:04,538 P46436 INFO ******** Test evaluation ********
2022-02-04 11:16:04,538 P46436 INFO Loading data...
2022-02-04 11:16:04,538 P46436 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-02-04 11:16:04,540 P46436 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-02-04 11:16:04,541 P46436 INFO Loading test data done.
2022-02-04 11:16:04,913 P46436 INFO [Metrics] AUC: 0.978241 - logloss: 0.235220

```
