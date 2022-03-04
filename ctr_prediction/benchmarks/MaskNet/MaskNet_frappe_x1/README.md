## MaskNet_frappe_x1

A hands-on guide to run the MaskNet model on the Frappe_x1 dataset.

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
  fuxictr: 1.1.1
  ```

### Dataset
Dataset ID: [Frappe_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Frappe/README.md#Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.1](https://github.com/xue-pai/FuxiCTR/tree/v1.1.1) for this experiment. See the model code: [MaskNet](https://github.com/xue-pai/FuxiCTR/blob/v1.1.1/fuxictr/pytorch/models/MaskNet.py).

Running steps:

1. Download [FuxiCTR-v1.1.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [MaskNet_frappe_x1_tuner_config_03](./MaskNet_frappe_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd MaskNet_frappe_x1
    nohup python run_expid.py --config ./MaskNet_frappe_x1_tuner_config_03 --expid MaskNet_frappe_x1_053_bd4428a9 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.984324 | 0.172650  |
| 2 | 0.984245 | 0.165011  |
| 3 | 0.982792 | 0.167315  |
| 4 | 0.983408 | 0.166238  |
| 5 | 0.983541 | 0.175635  |
| | | | 
| Avg | 0.983662 | 0.169370 |
| Std | &#177;0.00056819 | &#177;0.00407534 |


### Logs
```python
2022-01-29 09:05:19,793 P18187 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "dnn_hidden_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "emb_layernorm": "True",
    "embedding_dim": "10",
    "embedding_regularizer": "0.1",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "MaskNet",
    "model_id": "MaskNet_frappe_x1_053_bd4428a9",
    "model_root": "./Frappe/MaskNet_frappe_x1/",
    "model_type": "SerialMaskNet",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_layernorm": "False",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_block_dim": "64",
    "parallel_num_blocks": "1",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "1",
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
2022-01-29 09:05:19,793 P18187 INFO Set up feature encoder...
2022-01-29 09:05:19,793 P18187 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-01-29 09:05:19,794 P18187 INFO Loading data...
2022-01-29 09:05:19,796 P18187 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-01-29 09:05:19,807 P18187 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-01-29 09:05:19,811 P18187 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-01-29 09:05:19,811 P18187 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-01-29 09:05:19,811 P18187 INFO Loading train data done.
2022-01-29 09:05:23,307 P18187 INFO Total number of parameters: 836291.
2022-01-29 09:05:23,307 P18187 INFO Start training: 50 batches/epoch
2022-01-29 09:05:23,307 P18187 INFO ************ Epoch=1 start ************
2022-01-29 09:05:27,279 P18187 INFO [Metrics] AUC: 0.938796 - logloss: 0.284661
2022-01-29 09:05:27,279 P18187 INFO Save best model: monitor(max): 0.938796
2022-01-29 09:05:27,285 P18187 INFO --- 50/50 batches finished ---
2022-01-29 09:05:27,322 P18187 INFO Train loss: 0.425931
2022-01-29 09:05:27,322 P18187 INFO ************ Epoch=1 end ************
2022-01-29 09:05:32,214 P18187 INFO [Metrics] AUC: 0.961596 - logloss: 0.230285
2022-01-29 09:05:32,214 P18187 INFO Save best model: monitor(max): 0.961596
2022-01-29 09:05:32,221 P18187 INFO --- 50/50 batches finished ---
2022-01-29 09:05:32,257 P18187 INFO Train loss: 0.295892
2022-01-29 09:05:32,258 P18187 INFO ************ Epoch=2 end ************
2022-01-29 09:05:36,479 P18187 INFO [Metrics] AUC: 0.972209 - logloss: 0.194651
2022-01-29 09:05:36,480 P18187 INFO Save best model: monitor(max): 0.972209
2022-01-29 09:05:36,486 P18187 INFO --- 50/50 batches finished ---
2022-01-29 09:05:36,530 P18187 INFO Train loss: 0.259941
2022-01-29 09:05:36,530 P18187 INFO ************ Epoch=3 end ************
2022-01-29 09:05:40,378 P18187 INFO [Metrics] AUC: 0.976468 - logloss: 0.175908
2022-01-29 09:05:40,379 P18187 INFO Save best model: monitor(max): 0.976468
2022-01-29 09:05:40,385 P18187 INFO --- 50/50 batches finished ---
2022-01-29 09:05:40,424 P18187 INFO Train loss: 0.240686
2022-01-29 09:05:40,424 P18187 INFO ************ Epoch=4 end ************
2022-01-29 09:05:44,608 P18187 INFO [Metrics] AUC: 0.978210 - logloss: 0.168243
2022-01-29 09:05:44,609 P18187 INFO Save best model: monitor(max): 0.978210
2022-01-29 09:05:44,615 P18187 INFO --- 50/50 batches finished ---
2022-01-29 09:05:44,655 P18187 INFO Train loss: 0.227718
2022-01-29 09:05:44,655 P18187 INFO ************ Epoch=5 end ************
2022-01-29 09:05:49,792 P18187 INFO [Metrics] AUC: 0.978924 - logloss: 0.165224
2022-01-29 09:05:49,793 P18187 INFO Save best model: monitor(max): 0.978924
2022-01-29 09:05:49,799 P18187 INFO --- 50/50 batches finished ---
2022-01-29 09:05:49,834 P18187 INFO Train loss: 0.219551
2022-01-29 09:05:49,835 P18187 INFO ************ Epoch=6 end ************
2022-01-29 09:05:54,877 P18187 INFO [Metrics] AUC: 0.979004 - logloss: 0.166780
2022-01-29 09:05:54,877 P18187 INFO Save best model: monitor(max): 0.979004
2022-01-29 09:05:54,884 P18187 INFO --- 50/50 batches finished ---
2022-01-29 09:05:54,918 P18187 INFO Train loss: 0.214804
2022-01-29 09:05:54,918 P18187 INFO ************ Epoch=7 end ************
2022-01-29 09:06:00,142 P18187 INFO [Metrics] AUC: 0.979923 - logloss: 0.161069
2022-01-29 09:06:00,143 P18187 INFO Save best model: monitor(max): 0.979923
2022-01-29 09:06:00,149 P18187 INFO --- 50/50 batches finished ---
2022-01-29 09:06:00,185 P18187 INFO Train loss: 0.212766
2022-01-29 09:06:00,185 P18187 INFO ************ Epoch=8 end ************
2022-01-29 09:06:05,288 P18187 INFO [Metrics] AUC: 0.981073 - logloss: 0.157433
2022-01-29 09:06:05,288 P18187 INFO Save best model: monitor(max): 0.981073
2022-01-29 09:06:05,295 P18187 INFO --- 50/50 batches finished ---
2022-01-29 09:06:05,329 P18187 INFO Train loss: 0.207458
2022-01-29 09:06:05,329 P18187 INFO ************ Epoch=9 end ************
2022-01-29 09:06:10,667 P18187 INFO [Metrics] AUC: 0.980679 - logloss: 0.157450
2022-01-29 09:06:10,667 P18187 INFO Monitor(max) STOP: 0.980679 !
2022-01-29 09:06:10,667 P18187 INFO Reduce learning rate on plateau: 0.000100
2022-01-29 09:06:10,667 P18187 INFO --- 50/50 batches finished ---
2022-01-29 09:06:10,710 P18187 INFO Train loss: 0.205637
2022-01-29 09:06:10,710 P18187 INFO ************ Epoch=10 end ************
2022-01-29 09:06:15,275 P18187 INFO [Metrics] AUC: 0.983833 - logloss: 0.153822
2022-01-29 09:06:15,275 P18187 INFO Save best model: monitor(max): 0.983833
2022-01-29 09:06:15,282 P18187 INFO --- 50/50 batches finished ---
2022-01-29 09:06:15,322 P18187 INFO Train loss: 0.163112
2022-01-29 09:06:15,322 P18187 INFO ************ Epoch=11 end ************
2022-01-29 09:06:19,561 P18187 INFO [Metrics] AUC: 0.984651 - logloss: 0.157777
2022-01-29 09:06:19,562 P18187 INFO Save best model: monitor(max): 0.984651
2022-01-29 09:06:19,568 P18187 INFO --- 50/50 batches finished ---
2022-01-29 09:06:19,606 P18187 INFO Train loss: 0.127349
2022-01-29 09:06:19,606 P18187 INFO ************ Epoch=12 end ************
2022-01-29 09:06:23,652 P18187 INFO [Metrics] AUC: 0.984704 - logloss: 0.169387
2022-01-29 09:06:23,653 P18187 INFO Save best model: monitor(max): 0.984704
2022-01-29 09:06:23,659 P18187 INFO --- 50/50 batches finished ---
2022-01-29 09:06:23,697 P18187 INFO Train loss: 0.106299
2022-01-29 09:06:23,697 P18187 INFO ************ Epoch=13 end ************
2022-01-29 09:06:28,866 P18187 INFO [Metrics] AUC: 0.984417 - logloss: 0.181842
2022-01-29 09:06:28,867 P18187 INFO Monitor(max) STOP: 0.984417 !
2022-01-29 09:06:28,867 P18187 INFO Reduce learning rate on plateau: 0.000010
2022-01-29 09:06:28,867 P18187 INFO --- 50/50 batches finished ---
2022-01-29 09:06:28,902 P18187 INFO Train loss: 0.091695
2022-01-29 09:06:28,902 P18187 INFO ************ Epoch=14 end ************
2022-01-29 09:06:34,007 P18187 INFO [Metrics] AUC: 0.984299 - logloss: 0.191862
2022-01-29 09:06:34,008 P18187 INFO Monitor(max) STOP: 0.984299 !
2022-01-29 09:06:34,008 P18187 INFO Reduce learning rate on plateau: 0.000001
2022-01-29 09:06:34,008 P18187 INFO Early stopping at epoch=15
2022-01-29 09:06:34,008 P18187 INFO --- 50/50 batches finished ---
2022-01-29 09:06:34,059 P18187 INFO Train loss: 0.080797
2022-01-29 09:06:34,059 P18187 INFO Training finished.
2022-01-29 09:06:34,059 P18187 INFO Load best model: /home/XXX/benchmarks/Frappe/MaskNet_frappe_x1/frappe_x1_04e961e9/MaskNet_frappe_x1_053_bd4428a9.model
2022-01-29 09:06:34,077 P18187 INFO ****** Validation evaluation ******
2022-01-29 09:06:34,473 P18187 INFO [Metrics] AUC: 0.984704 - logloss: 0.169387
2022-01-29 09:06:34,531 P18187 INFO ******** Test evaluation ********
2022-01-29 09:06:34,531 P18187 INFO Loading data...
2022-01-29 09:06:34,531 P18187 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-01-29 09:06:34,534 P18187 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-01-29 09:06:34,534 P18187 INFO Loading test data done.
2022-01-29 09:06:34,783 P18187 INFO [Metrics] AUC: 0.984324 - logloss: 0.172650

```
