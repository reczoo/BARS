## DNN_frappe_x1

A hands-on guide to run the DNN model on the Frappe_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DNN](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DNN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DNN_frappe_x1_tuner_config_02](./DNN_frappe_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DNN_frappe_x1
    nohup python run_expid.py --config ./DNN_frappe_x1_tuner_config_02 --expid DNN_frappe_x1_001_ad0f0fce --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.983323 | 0.162229  |
| 2 | 0.984036 | 0.157776  |
| 3 | 0.983869 | 0.158045  |
| 4 | 0.984088 | 0.155575  |
| 5 | 0.984909 | 0.158224  |
| Avg | 0.984045 | 0.158370 |
| Std | &#177;0.00051004 | &#177;0.00215331 |


### Logs
```python
2022-02-10 19:48:23,765 P8897 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DNN",
    "model_id": "DNN_frappe_x1_001_ad0f0fce",
    "model_root": "./Frappe/DNN_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
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
2022-02-10 19:48:23,766 P8897 INFO Set up feature encoder...
2022-02-10 19:48:23,766 P8897 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-02-10 19:48:23,778 P8897 INFO Loading data...
2022-02-10 19:48:23,790 P8897 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-02-10 19:48:23,864 P8897 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-02-10 19:48:23,896 P8897 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-02-10 19:48:23,897 P8897 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-02-10 19:48:23,897 P8897 INFO Loading train data done.
2022-02-10 19:48:40,150 P8897 INFO Total number of parameters: 781892.
2022-02-10 19:48:40,151 P8897 INFO Start training: 50 batches/epoch
2022-02-10 19:48:40,151 P8897 INFO ************ Epoch=1 start ************
2022-02-10 19:48:50,052 P8897 INFO [Metrics] AUC: 0.935327 - logloss: 0.653639
2022-02-10 19:48:50,054 P8897 INFO Save best model: monitor(max): 0.935327
2022-02-10 19:48:50,078 P8897 INFO --- 50/50 batches finished ---
2022-02-10 19:48:50,294 P8897 INFO Train loss: 0.369379
2022-02-10 19:48:50,295 P8897 INFO ************ Epoch=1 end ************
2022-02-10 19:49:00,129 P8897 INFO [Metrics] AUC: 0.963559 - logloss: 0.235181
2022-02-10 19:49:00,139 P8897 INFO Save best model: monitor(max): 0.963559
2022-02-10 19:49:00,170 P8897 INFO --- 50/50 batches finished ---
2022-02-10 19:49:00,374 P8897 INFO Train loss: 0.270423
2022-02-10 19:49:00,375 P8897 INFO ************ Epoch=2 end ************
2022-02-10 19:49:10,494 P8897 INFO [Metrics] AUC: 0.974313 - logloss: 0.184344
2022-02-10 19:49:10,499 P8897 INFO Save best model: monitor(max): 0.974313
2022-02-10 19:49:10,567 P8897 INFO --- 50/50 batches finished ---
2022-02-10 19:49:10,794 P8897 INFO Train loss: 0.226715
2022-02-10 19:49:10,795 P8897 INFO ************ Epoch=3 end ************
2022-02-10 19:49:19,684 P8897 INFO [Metrics] AUC: 0.978184 - logloss: 0.171036
2022-02-10 19:49:19,693 P8897 INFO Save best model: monitor(max): 0.978184
2022-02-10 19:49:19,727 P8897 INFO --- 50/50 batches finished ---
2022-02-10 19:49:19,860 P8897 INFO Train loss: 0.201989
2022-02-10 19:49:19,860 P8897 INFO ************ Epoch=4 end ************
2022-02-10 19:49:28,671 P8897 INFO [Metrics] AUC: 0.979647 - logloss: 0.165831
2022-02-10 19:49:28,677 P8897 INFO Save best model: monitor(max): 0.979647
2022-02-10 19:49:28,713 P8897 INFO --- 50/50 batches finished ---
2022-02-10 19:49:28,955 P8897 INFO Train loss: 0.186265
2022-02-10 19:49:28,956 P8897 INFO ************ Epoch=5 end ************
2022-02-10 19:49:38,019 P8897 INFO [Metrics] AUC: 0.980853 - logloss: 0.160699
2022-02-10 19:49:38,021 P8897 INFO Save best model: monitor(max): 0.980853
2022-02-10 19:49:38,061 P8897 INFO --- 50/50 batches finished ---
2022-02-10 19:49:38,311 P8897 INFO Train loss: 0.178876
2022-02-10 19:49:38,311 P8897 INFO ************ Epoch=6 end ************
2022-02-10 19:49:47,555 P8897 INFO [Metrics] AUC: 0.981132 - logloss: 0.162531
2022-02-10 19:49:47,558 P8897 INFO Save best model: monitor(max): 0.981132
2022-02-10 19:49:47,595 P8897 INFO --- 50/50 batches finished ---
2022-02-10 19:49:47,749 P8897 INFO Train loss: 0.172960
2022-02-10 19:49:47,749 P8897 INFO ************ Epoch=7 end ************
2022-02-10 19:49:56,713 P8897 INFO [Metrics] AUC: 0.981780 - logloss: 0.157138
2022-02-10 19:49:56,721 P8897 INFO Save best model: monitor(max): 0.981780
2022-02-10 19:49:56,741 P8897 INFO --- 50/50 batches finished ---
2022-02-10 19:49:56,945 P8897 INFO Train loss: 0.168609
2022-02-10 19:49:56,946 P8897 INFO ************ Epoch=8 end ************
2022-02-10 19:50:05,676 P8897 INFO [Metrics] AUC: 0.982461 - logloss: 0.152486
2022-02-10 19:50:05,680 P8897 INFO Save best model: monitor(max): 0.982461
2022-02-10 19:50:05,697 P8897 INFO --- 50/50 batches finished ---
2022-02-10 19:50:05,926 P8897 INFO Train loss: 0.164111
2022-02-10 19:50:05,926 P8897 INFO ************ Epoch=9 end ************
2022-02-10 19:50:14,167 P8897 INFO [Metrics] AUC: 0.982074 - logloss: 0.154455
2022-02-10 19:50:14,174 P8897 INFO Monitor(max) STOP: 0.982074 !
2022-02-10 19:50:14,174 P8897 INFO Reduce learning rate on plateau: 0.000100
2022-02-10 19:50:14,174 P8897 INFO --- 50/50 batches finished ---
2022-02-10 19:50:14,332 P8897 INFO Train loss: 0.161748
2022-02-10 19:50:14,332 P8897 INFO ************ Epoch=10 end ************
2022-02-10 19:50:22,496 P8897 INFO [Metrics] AUC: 0.984005 - logloss: 0.147289
2022-02-10 19:50:22,500 P8897 INFO Save best model: monitor(max): 0.984005
2022-02-10 19:50:22,522 P8897 INFO --- 50/50 batches finished ---
2022-02-10 19:50:22,654 P8897 INFO Train loss: 0.136512
2022-02-10 19:50:22,654 P8897 INFO ************ Epoch=11 end ************
2022-02-10 19:50:30,873 P8897 INFO [Metrics] AUC: 0.984634 - logloss: 0.148080
2022-02-10 19:50:30,874 P8897 INFO Save best model: monitor(max): 0.984634
2022-02-10 19:50:30,897 P8897 INFO --- 50/50 batches finished ---
2022-02-10 19:50:31,058 P8897 INFO Train loss: 0.112925
2022-02-10 19:50:31,058 P8897 INFO ************ Epoch=12 end ************
2022-02-10 19:50:38,765 P8897 INFO [Metrics] AUC: 0.984786 - logloss: 0.150734
2022-02-10 19:50:38,768 P8897 INFO Save best model: monitor(max): 0.984786
2022-02-10 19:50:38,794 P8897 INFO --- 50/50 batches finished ---
2022-02-10 19:50:38,983 P8897 INFO Train loss: 0.099275
2022-02-10 19:50:38,983 P8897 INFO ************ Epoch=13 end ************
2022-02-10 19:50:47,931 P8897 INFO [Metrics] AUC: 0.984869 - logloss: 0.154243
2022-02-10 19:50:47,931 P8897 INFO Save best model: monitor(max): 0.984869
2022-02-10 19:50:47,965 P8897 INFO --- 50/50 batches finished ---
2022-02-10 19:50:48,151 P8897 INFO Train loss: 0.089964
2022-02-10 19:50:48,151 P8897 INFO ************ Epoch=14 end ************
2022-02-10 19:50:57,379 P8897 INFO [Metrics] AUC: 0.984660 - logloss: 0.159230
2022-02-10 19:50:57,386 P8897 INFO Monitor(max) STOP: 0.984660 !
2022-02-10 19:50:57,386 P8897 INFO Reduce learning rate on plateau: 0.000010
2022-02-10 19:50:57,386 P8897 INFO --- 50/50 batches finished ---
2022-02-10 19:50:57,623 P8897 INFO Train loss: 0.082066
2022-02-10 19:50:57,624 P8897 INFO ************ Epoch=15 end ************
2022-02-10 19:51:06,092 P8897 INFO [Metrics] AUC: 0.984726 - logloss: 0.159430
2022-02-10 19:51:06,118 P8897 INFO Monitor(max) STOP: 0.984726 !
2022-02-10 19:51:06,118 P8897 INFO Reduce learning rate on plateau: 0.000001
2022-02-10 19:51:06,118 P8897 INFO Early stopping at epoch=16
2022-02-10 19:51:06,118 P8897 INFO --- 50/50 batches finished ---
2022-02-10 19:51:06,275 P8897 INFO Train loss: 0.075929
2022-02-10 19:51:06,275 P8897 INFO Training finished.
2022-02-10 19:51:06,276 P8897 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Frappe/DNN_frappe_x1/frappe_x1_04e961e9/DNN_frappe_x1_001_ad0f0fce.model
2022-02-10 19:51:06,292 P8897 INFO ****** Validation evaluation ******
2022-02-10 19:51:08,049 P8897 INFO [Metrics] AUC: 0.984869 - logloss: 0.154243
2022-02-10 19:51:08,282 P8897 INFO ******** Test evaluation ********
2022-02-10 19:51:08,299 P8897 INFO Loading data...
2022-02-10 19:51:08,301 P8897 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-02-10 19:51:08,305 P8897 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-02-10 19:51:08,305 P8897 INFO Loading test data done.
2022-02-10 19:51:09,440 P8897 INFO [Metrics] AUC: 0.983323 - logloss: 0.162229

```
