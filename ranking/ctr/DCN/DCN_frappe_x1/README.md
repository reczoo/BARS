## DCN_frappe_x1

A hands-on guide to run the DCN model on the Frappe_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DCN](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DCN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DCN_frappe_x1_tuner_config_02](./DCN_frappe_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DCN_frappe_x1
    nohup python run_expid.py --config ./DCN_frappe_x1_tuner_config_02 --expid DCN_frappe_x1_013_efa58c31 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.983907 | 0.152660  |
| 2 | 0.982932 | 0.147549  |
| 3 | 0.983489 | 0.149382  |
| 4 | 0.983348 | 0.153989  |
| 5 | 0.983413 | 0.149799  |
| Avg | 0.983418 | 0.150676 |
| Std | &#177;0.00031154 | &#177;0.00232955 |


### Logs
```python
2022-01-18 11:05:14,211 P45242 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "crossing_layers": "3",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCN",
    "model_id": "DCN_frappe_x1_013_efa58c31",
    "model_root": "./Frappe/DCN_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
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
2022-01-18 11:05:14,212 P45242 INFO Set up feature encoder...
2022-01-18 11:05:14,212 P45242 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-01-18 11:05:14,212 P45242 INFO Loading data...
2022-01-18 11:05:14,215 P45242 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-01-18 11:05:14,226 P45242 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-01-18 11:05:14,231 P45242 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-01-18 11:05:14,231 P45242 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-01-18 11:05:14,231 P45242 INFO Loading train data done.
2022-01-18 11:05:16,984 P45242 INFO Total number of parameters: 418591.
2022-01-18 11:05:16,985 P45242 INFO Start training: 50 batches/epoch
2022-01-18 11:05:16,985 P45242 INFO ************ Epoch=1 start ************
2022-01-18 11:05:18,794 P45242 INFO [Metrics] AUC: 0.938178 - logloss: 0.589423
2022-01-18 11:05:18,794 P45242 INFO Save best model: monitor(max): 0.938178
2022-01-18 11:05:18,798 P45242 INFO --- 50/50 batches finished ---
2022-01-18 11:05:18,830 P45242 INFO Train loss: 0.366039
2022-01-18 11:05:18,830 P45242 INFO ************ Epoch=1 end ************
2022-01-18 11:05:20,566 P45242 INFO [Metrics] AUC: 0.964385 - logloss: 0.237764
2022-01-18 11:05:20,567 P45242 INFO Save best model: monitor(max): 0.964385
2022-01-18 11:05:20,571 P45242 INFO --- 50/50 batches finished ---
2022-01-18 11:05:20,604 P45242 INFO Train loss: 0.276074
2022-01-18 11:05:20,604 P45242 INFO ************ Epoch=2 end ************
2022-01-18 11:05:22,353 P45242 INFO [Metrics] AUC: 0.974080 - logloss: 0.184877
2022-01-18 11:05:22,353 P45242 INFO Save best model: monitor(max): 0.974080
2022-01-18 11:05:22,357 P45242 INFO --- 50/50 batches finished ---
2022-01-18 11:05:22,390 P45242 INFO Train loss: 0.229254
2022-01-18 11:05:22,390 P45242 INFO ************ Epoch=3 end ************
2022-01-18 11:05:24,136 P45242 INFO [Metrics] AUC: 0.977109 - logloss: 0.171899
2022-01-18 11:05:24,136 P45242 INFO Save best model: monitor(max): 0.977109
2022-01-18 11:05:24,141 P45242 INFO --- 50/50 batches finished ---
2022-01-18 11:05:24,180 P45242 INFO Train loss: 0.207806
2022-01-18 11:05:24,180 P45242 INFO ************ Epoch=4 end ************
2022-01-18 11:05:25,950 P45242 INFO [Metrics] AUC: 0.978758 - logloss: 0.168993
2022-01-18 11:05:25,950 P45242 INFO Save best model: monitor(max): 0.978758
2022-01-18 11:05:25,955 P45242 INFO --- 50/50 batches finished ---
2022-01-18 11:05:25,987 P45242 INFO Train loss: 0.195759
2022-01-18 11:05:25,987 P45242 INFO ************ Epoch=5 end ************
2022-01-18 11:05:27,764 P45242 INFO [Metrics] AUC: 0.979175 - logloss: 0.175415
2022-01-18 11:05:27,765 P45242 INFO Save best model: monitor(max): 0.979175
2022-01-18 11:05:27,769 P45242 INFO --- 50/50 batches finished ---
2022-01-18 11:05:27,801 P45242 INFO Train loss: 0.189789
2022-01-18 11:05:27,801 P45242 INFO ************ Epoch=6 end ************
2022-01-18 11:05:29,577 P45242 INFO [Metrics] AUC: 0.979667 - logloss: 0.180926
2022-01-18 11:05:29,577 P45242 INFO Save best model: monitor(max): 0.979667
2022-01-18 11:05:29,581 P45242 INFO --- 50/50 batches finished ---
2022-01-18 11:05:29,613 P45242 INFO Train loss: 0.184782
2022-01-18 11:05:29,613 P45242 INFO ************ Epoch=7 end ************
2022-01-18 11:05:31,400 P45242 INFO [Metrics] AUC: 0.980814 - logloss: 0.159869
2022-01-18 11:05:31,401 P45242 INFO Save best model: monitor(max): 0.980814
2022-01-18 11:05:31,405 P45242 INFO --- 50/50 batches finished ---
2022-01-18 11:05:31,437 P45242 INFO Train loss: 0.180876
2022-01-18 11:05:31,437 P45242 INFO ************ Epoch=8 end ************
2022-01-18 11:05:33,184 P45242 INFO [Metrics] AUC: 0.980862 - logloss: 0.158233
2022-01-18 11:05:33,185 P45242 INFO Save best model: monitor(max): 0.980862
2022-01-18 11:05:33,189 P45242 INFO --- 50/50 batches finished ---
2022-01-18 11:05:33,220 P45242 INFO Train loss: 0.179513
2022-01-18 11:05:33,220 P45242 INFO ************ Epoch=9 end ************
2022-01-18 11:05:34,983 P45242 INFO [Metrics] AUC: 0.979387 - logloss: 0.190057
2022-01-18 11:05:34,984 P45242 INFO Monitor(max) STOP: 0.979387 !
2022-01-18 11:05:34,984 P45242 INFO Reduce learning rate on plateau: 0.000100
2022-01-18 11:05:34,984 P45242 INFO --- 50/50 batches finished ---
2022-01-18 11:05:35,013 P45242 INFO Train loss: 0.176229
2022-01-18 11:05:35,013 P45242 INFO ************ Epoch=10 end ************
2022-01-18 11:05:36,786 P45242 INFO [Metrics] AUC: 0.983483 - logloss: 0.145844
2022-01-18 11:05:36,787 P45242 INFO Save best model: monitor(max): 0.983483
2022-01-18 11:05:36,792 P45242 INFO --- 50/50 batches finished ---
2022-01-18 11:05:36,824 P45242 INFO Train loss: 0.147572
2022-01-18 11:05:36,824 P45242 INFO ************ Epoch=11 end ************
2022-01-18 11:05:38,632 P45242 INFO [Metrics] AUC: 0.984231 - logloss: 0.143599
2022-01-18 11:05:38,632 P45242 INFO Save best model: monitor(max): 0.984231
2022-01-18 11:05:38,637 P45242 INFO --- 50/50 batches finished ---
2022-01-18 11:05:38,667 P45242 INFO Train loss: 0.123830
2022-01-18 11:05:38,668 P45242 INFO ************ Epoch=12 end ************
2022-01-18 11:05:40,454 P45242 INFO [Metrics] AUC: 0.984478 - logloss: 0.145871
2022-01-18 11:05:40,454 P45242 INFO Save best model: monitor(max): 0.984478
2022-01-18 11:05:40,459 P45242 INFO --- 50/50 batches finished ---
2022-01-18 11:05:40,497 P45242 INFO Train loss: 0.110417
2022-01-18 11:05:40,497 P45242 INFO ************ Epoch=13 end ************
2022-01-18 11:05:42,302 P45242 INFO [Metrics] AUC: 0.984487 - logloss: 0.148075
2022-01-18 11:05:42,302 P45242 INFO Save best model: monitor(max): 0.984487
2022-01-18 11:05:42,307 P45242 INFO --- 50/50 batches finished ---
2022-01-18 11:05:42,340 P45242 INFO Train loss: 0.100231
2022-01-18 11:05:42,340 P45242 INFO ************ Epoch=14 end ************
2022-01-18 11:05:44,138 P45242 INFO [Metrics] AUC: 0.984547 - logloss: 0.150780
2022-01-18 11:05:44,138 P45242 INFO Save best model: monitor(max): 0.984547
2022-01-18 11:05:44,143 P45242 INFO --- 50/50 batches finished ---
2022-01-18 11:05:44,176 P45242 INFO Train loss: 0.092173
2022-01-18 11:05:44,176 P45242 INFO ************ Epoch=15 end ************
2022-01-18 11:05:45,931 P45242 INFO [Metrics] AUC: 0.984470 - logloss: 0.153489
2022-01-18 11:05:45,932 P45242 INFO Monitor(max) STOP: 0.984470 !
2022-01-18 11:05:45,932 P45242 INFO Reduce learning rate on plateau: 0.000010
2022-01-18 11:05:45,932 P45242 INFO --- 50/50 batches finished ---
2022-01-18 11:05:45,965 P45242 INFO Train loss: 0.085723
2022-01-18 11:05:45,966 P45242 INFO ************ Epoch=16 end ************
2022-01-18 11:05:47,765 P45242 INFO [Metrics] AUC: 0.984533 - logloss: 0.153404
2022-01-18 11:05:47,765 P45242 INFO Monitor(max) STOP: 0.984533 !
2022-01-18 11:05:47,765 P45242 INFO Reduce learning rate on plateau: 0.000001
2022-01-18 11:05:47,765 P45242 INFO Early stopping at epoch=17
2022-01-18 11:05:47,765 P45242 INFO --- 50/50 batches finished ---
2022-01-18 11:05:47,796 P45242 INFO Train loss: 0.079403
2022-01-18 11:05:47,796 P45242 INFO Training finished.
2022-01-18 11:05:47,796 P45242 INFO Load best model: /home/XXX/FuxiCTR_github_v1.1/benchmarks/Frappe/DCN_frappe_x1/frappe_x1_04e961e9/DCN_frappe_x1_013_efa58c31.model
2022-01-18 11:05:51,836 P45242 INFO ****** Validation evaluation ******
2022-01-18 11:05:52,259 P45242 INFO [Metrics] AUC: 0.984547 - logloss: 0.150780
2022-01-18 11:05:52,287 P45242 INFO ******** Test evaluation ********
2022-01-18 11:05:52,287 P45242 INFO Loading data...
2022-01-18 11:05:52,288 P45242 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-01-18 11:05:52,290 P45242 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-01-18 11:05:52,290 P45242 INFO Loading test data done.
2022-01-18 11:05:52,534 P45242 INFO [Metrics] AUC: 0.983907 - logloss: 0.152660

```
