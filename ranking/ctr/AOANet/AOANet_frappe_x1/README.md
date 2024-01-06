## AOANet_frappe_x1

A hands-on guide to run the AOANet model on the Frappe_x1 dataset.

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
  fuxictr: 1.2.1
  ```

### Dataset
Dataset ID: [Frappe_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Frappe#Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [AOANet](https://github.com/xue-pai/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/AOANet.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AOANet_frappe_x1_tuner_config_02](./AOANet_frappe_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AOANet_frappe_x1
    nohup python run_expid.py --config ./AOANet_frappe_x1_tuner_config_02 --expid AOANet_frappe_x1_009_29c57772 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.984400 | 0.142379  |


### Logs
```python
2022-04-13 00:36:26,508 P32306 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "dnn_hidden_activations": "ReLU",
    "dnn_hidden_units": "[400, 400, 400]",
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
    "model": "AOANet",
    "model_id": "AOANet_frappe_x1_009_29c57772",
    "model_root": "./Frappe/AOANet_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_interaction_layers": "1",
    "num_subspaces": "2",
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
2022-04-13 00:36:26,509 P32306 INFO Set up feature encoder...
2022-04-13 00:36:26,509 P32306 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-04-13 00:36:26,510 P32306 INFO Loading data...
2022-04-13 00:36:26,513 P32306 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-04-13 00:36:26,525 P32306 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-04-13 00:36:26,530 P32306 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-04-13 00:36:26,530 P32306 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-04-13 00:36:26,530 P32306 INFO Loading train data done.
2022-04-13 00:36:30,988 P32306 INFO Total number of parameters: 418331.
2022-04-13 00:36:30,989 P32306 INFO Start training: 50 batches/epoch
2022-04-13 00:36:30,989 P32306 INFO ************ Epoch=1 start ************
2022-04-13 00:36:39,671 P32306 INFO [Metrics] AUC: 0.936314 - logloss: 0.727013
2022-04-13 00:36:39,672 P32306 INFO Save best model: monitor(max): 0.936314
2022-04-13 00:36:39,676 P32306 INFO --- 50/50 batches finished ---
2022-04-13 00:36:39,712 P32306 INFO Train loss: 0.394749
2022-04-13 00:36:39,712 P32306 INFO ************ Epoch=1 end ************
2022-04-13 00:36:48,472 P32306 INFO [Metrics] AUC: 0.954446 - logloss: 0.268002
2022-04-13 00:36:48,473 P32306 INFO Save best model: monitor(max): 0.954446
2022-04-13 00:36:48,479 P32306 INFO --- 50/50 batches finished ---
2022-04-13 00:36:48,520 P32306 INFO Train loss: 0.293335
2022-04-13 00:36:48,520 P32306 INFO ************ Epoch=2 end ************
2022-04-13 00:36:55,561 P32306 INFO [Metrics] AUC: 0.966939 - logloss: 0.216872
2022-04-13 00:36:55,562 P32306 INFO Save best model: monitor(max): 0.966939
2022-04-13 00:36:55,566 P32306 INFO --- 50/50 batches finished ---
2022-04-13 00:36:55,602 P32306 INFO Train loss: 0.262797
2022-04-13 00:36:55,602 P32306 INFO ************ Epoch=3 end ************
2022-04-13 00:37:02,344 P32306 INFO [Metrics] AUC: 0.972106 - logloss: 0.190556
2022-04-13 00:37:02,345 P32306 INFO Save best model: monitor(max): 0.972106
2022-04-13 00:37:02,351 P32306 INFO --- 50/50 batches finished ---
2022-04-13 00:37:02,389 P32306 INFO Train loss: 0.246463
2022-04-13 00:37:02,389 P32306 INFO ************ Epoch=4 end ************
2022-04-13 00:37:11,055 P32306 INFO [Metrics] AUC: 0.973515 - logloss: 0.188421
2022-04-13 00:37:11,055 P32306 INFO Save best model: monitor(max): 0.973515
2022-04-13 00:37:11,061 P32306 INFO --- 50/50 batches finished ---
2022-04-13 00:37:11,115 P32306 INFO Train loss: 0.237570
2022-04-13 00:37:11,116 P32306 INFO ************ Epoch=5 end ************
2022-04-13 00:37:19,571 P32306 INFO [Metrics] AUC: 0.974107 - logloss: 0.228146
2022-04-13 00:37:19,572 P32306 INFO Save best model: monitor(max): 0.974107
2022-04-13 00:37:19,579 P32306 INFO --- 50/50 batches finished ---
2022-04-13 00:37:19,623 P32306 INFO Train loss: 0.232030
2022-04-13 00:37:19,623 P32306 INFO ************ Epoch=6 end ************
2022-04-13 00:37:28,111 P32306 INFO [Metrics] AUC: 0.973322 - logloss: 0.190663
2022-04-13 00:37:28,112 P32306 INFO Monitor(max) STOP: 0.973322 !
2022-04-13 00:37:28,112 P32306 INFO Reduce learning rate on plateau: 0.000100
2022-04-13 00:37:28,112 P32306 INFO --- 50/50 batches finished ---
2022-04-13 00:37:28,165 P32306 INFO Train loss: 0.227103
2022-04-13 00:37:28,165 P32306 INFO ************ Epoch=7 end ************
2022-04-13 00:37:36,660 P32306 INFO [Metrics] AUC: 0.981090 - logloss: 0.150912
2022-04-13 00:37:36,661 P32306 INFO Save best model: monitor(max): 0.981090
2022-04-13 00:37:36,665 P32306 INFO --- 50/50 batches finished ---
2022-04-13 00:37:36,711 P32306 INFO Train loss: 0.193946
2022-04-13 00:37:36,711 P32306 INFO ************ Epoch=8 end ************
2022-04-13 00:37:45,232 P32306 INFO [Metrics] AUC: 0.983051 - logloss: 0.143658
2022-04-13 00:37:45,233 P32306 INFO Save best model: monitor(max): 0.983051
2022-04-13 00:37:45,238 P32306 INFO --- 50/50 batches finished ---
2022-04-13 00:37:45,296 P32306 INFO Train loss: 0.158726
2022-04-13 00:37:45,296 P32306 INFO ************ Epoch=9 end ************
2022-04-13 00:37:53,806 P32306 INFO [Metrics] AUC: 0.983878 - logloss: 0.142119
2022-04-13 00:37:53,807 P32306 INFO Save best model: monitor(max): 0.983878
2022-04-13 00:37:53,813 P32306 INFO --- 50/50 batches finished ---
2022-04-13 00:37:53,867 P32306 INFO Train loss: 0.136406
2022-04-13 00:37:53,867 P32306 INFO ************ Epoch=10 end ************
2022-04-13 00:38:02,417 P32306 INFO [Metrics] AUC: 0.984196 - logloss: 0.142770
2022-04-13 00:38:02,418 P32306 INFO Save best model: monitor(max): 0.984196
2022-04-13 00:38:02,424 P32306 INFO --- 50/50 batches finished ---
2022-04-13 00:38:02,465 P32306 INFO Train loss: 0.120680
2022-04-13 00:38:02,465 P32306 INFO ************ Epoch=11 end ************
2022-04-13 00:38:11,098 P32306 INFO [Metrics] AUC: 0.984486 - logloss: 0.142689
2022-04-13 00:38:11,099 P32306 INFO Save best model: monitor(max): 0.984486
2022-04-13 00:38:11,105 P32306 INFO --- 50/50 batches finished ---
2022-04-13 00:38:11,153 P32306 INFO Train loss: 0.108975
2022-04-13 00:38:11,153 P32306 INFO ************ Epoch=12 end ************
2022-04-13 00:38:19,736 P32306 INFO [Metrics] AUC: 0.984972 - logloss: 0.142321
2022-04-13 00:38:19,736 P32306 INFO Save best model: monitor(max): 0.984972
2022-04-13 00:38:19,742 P32306 INFO --- 50/50 batches finished ---
2022-04-13 00:38:19,792 P32306 INFO Train loss: 0.100400
2022-04-13 00:38:19,792 P32306 INFO ************ Epoch=13 end ************
2022-04-13 00:38:28,465 P32306 INFO [Metrics] AUC: 0.984567 - logloss: 0.146438
2022-04-13 00:38:28,466 P32306 INFO Monitor(max) STOP: 0.984567 !
2022-04-13 00:38:28,466 P32306 INFO Reduce learning rate on plateau: 0.000010
2022-04-13 00:38:28,466 P32306 INFO --- 50/50 batches finished ---
2022-04-13 00:38:28,503 P32306 INFO Train loss: 0.094982
2022-04-13 00:38:28,503 P32306 INFO ************ Epoch=14 end ************
2022-04-13 00:38:36,986 P32306 INFO [Metrics] AUC: 0.984691 - logloss: 0.146166
2022-04-13 00:38:36,987 P32306 INFO Monitor(max) STOP: 0.984691 !
2022-04-13 00:38:36,987 P32306 INFO Reduce learning rate on plateau: 0.000001
2022-04-13 00:38:36,987 P32306 INFO Early stopping at epoch=15
2022-04-13 00:38:36,987 P32306 INFO --- 50/50 batches finished ---
2022-04-13 00:38:37,021 P32306 INFO Train loss: 0.085368
2022-04-13 00:38:37,021 P32306 INFO Training finished.
2022-04-13 00:38:37,021 P32306 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Frappe/AOANet_frappe_x1/frappe_x1_04e961e9/AOANet_frappe_x1_009_29c57772.model
2022-04-13 00:38:37,074 P32306 INFO ****** Validation evaluation ******
2022-04-13 00:38:37,550 P32306 INFO [Metrics] AUC: 0.984972 - logloss: 0.142321
2022-04-13 00:38:37,583 P32306 INFO ******** Test evaluation ********
2022-04-13 00:38:37,583 P32306 INFO Loading data...
2022-04-13 00:38:37,584 P32306 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-04-13 00:38:37,587 P32306 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-04-13 00:38:37,587 P32306 INFO Loading test data done.
2022-04-13 00:38:37,956 P32306 INFO [Metrics] AUC: 0.984400 - logloss: 0.142379

```
