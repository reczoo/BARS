## DeepIM_frappe_x1

A hands-on guide to run the DeepIM model on the Frappe_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DeepIM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DeepIM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DeepIM_frappe_x1_tuner_config_01](./DeepIM_frappe_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd DeepIM_frappe_x1
   nohup python run_expid.py --config ./DeepIM_frappe_x1_tuner_config_01 --expid DeepIM_frappe_x1_005_a11ff117 --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

Total 5 runs:

| Runs | AUC              | logloss          |
|:----:|:----------------:|:----------------:|
| 1    | 0.984448         | 0.149010         |
| 2    | 0.983098         | 0.152701         |
| 3    | 0.984255         | 0.142934         |
| 4    | 0.982874         | 0.158292         |
| 5    | 0.983420         | 0.145777         |
| Avg  | 0.983619         | 0.149743         |
| Std  | &#177;0.00062575 | &#177;0.00537520 |

### Logs

```python
2022-11-04 14:57:37,249 P14425 INFO {
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
    "im_batch_norm": "False",
    "im_order": "3",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DeepIM",
    "model_id": "DeepIM_frappe_x1_005_a11ff117",
    "model_root": "./Frappe/DeepIM_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_batch_norm": "True",
    "net_dropout": "0.1",
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
2022-11-04 14:57:37,249 P14425 INFO Set up feature encoder...
2022-11-04 14:57:37,250 P14425 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-11-04 14:57:37,250 P14425 INFO Loading data...
2022-11-04 14:57:37,252 P14425 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-11-04 14:57:37,262 P14425 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-11-04 14:57:37,266 P14425 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-11-04 14:57:37,266 P14425 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-11-04 14:57:37,266 P14425 INFO Loading train data done.
2022-11-04 14:57:41,206 P14425 INFO Total number of parameters: 417922.
2022-11-04 14:57:41,207 P14425 INFO Start training: 50 batches/epoch
2022-11-04 14:57:41,207 P14425 INFO ************ Epoch=1 start ************
2022-11-04 14:57:45,325 P14425 INFO [Metrics] AUC: 0.936303 - logloss: 0.672567
2022-11-04 14:57:45,326 P14425 INFO Save best model: monitor(max): 0.936303
2022-11-04 14:57:45,332 P14425 INFO --- 50/50 batches finished ---
2022-11-04 14:57:45,368 P14425 INFO Train loss: 0.355558
2022-11-04 14:57:45,368 P14425 INFO ************ Epoch=1 end ************
2022-11-04 14:57:49,435 P14425 INFO [Metrics] AUC: 0.964309 - logloss: 0.239890
2022-11-04 14:57:49,436 P14425 INFO Save best model: monitor(max): 0.964309
2022-11-04 14:57:49,440 P14425 INFO --- 50/50 batches finished ---
2022-11-04 14:57:49,482 P14425 INFO Train loss: 0.265729
2022-11-04 14:57:49,482 P14425 INFO ************ Epoch=2 end ************
2022-11-04 14:57:53,473 P14425 INFO [Metrics] AUC: 0.975334 - logloss: 0.183212
2022-11-04 14:57:53,473 P14425 INFO Save best model: monitor(max): 0.975334
2022-11-04 14:57:53,478 P14425 INFO --- 50/50 batches finished ---
2022-11-04 14:57:53,512 P14425 INFO Train loss: 0.216922
2022-11-04 14:57:53,513 P14425 INFO ************ Epoch=3 end ************
2022-11-04 14:57:57,424 P14425 INFO [Metrics] AUC: 0.977162 - logloss: 0.183473
2022-11-04 14:57:57,425 P14425 INFO Save best model: monitor(max): 0.977162
2022-11-04 14:57:57,431 P14425 INFO --- 50/50 batches finished ---
2022-11-04 14:57:57,466 P14425 INFO Train loss: 0.191610
2022-11-04 14:57:57,466 P14425 INFO ************ Epoch=4 end ************
2022-11-04 14:58:01,458 P14425 INFO [Metrics] AUC: 0.978407 - logloss: 0.168452
2022-11-04 14:58:01,459 P14425 INFO Save best model: monitor(max): 0.978407
2022-11-04 14:58:01,465 P14425 INFO --- 50/50 batches finished ---
2022-11-04 14:58:01,498 P14425 INFO Train loss: 0.180325
2022-11-04 14:58:01,498 P14425 INFO ************ Epoch=5 end ************
2022-11-04 14:58:05,414 P14425 INFO [Metrics] AUC: 0.979042 - logloss: 0.208605
2022-11-04 14:58:05,414 P14425 INFO Save best model: monitor(max): 0.979042
2022-11-04 14:58:05,421 P14425 INFO --- 50/50 batches finished ---
2022-11-04 14:58:05,458 P14425 INFO Train loss: 0.176669
2022-11-04 14:58:05,458 P14425 INFO ************ Epoch=6 end ************
2022-11-04 14:58:09,443 P14425 INFO [Metrics] AUC: 0.980296 - logloss: 0.162306
2022-11-04 14:58:09,444 P14425 INFO Save best model: monitor(max): 0.980296
2022-11-04 14:58:09,450 P14425 INFO --- 50/50 batches finished ---
2022-11-04 14:58:09,495 P14425 INFO Train loss: 0.168830
2022-11-04 14:58:09,495 P14425 INFO ************ Epoch=7 end ************
2022-11-04 14:58:13,399 P14425 INFO [Metrics] AUC: 0.980675 - logloss: 0.180564
2022-11-04 14:58:13,400 P14425 INFO Save best model: monitor(max): 0.980675
2022-11-04 14:58:13,404 P14425 INFO --- 50/50 batches finished ---
2022-11-04 14:58:13,436 P14425 INFO Train loss: 0.168186
2022-11-04 14:58:13,436 P14425 INFO ************ Epoch=8 end ************
2022-11-04 14:58:17,443 P14425 INFO [Metrics] AUC: 0.981857 - logloss: 0.154874
2022-11-04 14:58:17,444 P14425 INFO Save best model: monitor(max): 0.981857
2022-11-04 14:58:17,451 P14425 INFO --- 50/50 batches finished ---
2022-11-04 14:58:17,486 P14425 INFO Train loss: 0.163475
2022-11-04 14:58:17,486 P14425 INFO ************ Epoch=9 end ************
2022-11-04 14:58:21,521 P14425 INFO [Metrics] AUC: 0.981827 - logloss: 0.158594
2022-11-04 14:58:21,522 P14425 INFO Monitor(max) STOP: 0.981827 !
2022-11-04 14:58:21,522 P14425 INFO Reduce learning rate on plateau: 0.000100
2022-11-04 14:58:21,522 P14425 INFO --- 50/50 batches finished ---
2022-11-04 14:58:21,567 P14425 INFO Train loss: 0.162784
2022-11-04 14:58:21,567 P14425 INFO ************ Epoch=10 end ************
2022-11-04 14:58:25,631 P14425 INFO [Metrics] AUC: 0.984017 - logloss: 0.145236
2022-11-04 14:58:25,632 P14425 INFO Save best model: monitor(max): 0.984017
2022-11-04 14:58:25,636 P14425 INFO --- 50/50 batches finished ---
2022-11-04 14:58:25,683 P14425 INFO Train loss: 0.134405
2022-11-04 14:58:25,683 P14425 INFO ************ Epoch=11 end ************
2022-11-04 14:58:29,737 P14425 INFO [Metrics] AUC: 0.984650 - logloss: 0.142095
2022-11-04 14:58:29,738 P14425 INFO Save best model: monitor(max): 0.984650
2022-11-04 14:58:29,742 P14425 INFO --- 50/50 batches finished ---
2022-11-04 14:58:29,781 P14425 INFO Train loss: 0.112249
2022-11-04 14:58:29,781 P14425 INFO ************ Epoch=12 end ************
2022-11-04 14:58:33,820 P14425 INFO [Metrics] AUC: 0.984763 - logloss: 0.145687
2022-11-04 14:58:33,820 P14425 INFO Save best model: monitor(max): 0.984763
2022-11-04 14:58:33,825 P14425 INFO --- 50/50 batches finished ---
2022-11-04 14:58:33,871 P14425 INFO Train loss: 0.098448
2022-11-04 14:58:33,871 P14425 INFO ************ Epoch=13 end ************
2022-11-04 14:58:38,029 P14425 INFO [Metrics] AUC: 0.984863 - logloss: 0.147055
2022-11-04 14:58:38,030 P14425 INFO Save best model: monitor(max): 0.984863
2022-11-04 14:58:38,036 P14425 INFO --- 50/50 batches finished ---
2022-11-04 14:58:38,075 P14425 INFO Train loss: 0.089370
2022-11-04 14:58:38,076 P14425 INFO ************ Epoch=14 end ************
2022-11-04 14:58:41,101 P14425 INFO [Metrics] AUC: 0.984824 - logloss: 0.149619
2022-11-04 14:58:41,102 P14425 INFO Monitor(max) STOP: 0.984824 !
2022-11-04 14:58:41,102 P14425 INFO Reduce learning rate on plateau: 0.000010
2022-11-04 14:58:41,102 P14425 INFO --- 50/50 batches finished ---
2022-11-04 14:58:41,144 P14425 INFO Train loss: 0.081502
2022-11-04 14:58:41,144 P14425 INFO ************ Epoch=15 end ************
2022-11-04 14:58:44,106 P14425 INFO [Metrics] AUC: 0.984859 - logloss: 0.151014
2022-11-04 14:58:44,106 P14425 INFO Monitor(max) STOP: 0.984859 !
2022-11-04 14:58:44,107 P14425 INFO Reduce learning rate on plateau: 0.000001
2022-11-04 14:58:44,107 P14425 INFO Early stopping at epoch=16
2022-11-04 14:58:44,107 P14425 INFO --- 50/50 batches finished ---
2022-11-04 14:58:44,137 P14425 INFO Train loss: 0.075351
2022-11-04 14:58:44,137 P14425 INFO Training finished.
2022-11-04 14:58:44,137 P14425 INFO Load best model: /home/FuxiCTR/benchmarks/Frappe/DeepIM_frappe_x1/frappe_x1_04e961e9/DeepIM_frappe_x1_005_a11ff117.model
2022-11-04 14:58:44,154 P14425 INFO ****** Validation evaluation ******
2022-11-04 14:58:44,483 P14425 INFO [Metrics] AUC: 0.984863 - logloss: 0.147055
2022-11-04 14:58:44,523 P14425 INFO ******** Test evaluation ********
2022-11-04 14:58:44,523 P14425 INFO Loading data...
2022-11-04 14:58:44,523 P14425 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-11-04 14:58:44,526 P14425 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-11-04 14:58:44,526 P14425 INFO Loading test data done.
2022-11-04 14:58:44,767 P14425 INFO [Metrics] AUC: 0.984448 - logloss: 0.149010
```
