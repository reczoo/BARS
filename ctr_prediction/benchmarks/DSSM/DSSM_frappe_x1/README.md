## DSSM_frappe_x1

A hands-on guide to run the DSSM model on the Frappe_x1 dataset.

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

We use [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [DSSM](https://github.com/xue-pai/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/DSSM.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DSSM_frappe_x1_tuner_config_02](./DSSM_frappe_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DSSM_frappe_x1
    nohup python run_expid.py --config ./DSSM_frappe_x1_tuner_config_02 --expid DSSM_frappe_x1_006_4be388eb --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.983221 | 0.187438  |


### Logs
```python
2022-04-13 11:55:40,527 P31863 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_db5a3f58",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.1",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'daytime', 'weekday', 'isweekend', 'homework', 'weather', 'country', 'city'], 'source': 'user', 'type': 'categorical'}, {'active': True, 'dtype': 'float', 'name': ['item', 'cost'], 'source': 'item', 'type': 'categorical'}]",
    "gpu": "1",
    "item_tower_activations": "ReLU",
    "item_tower_dropout": "0.2",
    "item_tower_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DSSM",
    "model_id": "DSSM_frappe_x1_006_4be388eb",
    "model_root": "./Frappe/DSSM_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
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
    "user_tower_activations": "ReLU",
    "user_tower_dropout": "0.1",
    "user_tower_units": "[400, 400, 400]",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-04-13 11:55:40,528 P31863 INFO Set up feature encoder...
2022-04-13 11:55:40,528 P31863 INFO Load feature_map from json: ../data/Frappe/frappe_x1_db5a3f58/feature_map.json
2022-04-13 11:55:40,528 P31863 INFO Loading data...
2022-04-13 11:55:40,531 P31863 INFO Loading data from h5: ../data/Frappe/frappe_x1_db5a3f58/train.h5
2022-04-13 11:55:40,542 P31863 INFO Loading data from h5: ../data/Frappe/frappe_x1_db5a3f58/valid.h5
2022-04-13 11:55:40,546 P31863 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-04-13 11:55:40,546 P31863 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-04-13 11:55:40,546 P31863 INFO Loading train data done.
2022-04-13 11:55:44,684 P31863 INFO Total number of parameters: 739490.
2022-04-13 11:55:44,685 P31863 INFO Start training: 50 batches/epoch
2022-04-13 11:55:44,685 P31863 INFO ************ Epoch=1 start ************
2022-04-13 11:55:50,880 P31863 INFO [Metrics] AUC: 0.900741 - logloss: 0.611823
2022-04-13 11:55:50,881 P31863 INFO Save best model: monitor(max): 0.900741
2022-04-13 11:55:50,886 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:55:50,929 P31863 INFO Train loss: 0.650903
2022-04-13 11:55:50,929 P31863 INFO ************ Epoch=1 end ************
2022-04-13 11:55:56,885 P31863 INFO [Metrics] AUC: 0.954946 - logloss: 0.279399
2022-04-13 11:55:56,886 P31863 INFO Save best model: monitor(max): 0.954946
2022-04-13 11:55:56,892 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:55:56,927 P31863 INFO Train loss: 0.331125
2022-04-13 11:55:56,927 P31863 INFO ************ Epoch=2 end ************
2022-04-13 11:56:02,535 P31863 INFO [Metrics] AUC: 0.967419 - logloss: 0.216603
2022-04-13 11:56:02,535 P31863 INFO Save best model: monitor(max): 0.967419
2022-04-13 11:56:02,542 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:56:02,583 P31863 INFO Train loss: 0.267234
2022-04-13 11:56:02,584 P31863 INFO ************ Epoch=3 end ************
2022-04-13 11:56:07,801 P31863 INFO [Metrics] AUC: 0.972068 - logloss: 0.210400
2022-04-13 11:56:07,801 P31863 INFO Save best model: monitor(max): 0.972068
2022-04-13 11:56:07,810 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:56:07,852 P31863 INFO Train loss: 0.241815
2022-04-13 11:56:07,852 P31863 INFO ************ Epoch=4 end ************
2022-04-13 11:56:13,194 P31863 INFO [Metrics] AUC: 0.973752 - logloss: 0.220257
2022-04-13 11:56:13,195 P31863 INFO Save best model: monitor(max): 0.973752
2022-04-13 11:56:13,201 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:56:13,242 P31863 INFO Train loss: 0.227181
2022-04-13 11:56:13,243 P31863 INFO ************ Epoch=5 end ************
2022-04-13 11:56:18,483 P31863 INFO [Metrics] AUC: 0.974975 - logloss: 0.189582
2022-04-13 11:56:18,484 P31863 INFO Save best model: monitor(max): 0.974975
2022-04-13 11:56:18,490 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:56:18,535 P31863 INFO Train loss: 0.219095
2022-04-13 11:56:18,535 P31863 INFO ************ Epoch=6 end ************
2022-04-13 11:56:23,707 P31863 INFO [Metrics] AUC: 0.976341 - logloss: 0.204962
2022-04-13 11:56:23,709 P31863 INFO Save best model: monitor(max): 0.976341
2022-04-13 11:56:23,718 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:56:23,761 P31863 INFO Train loss: 0.212992
2022-04-13 11:56:23,761 P31863 INFO ************ Epoch=7 end ************
2022-04-13 11:56:28,970 P31863 INFO [Metrics] AUC: 0.976750 - logloss: 0.180353
2022-04-13 11:56:28,971 P31863 INFO Save best model: monitor(max): 0.976750
2022-04-13 11:56:28,979 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:56:29,016 P31863 INFO Train loss: 0.210784
2022-04-13 11:56:29,016 P31863 INFO ************ Epoch=8 end ************
2022-04-13 11:56:34,938 P31863 INFO [Metrics] AUC: 0.977987 - logloss: 0.182927
2022-04-13 11:56:34,939 P31863 INFO Save best model: monitor(max): 0.977987
2022-04-13 11:56:34,946 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:56:34,992 P31863 INFO Train loss: 0.204913
2022-04-13 11:56:34,992 P31863 INFO ************ Epoch=9 end ************
2022-04-13 11:56:41,175 P31863 INFO [Metrics] AUC: 0.976822 - logloss: 0.199739
2022-04-13 11:56:41,176 P31863 INFO Monitor(max) STOP: 0.976822 !
2022-04-13 11:56:41,176 P31863 INFO Reduce learning rate on plateau: 0.000100
2022-04-13 11:56:41,176 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:56:41,221 P31863 INFO Train loss: 0.205779
2022-04-13 11:56:41,221 P31863 INFO ************ Epoch=10 end ************
2022-04-13 11:56:47,388 P31863 INFO [Metrics] AUC: 0.982380 - logloss: 0.158879
2022-04-13 11:56:47,389 P31863 INFO Save best model: monitor(max): 0.982380
2022-04-13 11:56:47,408 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:56:47,469 P31863 INFO Train loss: 0.173535
2022-04-13 11:56:47,469 P31863 INFO ************ Epoch=11 end ************
2022-04-13 11:56:53,624 P31863 INFO [Metrics] AUC: 0.983463 - logloss: 0.158115
2022-04-13 11:56:53,625 P31863 INFO Save best model: monitor(max): 0.983463
2022-04-13 11:56:53,634 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:56:53,681 P31863 INFO Train loss: 0.144106
2022-04-13 11:56:53,681 P31863 INFO ************ Epoch=12 end ************
2022-04-13 11:56:59,380 P31863 INFO [Metrics] AUC: 0.984005 - logloss: 0.159751
2022-04-13 11:56:59,381 P31863 INFO Save best model: monitor(max): 0.984005
2022-04-13 11:56:59,387 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:56:59,438 P31863 INFO Train loss: 0.126666
2022-04-13 11:56:59,438 P31863 INFO ************ Epoch=13 end ************
2022-04-13 11:57:04,229 P31863 INFO [Metrics] AUC: 0.984054 - logloss: 0.165974
2022-04-13 11:57:04,230 P31863 INFO Save best model: monitor(max): 0.984054
2022-04-13 11:57:04,236 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:57:04,274 P31863 INFO Train loss: 0.113567
2022-04-13 11:57:04,274 P31863 INFO ************ Epoch=14 end ************
2022-04-13 11:57:08,970 P31863 INFO [Metrics] AUC: 0.984203 - logloss: 0.169526
2022-04-13 11:57:08,970 P31863 INFO Save best model: monitor(max): 0.984203
2022-04-13 11:57:08,976 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:57:09,015 P31863 INFO Train loss: 0.102512
2022-04-13 11:57:09,015 P31863 INFO ************ Epoch=15 end ************
2022-04-13 11:57:14,990 P31863 INFO [Metrics] AUC: 0.984106 - logloss: 0.174788
2022-04-13 11:57:14,990 P31863 INFO Monitor(max) STOP: 0.984106 !
2022-04-13 11:57:14,990 P31863 INFO Reduce learning rate on plateau: 0.000010
2022-04-13 11:57:14,991 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:57:15,030 P31863 INFO Train loss: 0.095254
2022-04-13 11:57:15,030 P31863 INFO ************ Epoch=16 end ************
2022-04-13 11:57:19,600 P31863 INFO [Metrics] AUC: 0.984262 - logloss: 0.175753
2022-04-13 11:57:19,601 P31863 INFO Save best model: monitor(max): 0.984262
2022-04-13 11:57:19,610 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:57:19,646 P31863 INFO Train loss: 0.087010
2022-04-13 11:57:19,646 P31863 INFO ************ Epoch=17 end ************
2022-04-13 11:57:24,089 P31863 INFO [Metrics] AUC: 0.984265 - logloss: 0.177004
2022-04-13 11:57:24,090 P31863 INFO Save best model: monitor(max): 0.984265
2022-04-13 11:57:24,099 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:57:24,141 P31863 INFO Train loss: 0.084949
2022-04-13 11:57:24,141 P31863 INFO ************ Epoch=18 end ************
2022-04-13 11:57:28,611 P31863 INFO [Metrics] AUC: 0.984269 - logloss: 0.176973
2022-04-13 11:57:28,612 P31863 INFO Save best model: monitor(max): 0.984269
2022-04-13 11:57:28,620 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:57:28,666 P31863 INFO Train loss: 0.084518
2022-04-13 11:57:28,666 P31863 INFO ************ Epoch=19 end ************
2022-04-13 11:57:34,168 P31863 INFO [Metrics] AUC: 0.984280 - logloss: 0.178141
2022-04-13 11:57:34,168 P31863 INFO Save best model: monitor(max): 0.984280
2022-04-13 11:57:34,174 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:57:34,225 P31863 INFO Train loss: 0.083657
2022-04-13 11:57:34,225 P31863 INFO ************ Epoch=20 end ************
2022-04-13 11:57:39,887 P31863 INFO [Metrics] AUC: 0.984285 - logloss: 0.179113
2022-04-13 11:57:39,888 P31863 INFO Save best model: monitor(max): 0.984285
2022-04-13 11:57:39,896 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:57:39,940 P31863 INFO Train loss: 0.082723
2022-04-13 11:57:39,940 P31863 INFO ************ Epoch=21 end ************
2022-04-13 11:57:45,701 P31863 INFO [Metrics] AUC: 0.984264 - logloss: 0.179959
2022-04-13 11:57:45,702 P31863 INFO Monitor(max) STOP: 0.984264 !
2022-04-13 11:57:45,702 P31863 INFO Reduce learning rate on plateau: 0.000001
2022-04-13 11:57:45,702 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:57:45,746 P31863 INFO Train loss: 0.081612
2022-04-13 11:57:45,746 P31863 INFO ************ Epoch=22 end ************
2022-04-13 11:57:51,619 P31863 INFO [Metrics] AUC: 0.984255 - logloss: 0.180367
2022-04-13 11:57:51,620 P31863 INFO Monitor(max) STOP: 0.984255 !
2022-04-13 11:57:51,620 P31863 INFO Reduce learning rate on plateau: 0.000001
2022-04-13 11:57:51,621 P31863 INFO Early stopping at epoch=23
2022-04-13 11:57:51,621 P31863 INFO --- 50/50 batches finished ---
2022-04-13 11:57:51,662 P31863 INFO Train loss: 0.080517
2022-04-13 11:57:51,662 P31863 INFO Training finished.
2022-04-13 11:57:51,662 P31863 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Frappe/DSSM_frappe_x1/frappe_x1_db5a3f58/DSSM_frappe_x1_006_4be388eb.model
2022-04-13 11:57:55,845 P31863 INFO ****** Validation evaluation ******
2022-04-13 11:57:56,255 P31863 INFO [Metrics] AUC: 0.984285 - logloss: 0.179113
2022-04-13 11:57:56,314 P31863 INFO ******** Test evaluation ********
2022-04-13 11:57:56,314 P31863 INFO Loading data...
2022-04-13 11:57:56,315 P31863 INFO Loading data from h5: ../data/Frappe/frappe_x1_db5a3f58/test.h5
2022-04-13 11:57:56,318 P31863 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-04-13 11:57:56,319 P31863 INFO Loading test data done.
2022-04-13 11:57:56,617 P31863 INFO [Metrics] AUC: 0.983221 - logloss: 0.187438

```
