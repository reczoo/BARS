## HFM+_frappe_x1

A hands-on guide to run the HFM model on the Frappe_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [HFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/HFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HFM+_frappe_x1_tuner_config_04](./HFM+_frappe_x1_tuner_config_04). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HFM+_frappe_x1
    nohup python run_expid.py --config ./HFM+_frappe_x1_tuner_config_04 --expid HFM_frappe_x1_006_0caf1c14 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.982675 | 0.154335  |


### Logs
```python
2022-02-02 21:30:16,322 P34661 INFO {
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
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "interaction_type": "circular_correlation",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "HFM",
    "model_id": "HFM_frappe_x1_006_0caf1c14",
    "model_root": "./Frappe/HFM_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.4",
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
    "use_dnn": "True",
    "use_hdf5": "True",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-02-02 21:30:16,330 P34661 INFO Set up feature encoder...
2022-02-02 21:30:16,332 P34661 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-02-02 21:30:16,332 P34661 INFO Loading data...
2022-02-02 21:30:16,336 P34661 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-02-02 21:30:16,374 P34661 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-02-02 21:30:16,380 P34661 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-02-02 21:30:16,385 P34661 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-02-02 21:30:16,385 P34661 INFO Loading train data done.
2022-02-02 21:30:27,274 P34661 INFO Total number of parameters: 563281.
2022-02-02 21:30:27,275 P34661 INFO Start training: 50 batches/epoch
2022-02-02 21:30:27,275 P34661 INFO ************ Epoch=1 start ************
2022-02-02 21:30:38,838 P34661 INFO [Metrics] AUC: 0.942422 - logloss: 0.624702
2022-02-02 21:30:38,839 P34661 INFO Save best model: monitor(max): 0.942422
2022-02-02 21:30:38,858 P34661 INFO --- 50/50 batches finished ---
2022-02-02 21:30:38,973 P34661 INFO Train loss: 0.389045
2022-02-02 21:30:38,973 P34661 INFO ************ Epoch=1 end ************
2022-02-02 21:30:49,254 P34661 INFO [Metrics] AUC: 0.958303 - logloss: 0.288443
2022-02-02 21:30:49,255 P34661 INFO Save best model: monitor(max): 0.958303
2022-02-02 21:30:49,281 P34661 INFO --- 50/50 batches finished ---
2022-02-02 21:30:49,423 P34661 INFO Train loss: 0.277592
2022-02-02 21:30:49,423 P34661 INFO ************ Epoch=2 end ************
2022-02-02 21:31:00,206 P34661 INFO [Metrics] AUC: 0.972996 - logloss: 0.198972
2022-02-02 21:31:00,207 P34661 INFO Save best model: monitor(max): 0.972996
2022-02-02 21:31:00,225 P34661 INFO --- 50/50 batches finished ---
2022-02-02 21:31:00,352 P34661 INFO Train loss: 0.232715
2022-02-02 21:31:00,353 P34661 INFO ************ Epoch=3 end ************
2022-02-02 21:31:10,312 P34661 INFO [Metrics] AUC: 0.977378 - logloss: 0.175602
2022-02-02 21:31:10,312 P34661 INFO Save best model: monitor(max): 0.977378
2022-02-02 21:31:10,339 P34661 INFO --- 50/50 batches finished ---
2022-02-02 21:31:10,475 P34661 INFO Train loss: 0.198968
2022-02-02 21:31:10,476 P34661 INFO ************ Epoch=4 end ************
2022-02-02 21:31:21,094 P34661 INFO [Metrics] AUC: 0.978819 - logloss: 0.182541
2022-02-02 21:31:21,100 P34661 INFO Save best model: monitor(max): 0.978819
2022-02-02 21:31:21,112 P34661 INFO --- 50/50 batches finished ---
2022-02-02 21:31:21,250 P34661 INFO Train loss: 0.182939
2022-02-02 21:31:21,250 P34661 INFO ************ Epoch=5 end ************
2022-02-02 21:31:30,414 P34661 INFO [Metrics] AUC: 0.978835 - logloss: 0.220238
2022-02-02 21:31:30,414 P34661 INFO Save best model: monitor(max): 0.978835
2022-02-02 21:31:30,431 P34661 INFO --- 50/50 batches finished ---
2022-02-02 21:31:30,625 P34661 INFO Train loss: 0.176603
2022-02-02 21:31:30,626 P34661 INFO ************ Epoch=6 end ************
2022-02-02 21:31:40,295 P34661 INFO [Metrics] AUC: 0.977392 - logloss: 0.306929
2022-02-02 21:31:40,295 P34661 INFO Monitor(max) STOP: 0.977392 !
2022-02-02 21:31:40,295 P34661 INFO Reduce learning rate on plateau: 0.000100
2022-02-02 21:31:40,296 P34661 INFO --- 50/50 batches finished ---
2022-02-02 21:31:40,517 P34661 INFO Train loss: 0.174094
2022-02-02 21:31:40,517 P34661 INFO ************ Epoch=7 end ************
2022-02-02 21:31:51,562 P34661 INFO [Metrics] AUC: 0.981908 - logloss: 0.154221
2022-02-02 21:31:51,562 P34661 INFO Save best model: monitor(max): 0.981908
2022-02-02 21:31:51,581 P34661 INFO --- 50/50 batches finished ---
2022-02-02 21:31:51,682 P34661 INFO Train loss: 0.140431
2022-02-02 21:31:51,682 P34661 INFO ************ Epoch=8 end ************
2022-02-02 21:32:01,501 P34661 INFO [Metrics] AUC: 0.982303 - logloss: 0.157156
2022-02-02 21:32:01,501 P34661 INFO Save best model: monitor(max): 0.982303
2022-02-02 21:32:01,519 P34661 INFO --- 50/50 batches finished ---
2022-02-02 21:32:01,711 P34661 INFO Train loss: 0.116717
2022-02-02 21:32:01,711 P34661 INFO ************ Epoch=9 end ************
2022-02-02 21:32:11,837 P34661 INFO [Metrics] AUC: 0.982127 - logloss: 0.165557
2022-02-02 21:32:11,838 P34661 INFO Monitor(max) STOP: 0.982127 !
2022-02-02 21:32:11,838 P34661 INFO Reduce learning rate on plateau: 0.000010
2022-02-02 21:32:11,838 P34661 INFO --- 50/50 batches finished ---
2022-02-02 21:32:11,973 P34661 INFO Train loss: 0.101047
2022-02-02 21:32:11,973 P34661 INFO ************ Epoch=10 end ************
2022-02-02 21:32:22,321 P34661 INFO [Metrics] AUC: 0.982167 - logloss: 0.166114
2022-02-02 21:32:22,321 P34661 INFO Monitor(max) STOP: 0.982167 !
2022-02-02 21:32:22,321 P34661 INFO Reduce learning rate on plateau: 0.000001
2022-02-02 21:32:22,322 P34661 INFO Early stopping at epoch=11
2022-02-02 21:32:22,322 P34661 INFO --- 50/50 batches finished ---
2022-02-02 21:32:22,520 P34661 INFO Train loss: 0.091162
2022-02-02 21:32:22,520 P34661 INFO Training finished.
2022-02-02 21:32:22,520 P34661 INFO Load best model: /home/XXX/benchmarks/Frappe/HFM_frappe_x1/frappe_x1_04e961e9/HFM_frappe_x1_006_0caf1c14.model
2022-02-02 21:32:29,965 P34661 INFO ****** Validation evaluation ******
2022-02-02 21:32:32,875 P34661 INFO [Metrics] AUC: 0.982303 - logloss: 0.157156
2022-02-02 21:32:33,048 P34661 INFO ******** Test evaluation ********
2022-02-02 21:32:33,048 P34661 INFO Loading data...
2022-02-02 21:32:33,049 P34661 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-02-02 21:32:33,062 P34661 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-02-02 21:32:33,062 P34661 INFO Loading test data done.
2022-02-02 21:32:34,359 P34661 INFO [Metrics] AUC: 0.982675 - logloss: 0.154335

```
