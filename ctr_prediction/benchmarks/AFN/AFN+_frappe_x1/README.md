## AFN+_frappe_x1

A hands-on guide to run the AFN model on the Frappe_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [AFN](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/AFN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFN+_frappe_x1_tuner_config_02](./AFN+_frappe_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFN+_frappe_x1
    nohup python run_expid.py --config ./AFN+_frappe_x1_tuner_config_02 --expid AFN_frappe_x1_004_d2ea60c3 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.982563 | 0.213933  |
| 2 | 0.979261 | 0.235651  |
| 3 | 0.978932 | 0.223884  |
| 4 | 0.979346 | 0.239904  |
| 5 | 0.980789 | 0.222087  |
| | | | 
| Avg | 0.980178 | 0.227092 |
| Std | &#177;0.00135257 | &#177;0.00944356 |


### Logs
```python
2022-01-30 22:35:10,425 P36916 INFO {
    "afn_activations": "relu",
    "afn_dropout": "0.4",
    "afn_hidden_units": "[400]",
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_dropout": "0.2",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.001",
    "ensemble_dnn": "True",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "logarithmic_neurons": "1000",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AFN",
    "model_id": "AFN_frappe_x1_004_d2ea60c3",
    "model_root": "./Frappe/AFN_frappe_x1/",
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
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-30 22:35:10,426 P36916 INFO Set up feature encoder...
2022-01-30 22:35:10,427 P36916 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-01-30 22:35:10,427 P36916 INFO Loading data...
2022-01-30 22:35:10,439 P36916 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-01-30 22:35:10,473 P36916 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-01-30 22:35:10,479 P36916 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-01-30 22:35:10,488 P36916 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-01-30 22:35:10,488 P36916 INFO Loading train data done.
2022-01-30 22:35:23,348 P36916 INFO Total number of parameters: 4482205.
2022-01-30 22:35:23,349 P36916 INFO Start training: 50 batches/epoch
2022-01-30 22:35:23,349 P36916 INFO ************ Epoch=1 start ************
2022-01-30 22:35:40,745 P36916 INFO [Metrics] AUC: 0.932984 - logloss: 0.305990
2022-01-30 22:35:40,746 P36916 INFO Save best model: monitor(max): 0.932984
2022-01-30 22:35:40,813 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:35:40,918 P36916 INFO Train loss: 0.406739
2022-01-30 22:35:40,918 P36916 INFO ************ Epoch=1 end ************
2022-01-30 22:35:59,688 P36916 INFO [Metrics] AUC: 0.945790 - logloss: 0.264369
2022-01-30 22:35:59,692 P36916 INFO Save best model: monitor(max): 0.945790
2022-01-30 22:35:59,776 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:35:59,904 P36916 INFO Train loss: 0.282446
2022-01-30 22:35:59,905 P36916 INFO ************ Epoch=2 end ************
2022-01-30 22:36:16,740 P36916 INFO [Metrics] AUC: 0.955593 - logloss: 0.241280
2022-01-30 22:36:16,750 P36916 INFO Save best model: monitor(max): 0.955593
2022-01-30 22:36:16,905 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:36:17,071 P36916 INFO Train loss: 0.255018
2022-01-30 22:36:17,072 P36916 INFO ************ Epoch=3 end ************
2022-01-30 22:36:33,288 P36916 INFO [Metrics] AUC: 0.955911 - logloss: 0.245874
2022-01-30 22:36:33,294 P36916 INFO Save best model: monitor(max): 0.955911
2022-01-30 22:36:33,420 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:36:33,557 P36916 INFO Train loss: 0.229967
2022-01-30 22:36:33,557 P36916 INFO ************ Epoch=4 end ************
2022-01-30 22:36:48,893 P36916 INFO [Metrics] AUC: 0.969047 - logloss: 0.206815
2022-01-30 22:36:48,893 P36916 INFO Save best model: monitor(max): 0.969047
2022-01-30 22:36:48,932 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:36:49,011 P36916 INFO Train loss: 0.209896
2022-01-30 22:36:49,011 P36916 INFO ************ Epoch=5 end ************
2022-01-30 22:37:05,483 P36916 INFO [Metrics] AUC: 0.973316 - logloss: 0.193043
2022-01-30 22:37:05,484 P36916 INFO Save best model: monitor(max): 0.973316
2022-01-30 22:37:05,565 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:37:05,714 P36916 INFO Train loss: 0.175924
2022-01-30 22:37:05,714 P36916 INFO ************ Epoch=6 end ************
2022-01-30 22:37:22,315 P36916 INFO [Metrics] AUC: 0.975723 - logloss: 0.187480
2022-01-30 22:37:22,322 P36916 INFO Save best model: monitor(max): 0.975723
2022-01-30 22:37:22,402 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:37:22,480 P36916 INFO Train loss: 0.154888
2022-01-30 22:37:22,480 P36916 INFO ************ Epoch=7 end ************
2022-01-30 22:37:38,636 P36916 INFO [Metrics] AUC: 0.977480 - logloss: 0.186111
2022-01-30 22:37:38,639 P36916 INFO Save best model: monitor(max): 0.977480
2022-01-30 22:37:38,718 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:37:38,819 P36916 INFO Train loss: 0.139350
2022-01-30 22:37:38,819 P36916 INFO ************ Epoch=8 end ************
2022-01-30 22:37:54,061 P36916 INFO [Metrics] AUC: 0.978519 - logloss: 0.183879
2022-01-30 22:37:54,062 P36916 INFO Save best model: monitor(max): 0.978519
2022-01-30 22:37:54,100 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:37:54,212 P36916 INFO Train loss: 0.128097
2022-01-30 22:37:54,212 P36916 INFO ************ Epoch=9 end ************
2022-01-30 22:38:10,125 P36916 INFO [Metrics] AUC: 0.979079 - logloss: 0.186738
2022-01-30 22:38:10,125 P36916 INFO Save best model: monitor(max): 0.979079
2022-01-30 22:38:10,184 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:38:10,280 P36916 INFO Train loss: 0.118285
2022-01-30 22:38:10,280 P36916 INFO ************ Epoch=10 end ************
2022-01-30 22:38:26,281 P36916 INFO [Metrics] AUC: 0.979643 - logloss: 0.189886
2022-01-30 22:38:26,292 P36916 INFO Save best model: monitor(max): 0.979643
2022-01-30 22:38:26,399 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:38:26,495 P36916 INFO Train loss: 0.110552
2022-01-30 22:38:26,495 P36916 INFO ************ Epoch=11 end ************
2022-01-30 22:38:42,280 P36916 INFO [Metrics] AUC: 0.980122 - logloss: 0.188031
2022-01-30 22:38:42,281 P36916 INFO Save best model: monitor(max): 0.980122
2022-01-30 22:38:42,356 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:38:42,482 P36916 INFO Train loss: 0.103269
2022-01-30 22:38:42,482 P36916 INFO ************ Epoch=12 end ************
2022-01-30 22:38:59,631 P36916 INFO [Metrics] AUC: 0.980180 - logloss: 0.192972
2022-01-30 22:38:59,643 P36916 INFO Save best model: monitor(max): 0.980180
2022-01-30 22:38:59,759 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:38:59,848 P36916 INFO Train loss: 0.096586
2022-01-30 22:38:59,849 P36916 INFO ************ Epoch=13 end ************
2022-01-30 22:39:16,882 P36916 INFO [Metrics] AUC: 0.980547 - logloss: 0.194809
2022-01-30 22:39:16,883 P36916 INFO Save best model: monitor(max): 0.980547
2022-01-30 22:39:16,971 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:39:17,136 P36916 INFO Train loss: 0.091373
2022-01-30 22:39:17,137 P36916 INFO ************ Epoch=14 end ************
2022-01-30 22:39:33,762 P36916 INFO [Metrics] AUC: 0.980804 - logloss: 0.195714
2022-01-30 22:39:33,762 P36916 INFO Save best model: monitor(max): 0.980804
2022-01-30 22:39:33,799 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:39:33,886 P36916 INFO Train loss: 0.087670
2022-01-30 22:39:33,886 P36916 INFO ************ Epoch=15 end ************
2022-01-30 22:39:49,666 P36916 INFO [Metrics] AUC: 0.980595 - logloss: 0.204210
2022-01-30 22:39:49,667 P36916 INFO Monitor(max) STOP: 0.980595 !
2022-01-30 22:39:49,667 P36916 INFO Reduce learning rate on plateau: 0.000100
2022-01-30 22:39:49,667 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:39:49,744 P36916 INFO Train loss: 0.086086
2022-01-30 22:39:49,744 P36916 INFO ************ Epoch=16 end ************
2022-01-30 22:40:05,073 P36916 INFO [Metrics] AUC: 0.981614 - logloss: 0.203619
2022-01-30 22:40:05,074 P36916 INFO Save best model: monitor(max): 0.981614
2022-01-30 22:40:05,163 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:40:05,261 P36916 INFO Train loss: 0.068903
2022-01-30 22:40:05,261 P36916 INFO ************ Epoch=17 end ************
2022-01-30 22:40:22,102 P36916 INFO [Metrics] AUC: 0.981898 - logloss: 0.208315
2022-01-30 22:40:22,103 P36916 INFO Save best model: monitor(max): 0.981898
2022-01-30 22:40:22,222 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:40:22,301 P36916 INFO Train loss: 0.060410
2022-01-30 22:40:22,302 P36916 INFO ************ Epoch=18 end ************
2022-01-30 22:40:38,103 P36916 INFO [Metrics] AUC: 0.981946 - logloss: 0.210969
2022-01-30 22:40:38,103 P36916 INFO Save best model: monitor(max): 0.981946
2022-01-30 22:40:38,196 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:40:38,280 P36916 INFO Train loss: 0.056530
2022-01-30 22:40:38,281 P36916 INFO ************ Epoch=19 end ************
2022-01-30 22:40:54,741 P36916 INFO [Metrics] AUC: 0.982073 - logloss: 0.214431
2022-01-30 22:40:54,746 P36916 INFO Save best model: monitor(max): 0.982073
2022-01-30 22:40:54,837 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:40:54,933 P36916 INFO Train loss: 0.053697
2022-01-30 22:40:54,933 P36916 INFO ************ Epoch=20 end ************
2022-01-30 22:41:09,131 P36916 INFO [Metrics] AUC: 0.982030 - logloss: 0.217100
2022-01-30 22:41:09,132 P36916 INFO Monitor(max) STOP: 0.982030 !
2022-01-30 22:41:09,132 P36916 INFO Reduce learning rate on plateau: 0.000010
2022-01-30 22:41:09,132 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:41:09,220 P36916 INFO Train loss: 0.051779
2022-01-30 22:41:09,220 P36916 INFO ************ Epoch=21 end ************
2022-01-30 22:41:24,776 P36916 INFO [Metrics] AUC: 0.982035 - logloss: 0.218456
2022-01-30 22:41:24,777 P36916 INFO Monitor(max) STOP: 0.982035 !
2022-01-30 22:41:24,777 P36916 INFO Reduce learning rate on plateau: 0.000001
2022-01-30 22:41:24,777 P36916 INFO Early stopping at epoch=22
2022-01-30 22:41:24,777 P36916 INFO --- 50/50 batches finished ---
2022-01-30 22:41:24,863 P36916 INFO Train loss: 0.049437
2022-01-30 22:41:24,863 P36916 INFO Training finished.
2022-01-30 22:41:24,863 P36916 INFO Load best model: /home/XXX/benchmarks/Frappe/AFN_frappe_x1/frappe_x1_04e961e9/AFN_frappe_x1_004_d2ea60c3.model
2022-01-30 22:41:24,981 P36916 INFO ****** Validation evaluation ******
2022-01-30 22:41:26,542 P36916 INFO [Metrics] AUC: 0.982073 - logloss: 0.214431
2022-01-30 22:41:26,657 P36916 INFO ******** Test evaluation ********
2022-01-30 22:41:26,658 P36916 INFO Loading data...
2022-01-30 22:41:26,659 P36916 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-01-30 22:41:26,665 P36916 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-01-30 22:41:26,666 P36916 INFO Loading test data done.
2022-01-30 22:41:27,588 P36916 INFO [Metrics] AUC: 0.982563 - logloss: 0.213933

```
