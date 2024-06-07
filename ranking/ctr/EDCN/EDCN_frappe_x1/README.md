## EDCN_frappe_x1

A hands-on guide to run the EDCN model on the Frappe_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index

[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs) | [Revision History](#Revision-History)

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
  fuxictr: 1.2.2
  ```

### Dataset

Dataset ID: [Frappe_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Frappe#Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.2](https://github.com/reczoo/FuxiCTR/tree/v1.2.2) for this experiment. See the model code: [EDCN](https://github.com/reczoo/FuxiCTR/blob/v1.2.2/fuxictr/pytorch/models/EDCN.py).

Running steps:

1. Download [FuxiCTR-v1.2.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.2.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [fuxictr_version.py](./fuxictr_version.py#L3) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [EDCN_frappe_x1_tuner_config_02](./EDCN_frappe_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd EDCN_frappe_x1
   nohup python run_expid.py --config ./EDCN_frappe_x1_tuner_config_02 --expid EDCN_frappe_x1_006_5e8b9617 --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

| AUC      | logloss  |
|:--------:|:--------:|
| 0.985012 | 0.154740 |

### Logs

```python
2022-06-17 10:36:58,883 P11115 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "bridge_type": "hadamard_product",
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
    "hidden_activations": "ReLU",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "EDCN",
    "model_id": "EDCN_frappe_x1_006_5e8b9617",
    "model_root": "./Frappe/EDCN_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_cross_layers": "3",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "temperature": "5",
    "test_data": "../data/Frappe/Frappe_x1/test.csv",
    "train_data": "../data/Frappe/Frappe_x1/train.csv",
    "use_hdf5": "True",
    "use_regulation_module": "True",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-06-17 10:36:58,884 P11115 INFO Set up feature encoder...
2022-06-17 10:36:58,884 P11115 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-06-17 10:36:58,885 P11115 INFO Loading data...
2022-06-17 10:36:58,888 P11115 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-06-17 10:36:58,900 P11115 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-06-17 10:36:58,905 P11115 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-06-17 10:36:58,905 P11115 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-06-17 10:36:58,905 P11115 INFO Loading train data done.
2022-06-17 10:37:02,445 P11115 INFO Total number of parameters: 86351.
2022-06-17 10:37:02,445 P11115 INFO Start training: 50 batches/epoch
2022-06-17 10:37:02,446 P11115 INFO ************ Epoch=1 start ************
2022-06-17 10:37:05,733 P11115 INFO [Metrics] AUC: 0.923567 - logloss: 0.634561
2022-06-17 10:37:05,734 P11115 INFO Save best model: monitor(max): 0.923567
2022-06-17 10:37:05,749 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:37:05,781 P11115 INFO Train loss: 0.487524
2022-06-17 10:37:05,781 P11115 INFO ************ Epoch=1 end ************
2022-06-17 10:37:09,024 P11115 INFO [Metrics] AUC: 0.940441 - logloss: 0.385828
2022-06-17 10:37:09,024 P11115 INFO Save best model: monitor(max): 0.940441
2022-06-17 10:37:09,028 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:37:09,066 P11115 INFO Train loss: 0.317608
2022-06-17 10:37:09,066 P11115 INFO ************ Epoch=2 end ************
2022-06-17 10:37:12,337 P11115 INFO [Metrics] AUC: 0.957764 - logloss: 0.232331
2022-06-17 10:37:12,337 P11115 INFO Save best model: monitor(max): 0.957764
2022-06-17 10:37:12,343 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:37:12,380 P11115 INFO Train loss: 0.287962
2022-06-17 10:37:12,380 P11115 INFO ************ Epoch=3 end ************
2022-06-17 10:37:15,698 P11115 INFO [Metrics] AUC: 0.966722 - logloss: 0.208343
2022-06-17 10:37:15,698 P11115 INFO Save best model: monitor(max): 0.966722
2022-06-17 10:37:15,704 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:37:15,737 P11115 INFO Train loss: 0.266455
2022-06-17 10:37:15,737 P11115 INFO ************ Epoch=4 end ************
2022-06-17 10:37:19,207 P11115 INFO [Metrics] AUC: 0.970991 - logloss: 0.194302
2022-06-17 10:37:19,208 P11115 INFO Save best model: monitor(max): 0.970991
2022-06-17 10:37:19,212 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:37:19,252 P11115 INFO Train loss: 0.250951
2022-06-17 10:37:19,253 P11115 INFO ************ Epoch=5 end ************
2022-06-17 10:37:22,592 P11115 INFO [Metrics] AUC: 0.973928 - logloss: 0.184300
2022-06-17 10:37:22,592 P11115 INFO Save best model: monitor(max): 0.973928
2022-06-17 10:37:22,596 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:37:22,639 P11115 INFO Train loss: 0.239324
2022-06-17 10:37:22,639 P11115 INFO ************ Epoch=6 end ************
2022-06-17 10:37:26,118 P11115 INFO [Metrics] AUC: 0.975117 - logloss: 0.179804
2022-06-17 10:37:26,119 P11115 INFO Save best model: monitor(max): 0.975117
2022-06-17 10:37:26,125 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:37:26,190 P11115 INFO Train loss: 0.231904
2022-06-17 10:37:26,190 P11115 INFO ************ Epoch=7 end ************
2022-06-17 10:37:29,567 P11115 INFO [Metrics] AUC: 0.976421 - logloss: 0.174202
2022-06-17 10:37:29,567 P11115 INFO Save best model: monitor(max): 0.976421
2022-06-17 10:37:29,571 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:37:29,605 P11115 INFO Train loss: 0.224623
2022-06-17 10:37:29,605 P11115 INFO ************ Epoch=8 end ************
2022-06-17 10:37:33,023 P11115 INFO [Metrics] AUC: 0.978021 - logloss: 0.167625
2022-06-17 10:37:33,023 P11115 INFO Save best model: monitor(max): 0.978021
2022-06-17 10:37:33,030 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:37:33,087 P11115 INFO Train loss: 0.219969
2022-06-17 10:37:33,088 P11115 INFO ************ Epoch=9 end ************
2022-06-17 10:37:36,509 P11115 INFO [Metrics] AUC: 0.978175 - logloss: 0.166840
2022-06-17 10:37:36,510 P11115 INFO Save best model: monitor(max): 0.978175
2022-06-17 10:37:36,515 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:37:36,550 P11115 INFO Train loss: 0.215638
2022-06-17 10:37:36,551 P11115 INFO ************ Epoch=10 end ************
2022-06-17 10:37:39,867 P11115 INFO [Metrics] AUC: 0.978748 - logloss: 0.163673
2022-06-17 10:37:39,868 P11115 INFO Save best model: monitor(max): 0.978748
2022-06-17 10:37:39,872 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:37:39,915 P11115 INFO Train loss: 0.209935
2022-06-17 10:37:39,915 P11115 INFO ************ Epoch=11 end ************
2022-06-17 10:37:43,356 P11115 INFO [Metrics] AUC: 0.979133 - logloss: 0.162595
2022-06-17 10:37:43,357 P11115 INFO Save best model: monitor(max): 0.979133
2022-06-17 10:37:43,361 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:37:43,404 P11115 INFO Train loss: 0.207029
2022-06-17 10:37:43,404 P11115 INFO ************ Epoch=12 end ************
2022-06-17 10:37:46,803 P11115 INFO [Metrics] AUC: 0.979149 - logloss: 0.162339
2022-06-17 10:37:46,804 P11115 INFO Save best model: monitor(max): 0.979149
2022-06-17 10:37:46,808 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:37:46,847 P11115 INFO Train loss: 0.206296
2022-06-17 10:37:46,848 P11115 INFO ************ Epoch=13 end ************
2022-06-17 10:37:50,236 P11115 INFO [Metrics] AUC: 0.979671 - logloss: 0.160037
2022-06-17 10:37:50,237 P11115 INFO Save best model: monitor(max): 0.979671
2022-06-17 10:37:50,243 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:37:50,274 P11115 INFO Train loss: 0.202765
2022-06-17 10:37:50,274 P11115 INFO ************ Epoch=14 end ************
2022-06-17 10:37:53,711 P11115 INFO [Metrics] AUC: 0.979980 - logloss: 0.158859
2022-06-17 10:37:53,712 P11115 INFO Save best model: monitor(max): 0.979980
2022-06-17 10:37:53,717 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:37:53,770 P11115 INFO Train loss: 0.199952
2022-06-17 10:37:53,771 P11115 INFO ************ Epoch=15 end ************
2022-06-17 10:37:57,142 P11115 INFO [Metrics] AUC: 0.980411 - logloss: 0.157384
2022-06-17 10:37:57,142 P11115 INFO Save best model: monitor(max): 0.980411
2022-06-17 10:37:57,146 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:37:57,209 P11115 INFO Train loss: 0.200116
2022-06-17 10:37:57,209 P11115 INFO ************ Epoch=16 end ************
2022-06-17 10:38:00,609 P11115 INFO [Metrics] AUC: 0.980524 - logloss: 0.156414
2022-06-17 10:38:00,610 P11115 INFO Save best model: monitor(max): 0.980524
2022-06-17 10:38:00,614 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:38:00,672 P11115 INFO Train loss: 0.196681
2022-06-17 10:38:00,672 P11115 INFO ************ Epoch=17 end ************
2022-06-17 10:38:04,106 P11115 INFO [Metrics] AUC: 0.980684 - logloss: 0.155319
2022-06-17 10:38:04,107 P11115 INFO Save best model: monitor(max): 0.980684
2022-06-17 10:38:04,112 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:38:04,152 P11115 INFO Train loss: 0.194683
2022-06-17 10:38:04,152 P11115 INFO ************ Epoch=18 end ************
2022-06-17 10:38:07,547 P11115 INFO [Metrics] AUC: 0.981030 - logloss: 0.153903
2022-06-17 10:38:07,548 P11115 INFO Save best model: monitor(max): 0.981030
2022-06-17 10:38:07,552 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:38:07,586 P11115 INFO Train loss: 0.194086
2022-06-17 10:38:07,586 P11115 INFO ************ Epoch=19 end ************
2022-06-17 10:38:10,907 P11115 INFO [Metrics] AUC: 0.980959 - logloss: 0.154746
2022-06-17 10:38:10,908 P11115 INFO Monitor(max) STOP: 0.980959 !
2022-06-17 10:38:10,908 P11115 INFO Reduce learning rate on plateau: 0.000100
2022-06-17 10:38:10,908 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:38:10,939 P11115 INFO Train loss: 0.191342
2022-06-17 10:38:10,939 P11115 INFO ************ Epoch=20 end ************
2022-06-17 10:38:14,213 P11115 INFO [Metrics] AUC: 0.982819 - logloss: 0.149417
2022-06-17 10:38:14,213 P11115 INFO Save best model: monitor(max): 0.982819
2022-06-17 10:38:14,220 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:38:14,262 P11115 INFO Train loss: 0.164232
2022-06-17 10:38:14,262 P11115 INFO ************ Epoch=21 end ************
2022-06-17 10:38:17,564 P11115 INFO [Metrics] AUC: 0.983918 - logloss: 0.148284
2022-06-17 10:38:17,565 P11115 INFO Save best model: monitor(max): 0.983918
2022-06-17 10:38:17,571 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:38:17,622 P11115 INFO Train loss: 0.141489
2022-06-17 10:38:17,622 P11115 INFO ************ Epoch=22 end ************
2022-06-17 10:38:20,914 P11115 INFO [Metrics] AUC: 0.984512 - logloss: 0.147517
2022-06-17 10:38:20,914 P11115 INFO Save best model: monitor(max): 0.984512
2022-06-17 10:38:20,918 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:38:20,948 P11115 INFO Train loss: 0.127155
2022-06-17 10:38:20,948 P11115 INFO ************ Epoch=23 end ************
2022-06-17 10:38:24,140 P11115 INFO [Metrics] AUC: 0.984821 - logloss: 0.147594
2022-06-17 10:38:24,140 P11115 INFO Save best model: monitor(max): 0.984821
2022-06-17 10:38:24,144 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:38:24,176 P11115 INFO Train loss: 0.117613
2022-06-17 10:38:24,176 P11115 INFO ************ Epoch=24 end ************
2022-06-17 10:38:27,344 P11115 INFO [Metrics] AUC: 0.985099 - logloss: 0.147408
2022-06-17 10:38:27,345 P11115 INFO Save best model: monitor(max): 0.985099
2022-06-17 10:38:27,350 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:38:27,399 P11115 INFO Train loss: 0.109705
2022-06-17 10:38:27,399 P11115 INFO ************ Epoch=25 end ************
2022-06-17 10:38:30,528 P11115 INFO [Metrics] AUC: 0.985310 - logloss: 0.148142
2022-06-17 10:38:30,529 P11115 INFO Save best model: monitor(max): 0.985310
2022-06-17 10:38:30,535 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:38:30,566 P11115 INFO Train loss: 0.102259
2022-06-17 10:38:30,566 P11115 INFO ************ Epoch=26 end ************
2022-06-17 10:38:33,654 P11115 INFO [Metrics] AUC: 0.985344 - logloss: 0.149757
2022-06-17 10:38:33,655 P11115 INFO Save best model: monitor(max): 0.985344
2022-06-17 10:38:33,659 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:38:33,690 P11115 INFO Train loss: 0.097674
2022-06-17 10:38:33,690 P11115 INFO ************ Epoch=27 end ************
2022-06-17 10:38:36,724 P11115 INFO [Metrics] AUC: 0.985469 - logloss: 0.150015
2022-06-17 10:38:36,724 P11115 INFO Save best model: monitor(max): 0.985469
2022-06-17 10:38:36,728 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:38:36,758 P11115 INFO Train loss: 0.093279
2022-06-17 10:38:36,758 P11115 INFO ************ Epoch=28 end ************
2022-06-17 10:38:38,737 P11115 INFO [Metrics] AUC: 0.985596 - logloss: 0.151137
2022-06-17 10:38:38,738 P11115 INFO Save best model: monitor(max): 0.985596
2022-06-17 10:38:38,742 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:38:38,772 P11115 INFO Train loss: 0.089323
2022-06-17 10:38:38,772 P11115 INFO ************ Epoch=29 end ************
2022-06-17 10:38:40,640 P11115 INFO [Metrics] AUC: 0.985475 - logloss: 0.152399
2022-06-17 10:38:40,640 P11115 INFO Monitor(max) STOP: 0.985475 !
2022-06-17 10:38:40,640 P11115 INFO Reduce learning rate on plateau: 0.000010
2022-06-17 10:38:40,641 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:38:40,671 P11115 INFO Train loss: 0.086093
2022-06-17 10:38:40,671 P11115 INFO ************ Epoch=30 end ************
2022-06-17 10:38:42,730 P11115 INFO [Metrics] AUC: 0.985553 - logloss: 0.152531
2022-06-17 10:38:42,731 P11115 INFO Monitor(max) STOP: 0.985553 !
2022-06-17 10:38:42,731 P11115 INFO Reduce learning rate on plateau: 0.000001
2022-06-17 10:38:42,731 P11115 INFO Early stopping at epoch=31
2022-06-17 10:38:42,731 P11115 INFO --- 50/50 batches finished ---
2022-06-17 10:38:42,762 P11115 INFO Train loss: 0.081686
2022-06-17 10:38:42,762 P11115 INFO Training finished.
2022-06-17 10:38:42,762 P11115 INFO Load best model: /home/FuxiCTR/benchmarks_local/Frappe/EDCN_frappe_x1/frappe_x1_04e961e9/EDCN_frappe_x1_006_5e8b9617.model
2022-06-17 10:38:46,579 P11115 INFO ****** Validation evaluation ******
2022-06-17 10:38:46,909 P11115 INFO [Metrics] AUC: 0.985596 - logloss: 0.151137
2022-06-17 10:38:46,946 P11115 INFO ******** Test evaluation ********
2022-06-17 10:38:46,946 P11115 INFO Loading data...
2022-06-17 10:38:46,946 P11115 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-06-17 10:38:46,949 P11115 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-06-17 10:38:46,949 P11115 INFO Loading test data done.
2022-06-17 10:38:47,199 P11115 INFO [Metrics] AUC: 0.985012 - logloss: 0.154740
```

### Revision History

- [Version 1](https://github.com/openbenchmark/BARS/tree/88d3a0faa4565e975141ae89a52d35d3a8b56eda/ctr_prediction/benchmarks/EDCN/EDCN_frappe_x1): deprecated due to bug fix [#29](https://github.com/reczoo/FuxiCTR/issues/29) of FuxiCTR.
