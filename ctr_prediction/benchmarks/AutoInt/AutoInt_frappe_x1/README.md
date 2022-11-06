## AutoInt_frappe_x1

A hands-on guide to run the AutoInt model on the Frappe_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

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

We use [FuxiCTR-v1.2.2](https://github.com/xue-pai/FuxiCTR/tree/v1.2.2) for this experiment. See the model code: [AutoInt](https://github.com/xue-pai/FuxiCTR/blob/v1.2.2/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.2.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.2.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [fuxictr_version.py](./fuxictr_version.py#L3) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt_frappe_x1_tuner_config_05](./AutoInt_frappe_x1_tuner_config_05). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd AutoInt_frappe_x1
   nohup python run_expid.py --config ./AutoInt_frappe_x1_tuner_config_05 --expid AutoInt_frappe_x1_005_2b296630 --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

| AUC      | logloss  |
|:--------:|:--------:|
| 0.983085 | 0.163684 |

### Logs

```python
2022-06-27 08:11:52,662 P26323 INFO {
    "attention_dim": "128",
    "attention_layers": "6",
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AutoInt",
    "model_id": "AutoInt_frappe_x1_005_2b296630",
    "model_root": "./Frappe/AutoInt_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "4",
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
    "use_residual": "True",
    "use_scale": "False",
    "use_wide": "False",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-06-27 08:11:52,662 P26323 INFO Set up feature encoder...
2022-06-27 08:11:52,662 P26323 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-06-27 08:11:52,663 P26323 INFO Loading data...
2022-06-27 08:11:52,664 P26323 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-06-27 08:11:52,675 P26323 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-06-27 08:11:52,680 P26323 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-06-27 08:11:52,680 P26323 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-06-27 08:11:52,680 P26323 INFO Loading train data done.
2022-06-27 08:11:55,744 P26323 INFO Total number of parameters: 306051.
2022-06-27 08:11:55,745 P26323 INFO Start training: 50 batches/epoch
2022-06-27 08:11:55,745 P26323 INFO ************ Epoch=1 start ************
2022-06-27 08:12:09,903 P26323 INFO [Metrics] AUC: 0.929346 - logloss: 0.389975
2022-06-27 08:12:09,903 P26323 INFO Save best model: monitor(max): 0.929346
2022-06-27 08:12:09,907 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:12:09,940 P26323 INFO Train loss: 0.567680
2022-06-27 08:12:09,940 P26323 INFO ************ Epoch=1 end ************
2022-06-27 08:12:24,045 P26323 INFO [Metrics] AUC: 0.932565 - logloss: 0.291329
2022-06-27 08:12:24,046 P26323 INFO Save best model: monitor(max): 0.932565
2022-06-27 08:12:24,052 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:12:24,083 P26323 INFO Train loss: 0.341109
2022-06-27 08:12:24,083 P26323 INFO ************ Epoch=2 end ************
2022-06-27 08:12:37,996 P26323 INFO [Metrics] AUC: 0.936331 - logloss: 0.288659
2022-06-27 08:12:37,997 P26323 INFO Save best model: monitor(max): 0.936331
2022-06-27 08:12:38,003 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:12:38,035 P26323 INFO Train loss: 0.311964
2022-06-27 08:12:38,035 P26323 INFO ************ Epoch=3 end ************
2022-06-27 08:12:52,084 P26323 INFO [Metrics] AUC: 0.938286 - logloss: 0.285408
2022-06-27 08:12:52,085 P26323 INFO Save best model: monitor(max): 0.938286
2022-06-27 08:12:52,088 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:12:52,123 P26323 INFO Train loss: 0.306016
2022-06-27 08:12:52,123 P26323 INFO ************ Epoch=4 end ************
2022-06-27 08:13:06,183 P26323 INFO [Metrics] AUC: 0.942865 - logloss: 0.273632
2022-06-27 08:13:06,183 P26323 INFO Save best model: monitor(max): 0.942865
2022-06-27 08:13:06,187 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:13:06,221 P26323 INFO Train loss: 0.300813
2022-06-27 08:13:06,221 P26323 INFO ************ Epoch=5 end ************
2022-06-27 08:13:20,242 P26323 INFO [Metrics] AUC: 0.949183 - logloss: 0.258002
2022-06-27 08:13:20,243 P26323 INFO Save best model: monitor(max): 0.949183
2022-06-27 08:13:20,247 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:13:20,278 P26323 INFO Train loss: 0.294925
2022-06-27 08:13:20,278 P26323 INFO ************ Epoch=6 end ************
2022-06-27 08:13:34,183 P26323 INFO [Metrics] AUC: 0.955690 - logloss: 0.239714
2022-06-27 08:13:34,184 P26323 INFO Save best model: monitor(max): 0.955690
2022-06-27 08:13:34,189 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:13:34,222 P26323 INFO Train loss: 0.282001
2022-06-27 08:13:34,222 P26323 INFO ************ Epoch=7 end ************
2022-06-27 08:13:48,138 P26323 INFO [Metrics] AUC: 0.959647 - logloss: 0.228313
2022-06-27 08:13:48,138 P26323 INFO Save best model: monitor(max): 0.959647
2022-06-27 08:13:48,142 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:13:48,173 P26323 INFO Train loss: 0.270141
2022-06-27 08:13:48,173 P26323 INFO ************ Epoch=8 end ************
2022-06-27 08:14:02,017 P26323 INFO [Metrics] AUC: 0.961786 - logloss: 0.225262
2022-06-27 08:14:02,017 P26323 INFO Save best model: monitor(max): 0.961786
2022-06-27 08:14:02,021 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:14:02,063 P26323 INFO Train loss: 0.261778
2022-06-27 08:14:02,063 P26323 INFO ************ Epoch=9 end ************
2022-06-27 08:14:15,960 P26323 INFO [Metrics] AUC: 0.965417 - logloss: 0.212291
2022-06-27 08:14:15,961 P26323 INFO Save best model: monitor(max): 0.965417
2022-06-27 08:14:15,966 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:14:16,012 P26323 INFO Train loss: 0.254553
2022-06-27 08:14:16,012 P26323 INFO ************ Epoch=10 end ************
2022-06-27 08:14:29,995 P26323 INFO [Metrics] AUC: 0.966631 - logloss: 0.208825
2022-06-27 08:14:29,996 P26323 INFO Save best model: monitor(max): 0.966631
2022-06-27 08:14:30,001 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:14:30,052 P26323 INFO Train loss: 0.248408
2022-06-27 08:14:30,053 P26323 INFO ************ Epoch=11 end ************
2022-06-27 08:14:43,887 P26323 INFO [Metrics] AUC: 0.969565 - logloss: 0.198196
2022-06-27 08:14:43,888 P26323 INFO Save best model: monitor(max): 0.969565
2022-06-27 08:14:43,893 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:14:43,927 P26323 INFO Train loss: 0.244414
2022-06-27 08:14:43,927 P26323 INFO ************ Epoch=12 end ************
2022-06-27 08:14:57,876 P26323 INFO [Metrics] AUC: 0.970447 - logloss: 0.195624
2022-06-27 08:14:57,877 P26323 INFO Save best model: monitor(max): 0.970447
2022-06-27 08:14:57,880 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:14:57,913 P26323 INFO Train loss: 0.237312
2022-06-27 08:14:57,913 P26323 INFO ************ Epoch=13 end ************
2022-06-27 08:15:11,723 P26323 INFO [Metrics] AUC: 0.971478 - logloss: 0.191725
2022-06-27 08:15:11,724 P26323 INFO Save best model: monitor(max): 0.971478
2022-06-27 08:15:11,729 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:15:11,763 P26323 INFO Train loss: 0.232111
2022-06-27 08:15:11,763 P26323 INFO ************ Epoch=14 end ************
2022-06-27 08:15:25,700 P26323 INFO [Metrics] AUC: 0.972812 - logloss: 0.187026
2022-06-27 08:15:25,700 P26323 INFO Save best model: monitor(max): 0.972812
2022-06-27 08:15:25,704 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:15:25,749 P26323 INFO Train loss: 0.228942
2022-06-27 08:15:25,749 P26323 INFO ************ Epoch=15 end ************
2022-06-27 08:15:39,714 P26323 INFO [Metrics] AUC: 0.974048 - logloss: 0.183179
2022-06-27 08:15:39,714 P26323 INFO Save best model: monitor(max): 0.974048
2022-06-27 08:15:39,720 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:15:39,752 P26323 INFO Train loss: 0.223017
2022-06-27 08:15:39,752 P26323 INFO ************ Epoch=16 end ************
2022-06-27 08:15:53,689 P26323 INFO [Metrics] AUC: 0.974515 - logloss: 0.183152
2022-06-27 08:15:53,689 P26323 INFO Save best model: monitor(max): 0.974515
2022-06-27 08:15:53,693 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:15:53,738 P26323 INFO Train loss: 0.219969
2022-06-27 08:15:53,739 P26323 INFO ************ Epoch=17 end ************
2022-06-27 08:16:07,644 P26323 INFO [Metrics] AUC: 0.975063 - logloss: 0.179174
2022-06-27 08:16:07,644 P26323 INFO Save best model: monitor(max): 0.975063
2022-06-27 08:16:07,650 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:16:07,682 P26323 INFO Train loss: 0.217608
2022-06-27 08:16:07,682 P26323 INFO ************ Epoch=18 end ************
2022-06-27 08:16:21,476 P26323 INFO [Metrics] AUC: 0.975183 - logloss: 0.178508
2022-06-27 08:16:21,477 P26323 INFO Save best model: monitor(max): 0.975183
2022-06-27 08:16:21,482 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:16:21,515 P26323 INFO Train loss: 0.214375
2022-06-27 08:16:21,515 P26323 INFO ************ Epoch=19 end ************
2022-06-27 08:16:35,466 P26323 INFO [Metrics] AUC: 0.975895 - logloss: 0.177553
2022-06-27 08:16:35,466 P26323 INFO Save best model: monitor(max): 0.975895
2022-06-27 08:16:35,470 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:16:35,502 P26323 INFO Train loss: 0.210745
2022-06-27 08:16:35,502 P26323 INFO ************ Epoch=20 end ************
2022-06-27 08:16:49,276 P26323 INFO [Metrics] AUC: 0.975978 - logloss: 0.176675
2022-06-27 08:16:49,276 P26323 INFO Save best model: monitor(max): 0.975978
2022-06-27 08:16:49,282 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:16:49,329 P26323 INFO Train loss: 0.207246
2022-06-27 08:16:49,329 P26323 INFO ************ Epoch=21 end ************
2022-06-27 08:17:03,245 P26323 INFO [Metrics] AUC: 0.977693 - logloss: 0.168597
2022-06-27 08:17:03,246 P26323 INFO Save best model: monitor(max): 0.977693
2022-06-27 08:17:03,250 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:17:03,283 P26323 INFO Train loss: 0.206541
2022-06-27 08:17:03,283 P26323 INFO ************ Epoch=22 end ************
2022-06-27 08:17:17,102 P26323 INFO [Metrics] AUC: 0.977174 - logloss: 0.168754
2022-06-27 08:17:17,102 P26323 INFO Monitor(max) STOP: 0.977174 !
2022-06-27 08:17:17,103 P26323 INFO Reduce learning rate on plateau: 0.000100
2022-06-27 08:17:17,103 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:17:17,149 P26323 INFO Train loss: 0.202790
2022-06-27 08:17:17,149 P26323 INFO ************ Epoch=23 end ************
2022-06-27 08:17:31,132 P26323 INFO [Metrics] AUC: 0.980868 - logloss: 0.158463
2022-06-27 08:17:31,133 P26323 INFO Save best model: monitor(max): 0.980868
2022-06-27 08:17:31,136 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:17:31,168 P26323 INFO Train loss: 0.169093
2022-06-27 08:17:31,168 P26323 INFO ************ Epoch=24 end ************
2022-06-27 08:17:44,429 P26323 INFO [Metrics] AUC: 0.982040 - logloss: 0.156053
2022-06-27 08:17:44,429 P26323 INFO Save best model: monitor(max): 0.982040
2022-06-27 08:17:44,433 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:17:44,481 P26323 INFO Train loss: 0.140114
2022-06-27 08:17:44,482 P26323 INFO ************ Epoch=25 end ************
2022-06-27 08:17:54,859 P26323 INFO [Metrics] AUC: 0.982484 - logloss: 0.158594
2022-06-27 08:17:54,860 P26323 INFO Save best model: monitor(max): 0.982484
2022-06-27 08:17:54,864 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:17:54,905 P26323 INFO Train loss: 0.123262
2022-06-27 08:17:54,905 P26323 INFO ************ Epoch=26 end ************
2022-06-27 08:18:11,418 P26323 INFO [Metrics] AUC: 0.982674 - logloss: 0.159134
2022-06-27 08:18:11,419 P26323 INFO Save best model: monitor(max): 0.982674
2022-06-27 08:18:11,424 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:18:11,457 P26323 INFO Train loss: 0.111381
2022-06-27 08:18:11,458 P26323 INFO ************ Epoch=27 end ************
2022-06-27 08:18:28,343 P26323 INFO [Metrics] AUC: 0.982680 - logloss: 0.163802
2022-06-27 08:18:28,344 P26323 INFO Save best model: monitor(max): 0.982680
2022-06-27 08:18:28,348 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:18:28,394 P26323 INFO Train loss: 0.101737
2022-06-27 08:18:28,394 P26323 INFO ************ Epoch=28 end ************
2022-06-27 08:18:43,831 P26323 INFO [Metrics] AUC: 0.982810 - logloss: 0.164729
2022-06-27 08:18:43,832 P26323 INFO Save best model: monitor(max): 0.982810
2022-06-27 08:18:43,835 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:18:43,868 P26323 INFO Train loss: 0.094030
2022-06-27 08:18:43,868 P26323 INFO ************ Epoch=29 end ************
2022-06-27 08:19:05,576 P26323 INFO [Metrics] AUC: 0.982492 - logloss: 0.170600
2022-06-27 08:19:05,577 P26323 INFO Monitor(max) STOP: 0.982492 !
2022-06-27 08:19:05,577 P26323 INFO Reduce learning rate on plateau: 0.000010
2022-06-27 08:19:05,577 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:19:05,631 P26323 INFO Train loss: 0.087360
2022-06-27 08:19:05,632 P26323 INFO ************ Epoch=30 end ************
2022-06-27 08:19:27,414 P26323 INFO [Metrics] AUC: 0.982676 - logloss: 0.170811
2022-06-27 08:19:27,414 P26323 INFO Monitor(max) STOP: 0.982676 !
2022-06-27 08:19:27,414 P26323 INFO Reduce learning rate on plateau: 0.000001
2022-06-27 08:19:27,414 P26323 INFO Early stopping at epoch=31
2022-06-27 08:19:27,415 P26323 INFO --- 50/50 batches finished ---
2022-06-27 08:19:27,446 P26323 INFO Train loss: 0.078911
2022-06-27 08:19:27,446 P26323 INFO Training finished.
2022-06-27 08:19:27,446 P26323 INFO Load best model: /home/benchmarks/Frappe/AutoInt_frappe_x1/frappe_x1_04e961e9/AutoInt_frappe_x1_005_2b296630.model
2022-06-27 08:19:27,575 P26323 INFO ****** Validation evaluation ******
2022-06-27 08:19:28,818 P26323 INFO [Metrics] AUC: 0.982810 - logloss: 0.164729
2022-06-27 08:19:28,851 P26323 INFO ******** Test evaluation ********
2022-06-27 08:19:28,851 P26323 INFO Loading data...
2022-06-27 08:19:28,851 P26323 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-06-27 08:19:28,853 P26323 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-06-27 08:19:28,854 P26323 INFO Loading test data done.
2022-06-27 08:19:29,523 P26323 INFO [Metrics] AUC: 0.983085 - logloss: 0.163684
```

### Revision History

- [Version 1](https://github.com/openbenchmark/BARS/tree/88d3a0faa4565e975141ae89a52d35d3a8b56eda/ctr_prediction/benchmarks/AutoInt/AutoInt_frappe_x1#autoint_frappe_x1): deprecated due to bug fix [#30](https://github.com/xue-pai/FuxiCTR/issues/30) of FuxiCTR.
