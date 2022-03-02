## NFM_frappe_x1

A hands-on guide to run the NFM model on the Frappe_x1 dataset.

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

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [NFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/NFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [NFM_frappe_x1_tuner_config_01](./NFM_frappe_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd NFM_frappe_x1
    nohup python run_expid.py --config ./NFM_frappe_x1_tuner_config_01 --expid NFM_frappe_x1_012_3ebff53c --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.980409 | 0.205795  |


### Logs
```python
2022-01-28 16:16:32,867 P57305 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.0001",
    "epochs": "200",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "NFM",
    "model_id": "NFM_frappe_x1_012_3ebff53c",
    "model_root": "./Frappe/NFM_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.4",
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
2022-01-28 16:16:32,867 P57305 INFO Set up feature encoder...
2022-01-28 16:16:32,867 P57305 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-01-28 16:16:32,868 P57305 INFO Loading data...
2022-01-28 16:16:32,870 P57305 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-01-28 16:16:32,882 P57305 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-01-28 16:16:32,887 P57305 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-01-28 16:16:32,887 P57305 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-01-28 16:16:32,887 P57305 INFO Loading train data done.
2022-01-28 16:16:35,993 P57305 INFO Total number of parameters: 384880.
2022-01-28 16:16:35,994 P57305 INFO Start training: 50 batches/epoch
2022-01-28 16:16:35,994 P57305 INFO ************ Epoch=1 start ************
2022-01-28 16:16:39,253 P57305 INFO [Metrics] AUC: 0.927581 - logloss: 0.459683
2022-01-28 16:16:39,253 P57305 INFO Save best model: monitor(max): 0.927581
2022-01-28 16:16:39,257 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:16:39,303 P57305 INFO Train loss: 0.620545
2022-01-28 16:16:39,304 P57305 INFO ************ Epoch=1 end ************
2022-01-28 16:16:42,442 P57305 INFO [Metrics] AUC: 0.936050 - logloss: 0.289232
2022-01-28 16:16:42,443 P57305 INFO Save best model: monitor(max): 0.936050
2022-01-28 16:16:42,447 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:16:42,481 P57305 INFO Train loss: 0.344746
2022-01-28 16:16:42,481 P57305 INFO ************ Epoch=2 end ************
2022-01-28 16:16:45,823 P57305 INFO [Metrics] AUC: 0.938274 - logloss: 0.283714
2022-01-28 16:16:45,824 P57305 INFO Save best model: monitor(max): 0.938274
2022-01-28 16:16:45,829 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:16:45,864 P57305 INFO Train loss: 0.279621
2022-01-28 16:16:45,864 P57305 INFO ************ Epoch=3 end ************
2022-01-28 16:16:49,191 P57305 INFO [Metrics] AUC: 0.939415 - logloss: 0.282766
2022-01-28 16:16:49,191 P57305 INFO Save best model: monitor(max): 0.939415
2022-01-28 16:16:49,195 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:16:49,228 P57305 INFO Train loss: 0.273792
2022-01-28 16:16:49,228 P57305 INFO ************ Epoch=4 end ************
2022-01-28 16:16:52,506 P57305 INFO [Metrics] AUC: 0.940487 - logloss: 0.281213
2022-01-28 16:16:52,507 P57305 INFO Save best model: monitor(max): 0.940487
2022-01-28 16:16:52,511 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:16:52,545 P57305 INFO Train loss: 0.268960
2022-01-28 16:16:52,545 P57305 INFO ************ Epoch=5 end ************
2022-01-28 16:16:55,807 P57305 INFO [Metrics] AUC: 0.941501 - logloss: 0.279637
2022-01-28 16:16:55,807 P57305 INFO Save best model: monitor(max): 0.941501
2022-01-28 16:16:55,811 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:16:55,846 P57305 INFO Train loss: 0.264629
2022-01-28 16:16:55,846 P57305 INFO ************ Epoch=6 end ************
2022-01-28 16:16:59,109 P57305 INFO [Metrics] AUC: 0.942641 - logloss: 0.276341
2022-01-28 16:16:59,109 P57305 INFO Save best model: monitor(max): 0.942641
2022-01-28 16:16:59,113 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:16:59,150 P57305 INFO Train loss: 0.260787
2022-01-28 16:16:59,150 P57305 INFO ************ Epoch=7 end ************
2022-01-28 16:17:02,485 P57305 INFO [Metrics] AUC: 0.944231 - logloss: 0.274146
2022-01-28 16:17:02,486 P57305 INFO Save best model: monitor(max): 0.944231
2022-01-28 16:17:02,489 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:02,524 P57305 INFO Train loss: 0.257142
2022-01-28 16:17:02,524 P57305 INFO ************ Epoch=8 end ************
2022-01-28 16:17:05,824 P57305 INFO [Metrics] AUC: 0.945694 - logloss: 0.271713
2022-01-28 16:17:05,824 P57305 INFO Save best model: monitor(max): 0.945694
2022-01-28 16:17:05,828 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:05,865 P57305 INFO Train loss: 0.251779
2022-01-28 16:17:05,865 P57305 INFO ************ Epoch=9 end ************
2022-01-28 16:17:09,171 P57305 INFO [Metrics] AUC: 0.948131 - logloss: 0.267002
2022-01-28 16:17:09,171 P57305 INFO Save best model: monitor(max): 0.948131
2022-01-28 16:17:09,175 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:09,213 P57305 INFO Train loss: 0.247630
2022-01-28 16:17:09,213 P57305 INFO ************ Epoch=10 end ************
2022-01-28 16:17:12,423 P57305 INFO [Metrics] AUC: 0.950574 - logloss: 0.261687
2022-01-28 16:17:12,423 P57305 INFO Save best model: monitor(max): 0.950574
2022-01-28 16:17:12,427 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:12,462 P57305 INFO Train loss: 0.240930
2022-01-28 16:17:12,463 P57305 INFO ************ Epoch=11 end ************
2022-01-28 16:17:15,705 P57305 INFO [Metrics] AUC: 0.954182 - logloss: 0.252550
2022-01-28 16:17:15,705 P57305 INFO Save best model: monitor(max): 0.954182
2022-01-28 16:17:15,709 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:15,749 P57305 INFO Train loss: 0.231286
2022-01-28 16:17:15,749 P57305 INFO ************ Epoch=12 end ************
2022-01-28 16:17:19,120 P57305 INFO [Metrics] AUC: 0.959455 - logloss: 0.234125
2022-01-28 16:17:19,121 P57305 INFO Save best model: monitor(max): 0.959455
2022-01-28 16:17:19,125 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:19,162 P57305 INFO Train loss: 0.215835
2022-01-28 16:17:19,163 P57305 INFO ************ Epoch=13 end ************
2022-01-28 16:17:22,405 P57305 INFO [Metrics] AUC: 0.963866 - logloss: 0.225522
2022-01-28 16:17:22,405 P57305 INFO Save best model: monitor(max): 0.963866
2022-01-28 16:17:22,409 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:22,444 P57305 INFO Train loss: 0.193621
2022-01-28 16:17:22,444 P57305 INFO ************ Epoch=14 end ************
2022-01-28 16:17:25,689 P57305 INFO [Metrics] AUC: 0.967534 - logloss: 0.209270
2022-01-28 16:17:25,690 P57305 INFO Save best model: monitor(max): 0.967534
2022-01-28 16:17:25,694 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:25,742 P57305 INFO Train loss: 0.175981
2022-01-28 16:17:25,742 P57305 INFO ************ Epoch=15 end ************
2022-01-28 16:17:29,119 P57305 INFO [Metrics] AUC: 0.970858 - logloss: 0.199758
2022-01-28 16:17:29,119 P57305 INFO Save best model: monitor(max): 0.970858
2022-01-28 16:17:29,124 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:29,161 P57305 INFO Train loss: 0.155852
2022-01-28 16:17:29,161 P57305 INFO ************ Epoch=16 end ************
2022-01-28 16:17:32,523 P57305 INFO [Metrics] AUC: 0.973393 - logloss: 0.195573
2022-01-28 16:17:32,523 P57305 INFO Save best model: monitor(max): 0.973393
2022-01-28 16:17:32,527 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:32,565 P57305 INFO Train loss: 0.141989
2022-01-28 16:17:32,565 P57305 INFO ************ Epoch=17 end ************
2022-01-28 16:17:34,839 P57305 INFO [Metrics] AUC: 0.975310 - logloss: 0.196678
2022-01-28 16:17:34,839 P57305 INFO Save best model: monitor(max): 0.975310
2022-01-28 16:17:34,843 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:34,881 P57305 INFO Train loss: 0.129002
2022-01-28 16:17:34,882 P57305 INFO ************ Epoch=18 end ************
2022-01-28 16:17:37,035 P57305 INFO [Metrics] AUC: 0.976410 - logloss: 0.190073
2022-01-28 16:17:37,035 P57305 INFO Save best model: monitor(max): 0.976410
2022-01-28 16:17:37,039 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:37,076 P57305 INFO Train loss: 0.117451
2022-01-28 16:17:37,076 P57305 INFO ************ Epoch=19 end ************
2022-01-28 16:17:39,207 P57305 INFO [Metrics] AUC: 0.977615 - logloss: 0.186787
2022-01-28 16:17:39,207 P57305 INFO Save best model: monitor(max): 0.977615
2022-01-28 16:17:39,211 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:39,245 P57305 INFO Train loss: 0.108019
2022-01-28 16:17:39,245 P57305 INFO ************ Epoch=20 end ************
2022-01-28 16:17:41,393 P57305 INFO [Metrics] AUC: 0.978468 - logloss: 0.188857
2022-01-28 16:17:41,394 P57305 INFO Save best model: monitor(max): 0.978468
2022-01-28 16:17:41,398 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:41,434 P57305 INFO Train loss: 0.101777
2022-01-28 16:17:41,434 P57305 INFO ************ Epoch=21 end ************
2022-01-28 16:17:43,623 P57305 INFO [Metrics] AUC: 0.978749 - logloss: 0.187101
2022-01-28 16:17:43,624 P57305 INFO Save best model: monitor(max): 0.978749
2022-01-28 16:17:43,628 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:43,679 P57305 INFO Train loss: 0.096939
2022-01-28 16:17:43,679 P57305 INFO ************ Epoch=22 end ************
2022-01-28 16:17:45,916 P57305 INFO [Metrics] AUC: 0.979062 - logloss: 0.192979
2022-01-28 16:17:45,916 P57305 INFO Save best model: monitor(max): 0.979062
2022-01-28 16:17:45,920 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:45,955 P57305 INFO Train loss: 0.090136
2022-01-28 16:17:45,956 P57305 INFO ************ Epoch=23 end ************
2022-01-28 16:17:49,581 P57305 INFO [Metrics] AUC: 0.979390 - logloss: 0.198120
2022-01-28 16:17:49,581 P57305 INFO Save best model: monitor(max): 0.979390
2022-01-28 16:17:49,585 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:49,620 P57305 INFO Train loss: 0.085923
2022-01-28 16:17:49,620 P57305 INFO ************ Epoch=24 end ************
2022-01-28 16:17:53,300 P57305 INFO [Metrics] AUC: 0.979583 - logloss: 0.190171
2022-01-28 16:17:53,301 P57305 INFO Save best model: monitor(max): 0.979583
2022-01-28 16:17:53,305 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:53,342 P57305 INFO Train loss: 0.082078
2022-01-28 16:17:53,342 P57305 INFO ************ Epoch=25 end ************
2022-01-28 16:17:57,080 P57305 INFO [Metrics] AUC: 0.980287 - logloss: 0.191172
2022-01-28 16:17:57,080 P57305 INFO Save best model: monitor(max): 0.980287
2022-01-28 16:17:57,084 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:17:57,123 P57305 INFO Train loss: 0.080971
2022-01-28 16:17:57,124 P57305 INFO ************ Epoch=26 end ************
2022-01-28 16:18:00,858 P57305 INFO [Metrics] AUC: 0.979579 - logloss: 0.196943
2022-01-28 16:18:00,859 P57305 INFO Monitor(max) STOP: 0.979579 !
2022-01-28 16:18:00,859 P57305 INFO Reduce learning rate on plateau: 0.000100
2022-01-28 16:18:00,859 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:18:00,896 P57305 INFO Train loss: 0.078650
2022-01-28 16:18:00,897 P57305 INFO ************ Epoch=27 end ************
2022-01-28 16:18:04,514 P57305 INFO [Metrics] AUC: 0.980410 - logloss: 0.197547
2022-01-28 16:18:04,515 P57305 INFO Save best model: monitor(max): 0.980410
2022-01-28 16:18:04,518 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:18:04,553 P57305 INFO Train loss: 0.068382
2022-01-28 16:18:04,553 P57305 INFO ************ Epoch=28 end ************
2022-01-28 16:18:08,166 P57305 INFO [Metrics] AUC: 0.980412 - logloss: 0.200697
2022-01-28 16:18:08,166 P57305 INFO Save best model: monitor(max): 0.980412
2022-01-28 16:18:08,170 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:18:08,205 P57305 INFO Train loss: 0.064323
2022-01-28 16:18:08,205 P57305 INFO ************ Epoch=29 end ************
2022-01-28 16:18:11,804 P57305 INFO [Metrics] AUC: 0.980541 - logloss: 0.201140
2022-01-28 16:18:11,804 P57305 INFO Save best model: monitor(max): 0.980541
2022-01-28 16:18:11,808 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:18:11,843 P57305 INFO Train loss: 0.062241
2022-01-28 16:18:11,844 P57305 INFO ************ Epoch=30 end ************
2022-01-28 16:18:15,385 P57305 INFO [Metrics] AUC: 0.980689 - logloss: 0.201133
2022-01-28 16:18:15,386 P57305 INFO Save best model: monitor(max): 0.980689
2022-01-28 16:18:15,390 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:18:15,427 P57305 INFO Train loss: 0.061357
2022-01-28 16:18:15,427 P57305 INFO ************ Epoch=31 end ************
2022-01-28 16:18:18,963 P57305 INFO [Metrics] AUC: 0.980733 - logloss: 0.203337
2022-01-28 16:18:18,964 P57305 INFO Save best model: monitor(max): 0.980733
2022-01-28 16:18:18,968 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:18:19,019 P57305 INFO Train loss: 0.060321
2022-01-28 16:18:19,019 P57305 INFO ************ Epoch=32 end ************
2022-01-28 16:18:22,624 P57305 INFO [Metrics] AUC: 0.980573 - logloss: 0.203423
2022-01-28 16:18:22,624 P57305 INFO Monitor(max) STOP: 0.980573 !
2022-01-28 16:18:22,624 P57305 INFO Reduce learning rate on plateau: 0.000010
2022-01-28 16:18:22,625 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:18:22,662 P57305 INFO Train loss: 0.059471
2022-01-28 16:18:22,662 P57305 INFO ************ Epoch=33 end ************
2022-01-28 16:18:26,369 P57305 INFO [Metrics] AUC: 0.980658 - logloss: 0.203131
2022-01-28 16:18:26,369 P57305 INFO Monitor(max) STOP: 0.980658 !
2022-01-28 16:18:26,369 P57305 INFO Reduce learning rate on plateau: 0.000001
2022-01-28 16:18:26,369 P57305 INFO Early stopping at epoch=34
2022-01-28 16:18:26,369 P57305 INFO --- 50/50 batches finished ---
2022-01-28 16:18:26,403 P57305 INFO Train loss: 0.058393
2022-01-28 16:18:26,404 P57305 INFO Training finished.
2022-01-28 16:18:26,404 P57305 INFO Load best model: /home/XXX/benchmarks/Frappe/NFM_frappe_x1/frappe_x1_04e961e9/NFM_frappe_x1_012_3ebff53c.model
2022-01-28 16:18:30,124 P57305 INFO ****** Validation evaluation ******
2022-01-28 16:18:30,549 P57305 INFO [Metrics] AUC: 0.980733 - logloss: 0.203337
2022-01-28 16:18:30,605 P57305 INFO ******** Test evaluation ********
2022-01-28 16:18:30,606 P57305 INFO Loading data...
2022-01-28 16:18:30,606 P57305 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-01-28 16:18:30,609 P57305 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-01-28 16:18:30,609 P57305 INFO Loading test data done.
2022-01-28 16:18:30,898 P57305 INFO [Metrics] AUC: 0.980409 - logloss: 0.205795

```
