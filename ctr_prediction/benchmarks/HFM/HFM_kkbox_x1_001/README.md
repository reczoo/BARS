## HFM_KKBox_x4_001 

A notebook to benchmark HFM on KKBox_x4_001 dataset.

Author: [XUEPAI Team](https://github.com/xue-pai)


### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
  RAM: 500G+
  ```
+ Software

  ```python
  python: 3.6.5
  pandas: 1.0.0
  numpy: 1.18.1
  ```

### Dataset
In this setting, For all categorical fields, we replace infrequent features with a default ``<OOV>`` token by setting the threshold min_category_count=10. 

To make a fair comparison, we fix **embedding_dim=128**, which performs well.


### Code
1. Install FuxiCTR
  
    Install FuxiCTR via `pip install fuxictr==1.0` to get all dependencies ready. Then download [the FuxiCTR repository](https://github.com/huawei-noah/benchmark/archive/53e314461c19dbc7f462b42bf0f0bfae020dc398.zip) to your local path.

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/KKBox/KKBox_x4/split_kkbox_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [HFM_kkbox_x4_tuner_config_02.yaml](./HFM_kkbox_x4_tuner_config_02.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/HFM_kkbox_x4_tuner_config_02.yaml --tag 002 --gpu 0
  ```


### Results
```python
[Metrics] logloss: 0.497252 - AUC: 0.838807
```


### Logs
```python
2020-05-07 05:49:41,445 P14404 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "dataset_id": "kkbox_x4_001_c5c9c6e3",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "5e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[64, 64, 64]",
    "interaction_type": "circular_correlation",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "HFM",
    "model_id": "HFM_kkbox_x4_002_50e645d9",
    "model_root": "./KKBox/HFM_kkbox/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_dnn": "False",
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/KKBox/",
    "test_data": "../data/KKBox/kkbox_x4_001_c5c9c6e3/test.h5",
    "train_data": "../data/KKBox/kkbox_x4_001_c5c9c6e3/train.h5",
    "valid_data": "../data/KKBox/kkbox_x4_001_c5c9c6e3/valid.h5",
    "version": "pytorch",
    "gpu": "1"
}
2020-05-07 05:49:41,446 P14404 INFO Set up feature encoder...
2020-05-07 05:49:41,446 P14404 INFO Load feature_map from json: ../data/KKBox/kkbox_x4_001_c5c9c6e3/feature_map.json
2020-05-07 05:49:41,446 P14404 INFO Loading data...
2020-05-07 05:49:41,448 P14404 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/train.h5
2020-05-07 05:49:44,864 P14404 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/valid.h5
2020-05-07 05:49:46,270 P14404 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2020-05-07 05:49:46,298 P14404 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-05-07 05:49:46,298 P14404 INFO Loading train data done.
2020-05-07 05:49:55,577 P14404 INFO **** Start training: 591 batches/epoch ****
2020-05-07 05:53:34,468 P14404 INFO [Metrics] logloss: 0.555702 - AUC: 0.786895
2020-05-07 05:53:34,481 P14404 INFO Save best model: monitor(max): 0.231193
2020-05-07 05:53:34,523 P14404 INFO --- 591/591 batches finished ---
2020-05-07 05:53:34,567 P14404 INFO Train loss: 0.597183
2020-05-07 05:53:34,567 P14404 INFO ************ Epoch=1 end ************
2020-05-07 05:57:10,373 P14404 INFO [Metrics] logloss: 0.545353 - AUC: 0.796976
2020-05-07 05:57:10,392 P14404 INFO Save best model: monitor(max): 0.251623
2020-05-07 05:57:10,458 P14404 INFO --- 591/591 batches finished ---
2020-05-07 05:57:10,529 P14404 INFO Train loss: 0.581076
2020-05-07 05:57:10,529 P14404 INFO ************ Epoch=2 end ************
2020-05-07 06:00:45,817 P14404 INFO [Metrics] logloss: 0.539238 - AUC: 0.802381
2020-05-07 06:00:45,833 P14404 INFO Save best model: monitor(max): 0.263142
2020-05-07 06:00:45,890 P14404 INFO --- 591/591 batches finished ---
2020-05-07 06:00:45,957 P14404 INFO Train loss: 0.575700
2020-05-07 06:00:45,957 P14404 INFO ************ Epoch=3 end ************
2020-05-07 06:04:22,265 P14404 INFO [Metrics] logloss: 0.535431 - AUC: 0.805772
2020-05-07 06:04:22,278 P14404 INFO Save best model: monitor(max): 0.270340
2020-05-07 06:04:22,338 P14404 INFO --- 591/591 batches finished ---
2020-05-07 06:04:22,419 P14404 INFO Train loss: 0.571962
2020-05-07 06:04:22,420 P14404 INFO ************ Epoch=4 end ************
2020-05-07 06:07:57,666 P14404 INFO [Metrics] logloss: 0.531701 - AUC: 0.809208
2020-05-07 06:07:57,684 P14404 INFO Save best model: monitor(max): 0.277508
2020-05-07 06:07:57,742 P14404 INFO --- 591/591 batches finished ---
2020-05-07 06:07:57,814 P14404 INFO Train loss: 0.568981
2020-05-07 06:07:57,814 P14404 INFO ************ Epoch=5 end ************
2020-05-07 06:11:32,657 P14404 INFO [Metrics] logloss: 0.529541 - AUC: 0.810881
2020-05-07 06:11:32,675 P14404 INFO Save best model: monitor(max): 0.281340
2020-05-07 06:11:32,748 P14404 INFO --- 591/591 batches finished ---
2020-05-07 06:11:32,818 P14404 INFO Train loss: 0.566131
2020-05-07 06:11:32,818 P14404 INFO ************ Epoch=6 end ************
2020-05-07 06:15:07,860 P14404 INFO [Metrics] logloss: 0.526645 - AUC: 0.813427
2020-05-07 06:15:07,873 P14404 INFO Save best model: monitor(max): 0.286782
2020-05-07 06:15:07,935 P14404 INFO --- 591/591 batches finished ---
2020-05-07 06:15:07,997 P14404 INFO Train loss: 0.563486
2020-05-07 06:15:07,997 P14404 INFO ************ Epoch=7 end ************
2020-05-07 06:18:44,293 P14404 INFO [Metrics] logloss: 0.525341 - AUC: 0.815002
2020-05-07 06:18:44,306 P14404 INFO Save best model: monitor(max): 0.289661
2020-05-07 06:18:44,370 P14404 INFO --- 591/591 batches finished ---
2020-05-07 06:18:44,443 P14404 INFO Train loss: 0.561038
2020-05-07 06:18:44,443 P14404 INFO ************ Epoch=8 end ************
2020-05-07 06:22:19,154 P14404 INFO [Metrics] logloss: 0.521815 - AUC: 0.817516
2020-05-07 06:22:19,175 P14404 INFO Save best model: monitor(max): 0.295701
2020-05-07 06:22:19,254 P14404 INFO --- 591/591 batches finished ---
2020-05-07 06:22:19,323 P14404 INFO Train loss: 0.558640
2020-05-07 06:22:19,323 P14404 INFO ************ Epoch=9 end ************
2020-05-07 06:25:55,713 P14404 INFO [Metrics] logloss: 0.520885 - AUC: 0.818324
2020-05-07 06:25:55,730 P14404 INFO Save best model: monitor(max): 0.297439
2020-05-07 06:25:55,804 P14404 INFO --- 591/591 batches finished ---
2020-05-07 06:25:55,877 P14404 INFO Train loss: 0.556447
2020-05-07 06:25:55,878 P14404 INFO ************ Epoch=10 end ************
2020-05-07 06:29:30,745 P14404 INFO [Metrics] logloss: 0.519756 - AUC: 0.819432
2020-05-07 06:29:30,763 P14404 INFO Save best model: monitor(max): 0.299676
2020-05-07 06:29:30,825 P14404 INFO --- 591/591 batches finished ---
2020-05-07 06:29:30,883 P14404 INFO Train loss: 0.554051
2020-05-07 06:29:30,883 P14404 INFO ************ Epoch=11 end ************
2020-05-07 06:33:06,622 P14404 INFO [Metrics] logloss: 0.518116 - AUC: 0.820770
2020-05-07 06:33:06,639 P14404 INFO Save best model: monitor(max): 0.302654
2020-05-07 06:33:06,710 P14404 INFO --- 591/591 batches finished ---
2020-05-07 06:33:06,780 P14404 INFO Train loss: 0.552084
2020-05-07 06:33:06,780 P14404 INFO ************ Epoch=12 end ************
2020-05-07 06:36:41,690 P14404 INFO [Metrics] logloss: 0.516413 - AUC: 0.822139
2020-05-07 06:36:41,708 P14404 INFO Save best model: monitor(max): 0.305726
2020-05-07 06:36:41,787 P14404 INFO --- 591/591 batches finished ---
2020-05-07 06:36:41,853 P14404 INFO Train loss: 0.549862
2020-05-07 06:36:41,853 P14404 INFO ************ Epoch=13 end ************
2020-05-07 06:40:18,623 P14404 INFO [Metrics] logloss: 0.515958 - AUC: 0.822831
2020-05-07 06:40:18,641 P14404 INFO Save best model: monitor(max): 0.306873
2020-05-07 06:40:18,711 P14404 INFO --- 591/591 batches finished ---
2020-05-07 06:40:18,768 P14404 INFO Train loss: 0.547945
2020-05-07 06:40:18,768 P14404 INFO ************ Epoch=14 end ************
2020-05-07 06:43:53,602 P14404 INFO [Metrics] logloss: 0.514903 - AUC: 0.823540
2020-05-07 06:43:53,618 P14404 INFO Save best model: monitor(max): 0.308637
2020-05-07 06:43:53,682 P14404 INFO --- 591/591 batches finished ---
2020-05-07 06:43:53,743 P14404 INFO Train loss: 0.546198
2020-05-07 06:43:53,744 P14404 INFO ************ Epoch=15 end ************
2020-05-07 06:47:29,552 P14404 INFO [Metrics] logloss: 0.514436 - AUC: 0.824092
2020-05-07 06:47:29,564 P14404 INFO Save best model: monitor(max): 0.309656
2020-05-07 06:47:29,622 P14404 INFO --- 591/591 batches finished ---
2020-05-07 06:47:29,701 P14404 INFO Train loss: 0.544435
2020-05-07 06:47:29,701 P14404 INFO ************ Epoch=16 end ************
2020-05-07 06:51:04,463 P14404 INFO [Metrics] logloss: 0.513631 - AUC: 0.825098
2020-05-07 06:51:04,475 P14404 INFO Save best model: monitor(max): 0.311467
2020-05-07 06:51:04,530 P14404 INFO --- 591/591 batches finished ---
2020-05-07 06:51:04,591 P14404 INFO Train loss: 0.542805
2020-05-07 06:51:04,591 P14404 INFO ************ Epoch=17 end ************
2020-05-07 06:54:40,221 P14404 INFO [Metrics] logloss: 0.513200 - AUC: 0.825488
2020-05-07 06:54:40,235 P14404 INFO Save best model: monitor(max): 0.312288
2020-05-07 06:54:40,294 P14404 INFO --- 591/591 batches finished ---
2020-05-07 06:54:40,377 P14404 INFO Train loss: 0.541214
2020-05-07 06:54:40,377 P14404 INFO ************ Epoch=18 end ************
2020-05-07 06:58:15,108 P14404 INFO [Metrics] logloss: 0.512703 - AUC: 0.826331
2020-05-07 06:58:15,121 P14404 INFO Save best model: monitor(max): 0.313627
2020-05-07 06:58:15,177 P14404 INFO --- 591/591 batches finished ---
2020-05-07 06:58:15,258 P14404 INFO Train loss: 0.539648
2020-05-07 06:58:15,258 P14404 INFO ************ Epoch=19 end ************
2020-05-07 07:01:50,258 P14404 INFO [Metrics] logloss: 0.511789 - AUC: 0.826804
2020-05-07 07:01:50,269 P14404 INFO Save best model: monitor(max): 0.315016
2020-05-07 07:01:50,334 P14404 INFO --- 591/591 batches finished ---
2020-05-07 07:01:50,416 P14404 INFO Train loss: 0.538389
2020-05-07 07:01:50,416 P14404 INFO ************ Epoch=20 end ************
2020-05-07 07:05:25,441 P14404 INFO [Metrics] logloss: 0.512794 - AUC: 0.826485
2020-05-07 07:05:25,455 P14404 INFO Monitor(max) STOP: 0.313691 !
2020-05-07 07:05:25,455 P14404 INFO Reduce learning rate on plateau: 0.000100
2020-05-07 07:05:25,455 P14404 INFO --- 591/591 batches finished ---
2020-05-07 07:05:25,529 P14404 INFO Train loss: 0.536966
2020-05-07 07:05:25,529 P14404 INFO ************ Epoch=21 end ************
2020-05-07 07:09:00,829 P14404 INFO [Metrics] logloss: 0.498337 - AUC: 0.837042
2020-05-07 07:09:00,841 P14404 INFO Save best model: monitor(max): 0.338705
2020-05-07 07:09:00,906 P14404 INFO --- 591/591 batches finished ---
2020-05-07 07:09:00,986 P14404 INFO Train loss: 0.482476
2020-05-07 07:09:00,986 P14404 INFO ************ Epoch=22 end ************
2020-05-07 07:12:37,561 P14404 INFO [Metrics] logloss: 0.497029 - AUC: 0.838410
2020-05-07 07:12:37,575 P14404 INFO Save best model: monitor(max): 0.341381
2020-05-07 07:12:37,632 P14404 INFO --- 591/591 batches finished ---
2020-05-07 07:12:37,709 P14404 INFO Train loss: 0.466567
2020-05-07 07:12:37,709 P14404 INFO ************ Epoch=23 end ************
2020-05-07 07:16:12,347 P14404 INFO [Metrics] logloss: 0.497246 - AUC: 0.838884
2020-05-07 07:16:12,359 P14404 INFO Save best model: monitor(max): 0.341638
2020-05-07 07:16:12,416 P14404 INFO --- 591/591 batches finished ---
2020-05-07 07:16:12,496 P14404 INFO Train loss: 0.458596
2020-05-07 07:16:12,497 P14404 INFO ************ Epoch=24 end ************
2020-05-07 07:19:47,809 P14404 INFO [Metrics] logloss: 0.498176 - AUC: 0.838710
2020-05-07 07:19:47,820 P14404 INFO Monitor(max) STOP: 0.340534 !
2020-05-07 07:19:47,821 P14404 INFO Reduce learning rate on plateau: 0.000010
2020-05-07 07:19:47,821 P14404 INFO --- 591/591 batches finished ---
2020-05-07 07:19:47,901 P14404 INFO Train loss: 0.453245
2020-05-07 07:19:47,901 P14404 INFO ************ Epoch=25 end ************
2020-05-07 07:23:22,678 P14404 INFO [Metrics] logloss: 0.497800 - AUC: 0.839141
2020-05-07 07:23:22,692 P14404 INFO Monitor(max) STOP: 0.341340 !
2020-05-07 07:23:22,692 P14404 INFO Reduce learning rate on plateau: 0.000001
2020-05-07 07:23:22,692 P14404 INFO Early stopping at epoch=26
2020-05-07 07:23:22,693 P14404 INFO --- 591/591 batches finished ---
2020-05-07 07:23:22,773 P14404 INFO Train loss: 0.440386
2020-05-07 07:23:22,773 P14404 INFO Training finished.
2020-05-07 07:23:22,773 P14404 INFO Load best model: /home/hispace/container/data/xxx/FuxiCTR/benchmarks/KKBox/HFM_kkbox/kkbox_x4_001_c5c9c6e3/HFM_kkbox_x4_002_50e645d9_kkbox_x4_001_c5c9c6e3_model.ckpt
2020-05-07 07:23:22,846 P14404 INFO ****** Train/validation evaluation ******
2020-05-07 07:24:41,393 P14404 INFO [Metrics] logloss: 0.389339 - AUC: 0.910810
2020-05-07 07:24:50,593 P14404 INFO [Metrics] logloss: 0.497246 - AUC: 0.838884
2020-05-07 07:24:50,691 P14404 INFO ******** Test evaluation ********
2020-05-07 07:24:50,691 P14404 INFO Loading data...
2020-05-07 07:24:50,691 P14404 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/test.h5
2020-05-07 07:24:50,778 P14404 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-05-07 07:24:50,779 P14404 INFO Loading test data done.
2020-05-07 07:24:59,627 P14404 INFO [Metrics] logloss: 0.497252 - AUC: 0.838807
```
