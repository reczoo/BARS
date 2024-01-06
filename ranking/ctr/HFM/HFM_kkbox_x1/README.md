## HFM_kkbox_x1

A hands-on guide to run the HFM model on the KKBox_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) Gold 6278C CPU @ 2.60GHz
  GPU: Tesla V100 32G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 10.2
  python: 3.6.4
  pytorch: 1.0.0
  pandas: 0.22.0
  numpy: 1.19.2
  scipy: 1.5.4
  sklearn: 0.22.1
  pyyaml: 5.4.1
  h5py: 2.8.0
  tqdm: 4.60.0
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [KKBox_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/KKBox#KKBox_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [HFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/HFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HFM_kkbox_x1_tuner_config_03](./HFM_kkbox_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HFM_kkbox_x1
    nohup python run_expid.py --config ./HFM_kkbox_x1_tuner_config_03 --expid HFM_kkbox_x1_006_b549df4b --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.496967 | 0.839165  |


### Logs
```python
2022-03-11 15:14:35,074 P88881 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "5e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "5",
    "hidden_activations": "relu",
    "hidden_units": "[64, 64, 64]",
    "interaction_type": "circular_correlation",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "HFM",
    "model_id": "HFM_kkbox_x1_006_b549df4b",
    "model_root": "./KKBox/HFM_kkbox_x1/",
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
    "test_data": "../data/KKBox/KKBox_x1/test.csv",
    "train_data": "../data/KKBox/KKBox_x1/train.csv",
    "use_dnn": "False",
    "use_hdf5": "True",
    "valid_data": "../data/KKBox/KKBox_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2022-03-11 15:14:35,075 P88881 INFO Set up feature encoder...
2022-03-11 15:14:35,075 P88881 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-11 15:14:35,983 P88881 INFO Total number of parameters: 11899992.
2022-03-11 15:14:35,983 P88881 INFO Loading data...
2022-03-11 15:14:35,984 P88881 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-11 15:14:36,386 P88881 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-11 15:14:36,593 P88881 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-11 15:14:36,611 P88881 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-11 15:14:36,611 P88881 INFO Loading train data done.
2022-03-11 15:14:40,524 P88881 INFO Start training: 591 batches/epoch
2022-03-11 15:14:40,524 P88881 INFO ************ Epoch=1 start ************
2022-03-11 15:17:17,767 P88881 INFO [Metrics] logloss: 0.554428 - AUC: 0.788067
2022-03-11 15:17:17,771 P88881 INFO Save best model: monitor(max): 0.233639
2022-03-11 15:17:18,089 P88881 INFO --- 591/591 batches finished ---
2022-03-11 15:17:18,131 P88881 INFO Train loss: 0.596469
2022-03-11 15:17:18,131 P88881 INFO ************ Epoch=1 end ************
2022-03-11 15:19:54,924 P88881 INFO [Metrics] logloss: 0.545194 - AUC: 0.797156
2022-03-11 15:19:54,924 P88881 INFO Save best model: monitor(max): 0.251963
2022-03-11 15:19:54,990 P88881 INFO --- 591/591 batches finished ---
2022-03-11 15:19:55,024 P88881 INFO Train loss: 0.580448
2022-03-11 15:19:55,024 P88881 INFO ************ Epoch=2 end ************
2022-03-11 15:22:31,678 P88881 INFO [Metrics] logloss: 0.539089 - AUC: 0.802376
2022-03-11 15:22:31,679 P88881 INFO Save best model: monitor(max): 0.263287
2022-03-11 15:22:31,724 P88881 INFO --- 591/591 batches finished ---
2022-03-11 15:22:31,758 P88881 INFO Train loss: 0.574980
2022-03-11 15:22:31,759 P88881 INFO ************ Epoch=3 end ************
2022-03-11 15:25:08,590 P88881 INFO [Metrics] logloss: 0.534595 - AUC: 0.806422
2022-03-11 15:25:08,591 P88881 INFO Save best model: monitor(max): 0.271827
2022-03-11 15:25:08,638 P88881 INFO --- 591/591 batches finished ---
2022-03-11 15:25:08,678 P88881 INFO Train loss: 0.570875
2022-03-11 15:25:08,678 P88881 INFO ************ Epoch=4 end ************
2022-03-11 15:27:45,498 P88881 INFO [Metrics] logloss: 0.530597 - AUC: 0.810022
2022-03-11 15:27:45,499 P88881 INFO Save best model: monitor(max): 0.279425
2022-03-11 15:27:45,566 P88881 INFO --- 591/591 batches finished ---
2022-03-11 15:27:45,600 P88881 INFO Train loss: 0.567274
2022-03-11 15:27:45,601 P88881 INFO ************ Epoch=5 end ************
2022-03-11 15:30:22,340 P88881 INFO [Metrics] logloss: 0.527992 - AUC: 0.812138
2022-03-11 15:30:22,341 P88881 INFO Save best model: monitor(max): 0.284146
2022-03-11 15:30:22,398 P88881 INFO --- 591/591 batches finished ---
2022-03-11 15:30:22,435 P88881 INFO Train loss: 0.563865
2022-03-11 15:30:22,435 P88881 INFO ************ Epoch=6 end ************
2022-03-11 15:32:58,897 P88881 INFO [Metrics] logloss: 0.525229 - AUC: 0.814510
2022-03-11 15:32:58,898 P88881 INFO Save best model: monitor(max): 0.289281
2022-03-11 15:32:58,954 P88881 INFO --- 591/591 batches finished ---
2022-03-11 15:32:58,989 P88881 INFO Train loss: 0.560856
2022-03-11 15:32:58,990 P88881 INFO ************ Epoch=7 end ************
2022-03-11 15:35:35,883 P88881 INFO [Metrics] logloss: 0.524722 - AUC: 0.815538
2022-03-11 15:35:35,884 P88881 INFO Save best model: monitor(max): 0.290816
2022-03-11 15:35:35,938 P88881 INFO --- 591/591 batches finished ---
2022-03-11 15:35:35,973 P88881 INFO Train loss: 0.558178
2022-03-11 15:35:35,973 P88881 INFO ************ Epoch=8 end ************
2022-03-11 15:38:12,393 P88881 INFO [Metrics] logloss: 0.521129 - AUC: 0.818040
2022-03-11 15:38:12,393 P88881 INFO Save best model: monitor(max): 0.296911
2022-03-11 15:38:12,457 P88881 INFO --- 591/591 batches finished ---
2022-03-11 15:38:12,491 P88881 INFO Train loss: 0.555792
2022-03-11 15:38:12,491 P88881 INFO ************ Epoch=9 end ************
2022-03-11 15:40:49,485 P88881 INFO [Metrics] logloss: 0.520179 - AUC: 0.818953
2022-03-11 15:40:49,486 P88881 INFO Save best model: monitor(max): 0.298774
2022-03-11 15:40:49,532 P88881 INFO --- 591/591 batches finished ---
2022-03-11 15:40:49,572 P88881 INFO Train loss: 0.553706
2022-03-11 15:40:49,572 P88881 INFO ************ Epoch=10 end ************
2022-03-11 15:43:26,214 P88881 INFO [Metrics] logloss: 0.518533 - AUC: 0.820258
2022-03-11 15:43:26,215 P88881 INFO Save best model: monitor(max): 0.301725
2022-03-11 15:43:26,269 P88881 INFO --- 591/591 batches finished ---
2022-03-11 15:43:26,303 P88881 INFO Train loss: 0.551611
2022-03-11 15:43:26,303 P88881 INFO ************ Epoch=11 end ************
2022-03-11 15:46:03,042 P88881 INFO [Metrics] logloss: 0.517794 - AUC: 0.820874
2022-03-11 15:46:03,043 P88881 INFO Save best model: monitor(max): 0.303080
2022-03-11 15:46:03,097 P88881 INFO --- 591/591 batches finished ---
2022-03-11 15:46:03,138 P88881 INFO Train loss: 0.549997
2022-03-11 15:46:03,138 P88881 INFO ************ Epoch=12 end ************
2022-03-11 15:48:39,749 P88881 INFO [Metrics] logloss: 0.515848 - AUC: 0.822445
2022-03-11 15:48:39,750 P88881 INFO Save best model: monitor(max): 0.306598
2022-03-11 15:48:39,812 P88881 INFO --- 591/591 batches finished ---
2022-03-11 15:48:39,851 P88881 INFO Train loss: 0.548202
2022-03-11 15:48:39,851 P88881 INFO ************ Epoch=13 end ************
2022-03-11 15:51:16,442 P88881 INFO [Metrics] logloss: 0.516113 - AUC: 0.822738
2022-03-11 15:51:16,442 P88881 INFO Save best model: monitor(max): 0.306625
2022-03-11 15:51:16,489 P88881 INFO --- 591/591 batches finished ---
2022-03-11 15:51:16,526 P88881 INFO Train loss: 0.546675
2022-03-11 15:51:16,526 P88881 INFO ************ Epoch=14 end ************
2022-03-11 15:53:53,018 P88881 INFO [Metrics] logloss: 0.514767 - AUC: 0.823396
2022-03-11 15:53:53,018 P88881 INFO Save best model: monitor(max): 0.308630
2022-03-11 15:53:53,072 P88881 INFO --- 591/591 batches finished ---
2022-03-11 15:53:53,111 P88881 INFO Train loss: 0.545308
2022-03-11 15:53:53,111 P88881 INFO ************ Epoch=15 end ************
2022-03-11 15:56:29,646 P88881 INFO [Metrics] logloss: 0.514017 - AUC: 0.824306
2022-03-11 15:56:29,647 P88881 INFO Save best model: monitor(max): 0.310289
2022-03-11 15:56:29,693 P88881 INFO --- 591/591 batches finished ---
2022-03-11 15:56:29,733 P88881 INFO Train loss: 0.543917
2022-03-11 15:56:29,734 P88881 INFO ************ Epoch=16 end ************
2022-03-11 15:59:05,981 P88881 INFO [Metrics] logloss: 0.513833 - AUC: 0.824792
2022-03-11 15:59:05,982 P88881 INFO Save best model: monitor(max): 0.310959
2022-03-11 15:59:06,040 P88881 INFO --- 591/591 batches finished ---
2022-03-11 15:59:06,080 P88881 INFO Train loss: 0.542694
2022-03-11 15:59:06,080 P88881 INFO ************ Epoch=17 end ************
2022-03-11 16:01:42,932 P88881 INFO [Metrics] logloss: 0.513131 - AUC: 0.825340
2022-03-11 16:01:42,933 P88881 INFO Save best model: monitor(max): 0.312209
2022-03-11 16:01:42,980 P88881 INFO --- 591/591 batches finished ---
2022-03-11 16:01:43,021 P88881 INFO Train loss: 0.541359
2022-03-11 16:01:43,021 P88881 INFO ************ Epoch=18 end ************
2022-03-11 16:04:19,506 P88881 INFO [Metrics] logloss: 0.512596 - AUC: 0.825930
2022-03-11 16:04:19,506 P88881 INFO Save best model: monitor(max): 0.313334
2022-03-11 16:04:19,567 P88881 INFO --- 591/591 batches finished ---
2022-03-11 16:04:19,601 P88881 INFO Train loss: 0.540118
2022-03-11 16:04:19,601 P88881 INFO ************ Epoch=19 end ************
2022-03-11 16:06:56,066 P88881 INFO [Metrics] logloss: 0.511625 - AUC: 0.826714
2022-03-11 16:06:56,066 P88881 INFO Save best model: monitor(max): 0.315089
2022-03-11 16:06:56,115 P88881 INFO --- 591/591 batches finished ---
2022-03-11 16:06:56,155 P88881 INFO Train loss: 0.539082
2022-03-11 16:06:56,156 P88881 INFO ************ Epoch=20 end ************
2022-03-11 16:09:32,861 P88881 INFO [Metrics] logloss: 0.513005 - AUC: 0.826106
2022-03-11 16:09:32,862 P88881 INFO Monitor(max) STOP: 0.313100 !
2022-03-11 16:09:32,862 P88881 INFO Reduce learning rate on plateau: 0.000100
2022-03-11 16:09:32,862 P88881 INFO --- 591/591 batches finished ---
2022-03-11 16:09:32,905 P88881 INFO Train loss: 0.537960
2022-03-11 16:09:32,905 P88881 INFO ************ Epoch=21 end ************
2022-03-11 16:12:09,181 P88881 INFO [Metrics] logloss: 0.498592 - AUC: 0.836603
2022-03-11 16:12:09,182 P88881 INFO Save best model: monitor(max): 0.338011
2022-03-11 16:12:09,228 P88881 INFO --- 591/591 batches finished ---
2022-03-11 16:12:09,267 P88881 INFO Train loss: 0.486466
2022-03-11 16:12:09,267 P88881 INFO ************ Epoch=22 end ************
2022-03-11 16:14:45,985 P88881 INFO [Metrics] logloss: 0.496931 - AUC: 0.838118
2022-03-11 16:14:45,986 P88881 INFO Save best model: monitor(max): 0.341187
2022-03-11 16:14:46,041 P88881 INFO --- 591/591 batches finished ---
2022-03-11 16:14:46,076 P88881 INFO Train loss: 0.471215
2022-03-11 16:14:46,076 P88881 INFO ************ Epoch=23 end ************
2022-03-11 16:17:22,727 P88881 INFO [Metrics] logloss: 0.496958 - AUC: 0.838632
2022-03-11 16:17:22,728 P88881 INFO Save best model: monitor(max): 0.341674
2022-03-11 16:17:22,777 P88881 INFO --- 591/591 batches finished ---
2022-03-11 16:17:22,817 P88881 INFO Train loss: 0.463582
2022-03-11 16:17:22,817 P88881 INFO ************ Epoch=24 end ************
2022-03-11 16:19:59,367 P88881 INFO [Metrics] logloss: 0.497626 - AUC: 0.838537
2022-03-11 16:19:59,368 P88881 INFO Monitor(max) STOP: 0.340912 !
2022-03-11 16:19:59,368 P88881 INFO Reduce learning rate on plateau: 0.000010
2022-03-11 16:19:59,368 P88881 INFO --- 591/591 batches finished ---
2022-03-11 16:19:59,408 P88881 INFO Train loss: 0.458449
2022-03-11 16:19:59,408 P88881 INFO ************ Epoch=25 end ************
2022-03-11 16:22:36,012 P88881 INFO [Metrics] logloss: 0.497235 - AUC: 0.838981
2022-03-11 16:22:36,013 P88881 INFO Save best model: monitor(max): 0.341746
2022-03-11 16:22:36,060 P88881 INFO --- 591/591 batches finished ---
2022-03-11 16:22:36,102 P88881 INFO Train loss: 0.446055
2022-03-11 16:22:36,102 P88881 INFO ************ Epoch=26 end ************
2022-03-11 16:25:12,736 P88881 INFO [Metrics] logloss: 0.497249 - AUC: 0.838978
2022-03-11 16:25:12,737 P88881 INFO Monitor(max) STOP: 0.341728 !
2022-03-11 16:25:12,737 P88881 INFO Reduce learning rate on plateau: 0.000001
2022-03-11 16:25:12,738 P88881 INFO --- 591/591 batches finished ---
2022-03-11 16:25:12,775 P88881 INFO Train loss: 0.445276
2022-03-11 16:25:12,775 P88881 INFO ************ Epoch=27 end ************
2022-03-11 16:27:49,351 P88881 INFO [Metrics] logloss: 0.497250 - AUC: 0.838981
2022-03-11 16:27:49,353 P88881 INFO Monitor(max) STOP: 0.341731 !
2022-03-11 16:27:49,353 P88881 INFO Reduce learning rate on plateau: 0.000001
2022-03-11 16:27:49,353 P88881 INFO Early stopping at epoch=28
2022-03-11 16:27:49,353 P88881 INFO --- 591/591 batches finished ---
2022-03-11 16:27:49,393 P88881 INFO Train loss: 0.443902
2022-03-11 16:27:49,393 P88881 INFO Training finished.
2022-03-11 16:27:49,394 P88881 INFO Load best model: /cache/FuxiCTR/benchmarks/KKBox/HFM_kkbox_x1/kkbox_x1_227d337d/HFM_kkbox_x1_006_b549df4b_model.ckpt
2022-03-11 16:27:49,460 P88881 INFO ****** Validation evaluation ******
2022-03-11 16:27:56,418 P88881 INFO [Metrics] logloss: 0.497235 - AUC: 0.838981
2022-03-11 16:27:56,481 P88881 INFO ******** Test evaluation ********
2022-03-11 16:27:56,481 P88881 INFO Loading data...
2022-03-11 16:27:56,481 P88881 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-11 16:27:56,553 P88881 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-11 16:27:56,553 P88881 INFO Loading test data done.
2022-03-11 16:28:03,449 P88881 INFO [Metrics] logloss: 0.496967 - AUC: 0.839165

```
