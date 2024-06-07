## AFN+_kkbox_x1

A hands-on guide to run the AFN model on the KKBox_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

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

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [AFN](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/AFN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFN+_kkbox_x1_tuner_config_04](./AFN+_kkbox_x1_tuner_config_04). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFN+_kkbox_x1
    nohup python run_expid.py --config ./AFN+_kkbox_x1_tuner_config_04 --expid AFN_kkbox_x1_014_a8ea82ca --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.484192 | 0.848879  |


### Logs
```python
2022-03-09 20:22:13,217 P50150 INFO {
    "afn_activations": "relu",
    "afn_dropout": "0",
    "afn_hidden_units": "[1000, 1000, 1000]",
    "batch_norm": "True",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_dropout": "0.2",
    "dnn_hidden_units": "[1000, 1000, 1000]",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.0005",
    "ensemble_dnn": "True",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "5",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "logarithmic_neurons": "1500",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "AFN",
    "model_id": "AFN_kkbox_x1_014_a8ea82ca",
    "model_root": "./KKBox/AFN_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
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
    "use_hdf5": "True",
    "valid_data": "../data/KKBox/KKBox_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2022-03-09 20:22:13,217 P50150 INFO Set up feature encoder...
2022-03-09 20:22:13,217 P50150 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-09 20:22:17,284 P50150 INFO Total number of parameters: 221321763.
2022-03-09 20:22:17,284 P50150 INFO Loading data...
2022-03-09 20:22:17,284 P50150 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-09 20:22:17,676 P50150 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-09 20:22:17,874 P50150 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-09 20:22:17,892 P50150 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-09 20:22:17,892 P50150 INFO Loading train data done.
2022-03-09 20:22:21,646 P50150 INFO Start training: 591 batches/epoch
2022-03-09 20:22:21,646 P50150 INFO ************ Epoch=1 start ************
2022-03-09 20:38:17,045 P50150 INFO [Metrics] logloss: 0.549968 - AUC: 0.791682
2022-03-09 20:38:17,049 P50150 INFO Save best model: monitor(max): 0.241715
2022-03-09 20:38:18,279 P50150 INFO --- 591/591 batches finished ---
2022-03-09 20:38:18,319 P50150 INFO Train loss: 0.642826
2022-03-09 20:38:18,319 P50150 INFO ************ Epoch=1 end ************
2022-03-09 20:54:13,944 P50150 INFO [Metrics] logloss: 0.536037 - AUC: 0.804517
2022-03-09 20:54:13,946 P50150 INFO Save best model: monitor(max): 0.268480
2022-03-09 20:54:15,291 P50150 INFO --- 591/591 batches finished ---
2022-03-09 20:54:15,332 P50150 INFO Train loss: 0.645935
2022-03-09 20:54:15,332 P50150 INFO ************ Epoch=2 end ************
2022-03-09 21:10:11,095 P50150 INFO [Metrics] logloss: 0.529835 - AUC: 0.809767
2022-03-09 21:10:11,098 P50150 INFO Save best model: monitor(max): 0.279932
2022-03-09 21:10:12,396 P50150 INFO --- 591/591 batches finished ---
2022-03-09 21:10:12,438 P50150 INFO Train loss: 0.656286
2022-03-09 21:10:12,438 P50150 INFO ************ Epoch=3 end ************
2022-03-09 21:26:08,094 P50150 INFO [Metrics] logloss: 0.524723 - AUC: 0.814281
2022-03-09 21:26:08,098 P50150 INFO Save best model: monitor(max): 0.289558
2022-03-09 21:26:09,605 P50150 INFO --- 591/591 batches finished ---
2022-03-09 21:26:09,648 P50150 INFO Train loss: 0.651620
2022-03-09 21:26:09,648 P50150 INFO ************ Epoch=4 end ************
2022-03-09 21:42:04,741 P50150 INFO [Metrics] logloss: 0.522382 - AUC: 0.816374
2022-03-09 21:42:04,744 P50150 INFO Save best model: monitor(max): 0.293992
2022-03-09 21:42:06,287 P50150 INFO --- 591/591 batches finished ---
2022-03-09 21:42:06,329 P50150 INFO Train loss: 0.654604
2022-03-09 21:42:06,330 P50150 INFO ************ Epoch=5 end ************
2022-03-09 21:58:01,288 P50150 INFO [Metrics] logloss: 0.518361 - AUC: 0.819574
2022-03-09 21:58:01,290 P50150 INFO Save best model: monitor(max): 0.301213
2022-03-09 21:58:02,662 P50150 INFO --- 591/591 batches finished ---
2022-03-09 21:58:02,711 P50150 INFO Train loss: 0.668165
2022-03-09 21:58:02,712 P50150 INFO ************ Epoch=6 end ************
2022-03-09 22:13:57,368 P50150 INFO [Metrics] logloss: 0.516609 - AUC: 0.821168
2022-03-09 22:13:57,371 P50150 INFO Save best model: monitor(max): 0.304560
2022-03-09 22:13:58,697 P50150 INFO --- 591/591 batches finished ---
2022-03-09 22:13:58,737 P50150 INFO Train loss: 0.682851
2022-03-09 22:13:58,738 P50150 INFO ************ Epoch=7 end ************
2022-03-09 22:29:54,083 P50150 INFO [Metrics] logloss: 0.513966 - AUC: 0.823245
2022-03-09 22:29:54,086 P50150 INFO Save best model: monitor(max): 0.309279
2022-03-09 22:29:55,414 P50150 INFO --- 591/591 batches finished ---
2022-03-09 22:29:55,455 P50150 INFO Train loss: 0.691870
2022-03-09 22:29:55,455 P50150 INFO ************ Epoch=8 end ************
2022-03-09 22:45:50,166 P50150 INFO [Metrics] logloss: 0.512301 - AUC: 0.824648
2022-03-09 22:45:50,169 P50150 INFO Save best model: monitor(max): 0.312347
2022-03-09 22:45:51,467 P50150 INFO --- 591/591 batches finished ---
2022-03-09 22:45:51,508 P50150 INFO Train loss: 0.699372
2022-03-09 22:45:51,508 P50150 INFO ************ Epoch=9 end ************
2022-03-09 23:01:47,367 P50150 INFO [Metrics] logloss: 0.510794 - AUC: 0.825826
2022-03-09 23:01:47,371 P50150 INFO Save best model: monitor(max): 0.315032
2022-03-09 23:01:48,650 P50150 INFO --- 591/591 batches finished ---
2022-03-09 23:01:48,701 P50150 INFO Train loss: 0.706188
2022-03-09 23:01:48,701 P50150 INFO ************ Epoch=10 end ************
2022-03-09 23:17:43,795 P50150 INFO [Metrics] logloss: 0.509925 - AUC: 0.826631
2022-03-09 23:17:43,798 P50150 INFO Save best model: monitor(max): 0.316705
2022-03-09 23:17:45,177 P50150 INFO --- 591/591 batches finished ---
2022-03-09 23:17:45,219 P50150 INFO Train loss: 0.716689
2022-03-09 23:17:45,219 P50150 INFO ************ Epoch=11 end ************
2022-03-09 23:33:40,065 P50150 INFO [Metrics] logloss: 0.508480 - AUC: 0.828139
2022-03-09 23:33:40,068 P50150 INFO Save best model: monitor(max): 0.319659
2022-03-09 23:33:41,389 P50150 INFO --- 591/591 batches finished ---
2022-03-09 23:33:41,430 P50150 INFO Train loss: 0.720520
2022-03-09 23:33:41,430 P50150 INFO ************ Epoch=12 end ************
2022-03-09 23:49:36,756 P50150 INFO [Metrics] logloss: 0.506308 - AUC: 0.829353
2022-03-09 23:49:36,759 P50150 INFO Save best model: monitor(max): 0.323045
2022-03-09 23:49:38,054 P50150 INFO --- 591/591 batches finished ---
2022-03-09 23:49:38,096 P50150 INFO Train loss: 0.729115
2022-03-09 23:49:38,096 P50150 INFO ************ Epoch=13 end ************
2022-03-10 00:05:33,025 P50150 INFO [Metrics] logloss: 0.505634 - AUC: 0.829900
2022-03-10 00:05:33,028 P50150 INFO Save best model: monitor(max): 0.324266
2022-03-10 00:05:34,276 P50150 INFO --- 591/591 batches finished ---
2022-03-10 00:05:34,317 P50150 INFO Train loss: 0.740749
2022-03-10 00:05:34,317 P50150 INFO ************ Epoch=14 end ************
2022-03-10 00:21:29,202 P50150 INFO [Metrics] logloss: 0.505128 - AUC: 0.830299
2022-03-10 00:21:29,205 P50150 INFO Save best model: monitor(max): 0.325171
2022-03-10 00:21:30,459 P50150 INFO --- 591/591 batches finished ---
2022-03-10 00:21:30,503 P50150 INFO Train loss: 0.751454
2022-03-10 00:21:30,504 P50150 INFO ************ Epoch=15 end ************
2022-03-10 00:37:25,530 P50150 INFO [Metrics] logloss: 0.504657 - AUC: 0.830730
2022-03-10 00:37:25,533 P50150 INFO Save best model: monitor(max): 0.326073
2022-03-10 00:37:26,916 P50150 INFO --- 591/591 batches finished ---
2022-03-10 00:37:26,958 P50150 INFO Train loss: 0.763942
2022-03-10 00:37:26,958 P50150 INFO ************ Epoch=16 end ************
2022-03-10 00:53:21,962 P50150 INFO [Metrics] logloss: 0.503343 - AUC: 0.831807
2022-03-10 00:53:21,965 P50150 INFO Save best model: monitor(max): 0.328464
2022-03-10 00:53:23,260 P50150 INFO --- 591/591 batches finished ---
2022-03-10 00:53:23,301 P50150 INFO Train loss: 0.774106
2022-03-10 00:53:23,301 P50150 INFO ************ Epoch=17 end ************
2022-03-10 01:09:18,527 P50150 INFO [Metrics] logloss: 0.502648 - AUC: 0.832363
2022-03-10 01:09:18,530 P50150 INFO Save best model: monitor(max): 0.329715
2022-03-10 01:09:19,871 P50150 INFO --- 591/591 batches finished ---
2022-03-10 01:09:19,913 P50150 INFO Train loss: 0.785448
2022-03-10 01:09:19,913 P50150 INFO ************ Epoch=18 end ************
2022-03-10 01:25:15,391 P50150 INFO [Metrics] logloss: 0.501254 - AUC: 0.833336
2022-03-10 01:25:15,395 P50150 INFO Save best model: monitor(max): 0.332082
2022-03-10 01:25:16,714 P50150 INFO --- 591/591 batches finished ---
2022-03-10 01:25:16,755 P50150 INFO Train loss: 0.793539
2022-03-10 01:25:16,755 P50150 INFO ************ Epoch=19 end ************
2022-03-10 01:41:12,665 P50150 INFO [Metrics] logloss: 0.501840 - AUC: 0.833160
2022-03-10 01:41:12,668 P50150 INFO Monitor(max) STOP: 0.331320 !
2022-03-10 01:41:12,668 P50150 INFO Reduce learning rate on plateau: 0.000100
2022-03-10 01:41:12,668 P50150 INFO --- 591/591 batches finished ---
2022-03-10 01:41:12,710 P50150 INFO Train loss: 0.801821
2022-03-10 01:41:12,710 P50150 INFO ************ Epoch=20 end ************
2022-03-10 01:57:07,800 P50150 INFO [Metrics] logloss: 0.486370 - AUC: 0.845667
2022-03-10 01:57:07,804 P50150 INFO Save best model: monitor(max): 0.359298
2022-03-10 01:57:09,196 P50150 INFO --- 591/591 batches finished ---
2022-03-10 01:57:09,240 P50150 INFO Train loss: 0.735695
2022-03-10 01:57:09,240 P50150 INFO ************ Epoch=21 end ************
2022-03-10 02:13:03,894 P50150 INFO [Metrics] logloss: 0.484132 - AUC: 0.848044
2022-03-10 02:13:03,897 P50150 INFO Save best model: monitor(max): 0.363912
2022-03-10 02:13:05,225 P50150 INFO --- 591/591 batches finished ---
2022-03-10 02:13:05,265 P50150 INFO Train loss: 0.681115
2022-03-10 02:13:05,265 P50150 INFO ************ Epoch=22 end ************
2022-03-10 02:29:00,920 P50150 INFO [Metrics] logloss: 0.484375 - AUC: 0.848766
2022-03-10 02:29:00,923 P50150 INFO Save best model: monitor(max): 0.364392
2022-03-10 02:29:02,258 P50150 INFO --- 591/591 batches finished ---
2022-03-10 02:29:02,300 P50150 INFO Train loss: 0.646263
2022-03-10 02:29:02,300 P50150 INFO ************ Epoch=23 end ************
2022-03-10 02:44:57,475 P50150 INFO [Metrics] logloss: 0.485676 - AUC: 0.848775
2022-03-10 02:44:57,478 P50150 INFO Monitor(max) STOP: 0.363099 !
2022-03-10 02:44:57,478 P50150 INFO Reduce learning rate on plateau: 0.000010
2022-03-10 02:44:57,478 P50150 INFO --- 591/591 batches finished ---
2022-03-10 02:44:57,520 P50150 INFO Train loss: 0.617766
2022-03-10 02:44:57,520 P50150 INFO ************ Epoch=24 end ************
2022-03-10 03:00:52,487 P50150 INFO [Metrics] logloss: 0.492543 - AUC: 0.848486
2022-03-10 03:00:52,491 P50150 INFO Monitor(max) STOP: 0.355943 !
2022-03-10 03:00:52,491 P50150 INFO Reduce learning rate on plateau: 0.000001
2022-03-10 03:00:52,491 P50150 INFO Early stopping at epoch=25
2022-03-10 03:00:52,491 P50150 INFO --- 591/591 batches finished ---
2022-03-10 03:00:52,534 P50150 INFO Train loss: 0.585071
2022-03-10 03:00:52,534 P50150 INFO Training finished.
2022-03-10 03:00:52,534 P50150 INFO Load best model: /cache/FuxiCTR/benchmarks/KKBox/AFN_kkbox_x1/kkbox_x1_227d337d/AFN_kkbox_x1_014_a8ea82ca_model.ckpt
2022-03-10 03:00:53,989 P50150 INFO ****** Validation evaluation ******
2022-03-10 03:01:32,037 P50150 INFO [Metrics] logloss: 0.484375 - AUC: 0.848766
2022-03-10 03:01:32,088 P50150 INFO ******** Test evaluation ********
2022-03-10 03:01:32,088 P50150 INFO Loading data...
2022-03-10 03:01:32,089 P50150 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-10 03:01:32,170 P50150 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-10 03:01:32,170 P50150 INFO Loading test data done.
2022-03-10 03:02:10,113 P50150 INFO [Metrics] logloss: 0.484192 - AUC: 0.848879

```
