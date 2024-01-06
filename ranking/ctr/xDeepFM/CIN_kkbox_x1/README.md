## CIN_kkbox_x1

A hands-on guide to run the xDeepFM model on the KKBox_x1 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [xDeepFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/xDeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CIN_kkbox_x1_tuner_config_02](./CIN_kkbox_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CIN_kkbox_x1
    nohup python run_expid.py --config ./CIN_kkbox_x1_tuner_config_02 --expid xDeepFM_kkbox_x1_009_117199f0 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.490763 | 0.842723  |


### Logs
```python
2022-03-11 23:25:10,799 P32691 INFO {
    "batch_norm": "False",
    "batch_size": "5000",
    "cin_layer_units": "[78, 78]",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "dnn_hidden_units": "[]",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "5",
    "hidden_activations": "relu",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "xDeepFM",
    "model_id": "xDeepFM_kkbox_x1_009_117199f0",
    "model_root": "./KKBox/CIN_kkbox_x1/",
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
    "use_hdf5": "True",
    "valid_data": "../data/KKBox/KKBox_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2022-03-11 23:25:10,800 P32691 INFO Set up feature encoder...
2022-03-11 23:25:10,800 P32691 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-11 23:25:11,088 P32691 INFO Total number of parameters: 11992450.
2022-03-11 23:25:11,088 P32691 INFO Loading data...
2022-03-11 23:25:11,089 P32691 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-11 23:25:11,488 P32691 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-11 23:25:11,670 P32691 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-11 23:25:11,685 P32691 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-11 23:25:11,686 P32691 INFO Loading train data done.
2022-03-11 23:25:14,444 P32691 INFO Start training: 1181 batches/epoch
2022-03-11 23:25:14,444 P32691 INFO ************ Epoch=1 start ************
2022-03-11 23:33:14,382 P32691 INFO [Metrics] logloss: 0.569369 - AUC: 0.782518
2022-03-11 23:33:14,383 P32691 INFO Save best model: monitor(max): 0.213149
2022-03-11 23:33:14,624 P32691 INFO --- 1181/1181 batches finished ---
2022-03-11 23:33:14,652 P32691 INFO Train loss: 0.591440
2022-03-11 23:33:14,652 P32691 INFO ************ Epoch=1 end ************
2022-03-11 23:41:14,145 P32691 INFO [Metrics] logloss: 0.527387 - AUC: 0.813408
2022-03-11 23:41:14,145 P32691 INFO Save best model: monitor(max): 0.286022
2022-03-11 23:41:14,193 P32691 INFO --- 1181/1181 batches finished ---
2022-03-11 23:41:14,226 P32691 INFO Train loss: 0.558390
2022-03-11 23:41:14,226 P32691 INFO ************ Epoch=2 end ************
2022-03-11 23:49:13,815 P32691 INFO [Metrics] logloss: 0.500392 - AUC: 0.834398
2022-03-11 23:49:13,816 P32691 INFO Save best model: monitor(max): 0.334006
2022-03-11 23:49:13,865 P32691 INFO --- 1181/1181 batches finished ---
2022-03-11 23:49:13,912 P32691 INFO Train loss: 0.531708
2022-03-11 23:49:13,912 P32691 INFO ************ Epoch=3 end ************
2022-03-11 23:57:14,260 P32691 INFO [Metrics] logloss: 0.490377 - AUC: 0.842943
2022-03-11 23:57:14,261 P32691 INFO Save best model: monitor(max): 0.352567
2022-03-11 23:57:14,310 P32691 INFO --- 1181/1181 batches finished ---
2022-03-11 23:57:14,346 P32691 INFO Train loss: 0.507723
2022-03-11 23:57:14,346 P32691 INFO ************ Epoch=4 end ************
2022-03-12 00:05:14,053 P32691 INFO [Metrics] logloss: 0.499399 - AUC: 0.839914
2022-03-12 00:05:14,054 P32691 INFO Monitor(max) STOP: 0.340515 !
2022-03-12 00:05:14,054 P32691 INFO Reduce learning rate on plateau: 0.000100
2022-03-12 00:05:14,054 P32691 INFO --- 1181/1181 batches finished ---
2022-03-12 00:05:14,089 P32691 INFO Train loss: 0.479917
2022-03-12 00:05:14,089 P32691 INFO ************ Epoch=5 end ************
2022-03-12 00:13:13,589 P32691 INFO [Metrics] logloss: 0.628449 - AUC: 0.823116
2022-03-12 00:13:13,590 P32691 INFO Monitor(max) STOP: 0.194668 !
2022-03-12 00:13:13,590 P32691 INFO Reduce learning rate on plateau: 0.000010
2022-03-12 00:13:13,590 P32691 INFO Early stopping at epoch=6
2022-03-12 00:13:13,590 P32691 INFO --- 1181/1181 batches finished ---
2022-03-12 00:13:13,623 P32691 INFO Train loss: 0.331642
2022-03-12 00:13:13,623 P32691 INFO Training finished.
2022-03-12 00:13:13,623 P32691 INFO Load best model: /cache/FuxiCTR/benchmarks/KKBox/CIN_kkbox_x1/kkbox_x1_227d337d/xDeepFM_kkbox_x1_009_117199f0_model.ckpt
2022-03-12 00:13:13,704 P32691 INFO ****** Validation evaluation ******
2022-03-12 00:13:18,733 P32691 INFO [Metrics] logloss: 0.490377 - AUC: 0.842943
2022-03-12 00:13:18,778 P32691 INFO ******** Test evaluation ********
2022-03-12 00:13:18,778 P32691 INFO Loading data...
2022-03-12 00:13:18,778 P32691 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-12 00:13:18,852 P32691 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-12 00:13:18,852 P32691 INFO Loading test data done.
2022-03-12 00:13:23,821 P32691 INFO [Metrics] logloss: 0.490763 - AUC: 0.842723

```
