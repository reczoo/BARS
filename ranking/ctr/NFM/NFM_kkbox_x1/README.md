## NFM_kkbox_x1

A hands-on guide to run the NFM model on the KKBox_x1 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [NFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/NFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [NFM_kkbox_x1_tuner_config_03](./NFM_kkbox_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd NFM_kkbox_x1
    nohup python run_expid.py --config ./NFM_kkbox_x1_tuner_config_03 --expid NFM_kkbox_x1_015_cbae7b1e --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.510179 | 0.828495  |


### Logs
```python
2022-03-08 08:56:29,065 P4250 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "6",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "NFM",
    "model_id": "NFM_kkbox_x1_015_cbae7b1e",
    "model_root": "./KKBox/NFM_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.3",
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
2022-03-08 08:56:29,066 P4250 INFO Set up feature encoder...
2022-03-08 08:56:29,066 P4250 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-08 08:56:30,890 P4250 INFO Total number of parameters: 14031864.
2022-03-08 08:56:30,891 P4250 INFO Loading data...
2022-03-08 08:56:30,891 P4250 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-08 08:56:31,350 P4250 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-08 08:56:31,699 P4250 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-08 08:56:31,716 P4250 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-08 08:56:31,716 P4250 INFO Loading train data done.
2022-03-08 08:56:37,927 P4250 INFO Start training: 591 batches/epoch
2022-03-08 08:56:37,927 P4250 INFO ************ Epoch=1 start ************
2022-03-08 08:58:45,789 P4250 INFO [Metrics] logloss: 0.527666 - AUC: 0.812201
2022-03-08 08:58:45,792 P4250 INFO Save best model: monitor(max): 0.284535
2022-03-08 08:58:46,355 P4250 INFO --- 591/591 batches finished ---
2022-03-08 08:58:46,392 P4250 INFO Train loss: 0.572116
2022-03-08 08:58:46,392 P4250 INFO ************ Epoch=1 end ************
2022-03-08 09:00:52,894 P4250 INFO [Metrics] logloss: 0.510706 - AUC: 0.825792
2022-03-08 09:00:52,895 P4250 INFO Save best model: monitor(max): 0.315085
2022-03-08 09:00:52,956 P4250 INFO --- 591/591 batches finished ---
2022-03-08 09:00:52,997 P4250 INFO Train loss: 0.520221
2022-03-08 09:00:52,997 P4250 INFO ************ Epoch=2 end ************
2022-03-08 09:02:58,938 P4250 INFO [Metrics] logloss: 0.510013 - AUC: 0.828591
2022-03-08 09:02:58,938 P4250 INFO Save best model: monitor(max): 0.318579
2022-03-08 09:02:59,015 P4250 INFO --- 591/591 batches finished ---
2022-03-08 09:02:59,056 P4250 INFO Train loss: 0.490057
2022-03-08 09:02:59,056 P4250 INFO ************ Epoch=3 end ************
2022-03-08 09:05:04,599 P4250 INFO [Metrics] logloss: 0.526734 - AUC: 0.824098
2022-03-08 09:05:04,600 P4250 INFO Monitor(max) STOP: 0.297363 !
2022-03-08 09:05:04,600 P4250 INFO Reduce learning rate on plateau: 0.000100
2022-03-08 09:05:04,600 P4250 INFO --- 591/591 batches finished ---
2022-03-08 09:05:04,642 P4250 INFO Train loss: 0.463107
2022-03-08 09:05:04,642 P4250 INFO ************ Epoch=4 end ************
2022-03-08 09:07:10,213 P4250 INFO [Metrics] logloss: 0.633140 - AUC: 0.816162
2022-03-08 09:07:10,214 P4250 INFO Monitor(max) STOP: 0.183022 !
2022-03-08 09:07:10,214 P4250 INFO Reduce learning rate on plateau: 0.000010
2022-03-08 09:07:10,214 P4250 INFO Early stopping at epoch=5
2022-03-08 09:07:10,214 P4250 INFO --- 591/591 batches finished ---
2022-03-08 09:07:10,252 P4250 INFO Train loss: 0.371106
2022-03-08 09:07:10,252 P4250 INFO Training finished.
2022-03-08 09:07:10,252 P4250 INFO Load best model: /cache/FuxiCTR/benchmarks/KKBox/NFM_kkbox_x1/kkbox_x1_227d337d/NFM_kkbox_x1_015_cbae7b1e_model.ckpt
2022-03-08 09:07:10,408 P4250 INFO ****** Validation evaluation ******
2022-03-08 09:07:15,387 P4250 INFO [Metrics] logloss: 0.510013 - AUC: 0.828591
2022-03-08 09:07:15,450 P4250 INFO ******** Test evaluation ********
2022-03-08 09:07:15,450 P4250 INFO Loading data...
2022-03-08 09:07:15,450 P4250 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-08 09:07:15,514 P4250 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-08 09:07:15,514 P4250 INFO Loading test data done.
2022-03-08 09:07:20,482 P4250 INFO [Metrics] logloss: 0.510179 - AUC: 0.828495

```
