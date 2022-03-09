## FwFM_kkbox_x1

A hands-on guide to run the FwFM model on the KKBox_x1 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FwFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FwFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FwFM_kkbox_x1_tuner_config_01](./FwFM_kkbox_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FwFM_kkbox_x1
    nohup python run_expid.py --config ./FwFM_kkbox_x1_tuner_config_01 --expid FwFM_kkbox_x1_006_d4ec0630 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.497109 | 0.840632  |


### Logs
```python
2022-03-09 20:49:39,119 P56847 INFO {
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "embedding_dim": "128",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "5",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "linear_type": "FiLV",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "FwFM",
    "model_id": "FwFM_kkbox_x1_006_d4ec0630",
    "model_root": "./KKBox/FwFM_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-05",
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
2022-03-09 20:49:39,119 P56847 INFO Set up feature encoder...
2022-03-09 20:49:39,120 P56847 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-09 20:49:40,259 P56847 INFO Total number of parameters: 11809359.
2022-03-09 20:49:40,260 P56847 INFO Loading data...
2022-03-09 20:49:40,261 P56847 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-09 20:49:40,642 P56847 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-09 20:49:40,949 P56847 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-09 20:49:40,976 P56847 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-09 20:49:40,976 P56847 INFO Loading train data done.
2022-03-09 20:49:45,681 P56847 INFO Start training: 591 batches/epoch
2022-03-09 20:49:45,681 P56847 INFO ************ Epoch=1 start ************
2022-03-09 20:50:46,016 P56847 INFO [Metrics] logloss: 0.558657 - AUC: 0.783171
2022-03-09 20:50:46,020 P56847 INFO Save best model: monitor(max): 0.224514
2022-03-09 20:50:46,238 P56847 INFO --- 591/591 batches finished ---
2022-03-09 20:50:46,275 P56847 INFO Train loss: 0.592738
2022-03-09 20:50:46,276 P56847 INFO ************ Epoch=1 end ************
2022-03-09 20:51:42,214 P56847 INFO [Metrics] logloss: 0.532287 - AUC: 0.807866
2022-03-09 20:51:42,216 P56847 INFO Save best model: monitor(max): 0.275580
2022-03-09 20:51:42,277 P56847 INFO --- 591/591 batches finished ---
2022-03-09 20:51:42,309 P56847 INFO Train loss: 0.559196
2022-03-09 20:51:42,309 P56847 INFO ************ Epoch=2 end ************
2022-03-09 20:52:44,404 P56847 INFO [Metrics] logloss: 0.516569 - AUC: 0.821352
2022-03-09 20:52:44,408 P56847 INFO Save best model: monitor(max): 0.304783
2022-03-09 20:52:44,469 P56847 INFO --- 591/591 batches finished ---
2022-03-09 20:52:44,502 P56847 INFO Train loss: 0.538120
2022-03-09 20:52:44,503 P56847 INFO ************ Epoch=3 end ************
2022-03-09 20:53:45,141 P56847 INFO [Metrics] logloss: 0.507130 - AUC: 0.828976
2022-03-09 20:53:45,143 P56847 INFO Save best model: monitor(max): 0.321846
2022-03-09 20:53:45,205 P56847 INFO --- 591/591 batches finished ---
2022-03-09 20:53:45,239 P56847 INFO Train loss: 0.524122
2022-03-09 20:53:45,240 P56847 INFO ************ Epoch=4 end ************
2022-03-09 20:54:41,936 P56847 INFO [Metrics] logloss: 0.500812 - AUC: 0.834069
2022-03-09 20:54:41,939 P56847 INFO Save best model: monitor(max): 0.333257
2022-03-09 20:54:42,001 P56847 INFO --- 591/591 batches finished ---
2022-03-09 20:54:42,042 P56847 INFO Train loss: 0.512322
2022-03-09 20:54:42,042 P56847 INFO ************ Epoch=5 end ************
2022-03-09 20:55:43,042 P56847 INFO [Metrics] logloss: 0.497144 - AUC: 0.837228
2022-03-09 20:55:43,045 P56847 INFO Save best model: monitor(max): 0.340084
2022-03-09 20:55:43,099 P56847 INFO --- 591/591 batches finished ---
2022-03-09 20:55:43,135 P56847 INFO Train loss: 0.500676
2022-03-09 20:55:43,135 P56847 INFO ************ Epoch=6 end ************
2022-03-09 20:56:44,013 P56847 INFO [Metrics] logloss: 0.495856 - AUC: 0.839178
2022-03-09 20:56:44,017 P56847 INFO Save best model: monitor(max): 0.343322
2022-03-09 20:56:44,079 P56847 INFO --- 591/591 batches finished ---
2022-03-09 20:56:44,121 P56847 INFO Train loss: 0.488777
2022-03-09 20:56:44,122 P56847 INFO ************ Epoch=7 end ************
2022-03-09 20:57:41,290 P56847 INFO [Metrics] logloss: 0.497866 - AUC: 0.838922
2022-03-09 20:57:41,292 P56847 INFO Monitor(max) STOP: 0.341057 !
2022-03-09 20:57:41,293 P56847 INFO Reduce learning rate on plateau: 0.000100
2022-03-09 20:57:41,293 P56847 INFO --- 591/591 batches finished ---
2022-03-09 20:57:41,329 P56847 INFO Train loss: 0.476389
2022-03-09 20:57:41,329 P56847 INFO ************ Epoch=8 end ************
2022-03-09 20:58:41,992 P56847 INFO [Metrics] logloss: 0.496640 - AUC: 0.840880
2022-03-09 20:58:41,996 P56847 INFO Save best model: monitor(max): 0.344241
2022-03-09 20:58:42,057 P56847 INFO --- 591/591 batches finished ---
2022-03-09 20:58:42,099 P56847 INFO Train loss: 0.426631
2022-03-09 20:58:42,100 P56847 INFO ************ Epoch=9 end ************
2022-03-09 20:59:44,572 P56847 INFO [Metrics] logloss: 0.497829 - AUC: 0.841053
2022-03-09 20:59:44,576 P56847 INFO Monitor(max) STOP: 0.343225 !
2022-03-09 20:59:44,576 P56847 INFO Reduce learning rate on plateau: 0.000010
2022-03-09 20:59:44,576 P56847 INFO --- 591/591 batches finished ---
2022-03-09 20:59:44,619 P56847 INFO Train loss: 0.418938
2022-03-09 20:59:44,619 P56847 INFO ************ Epoch=10 end ************
2022-03-09 21:00:42,309 P56847 INFO [Metrics] logloss: 0.498002 - AUC: 0.841006
2022-03-09 21:00:42,313 P56847 INFO Monitor(max) STOP: 0.343004 !
2022-03-09 21:00:42,313 P56847 INFO Reduce learning rate on plateau: 0.000001
2022-03-09 21:00:42,313 P56847 INFO Early stopping at epoch=11
2022-03-09 21:00:42,314 P56847 INFO --- 591/591 batches finished ---
2022-03-09 21:00:42,355 P56847 INFO Train loss: 0.410363
2022-03-09 21:00:42,355 P56847 INFO Training finished.
2022-03-09 21:00:42,356 P56847 INFO Load best model: /cache/FuxiCTR/benchmarks/KKBox/FwFM_kkbox_x1/kkbox_x1_227d337d/FwFM_kkbox_x1_006_d4ec0630_model.ckpt
2022-03-09 21:00:42,462 P56847 INFO ****** Validation evaluation ******
2022-03-09 21:00:46,694 P56847 INFO [Metrics] logloss: 0.496640 - AUC: 0.840880
2022-03-09 21:00:46,750 P56847 INFO ******** Test evaluation ********
2022-03-09 21:00:46,751 P56847 INFO Loading data...
2022-03-09 21:00:46,751 P56847 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-09 21:00:46,830 P56847 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-09 21:00:46,830 P56847 INFO Loading test data done.
2022-03-09 21:00:51,046 P56847 INFO [Metrics] logloss: 0.497109 - AUC: 0.840632

```
