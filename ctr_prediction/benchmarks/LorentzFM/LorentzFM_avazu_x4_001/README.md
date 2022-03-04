## LorentzFM_avazu_x4_001

A hands-on guide to run the LorentzFM model on the Avazu_x4_001 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
  GPU: Tesla P100 16G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 10.0
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [Avazu_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [LorentzFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/LorentzFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [LorentzFM_avazu_x4_tuner_config_01](./LorentzFM_avazu_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd LorentzFM_avazu_x4_001
    nohup python run_expid.py --config ./LorentzFM_avazu_x4_tuner_config_01 --expid LorentzFM_avazu_x4_004_d25a301d --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.375609 | 0.788511  |


### Logs
```python
2022-03-02 17:31:56,819 P4816 INFO {
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'str', 'name': 'id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "2",
    "model": "LorentzFM",
    "model_id": "LorentzFM_avazu_x4_004_d25a301d",
    "model_root": "./Avazu/LorentzFM_avazu_x4_001/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-06",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x4/test.csv",
    "train_data": "../data/Avazu/Avazu_x4/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x4/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2022-03-02 17:31:56,819 P4816 INFO Set up feature encoder...
2022-03-02 17:31:56,819 P4816 INFO Load feature_encoder from pickle: ../data/Avazu/avazu_x4_3bbbc4c9/feature_encoder.pkl
2022-03-02 17:31:59,688 P4816 INFO Total number of parameters: 60015600.
2022-03-02 17:31:59,688 P4816 INFO Loading data...
2022-03-02 17:31:59,691 P4816 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2022-03-02 17:32:02,484 P4816 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2022-03-02 17:32:03,922 P4816 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2022-03-02 17:32:04,096 P4816 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2022-03-02 17:32:04,096 P4816 INFO Loading train data done.
2022-03-02 17:32:06,807 P4816 INFO Start training: 3235 batches/epoch
2022-03-02 17:32:06,807 P4816 INFO ************ Epoch=1 start ************
2022-03-02 17:38:06,257 P4816 INFO [Metrics] logloss: 0.386922 - AUC: 0.768417
2022-03-02 17:38:06,258 P4816 INFO Save best model: monitor(max): 0.381495
2022-03-02 17:38:06,507 P4816 INFO --- 3235/3235 batches finished ---
2022-03-02 17:38:06,549 P4816 INFO Train loss: 0.397970
2022-03-02 17:38:06,549 P4816 INFO ************ Epoch=1 end ************
2022-03-02 17:44:03,991 P4816 INFO [Metrics] logloss: 0.382647 - AUC: 0.775982
2022-03-02 17:44:03,995 P4816 INFO Save best model: monitor(max): 0.393335
2022-03-02 17:44:04,426 P4816 INFO --- 3235/3235 batches finished ---
2022-03-02 17:44:04,476 P4816 INFO Train loss: 0.390458
2022-03-02 17:44:04,477 P4816 INFO ************ Epoch=2 end ************
2022-03-02 17:50:01,233 P4816 INFO [Metrics] logloss: 0.379944 - AUC: 0.780303
2022-03-02 17:50:01,237 P4816 INFO Save best model: monitor(max): 0.400359
2022-03-02 17:50:01,714 P4816 INFO --- 3235/3235 batches finished ---
2022-03-02 17:50:01,759 P4816 INFO Train loss: 0.386698
2022-03-02 17:50:01,759 P4816 INFO ************ Epoch=3 end ************
2022-03-02 17:56:01,403 P4816 INFO [Metrics] logloss: 0.378817 - AUC: 0.782122
2022-03-02 17:56:01,407 P4816 INFO Save best model: monitor(max): 0.403304
2022-03-02 17:56:01,839 P4816 INFO --- 3235/3235 batches finished ---
2022-03-02 17:56:01,887 P4816 INFO Train loss: 0.383257
2022-03-02 17:56:01,887 P4816 INFO ************ Epoch=4 end ************
2022-03-02 18:01:59,351 P4816 INFO [Metrics] logloss: 0.378211 - AUC: 0.783431
2022-03-02 18:01:59,355 P4816 INFO Save best model: monitor(max): 0.405220
2022-03-02 18:01:59,797 P4816 INFO --- 3235/3235 batches finished ---
2022-03-02 18:01:59,848 P4816 INFO Train loss: 0.380321
2022-03-02 18:01:59,848 P4816 INFO ************ Epoch=5 end ************
2022-03-02 18:07:57,049 P4816 INFO [Metrics] logloss: 0.377871 - AUC: 0.784361
2022-03-02 18:07:57,053 P4816 INFO Save best model: monitor(max): 0.406490
2022-03-02 18:07:57,335 P4816 INFO --- 3235/3235 batches finished ---
2022-03-02 18:07:57,381 P4816 INFO Train loss: 0.377947
2022-03-02 18:07:57,381 P4816 INFO ************ Epoch=6 end ************
2022-03-02 18:13:54,853 P4816 INFO [Metrics] logloss: 0.377783 - AUC: 0.784544
2022-03-02 18:13:54,857 P4816 INFO Save best model: monitor(max): 0.406760
2022-03-02 18:13:55,314 P4816 INFO --- 3235/3235 batches finished ---
2022-03-02 18:13:55,362 P4816 INFO Train loss: 0.375920
2022-03-02 18:13:55,363 P4816 INFO ************ Epoch=7 end ************
2022-03-02 18:19:52,399 P4816 INFO [Metrics] logloss: 0.378111 - AUC: 0.785101
2022-03-02 18:19:52,405 P4816 INFO Save best model: monitor(max): 0.406989
2022-03-02 18:19:52,852 P4816 INFO --- 3235/3235 batches finished ---
2022-03-02 18:19:52,901 P4816 INFO Train loss: 0.374079
2022-03-02 18:19:52,901 P4816 INFO ************ Epoch=8 end ************
2022-03-02 18:25:49,797 P4816 INFO [Metrics] logloss: 0.378014 - AUC: 0.784594
2022-03-02 18:25:49,801 P4816 INFO Monitor(max) STOP: 0.406580 !
2022-03-02 18:25:49,801 P4816 INFO Reduce learning rate on plateau: 0.000100
2022-03-02 18:25:49,801 P4816 INFO --- 3235/3235 batches finished ---
2022-03-02 18:25:49,851 P4816 INFO Train loss: 0.372440
2022-03-02 18:25:49,851 P4816 INFO ************ Epoch=9 end ************
2022-03-02 18:31:47,249 P4816 INFO [Metrics] logloss: 0.375899 - AUC: 0.787891
2022-03-02 18:31:47,252 P4816 INFO Save best model: monitor(max): 0.411992
2022-03-02 18:31:47,721 P4816 INFO --- 3235/3235 batches finished ---
2022-03-02 18:31:47,767 P4816 INFO Train loss: 0.354220
2022-03-02 18:31:47,767 P4816 INFO ************ Epoch=10 end ************
2022-03-02 18:37:44,942 P4816 INFO [Metrics] logloss: 0.375728 - AUC: 0.788236
2022-03-02 18:37:44,946 P4816 INFO Save best model: monitor(max): 0.412509
2022-03-02 18:37:45,410 P4816 INFO --- 3235/3235 batches finished ---
2022-03-02 18:37:45,459 P4816 INFO Train loss: 0.352303
2022-03-02 18:37:45,459 P4816 INFO ************ Epoch=11 end ************
2022-03-02 18:43:43,042 P4816 INFO [Metrics] logloss: 0.375660 - AUC: 0.788385
2022-03-02 18:43:43,045 P4816 INFO Save best model: monitor(max): 0.412725
2022-03-02 18:43:43,518 P4816 INFO --- 3235/3235 batches finished ---
2022-03-02 18:43:43,566 P4816 INFO Train loss: 0.351316
2022-03-02 18:43:43,566 P4816 INFO ************ Epoch=12 end ************
2022-03-02 18:49:40,958 P4816 INFO [Metrics] logloss: 0.375721 - AUC: 0.788268
2022-03-02 18:49:40,963 P4816 INFO Monitor(max) STOP: 0.412547 !
2022-03-02 18:49:40,963 P4816 INFO Reduce learning rate on plateau: 0.000010
2022-03-02 18:49:40,963 P4816 INFO --- 3235/3235 batches finished ---
2022-03-02 18:49:41,015 P4816 INFO Train loss: 0.350627
2022-03-02 18:49:41,015 P4816 INFO ************ Epoch=13 end ************
2022-03-02 18:55:38,769 P4816 INFO [Metrics] logloss: 0.375607 - AUC: 0.788507
2022-03-02 18:55:38,775 P4816 INFO Save best model: monitor(max): 0.412900
2022-03-02 18:55:39,215 P4816 INFO --- 3235/3235 batches finished ---
2022-03-02 18:55:39,264 P4816 INFO Train loss: 0.347522
2022-03-02 18:55:39,264 P4816 INFO ************ Epoch=14 end ************
2022-03-02 19:01:34,718 P4816 INFO [Metrics] logloss: 0.375612 - AUC: 0.788506
2022-03-02 19:01:34,723 P4816 INFO Monitor(max) STOP: 0.412895 !
2022-03-02 19:01:34,723 P4816 INFO Reduce learning rate on plateau: 0.000001
2022-03-02 19:01:34,723 P4816 INFO --- 3235/3235 batches finished ---
2022-03-02 19:01:34,766 P4816 INFO Train loss: 0.347430
2022-03-02 19:01:34,766 P4816 INFO ************ Epoch=15 end ************
2022-03-02 19:07:31,883 P4816 INFO [Metrics] logloss: 0.375612 - AUC: 0.788500
2022-03-02 19:07:31,886 P4816 INFO Monitor(max) STOP: 0.412887 !
2022-03-02 19:07:31,887 P4816 INFO Reduce learning rate on plateau: 0.000001
2022-03-02 19:07:31,887 P4816 INFO Early stopping at epoch=16
2022-03-02 19:07:31,887 P4816 INFO --- 3235/3235 batches finished ---
2022-03-02 19:07:31,936 P4816 INFO Train loss: 0.347084
2022-03-02 19:07:31,936 P4816 INFO Training finished.
2022-03-02 19:07:31,936 P4816 INFO Load best model: /home/zhujieming/zhujieming/FuxiCTR_v1.0/benchmarks/Avazu/LorentzFM_avazu_x4_001/avazu_x4_3bbbc4c9/LorentzFM_avazu_x4_004_d25a301d_model.ckpt
2022-03-02 19:07:32,297 P4816 INFO ****** Validation evaluation ******
2022-03-02 19:07:55,845 P4816 INFO [Metrics] logloss: 0.375607 - AUC: 0.788507
2022-03-02 19:07:55,903 P4816 INFO ******** Test evaluation ********
2022-03-02 19:07:55,903 P4816 INFO Loading data...
2022-03-02 19:07:55,904 P4816 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2022-03-02 19:07:56,375 P4816 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2022-03-02 19:07:56,376 P4816 INFO Loading test data done.
2022-03-02 19:08:19,885 P4816 INFO [Metrics] logloss: 0.375609 - AUC: 0.788511

```
