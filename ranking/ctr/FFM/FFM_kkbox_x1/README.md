## FFM_kkbox_x1

A hands-on guide to run the FFM model on the KKBox_x1 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FFM](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FFM_kkbox_x1_tuner_config_03](./FFM_kkbox_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FFM_kkbox_x1
    nohup python run_expid.py --config ./FFM_kkbox_x1_tuner_config_03 --expid FFM_kkbox_x1_017_852fd9b3 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.497407 | 0.837603  |


### Logs
```python
2022-03-10 10:55:32,917 P56107 INFO {
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "embedding_dim": "64",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "7",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "FFM",
    "model_id": "FFM_kkbox_x1_017_852fd9b3",
    "model_root": "./KKBox/FFM_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "5e-06",
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
2022-03-10 10:55:32,918 P56107 INFO Set up feature encoder...
2022-03-10 10:55:32,918 P56107 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-10 10:55:34,401 P56107 INFO Total number of parameters: 70937944.
2022-03-10 10:55:34,401 P56107 INFO Loading data...
2022-03-10 10:55:34,402 P56107 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-10 10:55:34,807 P56107 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-10 10:55:34,993 P56107 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-10 10:55:35,009 P56107 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-10 10:55:35,009 P56107 INFO Loading train data done.
2022-03-10 10:55:39,862 P56107 INFO Start training: 591 batches/epoch
2022-03-10 10:55:39,863 P56107 INFO ************ Epoch=1 start ************
2022-03-10 11:03:21,264 P56107 INFO [Metrics] logloss: 0.527291 - AUC: 0.812617
2022-03-10 11:03:21,267 P56107 INFO Save best model: monitor(max): 0.285326
2022-03-10 11:03:21,769 P56107 INFO --- 591/591 batches finished ---
2022-03-10 11:03:21,805 P56107 INFO Train loss: 0.572839
2022-03-10 11:03:21,805 P56107 INFO ************ Epoch=1 end ************
2022-03-10 11:11:02,094 P56107 INFO [Metrics] logloss: 0.513248 - AUC: 0.824255
2022-03-10 11:11:02,096 P56107 INFO Save best model: monitor(max): 0.311006
2022-03-10 11:11:02,455 P56107 INFO --- 591/591 batches finished ---
2022-03-10 11:11:02,493 P56107 INFO Train loss: 0.539057
2022-03-10 11:11:02,493 P56107 INFO ************ Epoch=2 end ************
2022-03-10 11:18:41,756 P56107 INFO [Metrics] logloss: 0.507890 - AUC: 0.828587
2022-03-10 11:18:41,760 P56107 INFO Save best model: monitor(max): 0.320697
2022-03-10 11:18:42,136 P56107 INFO --- 591/591 batches finished ---
2022-03-10 11:18:42,173 P56107 INFO Train loss: 0.526530
2022-03-10 11:18:42,173 P56107 INFO ************ Epoch=3 end ************
2022-03-10 11:26:21,959 P56107 INFO [Metrics] logloss: 0.504579 - AUC: 0.831302
2022-03-10 11:26:21,963 P56107 INFO Save best model: monitor(max): 0.326723
2022-03-10 11:26:22,327 P56107 INFO --- 591/591 batches finished ---
2022-03-10 11:26:22,365 P56107 INFO Train loss: 0.518339
2022-03-10 11:26:22,365 P56107 INFO ************ Epoch=4 end ************
2022-03-10 11:34:03,208 P56107 INFO [Metrics] logloss: 0.502917 - AUC: 0.832817
2022-03-10 11:34:03,212 P56107 INFO Save best model: monitor(max): 0.329900
2022-03-10 11:34:03,609 P56107 INFO --- 591/591 batches finished ---
2022-03-10 11:34:03,646 P56107 INFO Train loss: 0.512083
2022-03-10 11:34:03,646 P56107 INFO ************ Epoch=5 end ************
2022-03-10 11:41:43,430 P56107 INFO [Metrics] logloss: 0.502515 - AUC: 0.833272
2022-03-10 11:41:43,434 P56107 INFO Save best model: monitor(max): 0.330757
2022-03-10 11:41:43,788 P56107 INFO --- 591/591 batches finished ---
2022-03-10 11:41:43,827 P56107 INFO Train loss: 0.506450
2022-03-10 11:41:43,827 P56107 INFO ************ Epoch=6 end ************
2022-03-10 11:49:23,622 P56107 INFO [Metrics] logloss: 0.502826 - AUC: 0.833315
2022-03-10 11:49:23,626 P56107 INFO Monitor(max) STOP: 0.330490 !
2022-03-10 11:49:23,626 P56107 INFO Reduce learning rate on plateau: 0.000100
2022-03-10 11:49:23,626 P56107 INFO --- 591/591 batches finished ---
2022-03-10 11:49:23,664 P56107 INFO Train loss: 0.501200
2022-03-10 11:49:23,664 P56107 INFO ************ Epoch=7 end ************
2022-03-10 11:57:02,786 P56107 INFO [Metrics] logloss: 0.498454 - AUC: 0.836654
2022-03-10 11:57:02,790 P56107 INFO Save best model: monitor(max): 0.338200
2022-03-10 11:57:03,108 P56107 INFO --- 591/591 batches finished ---
2022-03-10 11:57:03,154 P56107 INFO Train loss: 0.466921
2022-03-10 11:57:03,154 P56107 INFO ************ Epoch=8 end ************
2022-03-10 12:04:43,066 P56107 INFO [Metrics] logloss: 0.497798 - AUC: 0.837343
2022-03-10 12:04:43,070 P56107 INFO Save best model: monitor(max): 0.339545
2022-03-10 12:04:43,445 P56107 INFO --- 591/591 batches finished ---
2022-03-10 12:04:43,484 P56107 INFO Train loss: 0.461121
2022-03-10 12:04:43,485 P56107 INFO ************ Epoch=9 end ************
2022-03-10 12:11:36,447 P56107 INFO [Metrics] logloss: 0.497898 - AUC: 0.837380
2022-03-10 12:11:36,451 P56107 INFO Monitor(max) STOP: 0.339481 !
2022-03-10 12:11:36,451 P56107 INFO Reduce learning rate on plateau: 0.000010
2022-03-10 12:11:36,451 P56107 INFO --- 591/591 batches finished ---
2022-03-10 12:11:36,489 P56107 INFO Train loss: 0.457904
2022-03-10 12:11:36,489 P56107 INFO ************ Epoch=10 end ************
2022-03-10 12:15:40,770 P56107 INFO [Metrics] logloss: 0.497906 - AUC: 0.837425
2022-03-10 12:15:40,773 P56107 INFO Monitor(max) STOP: 0.339519 !
2022-03-10 12:15:40,773 P56107 INFO Reduce learning rate on plateau: 0.000001
2022-03-10 12:15:40,774 P56107 INFO Early stopping at epoch=11
2022-03-10 12:15:40,774 P56107 INFO --- 591/591 batches finished ---
2022-03-10 12:15:40,813 P56107 INFO Train loss: 0.451714
2022-03-10 12:15:40,814 P56107 INFO Training finished.
2022-03-10 12:15:40,814 P56107 INFO Load best model: /cache/FuxiCTR/benchmarks/KKBox/FFM_kkbox_x1/kkbox_x1_227d337d/FFM_kkbox_x1_017_852fd9b3_model.ckpt
2022-03-10 12:15:41,166 P56107 INFO ****** Validation evaluation ******
2022-03-10 12:15:45,394 P56107 INFO [Metrics] logloss: 0.497798 - AUC: 0.837343
2022-03-10 12:15:45,457 P56107 INFO ******** Test evaluation ********
2022-03-10 12:15:45,458 P56107 INFO Loading data...
2022-03-10 12:15:45,459 P56107 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-10 12:15:45,530 P56107 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-10 12:15:45,530 P56107 INFO Loading test data done.
2022-03-10 12:15:49,604 P56107 INFO [Metrics] logloss: 0.497407 - AUC: 0.837603

```
