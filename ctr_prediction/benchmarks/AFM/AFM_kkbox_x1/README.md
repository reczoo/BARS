## AFM_kkbox_x1

A hands-on guide to run the AFM model on the KKBox_x1 dataset.

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
Dataset ID: [KKBox_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/KKBox/README.md#KKBox_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [AFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/AFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFM_kkbox_x1_tuner_config_01](./AFM_kkbox_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFM_kkbox_x1
    nohup python run_expid.py --config ./AFM_kkbox_x1_tuner_config_01 --expid AFM_kkbox_x1_007_7d582ff2 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.524060 | 0.817460  |


### Logs
```python
2022-02-25 19:26:31,340 P7293 INFO {
    "attention_dim": "64",
    "attention_dropout": "[0, 0]",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_bf23df6c",
    "debug": "False",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.0001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "AFM",
    "model_id": "AFM_kkbox_x1_007_7d582ff2",
    "model_root": "./KKBox/AFM_kkbox/",
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
    "use_attention": "True",
    "use_hdf5": "True",
    "valid_data": "../data/KKBox/KKBox_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2022-02-25 19:26:31,341 P7293 INFO Set up feature encoder...
2022-02-25 19:26:31,341 P7293 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_bf23df6c/feature_encoder.pkl
2022-02-25 19:26:31,651 P7293 INFO Total number of parameters: 11908312.
2022-02-25 19:26:31,652 P7293 INFO Loading data...
2022-02-25 19:26:31,654 P7293 INFO Loading data from h5: ../data/KKBox/kkbox_x1_bf23df6c/train.h5
2022-02-25 19:26:32,019 P7293 INFO Loading data from h5: ../data/KKBox/kkbox_x1_bf23df6c/valid.h5
2022-02-25 19:26:32,221 P7293 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-02-25 19:26:32,240 P7293 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-02-25 19:26:32,240 P7293 INFO Loading train data done.
2022-02-25 19:26:34,976 P7293 INFO Start training: 591 batches/epoch
2022-02-25 19:26:34,976 P7293 INFO ************ Epoch=1 start ************
2022-02-25 19:27:50,142 P7293 INFO [Metrics] logloss: 0.629185 - AUC: 0.704059
2022-02-25 19:27:50,146 P7293 INFO Save best model: monitor(max): 0.074874
2022-02-25 19:27:50,188 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:27:50,230 P7293 INFO Train loss: 0.645415
2022-02-25 19:27:50,230 P7293 INFO ************ Epoch=1 end ************
2022-02-25 19:29:05,278 P7293 INFO [Metrics] logloss: 0.625116 - AUC: 0.709595
2022-02-25 19:29:05,282 P7293 INFO Save best model: monitor(max): 0.084479
2022-02-25 19:29:05,364 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:29:05,405 P7293 INFO Train loss: 0.633991
2022-02-25 19:29:05,405 P7293 INFO ************ Epoch=2 end ************
2022-02-25 19:30:20,583 P7293 INFO [Metrics] logloss: 0.602115 - AUC: 0.738924
2022-02-25 19:30:20,586 P7293 INFO Save best model: monitor(max): 0.136809
2022-02-25 19:30:20,664 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:30:20,706 P7293 INFO Train loss: 0.628852
2022-02-25 19:30:20,706 P7293 INFO ************ Epoch=3 end ************
2022-02-25 19:31:35,443 P7293 INFO [Metrics] logloss: 0.584775 - AUC: 0.758032
2022-02-25 19:31:35,446 P7293 INFO Save best model: monitor(max): 0.173257
2022-02-25 19:31:35,509 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:31:35,550 P7293 INFO Train loss: 0.610220
2022-02-25 19:31:35,550 P7293 INFO ************ Epoch=4 end ************
2022-02-25 19:32:52,045 P7293 INFO [Metrics] logloss: 0.572235 - AUC: 0.770913
2022-02-25 19:32:52,048 P7293 INFO Save best model: monitor(max): 0.198678
2022-02-25 19:32:52,110 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:32:52,151 P7293 INFO Train loss: 0.595460
2022-02-25 19:32:52,151 P7293 INFO ************ Epoch=5 end ************
2022-02-25 19:34:06,772 P7293 INFO [Metrics] logloss: 0.563425 - AUC: 0.779682
2022-02-25 19:34:06,775 P7293 INFO Save best model: monitor(max): 0.216257
2022-02-25 19:34:06,837 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:34:06,878 P7293 INFO Train loss: 0.585032
2022-02-25 19:34:06,878 P7293 INFO ************ Epoch=6 end ************
2022-02-25 19:35:21,439 P7293 INFO [Metrics] logloss: 0.555046 - AUC: 0.788002
2022-02-25 19:35:21,442 P7293 INFO Save best model: monitor(max): 0.232957
2022-02-25 19:35:21,505 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:35:21,547 P7293 INFO Train loss: 0.577565
2022-02-25 19:35:21,547 P7293 INFO ************ Epoch=7 end ************
2022-02-25 19:36:36,039 P7293 INFO [Metrics] logloss: 0.548488 - AUC: 0.793627
2022-02-25 19:36:36,042 P7293 INFO Save best model: monitor(max): 0.245139
2022-02-25 19:36:36,104 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:36:36,145 P7293 INFO Train loss: 0.570342
2022-02-25 19:36:36,145 P7293 INFO ************ Epoch=8 end ************
2022-02-25 19:37:50,599 P7293 INFO [Metrics] logloss: 0.543357 - AUC: 0.798315
2022-02-25 19:37:50,602 P7293 INFO Save best model: monitor(max): 0.254958
2022-02-25 19:37:50,670 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:37:50,711 P7293 INFO Train loss: 0.564392
2022-02-25 19:37:50,711 P7293 INFO ************ Epoch=9 end ************
2022-02-25 19:39:05,880 P7293 INFO [Metrics] logloss: 0.540450 - AUC: 0.800884
2022-02-25 19:39:05,883 P7293 INFO Save best model: monitor(max): 0.260435
2022-02-25 19:39:05,948 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:39:05,990 P7293 INFO Train loss: 0.559929
2022-02-25 19:39:05,990 P7293 INFO ************ Epoch=10 end ************
2022-02-25 19:40:21,109 P7293 INFO [Metrics] logloss: 0.538367 - AUC: 0.802900
2022-02-25 19:40:21,112 P7293 INFO Save best model: monitor(max): 0.264534
2022-02-25 19:40:21,179 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:40:21,221 P7293 INFO Train loss: 0.556417
2022-02-25 19:40:21,221 P7293 INFO ************ Epoch=11 end ************
2022-02-25 19:41:36,273 P7293 INFO [Metrics] logloss: 0.536494 - AUC: 0.804407
2022-02-25 19:41:36,276 P7293 INFO Save best model: monitor(max): 0.267913
2022-02-25 19:41:36,338 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:41:36,379 P7293 INFO Train loss: 0.553729
2022-02-25 19:41:36,379 P7293 INFO ************ Epoch=12 end ************
2022-02-25 19:42:51,196 P7293 INFO [Metrics] logloss: 0.535114 - AUC: 0.805606
2022-02-25 19:42:51,199 P7293 INFO Save best model: monitor(max): 0.270492
2022-02-25 19:42:51,261 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:42:51,303 P7293 INFO Train loss: 0.551592
2022-02-25 19:42:51,303 P7293 INFO ************ Epoch=13 end ************
2022-02-25 19:44:06,078 P7293 INFO [Metrics] logloss: 0.534870 - AUC: 0.805722
2022-02-25 19:44:06,081 P7293 INFO Save best model: monitor(max): 0.270852
2022-02-25 19:44:06,143 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:44:06,184 P7293 INFO Train loss: 0.549748
2022-02-25 19:44:06,184 P7293 INFO ************ Epoch=14 end ************
2022-02-25 19:45:20,948 P7293 INFO [Metrics] logloss: 0.533500 - AUC: 0.807145
2022-02-25 19:45:20,951 P7293 INFO Save best model: monitor(max): 0.273645
2022-02-25 19:45:21,022 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:45:21,064 P7293 INFO Train loss: 0.548230
2022-02-25 19:45:21,064 P7293 INFO ************ Epoch=15 end ************
2022-02-25 19:46:35,785 P7293 INFO [Metrics] logloss: 0.533156 - AUC: 0.807503
2022-02-25 19:46:35,788 P7293 INFO Save best model: monitor(max): 0.274347
2022-02-25 19:46:35,850 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:46:35,891 P7293 INFO Train loss: 0.547007
2022-02-25 19:46:35,891 P7293 INFO ************ Epoch=16 end ************
2022-02-25 19:47:50,791 P7293 INFO [Metrics] logloss: 0.532397 - AUC: 0.807971
2022-02-25 19:47:50,794 P7293 INFO Save best model: monitor(max): 0.275574
2022-02-25 19:47:50,857 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:47:50,899 P7293 INFO Train loss: 0.545792
2022-02-25 19:47:50,899 P7293 INFO ************ Epoch=17 end ************
2022-02-25 19:49:05,763 P7293 INFO [Metrics] logloss: 0.532422 - AUC: 0.808342
2022-02-25 19:49:05,765 P7293 INFO Save best model: monitor(max): 0.275919
2022-02-25 19:49:05,831 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:49:05,872 P7293 INFO Train loss: 0.544732
2022-02-25 19:49:05,873 P7293 INFO ************ Epoch=18 end ************
2022-02-25 19:50:20,625 P7293 INFO [Metrics] logloss: 0.531458 - AUC: 0.808956
2022-02-25 19:50:20,628 P7293 INFO Save best model: monitor(max): 0.277497
2022-02-25 19:50:20,693 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:50:20,734 P7293 INFO Train loss: 0.543747
2022-02-25 19:50:20,734 P7293 INFO ************ Epoch=19 end ************
2022-02-25 19:51:35,620 P7293 INFO [Metrics] logloss: 0.531511 - AUC: 0.809021
2022-02-25 19:51:35,623 P7293 INFO Save best model: monitor(max): 0.277510
2022-02-25 19:51:35,687 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:51:35,729 P7293 INFO Train loss: 0.543077
2022-02-25 19:51:35,729 P7293 INFO ************ Epoch=20 end ************
2022-02-25 19:52:50,705 P7293 INFO [Metrics] logloss: 0.531127 - AUC: 0.809340
2022-02-25 19:52:50,708 P7293 INFO Save best model: monitor(max): 0.278212
2022-02-25 19:52:50,777 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:52:50,820 P7293 INFO Train loss: 0.542136
2022-02-25 19:52:50,821 P7293 INFO ************ Epoch=21 end ************
2022-02-25 19:54:05,599 P7293 INFO [Metrics] logloss: 0.530742 - AUC: 0.809650
2022-02-25 19:54:05,602 P7293 INFO Save best model: monitor(max): 0.278908
2022-02-25 19:54:05,664 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:54:05,705 P7293 INFO Train loss: 0.541658
2022-02-25 19:54:05,705 P7293 INFO ************ Epoch=22 end ************
2022-02-25 19:55:20,589 P7293 INFO [Metrics] logloss: 0.530266 - AUC: 0.809955
2022-02-25 19:55:20,592 P7293 INFO Save best model: monitor(max): 0.279689
2022-02-25 19:55:20,656 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:55:20,697 P7293 INFO Train loss: 0.540791
2022-02-25 19:55:20,697 P7293 INFO ************ Epoch=23 end ************
2022-02-25 19:56:35,776 P7293 INFO [Metrics] logloss: 0.530091 - AUC: 0.810158
2022-02-25 19:56:35,779 P7293 INFO Save best model: monitor(max): 0.280067
2022-02-25 19:56:35,851 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:56:35,893 P7293 INFO Train loss: 0.539966
2022-02-25 19:56:35,893 P7293 INFO ************ Epoch=24 end ************
2022-02-25 19:57:50,670 P7293 INFO [Metrics] logloss: 0.529447 - AUC: 0.810983
2022-02-25 19:57:50,673 P7293 INFO Save best model: monitor(max): 0.281536
2022-02-25 19:57:50,748 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:57:50,790 P7293 INFO Train loss: 0.539240
2022-02-25 19:57:50,790 P7293 INFO ************ Epoch=25 end ************
2022-02-25 19:59:05,507 P7293 INFO [Metrics] logloss: 0.529455 - AUC: 0.810902
2022-02-25 19:59:05,509 P7293 INFO Monitor(max) STOP: 0.281448 !
2022-02-25 19:59:05,510 P7293 INFO Reduce learning rate on plateau: 0.000100
2022-02-25 19:59:05,510 P7293 INFO --- 591/591 batches finished ---
2022-02-25 19:59:05,550 P7293 INFO Train loss: 0.538528
2022-02-25 19:59:05,550 P7293 INFO ************ Epoch=26 end ************
2022-02-25 20:00:20,406 P7293 INFO [Metrics] logloss: 0.525675 - AUC: 0.815293
2022-02-25 20:00:20,410 P7293 INFO Save best model: monitor(max): 0.289618
2022-02-25 20:00:20,474 P7293 INFO --- 591/591 batches finished ---
2022-02-25 20:00:20,516 P7293 INFO Train loss: 0.513916
2022-02-25 20:00:20,516 P7293 INFO ************ Epoch=27 end ************
2022-02-25 20:01:36,327 P7293 INFO [Metrics] logloss: 0.524728 - AUC: 0.816577
2022-02-25 20:01:36,330 P7293 INFO Save best model: monitor(max): 0.291849
2022-02-25 20:01:36,393 P7293 INFO --- 591/591 batches finished ---
2022-02-25 20:01:36,434 P7293 INFO Train loss: 0.504962
2022-02-25 20:01:36,434 P7293 INFO ************ Epoch=28 end ************
2022-02-25 20:02:51,192 P7293 INFO [Metrics] logloss: 0.524827 - AUC: 0.816997
2022-02-25 20:02:51,195 P7293 INFO Save best model: monitor(max): 0.292170
2022-02-25 20:02:51,258 P7293 INFO --- 591/591 batches finished ---
2022-02-25 20:02:51,299 P7293 INFO Train loss: 0.500109
2022-02-25 20:02:51,299 P7293 INFO ************ Epoch=29 end ************
2022-02-25 20:04:06,080 P7293 INFO [Metrics] logloss: 0.525081 - AUC: 0.817189
2022-02-25 20:04:06,083 P7293 INFO Monitor(max) STOP: 0.292107 !
2022-02-25 20:04:06,083 P7293 INFO Reduce learning rate on plateau: 0.000010
2022-02-25 20:04:06,083 P7293 INFO --- 591/591 batches finished ---
2022-02-25 20:04:06,124 P7293 INFO Train loss: 0.496678
2022-02-25 20:04:06,124 P7293 INFO ************ Epoch=30 end ************
2022-02-25 20:05:20,777 P7293 INFO [Metrics] logloss: 0.525935 - AUC: 0.816999
2022-02-25 20:05:20,781 P7293 INFO Monitor(max) STOP: 0.291064 !
2022-02-25 20:05:20,781 P7293 INFO Reduce learning rate on plateau: 0.000001
2022-02-25 20:05:20,782 P7293 INFO Early stopping at epoch=31
2022-02-25 20:05:20,782 P7293 INFO --- 591/591 batches finished ---
2022-02-25 20:05:20,823 P7293 INFO Train loss: 0.489502
2022-02-25 20:05:20,823 P7293 INFO Training finished.
2022-02-25 20:05:20,823 P7293 INFO Load best model: /home/XXX/FuxiCTR_v1.0/benchmarks/KKBox/AFM_kkbox/kkbox_x1_bf23df6c/AFM_kkbox_x1_007_7d582ff2_model.ckpt
2022-02-25 20:05:20,884 P7293 INFO ****** Validation evaluation ******
2022-02-25 20:05:25,194 P7293 INFO [Metrics] logloss: 0.524827 - AUC: 0.816997
2022-02-25 20:05:25,236 P7293 INFO ******** Test evaluation ********
2022-02-25 20:05:25,236 P7293 INFO Loading data...
2022-02-25 20:05:25,237 P7293 INFO Loading data from h5: ../data/KKBox/kkbox_x1_bf23df6c/test.h5
2022-02-25 20:05:25,307 P7293 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-02-25 20:05:25,307 P7293 INFO Loading test data done.
2022-02-25 20:05:29,719 P7293 INFO [Metrics] logloss: 0.524060 - AUC: 0.817460

```
