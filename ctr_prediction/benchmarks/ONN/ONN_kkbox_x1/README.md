## ONN_kkbox_x1

A hands-on guide to run the ONN model on the KKBox_x1 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [ONN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/ONN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [ONN_kkbox_x1_tuner_config_02](./ONN_kkbox_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd ONN_kkbox_x1
    nohup python run_expid.py --config ./ONN_kkbox_x1_tuner_config_02 --expid ONN_kkbox_x1_024_650b448f --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.485572 | 0.849850  |


### Logs
```python
2022-03-08 19:04:45,773 P6876 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.0001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "3",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "ONN",
    "model_id": "ONN_kkbox_x1_024_650b448f",
    "model_root": "./KKBox/ONN_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.2",
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
2022-03-08 19:04:45,774 P6876 INFO Set up feature encoder...
2022-03-08 19:04:45,774 P6876 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-08 19:04:48,829 P6876 INFO Total number of parameters: 156244009.
2022-03-08 19:04:48,830 P6876 INFO Loading data...
2022-03-08 19:04:48,830 P6876 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-08 19:04:49,283 P6876 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-08 19:04:49,484 P6876 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-08 19:04:49,500 P6876 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-08 19:04:49,500 P6876 INFO Loading train data done.
2022-03-08 19:04:52,395 P6876 INFO Start training: 591 batches/epoch
2022-03-08 19:04:52,396 P6876 INFO ************ Epoch=1 start ************
2022-03-08 19:09:26,048 P6876 INFO [Metrics] logloss: 0.548579 - AUC: 0.792693
2022-03-08 19:09:26,052 P6876 INFO Save best model: monitor(max): 0.244114
2022-03-08 19:09:26,726 P6876 INFO --- 591/591 batches finished ---
2022-03-08 19:09:26,765 P6876 INFO Train loss: 0.590560
2022-03-08 19:09:26,765 P6876 INFO ************ Epoch=1 end ************
2022-03-08 19:14:00,349 P6876 INFO [Metrics] logloss: 0.530380 - AUC: 0.809497
2022-03-08 19:14:00,352 P6876 INFO Save best model: monitor(max): 0.279117
2022-03-08 19:14:01,140 P6876 INFO --- 591/591 batches finished ---
2022-03-08 19:14:01,185 P6876 INFO Train loss: 0.569592
2022-03-08 19:14:01,186 P6876 INFO ************ Epoch=2 end ************
2022-03-08 19:18:34,487 P6876 INFO [Metrics] logloss: 0.519652 - AUC: 0.818721
2022-03-08 19:18:34,490 P6876 INFO Save best model: monitor(max): 0.299069
2022-03-08 19:18:35,267 P6876 INFO --- 591/591 batches finished ---
2022-03-08 19:18:35,309 P6876 INFO Train loss: 0.559162
2022-03-08 19:18:35,309 P6876 INFO ************ Epoch=3 end ************
2022-03-08 19:23:08,911 P6876 INFO [Metrics] logloss: 0.512027 - AUC: 0.824718
2022-03-08 19:23:08,915 P6876 INFO Save best model: monitor(max): 0.312691
2022-03-08 19:23:09,714 P6876 INFO --- 591/591 batches finished ---
2022-03-08 19:23:09,753 P6876 INFO Train loss: 0.552182
2022-03-08 19:23:09,753 P6876 INFO ************ Epoch=4 end ************
2022-03-08 19:27:42,683 P6876 INFO [Metrics] logloss: 0.507272 - AUC: 0.828877
2022-03-08 19:27:42,687 P6876 INFO Save best model: monitor(max): 0.321604
2022-03-08 19:27:43,454 P6876 INFO --- 591/591 batches finished ---
2022-03-08 19:27:43,494 P6876 INFO Train loss: 0.547368
2022-03-08 19:27:43,495 P6876 INFO ************ Epoch=5 end ************
2022-03-08 19:32:18,278 P6876 INFO [Metrics] logloss: 0.502355 - AUC: 0.832518
2022-03-08 19:32:18,281 P6876 INFO Save best model: monitor(max): 0.330163
2022-03-08 19:32:19,085 P6876 INFO --- 591/591 batches finished ---
2022-03-08 19:32:19,133 P6876 INFO Train loss: 0.543966
2022-03-08 19:32:19,133 P6876 INFO ************ Epoch=6 end ************
2022-03-08 19:36:52,614 P6876 INFO [Metrics] logloss: 0.499648 - AUC: 0.834613
2022-03-08 19:36:52,618 P6876 INFO Save best model: monitor(max): 0.334964
2022-03-08 19:36:53,390 P6876 INFO --- 591/591 batches finished ---
2022-03-08 19:36:53,429 P6876 INFO Train loss: 0.540762
2022-03-08 19:36:53,430 P6876 INFO ************ Epoch=7 end ************
2022-03-08 19:41:26,393 P6876 INFO [Metrics] logloss: 0.496755 - AUC: 0.836759
2022-03-08 19:41:26,396 P6876 INFO Save best model: monitor(max): 0.340003
2022-03-08 19:41:27,216 P6876 INFO --- 591/591 batches finished ---
2022-03-08 19:41:27,257 P6876 INFO Train loss: 0.538219
2022-03-08 19:41:27,257 P6876 INFO ************ Epoch=8 end ************
2022-03-08 19:45:59,931 P6876 INFO [Metrics] logloss: 0.494898 - AUC: 0.838197
2022-03-08 19:45:59,934 P6876 INFO Save best model: monitor(max): 0.343299
2022-03-08 19:46:00,730 P6876 INFO --- 591/591 batches finished ---
2022-03-08 19:46:00,769 P6876 INFO Train loss: 0.536115
2022-03-08 19:46:00,769 P6876 INFO ************ Epoch=9 end ************
2022-03-08 19:50:34,081 P6876 INFO [Metrics] logloss: 0.492921 - AUC: 0.839712
2022-03-08 19:50:34,085 P6876 INFO Save best model: monitor(max): 0.346790
2022-03-08 19:50:34,967 P6876 INFO --- 591/591 batches finished ---
2022-03-08 19:50:35,006 P6876 INFO Train loss: 0.534274
2022-03-08 19:50:35,006 P6876 INFO ************ Epoch=10 end ************
2022-03-08 19:55:08,732 P6876 INFO [Metrics] logloss: 0.492187 - AUC: 0.840635
2022-03-08 19:55:08,736 P6876 INFO Save best model: monitor(max): 0.348447
2022-03-08 19:55:09,582 P6876 INFO --- 591/591 batches finished ---
2022-03-08 19:55:09,629 P6876 INFO Train loss: 0.533041
2022-03-08 19:55:09,629 P6876 INFO ************ Epoch=11 end ************
2022-03-08 19:59:44,027 P6876 INFO [Metrics] logloss: 0.491408 - AUC: 0.841320
2022-03-08 19:59:44,030 P6876 INFO Save best model: monitor(max): 0.349911
2022-03-08 19:59:44,958 P6876 INFO --- 591/591 batches finished ---
2022-03-08 19:59:45,005 P6876 INFO Train loss: 0.531557
2022-03-08 19:59:45,005 P6876 INFO ************ Epoch=12 end ************
2022-03-08 20:04:17,871 P6876 INFO [Metrics] logloss: 0.490140 - AUC: 0.842163
2022-03-08 20:04:17,874 P6876 INFO Save best model: monitor(max): 0.352022
2022-03-08 20:04:18,731 P6876 INFO --- 591/591 batches finished ---
2022-03-08 20:04:18,769 P6876 INFO Train loss: 0.530160
2022-03-08 20:04:18,769 P6876 INFO ************ Epoch=13 end ************
2022-03-08 20:08:51,595 P6876 INFO [Metrics] logloss: 0.489063 - AUC: 0.842986
2022-03-08 20:08:51,598 P6876 INFO Save best model: monitor(max): 0.353923
2022-03-08 20:08:52,417 P6876 INFO --- 591/591 batches finished ---
2022-03-08 20:08:52,463 P6876 INFO Train loss: 0.529414
2022-03-08 20:08:52,463 P6876 INFO ************ Epoch=14 end ************
2022-03-08 20:13:26,063 P6876 INFO [Metrics] logloss: 0.489325 - AUC: 0.842885
2022-03-08 20:13:26,066 P6876 INFO Monitor(max) STOP: 0.353560 !
2022-03-08 20:13:26,067 P6876 INFO Reduce learning rate on plateau: 0.000100
2022-03-08 20:13:26,067 P6876 INFO --- 591/591 batches finished ---
2022-03-08 20:13:26,109 P6876 INFO Train loss: 0.528521
2022-03-08 20:13:26,110 P6876 INFO ************ Epoch=15 end ************
2022-03-08 20:17:58,892 P6876 INFO [Metrics] logloss: 0.485695 - AUC: 0.849772
2022-03-08 20:17:58,897 P6876 INFO Save best model: monitor(max): 0.364077
2022-03-08 20:17:59,649 P6876 INFO --- 591/591 batches finished ---
2022-03-08 20:17:59,691 P6876 INFO Train loss: 0.471634
2022-03-08 20:17:59,692 P6876 INFO ************ Epoch=16 end ************
2022-03-08 20:22:32,099 P6876 INFO [Metrics] logloss: 0.488454 - AUC: 0.850709
2022-03-08 20:22:32,102 P6876 INFO Monitor(max) STOP: 0.362254 !
2022-03-08 20:22:32,102 P6876 INFO Reduce learning rate on plateau: 0.000010
2022-03-08 20:22:32,102 P6876 INFO --- 591/591 batches finished ---
2022-03-08 20:22:32,141 P6876 INFO Train loss: 0.444901
2022-03-08 20:22:32,141 P6876 INFO ************ Epoch=17 end ************
2022-03-08 20:27:04,972 P6876 INFO [Metrics] logloss: 0.496645 - AUC: 0.849678
2022-03-08 20:27:04,976 P6876 INFO Monitor(max) STOP: 0.353033 !
2022-03-08 20:27:04,976 P6876 INFO Reduce learning rate on plateau: 0.000001
2022-03-08 20:27:04,976 P6876 INFO Early stopping at epoch=18
2022-03-08 20:27:04,976 P6876 INFO --- 591/591 batches finished ---
2022-03-08 20:27:05,015 P6876 INFO Train loss: 0.421104
2022-03-08 20:27:05,015 P6876 INFO Training finished.
2022-03-08 20:27:05,015 P6876 INFO Load best model: /cache/FuxiCTR/benchmarks/KKBox/ONN_kkbox_x1/kkbox_x1_227d337d/ONN_kkbox_x1_024_650b448f_model.ckpt
2022-03-08 20:27:05,694 P6876 INFO ****** Validation evaluation ******
2022-03-08 20:27:09,847 P6876 INFO [Metrics] logloss: 0.485695 - AUC: 0.849772
2022-03-08 20:27:09,903 P6876 INFO ******** Test evaluation ********
2022-03-08 20:27:09,903 P6876 INFO Loading data...
2022-03-08 20:27:09,903 P6876 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-08 20:27:09,977 P6876 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-08 20:27:09,977 P6876 INFO Loading test data done.
2022-03-08 20:27:13,990 P6876 INFO [Metrics] logloss: 0.485572 - AUC: 0.849850

```
