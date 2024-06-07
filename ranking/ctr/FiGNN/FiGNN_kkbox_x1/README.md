## FiGNN_kkbox_x1

A hands-on guide to run the FiGNN model on the KKBox_x1 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FiGNN](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FiGNN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FiGNN_kkbox_x1_tuner_config_04](./FiGNN_kkbox_x1_tuner_config_04). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FiGNN_kkbox_x1
    nohup python run_expid.py --config ./FiGNN_kkbox_x1_tuner_config_04 --expid FiGNN_kkbox_x1_003_92b4cef8 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.489593 | 0.847155  |


### Logs
```python
2022-03-12 07:52:54,377 P47639 INFO {
    "batch_size": "2000",
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
    "gnn_layers": "6",
    "gpu": "2",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "FiGNN",
    "model_id": "FiGNN_kkbox_x1_003_92b4cef8",
    "model_root": "./KKBox/FiGNN_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reuse_graph_layer": "False",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/KKBox/KKBox_x1/test.csv",
    "train_data": "../data/KKBox/KKBox_x1/train.csv",
    "use_gru": "True",
    "use_hdf5": "True",
    "use_residual": "True",
    "valid_data": "../data/KKBox/KKBox_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2022-03-12 07:52:54,378 P47639 INFO Set up feature encoder...
2022-03-12 07:52:54,379 P47639 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-12 07:52:55,259 P47639 INFO Total number of parameters: 14485376.
2022-03-12 07:52:55,259 P47639 INFO Loading data...
2022-03-12 07:52:55,259 P47639 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-12 07:52:55,627 P47639 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-12 07:52:55,823 P47639 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-12 07:52:55,842 P47639 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-12 07:52:55,842 P47639 INFO Loading train data done.
2022-03-12 07:52:59,641 P47639 INFO Start training: 2951 batches/epoch
2022-03-12 07:52:59,641 P47639 INFO ************ Epoch=1 start ************
2022-03-12 08:06:59,321 P47639 INFO [Metrics] logloss: 0.547198 - AUC: 0.794177
2022-03-12 08:06:59,324 P47639 INFO Save best model: monitor(max): 0.246980
2022-03-12 08:06:59,560 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 08:06:59,589 P47639 INFO Train loss: 0.591372
2022-03-12 08:06:59,589 P47639 INFO ************ Epoch=1 end ************
2022-03-12 08:20:58,622 P47639 INFO [Metrics] logloss: 0.533039 - AUC: 0.807241
2022-03-12 08:20:58,625 P47639 INFO Save best model: monitor(max): 0.274202
2022-03-12 08:20:58,687 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 08:20:58,726 P47639 INFO Train loss: 0.566705
2022-03-12 08:20:58,726 P47639 INFO ************ Epoch=2 end ************
2022-03-12 08:34:59,122 P47639 INFO [Metrics] logloss: 0.525146 - AUC: 0.814153
2022-03-12 08:34:59,125 P47639 INFO Save best model: monitor(max): 0.289007
2022-03-12 08:34:59,192 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 08:34:59,230 P47639 INFO Train loss: 0.557188
2022-03-12 08:34:59,230 P47639 INFO ************ Epoch=3 end ************
2022-03-12 08:48:59,462 P47639 INFO [Metrics] logloss: 0.520442 - AUC: 0.817919
2022-03-12 08:48:59,465 P47639 INFO Save best model: monitor(max): 0.297477
2022-03-12 08:48:59,531 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 08:48:59,563 P47639 INFO Train loss: 0.552129
2022-03-12 08:48:59,563 P47639 INFO ************ Epoch=4 end ************
2022-03-12 09:02:59,155 P47639 INFO [Metrics] logloss: 0.516305 - AUC: 0.821295
2022-03-12 09:02:59,155 P47639 INFO Save best model: monitor(max): 0.304990
2022-03-12 09:02:59,222 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 09:02:59,253 P47639 INFO Train loss: 0.549180
2022-03-12 09:02:59,253 P47639 INFO ************ Epoch=5 end ************
2022-03-12 09:16:57,799 P47639 INFO [Metrics] logloss: 0.513276 - AUC: 0.823822
2022-03-12 09:16:57,800 P47639 INFO Save best model: monitor(max): 0.310546
2022-03-12 09:16:57,859 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 09:16:57,898 P47639 INFO Train loss: 0.546934
2022-03-12 09:16:57,898 P47639 INFO ************ Epoch=6 end ************
2022-03-12 09:30:58,143 P47639 INFO [Metrics] logloss: 0.511793 - AUC: 0.825316
2022-03-12 09:30:58,143 P47639 INFO Save best model: monitor(max): 0.313523
2022-03-12 09:30:58,201 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 09:30:58,237 P47639 INFO Train loss: 0.545030
2022-03-12 09:30:58,237 P47639 INFO ************ Epoch=7 end ************
2022-03-12 09:44:57,288 P47639 INFO [Metrics] logloss: 0.510450 - AUC: 0.826357
2022-03-12 09:44:57,292 P47639 INFO Save best model: monitor(max): 0.315907
2022-03-12 09:44:57,361 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 09:44:57,398 P47639 INFO Train loss: 0.543703
2022-03-12 09:44:57,398 P47639 INFO ************ Epoch=8 end ************
2022-03-12 09:58:55,783 P47639 INFO [Metrics] logloss: 0.508012 - AUC: 0.828280
2022-03-12 09:58:55,787 P47639 INFO Save best model: monitor(max): 0.320268
2022-03-12 09:58:55,853 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 09:58:55,891 P47639 INFO Train loss: 0.542347
2022-03-12 09:58:55,891 P47639 INFO ************ Epoch=9 end ************
2022-03-12 10:12:54,895 P47639 INFO [Metrics] logloss: 0.506771 - AUC: 0.829356
2022-03-12 10:12:54,898 P47639 INFO Save best model: monitor(max): 0.322585
2022-03-12 10:12:54,960 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 10:12:54,993 P47639 INFO Train loss: 0.541276
2022-03-12 10:12:54,993 P47639 INFO ************ Epoch=10 end ************
2022-03-12 10:26:53,569 P47639 INFO [Metrics] logloss: 0.506022 - AUC: 0.829868
2022-03-12 10:26:53,572 P47639 INFO Save best model: monitor(max): 0.323846
2022-03-12 10:26:53,637 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 10:26:53,674 P47639 INFO Train loss: 0.540264
2022-03-12 10:26:53,674 P47639 INFO ************ Epoch=11 end ************
2022-03-12 10:40:50,110 P47639 INFO [Metrics] logloss: 0.504779 - AUC: 0.830820
2022-03-12 10:40:50,113 P47639 INFO Save best model: monitor(max): 0.326041
2022-03-12 10:40:50,176 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 10:40:50,207 P47639 INFO Train loss: 0.539319
2022-03-12 10:40:50,207 P47639 INFO ************ Epoch=12 end ************
2022-03-12 10:54:46,823 P47639 INFO [Metrics] logloss: 0.503699 - AUC: 0.832103
2022-03-12 10:54:46,826 P47639 INFO Save best model: monitor(max): 0.328404
2022-03-12 10:54:46,896 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 10:54:46,925 P47639 INFO Train loss: 0.538414
2022-03-12 10:54:46,925 P47639 INFO ************ Epoch=13 end ************
2022-03-12 11:08:43,382 P47639 INFO [Metrics] logloss: 0.503026 - AUC: 0.832401
2022-03-12 11:08:43,384 P47639 INFO Save best model: monitor(max): 0.329375
2022-03-12 11:08:43,450 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 11:08:43,481 P47639 INFO Train loss: 0.537611
2022-03-12 11:08:43,481 P47639 INFO ************ Epoch=14 end ************
2022-03-12 11:22:40,237 P47639 INFO [Metrics] logloss: 0.502594 - AUC: 0.832658
2022-03-12 11:22:40,238 P47639 INFO Save best model: monitor(max): 0.330064
2022-03-12 11:22:40,297 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 11:22:40,327 P47639 INFO Train loss: 0.536884
2022-03-12 11:22:40,327 P47639 INFO ************ Epoch=15 end ************
2022-03-12 11:36:36,574 P47639 INFO [Metrics] logloss: 0.501443 - AUC: 0.833918
2022-03-12 11:36:36,574 P47639 INFO Save best model: monitor(max): 0.332476
2022-03-12 11:36:36,640 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 11:36:36,675 P47639 INFO Train loss: 0.536097
2022-03-12 11:36:36,675 P47639 INFO ************ Epoch=16 end ************
2022-03-12 11:50:32,808 P47639 INFO [Metrics] logloss: 0.501407 - AUC: 0.833709
2022-03-12 11:50:32,812 P47639 INFO Monitor(max) STOP: 0.332302 !
2022-03-12 11:50:32,813 P47639 INFO Reduce learning rate on plateau: 0.000100
2022-03-12 11:50:32,813 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 11:50:32,843 P47639 INFO Train loss: 0.535273
2022-03-12 11:50:32,843 P47639 INFO ************ Epoch=17 end ************
2022-03-12 12:04:28,908 P47639 INFO [Metrics] logloss: 0.488668 - AUC: 0.845942
2022-03-12 12:04:28,912 P47639 INFO Save best model: monitor(max): 0.357274
2022-03-12 12:04:28,980 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 12:04:29,012 P47639 INFO Train loss: 0.481175
2022-03-12 12:04:29,012 P47639 INFO ************ Epoch=18 end ************
2022-03-12 12:18:24,681 P47639 INFO [Metrics] logloss: 0.489386 - AUC: 0.847260
2022-03-12 12:18:24,684 P47639 INFO Save best model: monitor(max): 0.357874
2022-03-12 12:18:24,758 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 12:18:24,797 P47639 INFO Train loss: 0.455649
2022-03-12 12:18:24,797 P47639 INFO ************ Epoch=19 end ************
2022-03-12 12:32:20,678 P47639 INFO [Metrics] logloss: 0.492988 - AUC: 0.846941
2022-03-12 12:32:20,681 P47639 INFO Monitor(max) STOP: 0.353954 !
2022-03-12 12:32:20,681 P47639 INFO Reduce learning rate on plateau: 0.000010
2022-03-12 12:32:20,681 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 12:32:20,713 P47639 INFO Train loss: 0.442881
2022-03-12 12:32:20,713 P47639 INFO ************ Epoch=20 end ************
2022-03-12 12:46:16,440 P47639 INFO [Metrics] logloss: 0.502942 - AUC: 0.845406
2022-03-12 12:46:16,444 P47639 INFO Monitor(max) STOP: 0.342464 !
2022-03-12 12:46:16,444 P47639 INFO Reduce learning rate on plateau: 0.000001
2022-03-12 12:46:16,444 P47639 INFO Early stopping at epoch=21
2022-03-12 12:46:16,444 P47639 INFO --- 2951/2951 batches finished ---
2022-03-12 12:46:16,474 P47639 INFO Train loss: 0.419738
2022-03-12 12:46:16,474 P47639 INFO Training finished.
2022-03-12 12:46:16,474 P47639 INFO Load best model: /cache/FuxiCTR/benchmarks/KKBox/FiGNN_kkbox_x1/kkbox_x1_227d337d/FiGNN_kkbox_x1_003_92b4cef8_model.ckpt
2022-03-12 12:46:16,584 P47639 INFO ****** Validation evaluation ******
2022-03-12 12:46:49,578 P47639 INFO [Metrics] logloss: 0.489386 - AUC: 0.847260
2022-03-12 12:46:49,640 P47639 INFO ******** Test evaluation ********
2022-03-12 12:46:49,640 P47639 INFO Loading data...
2022-03-12 12:46:49,640 P47639 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-12 12:46:49,712 P47639 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-12 12:46:49,712 P47639 INFO Loading test data done.
2022-03-12 12:47:22,914 P47639 INFO [Metrics] logloss: 0.489593 - AUC: 0.847155

```
