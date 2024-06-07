## InterHAt_kkbox_x1

A hands-on guide to run the InterHAt model on the KKBox_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

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
Dataset ID: [KKBox_x1](https://github.com/reczoo/Datasets/tree/main/KKBox/KKBox_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [InterHAt](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/InterHAt.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [InterHAt_kkbox_x1_tuner_config_05](./InterHAt_kkbox_x1_tuner_config_05). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd InterHAt_kkbox_x1
    nohup python run_expid.py --config ./InterHAt_kkbox_x1_tuner_config_05 --expid InterHAt_kkbox_x1_007_5a759e50 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.486302 | 0.845923  |


### Logs
```python
2022-02-27 20:37:26,509 P1370 INFO {
    "attention_dim": "256",
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "embedding_dim": "128",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_dim": "1024",
    "hidden_units": "[1000, 1000]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "layer_norm": "True",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "InterHAt",
    "model_id": "InterHAt_kkbox_x1_007_5a759e50",
    "model_root": "./KKBox/InterHAt_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "2",
    "optimizer": "adam",
    "order": "4",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/KKBox/KKBox_x1/test.csv",
    "train_data": "../data/KKBox/KKBox_x1/train.csv",
    "use_hdf5": "True",
    "use_residual": "True",
    "valid_data": "../data/KKBox/KKBox_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2022-02-27 20:37:26,509 P1370 INFO Set up feature encoder...
2022-02-27 20:37:26,510 P1370 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-02-27 20:37:27,716 P1370 INFO Total number of parameters: 14130169.
2022-02-27 20:37:27,716 P1370 INFO Loading data...
2022-02-27 20:37:27,718 P1370 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-02-27 20:37:28,119 P1370 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-02-27 20:37:28,347 P1370 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-02-27 20:37:28,366 P1370 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-02-27 20:37:28,366 P1370 INFO Loading train data done.
2022-02-27 20:37:31,109 P1370 INFO Start training: 591 batches/epoch
2022-02-27 20:37:31,110 P1370 INFO ************ Epoch=1 start ************
2022-02-27 20:40:44,529 P1370 INFO [Metrics] logloss: 0.559703 - AUC: 0.781977
2022-02-27 20:40:44,530 P1370 INFO Save best model: monitor(max): 0.222274
2022-02-27 20:40:44,581 P1370 INFO --- 591/591 batches finished ---
2022-02-27 20:40:44,626 P1370 INFO Train loss: 0.646673
2022-02-27 20:40:44,626 P1370 INFO ************ Epoch=1 end ************
2022-02-27 20:43:58,095 P1370 INFO [Metrics] logloss: 0.548754 - AUC: 0.792722
2022-02-27 20:43:58,100 P1370 INFO Save best model: monitor(max): 0.243967
2022-02-27 20:43:58,177 P1370 INFO --- 591/591 batches finished ---
2022-02-27 20:43:58,222 P1370 INFO Train loss: 0.629355
2022-02-27 20:43:58,222 P1370 INFO ************ Epoch=2 end ************
2022-02-27 20:47:11,166 P1370 INFO [Metrics] logloss: 0.541665 - AUC: 0.799251
2022-02-27 20:47:11,169 P1370 INFO Save best model: monitor(max): 0.257586
2022-02-27 20:47:11,248 P1370 INFO --- 591/591 batches finished ---
2022-02-27 20:47:11,288 P1370 INFO Train loss: 0.620812
2022-02-27 20:47:11,288 P1370 INFO ************ Epoch=3 end ************
2022-02-27 20:50:24,050 P1370 INFO [Metrics] logloss: 0.537295 - AUC: 0.804455
2022-02-27 20:50:24,053 P1370 INFO Save best model: monitor(max): 0.267160
2022-02-27 20:50:24,130 P1370 INFO --- 591/591 batches finished ---
2022-02-27 20:50:24,172 P1370 INFO Train loss: 0.614568
2022-02-27 20:50:24,173 P1370 INFO ************ Epoch=4 end ************
2022-02-27 20:53:37,196 P1370 INFO [Metrics] logloss: 0.533304 - AUC: 0.807144
2022-02-27 20:53:37,200 P1370 INFO Save best model: monitor(max): 0.273839
2022-02-27 20:53:37,282 P1370 INFO --- 591/591 batches finished ---
2022-02-27 20:53:37,325 P1370 INFO Train loss: 0.611283
2022-02-27 20:53:37,325 P1370 INFO ************ Epoch=5 end ************
2022-02-27 20:56:50,344 P1370 INFO [Metrics] logloss: 0.529254 - AUC: 0.810495
2022-02-27 20:56:50,347 P1370 INFO Save best model: monitor(max): 0.281241
2022-02-27 20:56:50,426 P1370 INFO --- 591/591 batches finished ---
2022-02-27 20:56:50,468 P1370 INFO Train loss: 0.607801
2022-02-27 20:56:50,468 P1370 INFO ************ Epoch=6 end ************
2022-02-27 21:00:03,410 P1370 INFO [Metrics] logloss: 0.526305 - AUC: 0.813081
2022-02-27 21:00:03,413 P1370 INFO Save best model: monitor(max): 0.286776
2022-02-27 21:00:03,502 P1370 INFO --- 591/591 batches finished ---
2022-02-27 21:00:03,549 P1370 INFO Train loss: 0.603736
2022-02-27 21:00:03,549 P1370 INFO ************ Epoch=7 end ************
2022-02-27 21:03:16,253 P1370 INFO [Metrics] logloss: 0.523217 - AUC: 0.815786
2022-02-27 21:03:16,256 P1370 INFO Save best model: monitor(max): 0.292569
2022-02-27 21:03:16,335 P1370 INFO --- 591/591 batches finished ---
2022-02-27 21:03:16,377 P1370 INFO Train loss: 0.602120
2022-02-27 21:03:16,378 P1370 INFO ************ Epoch=8 end ************
2022-02-27 21:06:29,024 P1370 INFO [Metrics] logloss: 0.520785 - AUC: 0.818161
2022-02-27 21:06:29,027 P1370 INFO Save best model: monitor(max): 0.297376
2022-02-27 21:06:29,110 P1370 INFO --- 591/591 batches finished ---
2022-02-27 21:06:29,155 P1370 INFO Train loss: 0.599334
2022-02-27 21:06:29,155 P1370 INFO ************ Epoch=9 end ************
2022-02-27 21:09:42,252 P1370 INFO [Metrics] logloss: 0.518908 - AUC: 0.819533
2022-02-27 21:09:42,255 P1370 INFO Save best model: monitor(max): 0.300625
2022-02-27 21:09:42,338 P1370 INFO --- 591/591 batches finished ---
2022-02-27 21:09:42,388 P1370 INFO Train loss: 0.597244
2022-02-27 21:09:42,388 P1370 INFO ************ Epoch=10 end ************
2022-02-27 21:12:55,022 P1370 INFO [Metrics] logloss: 0.517630 - AUC: 0.820442
2022-02-27 21:12:55,025 P1370 INFO Save best model: monitor(max): 0.302812
2022-02-27 21:12:55,112 P1370 INFO --- 591/591 batches finished ---
2022-02-27 21:12:55,156 P1370 INFO Train loss: 0.594503
2022-02-27 21:12:55,156 P1370 INFO ************ Epoch=11 end ************
2022-02-27 21:16:08,061 P1370 INFO [Metrics] logloss: 0.515927 - AUC: 0.821800
2022-02-27 21:16:08,065 P1370 INFO Save best model: monitor(max): 0.305873
2022-02-27 21:16:08,149 P1370 INFO --- 591/591 batches finished ---
2022-02-27 21:16:08,194 P1370 INFO Train loss: 0.592965
2022-02-27 21:16:08,194 P1370 INFO ************ Epoch=12 end ************
2022-02-27 21:19:20,945 P1370 INFO [Metrics] logloss: 0.515302 - AUC: 0.822243
2022-02-27 21:19:20,949 P1370 INFO Save best model: monitor(max): 0.306941
2022-02-27 21:19:21,030 P1370 INFO --- 591/591 batches finished ---
2022-02-27 21:19:21,083 P1370 INFO Train loss: 0.591506
2022-02-27 21:19:21,083 P1370 INFO ************ Epoch=13 end ************
2022-02-27 21:22:34,057 P1370 INFO [Metrics] logloss: 0.515431 - AUC: 0.822797
2022-02-27 21:22:34,061 P1370 INFO Save best model: monitor(max): 0.307366
2022-02-27 21:22:34,138 P1370 INFO --- 591/591 batches finished ---
2022-02-27 21:22:34,181 P1370 INFO Train loss: 0.590253
2022-02-27 21:22:34,181 P1370 INFO ************ Epoch=14 end ************
2022-02-27 21:25:47,516 P1370 INFO [Metrics] logloss: 0.512845 - AUC: 0.824224
2022-02-27 21:25:47,520 P1370 INFO Save best model: monitor(max): 0.311379
2022-02-27 21:25:47,597 P1370 INFO --- 591/591 batches finished ---
2022-02-27 21:25:47,646 P1370 INFO Train loss: 0.588450
2022-02-27 21:25:47,646 P1370 INFO ************ Epoch=15 end ************
2022-02-27 21:29:00,544 P1370 INFO [Metrics] logloss: 0.513394 - AUC: 0.823710
2022-02-27 21:29:00,547 P1370 INFO Monitor(max) STOP: 0.310316 !
2022-02-27 21:29:00,547 P1370 INFO Reduce learning rate on plateau: 0.000100
2022-02-27 21:29:00,548 P1370 INFO --- 591/591 batches finished ---
2022-02-27 21:29:00,591 P1370 INFO Train loss: 0.587190
2022-02-27 21:29:00,591 P1370 INFO ************ Epoch=16 end ************
2022-02-27 21:32:13,387 P1370 INFO [Metrics] logloss: 0.488904 - AUC: 0.843135
2022-02-27 21:32:13,391 P1370 INFO Save best model: monitor(max): 0.354232
2022-02-27 21:32:13,467 P1370 INFO --- 591/591 batches finished ---
2022-02-27 21:32:13,516 P1370 INFO Train loss: 0.522910
2022-02-27 21:32:13,516 P1370 INFO ************ Epoch=17 end ************
2022-02-27 21:35:26,100 P1370 INFO [Metrics] logloss: 0.486820 - AUC: 0.845579
2022-02-27 21:35:26,103 P1370 INFO Save best model: monitor(max): 0.358759
2022-02-27 21:35:26,180 P1370 INFO --- 591/591 batches finished ---
2022-02-27 21:35:26,224 P1370 INFO Train loss: 0.487367
2022-02-27 21:35:26,224 P1370 INFO ************ Epoch=18 end ************
2022-02-27 21:38:39,240 P1370 INFO [Metrics] logloss: 0.492858 - AUC: 0.844013
2022-02-27 21:38:39,243 P1370 INFO Monitor(max) STOP: 0.351155 !
2022-02-27 21:38:39,244 P1370 INFO Reduce learning rate on plateau: 0.000010
2022-02-27 21:38:39,244 P1370 INFO --- 591/591 batches finished ---
2022-02-27 21:38:39,290 P1370 INFO Train loss: 0.468590
2022-02-27 21:38:39,290 P1370 INFO ************ Epoch=19 end ************
2022-02-27 21:41:52,336 P1370 INFO [Metrics] logloss: 0.526534 - AUC: 0.836668
2022-02-27 21:41:52,339 P1370 INFO Monitor(max) STOP: 0.310134 !
2022-02-27 21:41:52,339 P1370 INFO Reduce learning rate on plateau: 0.000001
2022-02-27 21:41:52,339 P1370 INFO Early stopping at epoch=20
2022-02-27 21:41:52,339 P1370 INFO --- 591/591 batches finished ---
2022-02-27 21:41:52,382 P1370 INFO Train loss: 0.418623
2022-02-27 21:41:52,382 P1370 INFO Training finished.
2022-02-27 21:41:52,382 P1370 INFO Load best model: /home/XXX/FuxiCTR_v1.0/benchmarks/KKBox/InterHAt_kkbox_x1/kkbox_x1_227d337d/InterHAt_kkbox_x1_007_5a759e50_model.ckpt
2022-02-27 21:41:52,458 P1370 INFO ****** Validation evaluation ******
2022-02-27 21:42:01,087 P1370 INFO [Metrics] logloss: 0.486820 - AUC: 0.845579
2022-02-27 21:42:01,137 P1370 INFO ******** Test evaluation ********
2022-02-27 21:42:01,137 P1370 INFO Loading data...
2022-02-27 21:42:01,137 P1370 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-02-27 21:42:01,205 P1370 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-02-27 21:42:01,205 P1370 INFO Loading test data done.
2022-02-27 21:42:09,784 P1370 INFO [Metrics] logloss: 0.486302 - AUC: 0.845923

```
