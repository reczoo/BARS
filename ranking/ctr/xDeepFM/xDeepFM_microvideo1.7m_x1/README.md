## xDeepFM_microvideo1.7m_x1

A hands-on guide to run the xDeepFM model on the MicroVideo1.7M_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)


| [Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs) |
|:-----------------------------:|:-----------:|:--------:|:--------:|-------|
### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) Gold 6278C CPU @ 2.60GHz
  GPU: Tesla V100 32G
  RAM: 755G

  ```

+ Software

  ```python
  cuda: 10.2
  python: 3.7.10
  pytorch: 1.11.0
  pandas: 1.1.5
  numpy: 1.19.5
  scipy: 1.5.2
  sklearn: 0.22.1
  pyyaml: 6.0
  h5py: 2.8.0
  tqdm: 4.64.0
  fuxictr: 2.0.1

  ```

### Dataset
Please refer to [MicroVideo1.7M_x1](https://github.com/reczoo/Datasets/tree/main/MicroVideo/MicroVideo1.7M_x1) to get the dataset details.

### Code

We use the [xDeepFM](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/xDeepFM) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/MicroVideo1.7M/MicroVideo1.7M_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [xDeepFM_microvideo1.7m_x1_tuner_config_04](./xDeepFM_microvideo1.7m_x1_tuner_config_04). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/xDeepFM
    nohup python run_expid.py --config XXX/benchmarks/xDeepFM/xDeepFM_microvideo1.7m_x1_tuner_config_04 --expid xDeepFM_microvideo1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.688869 | 0.736228 | 0.412212  |


### Logs
```python
2022-08-19 19:00:45,614 P61093 INFO Params: {
    "batch_norm": "True",
    "batch_size": "2048",
    "cin_hidden_units": "[32]",
    "data_format": "csv",
    "data_root": "../data/MicroVideo1.7M/",
    "dataset_id": "microvideo1.7m_x1_0d855fe6",
    "debug_mode": "False",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'name': 'clicked_items', 'padding': 'pre', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'clicked_categories', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'clicked_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'clicked_categories'}]",
    "gpu": "2",
    "group_id": "group_id",
    "hidden_activations": "relu",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "xDeepFM",
    "model_id": "xDeepFM_microvideo1.7m_x1_011_7df31553",
    "model_root": "./checkpoints/xDeepFM_microvideo1.7m_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2022",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "train_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/train.csv",
    "valid_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "verbose": "1"
}
2022-08-19 19:00:45,614 P61093 INFO Set up feature processor...
2022-08-19 19:00:45,614 P61093 INFO Load feature_map from json: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/feature_map.json
2022-08-19 19:00:45,615 P61093 INFO Set column index...
2022-08-19 19:00:45,615 P61093 INFO Feature specs: {
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514}",
    "clicked_categories": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514, 'max_len': 100}",
    "clicked_items": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'max_len': 100}",
    "group_id": "{'type': 'meta'}",
    "item_id": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10987, 'vocab_size': 10988}"
}
2022-08-19 19:00:57,900 P61093 INFO Total number of parameters: 5155638.
2022-08-19 19:00:57,901 P61093 INFO Loading data...
2022-08-19 19:00:57,901 P61093 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/train.h5
2022-08-19 19:01:15,634 P61093 INFO Train samples: total/8970309, blocks/1
2022-08-19 19:01:15,634 P61093 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/valid.h5
2022-08-19 19:01:23,865 P61093 INFO Validation samples: total/3767308, blocks/1
2022-08-19 19:01:23,865 P61093 INFO Loading train and validation data done.
2022-08-19 19:01:23,865 P61093 INFO Start training: 4381 batches/epoch
2022-08-19 19:01:23,865 P61093 INFO ************ Epoch=1 start ************
2022-08-19 19:29:11,769 P61093 INFO [Metrics] AUC: 0.717218 - gAUC: 0.672458
2022-08-19 19:29:11,783 P61093 INFO Save best model: monitor(max): 1.389676
2022-08-19 19:29:14,051 P61093 INFO --- 4381/4381 batches finished ---
2022-08-19 19:29:14,159 P61093 INFO Train loss: 0.470496
2022-08-19 19:29:14,159 P61093 INFO ************ Epoch=1 end ************
2022-08-19 19:57:02,364 P61093 INFO [Metrics] AUC: 0.720322 - gAUC: 0.674046
2022-08-19 19:57:02,416 P61093 INFO Save best model: monitor(max): 1.394368
2022-08-19 19:57:05,502 P61093 INFO --- 4381/4381 batches finished ---
2022-08-19 19:57:05,594 P61093 INFO Train loss: 0.451511
2022-08-19 19:57:05,594 P61093 INFO ************ Epoch=2 end ************
2022-08-19 20:25:04,432 P61093 INFO [Metrics] AUC: 0.722881 - gAUC: 0.675812
2022-08-19 20:25:04,442 P61093 INFO Save best model: monitor(max): 1.398693
2022-08-19 20:25:07,605 P61093 INFO --- 4381/4381 batches finished ---
2022-08-19 20:25:07,773 P61093 INFO Train loss: 0.449168
2022-08-19 20:25:07,773 P61093 INFO ************ Epoch=3 end ************
2022-08-19 20:53:07,171 P61093 INFO [Metrics] AUC: 0.723800 - gAUC: 0.676755
2022-08-19 20:53:07,179 P61093 INFO Save best model: monitor(max): 1.400556
2022-08-19 20:53:10,384 P61093 INFO --- 4381/4381 batches finished ---
2022-08-19 20:53:10,480 P61093 INFO Train loss: 0.447525
2022-08-19 20:53:10,480 P61093 INFO ************ Epoch=4 end ************
2022-08-19 21:21:18,052 P61093 INFO [Metrics] AUC: 0.723463 - gAUC: 0.677116
2022-08-19 21:21:18,071 P61093 INFO Save best model: monitor(max): 1.400579
2022-08-19 21:21:20,806 P61093 INFO --- 4381/4381 batches finished ---
2022-08-19 21:21:20,917 P61093 INFO Train loss: 0.446404
2022-08-19 21:21:20,917 P61093 INFO ************ Epoch=5 end ************
2022-08-19 21:49:27,375 P61093 INFO [Metrics] AUC: 0.724878 - gAUC: 0.678520
2022-08-19 21:49:27,390 P61093 INFO Save best model: monitor(max): 1.403397
2022-08-19 21:49:30,132 P61093 INFO --- 4381/4381 batches finished ---
2022-08-19 21:49:30,284 P61093 INFO Train loss: 0.445572
2022-08-19 21:49:30,284 P61093 INFO ************ Epoch=6 end ************
2022-08-19 22:17:41,151 P61093 INFO [Metrics] AUC: 0.725060 - gAUC: 0.678541
2022-08-19 22:17:41,156 P61093 INFO Save best model: monitor(max): 1.403601
2022-08-19 22:17:43,906 P61093 INFO --- 4381/4381 batches finished ---
2022-08-19 22:17:44,047 P61093 INFO Train loss: 0.444760
2022-08-19 22:17:44,048 P61093 INFO ************ Epoch=7 end ************
2022-08-19 22:45:48,727 P61093 INFO [Metrics] AUC: 0.726108 - gAUC: 0.678931
2022-08-19 22:45:48,736 P61093 INFO Save best model: monitor(max): 1.405039
2022-08-19 22:45:51,165 P61093 INFO --- 4381/4381 batches finished ---
2022-08-19 22:45:51,292 P61093 INFO Train loss: 0.444112
2022-08-19 22:45:51,292 P61093 INFO ************ Epoch=8 end ************
2022-08-19 23:13:55,905 P61093 INFO [Metrics] AUC: 0.726290 - gAUC: 0.679654
2022-08-19 23:13:55,911 P61093 INFO Save best model: monitor(max): 1.405944
2022-08-19 23:13:58,866 P61093 INFO --- 4381/4381 batches finished ---
2022-08-19 23:13:59,003 P61093 INFO Train loss: 0.443552
2022-08-19 23:13:59,004 P61093 INFO ************ Epoch=9 end ************
2022-08-19 23:41:58,471 P61093 INFO [Metrics] AUC: 0.726274 - gAUC: 0.679857
2022-08-19 23:41:58,476 P61093 INFO Save best model: monitor(max): 1.406131
2022-08-19 23:42:00,977 P61093 INFO --- 4381/4381 batches finished ---
2022-08-19 23:42:01,199 P61093 INFO Train loss: 0.442935
2022-08-19 23:42:01,200 P61093 INFO ************ Epoch=10 end ************
2022-08-20 00:10:08,931 P61093 INFO [Metrics] AUC: 0.726541 - gAUC: 0.680454
2022-08-20 00:10:08,963 P61093 INFO Save best model: monitor(max): 1.406995
2022-08-20 00:10:12,040 P61093 INFO --- 4381/4381 batches finished ---
2022-08-20 00:10:12,155 P61093 INFO Train loss: 0.442457
2022-08-20 00:10:12,156 P61093 INFO ************ Epoch=11 end ************
2022-08-20 00:38:21,191 P61093 INFO [Metrics] AUC: 0.726566 - gAUC: 0.680809
2022-08-20 00:38:21,197 P61093 INFO Save best model: monitor(max): 1.407375
2022-08-20 00:38:23,527 P61093 INFO --- 4381/4381 batches finished ---
2022-08-20 00:38:23,611 P61093 INFO Train loss: 0.441972
2022-08-20 00:38:23,611 P61093 INFO ************ Epoch=12 end ************
2022-08-20 01:03:14,260 P61093 INFO [Metrics] AUC: 0.726508 - gAUC: 0.679881
2022-08-20 01:03:14,264 P61093 INFO Monitor(max) STOP: 1.406389 !
2022-08-20 01:03:14,264 P61093 INFO Reduce learning rate on plateau: 0.000050
2022-08-20 01:03:14,264 P61093 INFO --- 4381/4381 batches finished ---
2022-08-20 01:03:14,383 P61093 INFO Train loss: 0.441561
2022-08-20 01:03:14,383 P61093 INFO ************ Epoch=13 end ************
2022-08-20 01:25:03,981 P61093 INFO [Metrics] AUC: 0.735064 - gAUC: 0.687504
2022-08-20 01:25:03,986 P61093 INFO Save best model: monitor(max): 1.422568
2022-08-20 01:25:06,327 P61093 INFO --- 4381/4381 batches finished ---
2022-08-20 01:25:06,429 P61093 INFO Train loss: 0.422943
2022-08-20 01:25:06,429 P61093 INFO ************ Epoch=14 end ************
2022-08-20 01:46:10,948 P61093 INFO [Metrics] AUC: 0.736228 - gAUC: 0.688869
2022-08-20 01:46:10,958 P61093 INFO Save best model: monitor(max): 1.425097
2022-08-20 01:46:13,121 P61093 INFO --- 4381/4381 batches finished ---
2022-08-20 01:46:13,237 P61093 INFO Train loss: 0.412733
2022-08-20 01:46:13,237 P61093 INFO ************ Epoch=15 end ************
2022-08-20 02:07:30,814 P61093 INFO [Metrics] AUC: 0.734942 - gAUC: 0.687273
2022-08-20 02:07:30,822 P61093 INFO Monitor(max) STOP: 1.422215 !
2022-08-20 02:07:30,822 P61093 INFO Reduce learning rate on plateau: 0.000005
2022-08-20 02:07:30,823 P61093 INFO --- 4381/4381 batches finished ---
2022-08-20 02:07:30,922 P61093 INFO Train loss: 0.406438
2022-08-20 02:07:30,922 P61093 INFO ************ Epoch=16 end ************
2022-08-20 02:25:35,843 P61093 INFO [Metrics] AUC: 0.734463 - gAUC: 0.686676
2022-08-20 02:25:35,859 P61093 INFO Monitor(max) STOP: 1.421138 !
2022-08-20 02:25:35,859 P61093 INFO Reduce learning rate on plateau: 0.000001
2022-08-20 02:25:35,859 P61093 INFO ********* Epoch==17 early stop *********
2022-08-20 02:25:35,860 P61093 INFO --- 4381/4381 batches finished ---
2022-08-20 02:25:35,959 P61093 INFO Train loss: 0.392227
2022-08-20 02:25:35,959 P61093 INFO Training finished.
2022-08-20 02:25:35,959 P61093 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/xDeepFM_microvideo1.7m_x1/microvideo1.7m_x1_0d855fe6/xDeepFM_microvideo1.7m_x1_011_7df31553.model
2022-08-20 02:25:37,122 P61093 INFO ****** Validation evaluation ******
2022-08-20 02:30:43,215 P61093 INFO [Metrics] gAUC: 0.688869 - AUC: 0.736228 - logloss: 0.412212
2022-08-20 02:30:43,351 P61093 INFO ******** Test evaluation ********
2022-08-20 02:30:43,351 P61093 INFO Loading data...
2022-08-20 02:30:43,351 P61093 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/test.h5
2022-08-20 02:30:48,348 P61093 INFO Test samples: total/3767308, blocks/1
2022-08-20 02:30:48,348 P61093 INFO Loading test data done.
2022-08-20 02:35:38,713 P61093 INFO [Metrics] gAUC: 0.688869 - AUC: 0.736228 - logloss: 0.412212

```
