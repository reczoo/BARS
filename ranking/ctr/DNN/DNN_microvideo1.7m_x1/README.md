## DNN_microvideo1.7m_x1

A hands-on guide to run the DNN model on the MicroVideo1.7M_x1 dataset.

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

We use the [DNN](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/DNN) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/MicroVideo1.7M/MicroVideo1.7M_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DNN_microvideo1.7m_x1_tuner_config_01](./DNN_microvideo1.7m_x1_tuner_config_01). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DNN
    nohup python run_expid.py --config XXX/benchmarks/DNN/DNN_microvideo1.7m_x1_tuner_config_01 --expid DNN_microvideo1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.685716 | 0.734555 | 0.411947  |


### Logs
```python
2022-11-17 16:58:32,788 P19219 INFO Params: {
    "batch_norm": "True",
    "batch_size": "2048",
    "data_format": "csv",
    "data_root": "../data/MicroVideo1.7M/",
    "dataset_id": "microvideo1.7m_x1_0d855fe6",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.0005",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'name': 'clicked_items', 'padding': 'pre', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'clicked_categories', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'clicked_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'clicked_categories'}]",
    "gpu": "6",
    "group_id": "group_id",
    "hidden_activations": "relu",
    "hidden_units": "[1024, 512, 256]",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DNN",
    "model_id": "DNN_microvideo1.7m_x1_020_eb07caea",
    "model_root": "./checkpoints/DNN_microvideo1.7m_x1/",
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
    "verbose": "0"
}
2022-11-17 16:58:32,789 P19219 INFO Set up feature processor...
2022-11-17 16:58:32,789 P19219 INFO Load feature_map from json: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/feature_map.json
2022-11-17 16:58:32,789 P19219 INFO Set column index...
2022-11-17 16:58:32,789 P19219 INFO Feature specs: {
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514}",
    "clicked_categories": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514, 'max_len': 100}",
    "clicked_items": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'max_len': 100}",
    "group_id": "{'type': 'meta'}",
    "item_id": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10987, 'vocab_size': 10988}"
}
2022-11-17 16:58:40,791 P19219 INFO Total number of parameters: 1732993.
2022-11-17 16:58:40,791 P19219 INFO Loading data...
2022-11-17 16:58:40,791 P19219 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/train.h5
2022-11-17 16:58:53,385 P19219 INFO Train samples: total/8970309, blocks/1
2022-11-17 16:58:53,385 P19219 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/valid.h5
2022-11-17 16:58:58,084 P19219 INFO Validation samples: total/3767308, blocks/1
2022-11-17 16:58:58,084 P19219 INFO Loading train and validation data done.
2022-11-17 16:58:58,084 P19219 INFO Start training: 4381 batches/epoch
2022-11-17 16:58:58,084 P19219 INFO ************ Epoch=1 start ************
2022-11-17 17:19:19,764 P19219 INFO [Metrics] AUC: 0.716399 - gAUC: 0.670146
2022-11-17 17:19:19,782 P19219 INFO Save best model: monitor(max): 1.386545
2022-11-17 17:19:21,674 P19219 INFO --- 4381/4381 batches finished ---
2022-11-17 17:19:21,749 P19219 INFO Train loss: 0.462000
2022-11-17 17:19:21,749 P19219 INFO ************ Epoch=1 end ************
2022-11-17 17:38:40,237 P19219 INFO [Metrics] AUC: 0.721406 - gAUC: 0.673064
2022-11-17 17:38:40,254 P19219 INFO Save best model: monitor(max): 1.394470
2022-11-17 17:38:42,293 P19219 INFO --- 4381/4381 batches finished ---
2022-11-17 17:38:42,403 P19219 INFO Train loss: 0.442982
2022-11-17 17:38:42,403 P19219 INFO ************ Epoch=2 end ************
2022-11-17 17:54:59,077 P19219 INFO [Metrics] AUC: 0.723169 - gAUC: 0.673627
2022-11-17 17:54:59,084 P19219 INFO Save best model: monitor(max): 1.396797
2022-11-17 17:55:01,169 P19219 INFO --- 4381/4381 batches finished ---
2022-11-17 17:55:01,262 P19219 INFO Train loss: 0.438898
2022-11-17 17:55:01,262 P19219 INFO ************ Epoch=3 end ************
2022-11-17 18:11:23,985 P19219 INFO [Metrics] AUC: 0.724782 - gAUC: 0.676639
2022-11-17 18:11:23,990 P19219 INFO Save best model: monitor(max): 1.401421
2022-11-17 18:11:25,983 P19219 INFO --- 4381/4381 batches finished ---
2022-11-17 18:11:26,072 P19219 INFO Train loss: 0.437472
2022-11-17 18:11:26,072 P19219 INFO ************ Epoch=4 end ************
2022-11-17 18:27:30,158 P19219 INFO [Metrics] AUC: 0.726003 - gAUC: 0.677563
2022-11-17 18:27:30,166 P19219 INFO Save best model: monitor(max): 1.403566
2022-11-17 18:27:32,131 P19219 INFO --- 4381/4381 batches finished ---
2022-11-17 18:27:32,240 P19219 INFO Train loss: 0.436427
2022-11-17 18:27:32,241 P19219 INFO ************ Epoch=5 end ************
2022-11-17 18:39:08,048 P19219 INFO [Metrics] AUC: 0.727155 - gAUC: 0.678594
2022-11-17 18:39:08,087 P19219 INFO Save best model: monitor(max): 1.405749
2022-11-17 18:39:10,039 P19219 INFO --- 4381/4381 batches finished ---
2022-11-17 18:39:10,124 P19219 INFO Train loss: 0.435621
2022-11-17 18:39:10,124 P19219 INFO ************ Epoch=6 end ************
2022-11-17 18:50:29,409 P19219 INFO [Metrics] AUC: 0.727332 - gAUC: 0.678742
2022-11-17 18:50:29,417 P19219 INFO Save best model: monitor(max): 1.406073
2022-11-17 18:50:31,348 P19219 INFO --- 4381/4381 batches finished ---
2022-11-17 18:50:31,447 P19219 INFO Train loss: 0.434965
2022-11-17 18:50:31,447 P19219 INFO ************ Epoch=7 end ************
2022-11-17 19:01:54,442 P19219 INFO [Metrics] AUC: 0.727553 - gAUC: 0.679312
2022-11-17 19:01:54,449 P19219 INFO Save best model: monitor(max): 1.406865
2022-11-17 19:01:56,425 P19219 INFO --- 4381/4381 batches finished ---
2022-11-17 19:01:56,504 P19219 INFO Train loss: 0.434295
2022-11-17 19:01:56,504 P19219 INFO ************ Epoch=8 end ************
2022-11-17 19:13:22,548 P19219 INFO [Metrics] AUC: 0.727801 - gAUC: 0.679893
2022-11-17 19:13:22,554 P19219 INFO Save best model: monitor(max): 1.407694
2022-11-17 19:13:24,465 P19219 INFO --- 4381/4381 batches finished ---
2022-11-17 19:13:24,548 P19219 INFO Train loss: 0.433749
2022-11-17 19:13:24,548 P19219 INFO ************ Epoch=9 end ************
2022-11-17 19:24:11,469 P19219 INFO [Metrics] AUC: 0.726975 - gAUC: 0.678907
2022-11-17 19:24:11,475 P19219 INFO Monitor(max) STOP: 1.405882 !
2022-11-17 19:24:11,475 P19219 INFO Reduce learning rate on plateau: 0.000050
2022-11-17 19:24:11,475 P19219 INFO --- 4381/4381 batches finished ---
2022-11-17 19:24:11,564 P19219 INFO Train loss: 0.433265
2022-11-17 19:24:11,564 P19219 INFO ************ Epoch=10 end ************
2022-11-17 19:34:01,992 P19219 INFO [Metrics] AUC: 0.733202 - gAUC: 0.684699
2022-11-17 19:34:01,998 P19219 INFO Save best model: monitor(max): 1.417901
2022-11-17 19:34:03,935 P19219 INFO --- 4381/4381 batches finished ---
2022-11-17 19:34:04,019 P19219 INFO Train loss: 0.423161
2022-11-17 19:34:04,019 P19219 INFO ************ Epoch=11 end ************
2022-11-17 19:43:49,129 P19219 INFO [Metrics] AUC: 0.733845 - gAUC: 0.685028
2022-11-17 19:43:49,134 P19219 INFO Save best model: monitor(max): 1.418873
2022-11-17 19:43:51,033 P19219 INFO --- 4381/4381 batches finished ---
2022-11-17 19:43:51,112 P19219 INFO Train loss: 0.418719
2022-11-17 19:43:51,112 P19219 INFO ************ Epoch=12 end ************
2022-11-17 19:52:27,883 P19219 INFO [Metrics] AUC: 0.733984 - gAUC: 0.685049
2022-11-17 19:52:27,891 P19219 INFO Save best model: monitor(max): 1.419033
2022-11-17 19:52:29,648 P19219 INFO --- 4381/4381 batches finished ---
2022-11-17 19:52:29,738 P19219 INFO Train loss: 0.416830
2022-11-17 19:52:29,740 P19219 INFO ************ Epoch=13 end ************
2022-11-17 20:01:09,854 P19219 INFO [Metrics] AUC: 0.734555 - gAUC: 0.685716
2022-11-17 20:01:09,860 P19219 INFO Save best model: monitor(max): 1.420271
2022-11-17 20:01:11,691 P19219 INFO --- 4381/4381 batches finished ---
2022-11-17 20:01:11,757 P19219 INFO Train loss: 0.415276
2022-11-17 20:01:11,757 P19219 INFO ************ Epoch=14 end ************
2022-11-17 20:09:23,735 P19219 INFO [Metrics] AUC: 0.734372 - gAUC: 0.685452
2022-11-17 20:09:23,742 P19219 INFO Monitor(max) STOP: 1.419824 !
2022-11-17 20:09:23,743 P19219 INFO Reduce learning rate on plateau: 0.000005
2022-11-17 20:09:23,744 P19219 INFO --- 4381/4381 batches finished ---
2022-11-17 20:09:23,827 P19219 INFO Train loss: 0.414130
2022-11-17 20:09:23,828 P19219 INFO ************ Epoch=15 end ************
2022-11-17 20:16:42,119 P19219 INFO [Metrics] AUC: 0.734272 - gAUC: 0.685428
2022-11-17 20:16:42,126 P19219 INFO Monitor(max) STOP: 1.419700 !
2022-11-17 20:16:42,126 P19219 INFO Reduce learning rate on plateau: 0.000001
2022-11-17 20:16:42,126 P19219 INFO ********* Epoch==16 early stop *********
2022-11-17 20:16:42,127 P19219 INFO --- 4381/4381 batches finished ---
2022-11-17 20:16:42,233 P19219 INFO Train loss: 0.410253
2022-11-17 20:16:42,233 P19219 INFO Training finished.
2022-11-17 20:16:42,233 P19219 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DNN_microvideo1.7m_x1/microvideo1.7m_x1_0d855fe6/DNN_microvideo1.7m_x1_020_eb07caea.model
2022-11-17 20:16:42,916 P19219 INFO ****** Validation evaluation ******
2022-11-17 20:18:51,521 P19219 INFO [Metrics] gAUC: 0.685716 - AUC: 0.734555 - logloss: 0.411947
2022-11-17 20:18:51,651 P19219 INFO ******** Test evaluation ********
2022-11-17 20:18:51,652 P19219 INFO Loading data...
2022-11-17 20:18:51,652 P19219 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/test.h5
2022-11-17 20:18:56,302 P19219 INFO Test samples: total/3767308, blocks/1
2022-11-17 20:18:56,302 P19219 INFO Loading test data done.
2022-11-17 20:20:58,798 P19219 INFO [Metrics] gAUC: 0.685716 - AUC: 0.734555 - logloss: 0.411947

```
