## DeepFM_microvideo1.7m_x1

A hands-on guide to run the DeepFM model on the MicroVideo1.7M_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)


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
Please refer to the BARS dataset [MicroVideo1.7M_x1](https://github.com/openbenchmark/BARS/blob/main/datasets/MicroVideo1.7M#MicroVideo17M_x1) to get data ready.

### Code

We use the [DeepFM](https://github.com/xue-pai/FuxiCTR/blob/v2.0.1/model_zoo/DeepFM) model code from [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/MicroVideo1.7M/MicroVideo1.7M_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DeepFM_microvideo1.7m_x1_tuner_config_04](./DeepFM_microvideo1.7m_x1_tuner_config_04). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DeepFM
    nohup python run_expid.py --config XXX/benchmarks/DeepFM/DeepFM_microvideo1.7m_x1_tuner_config_04 --expid DeepFM_microvideo1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.685246 | 0.733670 | 0.413181  |


### Logs
```python
2022-08-19 14:36:36,451 P59997 INFO Params: {
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
    "model": "DeepFM",
    "model_id": "DeepFM_microvideo1.7m_x1_023_bda67d29",
    "model_root": "./checkpoints/DeepFM_microvideo1.7m_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.3",
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
2022-08-19 14:36:36,451 P59997 INFO Set up feature processor...
2022-08-19 14:36:36,451 P59997 INFO Load feature_map from json: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/feature_map.json
2022-08-19 14:36:36,452 P59997 INFO Set column index...
2022-08-19 14:36:36,452 P59997 INFO Feature specs: {
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514}",
    "clicked_categories": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514, 'max_len': 100}",
    "clicked_items": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'max_len': 100}",
    "group_id": "{'type': 'meta'}",
    "item_id": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10987, 'vocab_size': 10988}"
}
2022-08-19 14:36:48,213 P59997 INFO Total number of parameters: 5154774.
2022-08-19 14:36:48,214 P59997 INFO Loading data...
2022-08-19 14:36:48,214 P59997 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/train.h5
2022-08-19 14:37:12,241 P59997 INFO Train samples: total/8970309, blocks/1
2022-08-19 14:37:12,241 P59997 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/valid.h5
2022-08-19 14:37:19,235 P59997 INFO Validation samples: total/3767308, blocks/1
2022-08-19 14:37:19,235 P59997 INFO Loading train and validation data done.
2022-08-19 14:37:19,235 P59997 INFO Start training: 4381 batches/epoch
2022-08-19 14:37:19,236 P59997 INFO ************ Epoch=1 start ************
2022-08-19 15:06:07,208 P59997 INFO [Metrics] AUC: 0.713301 - gAUC: 0.668283
2022-08-19 15:06:07,239 P59997 INFO Save best model: monitor(max): 1.381583
2022-08-19 15:06:10,674 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 15:06:10,820 P59997 INFO Train loss: 0.482776
2022-08-19 15:06:10,820 P59997 INFO ************ Epoch=1 end ************
2022-08-19 15:34:50,015 P59997 INFO [Metrics] AUC: 0.717891 - gAUC: 0.672873
2022-08-19 15:34:50,142 P59997 INFO Save best model: monitor(max): 1.390764
2022-08-19 15:34:52,792 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 15:34:52,863 P59997 INFO Train loss: 0.447675
2022-08-19 15:34:52,864 P59997 INFO ************ Epoch=2 end ************
2022-08-19 16:03:37,706 P59997 INFO [Metrics] AUC: 0.721154 - gAUC: 0.675541
2022-08-19 16:03:37,733 P59997 INFO Save best model: monitor(max): 1.396695
2022-08-19 16:03:40,529 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 16:03:40,659 P59997 INFO Train loss: 0.441851
2022-08-19 16:03:40,659 P59997 INFO ************ Epoch=3 end ************
2022-08-19 16:32:19,581 P59997 INFO [Metrics] AUC: 0.721465 - gAUC: 0.675985
2022-08-19 16:32:19,615 P59997 INFO Save best model: monitor(max): 1.397450
2022-08-19 16:32:22,321 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 16:32:22,433 P59997 INFO Train loss: 0.439503
2022-08-19 16:32:22,433 P59997 INFO ************ Epoch=4 end ************
2022-08-19 17:00:58,150 P59997 INFO [Metrics] AUC: 0.723604 - gAUC: 0.677759
2022-08-19 17:00:58,161 P59997 INFO Save best model: monitor(max): 1.401363
2022-08-19 17:01:00,966 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 17:01:01,160 P59997 INFO Train loss: 0.438518
2022-08-19 17:01:01,160 P59997 INFO ************ Epoch=5 end ************
2022-08-19 17:29:51,742 P59997 INFO [Metrics] AUC: 0.723895 - gAUC: 0.677806
2022-08-19 17:29:51,755 P59997 INFO Save best model: monitor(max): 1.401701
2022-08-19 17:29:54,600 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 17:29:54,724 P59997 INFO Train loss: 0.437927
2022-08-19 17:29:54,725 P59997 INFO ************ Epoch=6 end ************
2022-08-19 17:58:27,242 P59997 INFO [Metrics] AUC: 0.724686 - gAUC: 0.678200
2022-08-19 17:58:27,262 P59997 INFO Save best model: monitor(max): 1.402886
2022-08-19 17:58:30,109 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 17:58:30,351 P59997 INFO Train loss: 0.437476
2022-08-19 17:58:30,351 P59997 INFO ************ Epoch=7 end ************
2022-08-19 18:27:09,976 P59997 INFO [Metrics] AUC: 0.725712 - gAUC: 0.679694
2022-08-19 18:27:09,987 P59997 INFO Save best model: monitor(max): 1.405406
2022-08-19 18:27:12,445 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 18:27:12,635 P59997 INFO Train loss: 0.436933
2022-08-19 18:27:12,635 P59997 INFO ************ Epoch=8 end ************
2022-08-19 18:55:48,311 P59997 INFO [Metrics] AUC: 0.725736 - gAUC: 0.678781
2022-08-19 18:55:48,329 P59997 INFO Monitor(max) STOP: 1.404517 !
2022-08-19 18:55:48,329 P59997 INFO Reduce learning rate on plateau: 0.000050
2022-08-19 18:55:48,330 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 18:55:48,476 P59997 INFO Train loss: 0.436546
2022-08-19 18:55:48,476 P59997 INFO ************ Epoch=9 end ************
2022-08-19 19:24:20,670 P59997 INFO [Metrics] AUC: 0.731630 - gAUC: 0.683891
2022-08-19 19:24:20,749 P59997 INFO Save best model: monitor(max): 1.415521
2022-08-19 19:24:23,099 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 19:24:23,203 P59997 INFO Train loss: 0.425186
2022-08-19 19:24:23,203 P59997 INFO ************ Epoch=10 end ************
2022-08-19 19:51:16,899 P59997 INFO [Metrics] AUC: 0.732895 - gAUC: 0.684786
2022-08-19 19:51:16,912 P59997 INFO Save best model: monitor(max): 1.417681
2022-08-19 19:51:19,186 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 19:51:19,307 P59997 INFO Train loss: 0.420710
2022-08-19 19:51:19,307 P59997 INFO ************ Epoch=11 end ************
2022-08-19 20:15:47,331 P59997 INFO [Metrics] AUC: 0.733244 - gAUC: 0.684994
2022-08-19 20:15:47,344 P59997 INFO Save best model: monitor(max): 1.418238
2022-08-19 20:15:49,875 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 20:15:49,974 P59997 INFO Train loss: 0.418955
2022-08-19 20:15:49,974 P59997 INFO ************ Epoch=12 end ************
2022-08-19 20:39:59,521 P59997 INFO [Metrics] AUC: 0.733254 - gAUC: 0.685101
2022-08-19 20:39:59,536 P59997 INFO Save best model: monitor(max): 1.418355
2022-08-19 20:40:01,879 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 20:40:01,981 P59997 INFO Train loss: 0.417701
2022-08-19 20:40:01,981 P59997 INFO ************ Epoch=13 end ************
2022-08-19 21:02:00,989 P59997 INFO [Metrics] AUC: 0.733407 - gAUC: 0.685089
2022-08-19 21:02:01,001 P59997 INFO Save best model: monitor(max): 1.418496
2022-08-19 21:02:03,108 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 21:02:03,203 P59997 INFO Train loss: 0.416665
2022-08-19 21:02:03,203 P59997 INFO ************ Epoch=14 end ************
2022-08-19 21:17:11,504 P59997 INFO [Metrics] AUC: 0.733562 - gAUC: 0.685144
2022-08-19 21:17:11,510 P59997 INFO Save best model: monitor(max): 1.418706
2022-08-19 21:17:13,437 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 21:17:13,526 P59997 INFO Train loss: 0.415755
2022-08-19 21:17:13,526 P59997 INFO ************ Epoch=15 end ************
2022-08-19 21:26:15,587 P59997 INFO [Metrics] AUC: 0.733209 - gAUC: 0.684909
2022-08-19 21:26:15,594 P59997 INFO Monitor(max) STOP: 1.418118 !
2022-08-19 21:26:15,594 P59997 INFO Reduce learning rate on plateau: 0.000005
2022-08-19 21:26:15,594 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 21:26:15,669 P59997 INFO Train loss: 0.414894
2022-08-19 21:26:15,670 P59997 INFO ************ Epoch=16 end ************
2022-08-19 21:35:16,502 P59997 INFO [Metrics] AUC: 0.733670 - gAUC: 0.685246
2022-08-19 21:35:16,510 P59997 INFO Save best model: monitor(max): 1.418917
2022-08-19 21:35:18,429 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 21:35:18,506 P59997 INFO Train loss: 0.410478
2022-08-19 21:35:18,507 P59997 INFO ************ Epoch=17 end ************
2022-08-19 21:41:46,981 P59997 INFO [Metrics] AUC: 0.733606 - gAUC: 0.685182
2022-08-19 21:41:46,988 P59997 INFO Monitor(max) STOP: 1.418788 !
2022-08-19 21:41:46,988 P59997 INFO Reduce learning rate on plateau: 0.000001
2022-08-19 21:41:46,988 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 21:41:47,085 P59997 INFO Train loss: 0.409878
2022-08-19 21:41:47,085 P59997 INFO ************ Epoch=18 end ************
2022-08-19 21:45:43,769 P59997 INFO [Metrics] AUC: 0.733593 - gAUC: 0.685102
2022-08-19 21:45:43,775 P59997 INFO Monitor(max) STOP: 1.418695 !
2022-08-19 21:45:43,775 P59997 INFO Reduce learning rate on plateau: 0.000001
2022-08-19 21:45:43,775 P59997 INFO ********* Epoch==19 early stop *********
2022-08-19 21:45:43,776 P59997 INFO --- 4381/4381 batches finished ---
2022-08-19 21:45:43,838 P59997 INFO Train loss: 0.409177
2022-08-19 21:45:43,838 P59997 INFO Training finished.
2022-08-19 21:45:43,838 P59997 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DeepFM_microvideo1.7m_x1/microvideo1.7m_x1_0d855fe6/DeepFM_microvideo1.7m_x1_023_bda67d29.model
2022-08-19 21:45:44,782 P59997 INFO ****** Validation evaluation ******
2022-08-19 21:46:40,890 P59997 INFO [Metrics] gAUC: 0.685246 - AUC: 0.733670 - logloss: 0.413181
2022-08-19 21:46:41,075 P59997 INFO ******** Test evaluation ********
2022-08-19 21:46:41,075 P59997 INFO Loading data...
2022-08-19 21:46:41,075 P59997 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/test.h5
2022-08-19 21:46:45,629 P59997 INFO Test samples: total/3767308, blocks/1
2022-08-19 21:46:45,629 P59997 INFO Loading test data done.
2022-08-19 21:47:54,642 P59997 INFO [Metrics] gAUC: 0.685246 - AUC: 0.733670 - logloss: 0.413181

```
