## AutoInt_microvideo1.7m_x1

A hands-on guide to run the AutoInt model on the MicroVideo1.7M_x1 dataset.

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

We use the [AutoInt](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/AutoInt) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/MicroVideo1.7M/MicroVideo1.7M_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt+_microvideo1.7m_x1_tuner_config_08](./AutoInt+_microvideo1.7m_x1_tuner_config_08). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/AutoInt
    nohup python run_expid.py --config XXX/benchmarks/AutoInt/AutoInt+_microvideo1.7m_x1_tuner_config_08 --expid AutoInt_microvideo1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.684609 | 0.733822 | 0.413313  |


### Logs
```python
2022-08-17 18:22:39,779 P57261 INFO Params: {
    "attention_dim": "128",
    "attention_layers": "3",
    "batch_norm": "True",
    "batch_size": "2048",
    "data_format": "csv",
    "data_root": "../data/MicroVideo1.7M/",
    "dataset_id": "microvideo1.7m_x1_0d855fe6",
    "debug_mode": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'name': 'clicked_items', 'padding': 'pre', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'clicked_categories', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'clicked_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'clicked_categories'}]",
    "gpu": "4",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "layer_norm": "True",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AutoInt",
    "model_id": "AutoInt_microvideo1.7m_x1_029_f813da5f",
    "model_root": "./checkpoints/AutoInt_microvideo1.7m_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_heads": "2",
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
    "use_residual": "True",
    "use_scale": "False",
    "use_wide": "False",
    "valid_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "verbose": "1"
}
2022-08-17 18:22:39,780 P57261 INFO Set up feature processor...
2022-08-17 18:22:39,780 P57261 INFO Load feature_map from json: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/feature_map.json
2022-08-17 18:22:39,781 P57261 INFO Set column index...
2022-08-17 18:22:39,781 P57261 INFO Feature specs: {
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514}",
    "clicked_categories": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514, 'max_len': 100}",
    "clicked_items": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'max_len': 100}",
    "group_id": "{'type': 'meta'}",
    "item_id": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10987, 'vocab_size': 10988}"
}
2022-08-17 18:22:53,741 P57261 INFO Total number of parameters: 1865474.
2022-08-17 18:22:53,742 P57261 INFO Loading data...
2022-08-17 18:22:53,742 P57261 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/train.h5
2022-08-17 18:32:39,331 P57261 INFO Train samples: total/8970309, blocks/1
2022-08-17 18:32:39,331 P57261 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/valid.h5
2022-08-17 18:33:08,957 P57261 INFO Validation samples: total/3767308, blocks/1
2022-08-17 18:33:09,021 P57261 INFO Loading train and validation data done.
2022-08-17 18:33:09,022 P57261 INFO Start training: 4381 batches/epoch
2022-08-17 18:33:09,022 P57261 INFO ************ Epoch=1 start ************
2022-08-17 19:08:33,166 P57261 INFO [Metrics] AUC: 0.715070 - gAUC: 0.669289
2022-08-17 19:08:33,196 P57261 INFO Save best model: monitor(max): 1.384359
2022-08-17 19:08:40,561 P57261 INFO --- 4381/4381 batches finished ---
2022-08-17 19:08:40,986 P57261 INFO Train loss: 0.469980
2022-08-17 19:08:40,986 P57261 INFO ************ Epoch=1 end ************
2022-08-17 19:43:31,910 P57261 INFO [Metrics] AUC: 0.720693 - gAUC: 0.671380
2022-08-17 19:43:31,927 P57261 INFO Save best model: monitor(max): 1.392072
2022-08-17 19:43:44,596 P57261 INFO --- 4381/4381 batches finished ---
2022-08-17 19:43:44,990 P57261 INFO Train loss: 0.444076
2022-08-17 19:43:44,990 P57261 INFO ************ Epoch=2 end ************
2022-08-17 20:18:42,105 P57261 INFO [Metrics] AUC: 0.721349 - gAUC: 0.672441
2022-08-17 20:18:42,154 P57261 INFO Save best model: monitor(max): 1.393789
2022-08-17 20:20:01,424 P57261 INFO --- 4381/4381 batches finished ---
2022-08-17 20:20:01,858 P57261 INFO Train loss: 0.441814
2022-08-17 20:20:01,858 P57261 INFO ************ Epoch=3 end ************
2022-08-17 20:54:53,531 P57261 INFO [Metrics] AUC: 0.722729 - gAUC: 0.673458
2022-08-17 20:54:53,552 P57261 INFO Save best model: monitor(max): 1.396187
2022-08-17 20:55:19,990 P57261 INFO --- 4381/4381 batches finished ---
2022-08-17 20:55:20,648 P57261 INFO Train loss: 0.440998
2022-08-17 20:55:20,648 P57261 INFO ************ Epoch=4 end ************
2022-08-17 21:30:16,637 P57261 INFO [Metrics] AUC: 0.723825 - gAUC: 0.675988
2022-08-17 21:30:16,733 P57261 INFO Save best model: monitor(max): 1.399813
2022-08-17 21:30:22,762 P57261 INFO --- 4381/4381 batches finished ---
2022-08-17 21:30:23,342 P57261 INFO Train loss: 0.440125
2022-08-17 21:30:23,343 P57261 INFO ************ Epoch=5 end ************
2022-08-17 22:05:11,338 P57261 INFO [Metrics] AUC: 0.725223 - gAUC: 0.676843
2022-08-17 22:05:11,401 P57261 INFO Save best model: monitor(max): 1.402067
2022-08-17 22:05:24,547 P57261 INFO --- 4381/4381 batches finished ---
2022-08-17 22:05:25,007 P57261 INFO Train loss: 0.439273
2022-08-17 22:05:25,015 P57261 INFO ************ Epoch=6 end ************
2022-08-17 22:40:20,480 P57261 INFO [Metrics] AUC: 0.723360 - gAUC: 0.676409
2022-08-17 22:40:20,561 P57261 INFO Monitor(max) STOP: 1.399769 !
2022-08-17 22:40:20,562 P57261 INFO Reduce learning rate on plateau: 0.000050
2022-08-17 22:40:20,562 P57261 INFO --- 4381/4381 batches finished ---
2022-08-17 22:40:21,005 P57261 INFO Train loss: 0.438754
2022-08-17 22:40:21,005 P57261 INFO ************ Epoch=7 end ************
2022-08-17 23:15:06,449 P57261 INFO [Metrics] AUC: 0.731802 - gAUC: 0.682904
2022-08-17 23:15:06,474 P57261 INFO Save best model: monitor(max): 1.414706
2022-08-17 23:15:18,437 P57261 INFO --- 4381/4381 batches finished ---
2022-08-17 23:15:19,046 P57261 INFO Train loss: 0.427466
2022-08-17 23:15:19,047 P57261 INFO ************ Epoch=8 end ************
2022-08-17 23:50:14,739 P57261 INFO [Metrics] AUC: 0.732541 - gAUC: 0.683421
2022-08-17 23:50:14,772 P57261 INFO Save best model: monitor(max): 1.415962
2022-08-17 23:50:27,812 P57261 INFO --- 4381/4381 batches finished ---
2022-08-17 23:50:28,433 P57261 INFO Train loss: 0.423079
2022-08-17 23:50:28,433 P57261 INFO ************ Epoch=9 end ************
2022-08-18 00:25:16,523 P57261 INFO [Metrics] AUC: 0.733059 - gAUC: 0.683848
2022-08-18 00:25:16,565 P57261 INFO Save best model: monitor(max): 1.416907
2022-08-18 00:25:34,563 P57261 INFO --- 4381/4381 batches finished ---
2022-08-18 00:25:35,084 P57261 INFO Train loss: 0.421414
2022-08-18 00:25:35,084 P57261 INFO ************ Epoch=10 end ************
2022-08-18 01:00:19,790 P57261 INFO [Metrics] AUC: 0.733325 - gAUC: 0.684168
2022-08-18 01:00:19,810 P57261 INFO Save best model: monitor(max): 1.417493
2022-08-18 01:00:33,445 P57261 INFO --- 4381/4381 batches finished ---
2022-08-18 01:00:33,941 P57261 INFO Train loss: 0.420162
2022-08-18 01:00:33,941 P57261 INFO ************ Epoch=11 end ************
2022-08-18 01:35:05,046 P57261 INFO [Metrics] AUC: 0.733595 - gAUC: 0.684499
2022-08-18 01:35:05,057 P57261 INFO Save best model: monitor(max): 1.418094
2022-08-18 01:35:14,580 P57261 INFO --- 4381/4381 batches finished ---
2022-08-18 01:35:14,955 P57261 INFO Train loss: 0.419043
2022-08-18 01:35:14,956 P57261 INFO ************ Epoch=12 end ************
2022-08-18 02:09:50,929 P57261 INFO [Metrics] AUC: 0.733618 - gAUC: 0.684474
2022-08-18 02:09:50,936 P57261 INFO Monitor(max) STOP: 1.418093 !
2022-08-18 02:09:50,936 P57261 INFO Reduce learning rate on plateau: 0.000005
2022-08-18 02:09:50,937 P57261 INFO --- 4381/4381 batches finished ---
2022-08-18 02:09:51,312 P57261 INFO Train loss: 0.418121
2022-08-18 02:09:51,312 P57261 INFO ************ Epoch=13 end ************
2022-08-18 02:44:28,113 P57261 INFO [Metrics] AUC: 0.733724 - gAUC: 0.684594
2022-08-18 02:44:28,121 P57261 INFO Save best model: monitor(max): 1.418318
2022-08-18 02:44:32,112 P57261 INFO --- 4381/4381 batches finished ---
2022-08-18 02:44:32,497 P57261 INFO Train loss: 0.413599
2022-08-18 02:44:32,497 P57261 INFO ************ Epoch=14 end ************
2022-08-18 03:18:59,442 P57261 INFO [Metrics] AUC: 0.733685 - gAUC: 0.684479
2022-08-18 03:18:59,456 P57261 INFO Monitor(max) STOP: 1.418164 !
2022-08-18 03:18:59,456 P57261 INFO Reduce learning rate on plateau: 0.000001
2022-08-18 03:18:59,457 P57261 INFO --- 4381/4381 batches finished ---
2022-08-18 03:18:59,787 P57261 INFO Train loss: 0.412944
2022-08-18 03:18:59,787 P57261 INFO ************ Epoch=15 end ************
2022-08-18 03:50:24,665 P57261 INFO [Metrics] AUC: 0.733822 - gAUC: 0.684609
2022-08-18 03:50:24,672 P57261 INFO Save best model: monitor(max): 1.418431
2022-08-18 03:50:27,711 P57261 INFO --- 4381/4381 batches finished ---
2022-08-18 03:50:28,034 P57261 INFO Train loss: 0.412150
2022-08-18 03:50:28,034 P57261 INFO ************ Epoch=16 end ************
2022-08-18 04:17:47,068 P57261 INFO [Metrics] AUC: 0.733763 - gAUC: 0.684543
2022-08-18 04:17:47,087 P57261 INFO Monitor(max) STOP: 1.418305 !
2022-08-18 04:17:47,087 P57261 INFO Reduce learning rate on plateau: 0.000001
2022-08-18 04:17:47,088 P57261 INFO --- 4381/4381 batches finished ---
2022-08-18 04:17:47,514 P57261 INFO Train loss: 0.412029
2022-08-18 04:17:47,514 P57261 INFO ************ Epoch=17 end ************
2022-08-18 04:42:33,415 P57261 INFO [Metrics] AUC: 0.733773 - gAUC: 0.684525
2022-08-18 04:42:33,422 P57261 INFO Monitor(max) STOP: 1.418297 !
2022-08-18 04:42:33,422 P57261 INFO Reduce learning rate on plateau: 0.000001
2022-08-18 04:42:33,422 P57261 INFO ********* Epoch==18 early stop *********
2022-08-18 04:42:33,423 P57261 INFO --- 4381/4381 batches finished ---
2022-08-18 04:42:33,653 P57261 INFO Train loss: 0.411976
2022-08-18 04:42:33,653 P57261 INFO Training finished.
2022-08-18 04:42:33,653 P57261 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/AutoInt_microvideo1.7m_x1/microvideo1.7m_x1_0d855fe6/AutoInt_microvideo1.7m_x1_029_f813da5f.model
2022-08-18 04:42:34,410 P57261 INFO ****** Validation evaluation ******
2022-08-18 04:49:53,661 P57261 INFO [Metrics] gAUC: 0.684609 - AUC: 0.733822 - logloss: 0.413313
2022-08-18 04:49:54,594 P57261 INFO ******** Test evaluation ********
2022-08-18 04:49:54,595 P57261 INFO Loading data...
2022-08-18 04:49:54,595 P57261 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/test.h5
2022-08-18 04:50:00,277 P57261 INFO Test samples: total/3767308, blocks/1
2022-08-18 04:50:00,277 P57261 INFO Loading test data done.
2022-08-18 04:56:42,613 P57261 INFO [Metrics] gAUC: 0.684609 - AUC: 0.733822 - logloss: 0.413313

```
