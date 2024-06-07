## DIEN_kuaivideo_x1

A hands-on guide to run the DIEN model on the KuaiVideo_x1 dataset.

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
  fuxictr: 2.0.2

  ```

### Dataset
Please refer to [KuaiVideo_x1](https://github.com/reczoo/Datasets/tree/main/KuaiShou/KuaiVideo_x1) to get the dataset details.

### Code

We use the [DIEN](https://github.com/reczoo/FuxiCTR/blob/v2.0.2/model_zoo/DIEN) model code from [FuxiCTR-v2.0.2](https://github.com/reczoo/FuxiCTR/tree/v2.0.2) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.2.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.2
    ```

2. Create a data directory and put the downloaded data files in `../data/KuaiShou/KuaiVideo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DIEN_kuaivideo_x1_tuner_config_02](./DIEN_kuaivideo_x1_tuner_config_02). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DIEN
    nohup python run_expid.py --config XXX/benchmarks/DIEN/DIEN_kuaivideo_x1_tuner_config_02 --expid DIEN_kuaivideo_x1_009_657021a9 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.671148 | 0.750372 | 0.433049  |


### Logs
```python
2023-05-13 17:34:04,650 P80930 INFO Params: {
    "attention_activation": "Dice",
    "attention_dropout": "0.3",
    "attention_hidden_units": "[256, 256]",
    "attention_type": "din_attention",
    "aux_activation": "ReLU",
    "aux_hidden_units": "[100, 50]",
    "aux_loss_alpha": "0",
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/KuaiShou/",
    "dataset_id": "kuaivideo_x1_60f6c91a",
    "debug_mode": "False",
    "dien_neg_seq_field": "[]",
    "dien_sequence_field": "[('pos_items', 'pos_items_emb'), ('neg_items', 'neg_items_emb')]",
    "dien_target_field": "[('item_id', 'item_emb'), ('item_id', 'item_emb')]",
    "dnn_activations": "ReLU",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.0001",
    "enable_sum_pooling": "False",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'min_categr_count': 1, 'name': 'item_emb', 'preprocess': 'copy_from(item_id)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'pos_items', 'padding': 'post', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'neg_items', 'padding': 'post', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'pos_items_emb', 'padding': 'post', 'preprocess': 'copy_from(pos_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'neg_items_emb', 'padding': 'post', 'preprocess': 'copy_from(neg_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}]",
    "feature_config": "None",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_emb'}, {'feature_encoder': None, 'name': 'pos_items'}, {'feature_encoder': None, 'name': 'neg_items'}, {'feature_encoder': ['nn.Linear(64, 64, bias=False)'], 'name': 'pos_items_emb'}, {'feature_encoder': ['nn.Linear(64, 64, bias=False)'], 'name': 'neg_items_emb'}]",
    "gpu": "3",
    "group_id": "group_id",
    "gru_type": "AUGRU",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "DIEN",
    "model_id": "DIEN_kuaivideo_x1_009_657021a9",
    "model_root": "./checkpoints/DIEN_kuaivideo_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "20222023",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "train_data": "../data/KuaiShou/KuaiVideo_x1/train.csv",
    "use_attention_softmax": "True",
    "use_features": "None",
    "valid_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "verbose": "1"
}
2023-05-13 17:34:04,651 P80930 INFO Set up feature processor...
2023-05-13 17:34:04,651 P80930 WARNING Skip rebuilding ../data/KuaiShou/kuaivideo_x1_60f6c91a/feature_map.json. Please delete it manually if rebuilding is required.
2023-05-13 17:34:04,651 P80930 INFO Load feature_map from json: ../data/KuaiShou/kuaivideo_x1_60f6c91a/feature_map.json
2023-05-13 17:34:04,651 P80930 INFO Set column index...
2023-05-13 17:34:04,652 P80930 INFO Feature specs: {
    "group_id": "{'type': 'meta'}",
    "item_emb": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406}",
    "neg_items": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "neg_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "pos_items": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "pos_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10001, 'vocab_size': 10002}"
}
2023-05-13 17:34:11,709 P80930 INFO Total number of parameters: 43034627.
2023-05-13 17:34:11,709 P80930 INFO Loading data...
2023-05-13 17:34:11,709 P80930 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_60f6c91a/train.h5
2023-05-13 17:34:36,104 P80930 INFO Train samples: total/10931092, blocks/1
2023-05-13 17:34:36,104 P80930 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_60f6c91a/valid.h5
2023-05-13 17:34:43,070 P80930 INFO Validation samples: total/2730291, blocks/1
2023-05-13 17:34:43,070 P80930 INFO Loading train and validation data done.
2023-05-13 17:34:43,070 P80930 INFO Start training: 2669 batches/epoch
2023-05-13 17:34:43,070 P80930 INFO ************ Epoch=1 start ************
2023-05-13 18:17:12,714 P80930 INFO Train loss: 0.460592
2023-05-13 18:17:12,715 P80930 INFO Evaluation @epoch 1 - batch 2669: 
2023-05-13 18:19:33,875 P80930 INFO [Metrics] AUC: 0.728345 - gAUC: 0.638709
2023-05-13 18:19:33,881 P80930 INFO Save best model: monitor(max)=1.367054
2023-05-13 18:19:35,673 P80930 INFO ************ Epoch=1 end ************
2023-05-13 19:02:16,506 P80930 INFO Train loss: 0.449343
2023-05-13 19:02:16,507 P80930 INFO Evaluation @epoch 2 - batch 2669: 
2023-05-13 19:04:42,132 P80930 INFO [Metrics] AUC: 0.738410 - gAUC: 0.652429
2023-05-13 19:04:42,134 P80930 INFO Save best model: monitor(max)=1.390839
2023-05-13 19:04:44,479 P80930 INFO ************ Epoch=2 end ************
2023-05-13 19:47:32,108 P80930 INFO Train loss: 0.444577
2023-05-13 19:47:32,109 P80930 INFO Evaluation @epoch 3 - batch 2669: 
2023-05-13 19:50:24,517 P80930 INFO [Metrics] AUC: 0.740695 - gAUC: 0.654666
2023-05-13 19:50:24,524 P80930 INFO Save best model: monitor(max)=1.395360
2023-05-13 19:50:26,862 P80930 INFO ************ Epoch=3 end ************
2023-05-13 20:33:20,085 P80930 INFO Train loss: 0.442711
2023-05-13 20:33:20,086 P80930 INFO Evaluation @epoch 4 - batch 2669: 
2023-05-13 20:35:52,443 P80930 INFO [Metrics] AUC: 0.740337 - gAUC: 0.656451
2023-05-13 20:35:52,451 P80930 INFO Save best model: monitor(max)=1.396787
2023-05-13 20:35:54,789 P80930 INFO ************ Epoch=4 end ************
2023-05-13 21:18:44,040 P80930 INFO Train loss: 0.441169
2023-05-13 21:18:44,042 P80930 INFO Evaluation @epoch 5 - batch 2669: 
2023-05-13 21:21:18,028 P80930 INFO [Metrics] AUC: 0.741274 - gAUC: 0.657987
2023-05-13 21:21:18,030 P80930 INFO Save best model: monitor(max)=1.399261
2023-05-13 21:21:20,410 P80930 INFO ************ Epoch=5 end ************
2023-05-13 22:04:11,355 P80930 INFO Train loss: 0.439696
2023-05-13 22:04:11,357 P80930 INFO Evaluation @epoch 6 - batch 2669: 
2023-05-13 22:06:39,757 P80930 INFO [Metrics] AUC: 0.743642 - gAUC: 0.662018
2023-05-13 22:06:39,762 P80930 INFO Save best model: monitor(max)=1.405660
2023-05-13 22:06:42,132 P80930 INFO ************ Epoch=6 end ************
2023-05-13 22:49:42,834 P80930 INFO Train loss: 0.438842
2023-05-13 22:49:42,835 P80930 INFO Evaluation @epoch 7 - batch 2669: 
2023-05-13 22:52:14,129 P80930 INFO [Metrics] AUC: 0.744381 - gAUC: 0.663119
2023-05-13 22:52:14,131 P80930 INFO Save best model: monitor(max)=1.407500
2023-05-13 22:52:16,504 P80930 INFO ************ Epoch=7 end ************
2023-05-13 23:35:02,939 P80930 INFO Train loss: 0.438613
2023-05-13 23:35:02,941 P80930 INFO Evaluation @epoch 8 - batch 2669: 
2023-05-13 23:37:54,569 P80930 INFO [Metrics] AUC: 0.743420 - gAUC: 0.661469
2023-05-13 23:37:54,576 P80930 INFO Monitor(max)=1.404889 STOP!
2023-05-13 23:37:54,576 P80930 INFO Reduce learning rate on plateau: 0.000100
2023-05-13 23:37:54,679 P80930 INFO ************ Epoch=8 end ************
2023-05-14 00:20:38,408 P80930 INFO Train loss: 0.416218
2023-05-14 00:20:38,409 P80930 INFO Evaluation @epoch 9 - batch 2669: 
2023-05-14 00:23:02,025 P80930 INFO [Metrics] AUC: 0.749153 - gAUC: 0.669361
2023-05-14 00:23:02,030 P80930 INFO Save best model: monitor(max)=1.418514
2023-05-14 00:23:04,295 P80930 INFO ************ Epoch=9 end ************
2023-05-14 01:05:53,102 P80930 INFO Train loss: 0.407385
2023-05-14 01:05:53,103 P80930 INFO Evaluation @epoch 10 - batch 2669: 
2023-05-14 01:08:16,447 P80930 INFO [Metrics] AUC: 0.750515 - gAUC: 0.670932
2023-05-14 01:08:16,449 P80930 INFO Save best model: monitor(max)=1.421447
2023-05-14 01:08:18,757 P80930 INFO ************ Epoch=10 end ************
2023-05-14 01:51:03,480 P80930 INFO Train loss: 0.402786
2023-05-14 01:51:03,481 P80930 INFO Evaluation @epoch 11 - batch 2669: 
2023-05-14 01:53:31,868 P80930 INFO [Metrics] AUC: 0.750372 - gAUC: 0.671148
2023-05-14 01:53:31,870 P80930 INFO Save best model: monitor(max)=1.421520
2023-05-14 01:53:34,242 P80930 INFO ************ Epoch=11 end ************
2023-05-14 02:36:26,359 P80930 INFO Train loss: 0.399259
2023-05-14 02:36:26,360 P80930 INFO Evaluation @epoch 12 - batch 2669: 
2023-05-14 02:38:52,279 P80930 INFO [Metrics] AUC: 0.749782 - gAUC: 0.670719
2023-05-14 02:38:52,287 P80930 INFO Monitor(max)=1.420501 STOP!
2023-05-14 02:38:52,287 P80930 INFO Reduce learning rate on plateau: 0.000010
2023-05-14 02:38:52,379 P80930 INFO ************ Epoch=12 end ************
2023-05-14 03:21:50,250 P80930 INFO Train loss: 0.386945
2023-05-14 03:21:50,250 P80930 INFO Evaluation @epoch 13 - batch 2669: 
2023-05-14 03:24:25,420 P80930 INFO [Metrics] AUC: 0.748579 - gAUC: 0.669695
2023-05-14 03:24:25,425 P80930 INFO Monitor(max)=1.418274 STOP!
2023-05-14 03:24:25,425 P80930 INFO Reduce learning rate on plateau: 0.000001
2023-05-14 03:24:25,425 P80930 INFO ********* Epoch==13 early stop *********
2023-05-14 03:24:25,513 P80930 INFO Training finished.
2023-05-14 03:24:25,514 P80930 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DIEN_kuaivideo_x1/kuaivideo_x1_60f6c91a/DIEN_kuaivideo_x1_009_657021a9.model
2023-05-14 03:24:26,453 P80930 INFO ****** Validation evaluation ******
2023-05-14 03:26:53,204 P80930 INFO [Metrics] gAUC: 0.671148 - AUC: 0.750372 - logloss: 0.433049
2023-05-14 03:26:53,338 P80930 INFO ******** Test evaluation ********
2023-05-14 03:26:53,338 P80930 INFO Loading data...
2023-05-14 03:26:53,338 P80930 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_60f6c91a/test.h5
2023-05-14 03:26:59,423 P80930 INFO Test samples: total/2730291, blocks/1
2023-05-14 03:26:59,423 P80930 INFO Loading test data done.
2023-05-14 03:29:25,585 P80930 INFO [Metrics] gAUC: 0.671148 - AUC: 0.750372 - logloss: 0.433049

```
