## BST_kuaivideo_x1

A hands-on guide to run the BST model on the KuaiVideo_x1 dataset.

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
Please refer to the BARS dataset [KuaiVideo_x1](https://github.com/openbenchmark/BARS/blob/main/datasets/KuaiShou#KuaiVideo_x1) to get data ready.

### Code

We use the [BST](https://github.com/xue-pai/FuxiCTR/blob/v2.0.1/model_zoo/BST) model code from [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/KuaiShou/KuaiVideo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [BST_kuaivideo_x1_tuner_config_02](./BST_kuaivideo_x1_tuner_config_02). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/BST
    nohup python run_expid.py --config XXX/benchmarks/BST/BST_kuaivideo_x1_tuner_config_02 --expid BST_kuaivideo_x1_003_7ed4faca --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.669039 | 0.748407 | 0.433819  |


### Logs
```python
2022-09-08 09:47:23,492 P103231 INFO Params: {
    "attention_dropout": "0.1",
    "batch_norm": "False",
    "batch_size": "4096",
    "bst_sequence_field": "[('pos_items', 'pos_items_emb'), ('neg_items', 'neg_items_emb')]",
    "bst_target_field": "[('item_id', 'item_emb'), ('item_id', 'item_emb')]",
    "data_format": "csv",
    "data_root": "../data/KuaiShou/",
    "dataset_id": "kuaivideo_x1_dc7a3035",
    "debug_mode": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.0001",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'min_categr_count': 1, 'name': 'item_emb', 'preprocess': 'copy_from(item_id)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'pos_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'neg_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'pos_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(pos_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'neg_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(neg_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_emb'}, {'feature_encoder': None, 'name': 'pos_items'}, {'feature_encoder': None, 'name': 'neg_items'}, {'feature_encoder': ['nn.Linear(64, 64, bias=False)'], 'name': 'pos_items_emb'}, {'feature_encoder': ['nn.Linear(64, 64, bias=False)'], 'name': 'neg_items_emb'}]",
    "gpu": "2",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "BST",
    "model_id": "BST_kuaivideo_x1_003_7ed4faca",
    "model_root": "./checkpoints/BST_kuaivideo_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_heads": "8",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "20222023",
    "seq_pooling_type": "target",
    "shuffle": "True",
    "stacked_transformer_layers": "1",
    "task": "binary_classification",
    "test_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "train_data": "../data/KuaiShou/KuaiVideo_x1/train.csv",
    "use_causal_mask": "False",
    "use_position_emb": "True",
    "use_residual": "False",
    "valid_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "verbose": "1"
}
2022-09-08 09:47:23,493 P103231 INFO Set up feature processor...
2022-09-08 09:47:23,493 P103231 INFO Load feature_map from json: ../data/KuaiShou/kuaivideo_x1_dc7a3035/feature_map.json
2022-09-08 09:47:23,494 P103231 INFO Set column index...
2022-09-08 09:47:23,494 P103231 INFO Feature specs: {
    "group_id": "{'type': 'meta'}",
    "item_emb": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406}",
    "neg_items": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "neg_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "pos_items": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "pos_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10001, 'vocab_size': 10002}"
}
2022-09-08 09:47:31,907 P103231 INFO Total number of parameters: 42831233.
2022-09-08 09:47:31,908 P103231 INFO Loading data...
2022-09-08 09:47:31,908 P103231 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/train.h5
2022-09-08 09:47:58,560 P103231 INFO Train samples: total/10931092, blocks/1
2022-09-08 09:47:58,560 P103231 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/valid.h5
2022-09-08 09:48:04,468 P103231 INFO Validation samples: total/2730291, blocks/1
2022-09-08 09:48:04,468 P103231 INFO Loading train and validation data done.
2022-09-08 09:48:04,468 P103231 INFO Start training: 2669 batches/epoch
2022-09-08 09:48:04,468 P103231 INFO ************ Epoch=1 start ************
2022-09-08 10:06:51,587 P103231 INFO [Metrics] AUC: 0.726357 - gAUC: 0.638959
2022-09-08 10:06:51,604 P103231 INFO Save best model: monitor(max): 1.365316
2022-09-08 10:06:53,309 P103231 INFO --- 2669/2669 batches finished ---
2022-09-08 10:06:53,376 P103231 INFO Train loss: 0.457254
2022-09-08 10:06:53,377 P103231 INFO ************ Epoch=1 end ************
2022-09-08 10:25:37,045 P103231 INFO [Metrics] AUC: 0.732742 - gAUC: 0.648994
2022-09-08 10:25:37,058 P103231 INFO Save best model: monitor(max): 1.381737
2022-09-08 10:25:39,231 P103231 INFO --- 2669/2669 batches finished ---
2022-09-08 10:25:39,309 P103231 INFO Train loss: 0.447659
2022-09-08 10:25:39,309 P103231 INFO ************ Epoch=2 end ************
2022-09-08 10:44:35,780 P103231 INFO [Metrics] AUC: 0.736178 - gAUC: 0.652060
2022-09-08 10:44:35,793 P103231 INFO Save best model: monitor(max): 1.388238
2022-09-08 10:44:37,986 P103231 INFO --- 2669/2669 batches finished ---
2022-09-08 10:44:38,061 P103231 INFO Train loss: 0.442282
2022-09-08 10:44:38,061 P103231 INFO ************ Epoch=3 end ************
2022-09-08 11:03:32,078 P103231 INFO [Metrics] AUC: 0.738212 - gAUC: 0.654924
2022-09-08 11:03:32,089 P103231 INFO Save best model: monitor(max): 1.393136
2022-09-08 11:03:34,342 P103231 INFO --- 2669/2669 batches finished ---
2022-09-08 11:03:34,431 P103231 INFO Train loss: 0.439627
2022-09-08 11:03:34,431 P103231 INFO ************ Epoch=4 end ************
2022-09-08 11:22:29,281 P103231 INFO [Metrics] AUC: 0.739610 - gAUC: 0.657194
2022-09-08 11:22:29,291 P103231 INFO Save best model: monitor(max): 1.396804
2022-09-08 11:22:31,471 P103231 INFO --- 2669/2669 batches finished ---
2022-09-08 11:22:31,559 P103231 INFO Train loss: 0.438544
2022-09-08 11:22:31,559 P103231 INFO ************ Epoch=5 end ************
2022-09-08 11:41:20,576 P103231 INFO [Metrics] AUC: 0.739851 - gAUC: 0.657465
2022-09-08 11:41:20,607 P103231 INFO Save best model: monitor(max): 1.397317
2022-09-08 11:41:22,807 P103231 INFO --- 2669/2669 batches finished ---
2022-09-08 11:41:22,869 P103231 INFO Train loss: 0.438030
2022-09-08 11:41:22,869 P103231 INFO ************ Epoch=6 end ************
2022-09-08 12:00:11,255 P103231 INFO [Metrics] AUC: 0.740858 - gAUC: 0.658515
2022-09-08 12:00:11,262 P103231 INFO Save best model: monitor(max): 1.399373
2022-09-08 12:00:13,392 P103231 INFO --- 2669/2669 batches finished ---
2022-09-08 12:00:13,451 P103231 INFO Train loss: 0.437897
2022-09-08 12:00:13,452 P103231 INFO ************ Epoch=7 end ************
2022-09-08 12:18:51,527 P103231 INFO [Metrics] AUC: 0.740852 - gAUC: 0.658539
2022-09-08 12:18:51,538 P103231 INFO Save best model: monitor(max): 1.399391
2022-09-08 12:18:53,718 P103231 INFO --- 2669/2669 batches finished ---
2022-09-08 12:18:53,819 P103231 INFO Train loss: 0.438429
2022-09-08 12:18:53,819 P103231 INFO ************ Epoch=8 end ************
2022-09-08 12:37:30,614 P103231 INFO [Metrics] AUC: 0.741634 - gAUC: 0.659547
2022-09-08 12:37:30,624 P103231 INFO Save best model: monitor(max): 1.401181
2022-09-08 12:37:32,830 P103231 INFO --- 2669/2669 batches finished ---
2022-09-08 12:37:32,913 P103231 INFO Train loss: 0.439546
2022-09-08 12:37:32,913 P103231 INFO ************ Epoch=9 end ************
2022-09-08 12:56:10,023 P103231 INFO [Metrics] AUC: 0.742407 - gAUC: 0.660156
2022-09-08 12:56:10,032 P103231 INFO Save best model: monitor(max): 1.402563
2022-09-08 12:56:12,209 P103231 INFO --- 2669/2669 batches finished ---
2022-09-08 12:56:12,271 P103231 INFO Train loss: 0.440593
2022-09-08 12:56:12,272 P103231 INFO ************ Epoch=10 end ************
2022-09-08 13:14:52,376 P103231 INFO [Metrics] AUC: 0.740417 - gAUC: 0.658914
2022-09-08 13:14:52,386 P103231 INFO Monitor(max) STOP: 1.399332 !
2022-09-08 13:14:52,386 P103231 INFO Reduce learning rate on plateau: 0.000100
2022-09-08 13:14:52,386 P103231 INFO --- 2669/2669 batches finished ---
2022-09-08 13:14:52,461 P103231 INFO Train loss: 0.441351
2022-09-08 13:14:52,462 P103231 INFO ************ Epoch=11 end ************
2022-09-08 13:33:30,660 P103231 INFO [Metrics] AUC: 0.747143 - gAUC: 0.666986
2022-09-08 13:33:30,670 P103231 INFO Save best model: monitor(max): 1.414129
2022-09-08 13:33:32,832 P103231 INFO --- 2669/2669 batches finished ---
2022-09-08 13:33:32,922 P103231 INFO Train loss: 0.418913
2022-09-08 13:33:32,922 P103231 INFO ************ Epoch=12 end ************
2022-09-08 13:52:43,347 P103231 INFO [Metrics] AUC: 0.747965 - gAUC: 0.668347
2022-09-08 13:52:43,358 P103231 INFO Save best model: monitor(max): 1.416311
2022-09-08 13:52:45,524 P103231 INFO --- 2669/2669 batches finished ---
2022-09-08 13:52:45,592 P103231 INFO Train loss: 0.411441
2022-09-08 13:52:45,592 P103231 INFO ************ Epoch=13 end ************
2022-09-08 14:11:22,115 P103231 INFO [Metrics] AUC: 0.748407 - gAUC: 0.669039
2022-09-08 14:11:22,127 P103231 INFO Save best model: monitor(max): 1.417446
2022-09-08 14:11:24,264 P103231 INFO --- 2669/2669 batches finished ---
2022-09-08 14:11:24,326 P103231 INFO Train loss: 0.407264
2022-09-08 14:11:24,326 P103231 INFO ************ Epoch=14 end ************
2022-09-08 14:30:01,336 P103231 INFO [Metrics] AUC: 0.748178 - gAUC: 0.668968
2022-09-08 14:30:01,347 P103231 INFO Monitor(max) STOP: 1.417146 !
2022-09-08 14:30:01,347 P103231 INFO Reduce learning rate on plateau: 0.000010
2022-09-08 14:30:01,348 P103231 INFO --- 2669/2669 batches finished ---
2022-09-08 14:30:01,421 P103231 INFO Train loss: 0.404041
2022-09-08 14:30:01,422 P103231 INFO ************ Epoch=15 end ************
2022-09-08 14:48:41,836 P103231 INFO [Metrics] AUC: 0.746858 - gAUC: 0.667727
2022-09-08 14:48:41,845 P103231 INFO Monitor(max) STOP: 1.414585 !
2022-09-08 14:48:41,845 P103231 INFO Reduce learning rate on plateau: 0.000001
2022-09-08 14:48:41,845 P103231 INFO ********* Epoch==16 early stop *********
2022-09-08 14:48:41,846 P103231 INFO --- 2669/2669 batches finished ---
2022-09-08 14:48:41,907 P103231 INFO Train loss: 0.394157
2022-09-08 14:48:41,908 P103231 INFO Training finished.
2022-09-08 14:48:41,908 P103231 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/BST_kuaivideo_x1/kuaivideo_x1_dc7a3035/BST_kuaivideo_x1_003_7ed4faca.model
2022-09-08 14:48:43,315 P103231 INFO ****** Validation evaluation ******
2022-09-08 14:50:57,186 P103231 INFO [Metrics] gAUC: 0.669039 - AUC: 0.748407 - logloss: 0.433819
2022-09-08 14:50:57,355 P103231 INFO ******** Test evaluation ********
2022-09-08 14:50:57,355 P103231 INFO Loading data...
2022-09-08 14:50:57,356 P103231 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/test.h5
2022-09-08 14:51:03,444 P103231 INFO Test samples: total/2730291, blocks/1
2022-09-08 14:51:03,444 P103231 INFO Loading test data done.
2022-09-08 14:53:21,339 P103231 INFO [Metrics] gAUC: 0.669039 - AUC: 0.748407 - logloss: 0.433819

```
