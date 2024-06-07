## AutoInt_kuaivideo_x1

A hands-on guide to run the AutoInt model on the KuaiVideo_x1 dataset.

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
Please refer to [KuaiVideo_x1](https://github.com/reczoo/Datasets/tree/main/KuaiShou/KuaiVideo_x1) to get the dataset details.

### Code

We use the [AutoInt](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/AutoInt) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/KuaiShou/KuaiVideo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt+_kuaivideo_x1_tuner_config_02](./AutoInt+_kuaivideo_x1_tuner_config_02). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/AutoInt
    nohup python run_expid.py --config XXX/benchmarks/AutoInt/AutoInt+_kuaivideo_x1_tuner_config_02 --expid AutoInt_kuaivideo_x1_011_9ed2831b --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.666719 | 0.746899 | 0.435692  |


### Logs
```python
2022-08-23 21:16:08,271 P72848 INFO Params: {
    "attention_dim": "512",
    "attention_layers": "3",
    "batch_norm": "False",
    "batch_size": "8192",
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
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_emb'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'pos_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'neg_items'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'pos_items_emb'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'neg_items_emb'}]",
    "gpu": "0",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "AutoInt",
    "model_id": "AutoInt_kuaivideo_x1_011_9ed2831b",
    "model_root": "./checkpoints/AutoInt_kuaivideo_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_heads": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "20222023",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "train_data": "../data/KuaiShou/KuaiVideo_x1/train.csv",
    "use_residual": "True",
    "use_scale": "False",
    "use_wide": "False",
    "valid_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "verbose": "1"
}
2022-08-23 21:16:08,273 P72848 INFO Set up feature processor...
2022-08-23 21:16:08,273 P72848 INFO Load feature_map from json: ../data/KuaiShou/kuaivideo_x1_dc7a3035/feature_map.json
2022-08-23 21:16:08,273 P72848 INFO Set column index...
2022-08-23 21:16:08,273 P72848 INFO Feature specs: {
    "group_id": "{'type': 'meta'}",
    "item_emb": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406}",
    "neg_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "neg_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "pos_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "pos_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10001, 'vocab_size': 10002}"
}
2022-08-23 21:16:16,283 P72848 INFO Total number of parameters: 43950082.
2022-08-23 21:16:16,283 P72848 INFO Loading data...
2022-08-23 21:16:16,283 P72848 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/train.h5
2022-08-23 21:16:41,133 P72848 INFO Train samples: total/10931092, blocks/1
2022-08-23 21:16:41,133 P72848 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/valid.h5
2022-08-23 21:16:46,868 P72848 INFO Validation samples: total/2730291, blocks/1
2022-08-23 21:16:46,868 P72848 INFO Loading train and validation data done.
2022-08-23 21:16:46,869 P72848 INFO Start training: 1335 batches/epoch
2022-08-23 21:16:46,869 P72848 INFO ************ Epoch=1 start ************
2022-08-23 21:24:01,467 P72848 INFO [Metrics] AUC: 0.726639 - gAUC: 0.637584
2022-08-23 21:24:01,480 P72848 INFO Save best model: monitor(max): 1.364223
2022-08-23 21:24:03,316 P72848 INFO --- 1335/1335 batches finished ---
2022-08-23 21:24:03,389 P72848 INFO Train loss: 0.464143
2022-08-23 21:24:03,389 P72848 INFO ************ Epoch=1 end ************
2022-08-23 21:31:14,871 P72848 INFO [Metrics] AUC: 0.731792 - gAUC: 0.645143
2022-08-23 21:31:14,883 P72848 INFO Save best model: monitor(max): 1.376936
2022-08-23 21:31:17,128 P72848 INFO --- 1335/1335 batches finished ---
2022-08-23 21:31:17,213 P72848 INFO Train loss: 0.451669
2022-08-23 21:31:17,213 P72848 INFO ************ Epoch=2 end ************
2022-08-23 21:38:35,523 P72848 INFO [Metrics] AUC: 0.735088 - gAUC: 0.649924
2022-08-23 21:38:35,548 P72848 INFO Save best model: monitor(max): 1.385012
2022-08-23 21:38:37,664 P72848 INFO --- 1335/1335 batches finished ---
2022-08-23 21:38:37,784 P72848 INFO Train loss: 0.447777
2022-08-23 21:38:37,784 P72848 INFO ************ Epoch=3 end ************
2022-08-23 21:45:56,625 P72848 INFO [Metrics] AUC: 0.736753 - gAUC: 0.653395
2022-08-23 21:45:56,631 P72848 INFO Save best model: monitor(max): 1.390148
2022-08-23 21:45:58,878 P72848 INFO --- 1335/1335 batches finished ---
2022-08-23 21:45:58,983 P72848 INFO Train loss: 0.444823
2022-08-23 21:45:58,983 P72848 INFO ************ Epoch=4 end ************
2022-08-23 21:53:19,279 P72848 INFO [Metrics] AUC: 0.737165 - gAUC: 0.654465
2022-08-23 21:53:19,290 P72848 INFO Save best model: monitor(max): 1.391630
2022-08-23 21:53:21,530 P72848 INFO --- 1335/1335 batches finished ---
2022-08-23 21:53:21,624 P72848 INFO Train loss: 0.442999
2022-08-23 21:53:21,624 P72848 INFO ************ Epoch=5 end ************
2022-08-23 22:00:44,546 P72848 INFO [Metrics] AUC: 0.737725 - gAUC: 0.654237
2022-08-23 22:00:44,558 P72848 INFO Save best model: monitor(max): 1.391962
2022-08-23 22:00:46,774 P72848 INFO --- 1335/1335 batches finished ---
2022-08-23 22:00:46,852 P72848 INFO Train loss: 0.442032
2022-08-23 22:00:46,852 P72848 INFO ************ Epoch=6 end ************
2022-08-23 22:08:13,991 P72848 INFO [Metrics] AUC: 0.741623 - gAUC: 0.657846
2022-08-23 22:08:14,004 P72848 INFO Save best model: monitor(max): 1.399470
2022-08-23 22:08:16,211 P72848 INFO --- 1335/1335 batches finished ---
2022-08-23 22:08:16,319 P72848 INFO Train loss: 0.441423
2022-08-23 22:08:16,320 P72848 INFO ************ Epoch=7 end ************
2022-08-23 22:15:45,114 P72848 INFO [Metrics] AUC: 0.741336 - gAUC: 0.658218
2022-08-23 22:15:45,122 P72848 INFO Save best model: monitor(max): 1.399554
2022-08-23 22:15:47,206 P72848 INFO --- 1335/1335 batches finished ---
2022-08-23 22:15:47,293 P72848 INFO Train loss: 0.441095
2022-08-23 22:15:47,293 P72848 INFO ************ Epoch=8 end ************
2022-08-23 22:23:08,604 P72848 INFO [Metrics] AUC: 0.740962 - gAUC: 0.657606
2022-08-23 22:23:08,612 P72848 INFO Monitor(max) STOP: 1.398568 !
2022-08-23 22:23:08,613 P72848 INFO Reduce learning rate on plateau: 0.000100
2022-08-23 22:23:08,613 P72848 INFO --- 1335/1335 batches finished ---
2022-08-23 22:23:08,723 P72848 INFO Train loss: 0.440843
2022-08-23 22:23:08,723 P72848 INFO ************ Epoch=9 end ************
2022-08-23 22:30:39,672 P72848 INFO [Metrics] AUC: 0.745528 - gAUC: 0.664968
2022-08-23 22:30:39,685 P72848 INFO Save best model: monitor(max): 1.410496
2022-08-23 22:30:41,728 P72848 INFO --- 1335/1335 batches finished ---
2022-08-23 22:30:41,820 P72848 INFO Train loss: 0.420978
2022-08-23 22:30:41,820 P72848 INFO ************ Epoch=10 end ************
2022-08-23 22:37:48,786 P72848 INFO [Metrics] AUC: 0.746899 - gAUC: 0.666719
2022-08-23 22:37:48,794 P72848 INFO Save best model: monitor(max): 1.413618
2022-08-23 22:37:51,111 P72848 INFO --- 1335/1335 batches finished ---
2022-08-23 22:37:51,236 P72848 INFO Train loss: 0.414732
2022-08-23 22:37:51,240 P72848 INFO ************ Epoch=11 end ************
2022-08-23 22:44:49,855 P72848 INFO [Metrics] AUC: 0.746038 - gAUC: 0.666457
2022-08-23 22:44:49,868 P72848 INFO Monitor(max) STOP: 1.412495 !
2022-08-23 22:44:49,868 P72848 INFO Reduce learning rate on plateau: 0.000010
2022-08-23 22:44:49,869 P72848 INFO --- 1335/1335 batches finished ---
2022-08-23 22:44:49,963 P72848 INFO Train loss: 0.411359
2022-08-23 22:44:49,963 P72848 INFO ************ Epoch=12 end ************
2022-08-23 22:51:58,283 P72848 INFO [Metrics] AUC: 0.745015 - gAUC: 0.666013
2022-08-23 22:51:58,297 P72848 INFO Monitor(max) STOP: 1.411029 !
2022-08-23 22:51:58,297 P72848 INFO Reduce learning rate on plateau: 0.000001
2022-08-23 22:51:58,297 P72848 INFO ********* Epoch==13 early stop *********
2022-08-23 22:51:58,298 P72848 INFO --- 1335/1335 batches finished ---
2022-08-23 22:51:58,404 P72848 INFO Train loss: 0.403844
2022-08-23 22:51:58,404 P72848 INFO Training finished.
2022-08-23 22:51:58,404 P72848 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/AutoInt_kuaivideo_x1/kuaivideo_x1_dc7a3035/AutoInt_kuaivideo_x1_011_9ed2831b.model
2022-08-23 22:51:59,384 P72848 INFO ****** Validation evaluation ******
2022-08-23 22:53:20,355 P72848 INFO [Metrics] gAUC: 0.666719 - AUC: 0.746899 - logloss: 0.435692
2022-08-23 22:53:20,552 P72848 INFO ******** Test evaluation ********
2022-08-23 22:53:20,552 P72848 INFO Loading data...
2022-08-23 22:53:20,552 P72848 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/test.h5
2022-08-23 22:53:27,064 P72848 INFO Test samples: total/2730291, blocks/1
2022-08-23 22:53:27,064 P72848 INFO Loading test data done.
2022-08-23 22:54:40,704 P72848 INFO [Metrics] gAUC: 0.666719 - AUC: 0.746899 - logloss: 0.435692

```
