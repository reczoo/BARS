## DMIN_kuaivideo_x1

A hands-on guide to run the DMIN model on the KuaiVideo_x1 dataset.

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
  fuxictr: 2.0.3

  ```

### Dataset
Please refer to [KuaiVideo_x1](https://github.com/reczoo/Datasets/tree/main/KuaiShou/KuaiVideo_x1) to get the dataset details.

### Code

We use the [DMIN](https://github.com/reczoo/FuxiCTR/tree/v2.0.3/model_zoo/DMIN) model code from [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/tree/v2.0.3) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.3.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.3
    ```

2. Create a data directory and put the downloaded data files in `../data/KuaiShou/KuaiVideo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DMIN_kuaivideo_x1_tuner_config_02](./DMIN_kuaivideo_x1_tuner_config_02). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DMIN
    nohup python run_expid.py --config YOUR_PATH/DMIN/DMIN_kuaivideo_x1_tuner_config_02 --expid DMIN_kuaivideo_x1_020_d155f57e --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.672621 | 0.750777 | 0.430187  |


### Logs
```python
2023-05-24 02:21:17,101 P38184 INFO Params: {
    "attention_activation": "ReLU",
    "attention_dropout": "0.2",
    "attention_hidden_units": "[512, 256]",
    "aux_activation": "ReLU",
    "aux_hidden_units": "[100, 50]",
    "aux_loss_lambda": "0",
    "batch_norm": "False",
    "batch_size": "4096",
    "bn_only_once": "False",
    "data_format": "csv",
    "data_root": "../data/KuaiShou/",
    "dataset_id": "kuaivideo_x1_60f6c91a",
    "debug_mode": "False",
    "dnn_activations": "Dice",
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
    "gpu": "1",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "layer_norm": "True",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "DMIN",
    "model_id": "DMIN_kuaivideo_x1_020_d155f57e",
    "model_root": "./checkpoints/DMIN_kuaivideo_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "neg_seq_field": "None",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_heads": "2",
    "num_workers": "3",
    "optimizer": "adam",
    "pickle_feature_encoder": "True",
    "pos_emb_dim": "8",
    "save_best_only": "True",
    "seed": "20222023",
    "sequence_field": "[('pos_items', 'pos_items_emb'), ('neg_items', 'neg_items_emb')]",
    "shuffle": "True",
    "target_field": "[('item_id', 'item_emb'), ('item_id', 'item_emb')]",
    "task": "binary_classification",
    "test_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "train_data": "../data/KuaiShou/KuaiVideo_x1/train.csv",
    "use_behavior_refiner": "False",
    "use_features": "None",
    "use_pos_emb": "False",
    "valid_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "verbose": "1"
}
2023-05-24 02:21:17,102 P38184 INFO Set up feature processor...
2023-05-24 02:21:17,102 P38184 WARNING Skip rebuilding ../data/KuaiShou/kuaivideo_x1_60f6c91a/feature_map.json. Please delete it manually if rebuilding is required.
2023-05-24 02:21:17,102 P38184 INFO Load feature_map from json: ../data/KuaiShou/kuaivideo_x1_60f6c91a/feature_map.json
2023-05-24 02:21:17,103 P38184 INFO Set column index...
2023-05-24 02:21:17,103 P38184 INFO Feature specs: {
    "group_id": "{'type': 'meta'}",
    "item_emb": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406}",
    "neg_items": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "neg_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "pos_items": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "pos_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10001, 'vocab_size': 10002}"
}
2023-05-24 02:21:24,250 P38184 INFO Total number of parameters: 44479237.
2023-05-24 02:21:24,250 P38184 INFO Loading data...
2023-05-24 02:21:24,250 P38184 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_60f6c91a/train.h5
2023-05-24 02:21:48,371 P38184 INFO Train samples: total/10931092, blocks/1
2023-05-24 02:21:48,371 P38184 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_60f6c91a/valid.h5
2023-05-24 02:21:54,337 P38184 INFO Validation samples: total/2730291, blocks/1
2023-05-24 02:21:54,337 P38184 INFO Loading train and validation data done.
2023-05-24 02:21:54,337 P38184 INFO Start training: 2669 batches/epoch
2023-05-24 02:21:54,337 P38184 INFO ************ Epoch=1 start ************
2023-05-24 02:51:43,862 P38184 INFO Train loss: 0.459763
2023-05-24 02:51:43,863 P38184 INFO Evaluation @epoch 1 - batch 2669: 
2023-05-24 02:54:26,176 P38184 INFO [Metrics] AUC: 0.728968 - gAUC: 0.644054
2023-05-24 02:54:26,179 P38184 INFO Save best model: monitor(max)=1.373022
2023-05-24 02:54:28,391 P38184 INFO ************ Epoch=1 end ************
2023-05-24 03:24:25,138 P38184 INFO Train loss: 0.448659
2023-05-24 03:24:25,139 P38184 INFO Evaluation @epoch 2 - batch 2669: 
2023-05-24 03:27:09,608 P38184 INFO [Metrics] AUC: 0.736901 - gAUC: 0.653757
2023-05-24 03:27:09,612 P38184 INFO Save best model: monitor(max)=1.390658
2023-05-24 03:27:11,995 P38184 INFO ************ Epoch=2 end ************
2023-05-24 03:57:07,792 P38184 INFO Train loss: 0.442370
2023-05-24 03:57:07,793 P38184 INFO Evaluation @epoch 3 - batch 2669: 
2023-05-24 03:59:51,719 P38184 INFO [Metrics] AUC: 0.740633 - gAUC: 0.658931
2023-05-24 03:59:51,728 P38184 INFO Save best model: monitor(max)=1.399564
2023-05-24 03:59:53,981 P38184 INFO ************ Epoch=3 end ************
2023-05-24 04:29:39,203 P38184 INFO Train loss: 0.439325
2023-05-24 04:29:39,204 P38184 INFO Evaluation @epoch 4 - batch 2669: 
2023-05-24 04:32:21,538 P38184 INFO [Metrics] AUC: 0.743660 - gAUC: 0.661917
2023-05-24 04:32:21,541 P38184 INFO Save best model: monitor(max)=1.405577
2023-05-24 04:32:23,762 P38184 INFO ************ Epoch=4 end ************
2023-05-24 05:02:11,222 P38184 INFO Train loss: 0.437056
2023-05-24 05:02:11,223 P38184 INFO Evaluation @epoch 5 - batch 2669: 
2023-05-24 05:04:53,498 P38184 INFO [Metrics] AUC: 0.741948 - gAUC: 0.660783
2023-05-24 05:04:53,500 P38184 INFO Monitor(max)=1.402731 STOP!
2023-05-24 05:04:53,500 P38184 INFO Reduce learning rate on plateau: 0.000100
2023-05-24 05:04:53,579 P38184 INFO ************ Epoch=5 end ************
2023-05-24 05:34:34,084 P38184 INFO Train loss: 0.419416
2023-05-24 05:34:34,084 P38184 INFO Evaluation @epoch 6 - batch 2669: 
2023-05-24 05:37:14,934 P38184 INFO [Metrics] AUC: 0.749531 - gAUC: 0.670756
2023-05-24 05:37:14,941 P38184 INFO Save best model: monitor(max)=1.420288
2023-05-24 05:37:17,140 P38184 INFO ************ Epoch=6 end ************
2023-05-24 06:07:02,535 P38184 INFO Train loss: 0.413453
2023-05-24 06:07:02,535 P38184 INFO Evaluation @epoch 7 - batch 2669: 
2023-05-24 06:09:45,285 P38184 INFO [Metrics] AUC: 0.750247 - gAUC: 0.671811
2023-05-24 06:09:45,288 P38184 INFO Save best model: monitor(max)=1.422058
2023-05-24 06:09:47,545 P38184 INFO ************ Epoch=7 end ************
2023-05-24 06:39:56,163 P38184 INFO Train loss: 0.410211
2023-05-24 06:39:56,164 P38184 INFO Evaluation @epoch 8 - batch 2669: 
2023-05-24 06:42:41,946 P38184 INFO [Metrics] AUC: 0.750200 - gAUC: 0.672099
2023-05-24 06:42:41,953 P38184 INFO Save best model: monitor(max)=1.422300
2023-05-24 06:42:44,129 P38184 INFO ************ Epoch=8 end ************
2023-05-24 07:12:26,246 P38184 INFO Train loss: 0.407591
2023-05-24 07:12:26,246 P38184 INFO Evaluation @epoch 9 - batch 2669: 
2023-05-24 07:15:08,906 P38184 INFO [Metrics] AUC: 0.750777 - gAUC: 0.672621
2023-05-24 07:15:08,911 P38184 INFO Save best model: monitor(max)=1.423399
2023-05-24 07:15:11,171 P38184 INFO ************ Epoch=9 end ************
2023-05-24 07:45:09,335 P38184 INFO Train loss: 0.405235
2023-05-24 07:45:09,336 P38184 INFO Evaluation @epoch 10 - batch 2669: 
2023-05-24 07:47:53,016 P38184 INFO [Metrics] AUC: 0.749961 - gAUC: 0.671917
2023-05-24 07:47:53,020 P38184 INFO Monitor(max)=1.421877 STOP!
2023-05-24 07:47:53,020 P38184 INFO Reduce learning rate on plateau: 0.000010
2023-05-24 07:47:53,107 P38184 INFO ************ Epoch=10 end ************
2023-05-24 08:17:44,123 P38184 INFO Train loss: 0.397358
2023-05-24 08:17:44,124 P38184 INFO Evaluation @epoch 11 - batch 2669: 
2023-05-24 08:20:26,087 P38184 INFO [Metrics] AUC: 0.748279 - gAUC: 0.670200
2023-05-24 08:20:26,095 P38184 INFO Monitor(max)=1.418478 STOP!
2023-05-24 08:20:26,095 P38184 INFO Reduce learning rate on plateau: 0.000001
2023-05-24 08:20:26,095 P38184 INFO ********* Epoch==11 early stop *********
2023-05-24 08:20:26,167 P38184 INFO Training finished.
2023-05-24 08:20:26,167 P38184 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DMIN_kuaivideo_x1/kuaivideo_x1_60f6c91a/DMIN_kuaivideo_x1_020_d155f57e.model
2023-05-24 08:20:27,344 P38184 INFO ****** Validation evaluation ******
2023-05-24 08:23:09,237 P38184 INFO [Metrics] gAUC: 0.672621 - AUC: 0.750777 - logloss: 0.430187
2023-05-24 08:23:09,389 P38184 INFO ******** Test evaluation ********
2023-05-24 08:23:09,389 P38184 INFO Loading data...
2023-05-24 08:23:09,389 P38184 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_60f6c91a/test.h5
2023-05-24 08:23:16,034 P38184 INFO Test samples: total/2730291, blocks/1
2023-05-24 08:23:16,035 P38184 INFO Loading test data done.
2023-05-24 08:25:56,702 P38184 INFO [Metrics] gAUC: 0.672621 - AUC: 0.750777 - logloss: 0.430187

```
