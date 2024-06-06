## DMIN_microvideo1.7m_x1

A hands-on guide to run the DMIN model on the MicroVideo1.7M_x1 dataset.

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
Please refer to [MicroVideo1.7M_x1](https://github.com/reczoo/Datasets/tree/main/MicroVideo/MicroVideo1.7M_x1) to get the dataset details.

### Code

We use the [DMIN](https://github.com/reczoo/FuxiCTR/tree/v2.0.3/model_zoo/DMIN) model code from [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/tree/v2.0.3) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.3.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.3
    ```

2. Create a data directory and put the downloaded data files in `../data/MicroVideo/MicroVideo1.7M_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DMIN_microvideo1.7m_x1_tuner_config_04](./DMIN_microvideo1.7m_x1_tuner_config_04). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DMIN
    nohup python run_expid.py --config YOUR_PATH/DMIN/DMIN_microvideo1.7m_x1_tuner_config_04 --expid DMIN_microvideo1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.687910 | 0.733174 | 0.411456  |


### Logs
```python
2023-05-22 19:03:26,748 P1124 INFO Params: {
    "attention_activation": "ReLU",
    "attention_dropout": "0.1",
    "attention_hidden_units": "[512, 256]",
    "aux_activation": "ReLU",
    "aux_hidden_units": "[100, 50]",
    "aux_loss_lambda": "0",
    "batch_norm": "True",
    "batch_size": "2048",
    "bn_only_once": "False",
    "data_format": "csv",
    "data_root": "../data/MicroVideo1.7M/",
    "dataset_id": "microvideo1.7m_x1_83b31456",
    "debug_mode": "False",
    "dnn_activations": "ReLU",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.0005",
    "enable_sum_pooling": "False",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'name': 'clicked_items', 'padding': 'post', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'clicked_categories', 'padding': 'post', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "feature_config": "None",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id'}, {'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'clicked_items'}, {'feature_encoder': None, 'name': 'clicked_categories'}]",
    "gpu": "1",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "layer_norm": "True",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DMIN",
    "model_id": "DMIN_microvideo1.7m_x1_002_27dc4206",
    "model_root": "./checkpoints/DMIN_microvideo1.7m_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "neg_seq_field": "None",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_heads": "1",
    "num_workers": "3",
    "optimizer": "adam",
    "pickle_feature_encoder": "True",
    "pos_emb_dim": "2",
    "save_best_only": "True",
    "seed": "2022",
    "sequence_field": "('clicked_items', 'clicked_categories')",
    "shuffle": "True",
    "target_field": "('item_id', 'cate_id')",
    "task": "binary_classification",
    "test_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "train_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/train.csv",
    "use_behavior_refiner": "True",
    "use_features": "None",
    "use_pos_emb": "False",
    "valid_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "verbose": "1"
}
2023-05-22 19:03:26,748 P1124 INFO Set up feature processor...
2023-05-22 19:03:26,749 P1124 WARNING Skip rebuilding ../data/MicroVideo1.7M/microvideo1.7m_x1_83b31456/feature_map.json. Please delete it manually if rebuilding is required.
2023-05-22 19:03:26,749 P1124 INFO Load feature_map from json: ../data/MicroVideo1.7M/microvideo1.7m_x1_83b31456/feature_map.json
2023-05-22 19:03:26,749 P1124 INFO Set column index...
2023-05-22 19:03:26,749 P1124 INFO Feature specs: {
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514}",
    "clicked_categories": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514, 'max_len': 100}",
    "clicked_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'max_len': 100}",
    "group_id": "{'type': 'meta'}",
    "item_id": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10987, 'vocab_size': 10988}"
}
2023-05-22 19:03:34,413 P1124 INFO Total number of parameters: 2391170.
2023-05-22 19:03:34,413 P1124 INFO Loading data...
2023-05-22 19:03:34,414 P1124 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_83b31456/train.h5
2023-05-22 19:03:44,720 P1124 INFO Train samples: total/8970309, blocks/1
2023-05-22 19:03:44,721 P1124 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_83b31456/valid.h5
2023-05-22 19:03:48,822 P1124 INFO Validation samples: total/3767308, blocks/1
2023-05-22 19:03:48,822 P1124 INFO Loading train and validation data done.
2023-05-22 19:03:48,823 P1124 INFO Start training: 4381 batches/epoch
2023-05-22 19:03:48,823 P1124 INFO ************ Epoch=1 start ************
2023-05-22 19:14:34,894 P1124 INFO Train loss: 0.459424
2023-05-22 19:14:34,895 P1124 INFO Evaluation @epoch 1 - batch 4381: 
2023-05-22 19:17:02,982 P1124 INFO [Metrics] AUC: 0.719974 - gAUC: 0.675118
2023-05-22 19:17:02,986 P1124 INFO Save best model: monitor(max)=1.395092
2023-05-22 19:17:04,928 P1124 INFO ************ Epoch=1 end ************
2023-05-22 19:28:00,343 P1124 INFO Train loss: 0.440365
2023-05-22 19:28:00,344 P1124 INFO Evaluation @epoch 2 - batch 4381: 
2023-05-22 19:30:39,971 P1124 INFO [Metrics] AUC: 0.721605 - gAUC: 0.674904
2023-05-22 19:30:39,972 P1124 INFO Save best model: monitor(max)=1.396509
2023-05-22 19:30:41,925 P1124 INFO ************ Epoch=2 end ************
2023-05-22 19:41:58,600 P1124 INFO Train loss: 0.436459
2023-05-22 19:41:58,600 P1124 INFO Evaluation @epoch 3 - batch 4381: 
2023-05-22 19:44:29,938 P1124 INFO [Metrics] AUC: 0.726343 - gAUC: 0.679697
2023-05-22 19:44:29,939 P1124 INFO Save best model: monitor(max)=1.406041
2023-05-22 19:44:31,867 P1124 INFO ************ Epoch=3 end ************
2023-05-22 19:55:37,461 P1124 INFO Train loss: 0.433986
2023-05-22 19:55:37,461 P1124 INFO Evaluation @epoch 4 - batch 4381: 
2023-05-22 19:58:09,714 P1124 INFO [Metrics] AUC: 0.726451 - gAUC: 0.680044
2023-05-22 19:58:09,721 P1124 INFO Save best model: monitor(max)=1.406495
2023-05-22 19:58:11,601 P1124 INFO ************ Epoch=4 end ************
2023-05-22 20:09:37,210 P1124 INFO Train loss: 0.432523
2023-05-22 20:09:37,210 P1124 INFO Evaluation @epoch 5 - batch 4381: 
2023-05-22 20:12:17,393 P1124 INFO [Metrics] AUC: 0.728707 - gAUC: 0.682796
2023-05-22 20:12:17,399 P1124 INFO Save best model: monitor(max)=1.411503
2023-05-22 20:12:19,360 P1124 INFO ************ Epoch=5 end ************
2023-05-22 20:23:20,123 P1124 INFO Train loss: 0.431104
2023-05-22 20:23:20,123 P1124 INFO Evaluation @epoch 6 - batch 4381: 
2023-05-22 20:25:45,037 P1124 INFO [Metrics] AUC: 0.726938 - gAUC: 0.681092
2023-05-22 20:25:45,043 P1124 INFO Monitor(max)=1.408031 STOP!
2023-05-22 20:25:45,043 P1124 INFO Reduce learning rate on plateau: 0.000050
2023-05-22 20:25:45,089 P1124 INFO ************ Epoch=6 end ************
2023-05-22 20:36:58,858 P1124 INFO Train loss: 0.419381
2023-05-22 20:36:58,859 P1124 INFO Evaluation @epoch 7 - batch 4381: 
2023-05-22 20:39:21,346 P1124 INFO [Metrics] AUC: 0.733174 - gAUC: 0.687910
2023-05-22 20:39:21,347 P1124 INFO Save best model: monitor(max)=1.421083
2023-05-22 20:39:23,250 P1124 INFO ************ Epoch=7 end ************
2023-05-22 20:50:25,049 P1124 INFO Train loss: 0.414647
2023-05-22 20:50:25,050 P1124 INFO Evaluation @epoch 8 - batch 4381: 
2023-05-22 20:52:32,567 P1124 INFO [Metrics] AUC: 0.732513 - gAUC: 0.687943
2023-05-22 20:52:32,569 P1124 INFO Monitor(max)=1.420457 STOP!
2023-05-22 20:52:32,569 P1124 INFO Reduce learning rate on plateau: 0.000005
2023-05-22 20:52:32,627 P1124 INFO ************ Epoch=8 end ************
2023-05-22 21:03:32,610 P1124 INFO Train loss: 0.410793
2023-05-22 21:03:32,610 P1124 INFO Evaluation @epoch 9 - batch 4381: 
2023-05-22 21:05:56,759 P1124 INFO [Metrics] AUC: 0.732347 - gAUC: 0.687945
2023-05-22 21:05:56,760 P1124 INFO Monitor(max)=1.420292 STOP!
2023-05-22 21:05:56,760 P1124 INFO Reduce learning rate on plateau: 0.000001
2023-05-22 21:05:56,760 P1124 INFO ********* Epoch==9 early stop *********
2023-05-22 21:05:56,819 P1124 INFO Training finished.
2023-05-22 21:05:56,820 P1124 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DMIN_microvideo1.7m_x1/microvideo1.7m_x1_83b31456/DMIN_microvideo1.7m_x1_002_27dc4206.model
2023-05-22 21:05:57,423 P1124 INFO ****** Validation evaluation ******
2023-05-22 21:08:41,126 P1124 INFO [Metrics] gAUC: 0.687910 - AUC: 0.733174 - logloss: 0.411456
2023-05-22 21:08:41,252 P1124 INFO ******** Test evaluation ********
2023-05-22 21:08:41,252 P1124 INFO Loading data...
2023-05-22 21:08:41,252 P1124 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_83b31456/test.h5
2023-05-22 21:08:45,539 P1124 INFO Test samples: total/3767308, blocks/1
2023-05-22 21:08:45,539 P1124 INFO Loading test data done.
2023-05-22 21:11:01,894 P1124 INFO [Metrics] gAUC: 0.687910 - AUC: 0.733174 - logloss: 0.411456

```
