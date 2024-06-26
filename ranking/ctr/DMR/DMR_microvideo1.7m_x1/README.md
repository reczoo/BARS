## DMR_microvideo1.7m_x1

A hands-on guide to run the DMR model on the MicroVideo1.7M_x1 dataset.

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

We use the [DMR](https://github.com/reczoo/FuxiCTR/tree/v2.0.3/model_zoo/DMR) model code from [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/tree/v2.0.3) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.3.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.3
    ```

2. Create a data directory and put the downloaded data files in `../data/MicroVideo/MicroVideo1.7M_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DMR_microvideo1.7m_x1_tuner_config_01](./DMR_microvideo1.7m_x1_tuner_config_01). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DMR
    nohup python run_expid.py --config YOUR_PATH/DMR/DMR_microvideo1.7m_x1_tuner_config_01 --expid DMR_microvideo1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.687533 | 0.735436 | 0.412249  |


### Logs
```python
2023-05-24 14:41:47,007 P105461 INFO Params: {
    "attention_activation": "Dice",
    "attention_dropout": "0.2",
    "attention_hidden_units": "[512, 256]",
    "aux_loss_beta": "0",
    "batch_norm": "True",
    "batch_size": "2048",
    "bn_only_once": "False",
    "context_field": "None",
    "data_format": "csv",
    "data_root": "../data/MicroVideo1.7M/",
    "dataset_id": "microvideo1.7m_x1_83b31456",
    "debug_mode": "False",
    "dnn_activations": "ReLU",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.0005",
    "enable_i2i_rel": "False",
    "enable_sum_pooling": "False",
    "enable_u2i_rel": "False",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'name': 'clicked_items', 'padding': 'post', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'clicked_categories', 'padding': 'post', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "feature_config": "None",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id'}, {'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'clicked_items'}, {'feature_encoder': None, 'name': 'clicked_categories'}]",
    "gpu": "7",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DMR",
    "model_id": "DMR_microvideo1.7m_x1_008_dac65736",
    "model_root": "./checkpoints/DMR_microvideo1.7m_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "neg_seq_field": "None",
    "net_dropout": "0.1",
    "net_regularizer": "0",
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
    "use_features": "None",
    "use_pos_emb": "False",
    "valid_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "verbose": "1"
}
2023-05-24 14:41:47,007 P105461 INFO Set up feature processor...
2023-05-24 14:41:47,007 P105461 WARNING Skip rebuilding ../data/MicroVideo1.7M/microvideo1.7m_x1_83b31456/feature_map.json. Please delete it manually if rebuilding is required.
2023-05-24 14:41:47,007 P105461 INFO Load feature_map from json: ../data/MicroVideo1.7M/microvideo1.7m_x1_83b31456/feature_map.json
2023-05-24 14:41:47,008 P105461 INFO Set column index...
2023-05-24 14:41:47,008 P105461 INFO Feature specs: {
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514}",
    "clicked_categories": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514, 'max_len': 100}",
    "clicked_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'max_len': 100}",
    "group_id": "{'type': 'meta'}",
    "item_id": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10987, 'vocab_size': 10988}"
}
2023-05-24 14:41:54,935 P105461 INFO Total number of parameters: 111289858.
2023-05-24 14:41:54,935 P105461 INFO Loading data...
2023-05-24 14:41:54,936 P105461 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_83b31456/train.h5
2023-05-24 14:42:05,019 P105461 INFO Train samples: total/8970309, blocks/1
2023-05-24 14:42:05,020 P105461 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_83b31456/valid.h5
2023-05-24 14:42:09,244 P105461 INFO Validation samples: total/3767308, blocks/1
2023-05-24 14:42:09,245 P105461 INFO Loading train and validation data done.
2023-05-24 14:42:09,245 P105461 INFO Start training: 4381 batches/epoch
2023-05-24 14:42:09,245 P105461 INFO ************ Epoch=1 start ************
2023-05-24 14:50:52,629 P105461 INFO Train loss: 0.458405
2023-05-24 14:50:52,632 P105461 INFO Evaluation @epoch 1 - batch 4381: 
2023-05-24 14:53:40,779 P105461 INFO [Metrics] AUC: 0.718218 - gAUC: 0.673152
2023-05-24 14:53:40,780 P105461 INFO Save best model: monitor(max)=1.391369
2023-05-24 14:53:43,170 P105461 INFO ************ Epoch=1 end ************
2023-05-24 15:03:15,508 P105461 INFO Train loss: 0.440663
2023-05-24 15:03:15,509 P105461 INFO Evaluation @epoch 2 - batch 4381: 
2023-05-24 15:06:03,405 P105461 INFO [Metrics] AUC: 0.724735 - gAUC: 0.678255
2023-05-24 15:06:03,406 P105461 INFO Save best model: monitor(max)=1.402990
2023-05-24 15:06:06,470 P105461 INFO ************ Epoch=2 end ************
2023-05-24 15:15:01,613 P105461 INFO Train loss: 0.436843
2023-05-24 15:15:01,613 P105461 INFO Evaluation @epoch 3 - batch 4381: 
2023-05-24 15:17:43,544 P105461 INFO [Metrics] AUC: 0.725490 - gAUC: 0.678947
2023-05-24 15:17:43,549 P105461 INFO Save best model: monitor(max)=1.404437
2023-05-24 15:17:46,498 P105461 INFO ************ Epoch=3 end ************
2023-05-24 15:27:27,217 P105461 INFO Train loss: 0.434482
2023-05-24 15:27:27,217 P105461 INFO Evaluation @epoch 4 - batch 4381: 
2023-05-24 15:30:03,830 P105461 INFO [Metrics] AUC: 0.727900 - gAUC: 0.680990
2023-05-24 15:30:03,841 P105461 INFO Save best model: monitor(max)=1.408890
2023-05-24 15:30:06,978 P105461 INFO ************ Epoch=4 end ************
2023-05-24 15:39:14,581 P105461 INFO Train loss: 0.432584
2023-05-24 15:39:14,581 P105461 INFO Evaluation @epoch 5 - batch 4381: 
2023-05-24 15:41:52,382 P105461 INFO [Metrics] AUC: 0.728434 - gAUC: 0.681701
2023-05-24 15:41:52,384 P105461 INFO Save best model: monitor(max)=1.410135
2023-05-24 15:41:55,444 P105461 INFO ************ Epoch=5 end ************
2023-05-24 15:51:46,046 P105461 INFO Train loss: 0.431454
2023-05-24 15:51:46,046 P105461 INFO Evaluation @epoch 6 - batch 4381: 
2023-05-24 15:54:16,682 P105461 INFO [Metrics] AUC: 0.728414 - gAUC: 0.680855
2023-05-24 15:54:16,687 P105461 INFO Monitor(max)=1.409269 STOP!
2023-05-24 15:54:16,687 P105461 INFO Reduce learning rate on plateau: 0.000050
2023-05-24 15:54:16,744 P105461 INFO ************ Epoch=6 end ************
2023-05-24 16:03:31,668 P105461 INFO Train loss: 0.420687
2023-05-24 16:03:31,669 P105461 INFO Evaluation @epoch 7 - batch 4381: 
2023-05-24 16:06:00,820 P105461 INFO [Metrics] AUC: 0.734900 - gAUC: 0.687279
2023-05-24 16:06:00,823 P105461 INFO Save best model: monitor(max)=1.422179
2023-05-24 16:06:03,865 P105461 INFO ************ Epoch=7 end ************
2023-05-24 16:16:39,685 P105461 INFO Train loss: 0.416460
2023-05-24 16:16:39,686 P105461 INFO Evaluation @epoch 8 - batch 4381: 
2023-05-24 16:18:51,248 P105461 INFO [Metrics] AUC: 0.735273 - gAUC: 0.687546
2023-05-24 16:18:51,253 P105461 INFO Save best model: monitor(max)=1.422818
2023-05-24 16:18:54,415 P105461 INFO ************ Epoch=8 end ************
2023-05-24 16:27:32,459 P105461 INFO Train loss: 0.414496
2023-05-24 16:27:32,459 P105461 INFO Evaluation @epoch 9 - batch 4381: 
2023-05-24 16:29:48,733 P105461 INFO [Metrics] AUC: 0.735108 - gAUC: 0.687118
2023-05-24 16:29:48,737 P105461 INFO Monitor(max)=1.422227 STOP!
2023-05-24 16:29:48,737 P105461 INFO Reduce learning rate on plateau: 0.000005
2023-05-24 16:29:48,790 P105461 INFO ************ Epoch=9 end ************
2023-05-24 16:38:39,592 P105461 INFO Train loss: 0.410926
2023-05-24 16:38:39,593 P105461 INFO Evaluation @epoch 10 - batch 4381: 
2023-05-24 16:40:39,490 P105461 INFO [Metrics] AUC: 0.735421 - gAUC: 0.687478
2023-05-24 16:40:39,493 P105461 INFO Save best model: monitor(max)=1.422899
2023-05-24 16:40:42,508 P105461 INFO ************ Epoch=10 end ************
2023-05-24 16:49:19,371 P105461 INFO Train loss: 0.410466
2023-05-24 16:49:19,372 P105461 INFO Evaluation @epoch 11 - batch 4381: 
2023-05-24 16:51:45,219 P105461 INFO [Metrics] AUC: 0.735436 - gAUC: 0.687533
2023-05-24 16:51:45,224 P105461 INFO Save best model: monitor(max)=1.422969
2023-05-24 16:51:48,173 P105461 INFO ************ Epoch=11 end ************
2023-05-24 17:00:32,398 P105461 INFO Train loss: 0.410068
2023-05-24 17:00:32,398 P105461 INFO Evaluation @epoch 12 - batch 4381: 
2023-05-24 17:02:33,600 P105461 INFO [Metrics] AUC: 0.735284 - gAUC: 0.687278
2023-05-24 17:02:33,604 P105461 INFO Monitor(max)=1.422562 STOP!
2023-05-24 17:02:33,604 P105461 INFO Reduce learning rate on plateau: 0.000001
2023-05-24 17:02:33,657 P105461 INFO ************ Epoch=12 end ************
2023-05-24 17:11:30,346 P105461 INFO Train loss: 0.409650
2023-05-24 17:11:30,346 P105461 INFO Evaluation @epoch 13 - batch 4381: 
2023-05-24 17:12:52,070 P105461 INFO [Metrics] AUC: 0.735319 - gAUC: 0.687272
2023-05-24 17:12:52,072 P105461 INFO Monitor(max)=1.422591 STOP!
2023-05-24 17:12:52,072 P105461 INFO Reduce learning rate on plateau: 0.000001
2023-05-24 17:12:52,072 P105461 INFO ********* Epoch==13 early stop *********
2023-05-24 17:12:52,122 P105461 INFO Training finished.
2023-05-24 17:12:52,122 P105461 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DMR_microvideo1.7m_x1/microvideo1.7m_x1_83b31456/DMR_microvideo1.7m_x1_008_dac65736.model
2023-05-24 17:12:53,121 P105461 INFO ****** Validation evaluation ******
2023-05-24 17:14:13,474 P105461 INFO [Metrics] gAUC: 0.687533 - AUC: 0.735436 - logloss: 0.412249
2023-05-24 17:14:13,565 P105461 INFO ******** Test evaluation ********
2023-05-24 17:14:13,565 P105461 INFO Loading data...
2023-05-24 17:14:13,565 P105461 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_83b31456/test.h5
2023-05-24 17:14:17,811 P105461 INFO Test samples: total/3767308, blocks/1
2023-05-24 17:14:17,811 P105461 INFO Loading test data done.
2023-05-24 17:15:36,967 P105461 INFO [Metrics] gAUC: 0.687533 - AUC: 0.735436 - logloss: 0.412249

```
