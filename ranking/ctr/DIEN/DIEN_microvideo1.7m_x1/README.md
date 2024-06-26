## DIEN_microvideo1.7m_x1

A hands-on guide to run the DIEN model on the MicroVideo1.7M_x1 dataset.

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
Please refer to [MicroVideo1.7M_x1](https://github.com/reczoo/Datasets/tree/main/MicroVideo/MicroVideo1.7M_x1) to get the dataset details.

### Code

We use the [DIEN](https://github.com/reczoo/FuxiCTR/blob/v2.0.2/model_zoo/DIEN) model code from [FuxiCTR-v2.0.2](https://github.com/reczoo/FuxiCTR/tree/v2.0.2) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.2.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.2
    ```

2. Create a data directory and put the downloaded data files in `../data/MicroVideo1.7M/MicroVideo1.7M_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DIEN_microvideo1.7m_x1_tuner_config_03](./DIEN_microvideo1.7m_x1_tuner_config_03). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DIEN
    nohup python run_expid.py --config XXX/benchmarks/DIEN/DIEN_microvideo1.7m_x1_tuner_config_03 --expid DIEN_microvideo1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.686720 | 0.732075 | 0.412213  |


### Logs
```python
2023-05-13 18:57:47,811 P65602 INFO Params: {
    "attention_activation": "Dice",
    "attention_dropout": "0.2",
    "attention_hidden_units": "[256, 128]",
    "attention_type": "din_attention",
    "aux_activation": "ReLU",
    "aux_hidden_units": "[100, 50]",
    "aux_loss_alpha": "0",
    "batch_norm": "True",
    "batch_size": "2048",
    "data_format": "csv",
    "data_root": "../data/MicroVideo1.7M/",
    "dataset_id": "microvideo1.7m_x1_83b31456",
    "debug_mode": "False",
    "dien_neg_seq_field": "[]",
    "dien_sequence_field": "('clicked_items', 'clicked_categories')",
    "dien_target_field": "('item_id', 'cate_id')",
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
    "gpu": "3",
    "group_id": "group_id",
    "gru_type": "AUGRU",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DIEN",
    "model_id": "DIEN_microvideo1.7m_x1_013_a9650956",
    "model_root": "./checkpoints/DIEN_microvideo1.7m_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2022",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "train_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/train.csv",
    "use_attention_softmax": "True",
    "use_features": "None",
    "valid_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "verbose": "1"
}
2023-05-13 18:57:47,812 P65602 INFO Set up feature processor...
2023-05-13 18:57:47,812 P65602 WARNING Skip rebuilding ../data/MicroVideo1.7M/microvideo1.7m_x1_83b31456/feature_map.json. Please delete it manually if rebuilding is required.
2023-05-13 18:57:47,812 P65602 INFO Load feature_map from json: ../data/MicroVideo1.7M/microvideo1.7m_x1_83b31456/feature_map.json
2023-05-13 18:57:47,812 P65602 INFO Set column index...
2023-05-13 18:57:47,812 P65602 INFO Feature specs: {
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514}",
    "clicked_categories": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514, 'max_len': 100}",
    "clicked_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'max_len': 100}",
    "group_id": "{'type': 'meta'}",
    "item_id": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10987, 'vocab_size': 10988}"
}
2023-05-13 18:57:54,072 P65602 INFO Total number of parameters: 2095874.
2023-05-13 18:57:54,072 P65602 INFO Loading data...
2023-05-13 18:57:54,072 P65602 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_83b31456/train.h5
2023-05-13 18:58:03,311 P65602 INFO Train samples: total/8970309, blocks/1
2023-05-13 18:58:03,312 P65602 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_83b31456/valid.h5
2023-05-13 18:58:07,151 P65602 INFO Validation samples: total/3767308, blocks/1
2023-05-13 18:58:07,151 P65602 INFO Loading train and validation data done.
2023-05-13 18:58:07,151 P65602 INFO Start training: 4381 batches/epoch
2023-05-13 18:58:07,151 P65602 INFO ************ Epoch=1 start ************
2023-05-13 19:36:04,783 P65602 INFO Train loss: 0.460876
2023-05-13 19:36:04,784 P65602 INFO Evaluation @epoch 1 - batch 4381: 
2023-05-13 19:40:30,562 P65602 INFO [Metrics] AUC: 0.716681 - gAUC: 0.671589
2023-05-13 19:40:30,565 P65602 INFO Save best model: monitor(max)=1.388270
2023-05-13 19:40:32,211 P65602 INFO ************ Epoch=1 end ************
2023-05-13 20:18:17,570 P65602 INFO Train loss: 0.441729
2023-05-13 20:18:17,571 P65602 INFO Evaluation @epoch 2 - batch 4381: 
2023-05-13 20:22:04,994 P65602 INFO [Metrics] AUC: 0.721702 - gAUC: 0.675033
2023-05-13 20:22:04,998 P65602 INFO Save best model: monitor(max)=1.396735
2023-05-13 20:22:07,045 P65602 INFO ************ Epoch=2 end ************
2023-05-13 20:59:45,446 P65602 INFO Train loss: 0.438212
2023-05-13 20:59:45,446 P65602 INFO Evaluation @epoch 3 - batch 4381: 
2023-05-13 21:03:50,151 P65602 INFO [Metrics] AUC: 0.722878 - gAUC: 0.676235
2023-05-13 21:03:50,159 P65602 INFO Save best model: monitor(max)=1.399113
2023-05-13 21:03:52,237 P65602 INFO ************ Epoch=3 end ************
2023-05-13 21:41:29,931 P65602 INFO Train loss: 0.436219
2023-05-13 21:41:29,932 P65602 INFO Evaluation @epoch 4 - batch 4381: 
2023-05-13 21:45:41,656 P65602 INFO [Metrics] AUC: 0.725884 - gAUC: 0.680138
2023-05-13 21:45:41,669 P65602 INFO Save best model: monitor(max)=1.406022
2023-05-13 21:45:43,775 P65602 INFO ************ Epoch=4 end ************
2023-05-13 22:23:20,314 P65602 INFO Train loss: 0.434200
2023-05-13 22:23:20,315 P65602 INFO Evaluation @epoch 5 - batch 4381: 
2023-05-13 22:27:36,937 P65602 INFO [Metrics] AUC: 0.726550 - gAUC: 0.679151
2023-05-13 22:27:36,944 P65602 INFO Monitor(max)=1.405701 STOP!
2023-05-13 22:27:36,945 P65602 INFO Reduce learning rate on plateau: 0.000050
2023-05-13 22:27:37,041 P65602 INFO ************ Epoch=5 end ************
2023-05-13 23:05:13,581 P65602 INFO Train loss: 0.424132
2023-05-13 23:05:13,582 P65602 INFO Evaluation @epoch 6 - batch 4381: 
2023-05-13 23:09:31,603 P65602 INFO [Metrics] AUC: 0.731975 - gAUC: 0.685804
2023-05-13 23:09:31,604 P65602 INFO Save best model: monitor(max)=1.417779
2023-05-13 23:09:33,603 P65602 INFO ************ Epoch=6 end ************
2023-05-13 23:47:07,953 P65602 INFO Train loss: 0.419954
2023-05-13 23:47:07,954 P65602 INFO Evaluation @epoch 7 - batch 4381: 
2023-05-13 23:51:26,854 P65602 INFO [Metrics] AUC: 0.732008 - gAUC: 0.686177
2023-05-13 23:51:26,862 P65602 INFO Save best model: monitor(max)=1.418184
2023-05-13 23:51:28,966 P65602 INFO ************ Epoch=7 end ************
2023-05-14 00:29:10,775 P65602 INFO Train loss: 0.418204
2023-05-14 00:29:10,776 P65602 INFO Evaluation @epoch 8 - batch 4381: 
2023-05-14 00:33:20,302 P65602 INFO [Metrics] AUC: 0.731960 - gAUC: 0.686279
2023-05-14 00:33:20,305 P65602 INFO Save best model: monitor(max)=1.418240
2023-05-14 00:33:22,331 P65602 INFO ************ Epoch=8 end ************
2023-05-14 01:11:02,902 P65602 INFO Train loss: 0.416970
2023-05-14 01:11:02,903 P65602 INFO Evaluation @epoch 9 - batch 4381: 
2023-05-14 01:15:08,407 P65602 INFO [Metrics] AUC: 0.732099 - gAUC: 0.686315
2023-05-14 01:15:08,410 P65602 INFO Save best model: monitor(max)=1.418414
2023-05-14 01:15:10,457 P65602 INFO ************ Epoch=9 end ************
2023-05-14 01:52:45,812 P65602 INFO Train loss: 0.415838
2023-05-14 01:52:45,813 P65602 INFO Evaluation @epoch 10 - batch 4381: 
2023-05-14 01:56:48,051 P65602 INFO [Metrics] AUC: 0.732075 - gAUC: 0.686720
2023-05-14 01:56:48,057 P65602 INFO Save best model: monitor(max)=1.418795
2023-05-14 01:56:50,174 P65602 INFO ************ Epoch=10 end ************
2023-05-14 02:34:19,979 P65602 INFO Train loss: 0.414914
2023-05-14 02:34:19,980 P65602 INFO Evaluation @epoch 11 - batch 4381: 
2023-05-14 02:38:00,313 P65602 INFO [Metrics] AUC: 0.731416 - gAUC: 0.686012
2023-05-14 02:38:00,314 P65602 INFO Monitor(max)=1.417428 STOP!
2023-05-14 02:38:00,314 P65602 INFO Reduce learning rate on plateau: 0.000005
2023-05-14 02:38:00,384 P65602 INFO ************ Epoch=11 end ************
2023-05-14 03:15:30,207 P65602 INFO Train loss: 0.410890
2023-05-14 03:15:30,208 P65602 INFO Evaluation @epoch 12 - batch 4381: 
2023-05-14 03:19:24,128 P65602 INFO [Metrics] AUC: 0.731822 - gAUC: 0.686731
2023-05-14 03:19:24,131 P65602 INFO Monitor(max)=1.418553 STOP!
2023-05-14 03:19:24,131 P65602 INFO Reduce learning rate on plateau: 0.000001
2023-05-14 03:19:24,131 P65602 INFO ********* Epoch==12 early stop *********
2023-05-14 03:19:24,220 P65602 INFO Training finished.
2023-05-14 03:19:24,221 P65602 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DIEN_microvideo1.7m_x1/microvideo1.7m_x1_83b31456/DIEN_microvideo1.7m_x1_013_a9650956.model
2023-05-14 03:19:24,915 P65602 INFO ****** Validation evaluation ******
2023-05-14 03:22:54,366 P65602 INFO [Metrics] gAUC: 0.686720 - AUC: 0.732075 - logloss: 0.412213
2023-05-14 03:22:54,484 P65602 INFO ******** Test evaluation ********
2023-05-14 03:22:54,485 P65602 INFO Loading data...
2023-05-14 03:22:54,485 P65602 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_83b31456/test.h5
2023-05-14 03:22:58,956 P65602 INFO Test samples: total/3767308, blocks/1
2023-05-14 03:22:58,956 P65602 INFO Loading test data done.
2023-05-14 03:26:40,644 P65602 INFO [Metrics] gAUC: 0.686720 - AUC: 0.732075 - logloss: 0.412213

```
