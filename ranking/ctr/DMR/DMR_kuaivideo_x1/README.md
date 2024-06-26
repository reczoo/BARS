## DMR_kuaivideo_x1

A hands-on guide to run the DMR model on the KuaiVideo_x1 dataset.

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

We use the [DMR](https://github.com/reczoo/FuxiCTR/tree/v2.0.3/model_zoo/DMR) model code from [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/tree/v2.0.3) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.3.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.3
    ```

2. Create a data directory and put the downloaded data files in `../data/KuaiShou/KuaiVideo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DMR_kuaivideo_x1_tuner_config_01](./DMR_kuaivideo_x1_tuner_config_01). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DMR
    nohup python run_expid.py --config YOUR_PATH/DMR/DMR_kuaivideo_x1_tuner_config_01 --expid DMR_kuaivideo_x1_003_e9fb63c2 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.668880 | 0.748489 | 0.435421  |


### Logs
```python
2023-05-24 14:36:57,418 P91467 INFO Params: {
    "attention_activation": "ReLU",
    "attention_dropout": "0.1",
    "attention_hidden_units": "[512, 256]",
    "aux_loss_beta": "0",
    "batch_norm": "False",
    "batch_size": "4096",
    "bn_only_once": "False",
    "context_field": "None",
    "data_format": "csv",
    "data_root": "../data/KuaiShou/",
    "dataset_id": "kuaivideo_x1_60f6c91a",
    "debug_mode": "False",
    "dnn_activations": "Dice",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.0001",
    "enable_i2i_rel": "False",
    "enable_sum_pooling": "False",
    "enable_u2i_rel": "False",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'min_categr_count': 1, 'name': 'item_emb', 'preprocess': 'copy_from(item_id)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'pos_items', 'padding': 'post', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'neg_items', 'padding': 'post', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'pos_items_emb', 'padding': 'post', 'preprocess': 'copy_from(pos_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'neg_items_emb', 'padding': 'post', 'preprocess': 'copy_from(neg_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}]",
    "feature_config": "None",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_emb'}, {'feature_encoder': None, 'name': 'pos_items'}, {'feature_encoder': None, 'name': 'neg_items'}, {'feature_encoder': ['nn.Linear(64, 64, bias=False)'], 'name': 'pos_items_emb'}, {'feature_encoder': ['nn.Linear(64, 64, bias=False)'], 'name': 'neg_items_emb'}]",
    "gpu": "2",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "DMR",
    "model_id": "DMR_kuaivideo_x1_003_e9fb63c2",
    "model_root": "./checkpoints/DMR_kuaivideo_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "neg_seq_field": "None",
    "net_dropout": "0.1",
    "net_regularizer": "0",
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
    "use_features": "None",
    "use_pos_emb": "True",
    "valid_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "verbose": "1"
}
2023-05-24 14:36:57,418 P91467 INFO Set up feature processor...
2023-05-24 14:36:57,418 P91467 WARNING Skip rebuilding ../data/KuaiShou/kuaivideo_x1_60f6c91a/feature_map.json. Please delete it manually if rebuilding is required.
2023-05-24 14:36:57,419 P91467 INFO Load feature_map from json: ../data/KuaiShou/kuaivideo_x1_60f6c91a/feature_map.json
2023-05-24 14:36:57,419 P91467 INFO Set column index...
2023-05-24 14:36:57,419 P91467 INFO Feature specs: {
    "group_id": "{'type': 'meta'}",
    "item_emb": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406}",
    "neg_items": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "neg_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "pos_items": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "pos_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10001, 'vocab_size': 10002}"
}
2023-05-24 14:37:09,654 P91467 INFO Total number of parameters: 291051779.
2023-05-24 14:37:09,654 P91467 INFO Loading data...
2023-05-24 14:37:09,654 P91467 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_60f6c91a/train.h5
2023-05-24 14:37:33,165 P91467 INFO Train samples: total/10931092, blocks/1
2023-05-24 14:37:33,165 P91467 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_60f6c91a/valid.h5
2023-05-24 14:37:39,150 P91467 INFO Validation samples: total/2730291, blocks/1
2023-05-24 14:37:39,150 P91467 INFO Loading train and validation data done.
2023-05-24 14:37:39,150 P91467 INFO Start training: 2669 batches/epoch
2023-05-24 14:37:39,150 P91467 INFO ************ Epoch=1 start ************
2023-05-24 14:49:58,378 P91467 INFO Train loss: 0.459075
2023-05-24 14:49:58,378 P91467 INFO Evaluation @epoch 1 - batch 2669: 
2023-05-24 14:51:25,490 P91467 INFO [Metrics] AUC: 0.727751 - gAUC: 0.642543
2023-05-24 14:51:25,494 P91467 INFO Save best model: monitor(max)=1.370294
2023-05-24 14:51:29,482 P91467 INFO ************ Epoch=1 end ************
2023-05-24 15:03:47,690 P91467 INFO Train loss: 0.446603
2023-05-24 15:03:47,690 P91467 INFO Evaluation @epoch 2 - batch 2669: 
2023-05-24 15:05:46,484 P91467 INFO [Metrics] AUC: 0.737074 - gAUC: 0.653036
2023-05-24 15:05:46,485 P91467 INFO Save best model: monitor(max)=1.390110
2023-05-24 15:05:51,081 P91467 INFO ************ Epoch=2 end ************
2023-05-24 15:18:06,662 P91467 INFO Train loss: 0.442383
2023-05-24 15:18:06,663 P91467 INFO Evaluation @epoch 3 - batch 2669: 
2023-05-24 15:19:36,900 P91467 INFO [Metrics] AUC: 0.739617 - gAUC: 0.655949
2023-05-24 15:19:36,909 P91467 INFO Save best model: monitor(max)=1.395566
2023-05-24 15:19:41,223 P91467 INFO ************ Epoch=3 end ************
2023-05-24 15:32:04,674 P91467 INFO Train loss: 0.440136
2023-05-24 15:32:04,679 P91467 INFO Evaluation @epoch 4 - batch 2669: 
2023-05-24 15:34:01,025 P91467 INFO [Metrics] AUC: 0.741349 - gAUC: 0.658560
2023-05-24 15:34:01,029 P91467 INFO Save best model: monitor(max)=1.399909
2023-05-24 15:34:05,296 P91467 INFO ************ Epoch=4 end ************
2023-05-24 15:46:21,029 P91467 INFO Train loss: 0.438955
2023-05-24 15:46:21,029 P91467 INFO Evaluation @epoch 5 - batch 2669: 
2023-05-24 15:47:47,997 P91467 INFO [Metrics] AUC: 0.741713 - gAUC: 0.659622
2023-05-24 15:47:48,000 P91467 INFO Save best model: monitor(max)=1.401335
2023-05-24 15:47:52,284 P91467 INFO ************ Epoch=5 end ************
2023-05-24 16:00:15,145 P91467 INFO Train loss: 0.437948
2023-05-24 16:00:15,146 P91467 INFO Evaluation @epoch 6 - batch 2669: 
2023-05-24 16:02:03,322 P91467 INFO [Metrics] AUC: 0.743431 - gAUC: 0.661889
2023-05-24 16:02:03,324 P91467 INFO Save best model: monitor(max)=1.405321
2023-05-24 16:02:07,715 P91467 INFO ************ Epoch=6 end ************
2023-05-24 16:14:24,791 P91467 INFO Train loss: 0.437077
2023-05-24 16:14:24,792 P91467 INFO Evaluation @epoch 7 - batch 2669: 
2023-05-24 16:15:53,270 P91467 INFO [Metrics] AUC: 0.742143 - gAUC: 0.660690
2023-05-24 16:15:53,272 P91467 INFO Monitor(max)=1.402833 STOP!
2023-05-24 16:15:53,272 P91467 INFO Reduce learning rate on plateau: 0.000100
2023-05-24 16:15:53,335 P91467 INFO ************ Epoch=7 end ************
2023-05-24 16:28:10,982 P91467 INFO Train loss: 0.416633
2023-05-24 16:28:10,982 P91467 INFO Evaluation @epoch 8 - batch 2669: 
2023-05-24 16:29:58,291 P91467 INFO [Metrics] AUC: 0.747997 - gAUC: 0.667941
2023-05-24 16:29:58,294 P91467 INFO Save best model: monitor(max)=1.415937
2023-05-24 16:30:02,945 P91467 INFO ************ Epoch=8 end ************
2023-05-24 16:42:14,753 P91467 INFO Train loss: 0.409170
2023-05-24 16:42:14,754 P91467 INFO Evaluation @epoch 9 - batch 2669: 
2023-05-24 16:43:33,693 P91467 INFO [Metrics] AUC: 0.748489 - gAUC: 0.668880
2023-05-24 16:43:33,697 P91467 INFO Save best model: monitor(max)=1.417369
2023-05-24 16:43:37,898 P91467 INFO ************ Epoch=9 end ************
2023-05-24 16:55:48,721 P91467 INFO Train loss: 0.404871
2023-05-24 16:55:48,722 P91467 INFO Evaluation @epoch 10 - batch 2669: 
2023-05-24 16:57:17,796 P91467 INFO [Metrics] AUC: 0.748279 - gAUC: 0.668635
2023-05-24 16:57:17,802 P91467 INFO Monitor(max)=1.416913 STOP!
2023-05-24 16:57:17,802 P91467 INFO Reduce learning rate on plateau: 0.000010
2023-05-24 16:57:17,858 P91467 INFO ************ Epoch=10 end ************
2023-05-24 17:09:31,137 P91467 INFO Train loss: 0.394691
2023-05-24 17:09:31,138 P91467 INFO Evaluation @epoch 11 - batch 2669: 
2023-05-24 17:10:50,531 P91467 INFO [Metrics] AUC: 0.746504 - gAUC: 0.666924
2023-05-24 17:10:50,537 P91467 INFO Monitor(max)=1.413428 STOP!
2023-05-24 17:10:50,537 P91467 INFO Reduce learning rate on plateau: 0.000001
2023-05-24 17:10:50,537 P91467 INFO ********* Epoch==11 early stop *********
2023-05-24 17:10:50,591 P91467 INFO Training finished.
2023-05-24 17:10:50,591 P91467 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DMR_kuaivideo_x1/kuaivideo_x1_60f6c91a/DMR_kuaivideo_x1_003_e9fb63c2.model
2023-05-24 17:10:52,289 P91467 INFO ****** Validation evaluation ******
2023-05-24 17:12:13,072 P91467 INFO [Metrics] gAUC: 0.668880 - AUC: 0.748489 - logloss: 0.435421
2023-05-24 17:12:13,256 P91467 INFO ******** Test evaluation ********
2023-05-24 17:12:13,256 P91467 INFO Loading data...
2023-05-24 17:12:13,257 P91467 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_60f6c91a/test.h5
2023-05-24 17:12:19,203 P91467 INFO Test samples: total/2730291, blocks/1
2023-05-24 17:12:19,203 P91467 INFO Loading test data done.
2023-05-24 17:13:37,761 P91467 INFO [Metrics] gAUC: 0.668880 - AUC: 0.748489 - logloss: 0.435421

```
