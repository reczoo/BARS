## DMR_amazonelectronics_x1

A hands-on guide to run the DMR model on the AmazonElectronics_x1 dataset.

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
Please refer to [AmazonElectronics_x1](https://github.com/reczoo/Datasets/tree/main/Amazon/AmazonElectronics_x1) to get the dataset details.

### Code

We use the [DMR](https://github.com/reczoo/FuxiCTR/tree/v2.0.3/model_zoo/DMR) model code from [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/tree/v2.0.3) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.3.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.3
    ```

2. Create a data directory and put the downloaded data files in `../data/Amazon/AmazonElectronics_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DMR_amazonelectronics_x1_tuner_config_07](./DMR_amazonelectronics_x1_tuner_config_07). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DMR
    nohup python run_expid.py --config YOUR_PATH/DMR/DMR_amazonelectronics_x1_tuner_config_07 --expid DMR_amazonelectronics_x1_002_bc859be0 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.885142 | 0.887335 | 0.427744  |


### Logs
```python
2023-05-23 21:34:53,431 P18185 INFO Params: {
    "attention_activation": "Dice",
    "attention_dropout": "0.1",
    "attention_hidden_units": "[512, 256]",
    "aux_loss_beta": "0",
    "batch_norm": "True",
    "batch_size": "1024",
    "bn_only_once": "False",
    "context_field": "None",
    "data_format": "csv",
    "data_root": "../data/Amazon/",
    "dataset_id": "amazonelectronics_x1_51836f99",
    "debug_mode": "False",
    "dnn_activations": "ReLU",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.005",
    "enable_i2i_rel": "False",
    "enable_sum_pooling": "False",
    "enable_u2i_rel": "True",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'user_id', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'item_history', 'padding': 'post', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'cate_history', 'padding': 'post', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}]",
    "feature_config": "None",
    "feature_specs": "[{'feature_encoder': None, 'name': 'item_history'}, {'feature_encoder': None, 'name': 'cate_history'}]",
    "gpu": "1",
    "group_id": "user_id",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DMR",
    "model_id": "DMR_amazonelectronics_x1_002_bc859be0",
    "model_root": "./checkpoints/DMR_amazonelectronics_x1/",
    "monitor": "gAUC",
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
    "sequence_field": "('item_history', 'cate_history')",
    "shuffle": "True",
    "target_field": "('item_id', 'cate_id')",
    "task": "binary_classification",
    "test_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "train_data": "../data/Amazon/AmazonElectronics_x1/train.csv",
    "use_features": "None",
    "use_pos_emb": "True",
    "valid_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "verbose": "1"
}
2023-05-23 21:34:53,432 P18185 INFO Set up feature processor...
2023-05-23 21:34:53,432 P18185 WARNING Skip rebuilding ../data/Amazon/amazonelectronics_x1_51836f99/feature_map.json. Please delete it manually if rebuilding is required.
2023-05-23 21:34:53,432 P18185 INFO Load feature_map from json: ../data/Amazon/amazonelectronics_x1_51836f99/feature_map.json
2023-05-23 21:34:53,432 P18185 INFO Set column index...
2023-05-23 21:34:53,432 P18185 INFO Feature specs: {
    "cate_history": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803, 'max_len': 100}",
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803}",
    "item_history": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003, 'max_len': 100}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003}",
    "user_id": "{'type': 'meta'}"
}
2023-05-23 21:35:01,074 P18185 INFO Total number of parameters: 9918147.
2023-05-23 21:35:01,074 P18185 INFO Loading data...
2023-05-23 21:35:01,075 P18185 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_51836f99/train.h5
2023-05-23 21:35:04,778 P18185 INFO Train samples: total/2608764, blocks/1
2023-05-23 21:35:04,778 P18185 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_51836f99/valid.h5
2023-05-23 21:35:05,348 P18185 INFO Validation samples: total/384806, blocks/1
2023-05-23 21:35:05,349 P18185 INFO Loading train and validation data done.
2023-05-23 21:35:05,349 P18185 INFO Start training: 2548 batches/epoch
2023-05-23 21:35:05,349 P18185 INFO ************ Epoch=1 start ************
2023-05-23 21:46:19,091 P18185 INFO Train loss: 0.644987
2023-05-23 21:46:19,092 P18185 INFO Evaluation @epoch 1 - batch 2548: 
2023-05-23 21:48:41,607 P18185 INFO [Metrics] gAUC: 0.833932
2023-05-23 21:48:41,608 P18185 INFO Save best model: monitor(max)=0.833932
2023-05-23 21:48:41,746 P18185 INFO ************ Epoch=1 end ************
2023-05-23 21:59:53,774 P18185 INFO Train loss: 0.593059
2023-05-23 21:59:53,774 P18185 INFO Evaluation @epoch 2 - batch 2548: 
2023-05-23 22:02:13,341 P18185 INFO [Metrics] gAUC: 0.847398
2023-05-23 22:02:13,342 P18185 INFO Save best model: monitor(max)=0.847398
2023-05-23 22:02:13,530 P18185 INFO ************ Epoch=2 end ************
2023-05-23 22:13:28,921 P18185 INFO Train loss: 0.574215
2023-05-23 22:13:28,922 P18185 INFO Evaluation @epoch 3 - batch 2548: 
2023-05-23 22:15:46,619 P18185 INFO [Metrics] gAUC: 0.853116
2023-05-23 22:15:46,620 P18185 INFO Save best model: monitor(max)=0.853116
2023-05-23 22:15:46,796 P18185 INFO ************ Epoch=3 end ************
2023-05-23 22:26:58,887 P18185 INFO Train loss: 0.566249
2023-05-23 22:26:58,887 P18185 INFO Evaluation @epoch 4 - batch 2548: 
2023-05-23 22:29:17,239 P18185 INFO [Metrics] gAUC: 0.856650
2023-05-23 22:29:17,240 P18185 INFO Save best model: monitor(max)=0.856650
2023-05-23 22:29:17,416 P18185 INFO ************ Epoch=4 end ************
2023-05-23 22:40:35,757 P18185 INFO Train loss: 0.562551
2023-05-23 22:40:35,757 P18185 INFO Evaluation @epoch 5 - batch 2548: 
2023-05-23 22:42:54,551 P18185 INFO [Metrics] gAUC: 0.858214
2023-05-23 22:42:54,552 P18185 INFO Save best model: monitor(max)=0.858214
2023-05-23 22:42:54,753 P18185 INFO ************ Epoch=5 end ************
2023-05-23 22:54:09,036 P18185 INFO Train loss: 0.560197
2023-05-23 22:54:09,037 P18185 INFO Evaluation @epoch 6 - batch 2548: 
2023-05-23 22:56:27,023 P18185 INFO [Metrics] gAUC: 0.859514
2023-05-23 22:56:27,024 P18185 INFO Save best model: monitor(max)=0.859514
2023-05-23 22:56:27,372 P18185 INFO ************ Epoch=6 end ************
2023-05-23 23:07:41,177 P18185 INFO Train loss: 0.558907
2023-05-23 23:07:41,178 P18185 INFO Evaluation @epoch 7 - batch 2548: 
2023-05-23 23:09:57,111 P18185 INFO [Metrics] gAUC: 0.859961
2023-05-23 23:09:57,112 P18185 INFO Save best model: monitor(max)=0.859961
2023-05-23 23:09:57,298 P18185 INFO ************ Epoch=7 end ************
2023-05-23 23:21:14,408 P18185 INFO Train loss: 0.557464
2023-05-23 23:21:14,408 P18185 INFO Evaluation @epoch 8 - batch 2548: 
2023-05-23 23:23:32,196 P18185 INFO [Metrics] gAUC: 0.860376
2023-05-23 23:23:32,198 P18185 INFO Save best model: monitor(max)=0.860376
2023-05-23 23:23:32,390 P18185 INFO ************ Epoch=8 end ************
2023-05-23 23:34:45,299 P18185 INFO Train loss: 0.556701
2023-05-23 23:34:45,300 P18185 INFO Evaluation @epoch 9 - batch 2548: 
2023-05-23 23:37:03,792 P18185 INFO [Metrics] gAUC: 0.859145
2023-05-23 23:37:03,803 P18185 INFO Monitor(max)=0.859145 STOP!
2023-05-23 23:37:03,803 P18185 INFO Reduce learning rate on plateau: 0.000050
2023-05-23 23:37:03,888 P18185 INFO ************ Epoch=9 end ************
2023-05-23 23:48:11,547 P18185 INFO Train loss: 0.462638
2023-05-23 23:48:11,547 P18185 INFO Evaluation @epoch 10 - batch 2548: 
2023-05-23 23:50:19,955 P18185 INFO [Metrics] gAUC: 0.878531
2023-05-23 23:50:19,961 P18185 INFO Save best model: monitor(max)=0.878531
2023-05-23 23:50:20,141 P18185 INFO ************ Epoch=10 end ************
2023-05-24 00:01:39,500 P18185 INFO Train loss: 0.419315
2023-05-24 00:01:39,501 P18185 INFO Evaluation @epoch 11 - batch 2548: 
2023-05-24 00:03:50,824 P18185 INFO [Metrics] gAUC: 0.883214
2023-05-24 00:03:50,825 P18185 INFO Save best model: monitor(max)=0.883214
2023-05-24 00:03:51,006 P18185 INFO ************ Epoch=11 end ************
2023-05-24 00:15:06,635 P18185 INFO Train loss: 0.401688
2023-05-24 00:15:06,636 P18185 INFO Evaluation @epoch 12 - batch 2548: 
2023-05-24 00:16:58,274 P18185 INFO [Metrics] gAUC: 0.885142
2023-05-24 00:16:58,276 P18185 INFO Save best model: monitor(max)=0.885142
2023-05-24 00:16:58,450 P18185 INFO ************ Epoch=12 end ************
2023-05-24 00:27:33,679 P18185 INFO Train loss: 0.390809
2023-05-24 00:27:33,680 P18185 INFO Evaluation @epoch 13 - batch 2548: 
2023-05-24 00:29:23,541 P18185 INFO [Metrics] gAUC: 0.883843
2023-05-24 00:29:23,542 P18185 INFO Monitor(max)=0.883843 STOP!
2023-05-24 00:29:23,542 P18185 INFO Reduce learning rate on plateau: 0.000005
2023-05-24 00:29:23,634 P18185 INFO ************ Epoch=13 end ************
2023-05-24 00:39:31,120 P18185 INFO Train loss: 0.342710
2023-05-24 00:39:31,121 P18185 INFO Evaluation @epoch 14 - batch 2548: 
2023-05-24 00:41:11,799 P18185 INFO [Metrics] gAUC: 0.882263
2023-05-24 00:41:11,800 P18185 INFO Monitor(max)=0.882263 STOP!
2023-05-24 00:41:11,800 P18185 INFO Reduce learning rate on plateau: 0.000001
2023-05-24 00:41:11,800 P18185 INFO ********* Epoch==14 early stop *********
2023-05-24 00:41:11,876 P18185 INFO Training finished.
2023-05-24 00:41:11,876 P18185 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DMR_amazonelectronics_x1/amazonelectronics_x1_51836f99/DMR_amazonelectronics_x1_002_bc859be0.model
2023-05-24 00:41:12,005 P18185 INFO ****** Validation evaluation ******
2023-05-24 00:42:49,899 P18185 INFO [Metrics] gAUC: 0.885142 - AUC: 0.887335 - logloss: 0.427744
2023-05-24 00:42:49,990 P18185 INFO ******** Test evaluation ********
2023-05-24 00:42:49,990 P18185 INFO Loading data...
2023-05-24 00:42:49,990 P18185 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_51836f99/test.h5
2023-05-24 00:42:50,463 P18185 INFO Test samples: total/384806, blocks/1
2023-05-24 00:42:50,463 P18185 INFO Loading test data done.
2023-05-24 00:44:18,081 P18185 INFO [Metrics] gAUC: 0.885142 - AUC: 0.887335 - logloss: 0.427744

```
