## BST_microvideo1.7m_x1

A hands-on guide to run the BST model on the MicroVideo1.7M_x1 dataset.

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
Please refer to the BARS dataset [MicroVideo1.7M_x1](https://github.com/openbenchmark/BARS/blob/main/datasets/MicroVideo1.7M#MicroVideo17M_x1) to get data ready.

### Code

We use the [BST](https://github.com/xue-pai/FuxiCTR/blob/v2.0.1/model_zoo/BST) model code from [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/MicroVideo1.7M/MicroVideo1.7M_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [BST_microvideo1.7m_x1_tuner_config_04](./BST_microvideo1.7m_x1_tuner_config_04). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/BST
    nohup python run_expid.py --config XXX/benchmarks/BST/BST_microvideo1.7m_x1_tuner_config_04 --expid BST_microvideo1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.685436 | 0.734150 | 0.411837  |


### Logs
```python
2022-09-07 23:40:40,582 P44602 INFO Params: {
    "attention_dropout": "0.1",
    "batch_norm": "True",
    "batch_size": "2048",
    "bst_sequence_field": "('clicked_items', 'clicked_categories')",
    "bst_target_field": "('item_id', 'cate_id')",
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
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id'}, {'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'clicked_items'}, {'feature_encoder': None, 'name': 'clicked_categories'}]",
    "gpu": "4",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "layer_norm": "False",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "BST",
    "model_id": "BST_microvideo1.7m_x1_021_c05a2c83",
    "model_root": "./checkpoints/BST_microvideo1.7m_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_heads": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2022",
    "seq_pooling_type": "target",
    "shuffle": "True",
    "stacked_transformer_layers": "1",
    "task": "binary_classification",
    "test_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "train_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/train.csv",
    "use_causal_mask": "False",
    "use_position_emb": "False",
    "use_residual": "False",
    "valid_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "verbose": "1"
}
2022-09-07 23:40:40,583 P44602 INFO Set up feature processor...
2022-09-07 23:40:40,583 P44602 INFO Load feature_map from json: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/feature_map.json
2022-09-07 23:40:40,583 P44602 INFO Set column index...
2022-09-07 23:40:40,583 P44602 INFO Feature specs: {
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514}",
    "clicked_categories": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514, 'max_len': 100}",
    "clicked_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'max_len': 100}",
    "group_id": "{'type': 'meta'}",
    "item_id": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10987, 'vocab_size': 10988}"
}
2022-09-07 23:40:52,417 P44602 INFO Total number of parameters: 1832065.
2022-09-07 23:40:52,419 P44602 INFO Loading data...
2022-09-07 23:40:52,419 P44602 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/train.h5
2022-09-07 23:41:07,821 P44602 INFO Train samples: total/8970309, blocks/1
2022-09-07 23:41:07,821 P44602 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/valid.h5
2022-09-07 23:41:13,879 P44602 INFO Validation samples: total/3767308, blocks/1
2022-09-07 23:41:13,879 P44602 INFO Loading train and validation data done.
2022-09-07 23:41:13,879 P44602 INFO Start training: 4381 batches/epoch
2022-09-07 23:41:13,879 P44602 INFO ************ Epoch=1 start ************
2022-09-08 00:11:57,713 P44602 INFO [Metrics] AUC: 0.712728 - gAUC: 0.667338
2022-09-08 00:11:57,725 P44602 INFO Save best model: monitor(max): 1.380065
2022-09-08 00:12:00,699 P44602 INFO --- 4381/4381 batches finished ---
2022-09-08 00:12:00,875 P44602 INFO Train loss: 0.464781
2022-09-08 00:12:00,875 P44602 INFO ************ Epoch=1 end ************
2022-09-08 00:42:35,938 P44602 INFO [Metrics] AUC: 0.720858 - gAUC: 0.673019
2022-09-08 00:42:35,950 P44602 INFO Save best model: monitor(max): 1.393877
2022-09-08 00:42:38,845 P44602 INFO --- 4381/4381 batches finished ---
2022-09-08 00:42:38,958 P44602 INFO Train loss: 0.443799
2022-09-08 00:42:38,959 P44602 INFO ************ Epoch=2 end ************
2022-09-08 01:13:10,861 P44602 INFO [Metrics] AUC: 0.721667 - gAUC: 0.674922
2022-09-08 01:13:10,874 P44602 INFO Save best model: monitor(max): 1.396589
2022-09-08 01:13:14,111 P44602 INFO --- 4381/4381 batches finished ---
2022-09-08 01:13:14,224 P44602 INFO Train loss: 0.440716
2022-09-08 01:13:14,224 P44602 INFO ************ Epoch=3 end ************
2022-09-08 01:43:42,450 P44602 INFO [Metrics] AUC: 0.723687 - gAUC: 0.676650
2022-09-08 01:43:42,459 P44602 INFO Save best model: monitor(max): 1.400337
2022-09-08 01:43:44,854 P44602 INFO --- 4381/4381 batches finished ---
2022-09-08 01:43:44,974 P44602 INFO Train loss: 0.439213
2022-09-08 01:43:44,974 P44602 INFO ************ Epoch=4 end ************
2022-09-08 02:14:13,871 P44602 INFO [Metrics] AUC: 0.725835 - gAUC: 0.678674
2022-09-08 02:14:13,894 P44602 INFO Save best model: monitor(max): 1.404509
2022-09-08 02:14:16,978 P44602 INFO --- 4381/4381 batches finished ---
2022-09-08 02:14:17,176 P44602 INFO Train loss: 0.437789
2022-09-08 02:14:17,176 P44602 INFO ************ Epoch=5 end ************
2022-09-08 02:44:40,812 P44602 INFO [Metrics] AUC: 0.725884 - gAUC: 0.678452
2022-09-08 02:44:40,826 P44602 INFO Monitor(max) STOP: 1.404336 !
2022-09-08 02:44:40,826 P44602 INFO Reduce learning rate on plateau: 0.000050
2022-09-08 02:44:40,827 P44602 INFO --- 4381/4381 batches finished ---
2022-09-08 02:44:40,936 P44602 INFO Train loss: 0.436210
2022-09-08 02:44:40,936 P44602 INFO ************ Epoch=6 end ************
2022-09-08 03:15:08,988 P44602 INFO [Metrics] AUC: 0.732823 - gAUC: 0.684225
2022-09-08 03:15:09,005 P44602 INFO Save best model: monitor(max): 1.417048
2022-09-08 03:15:11,049 P44602 INFO --- 4381/4381 batches finished ---
2022-09-08 03:15:11,169 P44602 INFO Train loss: 0.425588
2022-09-08 03:15:11,169 P44602 INFO ************ Epoch=7 end ************
2022-09-08 03:45:36,029 P44602 INFO [Metrics] AUC: 0.733496 - gAUC: 0.684629
2022-09-08 03:45:36,039 P44602 INFO Save best model: monitor(max): 1.418126
2022-09-08 03:45:39,317 P44602 INFO --- 4381/4381 batches finished ---
2022-09-08 03:45:39,425 P44602 INFO Train loss: 0.421862
2022-09-08 03:45:39,425 P44602 INFO ************ Epoch=8 end ************
2022-09-08 04:16:07,684 P44602 INFO [Metrics] AUC: 0.733495 - gAUC: 0.684653
2022-09-08 04:16:07,704 P44602 INFO Save best model: monitor(max): 1.418148
2022-09-08 04:16:10,659 P44602 INFO --- 4381/4381 batches finished ---
2022-09-08 04:16:10,772 P44602 INFO Train loss: 0.420322
2022-09-08 04:16:10,772 P44602 INFO ************ Epoch=9 end ************
2022-09-08 04:46:37,693 P44602 INFO [Metrics] AUC: 0.733735 - gAUC: 0.684695
2022-09-08 04:46:37,700 P44602 INFO Save best model: monitor(max): 1.418430
2022-09-08 04:46:40,559 P44602 INFO --- 4381/4381 batches finished ---
2022-09-08 04:46:40,716 P44602 INFO Train loss: 0.419248
2022-09-08 04:46:40,716 P44602 INFO ************ Epoch=10 end ************
2022-09-08 05:17:06,652 P44602 INFO [Metrics] AUC: 0.733955 - gAUC: 0.685006
2022-09-08 05:17:06,660 P44602 INFO Save best model: monitor(max): 1.418961
2022-09-08 05:17:09,756 P44602 INFO --- 4381/4381 batches finished ---
2022-09-08 05:17:09,919 P44602 INFO Train loss: 0.418220
2022-09-08 05:17:09,919 P44602 INFO ************ Epoch=11 end ************
2022-09-08 05:47:27,091 P44602 INFO [Metrics] AUC: 0.733818 - gAUC: 0.684946
2022-09-08 05:47:27,103 P44602 INFO Monitor(max) STOP: 1.418764 !
2022-09-08 05:47:27,103 P44602 INFO Reduce learning rate on plateau: 0.000005
2022-09-08 05:47:27,107 P44602 INFO --- 4381/4381 batches finished ---
2022-09-08 05:47:27,201 P44602 INFO Train loss: 0.417407
2022-09-08 05:47:27,201 P44602 INFO ************ Epoch=12 end ************
2022-09-08 06:17:55,780 P44602 INFO [Metrics] AUC: 0.734150 - gAUC: 0.685436
2022-09-08 06:17:55,794 P44602 INFO Save best model: monitor(max): 1.419586
2022-09-08 06:17:58,378 P44602 INFO --- 4381/4381 batches finished ---
2022-09-08 06:17:58,483 P44602 INFO Train loss: 0.413375
2022-09-08 06:17:58,483 P44602 INFO ************ Epoch=13 end ************
2022-09-08 06:48:21,607 P44602 INFO [Metrics] AUC: 0.733932 - gAUC: 0.685176
2022-09-08 06:48:21,632 P44602 INFO Monitor(max) STOP: 1.419108 !
2022-09-08 06:48:21,632 P44602 INFO Reduce learning rate on plateau: 0.000001
2022-09-08 06:48:21,633 P44602 INFO --- 4381/4381 batches finished ---
2022-09-08 06:48:21,855 P44602 INFO Train loss: 0.412770
2022-09-08 06:48:21,855 P44602 INFO ************ Epoch=14 end ************
2022-09-08 07:18:52,596 P44602 INFO [Metrics] AUC: 0.734086 - gAUC: 0.685230
2022-09-08 07:18:52,688 P44602 INFO Monitor(max) STOP: 1.419316 !
2022-09-08 07:18:52,688 P44602 INFO Reduce learning rate on plateau: 0.000001
2022-09-08 07:18:52,688 P44602 INFO ********* Epoch==15 early stop *********
2022-09-08 07:18:52,688 P44602 INFO --- 4381/4381 batches finished ---
2022-09-08 07:18:52,762 P44602 INFO Train loss: 0.412045
2022-09-08 07:18:52,762 P44602 INFO Training finished.
2022-09-08 07:18:52,762 P44602 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/BST_microvideo1.7m_x1/microvideo1.7m_x1_0d855fe6/BST_microvideo1.7m_x1_021_c05a2c83.model
2022-09-08 07:18:53,715 P44602 INFO ****** Validation evaluation ******
2022-09-08 07:27:03,668 P44602 INFO [Metrics] gAUC: 0.685436 - AUC: 0.734150 - logloss: 0.411837
2022-09-08 07:27:03,833 P44602 INFO ******** Test evaluation ********
2022-09-08 07:27:03,833 P44602 INFO Loading data...
2022-09-08 07:27:03,833 P44602 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/test.h5
2022-09-08 07:27:10,813 P44602 INFO Test samples: total/3767308, blocks/1
2022-09-08 07:27:10,813 P44602 INFO Loading test data done.
2022-09-08 07:35:30,860 P44602 INFO [Metrics] gAUC: 0.685436 - AUC: 0.734150 - logloss: 0.411837

```
