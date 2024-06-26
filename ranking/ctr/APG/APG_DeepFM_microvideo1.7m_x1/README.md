## APG_DeepFM_microvideo1.7m_x1

A hands-on guide to run the APG model on the MicroVideo1.7M_x1 dataset.

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

We use the [APG](https://github.com/reczoo/FuxiCTR/tree/v2.0.3/model_zoo/APG) model code from [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/tree/v2.0.3) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.3.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.3
    ```

2. Create a data directory and put the downloaded data files in `../data/MicroVideo/MicroVideo1.7M_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [APG_DeepFM_microvideo1.7m_x1_tuner_config_01](./APG_DeepFM_microvideo1.7m_x1_tuner_config_01). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/APG
    nohup python run_expid.py --config YOUR_PATH/APG/APG_DeepFM_microvideo1.7m_x1_tuner_config_01 --expid APG_DeepFM_microvideo1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.683198 | 0.732537 | 0.413739  |


### Logs
```python
2023-06-04 08:27:58,069 P83006 INFO Params: {
    "batch_norm": "True",
    "batch_size": "2048",
    "condition_features": "None",
    "condition_mode": "self-wise",
    "data_format": "csv",
    "data_root": "../data/MicroVideo1.7M/",
    "dataset_id": "microvideo1.7m_x1_0d855fe6",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.0005",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'name': 'clicked_items', 'padding': 'pre', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'clicked_categories', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "feature_config": "None",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'clicked_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'clicked_categories'}]",
    "generate_bias": "True",
    "gpu": "4",
    "group_id": "group_id",
    "hidden_activations": "relu",
    "hidden_units": "[1024, 512, 256]",
    "hypernet_config": "{'dropout_rates': 0, 'hidden_activations': 'relu', 'hidden_units': []}",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "APG_DeepFM",
    "model_id": "APG_DeepFM_microvideo1.7m_x1_023_d3a3ee84",
    "model_root": "./checkpoints/APG_DeepFM_microvideo1.7m_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "new_condition_emb": "False",
    "num_workers": "3",
    "optimizer": "adam",
    "overparam_p": "[32, 16, 8]",
    "pickle_feature_encoder": "True",
    "rank_k": "[16, 8, 4]",
    "save_best_only": "True",
    "seed": "2022",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "train_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/train.csv",
    "use_features": "None",
    "valid_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "verbose": "1"
}
2023-06-04 08:27:58,070 P83006 INFO Set up feature processor...
2023-06-04 08:27:58,071 P83006 WARNING Skip rebuilding ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/feature_map.json. Please delete it manually if rebuilding is required.
2023-06-04 08:27:58,071 P83006 INFO Load feature_map from json: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/feature_map.json
2023-06-04 08:27:58,071 P83006 INFO Set column index...
2023-06-04 08:27:58,071 P83006 INFO Feature specs: {
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514}",
    "clicked_categories": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514, 'max_len': 100}",
    "clicked_items": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'max_len': 100}",
    "group_id": "{'type': 'meta'}",
    "item_id": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10987, 'vocab_size': 10988}"
}
2023-06-04 08:28:04,388 P83006 INFO Total number of parameters: 5385830.
2023-06-04 08:28:04,388 P83006 INFO Loading data...
2023-06-04 08:28:04,389 P83006 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/train.h5
2023-06-04 08:28:15,634 P83006 INFO Train samples: total/8970309, blocks/1
2023-06-04 08:28:15,634 P83006 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/valid.h5
2023-06-04 08:28:20,008 P83006 INFO Validation samples: total/3767308, blocks/1
2023-06-04 08:28:20,009 P83006 INFO Loading train and validation data done.
2023-06-04 08:28:20,009 P83006 INFO Start training: 4381 batches/epoch
2023-06-04 08:28:20,009 P83006 INFO ************ Epoch=1 start ************
2023-06-04 08:36:28,345 P83006 INFO Train loss: 0.482970
2023-06-04 08:36:28,346 P83006 INFO Evaluation @epoch 1 - batch 4381: 
2023-06-04 08:40:00,947 P83006 INFO [Metrics] AUC: 0.715466 - gAUC: 0.670076
2023-06-04 08:40:00,948 P83006 INFO Save best model: monitor(max)=1.385543
2023-06-04 08:40:02,615 P83006 INFO ************ Epoch=1 end ************
2023-06-04 08:48:11,283 P83006 INFO Train loss: 0.447571
2023-06-04 08:48:11,283 P83006 INFO Evaluation @epoch 2 - batch 4381: 
2023-06-04 08:51:42,963 P83006 INFO [Metrics] AUC: 0.718113 - gAUC: 0.671582
2023-06-04 08:51:42,963 P83006 INFO Save best model: monitor(max)=1.389695
2023-06-04 08:51:44,946 P83006 INFO ************ Epoch=2 end ************
2023-06-04 08:59:50,328 P83006 INFO Train loss: 0.441891
2023-06-04 08:59:50,329 P83006 INFO Evaluation @epoch 3 - batch 4381: 
2023-06-04 09:03:23,299 P83006 INFO [Metrics] AUC: 0.721060 - gAUC: 0.674756
2023-06-04 09:03:23,300 P83006 INFO Save best model: monitor(max)=1.395816
2023-06-04 09:03:25,285 P83006 INFO ************ Epoch=3 end ************
2023-06-04 09:11:34,532 P83006 INFO Train loss: 0.439808
2023-06-04 09:11:34,532 P83006 INFO Evaluation @epoch 4 - batch 4381: 
2023-06-04 09:15:10,786 P83006 INFO [Metrics] AUC: 0.722097 - gAUC: 0.675894
2023-06-04 09:15:10,788 P83006 INFO Save best model: monitor(max)=1.397991
2023-06-04 09:15:12,915 P83006 INFO ************ Epoch=4 end ************
2023-06-04 09:23:42,238 P83006 INFO Train loss: 0.438220
2023-06-04 09:23:42,238 P83006 INFO Evaluation @epoch 5 - batch 4381: 
2023-06-04 09:27:13,623 P83006 INFO [Metrics] AUC: 0.722427 - gAUC: 0.676863
2023-06-04 09:27:13,624 P83006 INFO Save best model: monitor(max)=1.399290
2023-06-04 09:27:15,851 P83006 INFO ************ Epoch=5 end ************
2023-06-04 09:35:33,393 P83006 INFO Train loss: 0.437070
2023-06-04 09:35:33,393 P83006 INFO Evaluation @epoch 6 - batch 4381: 
2023-06-04 09:39:02,513 P83006 INFO [Metrics] AUC: 0.723779 - gAUC: 0.676865
2023-06-04 09:39:02,514 P83006 INFO Save best model: monitor(max)=1.400644
2023-06-04 09:39:04,562 P83006 INFO ************ Epoch=6 end ************
2023-06-04 09:47:12,506 P83006 INFO Train loss: 0.436184
2023-06-04 09:47:12,507 P83006 INFO Evaluation @epoch 7 - batch 4381: 
2023-06-04 09:50:28,773 P83006 INFO [Metrics] AUC: 0.724318 - gAUC: 0.677541
2023-06-04 09:50:28,774 P83006 INFO Save best model: monitor(max)=1.401859
2023-06-04 09:50:30,795 P83006 INFO ************ Epoch=7 end ************
2023-06-04 09:57:46,890 P83006 INFO Train loss: 0.435608
2023-06-04 09:57:46,890 P83006 INFO Evaluation @epoch 8 - batch 4381: 
2023-06-04 10:00:55,569 P83006 INFO [Metrics] AUC: 0.724735 - gAUC: 0.677620
2023-06-04 10:00:55,570 P83006 INFO Save best model: monitor(max)=1.402355
2023-06-04 10:00:57,722 P83006 INFO ************ Epoch=8 end ************
2023-06-04 10:08:16,101 P83006 INFO Train loss: 0.435144
2023-06-04 10:08:16,102 P83006 INFO Evaluation @epoch 9 - batch 4381: 
2023-06-04 10:10:04,354 P83006 INFO [Metrics] AUC: 0.725101 - gAUC: 0.677391
2023-06-04 10:10:04,355 P83006 INFO Save best model: monitor(max)=1.402492
2023-06-04 10:10:06,383 P83006 INFO ************ Epoch=9 end ************
2023-06-04 10:14:14,665 P83006 INFO Train loss: 0.434660
2023-06-04 10:14:14,665 P83006 INFO Evaluation @epoch 10 - batch 4381: 
2023-06-04 10:15:42,830 P83006 INFO [Metrics] AUC: 0.725838 - gAUC: 0.679802
2023-06-04 10:15:42,831 P83006 INFO Save best model: monitor(max)=1.405640
2023-06-04 10:15:44,850 P83006 INFO ************ Epoch=10 end ************
2023-06-04 10:18:46,719 P83006 INFO Train loss: 0.434219
2023-06-04 10:18:46,719 P83006 INFO Evaluation @epoch 11 - batch 4381: 
2023-06-04 10:20:03,940 P83006 INFO [Metrics] AUC: 0.725347 - gAUC: 0.677700
2023-06-04 10:20:03,941 P83006 INFO Monitor(max)=1.403047 STOP!
2023-06-04 10:20:03,941 P83006 INFO Reduce learning rate on plateau: 0.000050
2023-06-04 10:20:04,002 P83006 INFO ************ Epoch=11 end ************
2023-06-04 10:22:49,099 P83006 INFO Train loss: 0.420980
2023-06-04 10:22:49,099 P83006 INFO Evaluation @epoch 12 - batch 4381: 
2023-06-04 10:23:21,470 P83006 INFO [Metrics] AUC: 0.731869 - gAUC: 0.683051
2023-06-04 10:23:21,471 P83006 INFO Save best model: monitor(max)=1.414920
2023-06-04 10:23:23,454 P83006 INFO ************ Epoch=12 end ************
2023-06-04 10:25:20,841 P83006 INFO Train loss: 0.415281
2023-06-04 10:25:20,841 P83006 INFO Evaluation @epoch 13 - batch 4381: 
2023-06-04 10:25:48,279 P83006 INFO [Metrics] AUC: 0.732537 - gAUC: 0.683198
2023-06-04 10:25:48,280 P83006 INFO Save best model: monitor(max)=1.415735
2023-06-04 10:25:50,236 P83006 INFO ************ Epoch=13 end ************
2023-06-04 10:27:15,878 P83006 INFO Train loss: 0.413059
2023-06-04 10:27:15,879 P83006 INFO Evaluation @epoch 14 - batch 4381: 
2023-06-04 10:27:37,123 P83006 INFO [Metrics] AUC: 0.732248 - gAUC: 0.683211
2023-06-04 10:27:37,123 P83006 INFO Monitor(max)=1.415459 STOP!
2023-06-04 10:27:37,124 P83006 INFO Reduce learning rate on plateau: 0.000005
2023-06-04 10:27:37,196 P83006 INFO ************ Epoch=14 end ************
2023-06-04 10:29:04,390 P83006 INFO Train loss: 0.408256
2023-06-04 10:29:04,390 P83006 INFO Evaluation @epoch 15 - batch 4381: 
2023-06-04 10:29:25,730 P83006 INFO [Metrics] AUC: 0.732236 - gAUC: 0.683034
2023-06-04 10:29:25,731 P83006 INFO Monitor(max)=1.415270 STOP!
2023-06-04 10:29:25,731 P83006 INFO Reduce learning rate on plateau: 0.000001
2023-06-04 10:29:25,731 P83006 INFO ********* Epoch==15 early stop *********
2023-06-04 10:29:25,800 P83006 INFO Training finished.
2023-06-04 10:29:25,800 P83006 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/APG_DeepFM_microvideo1.7m_x1/microvideo1.7m_x1_0d855fe6/APG_DeepFM_microvideo1.7m_x1_023_d3a3ee84.model
2023-06-04 10:29:26,302 P83006 INFO ****** Validation evaluation ******
2023-06-04 10:29:49,143 P83006 INFO [Metrics] gAUC: 0.683198 - AUC: 0.732537 - logloss: 0.413739
2023-06-04 10:29:49,222 P83006 INFO ******** Test evaluation ********
2023-06-04 10:29:49,222 P83006 INFO Loading data...
2023-06-04 10:29:49,223 P83006 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/test.h5
2023-06-04 10:29:53,364 P83006 INFO Test samples: total/3767308, blocks/1
2023-06-04 10:29:53,364 P83006 INFO Loading test data done.
2023-06-04 10:30:15,260 P83006 INFO [Metrics] gAUC: 0.683198 - AUC: 0.732537 - logloss: 0.413739

```
