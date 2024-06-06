## APG_DCNv2_microvideo1.7m_x1

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

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [APG_DCNv2_microvideo1.7m_x1_tuner_config_01](./APG_DCNv2_microvideo1.7m_x1_tuner_config_01). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/APG
    nohup python run_expid.py --config YOUR_PATH/APG/APG_DCNv2_microvideo1.7m_x1_tuner_config_01 --expid APG_DCNv2_microvideo1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.682735 | 0.732541 | 0.412684  |


### Logs
```python
2023-06-05 07:00:27,460 P75716 INFO Params: {
    "batch_norm": "True",
    "batch_size": "2048",
    "condition_features": "None",
    "condition_mode": "self-wise",
    "data_format": "csv",
    "data_root": "../data/MicroVideo1.7M/",
    "dataset_id": "microvideo1.7m_x1_0d855fe6",
    "debug_mode": "False",
    "dnn_activations": "relu",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'name': 'clicked_items', 'padding': 'pre', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'clicked_categories', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "feature_config": "None",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'clicked_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'clicked_categories'}]",
    "generate_bias": "True",
    "gpu": "6",
    "group_id": "group_id",
    "hypernet_config": "{'dropout_rates': 0.1, 'hidden_activations': 'relu', 'hidden_units': []}",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "low_rank": "32",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "APG_DCNv2",
    "model_id": "APG_DCNv2_microvideo1.7m_x1_007_863bfc5f",
    "model_root": "./checkpoints/APG_DCNv2_microvideo1.7m_x1/",
    "model_structure": "parallel",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "new_condition_emb": "False",
    "num_cross_layers": "3",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "overparam_p": "[32, 16, 8]",
    "parallel_dnn_hidden_units": "[1024, 512, 256]",
    "pickle_feature_encoder": "True",
    "rank_k": "[32, 16, 8]",
    "save_best_only": "True",
    "seed": "2022",
    "shuffle": "True",
    "stacked_dnn_hidden_units": "[500, 500, 500]",
    "task": "binary_classification",
    "test_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "train_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/train.csv",
    "use_features": "None",
    "use_low_rank_mixture": "False",
    "valid_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "verbose": "1"
}
2023-06-05 07:00:27,461 P75716 INFO Set up feature processor...
2023-06-05 07:00:27,461 P75716 WARNING Skip rebuilding ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/feature_map.json. Please delete it manually if rebuilding is required.
2023-06-05 07:00:27,461 P75716 INFO Load feature_map from json: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/feature_map.json
2023-06-05 07:00:27,461 P75716 INFO Set column index...
2023-06-05 07:00:27,461 P75716 INFO Feature specs: {
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514}",
    "clicked_categories": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514, 'max_len': 100}",
    "clicked_items": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'max_len': 100}",
    "group_id": "{'type': 'meta'}",
    "item_id": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10987, 'vocab_size': 10988}"
}
2023-06-05 07:00:35,594 P75716 INFO Total number of parameters: 2741825.
2023-06-05 07:00:35,595 P75716 INFO Loading data...
2023-06-05 07:00:35,595 P75716 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/train.h5
2023-06-05 07:00:45,726 P75716 INFO Train samples: total/8970309, blocks/1
2023-06-05 07:00:45,727 P75716 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/valid.h5
2023-06-05 07:00:49,865 P75716 INFO Validation samples: total/3767308, blocks/1
2023-06-05 07:00:49,865 P75716 INFO Loading train and validation data done.
2023-06-05 07:00:49,865 P75716 INFO Start training: 4381 batches/epoch
2023-06-05 07:00:49,865 P75716 INFO ************ Epoch=1 start ************
2023-06-05 07:04:49,365 P75716 INFO Train loss: 0.458991
2023-06-05 07:04:49,365 P75716 INFO Evaluation @epoch 1 - batch 4381: 
2023-06-05 07:06:38,963 P75716 INFO [Metrics] AUC: 0.719110 - gAUC: 0.670555
2023-06-05 07:06:38,969 P75716 INFO Save best model: monitor(max)=1.389665
2023-06-05 07:06:41,374 P75716 INFO ************ Epoch=1 end ************
2023-06-05 07:10:45,981 P75716 INFO Train loss: 0.440881
2023-06-05 07:10:45,981 P75716 INFO Evaluation @epoch 2 - batch 4381: 
2023-06-05 07:12:32,395 P75716 INFO [Metrics] AUC: 0.721810 - gAUC: 0.672201
2023-06-05 07:12:32,395 P75716 INFO Save best model: monitor(max)=1.394011
2023-06-05 07:12:34,368 P75716 INFO ************ Epoch=2 end ************
2023-06-05 07:16:58,462 P75716 INFO Train loss: 0.438760
2023-06-05 07:16:58,462 P75716 INFO Evaluation @epoch 3 - batch 4381: 
2023-06-05 07:18:41,773 P75716 INFO [Metrics] AUC: 0.723777 - gAUC: 0.674748
2023-06-05 07:18:41,774 P75716 INFO Save best model: monitor(max)=1.398525
2023-06-05 07:18:43,718 P75716 INFO ************ Epoch=3 end ************
2023-06-05 07:22:48,556 P75716 INFO Train loss: 0.437439
2023-06-05 07:22:48,556 P75716 INFO Evaluation @epoch 4 - batch 4381: 
2023-06-05 07:24:31,158 P75716 INFO [Metrics] AUC: 0.724921 - gAUC: 0.677001
2023-06-05 07:24:31,160 P75716 INFO Save best model: monitor(max)=1.401923
2023-06-05 07:24:33,146 P75716 INFO ************ Epoch=4 end ************
2023-06-05 07:28:32,968 P75716 INFO Train loss: 0.436534
2023-06-05 07:28:32,968 P75716 INFO Evaluation @epoch 5 - batch 4381: 
2023-06-05 07:30:18,914 P75716 INFO [Metrics] AUC: 0.724518 - gAUC: 0.676459
2023-06-05 07:30:18,915 P75716 INFO Monitor(max)=1.400977 STOP!
2023-06-05 07:30:18,915 P75716 INFO Reduce learning rate on plateau: 0.000050
2023-06-05 07:30:19,002 P75716 INFO ************ Epoch=5 end ************
2023-06-05 07:34:16,315 P75716 INFO Train loss: 0.423939
2023-06-05 07:34:16,315 P75716 INFO Evaluation @epoch 6 - batch 4381: 
2023-06-05 07:35:56,019 P75716 INFO [Metrics] AUC: 0.731578 - gAUC: 0.681731
2023-06-05 07:35:56,020 P75716 INFO Save best model: monitor(max)=1.413309
2023-06-05 07:35:57,900 P75716 INFO ************ Epoch=6 end ************
2023-06-05 07:39:56,593 P75716 INFO Train loss: 0.419004
2023-06-05 07:39:56,593 P75716 INFO Evaluation @epoch 7 - batch 4381: 
2023-06-05 07:41:31,031 P75716 INFO [Metrics] AUC: 0.732223 - gAUC: 0.682287
2023-06-05 07:41:31,032 P75716 INFO Save best model: monitor(max)=1.414510
2023-06-05 07:41:33,042 P75716 INFO ************ Epoch=7 end ************
2023-06-05 07:45:38,269 P75716 INFO Train loss: 0.416882
2023-06-05 07:45:38,269 P75716 INFO Evaluation @epoch 8 - batch 4381: 
2023-06-05 07:47:16,078 P75716 INFO [Metrics] AUC: 0.732484 - gAUC: 0.682546
2023-06-05 07:47:16,080 P75716 INFO Save best model: monitor(max)=1.415029
2023-06-05 07:47:18,056 P75716 INFO ************ Epoch=8 end ************
2023-06-05 07:51:23,447 P75716 INFO Train loss: 0.415176
2023-06-05 07:51:23,447 P75716 INFO Evaluation @epoch 9 - batch 4381: 
2023-06-05 07:53:01,166 P75716 INFO [Metrics] AUC: 0.732472 - gAUC: 0.682655
2023-06-05 07:53:01,168 P75716 INFO Save best model: monitor(max)=1.415127
2023-06-05 07:53:03,036 P75716 INFO ************ Epoch=9 end ************
2023-06-05 07:57:09,997 P75716 INFO Train loss: 0.413706
2023-06-05 07:57:09,998 P75716 INFO Evaluation @epoch 10 - batch 4381: 
2023-06-05 07:58:50,345 P75716 INFO [Metrics] AUC: 0.732541 - gAUC: 0.682735
2023-06-05 07:58:50,347 P75716 INFO Save best model: monitor(max)=1.415276
2023-06-05 07:58:52,314 P75716 INFO ************ Epoch=10 end ************
2023-06-05 08:03:06,305 P75716 INFO Train loss: 0.412341
2023-06-05 08:03:06,306 P75716 INFO Evaluation @epoch 11 - batch 4381: 
2023-06-05 08:04:41,451 P75716 INFO [Metrics] AUC: 0.732394 - gAUC: 0.682724
2023-06-05 08:04:41,452 P75716 INFO Monitor(max)=1.415118 STOP!
2023-06-05 08:04:41,452 P75716 INFO Reduce learning rate on plateau: 0.000005
2023-06-05 08:04:41,533 P75716 INFO ************ Epoch=11 end ************
2023-06-05 08:08:50,960 P75716 INFO Train loss: 0.406497
2023-06-05 08:08:50,961 P75716 INFO Evaluation @epoch 12 - batch 4381: 
2023-06-05 08:10:31,176 P75716 INFO [Metrics] AUC: 0.731831 - gAUC: 0.682006
2023-06-05 08:10:31,177 P75716 INFO Monitor(max)=1.413837 STOP!
2023-06-05 08:10:31,177 P75716 INFO Reduce learning rate on plateau: 0.000001
2023-06-05 08:10:31,177 P75716 INFO ********* Epoch==12 early stop *********
2023-06-05 08:10:31,264 P75716 INFO Training finished.
2023-06-05 08:10:31,264 P75716 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/APG_DCNv2_microvideo1.7m_x1/microvideo1.7m_x1_0d855fe6/APG_DCNv2_microvideo1.7m_x1_007_863bfc5f.model
2023-06-05 08:10:31,955 P75716 INFO ****** Validation evaluation ******
2023-06-05 08:11:39,195 P75716 INFO [Metrics] gAUC: 0.682735 - AUC: 0.732541 - logloss: 0.412684
2023-06-05 08:11:39,300 P75716 INFO ******** Test evaluation ********
2023-06-05 08:11:39,300 P75716 INFO Loading data...
2023-06-05 08:11:39,300 P75716 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/test.h5
2023-06-05 08:11:43,376 P75716 INFO Test samples: total/3767308, blocks/1
2023-06-05 08:11:43,376 P75716 INFO Loading test data done.
2023-06-05 08:12:19,350 P75716 INFO [Metrics] gAUC: 0.682735 - AUC: 0.732541 - logloss: 0.412684

```
