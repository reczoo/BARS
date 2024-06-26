## PPNet_microvideo1.7m_x1

A hands-on guide to run the PPNet model on the MicroVideo1.7M_x1 dataset.

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

We use the [PPNet](https://github.com/reczoo/FuxiCTR/tree/v2.0.3/model_zoo/PPNet) model code from [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/tree/v2.0.3) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.3.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.3
    ```

2. Create a data directory and put the downloaded data files in `../data/MicroVideo/MicroVideo1.7M_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [PPNet_microvideo1.7m_x1_tuner_config_01](./PPNet_microvideo1.7m_x1_tuner_config_01). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/PPNet
    nohup python run_expid.py --config YOUR_PATH/PPNet/PPNet_microvideo1.7m_x1_tuner_config_01 --expid PPNet_microvideo1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.686542 | 0.734938 | 0.411962  |


### Logs
```python
2023-06-04 02:38:36,259 P104561 INFO Params: {
    "batch_norm": "True",
    "batch_size": "2048",
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
    "gate_emb_dim": "64",
    "gate_hidden_dim": "128",
    "gate_priors": "['user_id']",
    "gpu": "0",
    "group_id": "group_id",
    "hidden_activations": "ReLU",
    "hidden_units": "[1024, 512, 256]",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "PPNet",
    "model_id": "PPNet_microvideo1.7m_x1_001_97742c0f",
    "model_root": "./checkpoints/PPNet_microvideo1.7m_x1/",
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
    "use_features": "None",
    "valid_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "verbose": "1"
}
2023-06-04 02:38:36,260 P104561 INFO Set up feature processor...
2023-06-04 02:38:36,260 P104561 WARNING Skip rebuilding ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/feature_map.json. Please delete it manually if rebuilding is required.
2023-06-04 02:38:36,260 P104561 INFO Load feature_map from json: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/feature_map.json
2023-06-04 02:38:36,260 P104561 INFO Set column index...
2023-06-04 02:38:36,260 P104561 INFO Feature specs: {
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514}",
    "clicked_categories": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514, 'max_len': 100}",
    "clicked_items": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'max_len': 100}",
    "group_id": "{'type': 'meta'}",
    "item_id": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10987, 'vocab_size': 10988}"
}
2023-06-04 02:38:43,782 P104561 INFO Total number of parameters: 2815233.
2023-06-04 02:38:43,783 P104561 INFO Loading data...
2023-06-04 02:38:43,783 P104561 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/train.h5
2023-06-04 02:38:53,767 P104561 INFO Train samples: total/8970309, blocks/1
2023-06-04 02:38:53,767 P104561 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/valid.h5
2023-06-04 02:38:58,043 P104561 INFO Validation samples: total/3767308, blocks/1
2023-06-04 02:38:58,043 P104561 INFO Loading train and validation data done.
2023-06-04 02:38:58,043 P104561 INFO Start training: 4381 batches/epoch
2023-06-04 02:38:58,043 P104561 INFO ************ Epoch=1 start ************
2023-06-04 02:47:02,330 P104561 INFO Train loss: 0.460393
2023-06-04 02:47:02,330 P104561 INFO Evaluation @epoch 1 - batch 4381: 
2023-06-04 02:50:36,625 P104561 INFO [Metrics] AUC: 0.716033 - gAUC: 0.670363
2023-06-04 02:50:36,631 P104561 INFO Save best model: monitor(max)=1.386396
2023-06-04 02:50:38,264 P104561 INFO ************ Epoch=1 end ************
2023-06-04 02:58:43,978 P104561 INFO Train loss: 0.442108
2023-06-04 02:58:43,978 P104561 INFO Evaluation @epoch 2 - batch 4381: 
2023-06-04 03:02:18,951 P104561 INFO [Metrics] AUC: 0.720923 - gAUC: 0.672270
2023-06-04 03:02:18,977 P104561 INFO Save best model: monitor(max)=1.393192
2023-06-04 03:02:21,011 P104561 INFO ************ Epoch=2 end ************
2023-06-04 03:10:16,104 P104561 INFO Train loss: 0.437907
2023-06-04 03:10:16,104 P104561 INFO Evaluation @epoch 3 - batch 4381: 
2023-06-04 03:13:48,307 P104561 INFO [Metrics] AUC: 0.724069 - gAUC: 0.676025
2023-06-04 03:13:48,308 P104561 INFO Save best model: monitor(max)=1.400094
2023-06-04 03:13:50,267 P104561 INFO ************ Epoch=3 end ************
2023-06-04 03:21:48,980 P104561 INFO Train loss: 0.436121
2023-06-04 03:21:48,981 P104561 INFO Evaluation @epoch 4 - batch 4381: 
2023-06-04 03:25:22,289 P104561 INFO [Metrics] AUC: 0.724788 - gAUC: 0.677178
2023-06-04 03:25:22,290 P104561 INFO Save best model: monitor(max)=1.401966
2023-06-04 03:25:24,263 P104561 INFO ************ Epoch=4 end ************
2023-06-04 03:33:28,958 P104561 INFO Train loss: 0.434862
2023-06-04 03:33:28,958 P104561 INFO Evaluation @epoch 5 - batch 4381: 
2023-06-04 03:37:01,145 P104561 INFO [Metrics] AUC: 0.726472 - gAUC: 0.678042
2023-06-04 03:37:01,149 P104561 INFO Save best model: monitor(max)=1.404514
2023-06-04 03:37:03,234 P104561 INFO ************ Epoch=5 end ************
2023-06-04 03:45:02,208 P104561 INFO Train loss: 0.433747
2023-06-04 03:45:02,209 P104561 INFO Evaluation @epoch 6 - batch 4381: 
2023-06-04 03:48:34,597 P104561 INFO [Metrics] AUC: 0.725813 - gAUC: 0.678635
2023-06-04 03:48:34,600 P104561 INFO Monitor(max)=1.404448 STOP!
2023-06-04 03:48:34,601 P104561 INFO Reduce learning rate on plateau: 0.000050
2023-06-04 03:48:34,682 P104561 INFO ************ Epoch=6 end ************
2023-06-04 03:56:41,576 P104561 INFO Train loss: 0.423679
2023-06-04 03:56:41,577 P104561 INFO Evaluation @epoch 7 - batch 4381: 
2023-06-04 04:00:12,600 P104561 INFO [Metrics] AUC: 0.733079 - gAUC: 0.684859
2023-06-04 04:00:12,605 P104561 INFO Save best model: monitor(max)=1.417938
2023-06-04 04:00:14,615 P104561 INFO ************ Epoch=7 end ************
2023-06-04 04:08:18,885 P104561 INFO Train loss: 0.419839
2023-06-04 04:08:18,886 P104561 INFO Evaluation @epoch 8 - batch 4381: 
2023-06-04 04:11:50,572 P104561 INFO [Metrics] AUC: 0.733664 - gAUC: 0.685047
2023-06-04 04:11:50,575 P104561 INFO Save best model: monitor(max)=1.418710
2023-06-04 04:11:52,634 P104561 INFO ************ Epoch=8 end ************
2023-06-04 04:19:58,946 P104561 INFO Train loss: 0.418099
2023-06-04 04:19:58,947 P104561 INFO Evaluation @epoch 9 - batch 4381: 
2023-06-04 04:23:31,665 P104561 INFO [Metrics] AUC: 0.733961 - gAUC: 0.685414
2023-06-04 04:23:31,668 P104561 INFO Save best model: monitor(max)=1.419375
2023-06-04 04:23:33,600 P104561 INFO ************ Epoch=9 end ************
2023-06-04 04:31:40,930 P104561 INFO Train loss: 0.416796
2023-06-04 04:31:40,931 P104561 INFO Evaluation @epoch 10 - batch 4381: 
2023-06-04 04:35:14,930 P104561 INFO [Metrics] AUC: 0.734369 - gAUC: 0.685685
2023-06-04 04:35:14,937 P104561 INFO Save best model: monitor(max)=1.420053
2023-06-04 04:35:17,003 P104561 INFO ************ Epoch=10 end ************
2023-06-04 04:43:33,153 P104561 INFO Train loss: 0.415682
2023-06-04 04:43:33,153 P104561 INFO Evaluation @epoch 11 - batch 4381: 
2023-06-04 04:47:05,456 P104561 INFO [Metrics] AUC: 0.734647 - gAUC: 0.685824
2023-06-04 04:47:05,459 P104561 INFO Save best model: monitor(max)=1.420472
2023-06-04 04:47:07,558 P104561 INFO ************ Epoch=11 end ************
2023-06-04 04:55:12,445 P104561 INFO Train loss: 0.414763
2023-06-04 04:55:12,445 P104561 INFO Evaluation @epoch 12 - batch 4381: 
2023-06-04 04:58:42,830 P104561 INFO [Metrics] AUC: 0.734531 - gAUC: 0.685963
2023-06-04 04:58:42,835 P104561 INFO Save best model: monitor(max)=1.420494
2023-06-04 04:58:44,811 P104561 INFO ************ Epoch=12 end ************
2023-06-04 05:06:51,051 P104561 INFO Train loss: 0.413799
2023-06-04 05:06:51,052 P104561 INFO Evaluation @epoch 13 - batch 4381: 
2023-06-04 05:10:26,252 P104561 INFO [Metrics] AUC: 0.734719 - gAUC: 0.686250
2023-06-04 05:10:26,255 P104561 INFO Save best model: monitor(max)=1.420969
2023-06-04 05:10:28,245 P104561 INFO ************ Epoch=13 end ************
2023-06-04 05:18:33,965 P104561 INFO Train loss: 0.413071
2023-06-04 05:18:33,966 P104561 INFO Evaluation @epoch 14 - batch 4381: 
2023-06-04 05:22:04,365 P104561 INFO [Metrics] AUC: 0.734938 - gAUC: 0.686542
2023-06-04 05:22:04,371 P104561 INFO Save best model: monitor(max)=1.421480
2023-06-04 05:22:06,489 P104561 INFO ************ Epoch=14 end ************
2023-06-04 05:30:14,676 P104561 INFO Train loss: 0.412329
2023-06-04 05:30:14,677 P104561 INFO Evaluation @epoch 15 - batch 4381: 
2023-06-04 05:33:44,383 P104561 INFO [Metrics] AUC: 0.734761 - gAUC: 0.686296
2023-06-04 05:33:44,386 P104561 INFO Monitor(max)=1.421057 STOP!
2023-06-04 05:33:44,387 P104561 INFO Reduce learning rate on plateau: 0.000005
2023-06-04 05:33:44,449 P104561 INFO ************ Epoch=15 end ************
2023-06-04 05:41:49,035 P104561 INFO Train loss: 0.406621
2023-06-04 05:41:49,036 P104561 INFO Evaluation @epoch 16 - batch 4381: 
2023-06-04 05:45:11,806 P104561 INFO [Metrics] AUC: 0.734859 - gAUC: 0.686399
2023-06-04 05:45:11,812 P104561 INFO Monitor(max)=1.421258 STOP!
2023-06-04 05:45:11,812 P104561 INFO Reduce learning rate on plateau: 0.000001
2023-06-04 05:45:11,812 P104561 INFO ********* Epoch==16 early stop *********
2023-06-04 05:45:11,874 P104561 INFO Training finished.
2023-06-04 05:45:11,874 P104561 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/PPNet_microvideo1.7m_x1/microvideo1.7m_x1_0d855fe6/PPNet_microvideo1.7m_x1_001_97742c0f.model
2023-06-04 05:45:12,597 P104561 INFO ****** Validation evaluation ******
2023-06-04 05:48:45,008 P104561 INFO [Metrics] gAUC: 0.686542 - AUC: 0.734938 - logloss: 0.411962
2023-06-04 05:48:45,099 P104561 INFO ******** Test evaluation ********
2023-06-04 05:48:45,099 P104561 INFO Loading data...
2023-06-04 05:48:45,099 P104561 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/test.h5
2023-06-04 05:48:49,643 P104561 INFO Test samples: total/3767308, blocks/1
2023-06-04 05:48:49,643 P104561 INFO Loading test data done.
2023-06-04 05:52:20,776 P104561 INFO [Metrics] gAUC: 0.686542 - AUC: 0.734938 - logloss: 0.411962

```
