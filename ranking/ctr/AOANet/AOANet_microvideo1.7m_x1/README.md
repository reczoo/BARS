## AOANet_microvideo1.7m_x1

A hands-on guide to run the AOANet model on the MicroVideo1.7M_x1 dataset.

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
  fuxictr: 2.0.1

  ```

### Dataset
Please refer to [MicroVideo1.7M_x1](https://github.com/reczoo/Datasets/tree/main/MicroVideo/MicroVideo1.7M_x1) to get the dataset details.

### Code

We use the [AOANet](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/AOANet) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/MicroVideo1.7M/MicroVideo1.7M_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AOANet_microvideo1.7m_x1_tuner_config_01](./AOANet_microvideo1.7m_x1_tuner_config_01). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/AOANet
    nohup python run_expid.py --config XXX/benchmarks/AOANet/AOANet_microvideo1.7m_x1_tuner_config_01 --expid AOANet_microvideo1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.685797 | 0.734644 | 0.412282  |


### Logs
```python
2022-08-15 10:06:33,855 P33337 INFO Params: {
    "batch_norm": "True",
    "batch_size": "2048",
    "data_format": "csv",
    "data_root": "../data/MicroVideo1.7M/",
    "dataset_id": "microvideo1.7m_x1_0d855fe6",
    "debug_mode": "False",
    "dnn_hidden_activations": "ReLU",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'name': 'clicked_items', 'padding': 'pre', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'clicked_categories', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'clicked_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'clicked_categories'}]",
    "gpu": "4",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AOANet",
    "model_id": "AOANet_microvideo1.7m_x1_005_5a9d3de4",
    "model_root": "./checkpoints/AOANet_microvideo1.7m_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_interaction_layers": "1",
    "num_subspaces": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2022",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "train_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/train.csv",
    "valid_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "verbose": "1"
}
2022-08-15 10:06:33,856 P33337 INFO Set up feature processor...
2022-08-15 10:06:33,856 P33337 INFO Load feature_map from json: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/feature_map.json
2022-08-15 10:06:33,856 P33337 INFO Set column index...
2022-08-15 10:06:33,856 P33337 INFO Feature specs: {
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514}",
    "clicked_categories": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514, 'max_len': 100}",
    "clicked_items": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'max_len': 100}",
    "group_id": "{'type': 'meta'}",
    "item_id": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10987, 'vocab_size': 10988}"
}
2022-08-15 10:06:44,113 P33337 INFO Total number of parameters: 1749989.
2022-08-15 10:06:44,113 P33337 INFO Loading data...
2022-08-15 10:06:44,113 P33337 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/train.h5
2022-08-15 10:06:56,420 P33337 INFO Train samples: total/8970309, blocks/1
2022-08-15 10:06:56,420 P33337 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/valid.h5
2022-08-15 10:07:01,196 P33337 INFO Validation samples: total/3767308, blocks/1
2022-08-15 10:07:01,196 P33337 INFO Loading train and validation data done.
2022-08-15 10:07:01,196 P33337 INFO Start training: 4381 batches/epoch
2022-08-15 10:07:01,196 P33337 INFO ************ Epoch=1 start ************
2022-08-15 10:28:09,953 P33337 INFO [Metrics] AUC: 0.713483 - gAUC: 0.666375
2022-08-15 10:28:09,974 P33337 INFO Save best model: monitor(max): 1.379858
2022-08-15 10:28:11,818 P33337 INFO --- 4381/4381 batches finished ---
2022-08-15 10:28:11,898 P33337 INFO Train loss: 0.472231
2022-08-15 10:28:11,899 P33337 INFO ************ Epoch=1 end ************
2022-08-15 10:49:13,808 P33337 INFO [Metrics] AUC: 0.719591 - gAUC: 0.671852
2022-08-15 10:49:13,829 P33337 INFO Save best model: monitor(max): 1.391444
2022-08-15 10:49:15,804 P33337 INFO --- 4381/4381 batches finished ---
2022-08-15 10:49:15,951 P33337 INFO Train loss: 0.444854
2022-08-15 10:49:15,951 P33337 INFO ************ Epoch=2 end ************
2022-08-15 11:10:19,847 P33337 INFO [Metrics] AUC: 0.721542 - gAUC: 0.673131
2022-08-15 11:10:19,863 P33337 INFO Save best model: monitor(max): 1.394672
2022-08-15 11:10:21,868 P33337 INFO --- 4381/4381 batches finished ---
2022-08-15 11:10:21,949 P33337 INFO Train loss: 0.442810
2022-08-15 11:10:21,949 P33337 INFO ************ Epoch=3 end ************
2022-08-15 11:31:26,734 P33337 INFO [Metrics] AUC: 0.723944 - gAUC: 0.676222
2022-08-15 11:31:26,745 P33337 INFO Save best model: monitor(max): 1.400166
2022-08-15 11:31:28,861 P33337 INFO --- 4381/4381 batches finished ---
2022-08-15 11:31:28,970 P33337 INFO Train loss: 0.441444
2022-08-15 11:31:28,970 P33337 INFO ************ Epoch=4 end ************
2022-08-15 11:52:40,456 P33337 INFO [Metrics] AUC: 0.724108 - gAUC: 0.676078
2022-08-15 11:52:40,467 P33337 INFO Save best model: monitor(max): 1.400187
2022-08-15 11:52:42,608 P33337 INFO --- 4381/4381 batches finished ---
2022-08-15 11:52:42,709 P33337 INFO Train loss: 0.440616
2022-08-15 11:52:42,709 P33337 INFO ************ Epoch=5 end ************
2022-08-15 12:13:43,534 P33337 INFO [Metrics] AUC: 0.725112 - gAUC: 0.676539
2022-08-15 12:13:43,549 P33337 INFO Save best model: monitor(max): 1.401651
2022-08-15 12:13:45,679 P33337 INFO --- 4381/4381 batches finished ---
2022-08-15 12:13:45,751 P33337 INFO Train loss: 0.440296
2022-08-15 12:13:45,751 P33337 INFO ************ Epoch=6 end ************
2022-08-15 12:34:47,549 P33337 INFO [Metrics] AUC: 0.726551 - gAUC: 0.678736
2022-08-15 12:34:47,561 P33337 INFO Save best model: monitor(max): 1.405286
2022-08-15 12:34:49,527 P33337 INFO --- 4381/4381 batches finished ---
2022-08-15 12:34:49,629 P33337 INFO Train loss: 0.439944
2022-08-15 12:34:49,629 P33337 INFO ************ Epoch=7 end ************
2022-08-15 12:55:59,448 P33337 INFO [Metrics] AUC: 0.725459 - gAUC: 0.678000
2022-08-15 12:55:59,480 P33337 INFO Monitor(max) STOP: 1.403459 !
2022-08-15 12:55:59,480 P33337 INFO Reduce learning rate on plateau: 0.000050
2022-08-15 12:55:59,480 P33337 INFO --- 4381/4381 batches finished ---
2022-08-15 12:55:59,556 P33337 INFO Train loss: 0.439722
2022-08-15 12:55:59,556 P33337 INFO ************ Epoch=8 end ************
2022-08-15 13:17:00,917 P33337 INFO [Metrics] AUC: 0.732833 - gAUC: 0.684378
2022-08-15 13:17:00,930 P33337 INFO Save best model: monitor(max): 1.417211
2022-08-15 13:17:03,241 P33337 INFO --- 4381/4381 batches finished ---
2022-08-15 13:17:03,333 P33337 INFO Train loss: 0.426951
2022-08-15 13:17:03,333 P33337 INFO ************ Epoch=9 end ************
2022-08-15 13:38:16,506 P33337 INFO [Metrics] AUC: 0.733745 - gAUC: 0.685011
2022-08-15 13:38:16,521 P33337 INFO Save best model: monitor(max): 1.418756
2022-08-15 13:38:18,524 P33337 INFO --- 4381/4381 batches finished ---
2022-08-15 13:38:18,634 P33337 INFO Train loss: 0.421657
2022-08-15 13:38:18,634 P33337 INFO ************ Epoch=10 end ************
2022-08-15 13:59:01,740 P33337 INFO [Metrics] AUC: 0.734088 - gAUC: 0.685153
2022-08-15 13:59:01,753 P33337 INFO Save best model: monitor(max): 1.419241
2022-08-15 13:59:03,877 P33337 INFO --- 4381/4381 batches finished ---
2022-08-15 13:59:03,954 P33337 INFO Train loss: 0.419623
2022-08-15 13:59:03,954 P33337 INFO ************ Epoch=11 end ************
2022-08-15 14:20:05,087 P33337 INFO [Metrics] AUC: 0.734475 - gAUC: 0.685598
2022-08-15 14:20:05,105 P33337 INFO Save best model: monitor(max): 1.420073
2022-08-15 14:20:07,265 P33337 INFO --- 4381/4381 batches finished ---
2022-08-15 14:20:07,371 P33337 INFO Train loss: 0.418028
2022-08-15 14:20:07,371 P33337 INFO ************ Epoch=12 end ************
2022-08-15 14:41:10,465 P33337 INFO [Metrics] AUC: 0.734641 - gAUC: 0.685718
2022-08-15 14:41:10,497 P33337 INFO Save best model: monitor(max): 1.420359
2022-08-15 14:41:12,690 P33337 INFO --- 4381/4381 batches finished ---
2022-08-15 14:41:12,799 P33337 INFO Train loss: 0.416585
2022-08-15 14:41:12,799 P33337 INFO ************ Epoch=13 end ************
2022-08-15 15:01:32,886 P33337 INFO [Metrics] AUC: 0.734538 - gAUC: 0.685620
2022-08-15 15:01:32,898 P33337 INFO Monitor(max) STOP: 1.420158 !
2022-08-15 15:01:32,898 P33337 INFO Reduce learning rate on plateau: 0.000005
2022-08-15 15:01:32,899 P33337 INFO --- 4381/4381 batches finished ---
2022-08-15 15:01:32,975 P33337 INFO Train loss: 0.415223
2022-08-15 15:01:32,975 P33337 INFO ************ Epoch=14 end ************
2022-08-15 15:21:34,379 P33337 INFO [Metrics] AUC: 0.734644 - gAUC: 0.685797
2022-08-15 15:21:34,391 P33337 INFO Save best model: monitor(max): 1.420441
2022-08-15 15:21:36,453 P33337 INFO --- 4381/4381 batches finished ---
2022-08-15 15:21:36,614 P33337 INFO Train loss: 0.409436
2022-08-15 15:21:36,614 P33337 INFO ************ Epoch=15 end ************
2022-08-15 15:41:37,738 P33337 INFO [Metrics] AUC: 0.734586 - gAUC: 0.685794
2022-08-15 15:41:37,751 P33337 INFO Monitor(max) STOP: 1.420380 !
2022-08-15 15:41:37,751 P33337 INFO Reduce learning rate on plateau: 0.000001
2022-08-15 15:41:37,751 P33337 INFO --- 4381/4381 batches finished ---
2022-08-15 15:41:37,822 P33337 INFO Train loss: 0.408784
2022-08-15 15:41:37,822 P33337 INFO ************ Epoch=16 end ************
2022-08-15 16:01:01,564 P33337 INFO [Metrics] AUC: 0.734531 - gAUC: 0.685754
2022-08-15 16:01:01,575 P33337 INFO Monitor(max) STOP: 1.420286 !
2022-08-15 16:01:01,576 P33337 INFO Reduce learning rate on plateau: 0.000001
2022-08-15 16:01:01,576 P33337 INFO ********* Epoch==17 early stop *********
2022-08-15 16:01:01,576 P33337 INFO --- 4381/4381 batches finished ---
2022-08-15 16:01:01,658 P33337 INFO Train loss: 0.407821
2022-08-15 16:01:01,659 P33337 INFO Training finished.
2022-08-15 16:01:01,659 P33337 INFO Load best model: /cache/FuxiCTRv2.0/benchmark/checkpoints/AOANet_microvideo1.7m_x1/microvideo1.7m_x1_0d855fe6/AOANet_microvideo1.7m_x1_005_5a9d3de4.model
2022-08-15 16:01:02,827 P33337 INFO ****** Validation evaluation ******
2022-08-15 16:06:26,852 P33337 INFO [Metrics] gAUC: 0.685797 - AUC: 0.734644 - logloss: 0.412282
2022-08-15 16:06:26,961 P33337 INFO ******** Test evaluation ********
2022-08-15 16:06:26,962 P33337 INFO Loading data...
2022-08-15 16:06:26,962 P33337 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/test.h5
2022-08-15 16:06:31,913 P33337 INFO Test samples: total/3767308, blocks/1
2022-08-15 16:06:31,913 P33337 INFO Loading test data done.
2022-08-15 16:11:58,280 P33337 INFO [Metrics] gAUC: 0.685797 - AUC: 0.734644 - logloss: 0.412282

```
