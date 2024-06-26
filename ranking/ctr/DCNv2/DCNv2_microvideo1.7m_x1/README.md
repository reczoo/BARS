## DCNv2_microvideo1.7m_x1

A hands-on guide to run the DCNv2 model on the MicroVideo1.7M_x1 dataset.

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

We use the [DCNv2](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/DCNv2) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/MicroVideo1.7M/MicroVideo1.7M_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DCNv2_microvideo1.7m_x1_tuner_config_05](./DCNv2_microvideo1.7m_x1_tuner_config_05). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DCNv2
    nohup python run_expid.py --config XXX/benchmarks/DCNv2/DCNv2_microvideo1.7m_x1_tuner_config_05 --expid DCNv2_microvideo1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.685946 | 0.734367 | 0.412206  |


### Logs
```python
2022-08-19 18:55:38,708 P21434 INFO Params: {
    "batch_norm": "True",
    "batch_size": "2048",
    "data_format": "csv",
    "data_root": "../data/MicroVideo1.7M/",
    "dataset_id": "microvideo1.7m_x1_0d855fe6",
    "debug_mode": "False",
    "dnn_activations": "relu",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'name': 'clicked_items', 'padding': 'pre', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'clicked_categories', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'clicked_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'clicked_categories'}]",
    "gpu": "0",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "low_rank": "32",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCNv2",
    "model_id": "DCNv2_microvideo1.7m_x1_017_9199218b",
    "model_root": "./checkpoints/DCNv2_microvideo1.7m_x1/",
    "model_structure": "parallel",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_cross_layers": "3",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "parallel_dnn_hidden_units": "[1024, 512, 256]",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2022",
    "shuffle": "True",
    "stacked_dnn_hidden_units": "[500, 500, 500]",
    "task": "binary_classification",
    "test_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "train_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/train.csv",
    "use_low_rank_mixture": "False",
    "valid_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "verbose": "1"
}
2022-08-19 18:55:38,709 P21434 INFO Set up feature processor...
2022-08-19 18:55:38,709 P21434 INFO Load feature_map from json: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/feature_map.json
2022-08-19 18:55:38,709 P21434 INFO Set column index...
2022-08-19 18:55:38,709 P21434 INFO Feature specs: {
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514}",
    "clicked_categories": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514, 'max_len': 100}",
    "clicked_items": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'max_len': 100}",
    "group_id": "{'type': 'meta'}",
    "item_id": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10987, 'vocab_size': 10988}"
}
2022-08-19 18:55:50,739 P21434 INFO Total number of parameters: 2041473.
2022-08-19 18:55:50,740 P21434 INFO Loading data...
2022-08-19 18:55:50,740 P21434 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/train.h5
2022-08-19 18:56:04,594 P21434 INFO Train samples: total/8970309, blocks/1
2022-08-19 18:56:04,595 P21434 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/valid.h5
2022-08-19 18:56:10,222 P21434 INFO Validation samples: total/3767308, blocks/1
2022-08-19 18:56:10,222 P21434 INFO Loading train and validation data done.
2022-08-19 18:56:10,222 P21434 INFO Start training: 4381 batches/epoch
2022-08-19 18:56:10,222 P21434 INFO ************ Epoch=1 start ************
2022-08-19 19:24:57,354 P21434 INFO [Metrics] AUC: 0.719126 - gAUC: 0.671827
2022-08-19 19:24:57,369 P21434 INFO Save best model: monitor(max): 1.390953
2022-08-19 19:24:59,483 P21434 INFO --- 4381/4381 batches finished ---
2022-08-19 19:24:59,600 P21434 INFO Train loss: 0.460405
2022-08-19 19:24:59,600 P21434 INFO ************ Epoch=1 end ************
2022-08-19 19:53:47,399 P21434 INFO [Metrics] AUC: 0.722097 - gAUC: 0.673882
2022-08-19 19:53:47,429 P21434 INFO Save best model: monitor(max): 1.395979
2022-08-19 19:53:50,704 P21434 INFO --- 4381/4381 batches finished ---
2022-08-19 19:53:50,922 P21434 INFO Train loss: 0.440669
2022-08-19 19:53:50,923 P21434 INFO ************ Epoch=2 end ************
2022-08-19 20:22:27,601 P21434 INFO [Metrics] AUC: 0.724128 - gAUC: 0.676651
2022-08-19 20:22:27,634 P21434 INFO Save best model: monitor(max): 1.400779
2022-08-19 20:22:30,117 P21434 INFO --- 4381/4381 batches finished ---
2022-08-19 20:22:30,268 P21434 INFO Train loss: 0.438625
2022-08-19 20:22:30,268 P21434 INFO ************ Epoch=3 end ************
2022-08-19 20:51:01,221 P21434 INFO [Metrics] AUC: 0.723993 - gAUC: 0.676090
2022-08-19 20:51:01,254 P21434 INFO Monitor(max) STOP: 1.400082 !
2022-08-19 20:51:01,254 P21434 INFO Reduce learning rate on plateau: 0.000050
2022-08-19 20:51:01,255 P21434 INFO --- 4381/4381 batches finished ---
2022-08-19 20:51:01,379 P21434 INFO Train loss: 0.437320
2022-08-19 20:51:01,379 P21434 INFO ************ Epoch=4 end ************
2022-08-19 21:19:37,519 P21434 INFO [Metrics] AUC: 0.732426 - gAUC: 0.683952
2022-08-19 21:19:37,530 P21434 INFO Save best model: monitor(max): 1.416378
2022-08-19 21:19:40,528 P21434 INFO --- 4381/4381 batches finished ---
2022-08-19 21:19:40,664 P21434 INFO Train loss: 0.425814
2022-08-19 21:19:40,664 P21434 INFO ************ Epoch=5 end ************
2022-08-19 21:48:20,817 P21434 INFO [Metrics] AUC: 0.733101 - gAUC: 0.684159
2022-08-19 21:48:20,832 P21434 INFO Save best model: monitor(max): 1.417260
2022-08-19 21:48:23,698 P21434 INFO --- 4381/4381 batches finished ---
2022-08-19 21:48:23,860 P21434 INFO Train loss: 0.421271
2022-08-19 21:48:23,860 P21434 INFO ************ Epoch=6 end ************
2022-08-19 22:16:59,781 P21434 INFO [Metrics] AUC: 0.733482 - gAUC: 0.684500
2022-08-19 22:16:59,810 P21434 INFO Save best model: monitor(max): 1.417982
2022-08-19 22:17:02,601 P21434 INFO --- 4381/4381 batches finished ---
2022-08-19 22:17:02,696 P21434 INFO Train loss: 0.419204
2022-08-19 22:17:02,697 P21434 INFO ************ Epoch=7 end ************
2022-08-19 22:45:58,559 P21434 INFO [Metrics] AUC: 0.733402 - gAUC: 0.684671
2022-08-19 22:45:58,569 P21434 INFO Save best model: monitor(max): 1.418074
2022-08-19 22:46:01,995 P21434 INFO --- 4381/4381 batches finished ---
2022-08-19 22:46:02,129 P21434 INFO Train loss: 0.417584
2022-08-19 22:46:02,129 P21434 INFO ************ Epoch=8 end ************
2022-08-19 23:14:41,656 P21434 INFO [Metrics] AUC: 0.733651 - gAUC: 0.685102
2022-08-19 23:14:41,671 P21434 INFO Save best model: monitor(max): 1.418753
2022-08-19 23:14:44,320 P21434 INFO --- 4381/4381 batches finished ---
2022-08-19 23:14:44,418 P21434 INFO Train loss: 0.416285
2022-08-19 23:14:44,418 P21434 INFO ************ Epoch=9 end ************
2022-08-19 23:43:23,405 P21434 INFO [Metrics] AUC: 0.733894 - gAUC: 0.685278
2022-08-19 23:43:23,468 P21434 INFO Save best model: monitor(max): 1.419172
2022-08-19 23:43:26,719 P21434 INFO --- 4381/4381 batches finished ---
2022-08-19 23:43:26,829 P21434 INFO Train loss: 0.415121
2022-08-19 23:43:26,829 P21434 INFO ************ Epoch=10 end ************
2022-08-20 00:11:55,279 P21434 INFO [Metrics] AUC: 0.734049 - gAUC: 0.685251
2022-08-20 00:11:55,396 P21434 INFO Save best model: monitor(max): 1.419300
2022-08-20 00:11:58,312 P21434 INFO --- 4381/4381 batches finished ---
2022-08-20 00:11:58,474 P21434 INFO Train loss: 0.414035
2022-08-20 00:11:58,474 P21434 INFO ************ Epoch=11 end ************
2022-08-20 00:40:44,326 P21434 INFO [Metrics] AUC: 0.734058 - gAUC: 0.685505
2022-08-20 00:40:44,335 P21434 INFO Save best model: monitor(max): 1.419562
2022-08-20 00:40:47,450 P21434 INFO --- 4381/4381 batches finished ---
2022-08-20 00:40:47,623 P21434 INFO Train loss: 0.413117
2022-08-20 00:40:47,624 P21434 INFO ************ Epoch=12 end ************
2022-08-20 01:09:15,468 P21434 INFO [Metrics] AUC: 0.734367 - gAUC: 0.685946
2022-08-20 01:09:15,479 P21434 INFO Save best model: monitor(max): 1.420313
2022-08-20 01:09:18,512 P21434 INFO --- 4381/4381 batches finished ---
2022-08-20 01:09:18,647 P21434 INFO Train loss: 0.412190
2022-08-20 01:09:18,647 P21434 INFO ************ Epoch=13 end ************
2022-08-20 01:37:53,024 P21434 INFO [Metrics] AUC: 0.734145 - gAUC: 0.685893
2022-08-20 01:37:53,039 P21434 INFO Monitor(max) STOP: 1.420038 !
2022-08-20 01:37:53,039 P21434 INFO Reduce learning rate on plateau: 0.000005
2022-08-20 01:37:53,040 P21434 INFO --- 4381/4381 batches finished ---
2022-08-20 01:37:53,157 P21434 INFO Train loss: 0.411289
2022-08-20 01:37:53,157 P21434 INFO ************ Epoch=14 end ************
2022-08-20 02:06:32,775 P21434 INFO [Metrics] AUC: 0.734109 - gAUC: 0.685755
2022-08-20 02:06:32,783 P21434 INFO Monitor(max) STOP: 1.419864 !
2022-08-20 02:06:32,783 P21434 INFO Reduce learning rate on plateau: 0.000001
2022-08-20 02:06:32,783 P21434 INFO ********* Epoch==15 early stop *********
2022-08-20 02:06:32,784 P21434 INFO --- 4381/4381 batches finished ---
2022-08-20 02:06:32,884 P21434 INFO Train loss: 0.404511
2022-08-20 02:06:32,884 P21434 INFO Training finished.
2022-08-20 02:06:32,884 P21434 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DCNv2_microvideo1.7m_x1/microvideo1.7m_x1_0d855fe6/DCNv2_microvideo1.7m_x1_017_9199218b.model
2022-08-20 02:06:33,866 P21434 INFO ****** Validation evaluation ******
2022-08-20 02:15:15,760 P21434 INFO [Metrics] gAUC: 0.685946 - AUC: 0.734367 - logloss: 0.412206
2022-08-20 02:15:15,977 P21434 INFO ******** Test evaluation ********
2022-08-20 02:15:15,977 P21434 INFO Loading data...
2022-08-20 02:15:15,977 P21434 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/test.h5
2022-08-20 02:15:23,861 P21434 INFO Test samples: total/3767308, blocks/1
2022-08-20 02:15:23,862 P21434 INFO Loading test data done.
2022-08-20 02:23:56,024 P21434 INFO [Metrics] gAUC: 0.685946 - AUC: 0.734367 - logloss: 0.412206

```
