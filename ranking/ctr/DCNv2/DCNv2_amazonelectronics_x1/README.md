## DCNv2_amazonelectronics_x1

A hands-on guide to run the DCNv2 model on the AmazonElectronics_x1 dataset.

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
Please refer to [AmazonElectronics_x1](https://github.com/reczoo/Datasets/tree/main/Amazon/AmazonElectronics_x1) to get the dataset details.

### Code

We use the [DCNv2](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/DCNv2) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Amazon/AmazonElectronics_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DCNv2_amazonelectronics_x1_tuner_config_04](./DCNv2_amazonelectronics_x1_tuner_config_04). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DCNv2
    nohup python run_expid.py --config XXX/benchmarks/DCNv2/DCNv2_amazonelectronics_x1_tuner_config_04 --expid DCNv2_amazonelectronics_x1_010_7212260c --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.879046 | 0.881160 | 0.444522  |


### Logs
```python
2022-08-11 13:23:08,008 P64748 INFO Params: {
    "batch_norm": "True",
    "batch_size": "1024",
    "data_format": "csv",
    "data_root": "../data/Amazon/",
    "dataset_id": "amazonelectronics_x1_b7a43f49",
    "debug_mode": "False",
    "dnn_activations": "relu",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.005",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'user_id', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'item_history', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'cate_history', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}]",
    "gpu": "1",
    "group_index": "user_id",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "low_rank": "32",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCNv2",
    "model_id": "DCNv2_amazonelectronics_x1_010_7212260c",
    "model_root": "./checkpoints/DCNv2_amazonelectronics_x1/",
    "model_structure": "parallel",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_cross_layers": "2",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_dnn_hidden_units": "[1024, 512, 256]",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2022",
    "shuffle": "True",
    "stacked_dnn_hidden_units": "[500, 500, 500]",
    "task": "binary_classification",
    "test_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "train_data": "../data/Amazon/AmazonElectronics_x1/train.csv",
    "use_low_rank_mixture": "False",
    "valid_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "verbose": "1"
}
2022-08-11 13:23:08,009 P64748 INFO Set up feature processor...
2022-08-11 13:23:08,009 P64748 INFO Load feature_map from json: ../data/Amazon/amazonelectronics_x1_b7a43f49/feature_map.json
2022-08-11 13:23:08,010 P64748 INFO Set column index...
2022-08-11 13:23:08,010 P64748 INFO Feature specs: {
    "cate_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803, 'max_len': 100}",
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803}",
    "item_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003, 'max_len': 100}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003}",
    "user_id": "{'type': 'meta'}"
}
2022-08-11 13:23:17,378 P64748 INFO Total number of parameters: 5138561.
2022-08-11 13:23:17,379 P64748 INFO Loading data...
2022-08-11 13:23:17,380 P64748 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/train.h5
2022-08-11 13:23:20,932 P64748 INFO Train samples: total/2608764, blocks/1
2022-08-11 13:23:20,932 P64748 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/valid.h5
2022-08-11 13:23:21,437 P64748 INFO Validation samples: total/384806, blocks/1
2022-08-11 13:23:21,437 P64748 INFO Loading train and validation data done.
2022-08-11 13:23:21,438 P64748 INFO Start training: 2548 batches/epoch
2022-08-11 13:23:21,438 P64748 INFO ************ Epoch=1 start ************
2022-08-11 13:40:03,583 P64748 INFO [Metrics] AUC: 0.833371 - gAUC: 0.832378
2022-08-11 13:40:03,892 P64748 INFO Save best model: monitor(max): 1.665749
2022-08-11 13:40:03,992 P64748 INFO --- 2548/2548 batches finished ---
2022-08-11 13:40:04,217 P64748 INFO Train loss: 0.637840
2022-08-11 13:40:04,217 P64748 INFO ************ Epoch=1 end ************
2022-08-11 13:56:25,264 P64748 INFO [Metrics] AUC: 0.846124 - gAUC: 0.843890
2022-08-11 13:56:25,595 P64748 INFO Save best model: monitor(max): 1.690014
2022-08-11 13:56:25,667 P64748 INFO --- 2548/2548 batches finished ---
2022-08-11 13:56:25,830 P64748 INFO Train loss: 0.591507
2022-08-11 13:56:25,831 P64748 INFO ************ Epoch=2 end ************
2022-08-11 14:13:19,050 P64748 INFO [Metrics] AUC: 0.852824 - gAUC: 0.850860
2022-08-11 14:13:19,239 P64748 INFO Save best model: monitor(max): 1.703684
2022-08-11 14:13:19,323 P64748 INFO --- 2548/2548 batches finished ---
2022-08-11 14:13:19,751 P64748 INFO Train loss: 0.574932
2022-08-11 14:13:19,751 P64748 INFO ************ Epoch=3 end ************
2022-08-11 14:30:14,484 P64748 INFO [Metrics] AUC: 0.855448 - gAUC: 0.853142
2022-08-11 14:30:14,729 P64748 INFO Save best model: monitor(max): 1.708590
2022-08-11 14:30:14,819 P64748 INFO --- 2548/2548 batches finished ---
2022-08-11 14:30:15,192 P64748 INFO Train loss: 0.569256
2022-08-11 14:30:15,192 P64748 INFO ************ Epoch=4 end ************
2022-08-11 14:47:14,018 P64748 INFO [Metrics] AUC: 0.856124 - gAUC: 0.853609
2022-08-11 14:47:14,384 P64748 INFO Save best model: monitor(max): 1.709733
2022-08-11 14:47:14,818 P64748 INFO --- 2548/2548 batches finished ---
2022-08-11 14:47:15,012 P64748 INFO Train loss: 0.565748
2022-08-11 14:47:15,012 P64748 INFO ************ Epoch=5 end ************
2022-08-11 15:04:22,912 P64748 INFO [Metrics] AUC: 0.856841 - gAUC: 0.854857
2022-08-11 15:04:23,231 P64748 INFO Save best model: monitor(max): 1.711698
2022-08-11 15:04:23,326 P64748 INFO --- 2548/2548 batches finished ---
2022-08-11 15:04:23,674 P64748 INFO Train loss: 0.562968
2022-08-11 15:04:23,674 P64748 INFO ************ Epoch=6 end ************
2022-08-11 15:21:04,150 P64748 INFO [Metrics] AUC: 0.857511 - gAUC: 0.855267
2022-08-11 15:21:04,462 P64748 INFO Save best model: monitor(max): 1.712778
2022-08-11 15:21:04,549 P64748 INFO --- 2548/2548 batches finished ---
2022-08-11 15:21:04,767 P64748 INFO Train loss: 0.561129
2022-08-11 15:21:04,767 P64748 INFO ************ Epoch=7 end ************
2022-08-11 15:37:23,517 P64748 INFO [Metrics] AUC: 0.858628 - gAUC: 0.856000
2022-08-11 15:37:23,746 P64748 INFO Save best model: monitor(max): 1.714628
2022-08-11 15:37:23,855 P64748 INFO --- 2548/2548 batches finished ---
2022-08-11 15:37:24,032 P64748 INFO Train loss: 0.560118
2022-08-11 15:37:24,032 P64748 INFO ************ Epoch=8 end ************
2022-08-11 15:53:14,859 P64748 INFO [Metrics] AUC: 0.857410 - gAUC: 0.855049
2022-08-11 15:53:15,088 P64748 INFO Monitor(max) STOP: 1.712459 !
2022-08-11 15:53:15,088 P64748 INFO Reduce learning rate on plateau: 0.000050
2022-08-11 15:53:15,089 P64748 INFO --- 2548/2548 batches finished ---
2022-08-11 15:53:15,253 P64748 INFO Train loss: 0.559008
2022-08-11 15:53:15,253 P64748 INFO ************ Epoch=9 end ************
2022-08-11 16:08:39,176 P64748 INFO [Metrics] AUC: 0.876588 - gAUC: 0.874311
2022-08-11 16:08:39,527 P64748 INFO Save best model: monitor(max): 1.750898
2022-08-11 16:08:39,601 P64748 INFO --- 2548/2548 batches finished ---
2022-08-11 16:08:39,817 P64748 INFO Train loss: 0.464809
2022-08-11 16:08:39,817 P64748 INFO ************ Epoch=10 end ************
2022-08-11 16:23:06,564 P64748 INFO [Metrics] AUC: 0.880173 - gAUC: 0.877585
2022-08-11 16:23:06,763 P64748 INFO Save best model: monitor(max): 1.757759
2022-08-11 16:23:06,835 P64748 INFO --- 2548/2548 batches finished ---
2022-08-11 16:23:07,015 P64748 INFO Train loss: 0.420528
2022-08-11 16:23:07,015 P64748 INFO ************ Epoch=11 end ************
2022-08-11 16:35:04,392 P64748 INFO [Metrics] AUC: 0.880314 - gAUC: 0.877990
2022-08-11 16:35:04,632 P64748 INFO Save best model: monitor(max): 1.758304
2022-08-11 16:35:04,699 P64748 INFO --- 2548/2548 batches finished ---
2022-08-11 16:35:04,833 P64748 INFO Train loss: 0.401271
2022-08-11 16:35:04,833 P64748 INFO ************ Epoch=12 end ************
2022-08-11 16:43:59,236 P64748 INFO [Metrics] AUC: 0.881160 - gAUC: 0.879046
2022-08-11 16:43:59,401 P64748 INFO Save best model: monitor(max): 1.760205
2022-08-11 16:43:59,451 P64748 INFO --- 2548/2548 batches finished ---
2022-08-11 16:43:59,554 P64748 INFO Train loss: 0.389149
2022-08-11 16:43:59,554 P64748 INFO ************ Epoch=13 end ************
2022-08-11 16:49:39,622 P64748 INFO [Metrics] AUC: 0.878222 - gAUC: 0.876005
2022-08-11 16:49:39,755 P64748 INFO Monitor(max) STOP: 1.754227 !
2022-08-11 16:49:39,755 P64748 INFO Reduce learning rate on plateau: 0.000005
2022-08-11 16:49:39,756 P64748 INFO --- 2548/2548 batches finished ---
2022-08-11 16:49:39,862 P64748 INFO Train loss: 0.379442
2022-08-11 16:49:39,862 P64748 INFO ************ Epoch=14 end ************
2022-08-11 16:53:18,663 P64748 INFO [Metrics] AUC: 0.875775 - gAUC: 0.873552
2022-08-11 16:53:18,821 P64748 INFO Monitor(max) STOP: 1.749327 !
2022-08-11 16:53:18,821 P64748 INFO Reduce learning rate on plateau: 0.000001
2022-08-11 16:53:18,822 P64748 INFO ********* Epoch==15 early stop *********
2022-08-11 16:53:18,822 P64748 INFO --- 2548/2548 batches finished ---
2022-08-11 16:53:18,897 P64748 INFO Train loss: 0.321335
2022-08-11 16:53:18,898 P64748 INFO Training finished.
2022-08-11 16:53:18,898 P64748 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DCNv2_amazonelectronics_x1/amazonelectronics_x1_b7a43f49/DCNv2_amazonelectronics_x1_010_7212260c.model
2022-08-11 16:53:19,083 P64748 INFO ****** Validation evaluation ******
2022-08-11 16:54:03,072 P64748 INFO [Metrics] gAUC: 0.879046 - AUC: 0.881160 - logloss: 0.444522
2022-08-11 16:54:03,291 P64748 INFO ******** Test evaluation ********
2022-08-11 16:54:03,291 P64748 INFO Loading data...
2022-08-11 16:54:03,291 P64748 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/test.h5
2022-08-11 16:54:03,713 P64748 INFO Test samples: total/384806, blocks/1
2022-08-11 16:54:03,714 P64748 INFO Loading test data done.
2022-08-11 16:54:47,113 P64748 INFO [Metrics] gAUC: 0.879046 - AUC: 0.881160 - logloss: 0.444522

```
