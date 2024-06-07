## AutoInt_amazonelectronics_x1

A hands-on guide to run the AutoInt model on the AmazonElectronics_x1 dataset.

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

We use the [AutoInt](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/AutoInt) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Amazon/AmazonElectronics_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt+_amazonelectronics_x1_tuner_config_03](./AutoInt+_amazonelectronics_x1_tuner_config_03). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/AutoInt
    nohup python run_expid.py --config XXX/benchmarks/AutoInt/AutoInt+_amazonelectronics_x1_tuner_config_03 --expid AutoInt_amazonelectronics_x1_004_7a937c94 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.878682 | 0.880402 | 0.444114  |


### Logs
```python
2022-08-13 10:16:07,058 P54239 INFO Params: {
    "attention_dim": "512",
    "attention_layers": "2",
    "batch_norm": "True",
    "batch_size": "1024",
    "data_format": "csv",
    "data_root": "../data/Amazon/",
    "dataset_id": "amazonelectronics_x1_b7a43f49",
    "debug_mode": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'user_id', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'item_history', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'cate_history', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "None",
    "gpu": "3",
    "group_id": "user_id",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "layer_norm": "False",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AutoInt",
    "model_id": "AutoInt_amazonelectronics_x1_004_7a937c94",
    "model_root": "./checkpoints/AutoInt_amazonelectronics_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_heads": "1",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2022",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "train_data": "../data/Amazon/AmazonElectronics_x1/train.csv",
    "use_residual": "False",
    "use_scale": "False",
    "use_wide": "True",
    "valid_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "verbose": "1"
}
2022-08-13 10:16:07,059 P54239 INFO Set up feature processor...
2022-08-13 10:16:07,059 P54239 INFO Load feature_map from json: ../data/Amazon/amazonelectronics_x1_b7a43f49/feature_map.json
2022-08-13 10:16:07,060 P54239 INFO Set column index...
2022-08-13 10:16:07,060 P54239 INFO Feature specs: {
    "cate_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803, 'max_len': 100}",
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803}",
    "item_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003, 'max_len': 100}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003}",
    "user_id": "{'type': 'meta'}"
}
2022-08-13 10:16:15,878 P54239 INFO Total number of parameters: 6021118.
2022-08-13 10:16:15,879 P54239 INFO Loading data...
2022-08-13 10:16:15,879 P54239 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/train.h5
2022-08-13 10:16:20,410 P54239 INFO Train samples: total/2608764, blocks/1
2022-08-13 10:16:20,410 P54239 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/valid.h5
2022-08-13 10:16:21,225 P54239 INFO Validation samples: total/384806, blocks/1
2022-08-13 10:16:21,225 P54239 INFO Loading train and validation data done.
2022-08-13 10:16:21,226 P54239 INFO Start training: 2548 batches/epoch
2022-08-13 10:16:21,226 P54239 INFO ************ Epoch=1 start ************
2022-08-13 10:33:56,610 P54239 INFO [Metrics] AUC: 0.844144 - gAUC: 0.841312
2022-08-13 10:33:56,860 P54239 INFO Save best model: monitor(max): 1.685456
2022-08-13 10:33:56,932 P54239 INFO --- 2548/2548 batches finished ---
2022-08-13 10:33:57,059 P54239 INFO Train loss: 0.577754
2022-08-13 10:33:57,059 P54239 INFO ************ Epoch=1 end ************
2022-08-13 10:48:26,616 P54239 INFO [Metrics] AUC: 0.856542 - gAUC: 0.853885
2022-08-13 10:48:26,877 P54239 INFO Save best model: monitor(max): 1.710427
2022-08-13 10:48:27,052 P54239 INFO --- 2548/2548 batches finished ---
2022-08-13 10:48:27,186 P54239 INFO Train loss: 0.544269
2022-08-13 10:48:27,186 P54239 INFO ************ Epoch=2 end ************
2022-08-13 11:03:13,677 P54239 INFO [Metrics] AUC: 0.863775 - gAUC: 0.861800
2022-08-13 11:03:13,965 P54239 INFO Save best model: monitor(max): 1.725576
2022-08-13 11:03:14,033 P54239 INFO --- 2548/2548 batches finished ---
2022-08-13 11:03:14,231 P54239 INFO Train loss: 0.530876
2022-08-13 11:03:14,231 P54239 INFO ************ Epoch=3 end ************
2022-08-13 11:18:11,552 P54239 INFO [Metrics] AUC: 0.866615 - gAUC: 0.864228
2022-08-13 11:18:11,766 P54239 INFO Save best model: monitor(max): 1.730843
2022-08-13 11:18:11,837 P54239 INFO --- 2548/2548 batches finished ---
2022-08-13 11:18:11,970 P54239 INFO Train loss: 0.521046
2022-08-13 11:18:11,970 P54239 INFO ************ Epoch=4 end ************
2022-08-13 11:33:13,493 P54239 INFO [Metrics] AUC: 0.869450 - gAUC: 0.866816
2022-08-13 11:33:13,688 P54239 INFO Save best model: monitor(max): 1.736266
2022-08-13 11:33:13,741 P54239 INFO --- 2548/2548 batches finished ---
2022-08-13 11:33:13,916 P54239 INFO Train loss: 0.514455
2022-08-13 11:33:13,916 P54239 INFO ************ Epoch=5 end ************
2022-08-13 11:48:25,278 P54239 INFO [Metrics] AUC: 0.870693 - gAUC: 0.868911
2022-08-13 11:48:25,616 P54239 INFO Save best model: monitor(max): 1.739603
2022-08-13 11:48:25,670 P54239 INFO --- 2548/2548 batches finished ---
2022-08-13 11:48:25,902 P54239 INFO Train loss: 0.510321
2022-08-13 11:48:25,902 P54239 INFO ************ Epoch=6 end ************
2022-08-13 12:03:32,289 P54239 INFO [Metrics] AUC: 0.872481 - gAUC: 0.869706
2022-08-13 12:03:32,484 P54239 INFO Save best model: monitor(max): 1.742187
2022-08-13 12:03:32,546 P54239 INFO --- 2548/2548 batches finished ---
2022-08-13 12:03:32,735 P54239 INFO Train loss: 0.507727
2022-08-13 12:03:32,735 P54239 INFO ************ Epoch=7 end ************
2022-08-13 12:18:37,247 P54239 INFO [Metrics] AUC: 0.873739 - gAUC: 0.871260
2022-08-13 12:18:37,425 P54239 INFO Save best model: monitor(max): 1.744998
2022-08-13 12:18:37,518 P54239 INFO --- 2548/2548 batches finished ---
2022-08-13 12:18:37,641 P54239 INFO Train loss: 0.505752
2022-08-13 12:18:37,641 P54239 INFO ************ Epoch=8 end ************
2022-08-13 12:33:50,822 P54239 INFO [Metrics] AUC: 0.875264 - gAUC: 0.872892
2022-08-13 12:33:51,035 P54239 INFO Save best model: monitor(max): 1.748156
2022-08-13 12:33:51,108 P54239 INFO --- 2548/2548 batches finished ---
2022-08-13 12:33:51,263 P54239 INFO Train loss: 0.504814
2022-08-13 12:33:51,265 P54239 INFO ************ Epoch=9 end ************
2022-08-13 12:49:13,734 P54239 INFO [Metrics] AUC: 0.874439 - gAUC: 0.872133
2022-08-13 12:49:13,965 P54239 INFO Monitor(max) STOP: 1.746572 !
2022-08-13 12:49:13,965 P54239 INFO Reduce learning rate on plateau: 0.000050
2022-08-13 12:49:13,967 P54239 INFO --- 2548/2548 batches finished ---
2022-08-13 12:49:14,132 P54239 INFO Train loss: 0.503244
2022-08-13 12:49:14,132 P54239 INFO ************ Epoch=10 end ************
2022-08-13 13:04:26,266 P54239 INFO [Metrics] AUC: 0.880402 - gAUC: 0.878682
2022-08-13 13:04:26,457 P54239 INFO Save best model: monitor(max): 1.759084
2022-08-13 13:04:26,508 P54239 INFO --- 2548/2548 batches finished ---
2022-08-13 13:04:26,652 P54239 INFO Train loss: 0.422795
2022-08-13 13:04:26,652 P54239 INFO ************ Epoch=11 end ************
2022-08-13 13:19:28,696 P54239 INFO [Metrics] AUC: 0.880065 - gAUC: 0.878094
2022-08-13 13:19:28,897 P54239 INFO Monitor(max) STOP: 1.758159 !
2022-08-13 13:19:28,897 P54239 INFO Reduce learning rate on plateau: 0.000005
2022-08-13 13:19:28,898 P54239 INFO --- 2548/2548 batches finished ---
2022-08-13 13:19:29,053 P54239 INFO Train loss: 0.384714
2022-08-13 13:19:29,053 P54239 INFO ************ Epoch=12 end ************
2022-08-13 13:32:55,734 P54239 INFO [Metrics] AUC: 0.878835 - gAUC: 0.876753
2022-08-13 13:32:55,918 P54239 INFO Monitor(max) STOP: 1.755589 !
2022-08-13 13:32:55,918 P54239 INFO Reduce learning rate on plateau: 0.000001
2022-08-13 13:32:55,918 P54239 INFO ********* Epoch==13 early stop *********
2022-08-13 13:32:55,919 P54239 INFO --- 2548/2548 batches finished ---
2022-08-13 13:32:56,021 P54239 INFO Train loss: 0.347026
2022-08-13 13:32:56,022 P54239 INFO Training finished.
2022-08-13 13:32:56,022 P54239 INFO Load best model: /cache/FuxiCTRv2.0/benchmark/checkpoints/AutoInt_amazonelectronics_x1/amazonelectronics_x1_b7a43f49/AutoInt_amazonelectronics_x1_004_7a937c94.model
2022-08-13 13:33:01,029 P54239 INFO ****** Validation evaluation ******
2022-08-13 13:35:31,002 P54239 INFO [Metrics] gAUC: 0.878682 - AUC: 0.880402 - logloss: 0.444114
2022-08-13 13:35:31,261 P54239 INFO ******** Test evaluation ********
2022-08-13 13:35:31,261 P54239 INFO Loading data...
2022-08-13 13:35:31,261 P54239 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/test.h5
2022-08-13 13:35:31,700 P54239 INFO Test samples: total/384806, blocks/1
2022-08-13 13:35:31,700 P54239 INFO Loading test data done.
2022-08-13 13:38:04,454 P54239 INFO [Metrics] gAUC: 0.878682 - AUC: 0.880402 - logloss: 0.444114

```
