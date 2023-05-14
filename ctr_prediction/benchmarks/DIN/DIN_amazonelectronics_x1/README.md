## DIN_amazonelectronics_x1

A hands-on guide to run the DIN model on the AmazonElectronics_x1 dataset.

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
Please refer to the BARS dataset [AmazonElectronics_x1](https://github.com/openbenchmark/BARS/blob/main/datasets/Amazon#AmazonElectronics_x1) to get data ready.

### Code

We use the [DIN](https://github.com/xue-pai/FuxiCTR/blob/v2.0.1/model_zoo/DIN) model code from [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Amazon/AmazonElectronics_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DIN_amazonelectronics_x1_tuner_config_11](./DIN_amazonelectronics_x1_tuner_config_11). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DIN
    nohup python run_expid.py --config XXX/benchmarks/DIN/DIN_amazonelectronics_x1_tuner_config_11 --expid DIN_amazonelectronics_x1_015_8539c013 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.883526 | 0.886028 | 0.430190  |


### Logs
```python
2022-08-15 15:48:22,078 P104103 INFO Params: {
    "attention_dropout": "0.1",
    "attention_hidden_activations": "Dice",
    "attention_hidden_units": "[512, 256]",
    "attention_output_activation": "None",
    "batch_norm": "True",
    "batch_size": "1024",
    "data_format": "csv",
    "data_root": "../data/Amazon/",
    "dataset_id": "amazonelectronics_x1_b7a43f49",
    "debug_mode": "False",
    "din_sequence_field": "('item_history', 'cate_history')",
    "din_target_field": "('item_id', 'cate_id')",
    "din_use_softmax": "True",
    "dnn_activations": "Dice",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.005",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'user_id', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'item_history', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'cate_history', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "[{'feature_encoder': None, 'name': 'item_history'}, {'feature_encoder': None, 'name': 'cate_history'}]",
    "gpu": "6",
    "group_id": "user_id",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DIN",
    "model_id": "DIN_amazonelectronics_x1_015_8539c013",
    "model_root": "./checkpoints/DIN_amazonelectronics_x1/",
    "monitor": "gAUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "20222023",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "train_data": "../data/Amazon/AmazonElectronics_x1/train.csv",
    "valid_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "verbose": "1"
}
2022-08-15 15:48:22,078 P104103 INFO Set up feature processor...
2022-08-15 15:48:22,079 P104103 INFO Load feature_map from json: ../data/Amazon/amazonelectronics_x1_b7a43f49/feature_map.json
2022-08-15 15:48:22,079 P104103 INFO Set column index...
2022-08-15 15:48:22,079 P104103 INFO Feature specs: {
    "cate_history": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803, 'max_len': 100}",
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803}",
    "item_history": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003, 'max_len': 100}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003}",
    "user_id": "{'type': 'meta'}"
}
2022-08-15 15:48:31,552 P104103 INFO Total number of parameters: 5403522.
2022-08-15 15:48:31,552 P104103 INFO Loading data...
2022-08-15 15:48:31,552 P104103 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/train.h5
2022-08-15 15:48:35,859 P104103 INFO Train samples: total/2608764, blocks/1
2022-08-15 15:48:35,859 P104103 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/valid.h5
2022-08-15 15:48:36,485 P104103 INFO Validation samples: total/384806, blocks/1
2022-08-15 15:48:36,485 P104103 INFO Loading train and validation data done.
2022-08-15 15:48:36,485 P104103 INFO Start training: 2548 batches/epoch
2022-08-15 15:48:36,485 P104103 INFO ************ Epoch=1 start ************
2022-08-15 16:05:15,165 P104103 INFO [Metrics] gAUC: 0.839202
2022-08-15 16:05:15,639 P104103 INFO Save best model: monitor(max): 0.839202
2022-08-15 16:05:15,686 P104103 INFO --- 2548/2548 batches finished ---
2022-08-15 16:05:15,788 P104103 INFO Train loss: 0.631589
2022-08-15 16:05:15,788 P104103 INFO ************ Epoch=1 end ************
2022-08-15 16:21:48,435 P104103 INFO [Metrics] gAUC: 0.851083
2022-08-15 16:21:48,644 P104103 INFO Save best model: monitor(max): 0.851083
2022-08-15 16:21:48,718 P104103 INFO --- 2548/2548 batches finished ---
2022-08-15 16:21:48,879 P104103 INFO Train loss: 0.578452
2022-08-15 16:21:48,879 P104103 INFO ************ Epoch=2 end ************
2022-08-15 16:38:27,419 P104103 INFO [Metrics] gAUC: 0.854997
2022-08-15 16:38:27,588 P104103 INFO Save best model: monitor(max): 0.854997
2022-08-15 16:38:27,742 P104103 INFO --- 2548/2548 batches finished ---
2022-08-15 16:38:27,968 P104103 INFO Train loss: 0.565074
2022-08-15 16:38:27,968 P104103 INFO ************ Epoch=3 end ************
2022-08-15 16:55:02,912 P104103 INFO [Metrics] gAUC: 0.859841
2022-08-15 16:55:03,063 P104103 INFO Save best model: monitor(max): 0.859841
2022-08-15 16:55:03,122 P104103 INFO --- 2548/2548 batches finished ---
2022-08-15 16:55:03,295 P104103 INFO Train loss: 0.559689
2022-08-15 16:55:03,295 P104103 INFO ************ Epoch=4 end ************
2022-08-15 17:11:34,836 P104103 INFO [Metrics] gAUC: 0.859191
2022-08-15 17:11:35,019 P104103 INFO Monitor(max) STOP: 0.859191 !
2022-08-15 17:11:35,019 P104103 INFO Reduce learning rate on plateau: 0.000050
2022-08-15 17:11:35,020 P104103 INFO --- 2548/2548 batches finished ---
2022-08-15 17:11:35,156 P104103 INFO Train loss: 0.556145
2022-08-15 17:11:35,156 P104103 INFO ************ Epoch=5 end ************
2022-08-15 17:28:09,940 P104103 INFO [Metrics] gAUC: 0.876431
2022-08-15 17:28:10,124 P104103 INFO Save best model: monitor(max): 0.876431
2022-08-15 17:28:10,210 P104103 INFO --- 2548/2548 batches finished ---
2022-08-15 17:28:10,314 P104103 INFO Train loss: 0.465179
2022-08-15 17:28:10,314 P104103 INFO ************ Epoch=6 end ************
2022-08-15 17:44:41,498 P104103 INFO [Metrics] gAUC: 0.880636
2022-08-15 17:44:41,668 P104103 INFO Save best model: monitor(max): 0.880636
2022-08-15 17:44:41,717 P104103 INFO --- 2548/2548 batches finished ---
2022-08-15 17:44:41,845 P104103 INFO Train loss: 0.425434
2022-08-15 17:44:41,845 P104103 INFO ************ Epoch=7 end ************
2022-08-15 18:01:19,301 P104103 INFO [Metrics] gAUC: 0.883526
2022-08-15 18:01:19,468 P104103 INFO Save best model: monitor(max): 0.883526
2022-08-15 18:01:19,532 P104103 INFO --- 2548/2548 batches finished ---
2022-08-15 18:01:19,633 P104103 INFO Train loss: 0.407147
2022-08-15 18:01:19,633 P104103 INFO ************ Epoch=8 end ************
2022-08-15 18:17:52,924 P104103 INFO [Metrics] gAUC: 0.882351
2022-08-15 18:17:53,087 P104103 INFO Monitor(max) STOP: 0.882351 !
2022-08-15 18:17:53,087 P104103 INFO Reduce learning rate on plateau: 0.000005
2022-08-15 18:17:53,088 P104103 INFO --- 2548/2548 batches finished ---
2022-08-15 18:17:53,264 P104103 INFO Train loss: 0.394253
2022-08-15 18:17:53,264 P104103 INFO ************ Epoch=9 end ************
2022-08-15 18:34:21,332 P104103 INFO [Metrics] gAUC: 0.879347
2022-08-15 18:34:21,525 P104103 INFO Monitor(max) STOP: 0.879347 !
2022-08-15 18:34:21,525 P104103 INFO Reduce learning rate on plateau: 0.000001
2022-08-15 18:34:21,525 P104103 INFO ********* Epoch==10 early stop *********
2022-08-15 18:34:21,526 P104103 INFO --- 2548/2548 batches finished ---
2022-08-15 18:34:21,771 P104103 INFO Train loss: 0.331825
2022-08-15 18:34:21,771 P104103 INFO Training finished.
2022-08-15 18:34:21,771 P104103 INFO Load best model: /cache/FuxiCTRv2.0/benchmark/checkpoints/DIN_amazonelectronics_x1/amazonelectronics_x1_b7a43f49/DIN_amazonelectronics_x1_015_8539c013.model
2022-08-15 18:34:39,487 P104103 INFO ****** Validation evaluation ******
2022-08-15 18:38:10,882 P104103 INFO [Metrics] gAUC: 0.883526 - AUC: 0.886028 - logloss: 0.430190
2022-08-15 18:38:11,160 P104103 INFO ******** Test evaluation ********
2022-08-15 18:38:11,160 P104103 INFO Loading data...
2022-08-15 18:38:11,160 P104103 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/test.h5
2022-08-15 18:38:11,614 P104103 INFO Test samples: total/384806, blocks/1
2022-08-15 18:38:11,614 P104103 INFO Loading test data done.
2022-08-15 18:41:08,640 P104103 INFO [Metrics] gAUC: 0.883526 - AUC: 0.886028 - logloss: 0.430190

```
