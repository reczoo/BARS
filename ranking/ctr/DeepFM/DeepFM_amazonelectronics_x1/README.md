## DeepFM_amazonelectronics_x1

A hands-on guide to run the DeepFM model on the AmazonElectronics_x1 dataset.

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

We use the [DeepFM](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/DeepFM) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Amazon/AmazonElectronics_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DeepFM_amazonelectronics_x1_tuner_config_02](./DeepFM_amazonelectronics_x1_tuner_config_02). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DeepFM
    nohup python run_expid.py --config XXX/benchmarks/DeepFM/DeepFM_amazonelectronics_x1_tuner_config_02 --expid DeDeepFM_amazonelectronics_x1_014_68eadb7e --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.878890 | 0.881583 | 0.437653  |


### Logs
```python
2022-08-15 23:07:23,636 P53652 INFO Params: {
    "batch_norm": "True",
    "batch_size": "1024",
    "data_format": "csv",
    "data_root": "../data/Amazon/",
    "dataset_id": "amazonelectronics_x1_b7a43f49",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.005",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'user_id', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'item_history', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'cate_history', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "None",
    "gpu": "5",
    "group_id": "user_id",
    "hidden_activations": "relu",
    "hidden_units": "[1024, 512, 256]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DeepFM",
    "model_id": "DeepFM_amazonelectronics_x1_014_68eadb7e",
    "model_root": "./checkpoints/DeepFM_amazonelectronics_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
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
2022-08-15 23:07:23,636 P53652 INFO Set up feature processor...
2022-08-15 23:07:23,636 P53652 INFO Load feature_map from json: ../data/Amazon/amazonelectronics_x1_b7a43f49/feature_map.json
2022-08-15 23:07:23,637 P53652 INFO Set column index...
2022-08-15 23:07:23,637 P53652 INFO Feature specs: {
    "cate_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803, 'max_len': 100}",
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803}",
    "item_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003, 'max_len': 100}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003}",
    "user_id": "{'type': 'meta'}"
}
2022-08-15 23:07:30,820 P53652 INFO Total number of parameters: 5134334.
2022-08-15 23:07:30,821 P53652 INFO Loading data...
2022-08-15 23:07:30,821 P53652 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/train.h5
2022-08-15 23:07:34,213 P53652 INFO Train samples: total/2608764, blocks/1
2022-08-15 23:07:34,213 P53652 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/valid.h5
2022-08-15 23:07:34,764 P53652 INFO Validation samples: total/384806, blocks/1
2022-08-15 23:07:34,764 P53652 INFO Loading train and validation data done.
2022-08-15 23:07:34,765 P53652 INFO Start training: 2548 batches/epoch
2022-08-15 23:07:34,765 P53652 INFO ************ Epoch=1 start ************
2022-08-15 23:18:49,801 P53652 INFO [Metrics] AUC: 0.835017 - gAUC: 0.833464
2022-08-15 23:18:49,973 P53652 INFO Save best model: monitor(max): 1.668481
2022-08-15 23:18:50,027 P53652 INFO --- 2548/2548 batches finished ---
2022-08-15 23:18:50,110 P53652 INFO Train loss: 0.647462
2022-08-15 23:18:50,111 P53652 INFO ************ Epoch=1 end ************
2022-08-15 23:29:58,664 P53652 INFO [Metrics] AUC: 0.847913 - gAUC: 0.846193
2022-08-15 23:29:58,931 P53652 INFO Save best model: monitor(max): 1.694105
2022-08-15 23:29:59,014 P53652 INFO --- 2548/2548 batches finished ---
2022-08-15 23:29:59,204 P53652 INFO Train loss: 0.605117
2022-08-15 23:29:59,205 P53652 INFO ************ Epoch=2 end ************
2022-08-15 23:41:04,966 P53652 INFO [Metrics] AUC: 0.851169 - gAUC: 0.849207
2022-08-15 23:41:05,269 P53652 INFO Save best model: monitor(max): 1.700376
2022-08-15 23:41:05,364 P53652 INFO --- 2548/2548 batches finished ---
2022-08-15 23:41:05,510 P53652 INFO Train loss: 0.583682
2022-08-15 23:41:05,510 P53652 INFO ************ Epoch=3 end ************
2022-08-15 23:52:13,796 P53652 INFO [Metrics] AUC: 0.852135 - gAUC: 0.850470
2022-08-15 23:52:14,019 P53652 INFO Save best model: monitor(max): 1.702605
2022-08-15 23:52:14,093 P53652 INFO --- 2548/2548 batches finished ---
2022-08-15 23:52:14,202 P53652 INFO Train loss: 0.574556
2022-08-15 23:52:14,202 P53652 INFO ************ Epoch=4 end ************
2022-08-16 00:03:15,958 P53652 INFO [Metrics] AUC: 0.855899 - gAUC: 0.853053
2022-08-16 00:03:16,170 P53652 INFO Save best model: monitor(max): 1.708952
2022-08-16 00:03:16,237 P53652 INFO --- 2548/2548 batches finished ---
2022-08-16 00:03:16,393 P53652 INFO Train loss: 0.571438
2022-08-16 00:03:16,393 P53652 INFO ************ Epoch=5 end ************
2022-08-16 00:14:11,877 P53652 INFO [Metrics] AUC: 0.855979 - gAUC: 0.853661
2022-08-16 00:14:12,150 P53652 INFO Save best model: monitor(max): 1.709640
2022-08-16 00:14:12,231 P53652 INFO --- 2548/2548 batches finished ---
2022-08-16 00:14:12,360 P53652 INFO Train loss: 0.568936
2022-08-16 00:14:12,360 P53652 INFO ************ Epoch=6 end ************
2022-08-16 00:25:11,929 P53652 INFO [Metrics] AUC: 0.856358 - gAUC: 0.854082
2022-08-16 00:25:12,209 P53652 INFO Save best model: monitor(max): 1.710440
2022-08-16 00:25:12,316 P53652 INFO --- 2548/2548 batches finished ---
2022-08-16 00:25:12,483 P53652 INFO Train loss: 0.568628
2022-08-16 00:25:12,483 P53652 INFO ************ Epoch=7 end ************
2022-08-16 00:36:10,695 P53652 INFO [Metrics] AUC: 0.856713 - gAUC: 0.854280
2022-08-16 00:36:10,936 P53652 INFO Save best model: monitor(max): 1.710993
2022-08-16 00:36:11,031 P53652 INFO --- 2548/2548 batches finished ---
2022-08-16 00:36:11,201 P53652 INFO Train loss: 0.567477
2022-08-16 00:36:11,201 P53652 INFO ************ Epoch=8 end ************
2022-08-16 00:46:09,498 P53652 INFO [Metrics] AUC: 0.858522 - gAUC: 0.857019
2022-08-16 00:46:09,724 P53652 INFO Save best model: monitor(max): 1.715541
2022-08-16 00:46:09,788 P53652 INFO --- 2548/2548 batches finished ---
2022-08-16 00:46:09,935 P53652 INFO Train loss: 0.567370
2022-08-16 00:46:09,936 P53652 INFO ************ Epoch=9 end ************
2022-08-16 00:55:30,817 P53652 INFO [Metrics] AUC: 0.857687 - gAUC: 0.855319
2022-08-16 00:55:31,059 P53652 INFO Monitor(max) STOP: 1.713007 !
2022-08-16 00:55:31,060 P53652 INFO Reduce learning rate on plateau: 0.000050
2022-08-16 00:55:31,060 P53652 INFO --- 2548/2548 batches finished ---
2022-08-16 00:55:31,177 P53652 INFO Train loss: 0.567132
2022-08-16 00:55:31,177 P53652 INFO ************ Epoch=10 end ************
2022-08-16 01:03:29,031 P53652 INFO [Metrics] AUC: 0.875430 - gAUC: 0.873115
2022-08-16 01:03:29,225 P53652 INFO Save best model: monitor(max): 1.748545
2022-08-16 01:03:29,294 P53652 INFO --- 2548/2548 batches finished ---
2022-08-16 01:03:29,403 P53652 INFO Train loss: 0.470898
2022-08-16 01:03:29,404 P53652 INFO ************ Epoch=11 end ************
2022-08-16 01:10:23,185 P53652 INFO [Metrics] AUC: 0.879779 - gAUC: 0.877174
2022-08-16 01:10:23,380 P53652 INFO Save best model: monitor(max): 1.756953
2022-08-16 01:10:23,440 P53652 INFO --- 2548/2548 batches finished ---
2022-08-16 01:10:23,627 P53652 INFO Train loss: 0.427326
2022-08-16 01:10:23,627 P53652 INFO ************ Epoch=12 end ************
2022-08-16 01:16:08,327 P53652 INFO [Metrics] AUC: 0.881583 - gAUC: 0.878890
2022-08-16 01:16:08,490 P53652 INFO Save best model: monitor(max): 1.760473
2022-08-16 01:16:08,541 P53652 INFO --- 2548/2548 batches finished ---
2022-08-16 01:16:08,641 P53652 INFO Train loss: 0.408635
2022-08-16 01:16:08,641 P53652 INFO ************ Epoch=13 end ************
2022-08-16 01:19:11,605 P53652 INFO [Metrics] AUC: 0.881249 - gAUC: 0.878635
2022-08-16 01:19:11,744 P53652 INFO Monitor(max) STOP: 1.759883 !
2022-08-16 01:19:11,745 P53652 INFO Reduce learning rate on plateau: 0.000005
2022-08-16 01:19:11,745 P53652 INFO --- 2548/2548 batches finished ---
2022-08-16 01:19:11,812 P53652 INFO Train loss: 0.396797
2022-08-16 01:19:11,812 P53652 INFO ************ Epoch=14 end ************
2022-08-16 01:20:29,576 P53652 INFO [Metrics] AUC: 0.879285 - gAUC: 0.877086
2022-08-16 01:20:29,721 P53652 INFO Monitor(max) STOP: 1.756371 !
2022-08-16 01:20:29,721 P53652 INFO Reduce learning rate on plateau: 0.000001
2022-08-16 01:20:29,721 P53652 INFO ********* Epoch==15 early stop *********
2022-08-16 01:20:29,722 P53652 INFO --- 2548/2548 batches finished ---
2022-08-16 01:20:29,785 P53652 INFO Train loss: 0.346555
2022-08-16 01:20:29,786 P53652 INFO Training finished.
2022-08-16 01:20:29,786 P53652 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DeepFM_amazonelectronics_x1/amazonelectronics_x1_b7a43f49/DeepFM_amazonelectronics_x1_014_68eadb7e.model
2022-08-16 01:20:29,975 P53652 INFO ****** Validation evaluation ******
2022-08-16 01:21:08,295 P53652 INFO [Metrics] gAUC: 0.878890 - AUC: 0.881583 - logloss: 0.437653
2022-08-16 01:21:08,568 P53652 INFO ******** Test evaluation ********
2022-08-16 01:21:08,569 P53652 INFO Loading data...
2022-08-16 01:21:08,569 P53652 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/test.h5
2022-08-16 01:21:09,012 P53652 INFO Test samples: total/384806, blocks/1
2022-08-16 01:21:09,012 P53652 INFO Loading test data done.
2022-08-16 01:21:46,408 P53652 INFO [Metrics] gAUC: 0.878890 - AUC: 0.881583 - logloss: 0.437653

```
