## AOANet_amazonelectronics_x1

A hands-on guide to run the AOANet model on the AmazonElectronics_x1 dataset.

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

We use the [AOANet](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/AOANet) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Amazon/AmazonElectronics_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AOANet_amazonelectronics_x1_tuner_config_02](./AOANet_amazonelectronics_x1_tuner_config_02). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/AOANet
    nohup python run_expid.py --config XXX/benchmarks/AOANet/AOANet_amazonelectronics_x1_tuner_config_02 --expid AOANet_amazonelectronics_x1_044_ac142790 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.879087 | 0.881228 | 0.439942  |


### Logs
```python
2022-08-13 22:15:25,163 P80671 INFO Params: {
    "batch_norm": "True",
    "batch_size": "1024",
    "data_format": "csv",
    "data_root": "../data/Amazon/",
    "dataset_id": "amazonelectronics_x1_b7a43f49",
    "debug_mode": "False",
    "dnn_hidden_activations": "ReLU",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.005",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'user_id', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'item_history', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'cate_history', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "None",
    "gpu": "7",
    "group_id": "user_id",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AOANet",
    "model_id": "AOANet_amazonelectronics_x1_044_ac142790",
    "model_root": "./checkpoints/AOANet_amazonelectronics_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_interaction_layers": "2",
    "num_subspaces": "16",
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
2022-08-13 22:15:25,168 P80671 INFO Set up feature processor...
2022-08-13 22:15:25,168 P80671 INFO Load feature_map from json: ../data/Amazon/amazonelectronics_x1_b7a43f49/feature_map.json
2022-08-13 22:15:25,169 P80671 INFO Set column index...
2022-08-13 22:15:25,169 P80671 INFO Feature specs: {
    "cate_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803, 'max_len': 100}",
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803}",
    "item_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003, 'max_len': 100}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003}",
    "user_id": "{'type': 'meta'}"
}
2022-08-13 22:15:30,207 P80671 INFO Total number of parameters: 5142145.
2022-08-13 22:15:30,207 P80671 INFO Loading data...
2022-08-13 22:15:30,208 P80671 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/train.h5
2022-08-13 22:15:34,254 P80671 INFO Train samples: total/2608764, blocks/1
2022-08-13 22:15:34,254 P80671 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/valid.h5
2022-08-13 22:15:34,775 P80671 INFO Validation samples: total/384806, blocks/1
2022-08-13 22:15:34,775 P80671 INFO Loading train and validation data done.
2022-08-13 22:15:34,775 P80671 INFO Start training: 2548 batches/epoch
2022-08-13 22:15:34,775 P80671 INFO ************ Epoch=1 start ************
2022-08-13 22:27:02,973 P80671 INFO [Metrics] AUC: 0.831304 - gAUC: 0.829514
2022-08-13 22:27:03,140 P80671 INFO Save best model: monitor(max): 1.660818
2022-08-13 22:27:03,207 P80671 INFO --- 2548/2548 batches finished ---
2022-08-13 22:27:03,332 P80671 INFO Train loss: 0.635173
2022-08-13 22:27:03,332 P80671 INFO ************ Epoch=1 end ************
2022-08-13 22:38:50,546 P80671 INFO [Metrics] AUC: 0.839919 - gAUC: 0.838027
2022-08-13 22:38:50,712 P80671 INFO Save best model: monitor(max): 1.677947
2022-08-13 22:38:50,764 P80671 INFO --- 2548/2548 batches finished ---
2022-08-13 22:38:50,868 P80671 INFO Train loss: 0.595474
2022-08-13 22:38:50,869 P80671 INFO ************ Epoch=2 end ************
2022-08-13 22:50:59,339 P80671 INFO [Metrics] AUC: 0.845043 - gAUC: 0.843464
2022-08-13 22:50:59,513 P80671 INFO Save best model: monitor(max): 1.688507
2022-08-13 22:50:59,555 P80671 INFO --- 2548/2548 batches finished ---
2022-08-13 22:50:59,651 P80671 INFO Train loss: 0.587923
2022-08-13 22:50:59,651 P80671 INFO ************ Epoch=3 end ************
2022-08-13 23:02:54,602 P80671 INFO [Metrics] AUC: 0.851140 - gAUC: 0.849181
2022-08-13 23:02:54,758 P80671 INFO Save best model: monitor(max): 1.700321
2022-08-13 23:02:54,804 P80671 INFO --- 2548/2548 batches finished ---
2022-08-13 23:02:54,903 P80671 INFO Train loss: 0.581880
2022-08-13 23:02:54,903 P80671 INFO ************ Epoch=4 end ************
2022-08-13 23:13:49,785 P80671 INFO [Metrics] AUC: 0.851653 - gAUC: 0.850247
2022-08-13 23:13:49,950 P80671 INFO Save best model: monitor(max): 1.701900
2022-08-13 23:13:50,006 P80671 INFO --- 2548/2548 batches finished ---
2022-08-13 23:13:50,116 P80671 INFO Train loss: 0.576663
2022-08-13 23:13:50,116 P80671 INFO ************ Epoch=5 end ************
2022-08-13 23:24:42,053 P80671 INFO [Metrics] AUC: 0.852821 - gAUC: 0.850990
2022-08-13 23:24:42,214 P80671 INFO Save best model: monitor(max): 1.703810
2022-08-13 23:24:42,269 P80671 INFO --- 2548/2548 batches finished ---
2022-08-13 23:24:42,413 P80671 INFO Train loss: 0.574354
2022-08-13 23:24:42,413 P80671 INFO ************ Epoch=6 end ************
2022-08-13 23:35:00,569 P80671 INFO [Metrics] AUC: 0.854035 - gAUC: 0.852710
2022-08-13 23:35:00,730 P80671 INFO Save best model: monitor(max): 1.706745
2022-08-13 23:35:00,780 P80671 INFO --- 2548/2548 batches finished ---
2022-08-13 23:35:00,876 P80671 INFO Train loss: 0.572369
2022-08-13 23:35:00,876 P80671 INFO ************ Epoch=7 end ************
2022-08-13 23:43:42,487 P80671 INFO [Metrics] AUC: 0.854414 - gAUC: 0.852492
2022-08-13 23:43:42,642 P80671 INFO Save best model: monitor(max): 1.706905
2022-08-13 23:43:42,698 P80671 INFO --- 2548/2548 batches finished ---
2022-08-13 23:43:42,792 P80671 INFO Train loss: 0.571173
2022-08-13 23:43:42,792 P80671 INFO ************ Epoch=8 end ************
2022-08-13 23:51:42,765 P80671 INFO [Metrics] AUC: 0.855522 - gAUC: 0.853625
2022-08-13 23:51:42,959 P80671 INFO Save best model: monitor(max): 1.709147
2022-08-13 23:51:43,016 P80671 INFO --- 2548/2548 batches finished ---
2022-08-13 23:51:43,119 P80671 INFO Train loss: 0.571248
2022-08-13 23:51:43,119 P80671 INFO ************ Epoch=9 end ************
2022-08-13 23:59:44,116 P80671 INFO [Metrics] AUC: 0.856688 - gAUC: 0.853375
2022-08-13 23:59:44,261 P80671 INFO Save best model: monitor(max): 1.710064
2022-08-13 23:59:44,308 P80671 INFO --- 2548/2548 batches finished ---
2022-08-13 23:59:44,397 P80671 INFO Train loss: 0.570974
2022-08-13 23:59:44,398 P80671 INFO ************ Epoch=10 end ************
2022-08-14 00:07:38,081 P80671 INFO [Metrics] AUC: 0.857127 - gAUC: 0.855101
2022-08-14 00:07:38,226 P80671 INFO Save best model: monitor(max): 1.712228
2022-08-14 00:07:38,271 P80671 INFO --- 2548/2548 batches finished ---
2022-08-14 00:07:38,350 P80671 INFO Train loss: 0.570978
2022-08-14 00:07:38,350 P80671 INFO ************ Epoch=11 end ************
2022-08-14 00:14:42,096 P80671 INFO [Metrics] AUC: 0.856842 - gAUC: 0.854878
2022-08-14 00:14:42,266 P80671 INFO Monitor(max) STOP: 1.711719 !
2022-08-14 00:14:42,266 P80671 INFO Reduce learning rate on plateau: 0.000050
2022-08-14 00:14:42,267 P80671 INFO --- 2548/2548 batches finished ---
2022-08-14 00:14:42,345 P80671 INFO Train loss: 0.570837
2022-08-14 00:14:42,346 P80671 INFO ************ Epoch=12 end ************
2022-08-14 00:20:16,942 P80671 INFO [Metrics] AUC: 0.874791 - gAUC: 0.872330
2022-08-14 00:20:17,110 P80671 INFO Save best model: monitor(max): 1.747122
2022-08-14 00:20:17,150 P80671 INFO --- 2548/2548 batches finished ---
2022-08-14 00:20:17,223 P80671 INFO Train loss: 0.475652
2022-08-14 00:20:17,223 P80671 INFO ************ Epoch=13 end ************
2022-08-14 00:26:00,093 P80671 INFO [Metrics] AUC: 0.879801 - gAUC: 0.877621
2022-08-14 00:26:00,247 P80671 INFO Save best model: monitor(max): 1.757423
2022-08-14 00:26:00,289 P80671 INFO --- 2548/2548 batches finished ---
2022-08-14 00:26:00,362 P80671 INFO Train loss: 0.432857
2022-08-14 00:26:00,362 P80671 INFO ************ Epoch=14 end ************
2022-08-14 00:29:40,524 P80671 INFO [Metrics] AUC: 0.880276 - gAUC: 0.877824
2022-08-14 00:29:40,692 P80671 INFO Save best model: monitor(max): 1.758100
2022-08-14 00:29:40,753 P80671 INFO --- 2548/2548 batches finished ---
2022-08-14 00:29:40,861 P80671 INFO Train loss: 0.416026
2022-08-14 00:29:40,861 P80671 INFO ************ Epoch=15 end ************
2022-08-14 00:33:22,828 P80671 INFO [Metrics] AUC: 0.881228 - gAUC: 0.879087
2022-08-14 00:33:22,960 P80671 INFO Save best model: monitor(max): 1.760315
2022-08-14 00:33:23,005 P80671 INFO --- 2548/2548 batches finished ---
2022-08-14 00:33:23,079 P80671 INFO Train loss: 0.405665
2022-08-14 00:33:23,079 P80671 INFO ************ Epoch=16 end ************
2022-08-14 00:36:30,992 P80671 INFO [Metrics] AUC: 0.880414 - gAUC: 0.878339
2022-08-14 00:36:31,129 P80671 INFO Monitor(max) STOP: 1.758753 !
2022-08-14 00:36:31,130 P80671 INFO Reduce learning rate on plateau: 0.000005
2022-08-14 00:36:31,130 P80671 INFO --- 2548/2548 batches finished ---
2022-08-14 00:36:31,200 P80671 INFO Train loss: 0.398078
2022-08-14 00:36:31,200 P80671 INFO ************ Epoch=17 end ************
2022-08-14 00:39:51,833 P80671 INFO [Metrics] AUC: 0.879063 - gAUC: 0.876837
2022-08-14 00:39:51,967 P80671 INFO Monitor(max) STOP: 1.755900 !
2022-08-14 00:39:51,967 P80671 INFO Reduce learning rate on plateau: 0.000001
2022-08-14 00:39:51,967 P80671 INFO ********* Epoch==18 early stop *********
2022-08-14 00:39:51,967 P80671 INFO --- 2548/2548 batches finished ---
2022-08-14 00:39:52,030 P80671 INFO Train loss: 0.349434
2022-08-14 00:39:52,031 P80671 INFO Training finished.
2022-08-14 00:39:52,031 P80671 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/AOANet_amazonelectronics_x1/amazonelectronics_x1_b7a43f49/AOANet_amazonelectronics_x1_044_ac142790.model
2022-08-14 00:39:52,220 P80671 INFO ****** Validation evaluation ******
2022-08-14 00:40:34,342 P80671 INFO [Metrics] gAUC: 0.879087 - AUC: 0.881228 - logloss: 0.439942
2022-08-14 00:40:34,557 P80671 INFO ******** Test evaluation ********
2022-08-14 00:40:34,557 P80671 INFO Loading data...
2022-08-14 00:40:34,557 P80671 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/test.h5
2022-08-14 00:40:34,995 P80671 INFO Test samples: total/384806, blocks/1
2022-08-14 00:40:34,995 P80671 INFO Loading test data done.
2022-08-14 00:41:17,161 P80671 INFO [Metrics] gAUC: 0.879087 - AUC: 0.881228 - logloss: 0.439942

```
