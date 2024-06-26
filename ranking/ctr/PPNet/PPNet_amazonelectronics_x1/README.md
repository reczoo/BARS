## PPNet_amazonelectronics_x1

A hands-on guide to run the PPNet model on the AmazonElectronics_x1 dataset.

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
Please refer to [AmazonElectronics_x1](https://github.com/reczoo/Datasets/tree/main/Amazon/AmazonElectronics_x1) to get the dataset details.

### Code

We use the [PPNet](https://github.com/reczoo/FuxiCTR/tree/v2.0.3/model_zoo/PPNet) model code from [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/tree/v2.0.3) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.3.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.3
    ```

2. Create a data directory and put the downloaded data files in `../data/Amazon/AmazonElectronics_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [PPNet_amazonelectronics_x1_tuner_config_03](./PPNet_amazonelectronics_x1_tuner_config_03). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/PPNet
    nohup python run_expid.py --config YOUR_PATH/PPNet/PPNet_amazonelectronics_x1_tuner_config_03 --expid PPNet_amazonelectronics_x1_023_0b353410 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.879690 | 0.881667 | 0.439829  |


### Logs
```python
2023-06-01 15:07:09,673 P45816 INFO Params: {
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
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'user_id', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'item_history', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'cate_history', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}]",
    "feature_config": "None",
    "feature_specs": "None",
    "gate_emb_dim": "32",
    "gate_hidden_dim": "64",
    "gate_priors": "['item_history']",
    "gpu": "5",
    "group_id": "user_id",
    "hidden_activations": "ReLU",
    "hidden_units": "[1024, 512, 256]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "PPNet",
    "model_id": "PPNet_amazonelectronics_x1_023_0b353410",
    "model_root": "./checkpoints/PPNet_amazonelectronics_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "20222023",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "train_data": "../data/Amazon/AmazonElectronics_x1/train.csv",
    "use_features": "None",
    "valid_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "verbose": "1"
}
2023-06-01 15:07:09,673 P45816 INFO Set up feature processor...
2023-06-01 15:07:09,674 P45816 WARNING Skip rebuilding ../data/Amazon/amazonelectronics_x1_b7a43f49/feature_map.json. Please delete it manually if rebuilding is required.
2023-06-01 15:07:09,674 P45816 INFO Load feature_map from json: ../data/Amazon/amazonelectronics_x1_b7a43f49/feature_map.json
2023-06-01 15:07:09,674 P45816 INFO Set column index...
2023-06-01 15:07:09,674 P45816 INFO Feature specs: {
    "cate_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803, 'max_len': 100}",
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803}",
    "item_history": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003, 'max_len': 100}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003}",
    "user_id": "{'type': 'meta'}"
}
2023-06-01 15:07:15,859 P45816 INFO Total number of parameters: 7194785.
2023-06-01 15:07:15,859 P45816 INFO Loading data...
2023-06-01 15:07:15,859 P45816 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/train.h5
2023-06-01 15:07:19,785 P45816 INFO Train samples: total/2608764, blocks/1
2023-06-01 15:07:19,785 P45816 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/valid.h5
2023-06-01 15:07:20,277 P45816 INFO Validation samples: total/384806, blocks/1
2023-06-01 15:07:20,277 P45816 INFO Loading train and validation data done.
2023-06-01 15:07:20,277 P45816 INFO Start training: 2548 batches/epoch
2023-06-01 15:07:20,277 P45816 INFO ************ Epoch=1 start ************
2023-06-01 15:15:38,685 P45816 INFO Train loss: 0.638819
2023-06-01 15:15:38,685 P45816 INFO Evaluation @epoch 1 - batch 2548: 
2023-06-01 15:18:04,118 P45816 INFO [Metrics] AUC: 0.835712 - gAUC: 0.834478
2023-06-01 15:18:04,119 P45816 INFO Save best model: monitor(max)=1.670190
2023-06-01 15:18:04,300 P45816 INFO ************ Epoch=1 end ************
2023-06-01 15:25:58,548 P45816 INFO Train loss: 0.591881
2023-06-01 15:25:58,558 P45816 INFO Evaluation @epoch 2 - batch 2548: 
2023-06-01 15:28:17,785 P45816 INFO [Metrics] AUC: 0.846837 - gAUC: 0.845397
2023-06-01 15:28:17,786 P45816 INFO Save best model: monitor(max)=1.692235
2023-06-01 15:28:17,966 P45816 INFO ************ Epoch=2 end ************
2023-06-01 15:34:54,539 P45816 INFO Train loss: 0.576393
2023-06-01 15:34:54,540 P45816 INFO Evaluation @epoch 3 - batch 2548: 
2023-06-01 15:36:54,513 P45816 INFO [Metrics] AUC: 0.852579 - gAUC: 0.849971
2023-06-01 15:36:54,514 P45816 INFO Save best model: monitor(max)=1.702550
2023-06-01 15:36:54,682 P45816 INFO ************ Epoch=3 end ************
2023-06-01 15:42:27,432 P45816 INFO Train loss: 0.569779
2023-06-01 15:42:27,433 P45816 INFO Evaluation @epoch 4 - batch 2548: 
2023-06-01 15:44:06,789 P45816 INFO [Metrics] AUC: 0.854074 - gAUC: 0.852149
2023-06-01 15:44:06,791 P45816 INFO Save best model: monitor(max)=1.706223
2023-06-01 15:44:06,942 P45816 INFO ************ Epoch=4 end ************
2023-06-01 15:49:17,545 P45816 INFO Train loss: 0.566076
2023-06-01 15:49:17,546 P45816 INFO Evaluation @epoch 5 - batch 2548: 
2023-06-01 15:50:53,414 P45816 INFO [Metrics] AUC: 0.854454 - gAUC: 0.852533
2023-06-01 15:50:53,415 P45816 INFO Save best model: monitor(max)=1.706988
2023-06-01 15:50:53,576 P45816 INFO ************ Epoch=5 end ************
2023-06-01 15:55:36,155 P45816 INFO Train loss: 0.564237
2023-06-01 15:55:36,155 P45816 INFO Evaluation @epoch 6 - batch 2548: 
2023-06-01 15:57:05,092 P45816 INFO [Metrics] AUC: 0.855354 - gAUC: 0.854503
2023-06-01 15:57:05,093 P45816 INFO Save best model: monitor(max)=1.709857
2023-06-01 15:57:05,240 P45816 INFO ************ Epoch=6 end ************
2023-06-01 16:01:48,854 P45816 INFO Train loss: 0.564162
2023-06-01 16:01:48,859 P45816 INFO Evaluation @epoch 7 - batch 2548: 
2023-06-01 16:03:16,675 P45816 INFO [Metrics] AUC: 0.855684 - gAUC: 0.854337
2023-06-01 16:03:16,676 P45816 INFO Save best model: monitor(max)=1.710021
2023-06-01 16:03:16,840 P45816 INFO ************ Epoch=7 end ************
2023-06-01 16:07:57,635 P45816 INFO Train loss: 0.564687
2023-06-01 16:07:57,635 P45816 INFO Evaluation @epoch 8 - batch 2548: 
2023-06-01 16:09:24,289 P45816 INFO [Metrics] AUC: 0.856126 - gAUC: 0.854072
2023-06-01 16:09:24,290 P45816 INFO Save best model: monitor(max)=1.710198
2023-06-01 16:09:24,454 P45816 INFO ************ Epoch=8 end ************
2023-06-01 16:13:41,197 P45816 INFO Train loss: 0.568022
2023-06-01 16:13:41,198 P45816 INFO Evaluation @epoch 9 - batch 2548: 
2023-06-01 16:15:04,923 P45816 INFO [Metrics] AUC: 0.859547 - gAUC: 0.858126
2023-06-01 16:15:04,925 P45816 INFO Save best model: monitor(max)=1.717673
2023-06-01 16:15:05,073 P45816 INFO ************ Epoch=9 end ************
2023-06-01 16:19:14,946 P45816 INFO Train loss: 0.573383
2023-06-01 16:19:14,947 P45816 INFO Evaluation @epoch 10 - batch 2548: 
2023-06-01 16:20:40,016 P45816 INFO [Metrics] AUC: 0.857831 - gAUC: 0.855096
2023-06-01 16:20:40,017 P45816 INFO Monitor(max)=1.712927 STOP!
2023-06-01 16:20:40,017 P45816 INFO Reduce learning rate on plateau: 0.000050
2023-06-01 16:20:40,106 P45816 INFO ************ Epoch=10 end ************
2023-06-01 16:24:44,738 P45816 INFO Train loss: 0.470273
2023-06-01 16:24:44,738 P45816 INFO Evaluation @epoch 11 - batch 2548: 
2023-06-01 16:26:08,739 P45816 INFO [Metrics] AUC: 0.877104 - gAUC: 0.874695
2023-06-01 16:26:08,740 P45816 INFO Save best model: monitor(max)=1.751799
2023-06-01 16:26:08,898 P45816 INFO ************ Epoch=11 end ************
2023-06-01 16:30:03,547 P45816 INFO Train loss: 0.421339
2023-06-01 16:30:03,547 P45816 INFO Evaluation @epoch 12 - batch 2548: 
2023-06-01 16:31:18,982 P45816 INFO [Metrics] AUC: 0.881220 - gAUC: 0.879643
2023-06-01 16:31:18,983 P45816 INFO Save best model: monitor(max)=1.760863
2023-06-01 16:31:19,131 P45816 INFO ************ Epoch=12 end ************
2023-06-01 16:33:48,819 P45816 INFO Train loss: 0.400627
2023-06-01 16:33:48,820 P45816 INFO Evaluation @epoch 13 - batch 2548: 
2023-06-01 16:34:34,062 P45816 INFO [Metrics] AUC: 0.881667 - gAUC: 0.879690
2023-06-01 16:34:34,063 P45816 INFO Save best model: monitor(max)=1.761357
2023-06-01 16:34:34,212 P45816 INFO ************ Epoch=13 end ************
2023-06-01 16:35:31,254 P45816 INFO Train loss: 0.387298
2023-06-01 16:35:31,254 P45816 INFO Evaluation @epoch 14 - batch 2548: 
2023-06-01 16:36:09,935 P45816 INFO [Metrics] AUC: 0.880315 - gAUC: 0.878120
2023-06-01 16:36:09,936 P45816 INFO Monitor(max)=1.758436 STOP!
2023-06-01 16:36:09,936 P45816 INFO Reduce learning rate on plateau: 0.000005
2023-06-01 16:36:10,017 P45816 INFO ************ Epoch=14 end ************
2023-06-01 16:36:48,874 P45816 INFO Train loss: 0.332448
2023-06-01 16:36:48,875 P45816 INFO Evaluation @epoch 15 - batch 2548: 
2023-06-01 16:37:26,805 P45816 INFO [Metrics] AUC: 0.877621 - gAUC: 0.875730
2023-06-01 16:37:26,806 P45816 INFO Monitor(max)=1.753351 STOP!
2023-06-01 16:37:26,807 P45816 INFO Reduce learning rate on plateau: 0.000001
2023-06-01 16:37:26,807 P45816 INFO ********* Epoch==15 early stop *********
2023-06-01 16:37:26,887 P45816 INFO Training finished.
2023-06-01 16:37:26,887 P45816 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/PPNet_amazonelectronics_x1/amazonelectronics_x1_b7a43f49/PPNet_amazonelectronics_x1_023_0b353410.model
2023-06-01 16:37:26,920 P45816 INFO ****** Validation evaluation ******
2023-06-01 16:38:04,651 P45816 INFO [Metrics] gAUC: 0.879690 - AUC: 0.881667 - logloss: 0.439829
2023-06-01 16:38:04,730 P45816 INFO ******** Test evaluation ********
2023-06-01 16:38:04,730 P45816 INFO Loading data...
2023-06-01 16:38:04,731 P45816 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_b7a43f49/test.h5
2023-06-01 16:38:05,112 P45816 INFO Test samples: total/384806, blocks/1
2023-06-01 16:38:05,112 P45816 INFO Loading test data done.
2023-06-01 16:38:41,506 P45816 INFO [Metrics] gAUC: 0.879690 - AUC: 0.881667 - logloss: 0.439829

```
