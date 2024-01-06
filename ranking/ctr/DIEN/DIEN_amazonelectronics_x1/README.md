## DIEN_amazonelectronics_x1

A hands-on guide to run the DIEN model on the AmazonElectronics_x1 dataset.

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
  fuxictr: 2.0.2

  ```

### Dataset
Please refer to the BARS dataset [AmazonElectronics_x1](https://github.com/openbenchmark/BARS/blob/main/datasets/Amazon#AmazonElectronics_x1) to get data ready.

### Code

We use the [DIEN](https://github.com/xue-pai/FuxiCTR/blob/v2.0.2/model_zoo/DIEN) model code from [FuxiCTR-v2.0.2](https://github.com/xue-pai/FuxiCTR/tree/v2.0.2) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v2.0.2.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.2
    ```

2. Create a data directory and put the downloaded data files in `../data/Amazon/AmazonElectronics_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DIEN_amazonelectronics_x1_tuner_config_03](./DIEN_amazonelectronics_x1_tuner_config_03). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DIEN
    nohup python run_expid.py --config XXX/benchmarks/DIEN/DIEN_amazonelectronics_x1_tuner_config_03 --expid DIEN_amazonelectronics_x1_022_a22ee885 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.885625 | 0.888777 | 0.425708  |


### Logs
```python
2023-05-13 06:07:53,283 P48775 INFO Params: {
    "attention_activation": "Dice",
    "attention_dropout": "0",
    "attention_hidden_units": "[256, 128]",
    "attention_type": "din_attention",
    "aux_activation": "ReLU",
    "aux_hidden_units": "[100, 50]",
    "aux_loss_alpha": "0",
    "batch_norm": "True",
    "batch_size": "1024",
    "data_format": "csv",
    "data_root": "../data/Amazon/",
    "dataset_id": "amazonelectronics_x1_51836f99",
    "debug_mode": "False",
    "dien_neg_seq_field": "[]",
    "dien_sequence_field": "('item_history', 'cate_history')",
    "dien_target_field": "('item_id', 'cate_id')",
    "dnn_activations": "ReLU",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.005",
    "enable_sum_pooling": "False",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'user_id', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'item_history', 'padding': 'post', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'feature_encoder': 'layers.MaskedAveragePooling()', 'max_len': 100, 'name': 'cate_history', 'padding': 'post', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}]",
    "feature_config": "None",
    "feature_specs": "[{'feature_encoder': None, 'name': 'item_history'}, {'feature_encoder': None, 'name': 'cate_history'}]",
    "gpu": "0",
    "group_id": "user_id",
    "gru_type": "AUGRU",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DIEN",
    "model_id": "DIEN_amazonelectronics_x1_022_a22ee885",
    "model_root": "./checkpoints/DIEN_amazonelectronics_x1/",
    "monitor": "gAUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
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
    "use_attention_softmax": "True",
    "use_features": "None",
    "valid_data": "../data/Amazon/AmazonElectronics_x1/test.csv",
    "verbose": "1"
}
2023-05-13 06:07:53,284 P48775 INFO Set up feature processor...
2023-05-13 06:07:53,284 P48775 WARNING Skip rebuilding ../data/Amazon/amazonelectronics_x1_51836f99/feature_map.json. Please delete it manually if rebuilding is required.
2023-05-13 06:07:53,284 P48775 INFO Load feature_map from json: ../data/Amazon/amazonelectronics_x1_51836f99/feature_map.json
2023-05-13 06:07:53,284 P48775 INFO Set column index...
2023-05-13 06:07:53,284 P48775 INFO Feature specs: {
    "cate_history": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803, 'max_len': 100}",
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 802, 'vocab_size': 803}",
    "item_history": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003, 'max_len': 100}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 63002, 'vocab_size': 63003}",
    "user_id": "{'type': 'meta'}"
}
2023-05-13 06:07:57,125 P48775 INFO Total number of parameters: 5369602.
2023-05-13 06:07:57,125 P48775 INFO Loading data...
2023-05-13 06:07:57,125 P48775 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_51836f99/train.h5
2023-05-13 06:08:00,053 P48775 INFO Train samples: total/2608764, blocks/1
2023-05-13 06:08:00,053 P48775 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_51836f99/valid.h5
2023-05-13 06:08:00,481 P48775 INFO Validation samples: total/384806, blocks/1
2023-05-13 06:08:00,481 P48775 INFO Loading train and validation data done.
2023-05-13 06:08:00,482 P48775 INFO Start training: 2548 batches/epoch
2023-05-13 06:08:00,482 P48775 INFO ************ Epoch=1 start ************
2023-05-13 06:14:23,817 P48775 INFO Train loss: 0.645549
2023-05-13 06:14:23,818 P48775 INFO Evaluation @epoch 1 - batch 2548: 
2023-05-13 06:15:15,245 P48775 INFO [Metrics] gAUC: 0.830730
2023-05-13 06:15:15,246 P48775 INFO Save best model: monitor(max)=0.830730
2023-05-13 06:15:15,344 P48775 INFO ************ Epoch=1 end ************
2023-05-13 06:22:17,133 P48775 INFO Train loss: 0.583323
2023-05-13 06:22:17,134 P48775 INFO Evaluation @epoch 2 - batch 2548: 
2023-05-13 06:23:08,922 P48775 INFO [Metrics] gAUC: 0.851432
2023-05-13 06:23:08,923 P48775 INFO Save best model: monitor(max)=0.851432
2023-05-13 06:23:09,053 P48775 INFO ************ Epoch=2 end ************
2023-05-13 06:29:33,348 P48775 INFO Train loss: 0.562884
2023-05-13 06:29:33,349 P48775 INFO Evaluation @epoch 3 - batch 2548: 
2023-05-13 06:30:27,846 P48775 INFO [Metrics] gAUC: 0.854945
2023-05-13 06:30:27,848 P48775 INFO Save best model: monitor(max)=0.854945
2023-05-13 06:30:27,963 P48775 INFO ************ Epoch=3 end ************
2023-05-13 06:36:54,015 P48775 INFO Train loss: 0.556788
2023-05-13 06:36:54,016 P48775 INFO Evaluation @epoch 4 - batch 2548: 
2023-05-13 06:37:47,720 P48775 INFO [Metrics] gAUC: 0.857336
2023-05-13 06:37:47,721 P48775 INFO Save best model: monitor(max)=0.857336
2023-05-13 06:37:47,830 P48775 INFO ************ Epoch=4 end ************
2023-05-13 06:44:16,553 P48775 INFO Train loss: 0.552307
2023-05-13 06:44:16,554 P48775 INFO Evaluation @epoch 5 - batch 2548: 
2023-05-13 06:45:08,426 P48775 INFO [Metrics] gAUC: 0.860127
2023-05-13 06:45:08,427 P48775 INFO Save best model: monitor(max)=0.860127
2023-05-13 06:45:08,554 P48775 INFO ************ Epoch=5 end ************
2023-05-13 06:51:58,084 P48775 INFO Train loss: 0.550696
2023-05-13 06:51:58,085 P48775 INFO Evaluation @epoch 6 - batch 2548: 
2023-05-13 06:52:50,519 P48775 INFO [Metrics] gAUC: 0.862570
2023-05-13 06:52:50,519 P48775 INFO Save best model: monitor(max)=0.862570
2023-05-13 06:52:50,644 P48775 INFO ************ Epoch=6 end ************
2023-05-13 06:59:31,113 P48775 INFO Train loss: 0.549682
2023-05-13 06:59:31,113 P48775 INFO Evaluation @epoch 7 - batch 2548: 
2023-05-13 07:00:22,470 P48775 INFO [Metrics] gAUC: 0.861166
2023-05-13 07:00:22,471 P48775 INFO Monitor(max)=0.861166 STOP!
2023-05-13 07:00:22,471 P48775 INFO Reduce learning rate on plateau: 0.000050
2023-05-13 07:00:22,548 P48775 INFO ************ Epoch=7 end ************
2023-05-13 07:07:02,853 P48775 INFO Train loss: 0.459559
2023-05-13 07:07:02,854 P48775 INFO Evaluation @epoch 8 - batch 2548: 
2023-05-13 07:07:55,533 P48775 INFO [Metrics] gAUC: 0.879441
2023-05-13 07:07:55,534 P48775 INFO Save best model: monitor(max)=0.879441
2023-05-13 07:07:55,658 P48775 INFO ************ Epoch=8 end ************
2023-05-13 07:14:40,089 P48775 INFO Train loss: 0.417507
2023-05-13 07:14:40,090 P48775 INFO Evaluation @epoch 9 - batch 2548: 
2023-05-13 07:15:32,108 P48775 INFO [Metrics] gAUC: 0.884196
2023-05-13 07:15:32,109 P48775 INFO Save best model: monitor(max)=0.884196
2023-05-13 07:15:32,233 P48775 INFO ************ Epoch=9 end ************
2023-05-13 07:22:18,419 P48775 INFO Train loss: 0.398640
2023-05-13 07:22:18,420 P48775 INFO Evaluation @epoch 10 - batch 2548: 
2023-05-13 07:23:10,705 P48775 INFO [Metrics] gAUC: 0.885625
2023-05-13 07:23:10,706 P48775 INFO Save best model: monitor(max)=0.885625
2023-05-13 07:23:10,818 P48775 INFO ************ Epoch=10 end ************
2023-05-13 07:29:33,487 P48775 INFO Train loss: 0.386624
2023-05-13 07:29:33,487 P48775 INFO Evaluation @epoch 11 - batch 2548: 
2023-05-13 07:30:25,366 P48775 INFO [Metrics] gAUC: 0.885199
2023-05-13 07:30:25,367 P48775 INFO Monitor(max)=0.885199 STOP!
2023-05-13 07:30:25,367 P48775 INFO Reduce learning rate on plateau: 0.000005
2023-05-13 07:30:25,448 P48775 INFO ************ Epoch=11 end ************
2023-05-13 07:36:55,796 P48775 INFO Train loss: 0.335177
2023-05-13 07:36:55,797 P48775 INFO Evaluation @epoch 12 - batch 2548: 
2023-05-13 07:37:47,094 P48775 INFO [Metrics] gAUC: 0.881743
2023-05-13 07:37:47,095 P48775 INFO Monitor(max)=0.881743 STOP!
2023-05-13 07:37:47,095 P48775 INFO Reduce learning rate on plateau: 0.000001
2023-05-13 07:37:47,095 P48775 INFO ********* Epoch==12 early stop *********
2023-05-13 07:37:47,164 P48775 INFO Training finished.
2023-05-13 07:37:47,165 P48775 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DIEN_amazonelectronics_x1/amazonelectronics_x1_51836f99/DIEN_amazonelectronics_x1_022_a22ee885.model
2023-05-13 07:37:47,198 P48775 INFO ****** Validation evaluation ******
2023-05-13 07:38:39,764 P48775 INFO [Metrics] gAUC: 0.885625 - AUC: 0.888777 - logloss: 0.425708
2023-05-13 07:38:39,850 P48775 INFO ******** Test evaluation ********
2023-05-13 07:38:39,850 P48775 INFO Loading data...
2023-05-13 07:38:39,850 P48775 INFO Loading data from h5: ../data/Amazon/amazonelectronics_x1_51836f99/test.h5
2023-05-13 07:38:40,292 P48775 INFO Test samples: total/384806, blocks/1
2023-05-13 07:38:40,292 P48775 INFO Loading test data done.
2023-05-13 07:39:32,056 P48775 INFO [Metrics] gAUC: 0.885625 - AUC: 0.888777 - logloss: 0.425708

```
