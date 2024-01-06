## AOANet_kuaivideo_x1

A hands-on guide to run the AOANet model on the KuaiVideo_x1 dataset.

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
Please refer to the BARS dataset [KuaiVideo_x1](https://github.com/openbenchmark/BARS/blob/main/datasets/KuaiShou#KuaiVideo_x1) to get data ready.

### Code

We use the [AOANet](https://github.com/xue-pai/FuxiCTR/blob/v2.0.1/model_zoo/AOANet) model code from [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/KuaiShou/KuaiVideo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AOANet_kuaivideo_x1_tuner_config_03](./AOANet_kuaivideo_x1_tuner_config_03). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/AOANet
    nohup python run_expid.py --config XXX/benchmarks/AOANet/AOANet_kuaivideo_x1_tuner_config_03 --expid AOANet_kuaivideo_x1_006_71fb5d8c --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.667875 | 0.747020 | 0.438034  |


### Logs
```python
2022-08-24 20:52:20,944 P23932 INFO Params: {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/KuaiShou/",
    "dataset_id": "kuaivideo_x1_dc7a3035",
    "debug_mode": "False",
    "dnn_hidden_activations": "ReLU",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.0001",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'min_categr_count': 1, 'name': 'item_emb', 'preprocess': 'copy_from(item_id)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'pos_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'neg_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'pos_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(pos_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'neg_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(neg_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}]",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_emb'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'pos_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'neg_items'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'pos_items_emb'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'neg_items_emb'}]",
    "gpu": "5",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "AOANet",
    "model_id": "AOANet_kuaivideo_x1_006_71fb5d8c",
    "model_root": "./checkpoints/AOANet_kuaivideo_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_interaction_layers": "1",
    "num_subspaces": "1",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "20222023",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "train_data": "../data/KuaiShou/KuaiVideo_x1/train.csv",
    "valid_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "verbose": "1"
}
2022-08-24 20:52:20,945 P23932 INFO Set up feature processor...
2022-08-24 20:52:20,945 P23932 INFO Load feature_map from json: ../data/KuaiShou/kuaivideo_x1_dc7a3035/feature_map.json
2022-08-24 20:52:20,946 P23932 INFO Set column index...
2022-08-24 20:52:20,946 P23932 INFO Feature specs: {
    "group_id": "{'type': 'meta'}",
    "item_emb": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406}",
    "neg_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "neg_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "pos_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "pos_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10001, 'vocab_size': 10002}"
}
2022-08-24 20:52:29,121 P23932 INFO Total number of parameters: 42246834.
2022-08-24 20:52:29,121 P23932 INFO Loading data...
2022-08-24 20:52:29,121 P23932 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/train.h5
2022-08-24 20:52:53,025 P23932 INFO Train samples: total/10931092, blocks/1
2022-08-24 20:52:53,025 P23932 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/valid.h5
2022-08-24 20:52:58,815 P23932 INFO Validation samples: total/2730291, blocks/1
2022-08-24 20:52:58,815 P23932 INFO Loading train and validation data done.
2022-08-24 20:52:58,815 P23932 INFO Start training: 2669 batches/epoch
2022-08-24 20:52:58,815 P23932 INFO ************ Epoch=1 start ************
2022-08-24 21:04:11,849 P23932 INFO [Metrics] AUC: 0.730401 - gAUC: 0.644536
2022-08-24 21:04:11,861 P23932 INFO Save best model: monitor(max): 1.374938
2022-08-24 21:04:13,631 P23932 INFO --- 2669/2669 batches finished ---
2022-08-24 21:04:13,707 P23932 INFO Train loss: 0.460896
2022-08-24 21:04:13,707 P23932 INFO ************ Epoch=1 end ************
2022-08-24 21:15:36,198 P23932 INFO [Metrics] AUC: 0.736379 - gAUC: 0.651744
2022-08-24 21:15:36,209 P23932 INFO Save best model: monitor(max): 1.388124
2022-08-24 21:15:38,672 P23932 INFO --- 2669/2669 batches finished ---
2022-08-24 21:15:38,752 P23932 INFO Train loss: 0.448256
2022-08-24 21:15:38,752 P23932 INFO ************ Epoch=2 end ************
2022-08-24 21:26:48,214 P23932 INFO [Metrics] AUC: 0.736704 - gAUC: 0.653011
2022-08-24 21:26:48,236 P23932 INFO Save best model: monitor(max): 1.389715
2022-08-24 21:26:50,363 P23932 INFO --- 2669/2669 batches finished ---
2022-08-24 21:26:50,452 P23932 INFO Train loss: 0.445617
2022-08-24 21:26:50,452 P23932 INFO ************ Epoch=3 end ************
2022-08-24 21:38:18,051 P23932 INFO [Metrics] AUC: 0.735972 - gAUC: 0.653728
2022-08-24 21:38:18,060 P23932 INFO Monitor(max) STOP: 1.389700 !
2022-08-24 21:38:18,060 P23932 INFO Reduce learning rate on plateau: 0.000100
2022-08-24 21:38:18,060 P23932 INFO --- 2669/2669 batches finished ---
2022-08-24 21:38:18,138 P23932 INFO Train loss: 0.444526
2022-08-24 21:38:18,138 P23932 INFO ************ Epoch=4 end ************
2022-08-24 21:49:34,395 P23932 INFO [Metrics] AUC: 0.745799 - gAUC: 0.664983
2022-08-24 21:49:34,402 P23932 INFO Save best model: monitor(max): 1.410783
2022-08-24 21:49:36,689 P23932 INFO --- 2669/2669 batches finished ---
2022-08-24 21:49:36,763 P23932 INFO Train loss: 0.424377
2022-08-24 21:49:36,763 P23932 INFO ************ Epoch=5 end ************
2022-08-24 22:01:26,512 P23932 INFO [Metrics] AUC: 0.746044 - gAUC: 0.666445
2022-08-24 22:01:26,527 P23932 INFO Save best model: monitor(max): 1.412489
2022-08-24 22:01:28,728 P23932 INFO --- 2669/2669 batches finished ---
2022-08-24 22:01:28,800 P23932 INFO Train loss: 0.416678
2022-08-24 22:01:28,803 P23932 INFO ************ Epoch=6 end ************
2022-08-24 22:13:14,512 P23932 INFO [Metrics] AUC: 0.747020 - gAUC: 0.667875
2022-08-24 22:13:14,522 P23932 INFO Save best model: monitor(max): 1.414896
2022-08-24 22:13:16,729 P23932 INFO --- 2669/2669 batches finished ---
2022-08-24 22:13:16,816 P23932 INFO Train loss: 0.411805
2022-08-24 22:13:16,816 P23932 INFO ************ Epoch=7 end ************
2022-08-24 22:24:57,191 P23932 INFO [Metrics] AUC: 0.745296 - gAUC: 0.667209
2022-08-24 22:24:57,201 P23932 INFO Monitor(max) STOP: 1.412505 !
2022-08-24 22:24:57,201 P23932 INFO Reduce learning rate on plateau: 0.000010
2022-08-24 22:24:57,201 P23932 INFO --- 2669/2669 batches finished ---
2022-08-24 22:24:57,277 P23932 INFO Train loss: 0.407524
2022-08-24 22:24:57,277 P23932 INFO ************ Epoch=8 end ************
2022-08-24 22:36:30,383 P23932 INFO [Metrics] AUC: 0.745161 - gAUC: 0.667427
2022-08-24 22:36:30,397 P23932 INFO Monitor(max) STOP: 1.412587 !
2022-08-24 22:36:30,397 P23932 INFO Reduce learning rate on plateau: 0.000001
2022-08-24 22:36:30,397 P23932 INFO ********* Epoch==9 early stop *********
2022-08-24 22:36:30,397 P23932 INFO --- 2669/2669 batches finished ---
2022-08-24 22:36:30,475 P23932 INFO Train loss: 0.395520
2022-08-24 22:36:30,475 P23932 INFO Training finished.
2022-08-24 22:36:30,475 P23932 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/AOANet_kuaivideo_x1/kuaivideo_x1_dc7a3035/AOANet_kuaivideo_x1_006_71fb5d8c.model
2022-08-24 22:36:31,684 P23932 INFO ****** Validation evaluation ******
2022-08-24 22:38:34,368 P23932 INFO [Metrics] gAUC: 0.667875 - AUC: 0.747020 - logloss: 0.438034
2022-08-24 22:38:34,588 P23932 INFO ******** Test evaluation ********
2022-08-24 22:38:34,588 P23932 INFO Loading data...
2022-08-24 22:38:34,588 P23932 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/test.h5
2022-08-24 22:38:42,312 P23932 INFO Test samples: total/2730291, blocks/1
2022-08-24 22:38:42,312 P23932 INFO Loading test data done.
2022-08-24 22:40:42,985 P23932 INFO [Metrics] gAUC: 0.667875 - AUC: 0.747020 - logloss: 0.438034

```
