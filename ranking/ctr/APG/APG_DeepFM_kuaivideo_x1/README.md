## APG_DeepFM_kuaivideo_x1

A hands-on guide to run the APG model on the KuaiVideo_x1 dataset.

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
Please refer to [KuaiVideo_x1](https://github.com/reczoo/Datasets/tree/main/KuaiShou/KuaiVideo_x1) to get the dataset details.

### Code

We use the [APG](https://github.com/reczoo/FuxiCTR/tree/v2.0.3/model_zoo/APG) model code from [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/tree/v2.0.3) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.3.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.3
    ```

2. Create a data directory and put the downloaded data files in `../data/KuaiShou/KuaiVideo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [APG_DeepFM_kuaivideo_x1_tuner_config_01](./APG_DeepFM_kuaivideo_x1_tuner_config_01). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/APG
    nohup python run_expid.py --config YOUR_PATH/APG/APG_DeepFM_kuaivideo_x1_tuner_config_01 --expid APG_DeepFM_kuaivideo_x1_007_5838d908 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.667573 | 0.745036 | 0.441939  |


### Logs
```python
2023-06-03 06:32:40,135 P70166 INFO Params: {
    "batch_norm": "False",
    "batch_size": "8192",
    "condition_features": "None",
    "condition_mode": "self-wise",
    "data_format": "csv",
    "data_root": "../data/KuaiShou/",
    "dataset_id": "kuaivideo_x1_dc7a3035",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.0001",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'min_categr_count': 1, 'name': 'item_emb', 'preprocess': 'copy_from(item_id)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'pos_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'neg_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'pos_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(pos_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'neg_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(neg_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}]",
    "feature_config": "None",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_emb'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'pos_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'neg_items'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'pos_items_emb'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'neg_items_emb'}]",
    "generate_bias": "True",
    "gpu": "6",
    "group_id": "user_id",
    "hidden_activations": "relu",
    "hidden_units": "[1024, 512, 256]",
    "hypernet_config": "{'dropout_rates': 0.1, 'hidden_activations': 'relu', 'hidden_units': []}",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "APG_DeepFM",
    "model_id": "APG_DeepFM_kuaivideo_x1_007_5838d908",
    "model_root": "./checkpoints/APG_DeepFM_kuaivideo_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "new_condition_emb": "False",
    "num_workers": "3",
    "optimizer": "adam",
    "overparam_p": "[32, 16, 8]",
    "pickle_feature_encoder": "True",
    "rank_k": "[32, 16, 8]",
    "save_best_only": "True",
    "seed": "20222023",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "train_data": "../data/KuaiShou/KuaiVideo_x1/train.csv",
    "use_features": "None",
    "valid_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "verbose": "1"
}
2023-06-03 06:32:40,136 P70166 INFO Set up feature processor...
2023-06-03 06:32:40,136 P70166 WARNING Skip rebuilding ../data/KuaiShou/kuaivideo_x1_dc7a3035/feature_map.json. Please delete it manually if rebuilding is required.
2023-06-03 06:32:40,136 P70166 INFO Load feature_map from json: ../data/KuaiShou/kuaivideo_x1_dc7a3035/feature_map.json
2023-06-03 06:32:40,136 P70166 INFO Set column index...
2023-06-03 06:32:40,137 P70166 INFO Feature specs: {
    "group_id": "{'type': 'meta'}",
    "item_emb": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406}",
    "neg_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "neg_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "pos_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "pos_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10001, 'vocab_size': 10002}"
}
2023-06-03 06:32:49,416 P70166 INFO Total number of parameters: 54712253.
2023-06-03 06:32:49,416 P70166 INFO Loading data...
2023-06-03 06:32:49,416 P70166 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/train.h5
2023-06-03 06:33:13,469 P70166 INFO Train samples: total/10931092, blocks/1
2023-06-03 06:33:13,470 P70166 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/valid.h5
2023-06-03 06:33:19,255 P70166 INFO Validation samples: total/2730291, blocks/1
2023-06-03 06:33:19,256 P70166 INFO Loading train and validation data done.
2023-06-03 06:33:19,256 P70166 INFO Start training: 1335 batches/epoch
2023-06-03 06:33:19,256 P70166 INFO ************ Epoch=1 start ************
2023-06-03 06:37:25,032 P70166 INFO Train loss: 0.515158
2023-06-03 06:37:25,033 P70166 INFO Evaluation @epoch 1 - batch 1335: 
2023-06-03 06:38:44,085 P70166 INFO [Metrics] AUC: 0.713362 - gAUC: 0.639544
2023-06-03 06:38:44,087 P70166 INFO Save best model: monitor(max)=1.352906
2023-06-03 06:38:46,690 P70166 INFO ************ Epoch=1 end ************
2023-06-03 06:44:04,628 P70166 INFO Train loss: 0.468914
2023-06-03 06:44:04,631 P70166 INFO Evaluation @epoch 2 - batch 1335: 
2023-06-03 06:45:25,534 P70166 INFO [Metrics] AUC: 0.713932 - gAUC: 0.643671
2023-06-03 06:45:25,538 P70166 INFO Save best model: monitor(max)=1.357603
2023-06-03 06:45:28,150 P70166 INFO ************ Epoch=2 end ************
2023-06-03 06:50:42,165 P70166 INFO Train loss: 0.463832
2023-06-03 06:50:42,165 P70166 INFO Evaluation @epoch 3 - batch 1335: 
2023-06-03 06:52:05,910 P70166 INFO [Metrics] AUC: 0.718080 - gAUC: 0.649056
2023-06-03 06:52:05,912 P70166 INFO Save best model: monitor(max)=1.367135
2023-06-03 06:52:08,391 P70166 INFO ************ Epoch=3 end ************
2023-06-03 06:57:30,530 P70166 INFO Train loss: 0.461261
2023-06-03 06:57:30,531 P70166 INFO Evaluation @epoch 4 - batch 1335: 
2023-06-03 06:58:51,402 P70166 INFO [Metrics] AUC: 0.717861 - gAUC: 0.651169
2023-06-03 06:58:51,405 P70166 INFO Save best model: monitor(max)=1.369030
2023-06-03 06:58:53,795 P70166 INFO ************ Epoch=4 end ************
2023-06-03 07:04:09,901 P70166 INFO Train loss: 0.459384
2023-06-03 07:04:09,906 P70166 INFO Evaluation @epoch 5 - batch 1335: 
2023-06-03 07:05:32,094 P70166 INFO [Metrics] AUC: 0.721888 - gAUC: 0.655096
2023-06-03 07:05:32,099 P70166 INFO Save best model: monitor(max)=1.376984
2023-06-03 07:05:34,645 P70166 INFO ************ Epoch=5 end ************
2023-06-03 07:10:48,983 P70166 INFO Train loss: 0.457978
2023-06-03 07:10:48,984 P70166 INFO Evaluation @epoch 6 - batch 1335: 
2023-06-03 07:12:09,711 P70166 INFO [Metrics] AUC: 0.722433 - gAUC: 0.656997
2023-06-03 07:12:09,713 P70166 INFO Save best model: monitor(max)=1.379430
2023-06-03 07:12:12,330 P70166 INFO ************ Epoch=6 end ************
2023-06-03 07:17:25,567 P70166 INFO Train loss: 0.456872
2023-06-03 07:17:25,572 P70166 INFO Evaluation @epoch 7 - batch 1335: 
2023-06-03 07:18:47,539 P70166 INFO [Metrics] AUC: 0.721520 - gAUC: 0.656891
2023-06-03 07:18:47,540 P70166 INFO Monitor(max)=1.378412 STOP!
2023-06-03 07:18:47,540 P70166 INFO Reduce learning rate on plateau: 0.000100
2023-06-03 07:18:47,611 P70166 INFO ************ Epoch=7 end ************
2023-06-03 07:24:06,525 P70166 INFO Train loss: 0.418989
2023-06-03 07:24:06,526 P70166 INFO Evaluation @epoch 8 - batch 1335: 
2023-06-03 07:25:28,575 P70166 INFO [Metrics] AUC: 0.744775 - gAUC: 0.666257
2023-06-03 07:25:28,576 P70166 INFO Save best model: monitor(max)=1.411032
2023-06-03 07:25:31,005 P70166 INFO ************ Epoch=8 end ************
2023-06-03 07:30:30,429 P70166 INFO Train loss: 0.408809
2023-06-03 07:30:30,429 P70166 INFO Evaluation @epoch 9 - batch 1335: 
2023-06-03 07:31:43,647 P70166 INFO [Metrics] AUC: 0.745036 - gAUC: 0.667573
2023-06-03 07:31:43,649 P70166 INFO Save best model: monitor(max)=1.412609
2023-06-03 07:31:45,961 P70166 INFO ************ Epoch=9 end ************
2023-06-03 07:36:21,724 P70166 INFO Train loss: 0.403375
2023-06-03 07:36:21,724 P70166 INFO Evaluation @epoch 10 - batch 1335: 
2023-06-03 07:37:32,543 P70166 INFO [Metrics] AUC: 0.744316 - gAUC: 0.667522
2023-06-03 07:37:32,547 P70166 INFO Monitor(max)=1.411838 STOP!
2023-06-03 07:37:32,547 P70166 INFO Reduce learning rate on plateau: 0.000010
2023-06-03 07:37:32,608 P70166 INFO ************ Epoch=10 end ************
2023-06-03 07:41:28,141 P70166 INFO Train loss: 0.388930
2023-06-03 07:41:28,142 P70166 INFO Evaluation @epoch 11 - batch 1335: 
2023-06-03 07:42:10,860 P70166 INFO [Metrics] AUC: 0.742410 - gAUC: 0.665792
2023-06-03 07:42:10,863 P70166 INFO Monitor(max)=1.408203 STOP!
2023-06-03 07:42:10,863 P70166 INFO Reduce learning rate on plateau: 0.000001
2023-06-03 07:42:10,863 P70166 INFO ********* Epoch==11 early stop *********
2023-06-03 07:42:10,941 P70166 INFO Training finished.
2023-06-03 07:42:10,941 P70166 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/APG_DeepFM_kuaivideo_x1/kuaivideo_x1_dc7a3035/APG_DeepFM_kuaivideo_x1_007_5838d908.model
2023-06-03 07:42:11,898 P70166 INFO ****** Validation evaluation ******
2023-06-03 07:42:59,597 P70166 INFO [Metrics] gAUC: 0.667573 - AUC: 0.745036 - logloss: 0.441939
2023-06-03 07:42:59,779 P70166 INFO ******** Test evaluation ********
2023-06-03 07:42:59,779 P70166 INFO Loading data...
2023-06-03 07:42:59,779 P70166 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/test.h5
2023-06-03 07:43:06,513 P70166 INFO Test samples: total/2730291, blocks/1
2023-06-03 07:43:06,513 P70166 INFO Loading test data done.
2023-06-03 07:43:54,144 P70166 INFO [Metrics] gAUC: 0.667573 - AUC: 0.745036 - logloss: 0.441939

```
