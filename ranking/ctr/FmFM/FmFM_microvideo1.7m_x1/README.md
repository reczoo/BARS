## FmFM_microvideo1.7m_x1

A hands-on guide to run the FmFM model on the MicroVideo1.7M_x1 dataset.

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

We use the [FmFM](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/FmFM) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/MicroVideo1.7M/MicroVideo1.7M_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FmFM_microvideo1.7m_x1_tuner_config_03](./FmFM_microvideo1.7m_x1_tuner_config_03). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/FmFM
    nohup python run_expid.py --config XXX/benchmarks/FmFM/FmFM_microvideo1.7m_x1_tuner_config_03 --expid FmFM_microvideo1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.673692 | 0.721519 | 0.418067  |


### Logs
```python
2022-08-19 14:37:14,555 P87638 INFO Params: {
    "batch_size": "2048",
    "data_format": "csv",
    "data_root": "../data/MicroVideo1.7M/",
    "dataset_id": "microvideo1.7m_x1_0d855fe6",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'name': 'clicked_items', 'padding': 'pre', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'clicked_categories', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'clicked_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'clicked_categories'}]",
    "field_interaction_type": "vectorized",
    "gpu": "7",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FmFM",
    "model_id": "FmFM_microvideo1.7m_x1_008_b522838a",
    "model_root": "./checkpoints/FmFM_microvideo1.7m_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "regularizer": "5e-05",
    "save_best_only": "True",
    "seed": "2022",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "train_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/train.csv",
    "valid_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "verbose": "1"
}
2022-08-19 14:37:14,555 P87638 INFO Set up feature processor...
2022-08-19 14:37:14,555 P87638 INFO Load feature_map from json: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/feature_map.json
2022-08-19 14:37:14,556 P87638 INFO Set column index...
2022-08-19 14:37:14,556 P87638 INFO Feature specs: {
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514}",
    "clicked_categories": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514, 'max_len': 100}",
    "clicked_items": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'max_len': 100}",
    "group_id": "{'type': 'meta'}",
    "item_id": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10987, 'vocab_size': 10988}"
}
2022-08-19 14:37:26,938 P87638 INFO Total number of parameters: 4166741.
2022-08-19 14:37:26,938 P87638 INFO Loading data...
2022-08-19 14:37:26,939 P87638 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/train.h5
2022-08-19 14:40:17,825 P87638 INFO Train samples: total/8970309, blocks/1
2022-08-19 14:40:17,825 P87638 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/valid.h5
2022-08-19 14:42:49,778 P87638 INFO Validation samples: total/3767308, blocks/1
2022-08-19 14:42:49,778 P87638 INFO Loading train and validation data done.
2022-08-19 14:42:49,778 P87638 INFO Start training: 4381 batches/epoch
2022-08-19 14:42:49,778 P87638 INFO ************ Epoch=1 start ************
2022-08-19 15:12:31,461 P87638 INFO [Metrics] AUC: 0.713818 - gAUC: 0.668221
2022-08-19 15:12:31,488 P87638 INFO Save best model: monitor(max): 1.382039
2022-08-19 15:12:34,363 P87638 INFO --- 4381/4381 batches finished ---
2022-08-19 15:12:34,841 P87638 INFO Train loss: 0.447963
2022-08-19 15:12:34,841 P87638 INFO ************ Epoch=1 end ************
2022-08-19 15:41:34,879 P87638 INFO [Metrics] AUC: 0.715846 - gAUC: 0.670929
2022-08-19 15:41:35,005 P87638 INFO Save best model: monitor(max): 1.386775
2022-08-19 15:41:37,937 P87638 INFO --- 4381/4381 batches finished ---
2022-08-19 15:41:38,288 P87638 INFO Train loss: 0.434585
2022-08-19 15:41:38,288 P87638 INFO ************ Epoch=2 end ************
2022-08-19 16:10:40,275 P87638 INFO [Metrics] AUC: 0.716974 - gAUC: 0.672127
2022-08-19 16:10:40,289 P87638 INFO Save best model: monitor(max): 1.389101
2022-08-19 16:10:43,570 P87638 INFO --- 4381/4381 batches finished ---
2022-08-19 16:10:43,851 P87638 INFO Train loss: 0.431487
2022-08-19 16:10:43,852 P87638 INFO ************ Epoch=3 end ************
2022-08-19 16:37:19,201 P87638 INFO [Metrics] AUC: 0.717680 - gAUC: 0.672624
2022-08-19 16:37:19,223 P87638 INFO Save best model: monitor(max): 1.390304
2022-08-19 16:37:21,958 P87638 INFO --- 4381/4381 batches finished ---
2022-08-19 16:37:22,306 P87638 INFO Train loss: 0.429749
2022-08-19 16:37:22,306 P87638 INFO ************ Epoch=4 end ************
2022-08-19 16:58:59,999 P87638 INFO [Metrics] AUC: 0.717986 - gAUC: 0.672777
2022-08-19 16:59:00,011 P87638 INFO Save best model: monitor(max): 1.390763
2022-08-19 16:59:02,243 P87638 INFO --- 4381/4381 batches finished ---
2022-08-19 16:59:02,472 P87638 INFO Train loss: 0.428595
2022-08-19 16:59:02,472 P87638 INFO ************ Epoch=5 end ************
2022-08-19 17:15:41,600 P87638 INFO [Metrics] AUC: 0.717704 - gAUC: 0.672823
2022-08-19 17:15:41,610 P87638 INFO Monitor(max) STOP: 1.390527 !
2022-08-19 17:15:41,610 P87638 INFO Reduce learning rate on plateau: 0.000050
2022-08-19 17:15:41,610 P87638 INFO --- 4381/4381 batches finished ---
2022-08-19 17:15:41,804 P87638 INFO Train loss: 0.427882
2022-08-19 17:15:41,804 P87638 INFO ************ Epoch=6 end ************
2022-08-19 17:30:02,703 P87638 INFO [Metrics] AUC: 0.721519 - gAUC: 0.673692
2022-08-19 17:30:02,713 P87638 INFO Save best model: monitor(max): 1.395211
2022-08-19 17:30:04,952 P87638 INFO --- 4381/4381 batches finished ---
2022-08-19 17:30:05,175 P87638 INFO Train loss: 0.414288
2022-08-19 17:30:05,175 P87638 INFO ************ Epoch=7 end ************
2022-08-19 17:42:49,225 P87638 INFO [Metrics] AUC: 0.721607 - gAUC: 0.672971
2022-08-19 17:42:49,235 P87638 INFO Monitor(max) STOP: 1.394577 !
2022-08-19 17:42:49,235 P87638 INFO Reduce learning rate on plateau: 0.000005
2022-08-19 17:42:49,236 P87638 INFO --- 4381/4381 batches finished ---
2022-08-19 17:42:49,465 P87638 INFO Train loss: 0.411273
2022-08-19 17:42:49,465 P87638 INFO ************ Epoch=8 end ************
2022-08-19 17:54:54,416 P87638 INFO [Metrics] AUC: 0.721623 - gAUC: 0.672934
2022-08-19 17:54:54,426 P87638 INFO Monitor(max) STOP: 1.394557 !
2022-08-19 17:54:54,426 P87638 INFO Reduce learning rate on plateau: 0.000001
2022-08-19 17:54:54,426 P87638 INFO ********* Epoch==9 early stop *********
2022-08-19 17:54:54,426 P87638 INFO --- 4381/4381 batches finished ---
2022-08-19 17:54:54,653 P87638 INFO Train loss: 0.407949
2022-08-19 17:54:54,653 P87638 INFO Training finished.
2022-08-19 17:54:54,653 P87638 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/FmFM_microvideo1.7m_x1/microvideo1.7m_x1_0d855fe6/FmFM_microvideo1.7m_x1_008_b522838a.model
2022-08-19 17:54:55,355 P87638 INFO ****** Validation evaluation ******
2022-08-19 17:58:41,245 P87638 INFO [Metrics] gAUC: 0.673692 - AUC: 0.721519 - logloss: 0.418067
2022-08-19 17:58:41,607 P87638 INFO ******** Test evaluation ********
2022-08-19 17:58:41,608 P87638 INFO Loading data...
2022-08-19 17:58:41,608 P87638 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/test.h5
2022-08-19 17:58:46,148 P87638 INFO Test samples: total/3767308, blocks/1
2022-08-19 17:58:46,148 P87638 INFO Loading test data done.
2022-08-19 18:02:33,069 P87638 INFO [Metrics] gAUC: 0.673692 - AUC: 0.721519 - logloss: 0.418067

```
