## DIN_microvideo1.7m_x1

A hands-on guide to run the DIN model on the MicroVideo1.7M_x1 dataset.

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

We use the [DIN](https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/DIN) model code from [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/tree/v2.0.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.1
    ```

2. Create a data directory and put the downloaded data files in `../data/MicroVideo1.7M/MicroVideo1.7M_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DIN_microvideo1.7m_x1_tuner_config_06](./DIN_microvideo1.7m_x1_tuner_config_06). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DIN
    nohup python run_expid.py --config XXX/benchmarks/DIN/DIN_microvideo1.7m_x1_tuner_config_06 --expid DIN_microvideo1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.688282 | 0.736006 | 0.411558  |


### Logs
```python
2022-08-18 13:09:12,359 P87342 INFO Params: {
    "attention_dropout": "0.2",
    "attention_hidden_activations": "ReLU",
    "attention_hidden_units": "[512, 256]",
    "attention_output_activation": "None",
    "batch_norm": "True",
    "batch_size": "2048",
    "data_format": "csv",
    "data_root": "../data/MicroVideo1.7M/",
    "dataset_id": "microvideo1.7m_x1_0d855fe6",
    "debug_mode": "False",
    "din_sequence_field": "('clicked_items', 'clicked_categories')",
    "din_target_field": "('item_id', 'cate_id')",
    "din_use_softmax": "True",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'name': 'clicked_items', 'padding': 'pre', 'pretrained_emb': '../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'clicked_categories', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id'}, {'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'clicked_items'}, {'feature_encoder': None, 'name': 'clicked_categories'}]",
    "gpu": "5",
    "group_id": "group_id",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DIN",
    "model_id": "DIN_microvideo1.7m_x1_006_ab4e3b7f",
    "model_root": "./checkpoints/DIN_microvideo1.7m_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2022",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "train_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/train.csv",
    "valid_data": "../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv",
    "verbose": "1"
}
2022-08-18 13:09:12,360 P87342 INFO Set up feature processor...
2022-08-18 13:09:12,360 P87342 INFO Load feature_map from json: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/feature_map.json
2022-08-18 13:09:12,360 P87342 INFO Set column index...
2022-08-18 13:09:12,361 P87342 INFO Feature specs: {
    "cate_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514}",
    "clicked_categories": "{'source': '', 'type': 'sequence', 'feature_encoder': None, 'share_embedding': 'cate_id', 'padding_idx': 0, 'oov_idx': 513, 'vocab_size': 514, 'max_len': 100}",
    "clicked_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'max_len': 100}",
    "group_id": "{'type': 'meta'}",
    "item_id": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 1704881, 'vocab_size': 1704882, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10987, 'vocab_size': 10988}"
}
2022-08-18 13:09:23,711 P87342 INFO Total number of parameters: 2127234.
2022-08-18 13:09:23,712 P87342 INFO Loading data...
2022-08-18 13:09:23,712 P87342 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/train.h5
2022-08-18 13:09:37,812 P87342 INFO Train samples: total/8970309, blocks/1
2022-08-18 13:09:37,813 P87342 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/valid.h5
2022-08-18 13:09:43,269 P87342 INFO Validation samples: total/3767308, blocks/1
2022-08-18 13:09:43,269 P87342 INFO Loading train and validation data done.
2022-08-18 13:09:43,269 P87342 INFO Start training: 4381 batches/epoch
2022-08-18 13:09:43,270 P87342 INFO ************ Epoch=1 start ************
2022-08-18 13:41:19,803 P87342 INFO [Metrics] AUC: 0.718361 - gAUC: 0.674294
2022-08-18 13:41:19,820 P87342 INFO Save best model: monitor(max): 1.392655
2022-08-18 13:41:22,390 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 13:41:22,547 P87342 INFO Train loss: 0.466580
2022-08-18 13:41:22,547 P87342 INFO ************ Epoch=1 end ************
2022-08-18 14:12:53,180 P87342 INFO [Metrics] AUC: 0.722517 - gAUC: 0.677523
2022-08-18 14:12:53,189 P87342 INFO Save best model: monitor(max): 1.400040
2022-08-18 14:12:55,645 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 14:12:55,699 P87342 INFO Train loss: 0.442923
2022-08-18 14:12:55,699 P87342 INFO ************ Epoch=2 end ************
2022-08-18 14:44:26,892 P87342 INFO [Metrics] AUC: 0.725732 - gAUC: 0.679456
2022-08-18 14:44:26,922 P87342 INFO Save best model: monitor(max): 1.405188
2022-08-18 14:44:29,886 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 14:44:29,981 P87342 INFO Train loss: 0.440458
2022-08-18 14:44:29,981 P87342 INFO ************ Epoch=3 end ************
2022-08-18 15:15:53,394 P87342 INFO [Metrics] AUC: 0.726070 - gAUC: 0.680026
2022-08-18 15:15:53,411 P87342 INFO Save best model: monitor(max): 1.406096
2022-08-18 15:15:56,631 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 15:15:56,731 P87342 INFO Train loss: 0.438041
2022-08-18 15:15:56,731 P87342 INFO ************ Epoch=4 end ************
2022-08-18 15:47:15,990 P87342 INFO [Metrics] AUC: 0.726240 - gAUC: 0.680664
2022-08-18 15:47:16,003 P87342 INFO Save best model: monitor(max): 1.406904
2022-08-18 15:47:18,197 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 15:47:18,336 P87342 INFO Train loss: 0.436858
2022-08-18 15:47:18,336 P87342 INFO ************ Epoch=5 end ************
2022-08-18 16:18:41,930 P87342 INFO [Metrics] AUC: 0.727840 - gAUC: 0.681167
2022-08-18 16:18:41,941 P87342 INFO Save best model: monitor(max): 1.409007
2022-08-18 16:18:45,058 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 16:18:45,227 P87342 INFO Train loss: 0.435995
2022-08-18 16:18:45,227 P87342 INFO ************ Epoch=6 end ************
2022-08-18 16:50:02,604 P87342 INFO [Metrics] AUC: 0.726686 - gAUC: 0.680090
2022-08-18 16:50:02,624 P87342 INFO Monitor(max) STOP: 1.406776 !
2022-08-18 16:50:02,624 P87342 INFO Reduce learning rate on plateau: 0.000050
2022-08-18 16:50:02,625 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 16:50:02,727 P87342 INFO Train loss: 0.435214
2022-08-18 16:50:02,727 P87342 INFO ************ Epoch=7 end ************
2022-08-18 17:21:22,132 P87342 INFO [Metrics] AUC: 0.734210 - gAUC: 0.687178
2022-08-18 17:21:22,144 P87342 INFO Save best model: monitor(max): 1.421388
2022-08-18 17:21:24,417 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 17:21:24,537 P87342 INFO Train loss: 0.425088
2022-08-18 17:21:24,538 P87342 INFO ************ Epoch=8 end ************
2022-08-18 17:52:39,879 P87342 INFO [Metrics] AUC: 0.734825 - gAUC: 0.687170
2022-08-18 17:52:39,897 P87342 INFO Save best model: monitor(max): 1.421995
2022-08-18 17:52:42,587 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 17:52:42,707 P87342 INFO Train loss: 0.421457
2022-08-18 17:52:42,707 P87342 INFO ************ Epoch=9 end ************
2022-08-18 18:23:54,220 P87342 INFO [Metrics] AUC: 0.735162 - gAUC: 0.687399
2022-08-18 18:23:54,237 P87342 INFO Save best model: monitor(max): 1.422561
2022-08-18 18:23:57,013 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 18:23:57,119 P87342 INFO Train loss: 0.420101
2022-08-18 18:23:57,119 P87342 INFO ************ Epoch=10 end ************
2022-08-18 18:55:09,620 P87342 INFO [Metrics] AUC: 0.735736 - gAUC: 0.688266
2022-08-18 18:55:09,637 P87342 INFO Save best model: monitor(max): 1.424001
2022-08-18 18:55:12,080 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 18:55:12,222 P87342 INFO Train loss: 0.419014
2022-08-18 18:55:12,222 P87342 INFO ************ Epoch=11 end ************
2022-08-18 19:26:21,042 P87342 INFO [Metrics] AUC: 0.735457 - gAUC: 0.687930
2022-08-18 19:26:21,053 P87342 INFO Monitor(max) STOP: 1.423388 !
2022-08-18 19:26:21,054 P87342 INFO Reduce learning rate on plateau: 0.000005
2022-08-18 19:26:21,054 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 19:26:21,109 P87342 INFO Train loss: 0.418164
2022-08-18 19:26:21,110 P87342 INFO ************ Epoch=12 end ************
2022-08-18 19:55:15,736 P87342 INFO [Metrics] AUC: 0.735923 - gAUC: 0.688252
2022-08-18 19:55:15,750 P87342 INFO Save best model: monitor(max): 1.424175
2022-08-18 19:55:19,304 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 19:55:19,382 P87342 INFO Train loss: 0.414825
2022-08-18 19:55:19,382 P87342 INFO ************ Epoch=13 end ************
2022-08-18 20:15:55,734 P87342 INFO [Metrics] AUC: 0.735980 - gAUC: 0.688255
2022-08-18 20:15:55,746 P87342 INFO Save best model: monitor(max): 1.424235
2022-08-18 20:15:57,729 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 20:15:57,809 P87342 INFO Train loss: 0.414237
2022-08-18 20:15:57,810 P87342 INFO ************ Epoch=14 end ************
2022-08-18 20:31:52,846 P87342 INFO [Metrics] AUC: 0.735946 - gAUC: 0.688308
2022-08-18 20:31:52,852 P87342 INFO Save best model: monitor(max): 1.424254
2022-08-18 20:31:54,577 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 20:31:54,649 P87342 INFO Train loss: 0.413959
2022-08-18 20:31:54,649 P87342 INFO ************ Epoch=15 end ************
2022-08-18 20:44:35,501 P87342 INFO [Metrics] AUC: 0.735923 - gAUC: 0.688249
2022-08-18 20:44:35,510 P87342 INFO Monitor(max) STOP: 1.424172 !
2022-08-18 20:44:35,510 P87342 INFO Reduce learning rate on plateau: 0.000001
2022-08-18 20:44:35,511 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 20:44:35,577 P87342 INFO Train loss: 0.413707
2022-08-18 20:44:35,578 P87342 INFO ************ Epoch=16 end ************
2022-08-18 20:53:42,149 P87342 INFO [Metrics] AUC: 0.735978 - gAUC: 0.688299
2022-08-18 20:53:42,157 P87342 INFO Save best model: monitor(max): 1.424277
2022-08-18 20:53:44,157 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 20:53:44,220 P87342 INFO Train loss: 0.413106
2022-08-18 20:53:44,220 P87342 INFO ************ Epoch=17 end ************
2022-08-18 21:02:20,932 P87342 INFO [Metrics] AUC: 0.736006 - gAUC: 0.688282
2022-08-18 21:02:20,941 P87342 INFO Save best model: monitor(max): 1.424289
2022-08-18 21:02:22,931 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 21:02:22,991 P87342 INFO Train loss: 0.413031
2022-08-18 21:02:22,991 P87342 INFO ************ Epoch=18 end ************
2022-08-18 21:10:40,443 P87342 INFO [Metrics] AUC: 0.735928 - gAUC: 0.688199
2022-08-18 21:10:40,450 P87342 INFO Monitor(max) STOP: 1.424127 !
2022-08-18 21:10:40,451 P87342 INFO Reduce learning rate on plateau: 0.000001
2022-08-18 21:10:40,451 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 21:10:40,512 P87342 INFO Train loss: 0.412985
2022-08-18 21:10:40,513 P87342 INFO ************ Epoch=19 end ************
2022-08-18 21:18:10,393 P87342 INFO [Metrics] AUC: 0.735886 - gAUC: 0.688118
2022-08-18 21:18:10,399 P87342 INFO Monitor(max) STOP: 1.424004 !
2022-08-18 21:18:10,400 P87342 INFO Reduce learning rate on plateau: 0.000001
2022-08-18 21:18:10,400 P87342 INFO ********* Epoch==20 early stop *********
2022-08-18 21:18:10,400 P87342 INFO --- 4381/4381 batches finished ---
2022-08-18 21:18:10,452 P87342 INFO Train loss: 0.413004
2022-08-18 21:18:10,452 P87342 INFO Training finished.
2022-08-18 21:18:10,452 P87342 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/DIN_microvideo1.7m_x1/microvideo1.7m_x1_0d855fe6/DIN_microvideo1.7m_x1_006_ab4e3b7f.model
2022-08-18 21:18:11,113 P87342 INFO ****** Validation evaluation ******
2022-08-18 21:19:39,187 P87342 INFO [Metrics] gAUC: 0.688282 - AUC: 0.736006 - logloss: 0.411558
2022-08-18 21:19:39,305 P87342 INFO ******** Test evaluation ********
2022-08-18 21:19:39,305 P87342 INFO Loading data...
2022-08-18 21:19:39,305 P87342 INFO Loading data from h5: ../data/MicroVideo1.7M/microvideo1.7m_x1_0d855fe6/test.h5
2022-08-18 21:19:43,307 P87342 INFO Test samples: total/3767308, blocks/1
2022-08-18 21:19:43,307 P87342 INFO Loading test data done.
2022-08-18 21:21:13,418 P87342 INFO [Metrics] gAUC: 0.688282 - AUC: 0.736006 - logloss: 0.411558

```
