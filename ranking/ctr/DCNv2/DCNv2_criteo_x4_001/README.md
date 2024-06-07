## DCNv2_criteo_x4_001

A hands-on guide to run the DCNv2 model on the Criteo_x4 dataset.

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
  CUDA: 10.2
  python: 3.7.10
  pytorch: 1.10.2+cu102
  pandas: 1.1.5
  numpy: 1.19.5
  scipy: 1.5.2
  sklearn: 0.22.1
  pyyaml: 6.0.1
  h5py: 2.8.0
  tqdm: 4.64.0
  keras_preprocessing: 1.1.2
  fuxictr: 2.2.0
  ```

### Dataset
Please refer to [Criteo_x4](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x4) to get the dataset details.

### Code

We use the [DCNv2](https://github.com/reczoo/FuxiCTR/tree/v2.2.0/model_zoo/DCNv2) model code from [FuxiCTR-v2.2.0](https://github.com/reczoo/FuxiCTR/tree/v2.2.0) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.2.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.2.0.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.2.0
    ```

2. Create a data directory and put the downloaded data files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DCNv2_criteo_x4_tuner_config_01](./DCNv2_criteo_x4_tuner_config_01). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DCNv2
    nohup python run_expid.py --config YOUR_PATH/DCNv2/DCNv2_criteo_x4_tuner_config_01 --expid DCNv2_criteo_x4_001_005_c2376d55 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.814514 | 0.437631  |


### Logs
```python
2024-02-19 11:28:52,376 P3740343 INFO Params: {
    "batch_norm": "True",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_001_a5e05ce7",
    "debug_mode": "False",
    "dnn_activations": "relu",
    "early_stop_patience": "2",
    "embedding_dim": "16",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'fill_na': 0, 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'preprocess': 'convert_to_bucket', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'fill_na': '', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "feature_config": "None",
    "feature_specs": "None",
    "gpu": "4",
    "group_id": "None",
    "label_col": "{'dtype': 'float', 'name': 'Label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "low_rank": "32",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "DCNv2",
    "model_id": "DCNv2_criteo_x4_001_005_c2376d55",
    "model_root": "./checkpoints/",
    "model_structure": "parallel",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_cross_layers": "3",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_dnn_hidden_units": "[1000, 1000, 1000, 1000, 1000]",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "stacked_dnn_hidden_units": "[500, 500, 500]",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x4/test.csv",
    "train_data": "../data/Criteo/Criteo_x4/train.csv",
    "use_features": "None",
    "use_low_rank_mixture": "False",
    "valid_data": "../data/Criteo/Criteo_x4/valid.csv",
    "verbose": "1"
}
2024-02-19 11:28:52,376 P3740343 INFO Set up feature processor...
2024-02-19 11:28:52,377 P3740343 WARNING Skip rebuilding ../data/Criteo/criteo_x4_001_a5e05ce7/feature_map.json. Please delete it manually if rebuilding is required.
2024-02-19 11:28:52,377 P3740343 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_a5e05ce7/feature_map.json
2024-02-19 11:28:52,377 P3740343 INFO Set column index...
2024-02-19 11:28:52,377 P3740343 INFO Feature specs: {
    "C1": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 1446}",
    "C10": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 39530}",
    "C11": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 5131}",
    "C12": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 156656}",
    "C13": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 3176}",
    "C14": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 28}",
    "C15": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 11043}",
    "C16": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 148913}",
    "C17": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 12}",
    "C18": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 4560}",
    "C19": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 2003}",
    "C2": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 554}",
    "C20": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 5}",
    "C21": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 154564}",
    "C22": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 18}",
    "C23": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 17}",
    "C24": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 53031}",
    "C25": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 82}",
    "C26": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 40955}",
    "C3": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 157339}",
    "C4": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 117822}",
    "C5": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 306}",
    "C6": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 18}",
    "C7": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 11882}",
    "C8": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 630}",
    "C9": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 5}",
    "I1": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 44}",
    "I10": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 6}",
    "I11": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 27}",
    "I12": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 37}",
    "I13": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 72}",
    "I2": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 99}",
    "I3": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 122}",
    "I4": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 41}",
    "I5": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 220}",
    "I6": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 112}",
    "I7": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 80}",
    "I8": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 69}",
    "I9": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 92}"
}
2024-02-19 11:28:57,677 P3740343 INFO Total number of parameters: 20382577.
2024-02-19 11:28:57,677 P3740343 INFO Loading datasets...
2024-02-19 11:29:33,844 P3740343 INFO Train samples: total/36672493, blocks/1
2024-02-19 11:29:38,337 P3740343 INFO Validation samples: total/4584062, blocks/1
2024-02-19 11:29:38,337 P3740343 INFO Loading train and validation data done.
2024-02-19 11:29:38,338 P3740343 INFO Start training: 3668 batches/epoch
2024-02-19 11:29:38,338 P3740343 INFO ************ Epoch=1 start ************
2024-02-19 11:34:45,532 P3740343 INFO Train loss: 0.458952
2024-02-19 11:34:45,533 P3740343 INFO Evaluation @epoch 1 - batch 3668: 
2024-02-19 11:35:00,517 P3740343 INFO [Metrics] AUC: 0.806450
2024-02-19 11:35:00,521 P3740343 INFO Save best model: monitor(max)=0.806450
2024-02-19 11:35:00,679 P3740343 INFO ************ Epoch=1 end ************
2024-02-19 11:40:08,747 P3740343 INFO Train loss: 0.451408
2024-02-19 11:40:08,747 P3740343 INFO Evaluation @epoch 2 - batch 3668: 
2024-02-19 11:40:24,340 P3740343 INFO [Metrics] AUC: 0.809079
2024-02-19 11:40:24,344 P3740343 INFO Save best model: monitor(max)=0.809079
2024-02-19 11:40:24,518 P3740343 INFO ************ Epoch=2 end ************
2024-02-19 11:45:34,026 P3740343 INFO Train loss: 0.449765
2024-02-19 11:45:34,027 P3740343 INFO Evaluation @epoch 3 - batch 3668: 
2024-02-19 11:45:49,178 P3740343 INFO [Metrics] AUC: 0.810513
2024-02-19 11:45:49,180 P3740343 INFO Save best model: monitor(max)=0.810513
2024-02-19 11:45:49,358 P3740343 INFO ************ Epoch=3 end ************
2024-02-19 11:50:55,398 P3740343 INFO Train loss: 0.448540
2024-02-19 11:50:55,398 P3740343 INFO Evaluation @epoch 4 - batch 3668: 
2024-02-19 11:51:10,992 P3740343 INFO [Metrics] AUC: 0.811169
2024-02-19 11:51:10,993 P3740343 INFO Save best model: monitor(max)=0.811169
2024-02-19 11:51:11,171 P3740343 INFO ************ Epoch=4 end ************
2024-02-19 11:56:19,561 P3740343 INFO Train loss: 0.447629
2024-02-19 11:56:19,562 P3740343 INFO Evaluation @epoch 5 - batch 3668: 
2024-02-19 11:56:34,608 P3740343 INFO [Metrics] AUC: 0.811757
2024-02-19 11:56:34,609 P3740343 INFO Save best model: monitor(max)=0.811757
2024-02-19 11:56:34,790 P3740343 INFO ************ Epoch=5 end ************
2024-02-19 12:01:40,413 P3740343 INFO Train loss: 0.446832
2024-02-19 12:01:40,413 P3740343 INFO Evaluation @epoch 6 - batch 3668: 
2024-02-19 12:01:55,503 P3740343 INFO [Metrics] AUC: 0.811747
2024-02-19 12:01:55,506 P3740343 INFO Monitor(max)=0.811747 STOP!
2024-02-19 12:01:55,506 P3740343 INFO Reduce learning rate on plateau: 0.000100
2024-02-19 12:01:55,559 P3740343 INFO ************ Epoch=6 end ************
2024-02-19 12:07:00,817 P3740343 INFO Train loss: 0.435416
2024-02-19 12:07:00,817 P3740343 INFO Evaluation @epoch 7 - batch 3668: 
2024-02-19 12:07:15,793 P3740343 INFO [Metrics] AUC: 0.814037
2024-02-19 12:07:15,794 P3740343 INFO Save best model: monitor(max)=0.814037
2024-02-19 12:07:15,963 P3740343 INFO ************ Epoch=7 end ************
2024-02-19 12:12:25,653 P3740343 INFO Train loss: 0.431232
2024-02-19 12:12:25,654 P3740343 INFO Evaluation @epoch 8 - batch 3668: 
2024-02-19 12:12:40,871 P3740343 INFO [Metrics] AUC: 0.813989
2024-02-19 12:12:40,872 P3740343 INFO Monitor(max)=0.813989 STOP!
2024-02-19 12:12:40,872 P3740343 INFO Reduce learning rate on plateau: 0.000010
2024-02-19 12:12:40,923 P3740343 INFO ************ Epoch=8 end ************
2024-02-19 12:17:46,267 P3740343 INFO Train loss: 0.427107
2024-02-19 12:17:46,267 P3740343 INFO Evaluation @epoch 9 - batch 3668: 
2024-02-19 12:18:00,971 P3740343 INFO [Metrics] AUC: 0.813659
2024-02-19 12:18:00,972 P3740343 INFO Monitor(max)=0.813659 STOP!
2024-02-19 12:18:00,972 P3740343 INFO Reduce learning rate on plateau: 0.000001
2024-02-19 12:18:00,972 P3740343 INFO ********* Epoch==9 early stop *********
2024-02-19 12:18:01,025 P3740343 INFO Training finished.
2024-02-19 12:18:01,026 P3740343 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/criteo_x4_001_a5e05ce7/DCNv2_criteo_x4_001_005_c2376d55.model
2024-02-19 12:18:01,096 P3740343 INFO ****** Validation evaluation ******
2024-02-19 12:18:17,843 P3740343 INFO [Metrics] AUC: 0.814037 - logloss: 0.438087
2024-02-19 12:18:17,963 P3740343 INFO ******** Test evaluation ********
2024-02-19 12:18:17,964 P3740343 INFO Loading datasets...
2024-02-19 12:18:22,506 P3740343 INFO Test samples: total/4584062, blocks/1
2024-02-19 12:18:22,507 P3740343 INFO Loading test data done.
2024-02-19 12:18:38,751 P3740343 INFO [Metrics] AUC: 0.814514 - logloss: 0.437631

```
