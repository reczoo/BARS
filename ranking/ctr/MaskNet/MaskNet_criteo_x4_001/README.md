## MaskNet_criteo_x4_001

A hands-on guide to run the MaskNet model on the Criteo_x4 dataset.

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

We use the [MaskNet](https://github.com/reczoo/FuxiCTR/tree/v2.2.0/model_zoo/MaskNet) model code from [FuxiCTR-v2.2.0](https://github.com/reczoo/FuxiCTR/tree/v2.2.0) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.2.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.2.0.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.2.0
    ```

2. Create a data directory and put the downloaded data files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [MaskNet_criteo_x4_tuner_config_06](./MaskNet_criteo_x4_tuner_config_06). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/MaskNet
    nohup python run_expid.py --config YOUR_PATH/MaskNet/MaskNet_criteo_x4_tuner_config_06 --expid MaskNet_criteo_x4_001_018_ccc857cd --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.813420 | 0.438748  |


### Logs
```python
2024-02-23 11:34:31,352 P3402379 INFO Params: {
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_001_a5e05ce7",
    "debug_mode": "False",
    "dnn_hidden_activations": "relu",
    "dnn_hidden_units": "[1000, 1000, 1000, 1000, 1000]",
    "early_stop_patience": "2",
    "emb_layernorm": "False",
    "embedding_dim": "16",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'fill_na': 0, 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'preprocess': 'convert_to_bucket', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'fill_na': '', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "feature_config": "None",
    "feature_specs": "None",
    "gpu": "1",
    "group_id": "None",
    "label_col": "{'dtype': 'float', 'name': 'Label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "MaskNet",
    "model_id": "MaskNet_criteo_x4_001_018_ccc857cd",
    "model_root": "./checkpoints/",
    "model_type": "ParallelMaskNet",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_layernorm": "False",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_block_dim": "500",
    "parallel_num_blocks": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "0.1",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x4/test.csv",
    "train_data": "../data/Criteo/Criteo_x4/train.csv",
    "use_features": "None",
    "valid_data": "../data/Criteo/Criteo_x4/valid.csv",
    "verbose": "1"
}
2024-02-23 11:34:31,353 P3402379 INFO Set up feature processor...
2024-02-23 11:34:31,353 P3402379 WARNING Skip rebuilding ../data/Criteo/criteo_x4_001_a5e05ce7/feature_map.json. Please delete it manually if rebuilding is required.
2024-02-23 11:34:31,353 P3402379 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_a5e05ce7/feature_map.json
2024-02-23 11:34:31,354 P3402379 INFO Set column index...
2024-02-23 11:34:31,354 P3402379 INFO Feature specs: {
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
2024-02-23 11:34:35,198 P3402379 INFO Total number of parameters: 20358077.
2024-02-23 11:34:35,198 P3402379 INFO Loading datasets...
2024-02-23 11:35:11,583 P3402379 INFO Train samples: total/36672493, blocks/1
2024-02-23 11:35:16,175 P3402379 INFO Validation samples: total/4584062, blocks/1
2024-02-23 11:35:16,175 P3402379 INFO Loading train and validation data done.
2024-02-23 11:35:16,175 P3402379 INFO Start training: 3668 batches/epoch
2024-02-23 11:35:16,175 P3402379 INFO ************ Epoch=1 start ************
2024-02-23 11:40:13,140 P3402379 INFO Train loss: 0.459513
2024-02-23 11:40:13,141 P3402379 INFO Evaluation @epoch 1 - batch 3668: 
2024-02-23 11:40:27,684 P3402379 INFO [Metrics] AUC: 0.806480
2024-02-23 11:40:27,686 P3402379 INFO Save best model: monitor(max)=0.806480
2024-02-23 11:40:27,864 P3402379 INFO ************ Epoch=1 end ************
2024-02-23 11:45:24,583 P3402379 INFO Train loss: 0.452990
2024-02-23 11:45:24,583 P3402379 INFO Evaluation @epoch 2 - batch 3668: 
2024-02-23 11:45:39,011 P3402379 INFO [Metrics] AUC: 0.808234
2024-02-23 11:45:39,012 P3402379 INFO Save best model: monitor(max)=0.808234
2024-02-23 11:45:39,189 P3402379 INFO ************ Epoch=2 end ************
2024-02-23 11:50:35,804 P3402379 INFO Train loss: 0.451728
2024-02-23 11:50:35,804 P3402379 INFO Evaluation @epoch 3 - batch 3668: 
2024-02-23 11:50:50,241 P3402379 INFO [Metrics] AUC: 0.809256
2024-02-23 11:50:50,242 P3402379 INFO Save best model: monitor(max)=0.809256
2024-02-23 11:50:50,429 P3402379 INFO ************ Epoch=3 end ************
2024-02-23 11:55:47,287 P3402379 INFO Train loss: 0.451147
2024-02-23 11:55:47,287 P3402379 INFO Evaluation @epoch 4 - batch 3668: 
2024-02-23 11:56:01,675 P3402379 INFO [Metrics] AUC: 0.809425
2024-02-23 11:56:01,676 P3402379 INFO Save best model: monitor(max)=0.809425
2024-02-23 11:56:01,848 P3402379 INFO ************ Epoch=4 end ************
2024-02-23 12:00:59,781 P3402379 INFO Train loss: 0.450759
2024-02-23 12:00:59,782 P3402379 INFO Evaluation @epoch 5 - batch 3668: 
2024-02-23 12:01:14,401 P3402379 INFO [Metrics] AUC: 0.809680
2024-02-23 12:01:14,405 P3402379 INFO Save best model: monitor(max)=0.809680
2024-02-23 12:01:14,596 P3402379 INFO ************ Epoch=5 end ************
2024-02-23 12:06:12,225 P3402379 INFO Train loss: 0.450590
2024-02-23 12:06:12,226 P3402379 INFO Evaluation @epoch 6 - batch 3668: 
2024-02-23 12:06:27,233 P3402379 INFO [Metrics] AUC: 0.809876
2024-02-23 12:06:27,234 P3402379 INFO Save best model: monitor(max)=0.809876
2024-02-23 12:06:27,415 P3402379 INFO ************ Epoch=6 end ************
2024-02-23 12:11:25,413 P3402379 INFO Train loss: 0.450475
2024-02-23 12:11:25,413 P3402379 INFO Evaluation @epoch 7 - batch 3668: 
2024-02-23 12:11:40,046 P3402379 INFO [Metrics] AUC: 0.809802
2024-02-23 12:11:40,048 P3402379 INFO Monitor(max)=0.809802 STOP!
2024-02-23 12:11:40,048 P3402379 INFO Reduce learning rate on plateau: 0.000100
2024-02-23 12:11:40,098 P3402379 INFO ************ Epoch=7 end ************
2024-02-23 12:16:36,465 P3402379 INFO Train loss: 0.440398
2024-02-23 12:16:36,466 P3402379 INFO Evaluation @epoch 8 - batch 3668: 
2024-02-23 12:16:51,101 P3402379 INFO [Metrics] AUC: 0.812577
2024-02-23 12:16:51,103 P3402379 INFO Save best model: monitor(max)=0.812577
2024-02-23 12:16:51,276 P3402379 INFO ************ Epoch=8 end ************
2024-02-23 12:21:47,033 P3402379 INFO Train loss: 0.436010
2024-02-23 12:21:47,034 P3402379 INFO Evaluation @epoch 9 - batch 3668: 
2024-02-23 12:22:02,056 P3402379 INFO [Metrics] AUC: 0.812951
2024-02-23 12:22:02,060 P3402379 INFO Save best model: monitor(max)=0.812951
2024-02-23 12:22:02,238 P3402379 INFO ************ Epoch=9 end ************
2024-02-23 12:26:56,632 P3402379 INFO Train loss: 0.433695
2024-02-23 12:26:56,632 P3402379 INFO Evaluation @epoch 10 - batch 3668: 
2024-02-23 12:27:11,256 P3402379 INFO [Metrics] AUC: 0.812880
2024-02-23 12:27:11,258 P3402379 INFO Monitor(max)=0.812880 STOP!
2024-02-23 12:27:11,258 P3402379 INFO Reduce learning rate on plateau: 0.000010
2024-02-23 12:27:11,308 P3402379 INFO ************ Epoch=10 end ************
2024-02-23 12:32:05,896 P3402379 INFO Train loss: 0.430003
2024-02-23 12:32:05,896 P3402379 INFO Evaluation @epoch 11 - batch 3668: 
2024-02-23 12:32:20,496 P3402379 INFO [Metrics] AUC: 0.812549
2024-02-23 12:32:20,497 P3402379 INFO Monitor(max)=0.812549 STOP!
2024-02-23 12:32:20,498 P3402379 INFO Reduce learning rate on plateau: 0.000001
2024-02-23 12:32:20,498 P3402379 INFO ********* Epoch==11 early stop *********
2024-02-23 12:32:20,560 P3402379 INFO Training finished.
2024-02-23 12:32:20,560 P3402379 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/criteo_x4_001_a5e05ce7/MaskNet_criteo_x4_001_018_ccc857cd.model
2024-02-23 12:32:20,644 P3402379 INFO ****** Validation evaluation ******
2024-02-23 12:32:36,461 P3402379 INFO [Metrics] AUC: 0.812951 - logloss: 0.439143
2024-02-23 12:32:36,559 P3402379 INFO ******** Test evaluation ********
2024-02-23 12:32:36,559 P3402379 INFO Loading datasets...
2024-02-23 12:32:41,178 P3402379 INFO Test samples: total/4584062, blocks/1
2024-02-23 12:32:41,178 P3402379 INFO Loading test data done.
2024-02-23 12:32:57,223 P3402379 INFO [Metrics] AUC: 0.813420 - logloss: 0.438748

```
