## AutoInt_avazu_x1

A hands-on guide to run the AutoInt model on the Avazu_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index

[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs) | [Revision History](#Revision-History)

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
  python: 3.6.4
  pytorch: 1.0.0
  pandas: 0.22.0
  numpy: 1.19.2
  scipy: 1.5.4
  sklearn: 0.22.1
  pyyaml: 5.4.1
  h5py: 2.8.0
  tqdm: 4.60.0
  fuxictr: 1.2.2
  ```

### Dataset

Dataset ID: [Avazu_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu#Avazu_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.2](https://github.com/reczoo/FuxiCTR/tree/v1.2.2) for this experiment. See the model code: [AutoInt](https://github.com/reczoo/FuxiCTR/blob/v1.2.2/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.2.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.2.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [fuxictr_version.py](./fuxictr_version.py#L3) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt_avazu_x1_tuner_config_03](./AutoInt_avazu_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd AutoInt_avazu_x1
   nohup python run_expid.py --config ./AutoInt_avazu_x1_tuner_config_03 --expid AutoInt_avazu_x1_001_af7937e6 --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

| AUC      | logloss  |
|:--------:|:--------:|
| 0.762373 | 0.367835 |

### Logs

```python
2022-06-26 11:15:13,911 P2572 INFO {
    "attention_dim": "128",
    "attention_layers": "4",
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "layer_norm": "True",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AutoInt",
    "model_id": "AutoInt_avazu_x1_001_af7937e6",
    "model_root": "./Avazu/AutoInt_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "1",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_hdf5": "True",
    "use_residual": "True",
    "use_scale": "True",
    "use_wide": "False",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-06-26 11:15:13,912 P2572 INFO Set up feature encoder...
2022-06-26 11:15:13,912 P2572 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-06-26 11:15:13,913 P2572 INFO Loading data...
2022-06-26 11:15:13,913 P2572 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-06-26 11:15:17,090 P2572 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-06-26 11:15:17,528 P2572 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-06-26 11:15:17,528 P2572 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-06-26 11:15:17,528 P2572 INFO Loading train data done.
2022-06-26 11:15:23,971 P2572 INFO Total number of parameters: 13142407.
2022-06-26 11:15:23,971 P2572 INFO Start training: 6910 batches/epoch
2022-06-26 11:15:23,971 P2572 INFO ************ Epoch=1 start ************
2022-06-26 11:34:04,397 P2572 INFO [Metrics] AUC: 0.739827 - logloss: 0.399477
2022-06-26 11:34:04,399 P2572 INFO Save best model: monitor(max): 0.739827
2022-06-26 11:34:04,457 P2572 INFO --- 6910/6910 batches finished ---
2022-06-26 11:34:04,506 P2572 INFO Train loss: 0.464762
2022-06-26 11:34:04,506 P2572 INFO ************ Epoch=1 end ************
2022-06-26 11:52:43,441 P2572 INFO [Metrics] AUC: 0.740997 - logloss: 0.399898
2022-06-26 11:52:43,443 P2572 INFO Save best model: monitor(max): 0.740997
2022-06-26 11:52:43,514 P2572 INFO --- 6910/6910 batches finished ---
2022-06-26 11:52:43,556 P2572 INFO Train loss: 0.480839
2022-06-26 11:52:43,557 P2572 INFO ************ Epoch=2 end ************
2022-06-26 12:11:21,118 P2572 INFO [Metrics] AUC: 0.740723 - logloss: 0.398973
2022-06-26 12:11:21,119 P2572 INFO Monitor(max) STOP: 0.740723 !
2022-06-26 12:11:21,120 P2572 INFO Reduce learning rate on plateau: 0.000100
2022-06-26 12:11:21,120 P2572 INFO --- 6910/6910 batches finished ---
2022-06-26 12:11:21,163 P2572 INFO Train loss: 0.501063
2022-06-26 12:11:21,163 P2572 INFO ************ Epoch=3 end ************
2022-06-26 12:29:56,816 P2572 INFO [Metrics] AUC: 0.744060 - logloss: 0.396996
2022-06-26 12:29:56,817 P2572 INFO Save best model: monitor(max): 0.744060
2022-06-26 12:29:56,889 P2572 INFO --- 6910/6910 batches finished ---
2022-06-26 12:29:56,930 P2572 INFO Train loss: 0.418401
2022-06-26 12:29:56,930 P2572 INFO ************ Epoch=4 end ************
2022-06-26 12:48:31,054 P2572 INFO [Metrics] AUC: 0.742162 - logloss: 0.398282
2022-06-26 12:48:31,056 P2572 INFO Monitor(max) STOP: 0.742162 !
2022-06-26 12:48:31,056 P2572 INFO Reduce learning rate on plateau: 0.000010
2022-06-26 12:48:31,056 P2572 INFO --- 6910/6910 batches finished ---
2022-06-26 12:48:31,099 P2572 INFO Train loss: 0.425549
2022-06-26 12:48:31,099 P2572 INFO ************ Epoch=5 end ************
2022-06-26 13:07:06,708 P2572 INFO [Metrics] AUC: 0.739204 - logloss: 0.400295
2022-06-26 13:07:06,710 P2572 INFO Monitor(max) STOP: 0.739204 !
2022-06-26 13:07:06,710 P2572 INFO Reduce learning rate on plateau: 0.000001
2022-06-26 13:07:06,710 P2572 INFO Early stopping at epoch=6
2022-06-26 13:07:06,710 P2572 INFO --- 6910/6910 batches finished ---
2022-06-26 13:07:06,761 P2572 INFO Train loss: 0.397311
2022-06-26 13:07:06,761 P2572 INFO Training finished.
2022-06-26 13:07:06,761 P2572 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/AutoInt_avazu_x1/avazu_x1_3fb65689/AutoInt_avazu_x1_001_af7937e6.model
2022-06-26 13:07:06,846 P2572 INFO ****** Validation evaluation ******
2022-06-26 13:07:44,123 P2572 INFO [Metrics] AUC: 0.744060 - logloss: 0.396996
2022-06-26 13:07:44,193 P2572 INFO ******** Test evaluation ********
2022-06-26 13:07:44,194 P2572 INFO Loading data...
2022-06-26 13:07:44,194 P2572 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-06-26 13:07:44,921 P2572 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-06-26 13:07:44,921 P2572 INFO Loading test data done.
2022-06-26 13:08:59,611 P2572 INFO [Metrics] AUC: 0.762373 - logloss: 0.367835
```

### Revision History

+ [Version 1](https://github.com/openbenchmark/BARS/tree/88d3a0faa4565e975141ae89a52d35d3a8b56eda/ctr_prediction/benchmarks/AutoInt/AutoInt_avazu_x1#autoint_avazu_x1): deprecated due to bug fix [#30](https://github.com/reczoo/FuxiCTR/issues/30) of FuxiCTR.
