## AutoInt+_avazu_x1

A hands-on guide to run the AutoInt model on the Avazu_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

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

We use [FuxiCTR-v1.2.2](https://github.com/xue-pai/FuxiCTR/tree/v1.2.2) for this experiment. See the model code: [AutoInt](https://github.com/xue-pai/FuxiCTR/blob/v1.2.2/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.2.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.2.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [fuxictr_version.py](./fuxictr_version.py#L3) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt+_avazu_x1_tuner_config_06](./AutoInt+_avazu_x1_tuner_config_06). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd AutoInt+_avazu_x1
   nohup python run_expid.py --config ./AutoInt+_avazu_x1_tuner_config_06 --expid AutoInt_avazu_x1_005_73d0b026 --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

| AUC      | logloss  |
|:--------:|:--------:|
| 0.764524 | 0.366815 |

### Logs

```python
2022-07-04 10:13:49,391 P90905 INFO {
    "attention_dim": "256",
    "attention_layers": "5",
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "4",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "layer_norm": "True",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AutoInt",
    "model_id": "AutoInt_avazu_x1_005_73d0b026",
    "model_root": "./Avazu/AutoInt_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_heads": "2",
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
    "verbose": "0",
    "version": "pytorch"
}
2022-07-04 10:13:49,392 P90905 INFO Set up feature encoder...
2022-07-04 10:13:49,392 P90905 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-07-04 10:13:49,392 P90905 INFO Loading data...
2022-07-04 10:13:49,393 P90905 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-07-04 10:13:51,652 P90905 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-07-04 10:13:51,980 P90905 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-07-04 10:13:51,980 P90905 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-07-04 10:13:51,980 P90905 INFO Loading train data done.
2022-07-04 10:13:56,869 P90905 INFO Total number of parameters: 14202856.
2022-07-04 10:13:56,869 P90905 INFO Start training: 6910 batches/epoch
2022-07-04 10:13:56,869 P90905 INFO ************ Epoch=1 start ************
2022-07-04 10:33:01,996 P90905 INFO [Metrics] AUC: 0.741912 - logloss: 0.398438
2022-07-04 10:33:01,999 P90905 INFO Save best model: monitor(max): 0.741912
2022-07-04 10:33:02,317 P90905 INFO --- 6910/6910 batches finished ---
2022-07-04 10:33:02,353 P90905 INFO Train loss: 0.436511
2022-07-04 10:33:02,354 P90905 INFO ************ Epoch=1 end ************
2022-07-04 10:52:05,913 P90905 INFO [Metrics] AUC: 0.743150 - logloss: 0.397991
2022-07-04 10:52:05,915 P90905 INFO Save best model: monitor(max): 0.743150
2022-07-04 10:52:05,993 P90905 INFO --- 6910/6910 batches finished ---
2022-07-04 10:52:06,033 P90905 INFO Train loss: 0.429546
2022-07-04 10:52:06,034 P90905 INFO ************ Epoch=2 end ************
2022-07-04 11:11:12,537 P90905 INFO [Metrics] AUC: 0.743018 - logloss: 0.397521
2022-07-04 11:11:12,539 P90905 INFO Monitor(max) STOP: 0.743018 !
2022-07-04 11:11:12,539 P90905 INFO Reduce learning rate on plateau: 0.000100
2022-07-04 11:11:12,539 P90905 INFO --- 6910/6910 batches finished ---
2022-07-04 11:11:12,585 P90905 INFO Train loss: 0.430475
2022-07-04 11:11:12,585 P90905 INFO ************ Epoch=3 end ************
2022-07-04 11:30:15,696 P90905 INFO [Metrics] AUC: 0.746459 - logloss: 0.396205
2022-07-04 11:30:15,699 P90905 INFO Save best model: monitor(max): 0.746459
2022-07-04 11:30:15,773 P90905 INFO --- 6910/6910 batches finished ---
2022-07-04 11:30:15,813 P90905 INFO Train loss: 0.406038
2022-07-04 11:30:15,814 P90905 INFO ************ Epoch=4 end ************
2022-07-04 11:49:18,884 P90905 INFO [Metrics] AUC: 0.745633 - logloss: 0.396847
2022-07-04 11:49:18,888 P90905 INFO Monitor(max) STOP: 0.745633 !
2022-07-04 11:49:18,888 P90905 INFO Reduce learning rate on plateau: 0.000010
2022-07-04 11:49:18,888 P90905 INFO --- 6910/6910 batches finished ---
2022-07-04 11:49:18,929 P90905 INFO Train loss: 0.405376
2022-07-04 11:49:18,929 P90905 INFO ************ Epoch=5 end ************
2022-07-04 12:08:21,645 P90905 INFO [Metrics] AUC: 0.741566 - logloss: 0.399525
2022-07-04 12:08:21,649 P90905 INFO Monitor(max) STOP: 0.741566 !
2022-07-04 12:08:21,649 P90905 INFO Reduce learning rate on plateau: 0.000001
2022-07-04 12:08:21,649 P90905 INFO Early stopping at epoch=6
2022-07-04 12:08:21,649 P90905 INFO --- 6910/6910 batches finished ---
2022-07-04 12:08:21,690 P90905 INFO Train loss: 0.392677
2022-07-04 12:08:21,690 P90905 INFO Training finished.
2022-07-04 12:08:21,691 P90905 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/AutoInt_avazu_x1/avazu_x1_3fb65689/AutoInt_avazu_x1_005_73d0b026.model
2022-07-04 12:08:29,038 P90905 INFO ****** Validation evaluation ******
2022-07-04 12:09:18,992 P90905 INFO [Metrics] AUC: 0.746459 - logloss: 0.396205
2022-07-04 12:09:19,059 P90905 INFO ******** Test evaluation ********
2022-07-04 12:09:19,060 P90905 INFO Loading data...
2022-07-04 12:09:19,060 P90905 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-07-04 12:09:19,865 P90905 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-07-04 12:09:19,866 P90905 INFO Loading test data done.
2022-07-04 12:11:00,175 P90905 INFO [Metrics] AUC: 0.764524 - logloss: 0.366815
```

Revision History

- [Version 1](https://github.com/openbenchmark/BARS/tree/88d3a0faa4565e975141ae89a52d35d3a8b56eda/ctr_prediction/benchmarks/AutoInt/AutoInt%2B_avazu_x1#autoint_avazu_x1): deprecated due to bug fix [#30](https://github.com/xue-pai/FuxiCTR/issues/30).
