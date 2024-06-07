## EDCN_avazu_x1

A hands-on guide to run the EDCN model on the Avazu_x1 dataset.

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

We use [FuxiCTR-v1.2.2](https://github.com/reczoo/FuxiCTR/tree/v1.2.2) for this experiment. See the model code: [EDCN](https://github.com/reczoo/FuxiCTR/blob/v1.2.2/fuxictr/pytorch/models/EDCN.py).

Running steps:

1. Download [FuxiCTR-v1.2.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.2.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [fuxictr_version.py](./fuxictr_version.py#L3) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [EDCN_avazu_x1_tuner_config_03](./EDCN_avazu_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd EDCN_avazu_x1
   nohup python run_expid.py --config ./EDCN_avazu_x1_tuner_config_03 --expid EDCN_avazu_x1_030_97df3f6c --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

| AUC      | logloss  |
|:--------:|:--------:|
| 0.765241 | 0.367039 |

### Logs

```python
2022-06-16 21:38:08,326 P55256 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "bridge_type": "hadamard_product",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "4",
    "hidden_activations": "ReLU",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "EDCN",
    "model_id": "EDCN_avazu_x1_030_97df3f6c",
    "model_root": "./Avazu/EDCN_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_cross_layers": "3",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "temperature": "1",
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_hdf5": "True",
    "use_regulation_module": "False",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-06-16 21:38:08,327 P55256 INFO Set up feature encoder...
2022-06-16 21:38:08,327 P55256 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-06-16 21:38:08,327 P55256 INFO Loading data...
2022-06-16 21:38:08,328 P55256 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-06-16 21:38:10,735 P55256 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-06-16 21:38:11,071 P55256 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-06-16 21:38:11,072 P55256 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-06-16 21:38:11,072 P55256 INFO Loading train data done.
2022-06-16 21:38:14,519 P55256 INFO Total number of parameters: 13136471.
2022-06-16 21:38:14,519 P55256 INFO Start training: 6910 batches/epoch
2022-06-16 21:38:14,519 P55256 INFO ************ Epoch=1 start ************
2022-06-16 21:50:51,069 P55256 INFO [Metrics] AUC: 0.738545 - logloss: 0.400676
2022-06-16 21:50:51,071 P55256 INFO Save best model: monitor(max): 0.738545
2022-06-16 21:50:51,327 P55256 INFO --- 6910/6910 batches finished ---
2022-06-16 21:50:51,370 P55256 INFO Train loss: 0.451488
2022-06-16 21:50:51,370 P55256 INFO ************ Epoch=1 end ************
2022-06-16 22:03:26,633 P55256 INFO [Metrics] AUC: 0.739187 - logloss: 0.401015
2022-06-16 22:03:26,636 P55256 INFO Save best model: monitor(max): 0.739187
2022-06-16 22:03:26,710 P55256 INFO --- 6910/6910 batches finished ---
2022-06-16 22:03:26,756 P55256 INFO Train loss: 0.443266
2022-06-16 22:03:26,756 P55256 INFO ************ Epoch=2 end ************
2022-06-16 22:15:56,200 P55256 INFO [Metrics] AUC: 0.738000 - logloss: 0.400357
2022-06-16 22:15:56,203 P55256 INFO Monitor(max) STOP: 0.738000 !
2022-06-16 22:15:56,204 P55256 INFO Reduce learning rate on plateau: 0.000100
2022-06-16 22:15:56,204 P55256 INFO --- 6910/6910 batches finished ---
2022-06-16 22:15:56,259 P55256 INFO Train loss: 0.443215
2022-06-16 22:15:56,259 P55256 INFO ************ Epoch=3 end ************
2022-06-16 22:28:27,706 P55256 INFO [Metrics] AUC: 0.746973 - logloss: 0.395593
2022-06-16 22:28:27,710 P55256 INFO Save best model: monitor(max): 0.746973
2022-06-16 22:28:27,783 P55256 INFO --- 6910/6910 batches finished ---
2022-06-16 22:28:27,846 P55256 INFO Train loss: 0.409797
2022-06-16 22:28:27,847 P55256 INFO ************ Epoch=4 end ************
2022-06-16 22:40:59,267 P55256 INFO [Metrics] AUC: 0.746601 - logloss: 0.395412
2022-06-16 22:40:59,270 P55256 INFO Monitor(max) STOP: 0.746601 !
2022-06-16 22:40:59,270 P55256 INFO Reduce learning rate on plateau: 0.000010
2022-06-16 22:40:59,270 P55256 INFO --- 6910/6910 batches finished ---
2022-06-16 22:40:59,318 P55256 INFO Train loss: 0.411322
2022-06-16 22:40:59,318 P55256 INFO ************ Epoch=5 end ************
2022-06-16 22:53:29,345 P55256 INFO [Metrics] AUC: 0.748368 - logloss: 0.395691
2022-06-16 22:53:29,349 P55256 INFO Save best model: monitor(max): 0.748368
2022-06-16 22:53:29,423 P55256 INFO --- 6910/6910 batches finished ---
2022-06-16 22:53:29,462 P55256 INFO Train loss: 0.396843
2022-06-16 22:53:29,462 P55256 INFO ************ Epoch=6 end ************
2022-06-16 23:05:57,854 P55256 INFO [Metrics] AUC: 0.746022 - logloss: 0.397059
2022-06-16 23:05:57,857 P55256 INFO Monitor(max) STOP: 0.746022 !
2022-06-16 23:05:57,857 P55256 INFO Reduce learning rate on plateau: 0.000001
2022-06-16 23:05:57,857 P55256 INFO --- 6910/6910 batches finished ---
2022-06-16 23:05:57,909 P55256 INFO Train loss: 0.393571
2022-06-16 23:05:57,909 P55256 INFO ************ Epoch=7 end ************
2022-06-16 23:11:51,514 P55256 INFO [Metrics] AUC: 0.743069 - logloss: 0.399253
2022-06-16 23:11:51,518 P55256 INFO Monitor(max) STOP: 0.743069 !
2022-06-16 23:11:51,518 P55256 INFO Reduce learning rate on plateau: 0.000001
2022-06-16 23:11:51,518 P55256 INFO Early stopping at epoch=8
2022-06-16 23:11:51,518 P55256 INFO --- 6910/6910 batches finished ---
2022-06-16 23:11:51,565 P55256 INFO Train loss: 0.385598
2022-06-16 23:11:51,565 P55256 INFO Training finished.
2022-06-16 23:11:51,565 P55256 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/EDCN_avazu_x1/avazu_x1_3fb65689/EDCN_avazu_x1_030_97df3f6c.model
2022-06-16 23:11:56,137 P55256 INFO ****** Validation evaluation ******
2022-06-16 23:12:07,234 P55256 INFO [Metrics] AUC: 0.748368 - logloss: 0.395691
2022-06-16 23:12:07,306 P55256 INFO ******** Test evaluation ********
2022-06-16 23:12:07,306 P55256 INFO Loading data...
2022-06-16 23:12:07,306 P55256 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-06-16 23:12:08,150 P55256 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-06-16 23:12:08,150 P55256 INFO Loading test data done.
2022-06-16 23:12:33,237 P55256 INFO [Metrics] AUC: 0.765241 - logloss: 0.367039
```

### Revision History

- [Version 1](https://github.com/openbenchmark/BARS/tree/88d3a0faa4565e975141ae89a52d35d3a8b56eda/ctr_prediction/benchmarks/EDCN/EDCN_avazu_x1): deprecated due to bug fix [#29](https://github.com/reczoo/FuxiCTR/issues/29) of FuxiCTR.
