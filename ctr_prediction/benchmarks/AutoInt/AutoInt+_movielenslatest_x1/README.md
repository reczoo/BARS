## AutoInt+_movielenslatest_x1

A hands-on guide to run the AutoInt model on the Movielenslatest_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

### Index

[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs) | [Revision History](#Revision-History)

### Environments

+ Hardware
  
  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
  GPU: Tesla P100 16G
  RAM: 755G
  ```

+ Software
  
  ```python
  CUDA: 11.4
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  fuxictr: 1.2.2
  ```

### Dataset

Dataset ID: [Movielenslatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Movielenslatest#Movielenslatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.2](https://github.com/xue-pai/FuxiCTR/tree/v1.2.2) for this experiment. See the model code: [AutoInt](https://github.com/xue-pai/FuxiCTR/blob/v1.2.2/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.2.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.2.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [fuxictr_version.py](./fuxictr_version.py#L3) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Movielenslatest/Movielenslatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt+_movielenslatest_x1_tuner_config_08](./AutoInt+_movielenslatest_x1_tuner_config_08). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd AutoInt_movielenslatest_x1
   nohup python run_expid.py --config ./AutoInt+_movielenslatest_x1_tuner_config_08 --expid AutoInt_movielenslatest_x1_006_a5e56596 --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

| AUC      | logloss  |
|:--------:|:--------:|
| 0.969188 | 0.214772 |

### Logs

```python
2022-07-03 22:29:05,838 P13120 INFO {
    "attention_dim": "128",
    "attention_layers": "2",
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "layer_norm": "True",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AutoInt",
    "model_id": "AutoInt_movielenslatest_x1_006_a5e56596",
    "model_root": "./Movielens/AutoInt_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_heads": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Movielens/MovielensLatest_x1/test.csv",
    "train_data": "../data/Movielens/MovielensLatest_x1/train.csv",
    "use_hdf5": "True",
    "use_residual": "True",
    "use_scale": "True",
    "use_wide": "True",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-07-03 22:29:05,839 P13120 INFO Set up feature encoder...
2022-07-03 22:29:05,839 P13120 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-07-03 22:29:05,839 P13120 INFO Loading data...
2022-07-03 22:29:05,841 P13120 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-07-03 22:29:05,867 P13120 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-07-03 22:29:05,875 P13120 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-07-03 22:29:05,875 P13120 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-07-03 22:29:05,875 P13120 INFO Loading train data done.
2022-07-03 22:29:10,016 P13120 INFO Total number of parameters: 1383799.
2022-07-03 22:29:10,017 P13120 INFO Start training: 343 batches/epoch
2022-07-03 22:29:10,017 P13120 INFO ************ Epoch=1 start ************
2022-07-03 22:30:13,019 P13120 INFO [Metrics] AUC: 0.937663 - logloss: 0.284440
2022-07-03 22:30:13,020 P13120 INFO Save best model: monitor(max): 0.937663
2022-07-03 22:30:13,028 P13120 INFO --- 343/343 batches finished ---
2022-07-03 22:30:13,058 P13120 INFO Train loss: 0.385147
2022-07-03 22:30:13,058 P13120 INFO ************ Epoch=1 end ************
2022-07-03 22:31:18,612 P13120 INFO [Metrics] AUC: 0.946182 - logloss: 0.287991
2022-07-03 22:31:18,613 P13120 INFO Save best model: monitor(max): 0.946182
2022-07-03 22:31:18,622 P13120 INFO --- 343/343 batches finished ---
2022-07-03 22:31:18,655 P13120 INFO Train loss: 0.368093
2022-07-03 22:31:18,655 P13120 INFO ************ Epoch=2 end ************
2022-07-03 22:32:24,765 P13120 INFO [Metrics] AUC: 0.950263 - logloss: 0.250355
2022-07-03 22:32:24,766 P13120 INFO Save best model: monitor(max): 0.950263
2022-07-03 22:32:24,775 P13120 INFO --- 343/343 batches finished ---
2022-07-03 22:32:24,805 P13120 INFO Train loss: 0.369451
2022-07-03 22:32:24,805 P13120 INFO ************ Epoch=3 end ************
2022-07-03 22:33:30,191 P13120 INFO [Metrics] AUC: 0.951988 - logloss: 0.253689
2022-07-03 22:33:30,191 P13120 INFO Save best model: monitor(max): 0.951988
2022-07-03 22:33:30,200 P13120 INFO --- 343/343 batches finished ---
2022-07-03 22:33:30,236 P13120 INFO Train loss: 0.372838
2022-07-03 22:33:30,236 P13120 INFO ************ Epoch=4 end ************
2022-07-03 22:34:34,897 P13120 INFO [Metrics] AUC: 0.953812 - logloss: 0.242734
2022-07-03 22:34:34,897 P13120 INFO Save best model: monitor(max): 0.953812
2022-07-03 22:34:34,907 P13120 INFO --- 343/343 batches finished ---
2022-07-03 22:34:34,938 P13120 INFO Train loss: 0.375296
2022-07-03 22:34:34,938 P13120 INFO ************ Epoch=5 end ************
2022-07-03 22:35:41,128 P13120 INFO [Metrics] AUC: 0.954291 - logloss: 0.239010
2022-07-03 22:35:41,129 P13120 INFO Save best model: monitor(max): 0.954291
2022-07-03 22:35:41,138 P13120 INFO --- 343/343 batches finished ---
2022-07-03 22:35:41,166 P13120 INFO Train loss: 0.376394
2022-07-03 22:35:41,167 P13120 INFO ************ Epoch=6 end ************
2022-07-03 22:36:47,508 P13120 INFO [Metrics] AUC: 0.954845 - logloss: 0.238665
2022-07-03 22:36:47,509 P13120 INFO Save best model: monitor(max): 0.954845
2022-07-03 22:36:47,518 P13120 INFO --- 343/343 batches finished ---
2022-07-03 22:36:47,556 P13120 INFO Train loss: 0.377731
2022-07-03 22:36:47,557 P13120 INFO ************ Epoch=7 end ************
2022-07-03 22:37:52,600 P13120 INFO [Metrics] AUC: 0.954765 - logloss: 0.238342
2022-07-03 22:37:52,600 P13120 INFO Monitor(max) STOP: 0.954765 !
2022-07-03 22:37:52,600 P13120 INFO Reduce learning rate on plateau: 0.000100
2022-07-03 22:37:52,600 P13120 INFO --- 343/343 batches finished ---
2022-07-03 22:37:52,626 P13120 INFO Train loss: 0.378794
2022-07-03 22:37:52,627 P13120 INFO ************ Epoch=8 end ************
2022-07-03 22:38:57,616 P13120 INFO [Metrics] AUC: 0.967635 - logloss: 0.206114
2022-07-03 22:38:57,616 P13120 INFO Save best model: monitor(max): 0.967635
2022-07-03 22:38:57,625 P13120 INFO --- 343/343 batches finished ---
2022-07-03 22:38:57,656 P13120 INFO Train loss: 0.274431
2022-07-03 22:38:57,656 P13120 INFO ************ Epoch=9 end ************
2022-07-03 22:40:02,269 P13120 INFO [Metrics] AUC: 0.969328 - logloss: 0.214030
2022-07-03 22:40:02,270 P13120 INFO Save best model: monitor(max): 0.969328
2022-07-03 22:40:02,279 P13120 INFO --- 343/343 batches finished ---
2022-07-03 22:40:02,311 P13120 INFO Train loss: 0.182387
2022-07-03 22:40:02,311 P13120 INFO ************ Epoch=10 end ************
2022-07-03 22:41:08,341 P13120 INFO [Metrics] AUC: 0.967850 - logloss: 0.240925
2022-07-03 22:41:08,342 P13120 INFO Monitor(max) STOP: 0.967850 !
2022-07-03 22:41:08,342 P13120 INFO Reduce learning rate on plateau: 0.000010
2022-07-03 22:41:08,342 P13120 INFO --- 343/343 batches finished ---
2022-07-03 22:41:08,373 P13120 INFO Train loss: 0.134998
2022-07-03 22:41:08,373 P13120 INFO ************ Epoch=11 end ************
2022-07-03 22:42:13,252 P13120 INFO [Metrics] AUC: 0.967451 - logloss: 0.254764
2022-07-03 22:42:13,252 P13120 INFO Monitor(max) STOP: 0.967451 !
2022-07-03 22:42:13,252 P13120 INFO Reduce learning rate on plateau: 0.000001
2022-07-03 22:42:13,253 P13120 INFO Early stopping at epoch=12
2022-07-03 22:42:13,253 P13120 INFO --- 343/343 batches finished ---
2022-07-03 22:42:13,282 P13120 INFO Train loss: 0.103227
2022-07-03 22:42:13,282 P13120 INFO Training finished.
2022-07-03 22:42:13,282 P13120 INFO Load best model: /home/FuxiCTR/benchmarks/Movielens/AutoInt_movielenslatest_x1/movielenslatest_x1_cd32d937/AutoInt_movielenslatest_x1_006_a5e56596.model
2022-07-03 22:42:18,179 P13120 INFO ****** Validation evaluation ******
2022-07-03 22:42:20,947 P13120 INFO [Metrics] AUC: 0.969328 - logloss: 0.214030
2022-07-03 22:42:20,984 P13120 INFO ******** Test evaluation ********
2022-07-03 22:42:20,984 P13120 INFO Loading data...
2022-07-03 22:42:20,984 P13120 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-07-03 22:42:20,988 P13120 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-07-03 22:42:20,988 P13120 INFO Loading test data done.
2022-07-03 22:42:22,481 P13120 INFO [Metrics] AUC: 0.969188 - logloss: 0.214772
```

### Revision History

- [Version 1](https://github.com/openbenchmark/BARS/tree/88d3a0faa4565e975141ae89a52d35d3a8b56eda/ctr_prediction/benchmarks/AutoInt/AutoInt%2B_movielenslatest_x1#autoint_movielenslatest_x1): deprecated due to bug fix [#30](https://github.com/xue-pai/FuxiCTR/issues/30) of FuxiCTR.
