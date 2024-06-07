## AOANet_movielenslatest_x1

A hands-on guide to run the AOANet model on the MovielensLatest_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

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
  fuxictr: 1.2.1
  ```

### Dataset
Dataset ID: [MovielensLatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/MovieLens#MovielensLatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.1](https://github.com/reczoo/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [AOANet](https://github.com/reczoo/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/AOANet.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AOANet_movielenslatest_x1_tuner_config_01](./AOANet_movielenslatest_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AOANet_movielenslatest_x1
    nohup python run_expid.py --config ./AOANet_movielenslatest_x1_tuner_config_01 --expid AOANet_movielenslatest_x1_031_bbf8c17a --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.969401 | 0.210458  |


### Logs
```python
2022-05-31 18:46:05,745 P16920 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "dnn_hidden_activations": "ReLU",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AOANet",
    "model_id": "AOANet_movielenslatest_x1_031_bbf8c17a",
    "model_root": "./Movielens/AOANet_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_interaction_layers": "3",
    "num_subspaces": "2",
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
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-05-31 18:46:05,746 P16920 INFO Set up feature encoder...
2022-05-31 18:46:05,746 P16920 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-05-31 18:46:05,746 P16920 INFO Loading data...
2022-05-31 18:46:05,749 P16920 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-05-31 18:46:05,777 P16920 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-05-31 18:46:05,785 P16920 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-05-31 18:46:05,785 P16920 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-05-31 18:46:05,785 P16920 INFO Loading train data done.
2022-05-31 18:46:09,635 P16920 INFO Total number of parameters: 1239113.
2022-05-31 18:46:09,635 P16920 INFO Start training: 343 batches/epoch
2022-05-31 18:46:09,636 P16920 INFO ************ Epoch=1 start ************
2022-05-31 18:46:41,510 P16920 INFO [Metrics] AUC: 0.933901 - logloss: 0.292700
2022-05-31 18:46:41,511 P16920 INFO Save best model: monitor(max): 0.933901
2022-05-31 18:46:41,520 P16920 INFO --- 343/343 batches finished ---
2022-05-31 18:46:41,559 P16920 INFO Train loss: 0.377147
2022-05-31 18:46:41,559 P16920 INFO ************ Epoch=1 end ************
2022-05-31 18:47:13,297 P16920 INFO [Metrics] AUC: 0.944238 - logloss: 0.272118
2022-05-31 18:47:13,299 P16920 INFO Save best model: monitor(max): 0.944238
2022-05-31 18:47:13,310 P16920 INFO --- 343/343 batches finished ---
2022-05-31 18:47:13,361 P16920 INFO Train loss: 0.361309
2022-05-31 18:47:13,361 P16920 INFO ************ Epoch=2 end ************
2022-05-31 18:47:45,052 P16920 INFO [Metrics] AUC: 0.948313 - logloss: 0.261492
2022-05-31 18:47:45,053 P16920 INFO Save best model: monitor(max): 0.948313
2022-05-31 18:47:45,064 P16920 INFO --- 343/343 batches finished ---
2022-05-31 18:47:45,112 P16920 INFO Train loss: 0.362418
2022-05-31 18:47:45,112 P16920 INFO ************ Epoch=3 end ************
2022-05-31 18:48:16,983 P16920 INFO [Metrics] AUC: 0.950069 - logloss: 0.250828
2022-05-31 18:48:16,984 P16920 INFO Save best model: monitor(max): 0.950069
2022-05-31 18:48:16,995 P16920 INFO --- 343/343 batches finished ---
2022-05-31 18:48:17,037 P16920 INFO Train loss: 0.365230
2022-05-31 18:48:17,037 P16920 INFO ************ Epoch=4 end ************
2022-05-31 18:48:48,797 P16920 INFO [Metrics] AUC: 0.951998 - logloss: 0.246179
2022-05-31 18:48:48,798 P16920 INFO Save best model: monitor(max): 0.951998
2022-05-31 18:48:48,809 P16920 INFO --- 343/343 batches finished ---
2022-05-31 18:48:48,882 P16920 INFO Train loss: 0.368801
2022-05-31 18:48:48,882 P16920 INFO ************ Epoch=5 end ************
2022-05-31 18:49:16,351 P16920 INFO [Metrics] AUC: 0.952251 - logloss: 0.244044
2022-05-31 18:49:16,352 P16920 INFO Save best model: monitor(max): 0.952251
2022-05-31 18:49:16,362 P16920 INFO --- 343/343 batches finished ---
2022-05-31 18:49:16,412 P16920 INFO Train loss: 0.371365
2022-05-31 18:49:16,412 P16920 INFO ************ Epoch=6 end ************
2022-05-31 18:49:48,759 P16920 INFO [Metrics] AUC: 0.953469 - logloss: 0.241305
2022-05-31 18:49:48,760 P16920 INFO Save best model: monitor(max): 0.953469
2022-05-31 18:49:48,768 P16920 INFO --- 343/343 batches finished ---
2022-05-31 18:49:48,823 P16920 INFO Train loss: 0.373331
2022-05-31 18:49:48,823 P16920 INFO ************ Epoch=7 end ************
2022-05-31 18:50:21,229 P16920 INFO [Metrics] AUC: 0.953871 - logloss: 0.240481
2022-05-31 18:50:21,230 P16920 INFO Save best model: monitor(max): 0.953871
2022-05-31 18:50:21,238 P16920 INFO --- 343/343 batches finished ---
2022-05-31 18:50:21,292 P16920 INFO Train loss: 0.374384
2022-05-31 18:50:21,292 P16920 INFO ************ Epoch=8 end ************
2022-05-31 18:50:50,300 P16920 INFO [Metrics] AUC: 0.954370 - logloss: 0.240454
2022-05-31 18:50:50,301 P16920 INFO Save best model: monitor(max): 0.954370
2022-05-31 18:50:50,313 P16920 INFO --- 343/343 batches finished ---
2022-05-31 18:50:50,372 P16920 INFO Train loss: 0.374569
2022-05-31 18:50:50,372 P16920 INFO ************ Epoch=9 end ************
2022-05-31 18:51:26,099 P16920 INFO [Metrics] AUC: 0.954567 - logloss: 0.239315
2022-05-31 18:51:26,100 P16920 INFO Save best model: monitor(max): 0.954567
2022-05-31 18:51:26,110 P16920 INFO --- 343/343 batches finished ---
2022-05-31 18:51:26,148 P16920 INFO Train loss: 0.374840
2022-05-31 18:51:26,148 P16920 INFO ************ Epoch=10 end ************
2022-05-31 18:52:01,834 P16920 INFO [Metrics] AUC: 0.955103 - logloss: 0.238806
2022-05-31 18:52:01,836 P16920 INFO Save best model: monitor(max): 0.955103
2022-05-31 18:52:01,847 P16920 INFO --- 343/343 batches finished ---
2022-05-31 18:52:01,921 P16920 INFO Train loss: 0.375471
2022-05-31 18:52:01,922 P16920 INFO ************ Epoch=11 end ************
2022-05-31 18:52:37,467 P16920 INFO [Metrics] AUC: 0.955896 - logloss: 0.235422
2022-05-31 18:52:37,468 P16920 INFO Save best model: monitor(max): 0.955896
2022-05-31 18:52:37,478 P16920 INFO --- 343/343 batches finished ---
2022-05-31 18:52:37,519 P16920 INFO Train loss: 0.376710
2022-05-31 18:52:37,519 P16920 INFO ************ Epoch=12 end ************
2022-05-31 18:53:13,017 P16920 INFO [Metrics] AUC: 0.956460 - logloss: 0.236233
2022-05-31 18:53:13,017 P16920 INFO Save best model: monitor(max): 0.956460
2022-05-31 18:53:13,025 P16920 INFO --- 343/343 batches finished ---
2022-05-31 18:53:13,080 P16920 INFO Train loss: 0.375659
2022-05-31 18:53:13,080 P16920 INFO ************ Epoch=13 end ************
2022-05-31 18:53:48,775 P16920 INFO [Metrics] AUC: 0.956391 - logloss: 0.232838
2022-05-31 18:53:48,776 P16920 INFO Monitor(max) STOP: 0.956391 !
2022-05-31 18:53:48,777 P16920 INFO Reduce learning rate on plateau: 0.000100
2022-05-31 18:53:48,777 P16920 INFO --- 343/343 batches finished ---
2022-05-31 18:53:48,819 P16920 INFO Train loss: 0.376075
2022-05-31 18:53:48,819 P16920 INFO ************ Epoch=14 end ************
2022-05-31 18:54:24,283 P16920 INFO [Metrics] AUC: 0.967931 - logloss: 0.203813
2022-05-31 18:54:24,283 P16920 INFO Save best model: monitor(max): 0.967931
2022-05-31 18:54:24,291 P16920 INFO --- 343/343 batches finished ---
2022-05-31 18:54:24,341 P16920 INFO Train loss: 0.276614
2022-05-31 18:54:24,341 P16920 INFO ************ Epoch=15 end ************
2022-05-31 18:55:00,032 P16920 INFO [Metrics] AUC: 0.969771 - logloss: 0.208504
2022-05-31 18:55:00,033 P16920 INFO Save best model: monitor(max): 0.969771
2022-05-31 18:55:00,043 P16920 INFO --- 343/343 batches finished ---
2022-05-31 18:55:00,095 P16920 INFO Train loss: 0.191585
2022-05-31 18:55:00,096 P16920 INFO ************ Epoch=16 end ************
2022-05-31 18:55:31,185 P16920 INFO [Metrics] AUC: 0.968663 - logloss: 0.228933
2022-05-31 18:55:31,186 P16920 INFO Monitor(max) STOP: 0.968663 !
2022-05-31 18:55:31,186 P16920 INFO Reduce learning rate on plateau: 0.000010
2022-05-31 18:55:31,186 P16920 INFO --- 343/343 batches finished ---
2022-05-31 18:55:31,235 P16920 INFO Train loss: 0.148550
2022-05-31 18:55:31,236 P16920 INFO ************ Epoch=17 end ************
2022-05-31 18:55:57,231 P16920 INFO [Metrics] AUC: 0.968163 - logloss: 0.243116
2022-05-31 18:55:57,232 P16920 INFO Monitor(max) STOP: 0.968163 !
2022-05-31 18:55:57,232 P16920 INFO Reduce learning rate on plateau: 0.000001
2022-05-31 18:55:57,232 P16920 INFO Early stopping at epoch=18
2022-05-31 18:55:57,233 P16920 INFO --- 343/343 batches finished ---
2022-05-31 18:55:57,330 P16920 INFO Train loss: 0.116393
2022-05-31 18:55:57,330 P16920 INFO Training finished.
2022-05-31 18:55:57,331 P16920 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Movielens/AOANet_movielenslatest_x1/movielenslatest_x1_cd32d937/AOANet_movielenslatest_x1_031_bbf8c17a.model
2022-05-31 18:55:57,354 P16920 INFO ****** Validation evaluation ******
2022-05-31 18:55:59,102 P16920 INFO [Metrics] AUC: 0.969771 - logloss: 0.208504
2022-05-31 18:55:59,149 P16920 INFO ******** Test evaluation ********
2022-05-31 18:55:59,150 P16920 INFO Loading data...
2022-05-31 18:55:59,150 P16920 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-05-31 18:55:59,156 P16920 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-05-31 18:55:59,156 P16920 INFO Loading test data done.
2022-05-31 18:56:00,022 P16920 INFO [Metrics] AUC: 0.969401 - logloss: 0.210458

```
