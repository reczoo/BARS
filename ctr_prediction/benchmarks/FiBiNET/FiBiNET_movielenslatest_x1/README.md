## FiBiNET_movielenslatest_x1

A hands-on guide to run the FiBiNET model on the Movielenslatest_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

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
  fuxictr: 1.1.0
  ```

### Dataset
Dataset ID: [Movielenslatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Movielenslatest/README.md#Movielenslatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [FiBiNET](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FiBiNET.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FiBiNET_movielenslatest_x1_tuner_config_03](./FiBiNET_movielenslatest_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FiBiNET_movielenslatest_x1
    nohup python run_expid.py --config ./FiBiNET_movielenslatest_x1_tuner_config_03 --expid FiBiNET_movielenslatest_x1_017_24eff5cf --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.957579 | 0.251849  |


### Logs
```python
2022-02-01 18:09:40,871 P6373 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "bilinear_type": "field_all",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FiBiNET",
    "model_id": "FiBiNET_movielenslatest_x1_017_24eff5cf",
    "model_root": "./Movielens/FiBiNET_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "0.3",
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
2022-02-01 18:09:40,873 P6373 INFO Set up feature encoder...
2022-02-01 18:09:40,873 P6373 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-02-01 18:09:40,873 P6373 INFO Loading data...
2022-02-01 18:09:40,877 P6373 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-02-01 18:09:40,910 P6373 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-02-01 18:09:40,921 P6373 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-02-01 18:09:40,921 P6373 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-02-01 18:09:40,921 P6373 INFO Loading train data done.
2022-02-01 18:09:46,535 P6373 INFO Total number of parameters: 1340790.
2022-02-01 18:09:46,536 P6373 INFO Start training: 343 batches/epoch
2022-02-01 18:09:46,536 P6373 INFO ************ Epoch=1 start ************
2022-02-01 18:10:09,804 P6373 INFO [Metrics] AUC: 0.500000 - logloss: 10.727955
2022-02-01 18:10:09,805 P6373 INFO Save best model: monitor(max): 0.500000
2022-02-01 18:10:09,845 P6373 INFO --- 343/343 batches finished ---
2022-02-01 18:10:09,943 P6373 INFO Train loss: 0.492610
2022-02-01 18:10:09,943 P6373 INFO ************ Epoch=1 end ************
2022-02-01 18:10:32,108 P6373 INFO [Metrics] AUC: 0.692635 - logloss: 4.132846
2022-02-01 18:10:32,109 P6373 INFO Save best model: monitor(max): 0.692635
2022-02-01 18:10:32,122 P6373 INFO --- 343/343 batches finished ---
2022-02-01 18:10:32,238 P6373 INFO Train loss: 0.469429
2022-02-01 18:10:32,238 P6373 INFO ************ Epoch=2 end ************
2022-02-01 18:10:54,502 P6373 INFO [Metrics] AUC: 0.782614 - logloss: 6.399087
2022-02-01 18:10:54,503 P6373 INFO Save best model: monitor(max): 0.782614
2022-02-01 18:10:54,527 P6373 INFO --- 343/343 batches finished ---
2022-02-01 18:10:54,681 P6373 INFO Train loss: 0.474188
2022-02-01 18:10:54,682 P6373 INFO ************ Epoch=3 end ************
2022-02-01 18:11:18,270 P6373 INFO [Metrics] AUC: 0.765277 - logloss: 3.527854
2022-02-01 18:11:18,271 P6373 INFO Monitor(max) STOP: 0.765277 !
2022-02-01 18:11:18,271 P6373 INFO Reduce learning rate on plateau: 0.000100
2022-02-01 18:11:18,271 P6373 INFO --- 343/343 batches finished ---
2022-02-01 18:11:18,403 P6373 INFO Train loss: 0.473872
2022-02-01 18:11:18,403 P6373 INFO ************ Epoch=4 end ************
2022-02-01 18:11:41,059 P6373 INFO [Metrics] AUC: 0.858960 - logloss: 3.822789
2022-02-01 18:11:41,059 P6373 INFO Save best model: monitor(max): 0.858960
2022-02-01 18:11:41,095 P6373 INFO --- 343/343 batches finished ---
2022-02-01 18:11:41,308 P6373 INFO Train loss: 0.350077
2022-02-01 18:11:41,308 P6373 INFO ************ Epoch=5 end ************
2022-02-01 18:12:11,573 P6373 INFO [Metrics] AUC: 0.817997 - logloss: 5.782039
2022-02-01 18:12:11,574 P6373 INFO Monitor(max) STOP: 0.817997 !
2022-02-01 18:12:11,574 P6373 INFO Reduce learning rate on plateau: 0.000010
2022-02-01 18:12:11,574 P6373 INFO --- 343/343 batches finished ---
2022-02-01 18:12:11,748 P6373 INFO Train loss: 0.280009
2022-02-01 18:12:11,748 P6373 INFO ************ Epoch=6 end ************
2022-02-01 18:12:41,542 P6373 INFO [Metrics] AUC: 0.939333 - logloss: 0.284040
2022-02-01 18:12:41,552 P6373 INFO Save best model: monitor(max): 0.939333
2022-02-01 18:12:41,573 P6373 INFO --- 343/343 batches finished ---
2022-02-01 18:12:41,713 P6373 INFO Train loss: 0.210929
2022-02-01 18:12:41,714 P6373 INFO ************ Epoch=7 end ************
2022-02-01 18:13:11,080 P6373 INFO [Metrics] AUC: 0.939179 - logloss: 0.356365
2022-02-01 18:13:11,081 P6373 INFO Monitor(max) STOP: 0.939179 !
2022-02-01 18:13:11,081 P6373 INFO Reduce learning rate on plateau: 0.000001
2022-02-01 18:13:11,081 P6373 INFO --- 343/343 batches finished ---
2022-02-01 18:13:11,283 P6373 INFO Train loss: 0.183743
2022-02-01 18:13:11,283 P6373 INFO ************ Epoch=8 end ************
2022-02-01 18:13:41,974 P6373 INFO [Metrics] AUC: 0.957717 - logloss: 0.247564
2022-02-01 18:13:41,974 P6373 INFO Save best model: monitor(max): 0.957717
2022-02-01 18:13:41,995 P6373 INFO --- 343/343 batches finished ---
2022-02-01 18:13:42,244 P6373 INFO Train loss: 0.164272
2022-02-01 18:13:42,244 P6373 INFO ************ Epoch=9 end ************
2022-02-01 18:14:09,800 P6373 INFO [Metrics] AUC: 0.957429 - logloss: 0.253700
2022-02-01 18:14:09,805 P6373 INFO Monitor(max) STOP: 0.957429 !
2022-02-01 18:14:09,806 P6373 INFO Reduce learning rate on plateau: 0.000001
2022-02-01 18:14:09,806 P6373 INFO --- 343/343 batches finished ---
2022-02-01 18:14:09,948 P6373 INFO Train loss: 0.162201
2022-02-01 18:14:09,948 P6373 INFO ************ Epoch=10 end ************
2022-02-01 18:14:37,679 P6373 INFO [Metrics] AUC: 0.958005 - logloss: 0.250501
2022-02-01 18:14:37,692 P6373 INFO Save best model: monitor(max): 0.958005
2022-02-01 18:14:37,732 P6373 INFO --- 343/343 batches finished ---
2022-02-01 18:14:37,907 P6373 INFO Train loss: 0.160276
2022-02-01 18:14:37,907 P6373 INFO ************ Epoch=11 end ************
2022-02-01 18:15:09,209 P6373 INFO [Metrics] AUC: 0.957578 - logloss: 0.260636
2022-02-01 18:15:09,209 P6373 INFO Monitor(max) STOP: 0.957578 !
2022-02-01 18:15:09,209 P6373 INFO Reduce learning rate on plateau: 0.000001
2022-02-01 18:15:09,210 P6373 INFO --- 343/343 batches finished ---
2022-02-01 18:15:09,466 P6373 INFO Train loss: 0.158480
2022-02-01 18:15:09,466 P6373 INFO ************ Epoch=12 end ************
2022-02-01 18:15:40,138 P6373 INFO [Metrics] AUC: 0.957716 - logloss: 0.262977
2022-02-01 18:15:40,139 P6373 INFO Monitor(max) STOP: 0.957716 !
2022-02-01 18:15:40,139 P6373 INFO Reduce learning rate on plateau: 0.000001
2022-02-01 18:15:40,149 P6373 INFO Early stopping at epoch=13
2022-02-01 18:15:40,149 P6373 INFO --- 343/343 batches finished ---
2022-02-01 18:15:40,394 P6373 INFO Train loss: 0.156616
2022-02-01 18:15:40,394 P6373 INFO Training finished.
2022-02-01 18:15:40,394 P6373 INFO Load best model: /home/XXX/benchmarks/Movielens/FiBiNET_movielenslatest_x1/movielenslatest_x1_cd32d937/FiBiNET_movielenslatest_x1_017_24eff5cf.model
2022-02-01 18:15:49,357 P6373 INFO ****** Validation evaluation ******
2022-02-01 18:15:56,006 P6373 INFO [Metrics] AUC: 0.958005 - logloss: 0.250501
2022-02-01 18:15:56,213 P6373 INFO ******** Test evaluation ********
2022-02-01 18:15:56,214 P6373 INFO Loading data...
2022-02-01 18:15:56,214 P6373 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-02-01 18:15:56,234 P6373 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-02-01 18:15:56,234 P6373 INFO Loading test data done.
2022-02-01 18:16:00,063 P6373 INFO [Metrics] AUC: 0.957579 - logloss: 0.251849

```
