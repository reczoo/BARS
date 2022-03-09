## NFM_movielenslatest_x1

A hands-on guide to run the NFM model on the Movielenslatest_x1 dataset.

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
Dataset ID: [Movielenslatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/MovieLens/README.md#Movielenslatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [NFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/NFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [NFM_movielenslatest_x1_tuner_config_01](./NFM_movielenslatest_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd NFM_movielenslatest_x1
    nohup python run_expid.py --config ./NFM_movielenslatest_x1_tuner_config_01 --expid NFM_movielenslatest_x1_010_f3d546bc --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.949597 | 0.266413  |


### Logs
```python
2022-01-28 18:55:58,547 P19230 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.0001",
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
    "model": "NFM",
    "model_id": "NFM_movielenslatest_x1_010_f3d546bc",
    "model_root": "./Movielens/NFM_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
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
2022-01-28 18:55:58,548 P19230 INFO Set up feature encoder...
2022-01-28 18:55:58,548 P19230 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-01-28 18:55:58,548 P19230 INFO Loading data...
2022-01-28 18:55:58,550 P19230 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-01-28 18:55:58,581 P19230 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-01-28 18:55:58,591 P19230 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-01-28 18:55:58,591 P19230 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-01-28 18:55:58,591 P19230 INFO Loading train data done.
2022-01-28 18:56:01,755 P19230 INFO Total number of parameters: 1318230.
2022-01-28 18:56:01,756 P19230 INFO Start training: 343 batches/epoch
2022-01-28 18:56:01,756 P19230 INFO ************ Epoch=1 start ************
2022-01-28 18:56:11,404 P19230 INFO [Metrics] AUC: 0.940145 - logloss: 0.279049
2022-01-28 18:56:11,404 P19230 INFO Save best model: monitor(max): 0.940145
2022-01-28 18:56:11,411 P19230 INFO --- 343/343 batches finished ---
2022-01-28 18:56:11,447 P19230 INFO Train loss: 0.430998
2022-01-28 18:56:11,448 P19230 INFO ************ Epoch=1 end ************
2022-01-28 18:56:21,177 P19230 INFO [Metrics] AUC: 0.948656 - logloss: 0.259961
2022-01-28 18:56:21,178 P19230 INFO Save best model: monitor(max): 0.948656
2022-01-28 18:56:21,185 P19230 INFO --- 343/343 batches finished ---
2022-01-28 18:56:21,222 P19230 INFO Train loss: 0.267695
2022-01-28 18:56:21,222 P19230 INFO ************ Epoch=2 end ************
2022-01-28 18:56:30,805 P19230 INFO [Metrics] AUC: 0.949862 - logloss: 0.265737
2022-01-28 18:56:30,805 P19230 INFO Save best model: monitor(max): 0.949862
2022-01-28 18:56:30,813 P19230 INFO --- 343/343 batches finished ---
2022-01-28 18:56:30,850 P19230 INFO Train loss: 0.220911
2022-01-28 18:56:30,850 P19230 INFO ************ Epoch=3 end ************
2022-01-28 18:56:40,187 P19230 INFO [Metrics] AUC: 0.948907 - logloss: 0.290494
2022-01-28 18:56:40,187 P19230 INFO Monitor(max) STOP: 0.948907 !
2022-01-28 18:56:40,188 P19230 INFO Reduce learning rate on plateau: 0.000100
2022-01-28 18:56:40,188 P19230 INFO --- 343/343 batches finished ---
2022-01-28 18:56:40,224 P19230 INFO Train loss: 0.183626
2022-01-28 18:56:40,224 P19230 INFO ************ Epoch=4 end ************
2022-01-28 18:56:49,621 P19230 INFO [Metrics] AUC: 0.947341 - logloss: 0.361041
2022-01-28 18:56:49,622 P19230 INFO Monitor(max) STOP: 0.947341 !
2022-01-28 18:56:49,622 P19230 INFO Reduce learning rate on plateau: 0.000010
2022-01-28 18:56:49,622 P19230 INFO Early stopping at epoch=5
2022-01-28 18:56:49,622 P19230 INFO --- 343/343 batches finished ---
2022-01-28 18:56:49,662 P19230 INFO Train loss: 0.121798
2022-01-28 18:56:49,662 P19230 INFO Training finished.
2022-01-28 18:56:49,662 P19230 INFO Load best model: /home/XXX/benchmarks/Movielens/NFM_movielenslatest_x1/movielenslatest_x1_cd32d937/NFM_movielenslatest_x1_010_f3d546bc.model
2022-01-28 18:56:52,771 P19230 INFO ****** Validation evaluation ******
2022-01-28 18:56:54,210 P19230 INFO [Metrics] AUC: 0.949862 - logloss: 0.265737
2022-01-28 18:56:54,252 P19230 INFO ******** Test evaluation ********
2022-01-28 18:56:54,252 P19230 INFO Loading data...
2022-01-28 18:56:54,252 P19230 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-01-28 18:56:54,256 P19230 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-01-28 18:56:54,256 P19230 INFO Loading test data done.
2022-01-28 18:56:54,968 P19230 INFO [Metrics] AUC: 0.949597 - logloss: 0.266413

```
