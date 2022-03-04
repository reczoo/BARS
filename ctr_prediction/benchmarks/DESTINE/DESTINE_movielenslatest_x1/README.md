## DESTINE_movielenslatest_x1

A hands-on guide to run the DESTINE model on the Movielenslatest_x1 dataset.

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
  fuxictr: 1.1.1
  ```

### Dataset
Dataset ID: [Movielenslatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/MovieLens/README.md#Movielenslatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.1](https://github.com/xue-pai/FuxiCTR/tree/v1.1.1) for this experiment. See the model code: [DESTINE](https://github.com/xue-pai/FuxiCTR/blob/v1.1.1/fuxictr/pytorch/models/DESTINE.py).

Running steps:

1. Download [FuxiCTR-v1.1.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DESTINE_movielenslatest_x1_tuner_config_03](./DESTINE_movielenslatest_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DESTINE_movielenslatest_x1
    nohup python run_expid.py --config ./DESTINE_movielenslatest_x1_tuner_config_03 --expid DESTINE_movielenslatest_x1_011_f64ca64d --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.969395 | 0.212452  |


### Logs
```python
2022-02-19 00:19:35,759 P42463 INFO {
    "att_dropout": "0.1",
    "attention_dim": "64",
    "attention_layers": "3",
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
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DESTINE",
    "model_id": "DESTINE_movielenslatest_x1_011_f64ca64d",
    "model_root": "./Movielens/DESTINE_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_heads": "1",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "relu_before_att": "False",
    "residual_mode": "each_layer",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Movielens/MovielensLatest_x1/test.csv",
    "train_data": "../data/Movielens/MovielensLatest_x1/train.csv",
    "use_hdf5": "True",
    "use_scale": "True",
    "use_wide": "False",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-02-19 00:19:35,760 P42463 INFO Set up feature encoder...
2022-02-19 00:19:35,760 P42463 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-02-19 00:19:35,760 P42463 INFO Loading data...
2022-02-19 00:19:35,763 P42463 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-02-19 00:19:35,792 P42463 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-02-19 00:19:35,801 P42463 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-02-19 00:19:35,801 P42463 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-02-19 00:19:35,801 P42463 INFO Loading train data done.
2022-02-19 00:19:39,783 P42463 INFO Total number of parameters: 1274821.
2022-02-19 00:19:39,783 P42463 INFO Start training: 343 batches/epoch
2022-02-19 00:19:39,784 P42463 INFO ************ Epoch=1 start ************
2022-02-19 00:20:16,878 P42463 INFO [Metrics] AUC: 0.931520 - logloss: 0.297814
2022-02-19 00:20:16,879 P42463 INFO Save best model: monitor(max): 0.931520
2022-02-19 00:20:16,890 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:20:16,925 P42463 INFO Train loss: 0.385184
2022-02-19 00:20:16,925 P42463 INFO ************ Epoch=1 end ************
2022-02-19 00:20:58,102 P42463 INFO [Metrics] AUC: 0.942267 - logloss: 0.281503
2022-02-19 00:20:58,102 P42463 INFO Save best model: monitor(max): 0.942267
2022-02-19 00:20:58,112 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:20:58,147 P42463 INFO Train loss: 0.370614
2022-02-19 00:20:58,147 P42463 INFO ************ Epoch=2 end ************
2022-02-19 00:21:39,293 P42463 INFO [Metrics] AUC: 0.945850 - logloss: 0.272156
2022-02-19 00:21:39,294 P42463 INFO Save best model: monitor(max): 0.945850
2022-02-19 00:21:39,303 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:21:39,342 P42463 INFO Train loss: 0.370154
2022-02-19 00:21:39,342 P42463 INFO ************ Epoch=3 end ************
2022-02-19 00:22:20,281 P42463 INFO [Metrics] AUC: 0.948812 - logloss: 0.256134
2022-02-19 00:22:20,281 P42463 INFO Save best model: monitor(max): 0.948812
2022-02-19 00:22:20,291 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:22:20,332 P42463 INFO Train loss: 0.371447
2022-02-19 00:22:20,332 P42463 INFO ************ Epoch=4 end ************
2022-02-19 00:23:01,380 P42463 INFO [Metrics] AUC: 0.951121 - logloss: 0.247868
2022-02-19 00:23:01,381 P42463 INFO Save best model: monitor(max): 0.951121
2022-02-19 00:23:01,396 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:23:01,453 P42463 INFO Train loss: 0.374460
2022-02-19 00:23:01,453 P42463 INFO ************ Epoch=5 end ************
2022-02-19 00:23:42,297 P42463 INFO [Metrics] AUC: 0.952252 - logloss: 0.244469
2022-02-19 00:23:42,298 P42463 INFO Save best model: monitor(max): 0.952252
2022-02-19 00:23:42,307 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:23:42,346 P42463 INFO Train loss: 0.375270
2022-02-19 00:23:42,346 P42463 INFO ************ Epoch=6 end ************
2022-02-19 00:24:23,038 P42463 INFO [Metrics] AUC: 0.953870 - logloss: 0.239837
2022-02-19 00:24:23,039 P42463 INFO Save best model: monitor(max): 0.953870
2022-02-19 00:24:23,054 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:24:23,091 P42463 INFO Train loss: 0.376610
2022-02-19 00:24:23,091 P42463 INFO ************ Epoch=7 end ************
2022-02-19 00:25:03,912 P42463 INFO [Metrics] AUC: 0.954345 - logloss: 0.238242
2022-02-19 00:25:03,913 P42463 INFO Save best model: monitor(max): 0.954345
2022-02-19 00:25:03,923 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:25:03,965 P42463 INFO Train loss: 0.377530
2022-02-19 00:25:03,965 P42463 INFO ************ Epoch=8 end ************
2022-02-19 00:25:44,680 P42463 INFO [Metrics] AUC: 0.954880 - logloss: 0.238565
2022-02-19 00:25:44,681 P42463 INFO Save best model: monitor(max): 0.954880
2022-02-19 00:25:44,691 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:25:44,739 P42463 INFO Train loss: 0.375859
2022-02-19 00:25:44,739 P42463 INFO ************ Epoch=9 end ************
2022-02-19 00:26:25,538 P42463 INFO [Metrics] AUC: 0.955331 - logloss: 0.235825
2022-02-19 00:26:25,538 P42463 INFO Save best model: monitor(max): 0.955331
2022-02-19 00:26:25,548 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:26:25,591 P42463 INFO Train loss: 0.376940
2022-02-19 00:26:25,591 P42463 INFO ************ Epoch=10 end ************
2022-02-19 00:27:06,578 P42463 INFO [Metrics] AUC: 0.955740 - logloss: 0.236645
2022-02-19 00:27:06,578 P42463 INFO Save best model: monitor(max): 0.955740
2022-02-19 00:27:06,588 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:27:06,626 P42463 INFO Train loss: 0.377148
2022-02-19 00:27:06,626 P42463 INFO ************ Epoch=11 end ************
2022-02-19 00:27:47,551 P42463 INFO [Metrics] AUC: 0.956200 - logloss: 0.233234
2022-02-19 00:27:47,552 P42463 INFO Save best model: monitor(max): 0.956200
2022-02-19 00:27:47,562 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:27:47,602 P42463 INFO Train loss: 0.377508
2022-02-19 00:27:47,602 P42463 INFO ************ Epoch=12 end ************
2022-02-19 00:28:28,701 P42463 INFO [Metrics] AUC: 0.956330 - logloss: 0.234603
2022-02-19 00:28:28,701 P42463 INFO Save best model: monitor(max): 0.956330
2022-02-19 00:28:28,713 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:28:28,759 P42463 INFO Train loss: 0.376198
2022-02-19 00:28:28,759 P42463 INFO ************ Epoch=13 end ************
2022-02-19 00:28:50,157 P42463 INFO [Metrics] AUC: 0.957096 - logloss: 0.232061
2022-02-19 00:28:50,157 P42463 INFO Save best model: monitor(max): 0.957096
2022-02-19 00:28:50,167 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:28:50,207 P42463 INFO Train loss: 0.376831
2022-02-19 00:28:50,207 P42463 INFO ************ Epoch=14 end ************
2022-02-19 00:29:10,642 P42463 INFO [Metrics] AUC: 0.957166 - logloss: 0.231509
2022-02-19 00:29:10,642 P42463 INFO Save best model: monitor(max): 0.957166
2022-02-19 00:29:10,652 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:29:10,690 P42463 INFO Train loss: 0.377383
2022-02-19 00:29:10,690 P42463 INFO ************ Epoch=15 end ************
2022-02-19 00:29:22,183 P42463 INFO [Metrics] AUC: 0.956028 - logloss: 0.239762
2022-02-19 00:29:22,184 P42463 INFO Monitor(max) STOP: 0.956028 !
2022-02-19 00:29:22,184 P42463 INFO Reduce learning rate on plateau: 0.000100
2022-02-19 00:29:22,184 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:29:22,222 P42463 INFO Train loss: 0.377671
2022-02-19 00:29:22,222 P42463 INFO ************ Epoch=16 end ************
2022-02-19 00:29:33,686 P42463 INFO [Metrics] AUC: 0.968027 - logloss: 0.203946
2022-02-19 00:29:33,686 P42463 INFO Save best model: monitor(max): 0.968027
2022-02-19 00:29:33,696 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:29:33,740 P42463 INFO Train loss: 0.278238
2022-02-19 00:29:33,740 P42463 INFO ************ Epoch=17 end ************
2022-02-19 00:29:45,312 P42463 INFO [Metrics] AUC: 0.969631 - logloss: 0.211023
2022-02-19 00:29:45,312 P42463 INFO Save best model: monitor(max): 0.969631
2022-02-19 00:29:45,328 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:29:45,365 P42463 INFO Train loss: 0.192021
2022-02-19 00:29:45,365 P42463 INFO ************ Epoch=18 end ************
2022-02-19 00:29:56,917 P42463 INFO [Metrics] AUC: 0.968262 - logloss: 0.233335
2022-02-19 00:29:56,917 P42463 INFO Monitor(max) STOP: 0.968262 !
2022-02-19 00:29:56,918 P42463 INFO Reduce learning rate on plateau: 0.000010
2022-02-19 00:29:56,918 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:29:56,959 P42463 INFO Train loss: 0.148547
2022-02-19 00:29:56,959 P42463 INFO ************ Epoch=19 end ************
2022-02-19 00:30:08,526 P42463 INFO [Metrics] AUC: 0.967809 - logloss: 0.247005
2022-02-19 00:30:08,527 P42463 INFO Monitor(max) STOP: 0.967809 !
2022-02-19 00:30:08,527 P42463 INFO Reduce learning rate on plateau: 0.000001
2022-02-19 00:30:08,527 P42463 INFO Early stopping at epoch=20
2022-02-19 00:30:08,527 P42463 INFO --- 343/343 batches finished ---
2022-02-19 00:30:08,562 P42463 INFO Train loss: 0.115975
2022-02-19 00:30:08,563 P42463 INFO Training finished.
2022-02-19 00:30:08,563 P42463 INFO Load best model: /home/XXX/benchmarks/Movielens/DESTINE_movielenslatest_x1/movielenslatest_x1_cd32d937/DESTINE_movielenslatest_x1_011_f64ca64d.model
2022-02-19 00:30:08,579 P42463 INFO ****** Validation evaluation ******
2022-02-19 00:30:09,935 P42463 INFO [Metrics] AUC: 0.969631 - logloss: 0.211023
2022-02-19 00:30:09,982 P42463 INFO ******** Test evaluation ********
2022-02-19 00:30:09,982 P42463 INFO Loading data...
2022-02-19 00:30:09,982 P42463 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-02-19 00:30:09,987 P42463 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-02-19 00:30:09,987 P42463 INFO Loading test data done.
2022-02-19 00:30:10,732 P42463 INFO [Metrics] AUC: 0.969395 - logloss: 0.212452

```
