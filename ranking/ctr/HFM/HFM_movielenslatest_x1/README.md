## HFM_movielenslatest_x1

A hands-on guide to run the HFM model on the Movielenslatest_x1 dataset.

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
  fuxictr: 1.1.0
  ```

### Dataset
Dataset ID: [MovielensLatest_x1](https://github.com/reczoo/Datasets/tree/main/MovieLens/MovielensLatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [HFM](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/HFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HFM_movielenslatest_x1_tuner_config_05](./HFM_movielenslatest_x1_tuner_config_05). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HFM_movielenslatest_x1
    nohup python run_expid.py --config ./HFM_movielenslatest_x1_tuner_config_05 --expid HFM_movielenslatest_x1_002_e5873ea8 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.937170 | 0.294840  |


### Logs
```python
2022-02-01 19:22:13,620 P21450 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "5e-06",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[64, 64, 64]",
    "interaction_type": "circular_correlation",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "HFM",
    "model_id": "HFM_movielenslatest_x1_002_e5873ea8",
    "model_root": "./Movielens/HFM_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0.0001",
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
    "use_dnn": "False",
    "use_hdf5": "True",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-02-01 19:22:13,621 P21450 INFO Set up feature encoder...
2022-02-01 19:22:13,621 P21450 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-02-01 19:22:13,622 P21450 INFO Loading data...
2022-02-01 19:22:13,624 P21450 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-02-01 19:22:13,651 P21450 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-02-01 19:22:13,659 P21450 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-02-01 19:22:13,660 P21450 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-02-01 19:22:13,660 P21450 INFO Loading train data done.
2022-02-01 19:22:17,070 P21450 INFO Total number of parameters: 992640.
2022-02-01 19:22:17,071 P21450 INFO Start training: 343 batches/epoch
2022-02-01 19:22:17,071 P21450 INFO ************ Epoch=1 start ************
2022-02-01 19:22:22,288 P21450 INFO [Metrics] AUC: 0.906958 - logloss: 0.358994
2022-02-01 19:22:22,288 P21450 INFO Save best model: monitor(max): 0.906958
2022-02-01 19:22:22,293 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:22:22,335 P21450 INFO Train loss: 0.529254
2022-02-01 19:22:22,335 P21450 INFO ************ Epoch=1 end ************
2022-02-01 19:22:27,407 P21450 INFO [Metrics] AUC: 0.926548 - logloss: 0.312022
2022-02-01 19:22:27,407 P21450 INFO Save best model: monitor(max): 0.926548
2022-02-01 19:22:27,413 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:22:27,452 P21450 INFO Train loss: 0.325583
2022-02-01 19:22:27,452 P21450 INFO ************ Epoch=2 end ************
2022-02-01 19:22:32,450 P21450 INFO [Metrics] AUC: 0.930374 - logloss: 0.304550
2022-02-01 19:22:32,450 P21450 INFO Save best model: monitor(max): 0.930374
2022-02-01 19:22:32,457 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:22:32,500 P21450 INFO Train loss: 0.299503
2022-02-01 19:22:32,500 P21450 INFO ************ Epoch=3 end ************
2022-02-01 19:22:37,560 P21450 INFO [Metrics] AUC: 0.931764 - logloss: 0.302109
2022-02-01 19:22:37,561 P21450 INFO Save best model: monitor(max): 0.931764
2022-02-01 19:22:37,567 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:22:37,601 P21450 INFO Train loss: 0.291429
2022-02-01 19:22:37,601 P21450 INFO ************ Epoch=4 end ************
2022-02-01 19:22:42,734 P21450 INFO [Metrics] AUC: 0.932397 - logloss: 0.301058
2022-02-01 19:22:42,734 P21450 INFO Save best model: monitor(max): 0.932397
2022-02-01 19:22:42,741 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:22:42,778 P21450 INFO Train loss: 0.287230
2022-02-01 19:22:42,778 P21450 INFO ************ Epoch=5 end ************
2022-02-01 19:22:48,018 P21450 INFO [Metrics] AUC: 0.932843 - logloss: 0.300518
2022-02-01 19:22:48,018 P21450 INFO Save best model: monitor(max): 0.932843
2022-02-01 19:22:48,024 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:22:48,064 P21450 INFO Train loss: 0.284445
2022-02-01 19:22:48,064 P21450 INFO ************ Epoch=6 end ************
2022-02-01 19:22:53,331 P21450 INFO [Metrics] AUC: 0.933079 - logloss: 0.300271
2022-02-01 19:22:53,331 P21450 INFO Save best model: monitor(max): 0.933079
2022-02-01 19:22:53,337 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:22:53,373 P21450 INFO Train loss: 0.282485
2022-02-01 19:22:53,373 P21450 INFO ************ Epoch=7 end ************
2022-02-01 19:22:58,489 P21450 INFO [Metrics] AUC: 0.933350 - logloss: 0.300034
2022-02-01 19:22:58,490 P21450 INFO Save best model: monitor(max): 0.933350
2022-02-01 19:22:58,496 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:22:58,538 P21450 INFO Train loss: 0.280914
2022-02-01 19:22:58,538 P21450 INFO ************ Epoch=8 end ************
2022-02-01 19:23:03,583 P21450 INFO [Metrics] AUC: 0.933517 - logloss: 0.299820
2022-02-01 19:23:03,584 P21450 INFO Save best model: monitor(max): 0.933517
2022-02-01 19:23:03,589 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:23:03,625 P21450 INFO Train loss: 0.279556
2022-02-01 19:23:03,625 P21450 INFO ************ Epoch=9 end ************
2022-02-01 19:23:08,598 P21450 INFO [Metrics] AUC: 0.933756 - logloss: 0.299462
2022-02-01 19:23:08,599 P21450 INFO Save best model: monitor(max): 0.933756
2022-02-01 19:23:08,604 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:23:08,640 P21450 INFO Train loss: 0.278406
2022-02-01 19:23:08,640 P21450 INFO ************ Epoch=10 end ************
2022-02-01 19:23:13,648 P21450 INFO [Metrics] AUC: 0.933787 - logloss: 0.299514
2022-02-01 19:23:13,648 P21450 INFO Save best model: monitor(max): 0.933787
2022-02-01 19:23:13,654 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:23:13,689 P21450 INFO Train loss: 0.277333
2022-02-01 19:23:13,689 P21450 INFO ************ Epoch=11 end ************
2022-02-01 19:23:18,783 P21450 INFO [Metrics] AUC: 0.933963 - logloss: 0.299196
2022-02-01 19:23:18,783 P21450 INFO Save best model: monitor(max): 0.933963
2022-02-01 19:23:18,789 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:23:18,826 P21450 INFO Train loss: 0.276454
2022-02-01 19:23:18,827 P21450 INFO ************ Epoch=12 end ************
2022-02-01 19:23:23,852 P21450 INFO [Metrics] AUC: 0.934026 - logloss: 0.299286
2022-02-01 19:23:23,853 P21450 INFO Save best model: monitor(max): 0.934026
2022-02-01 19:23:23,859 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:23:23,899 P21450 INFO Train loss: 0.275629
2022-02-01 19:23:23,899 P21450 INFO ************ Epoch=13 end ************
2022-02-01 19:23:28,979 P21450 INFO [Metrics] AUC: 0.934027 - logloss: 0.299277
2022-02-01 19:23:28,979 P21450 INFO Save best model: monitor(max): 0.934027
2022-02-01 19:23:28,985 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:23:29,025 P21450 INFO Train loss: 0.274842
2022-02-01 19:23:29,025 P21450 INFO ************ Epoch=14 end ************
2022-02-01 19:23:34,196 P21450 INFO [Metrics] AUC: 0.934157 - logloss: 0.299055
2022-02-01 19:23:34,197 P21450 INFO Save best model: monitor(max): 0.934157
2022-02-01 19:23:34,203 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:23:34,239 P21450 INFO Train loss: 0.273996
2022-02-01 19:23:34,239 P21450 INFO ************ Epoch=15 end ************
2022-02-01 19:23:39,140 P21450 INFO [Metrics] AUC: 0.934349 - logloss: 0.298760
2022-02-01 19:23:39,140 P21450 INFO Save best model: monitor(max): 0.934349
2022-02-01 19:23:39,146 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:23:39,183 P21450 INFO Train loss: 0.272993
2022-02-01 19:23:39,184 P21450 INFO ************ Epoch=16 end ************
2022-02-01 19:23:44,214 P21450 INFO [Metrics] AUC: 0.934787 - logloss: 0.297463
2022-02-01 19:23:44,215 P21450 INFO Save best model: monitor(max): 0.934787
2022-02-01 19:23:44,220 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:23:44,258 P21450 INFO Train loss: 0.271289
2022-02-01 19:23:44,258 P21450 INFO ************ Epoch=17 end ************
2022-02-01 19:23:49,340 P21450 INFO [Metrics] AUC: 0.935092 - logloss: 0.296213
2022-02-01 19:23:49,340 P21450 INFO Save best model: monitor(max): 0.935092
2022-02-01 19:23:49,346 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:23:49,383 P21450 INFO Train loss: 0.268315
2022-02-01 19:23:49,383 P21450 INFO ************ Epoch=18 end ************
2022-02-01 19:23:54,530 P21450 INFO [Metrics] AUC: 0.935236 - logloss: 0.295521
2022-02-01 19:23:54,530 P21450 INFO Save best model: monitor(max): 0.935236
2022-02-01 19:23:54,536 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:23:54,572 P21450 INFO Train loss: 0.264572
2022-02-01 19:23:54,572 P21450 INFO ************ Epoch=19 end ************
2022-02-01 19:23:59,706 P21450 INFO [Metrics] AUC: 0.935540 - logloss: 0.294801
2022-02-01 19:23:59,707 P21450 INFO Save best model: monitor(max): 0.935540
2022-02-01 19:23:59,713 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:23:59,754 P21450 INFO Train loss: 0.260788
2022-02-01 19:23:59,754 P21450 INFO ************ Epoch=20 end ************
2022-02-01 19:24:05,023 P21450 INFO [Metrics] AUC: 0.936118 - logloss: 0.293794
2022-02-01 19:24:05,024 P21450 INFO Save best model: monitor(max): 0.936118
2022-02-01 19:24:05,029 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:24:05,082 P21450 INFO Train loss: 0.256572
2022-02-01 19:24:05,082 P21450 INFO ************ Epoch=21 end ************
2022-02-01 19:24:10,268 P21450 INFO [Metrics] AUC: 0.936788 - logloss: 0.292820
2022-02-01 19:24:10,268 P21450 INFO Save best model: monitor(max): 0.936788
2022-02-01 19:24:10,274 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:24:10,309 P21450 INFO Train loss: 0.251452
2022-02-01 19:24:10,310 P21450 INFO ************ Epoch=22 end ************
2022-02-01 19:24:15,313 P21450 INFO [Metrics] AUC: 0.937257 - logloss: 0.292769
2022-02-01 19:24:15,314 P21450 INFO Save best model: monitor(max): 0.937257
2022-02-01 19:24:15,319 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:24:15,355 P21450 INFO Train loss: 0.245838
2022-02-01 19:24:15,355 P21450 INFO ************ Epoch=23 end ************
2022-02-01 19:24:20,380 P21450 INFO [Metrics] AUC: 0.937442 - logloss: 0.293956
2022-02-01 19:24:20,381 P21450 INFO Save best model: monitor(max): 0.937442
2022-02-01 19:24:20,387 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:24:20,422 P21450 INFO Train loss: 0.240288
2022-02-01 19:24:20,422 P21450 INFO ************ Epoch=24 end ************
2022-02-01 19:24:25,492 P21450 INFO [Metrics] AUC: 0.937326 - logloss: 0.296305
2022-02-01 19:24:25,493 P21450 INFO Monitor(max) STOP: 0.937326 !
2022-02-01 19:24:25,493 P21450 INFO Reduce learning rate on plateau: 0.000100
2022-02-01 19:24:25,493 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:24:25,529 P21450 INFO Train loss: 0.234975
2022-02-01 19:24:25,529 P21450 INFO ************ Epoch=25 end ************
2022-02-01 19:24:30,456 P21450 INFO [Metrics] AUC: 0.937372 - logloss: 0.297032
2022-02-01 19:24:30,457 P21450 INFO Monitor(max) STOP: 0.937372 !
2022-02-01 19:24:30,457 P21450 INFO Reduce learning rate on plateau: 0.000010
2022-02-01 19:24:30,457 P21450 INFO Early stopping at epoch=26
2022-02-01 19:24:30,457 P21450 INFO --- 343/343 batches finished ---
2022-02-01 19:24:30,493 P21450 INFO Train loss: 0.219176
2022-02-01 19:24:30,493 P21450 INFO Training finished.
2022-02-01 19:24:30,493 P21450 INFO Load best model: /home/XXX/benchmarks/Movielens/HFM_movielenslatest_x1/movielenslatest_x1_cd32d937/HFM_movielenslatest_x1_002_e5873ea8.model
2022-02-01 19:24:33,255 P21450 INFO ****** Validation evaluation ******
2022-02-01 19:24:34,521 P21450 INFO [Metrics] AUC: 0.937442 - logloss: 0.293956
2022-02-01 19:24:34,571 P21450 INFO ******** Test evaluation ********
2022-02-01 19:24:34,571 P21450 INFO Loading data...
2022-02-01 19:24:34,572 P21450 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-02-01 19:24:34,576 P21450 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-02-01 19:24:34,576 P21450 INFO Loading test data done.
2022-02-01 19:24:35,326 P21450 INFO [Metrics] AUC: 0.937170 - logloss: 0.294840

```
