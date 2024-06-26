## SAM_movielenslatest_x1

A hands-on guide to run the SAM model on the MovielensLatest_x1 dataset.

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

We use [FuxiCTR-v1.2.1](https://github.com/reczoo/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [SAM](https://github.com/reczoo/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/SAM.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [SAM_movielenslatest_x1_tuner_config_06](./SAM_movielenslatest_x1_tuner_config_06). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd SAM_movielenslatest_x1
    nohup python run_expid.py --config ./SAM_movielenslatest_x1_tuner_config_06 --expid SAM_movielenslatest_x1_013_68a6bc8b --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.963104 | 0.266696  |


### Logs
```python
2022-04-12 16:54:38,091 P12228 INFO {
    "aggregation": "weighted_pooling",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "1",
    "interaction_type": "SAM3A",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "SAM",
    "model_id": "SAM_movielenslatest_x1_013_68a6bc8b",
    "model_root": "./Movielens/SAM_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_interaction_layers": "2",
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
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-04-12 16:54:38,092 P12228 INFO Set up feature encoder...
2022-04-12 16:54:38,092 P12228 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-04-12 16:54:38,092 P12228 INFO Loading data...
2022-04-12 16:54:38,095 P12228 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-04-12 16:54:38,123 P12228 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-04-12 16:54:38,132 P12228 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-04-12 16:54:38,133 P12228 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-04-12 16:54:38,133 P12228 INFO Loading train data done.
2022-04-12 16:54:41,830 P12228 INFO Total number of parameters: 902984.
2022-04-12 16:54:41,830 P12228 INFO Start training: 343 batches/epoch
2022-04-12 16:54:41,831 P12228 INFO ************ Epoch=1 start ************
2022-04-12 16:54:51,666 P12228 INFO [Metrics] AUC: 0.930376 - logloss: 0.301244
2022-04-12 16:54:51,667 P12228 INFO Save best model: monitor(max): 0.930376
2022-04-12 16:54:51,673 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:54:51,749 P12228 INFO Train loss: 0.427487
2022-04-12 16:54:51,750 P12228 INFO ************ Epoch=1 end ************
2022-04-12 16:55:00,419 P12228 INFO [Metrics] AUC: 0.933289 - logloss: 0.296415
2022-04-12 16:55:00,419 P12228 INFO Save best model: monitor(max): 0.933289
2022-04-12 16:55:00,426 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:55:00,541 P12228 INFO Train loss: 0.334632
2022-04-12 16:55:00,541 P12228 INFO ************ Epoch=2 end ************
2022-04-12 16:55:08,977 P12228 INFO [Metrics] AUC: 0.937246 - logloss: 0.287503
2022-04-12 16:55:08,977 P12228 INFO Save best model: monitor(max): 0.937246
2022-04-12 16:55:08,985 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:55:09,047 P12228 INFO Train loss: 0.321591
2022-04-12 16:55:09,047 P12228 INFO ************ Epoch=3 end ************
2022-04-12 16:55:18,340 P12228 INFO [Metrics] AUC: 0.940396 - logloss: 0.279409
2022-04-12 16:55:18,340 P12228 INFO Save best model: monitor(max): 0.940396
2022-04-12 16:55:18,348 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:55:18,401 P12228 INFO Train loss: 0.314695
2022-04-12 16:55:18,401 P12228 INFO ************ Epoch=4 end ************
2022-04-12 16:55:27,779 P12228 INFO [Metrics] AUC: 0.942787 - logloss: 0.273496
2022-04-12 16:55:27,780 P12228 INFO Save best model: monitor(max): 0.942787
2022-04-12 16:55:27,787 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:55:27,856 P12228 INFO Train loss: 0.306847
2022-04-12 16:55:27,856 P12228 INFO ************ Epoch=5 end ************
2022-04-12 16:55:37,908 P12228 INFO [Metrics] AUC: 0.944759 - logloss: 0.269057
2022-04-12 16:55:37,909 P12228 INFO Save best model: monitor(max): 0.944759
2022-04-12 16:55:37,920 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:55:37,978 P12228 INFO Train loss: 0.299294
2022-04-12 16:55:37,978 P12228 INFO ************ Epoch=6 end ************
2022-04-12 16:55:47,691 P12228 INFO [Metrics] AUC: 0.946652 - logloss: 0.264696
2022-04-12 16:55:47,692 P12228 INFO Save best model: monitor(max): 0.946652
2022-04-12 16:55:47,700 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:55:47,758 P12228 INFO Train loss: 0.291928
2022-04-12 16:55:47,758 P12228 INFO ************ Epoch=7 end ************
2022-04-12 16:55:57,385 P12228 INFO [Metrics] AUC: 0.949001 - logloss: 0.258957
2022-04-12 16:55:57,386 P12228 INFO Save best model: monitor(max): 0.949001
2022-04-12 16:55:57,393 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:55:57,467 P12228 INFO Train loss: 0.284887
2022-04-12 16:55:57,467 P12228 INFO ************ Epoch=8 end ************
2022-04-12 16:56:06,608 P12228 INFO [Metrics] AUC: 0.951515 - logloss: 0.253123
2022-04-12 16:56:06,609 P12228 INFO Save best model: monitor(max): 0.951515
2022-04-12 16:56:06,617 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:56:06,675 P12228 INFO Train loss: 0.277926
2022-04-12 16:56:06,675 P12228 INFO ************ Epoch=9 end ************
2022-04-12 16:56:16,250 P12228 INFO [Metrics] AUC: 0.953628 - logloss: 0.246790
2022-04-12 16:56:16,251 P12228 INFO Save best model: monitor(max): 0.953628
2022-04-12 16:56:16,258 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:56:16,312 P12228 INFO Train loss: 0.271264
2022-04-12 16:56:16,312 P12228 INFO ************ Epoch=10 end ************
2022-04-12 16:56:25,827 P12228 INFO [Metrics] AUC: 0.955708 - logloss: 0.241655
2022-04-12 16:56:25,828 P12228 INFO Save best model: monitor(max): 0.955708
2022-04-12 16:56:25,835 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:56:25,901 P12228 INFO Train loss: 0.264930
2022-04-12 16:56:25,901 P12228 INFO ************ Epoch=11 end ************
2022-04-12 16:56:35,385 P12228 INFO [Metrics] AUC: 0.957115 - logloss: 0.239097
2022-04-12 16:56:35,385 P12228 INFO Save best model: monitor(max): 0.957115
2022-04-12 16:56:35,392 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:56:35,459 P12228 INFO Train loss: 0.258424
2022-04-12 16:56:35,460 P12228 INFO ************ Epoch=12 end ************
2022-04-12 16:56:44,696 P12228 INFO [Metrics] AUC: 0.958158 - logloss: 0.237206
2022-04-12 16:56:44,697 P12228 INFO Save best model: monitor(max): 0.958158
2022-04-12 16:56:44,704 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:56:44,754 P12228 INFO Train loss: 0.250939
2022-04-12 16:56:44,754 P12228 INFO ************ Epoch=13 end ************
2022-04-12 16:56:54,347 P12228 INFO [Metrics] AUC: 0.958425 - logloss: 0.238402
2022-04-12 16:56:54,348 P12228 INFO Save best model: monitor(max): 0.958425
2022-04-12 16:56:54,355 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:56:54,427 P12228 INFO Train loss: 0.242757
2022-04-12 16:56:54,427 P12228 INFO ************ Epoch=14 end ************
2022-04-12 16:57:04,075 P12228 INFO [Metrics] AUC: 0.958673 - logloss: 0.240255
2022-04-12 16:57:04,076 P12228 INFO Save best model: monitor(max): 0.958673
2022-04-12 16:57:04,083 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:57:04,148 P12228 INFO Train loss: 0.236126
2022-04-12 16:57:04,149 P12228 INFO ************ Epoch=15 end ************
2022-04-12 16:57:13,373 P12228 INFO [Metrics] AUC: 0.958731 - logloss: 0.243396
2022-04-12 16:57:13,374 P12228 INFO Save best model: monitor(max): 0.958731
2022-04-12 16:57:13,380 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:57:13,433 P12228 INFO Train loss: 0.229093
2022-04-12 16:57:13,433 P12228 INFO ************ Epoch=16 end ************
2022-04-12 16:57:22,435 P12228 INFO [Metrics] AUC: 0.958769 - logloss: 0.247040
2022-04-12 16:57:22,436 P12228 INFO Save best model: monitor(max): 0.958769
2022-04-12 16:57:22,443 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:57:22,501 P12228 INFO Train loss: 0.223994
2022-04-12 16:57:22,501 P12228 INFO ************ Epoch=17 end ************
2022-04-12 16:57:31,730 P12228 INFO [Metrics] AUC: 0.958379 - logloss: 0.250961
2022-04-12 16:57:31,731 P12228 INFO Monitor(max) STOP: 0.958379 !
2022-04-12 16:57:31,731 P12228 INFO Reduce learning rate on plateau: 0.000100
2022-04-12 16:57:31,731 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:57:31,796 P12228 INFO Train loss: 0.218576
2022-04-12 16:57:31,796 P12228 INFO ************ Epoch=18 end ************
2022-04-12 16:57:41,114 P12228 INFO [Metrics] AUC: 0.961894 - logloss: 0.250288
2022-04-12 16:57:41,115 P12228 INFO Save best model: monitor(max): 0.961894
2022-04-12 16:57:41,122 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:57:41,184 P12228 INFO Train loss: 0.172328
2022-04-12 16:57:41,184 P12228 INFO ************ Epoch=19 end ************
2022-04-12 16:57:50,463 P12228 INFO [Metrics] AUC: 0.962941 - logloss: 0.257173
2022-04-12 16:57:50,464 P12228 INFO Save best model: monitor(max): 0.962941
2022-04-12 16:57:50,471 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:57:50,545 P12228 INFO Train loss: 0.141895
2022-04-12 16:57:50,546 P12228 INFO ************ Epoch=20 end ************
2022-04-12 16:57:59,113 P12228 INFO [Metrics] AUC: 0.963267 - logloss: 0.266475
2022-04-12 16:57:59,113 P12228 INFO Save best model: monitor(max): 0.963267
2022-04-12 16:57:59,120 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:57:59,182 P12228 INFO Train loss: 0.123147
2022-04-12 16:57:59,182 P12228 INFO ************ Epoch=21 end ************
2022-04-12 16:58:07,262 P12228 INFO [Metrics] AUC: 0.963191 - logloss: 0.276652
2022-04-12 16:58:07,263 P12228 INFO Monitor(max) STOP: 0.963191 !
2022-04-12 16:58:07,263 P12228 INFO Reduce learning rate on plateau: 0.000010
2022-04-12 16:58:07,263 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:58:07,315 P12228 INFO Train loss: 0.109783
2022-04-12 16:58:07,316 P12228 INFO ************ Epoch=22 end ************
2022-04-12 16:58:15,280 P12228 INFO [Metrics] AUC: 0.963187 - logloss: 0.277999
2022-04-12 16:58:15,280 P12228 INFO Monitor(max) STOP: 0.963187 !
2022-04-12 16:58:15,281 P12228 INFO Reduce learning rate on plateau: 0.000001
2022-04-12 16:58:15,281 P12228 INFO Early stopping at epoch=23
2022-04-12 16:58:15,281 P12228 INFO --- 343/343 batches finished ---
2022-04-12 16:58:15,334 P12228 INFO Train loss: 0.098557
2022-04-12 16:58:15,334 P12228 INFO Training finished.
2022-04-12 16:58:15,334 P12228 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Movielens/SAM_movielenslatest_x1/movielenslatest_x1_cd32d937/SAM_movielenslatest_x1_013_68a6bc8b.model
2022-04-12 16:58:18,900 P12228 INFO ****** Validation evaluation ******
2022-04-12 16:58:21,366 P12228 INFO [Metrics] AUC: 0.963267 - logloss: 0.266475
2022-04-12 16:58:21,444 P12228 INFO ******** Test evaluation ********
2022-04-12 16:58:21,445 P12228 INFO Loading data...
2022-04-12 16:58:21,445 P12228 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-04-12 16:58:21,451 P12228 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-04-12 16:58:21,452 P12228 INFO Loading test data done.
2022-04-12 16:58:22,400 P12228 INFO [Metrics] AUC: 0.963104 - logloss: 0.266696

```
