## CrossNetv2_movielenslatest_x1

A hands-on guide to run the DCNv2 model on the MovielensLatest_x1 dataset.

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
Dataset ID: [MovielensLatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/MovieLens#MovielensLatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DCNv2](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DCNv2.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CrossNetv2_movielenslatest_x1_tuner_config_01](./CrossNetv2_movielenslatest_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CrossNetv2_movielenslatest_x1
    nohup python run_expid.py --config ./CrossNetv2_movielenslatest_x1_tuner_config_01 --expid DCNv2_movielenslatest_x1_005_a53d8bd5 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.960600 | 0.257848  |


### Logs
```python
2022-01-23 13:32:02,119 P14331 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "dnn_activations": "relu",
    "embedding_dim": "10",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "low_rank": "32",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCNv2",
    "model_id": "DCNv2_movielenslatest_x1_005_a53d8bd5",
    "model_root": "./Frappe/DCNv2_movielenslatest_x1/",
    "model_structure": "crossnet_only",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_cross_layers": "8",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_dnn_hidden_units": "[500, 500, 500]",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "stacked_dnn_hidden_units": "[500, 500, 500]",
    "task": "binary_classification",
    "test_data": "../data/Movielens/MovielensLatest_x1/test.csv",
    "train_data": "../data/Movielens/MovielensLatest_x1/train.csv",
    "use_hdf5": "True",
    "use_low_rank_mixture": "False",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-23 13:32:02,120 P14331 INFO Set up feature encoder...
2022-01-23 13:32:02,120 P14331 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-01-23 13:32:02,121 P14331 INFO Loading data...
2022-01-23 13:32:02,123 P14331 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-01-23 13:32:02,152 P14331 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-01-23 13:32:02,160 P14331 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-01-23 13:32:02,160 P14331 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-01-23 13:32:02,160 P14331 INFO Loading train data done.
2022-01-23 13:32:07,211 P14331 INFO Total number of parameters: 909861.
2022-01-23 13:32:07,212 P14331 INFO Start training: 343 batches/epoch
2022-01-23 13:32:07,212 P14331 INFO ************ Epoch=1 start ************
2022-01-23 13:32:31,195 P14331 INFO [Metrics] AUC: 0.927115 - logloss: 0.307313
2022-01-23 13:32:31,195 P14331 INFO Save best model: monitor(max): 0.927115
2022-01-23 13:32:31,200 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:32:31,253 P14331 INFO Train loss: 0.468740
2022-01-23 13:32:31,253 P14331 INFO ************ Epoch=1 end ************
2022-01-23 13:32:39,231 P14331 INFO [Metrics] AUC: 0.932153 - logloss: 0.299090
2022-01-23 13:32:39,232 P14331 INFO Save best model: monitor(max): 0.932153
2022-01-23 13:32:39,237 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:32:39,294 P14331 INFO Train loss: 0.353250
2022-01-23 13:32:39,294 P14331 INFO ************ Epoch=2 end ************
2022-01-23 13:32:47,195 P14331 INFO [Metrics] AUC: 0.933659 - logloss: 0.296147
2022-01-23 13:32:47,196 P14331 INFO Save best model: monitor(max): 0.933659
2022-01-23 13:32:47,201 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:32:47,257 P14331 INFO Train loss: 0.338897
2022-01-23 13:32:47,257 P14331 INFO ************ Epoch=3 end ************
2022-01-23 13:33:15,034 P14331 INFO [Metrics] AUC: 0.934972 - logloss: 0.293528
2022-01-23 13:33:15,035 P14331 INFO Save best model: monitor(max): 0.934972
2022-01-23 13:33:15,040 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:33:15,093 P14331 INFO Train loss: 0.331366
2022-01-23 13:33:15,093 P14331 INFO ************ Epoch=4 end ************
2022-01-23 13:33:58,454 P14331 INFO [Metrics] AUC: 0.935891 - logloss: 0.291285
2022-01-23 13:33:58,454 P14331 INFO Save best model: monitor(max): 0.935891
2022-01-23 13:33:58,460 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:33:58,519 P14331 INFO Train loss: 0.325854
2022-01-23 13:33:58,519 P14331 INFO ************ Epoch=5 end ************
2022-01-23 13:34:42,534 P14331 INFO [Metrics] AUC: 0.938383 - logloss: 0.285677
2022-01-23 13:34:42,535 P14331 INFO Save best model: monitor(max): 0.938383
2022-01-23 13:34:42,541 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:34:42,593 P14331 INFO Train loss: 0.320053
2022-01-23 13:34:42,593 P14331 INFO ************ Epoch=6 end ************
2022-01-23 13:35:26,158 P14331 INFO [Metrics] AUC: 0.941837 - logloss: 0.277530
2022-01-23 13:35:26,158 P14331 INFO Save best model: monitor(max): 0.941837
2022-01-23 13:35:26,164 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:35:26,212 P14331 INFO Train loss: 0.311736
2022-01-23 13:35:26,212 P14331 INFO ************ Epoch=7 end ************
2022-01-23 13:36:10,017 P14331 INFO [Metrics] AUC: 0.944402 - logloss: 0.271048
2022-01-23 13:36:10,017 P14331 INFO Save best model: monitor(max): 0.944402
2022-01-23 13:36:10,023 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:36:10,073 P14331 INFO Train loss: 0.304064
2022-01-23 13:36:10,073 P14331 INFO ************ Epoch=8 end ************
2022-01-23 13:36:53,642 P14331 INFO [Metrics] AUC: 0.947380 - logloss: 0.264100
2022-01-23 13:36:53,642 P14331 INFO Save best model: monitor(max): 0.947380
2022-01-23 13:36:53,648 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:36:53,695 P14331 INFO Train loss: 0.296497
2022-01-23 13:36:53,696 P14331 INFO ************ Epoch=9 end ************
2022-01-23 13:37:37,226 P14331 INFO [Metrics] AUC: 0.949104 - logloss: 0.260154
2022-01-23 13:37:37,226 P14331 INFO Save best model: monitor(max): 0.949104
2022-01-23 13:37:37,232 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:37:37,279 P14331 INFO Train loss: 0.286990
2022-01-23 13:37:37,279 P14331 INFO ************ Epoch=10 end ************
2022-01-23 13:38:21,087 P14331 INFO [Metrics] AUC: 0.950721 - logloss: 0.256409
2022-01-23 13:38:21,088 P14331 INFO Save best model: monitor(max): 0.950721
2022-01-23 13:38:21,094 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:38:21,143 P14331 INFO Train loss: 0.279196
2022-01-23 13:38:21,143 P14331 INFO ************ Epoch=11 end ************
2022-01-23 13:39:05,407 P14331 INFO [Metrics] AUC: 0.951915 - logloss: 0.253872
2022-01-23 13:39:05,407 P14331 INFO Save best model: monitor(max): 0.951915
2022-01-23 13:39:05,413 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:39:05,492 P14331 INFO Train loss: 0.272639
2022-01-23 13:39:05,493 P14331 INFO ************ Epoch=12 end ************
2022-01-23 13:39:49,580 P14331 INFO [Metrics] AUC: 0.952646 - logloss: 0.252227
2022-01-23 13:39:49,581 P14331 INFO Save best model: monitor(max): 0.952646
2022-01-23 13:39:49,586 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:39:49,638 P14331 INFO Train loss: 0.267295
2022-01-23 13:39:49,638 P14331 INFO ************ Epoch=13 end ************
2022-01-23 13:40:33,519 P14331 INFO [Metrics] AUC: 0.953356 - logloss: 0.250443
2022-01-23 13:40:33,519 P14331 INFO Save best model: monitor(max): 0.953356
2022-01-23 13:40:33,525 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:40:33,571 P14331 INFO Train loss: 0.262843
2022-01-23 13:40:33,571 P14331 INFO ************ Epoch=14 end ************
2022-01-23 13:41:17,246 P14331 INFO [Metrics] AUC: 0.953862 - logloss: 0.249957
2022-01-23 13:41:17,246 P14331 INFO Save best model: monitor(max): 0.953862
2022-01-23 13:41:17,252 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:41:17,299 P14331 INFO Train loss: 0.258983
2022-01-23 13:41:17,299 P14331 INFO ************ Epoch=15 end ************
2022-01-23 13:42:00,881 P14331 INFO [Metrics] AUC: 0.954835 - logloss: 0.247557
2022-01-23 13:42:00,881 P14331 INFO Save best model: monitor(max): 0.954835
2022-01-23 13:42:00,887 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:42:00,934 P14331 INFO Train loss: 0.255221
2022-01-23 13:42:00,934 P14331 INFO ************ Epoch=16 end ************
2022-01-23 13:42:44,366 P14331 INFO [Metrics] AUC: 0.955381 - logloss: 0.246348
2022-01-23 13:42:44,366 P14331 INFO Save best model: monitor(max): 0.955381
2022-01-23 13:42:44,372 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:42:44,418 P14331 INFO Train loss: 0.251695
2022-01-23 13:42:44,418 P14331 INFO ************ Epoch=17 end ************
2022-01-23 13:43:27,947 P14331 INFO [Metrics] AUC: 0.955865 - logloss: 0.245874
2022-01-23 13:43:27,947 P14331 INFO Save best model: monitor(max): 0.955865
2022-01-23 13:43:27,953 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:43:27,999 P14331 INFO Train loss: 0.247962
2022-01-23 13:43:27,999 P14331 INFO ************ Epoch=18 end ************
2022-01-23 13:44:12,125 P14331 INFO [Metrics] AUC: 0.956488 - logloss: 0.245183
2022-01-23 13:44:12,126 P14331 INFO Save best model: monitor(max): 0.956488
2022-01-23 13:44:12,132 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:44:12,182 P14331 INFO Train loss: 0.243939
2022-01-23 13:44:12,182 P14331 INFO ************ Epoch=19 end ************
2022-01-23 13:44:55,588 P14331 INFO [Metrics] AUC: 0.957065 - logloss: 0.243498
2022-01-23 13:44:55,588 P14331 INFO Save best model: monitor(max): 0.957065
2022-01-23 13:44:55,594 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:44:55,642 P14331 INFO Train loss: 0.240151
2022-01-23 13:44:55,642 P14331 INFO ************ Epoch=20 end ************
2022-01-23 13:45:39,335 P14331 INFO [Metrics] AUC: 0.957455 - logloss: 0.244029
2022-01-23 13:45:39,335 P14331 INFO Save best model: monitor(max): 0.957455
2022-01-23 13:45:39,341 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:45:39,390 P14331 INFO Train loss: 0.236745
2022-01-23 13:45:39,390 P14331 INFO ************ Epoch=21 end ************
2022-01-23 13:46:23,262 P14331 INFO [Metrics] AUC: 0.957577 - logloss: 0.244996
2022-01-23 13:46:23,263 P14331 INFO Save best model: monitor(max): 0.957577
2022-01-23 13:46:23,269 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:46:23,318 P14331 INFO Train loss: 0.233452
2022-01-23 13:46:23,318 P14331 INFO ************ Epoch=22 end ************
2022-01-23 13:47:07,423 P14331 INFO [Metrics] AUC: 0.957621 - logloss: 0.245759
2022-01-23 13:47:07,423 P14331 INFO Save best model: monitor(max): 0.957621
2022-01-23 13:47:07,430 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:47:07,476 P14331 INFO Train loss: 0.230427
2022-01-23 13:47:07,478 P14331 INFO ************ Epoch=23 end ************
2022-01-23 13:47:50,921 P14331 INFO [Metrics] AUC: 0.958001 - logloss: 0.246918
2022-01-23 13:47:50,922 P14331 INFO Save best model: monitor(max): 0.958001
2022-01-23 13:47:50,929 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:47:50,979 P14331 INFO Train loss: 0.227437
2022-01-23 13:47:50,979 P14331 INFO ************ Epoch=24 end ************
2022-01-23 13:48:33,936 P14331 INFO [Metrics] AUC: 0.957814 - logloss: 0.248785
2022-01-23 13:48:33,937 P14331 INFO Monitor(max) STOP: 0.957814 !
2022-01-23 13:48:33,937 P14331 INFO Reduce learning rate on plateau: 0.000100
2022-01-23 13:48:33,937 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:48:33,984 P14331 INFO Train loss: 0.224838
2022-01-23 13:48:33,985 P14331 INFO ************ Epoch=25 end ************
2022-01-23 13:49:17,302 P14331 INFO [Metrics] AUC: 0.960105 - logloss: 0.247866
2022-01-23 13:49:17,302 P14331 INFO Save best model: monitor(max): 0.960105
2022-01-23 13:49:17,308 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:49:17,368 P14331 INFO Train loss: 0.182522
2022-01-23 13:49:17,368 P14331 INFO ************ Epoch=26 end ************
2022-01-23 13:50:00,952 P14331 INFO [Metrics] AUC: 0.960734 - logloss: 0.251583
2022-01-23 13:50:00,952 P14331 INFO Save best model: monitor(max): 0.960734
2022-01-23 13:50:00,958 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:50:01,040 P14331 INFO Train loss: 0.162325
2022-01-23 13:50:01,040 P14331 INFO ************ Epoch=27 end ************
2022-01-23 13:50:44,974 P14331 INFO [Metrics] AUC: 0.960746 - logloss: 0.257197
2022-01-23 13:50:44,974 P14331 INFO Save best model: monitor(max): 0.960746
2022-01-23 13:50:44,980 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:50:45,026 P14331 INFO Train loss: 0.150766
2022-01-23 13:50:45,026 P14331 INFO ************ Epoch=28 end ************
2022-01-23 13:51:29,111 P14331 INFO [Metrics] AUC: 0.960468 - logloss: 0.263855
2022-01-23 13:51:29,111 P14331 INFO Monitor(max) STOP: 0.960468 !
2022-01-23 13:51:29,111 P14331 INFO Reduce learning rate on plateau: 0.000010
2022-01-23 13:51:29,111 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:51:29,163 P14331 INFO Train loss: 0.142814
2022-01-23 13:51:29,163 P14331 INFO ************ Epoch=29 end ************
2022-01-23 13:52:13,209 P14331 INFO [Metrics] AUC: 0.960383 - logloss: 0.265112
2022-01-23 13:52:13,210 P14331 INFO Monitor(max) STOP: 0.960383 !
2022-01-23 13:52:13,210 P14331 INFO Reduce learning rate on plateau: 0.000001
2022-01-23 13:52:13,210 P14331 INFO Early stopping at epoch=30
2022-01-23 13:52:13,210 P14331 INFO --- 343/343 batches finished ---
2022-01-23 13:52:13,263 P14331 INFO Train loss: 0.132949
2022-01-23 13:52:13,263 P14331 INFO Training finished.
2022-01-23 13:52:13,264 P14331 INFO Load best model: /home/FuxiCTR/benchmarks/Frappe/DCNv2_movielenslatest_x1/movielenslatest_x1_cd32d937/DCNv2_movielenslatest_x1_005_a53d8bd5.model
2022-01-23 13:52:13,313 P14331 INFO ****** Validation evaluation ******
2022-01-23 13:52:14,890 P14331 INFO [Metrics] AUC: 0.960746 - logloss: 0.257197
2022-01-23 13:52:14,937 P14331 INFO ******** Test evaluation ********
2022-01-23 13:52:14,938 P14331 INFO Loading data...
2022-01-23 13:52:14,938 P14331 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-01-23 13:52:14,943 P14331 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-01-23 13:52:14,943 P14331 INFO Loading test data done.
2022-01-23 13:52:15,794 P14331 INFO [Metrics] AUC: 0.960600 - logloss: 0.257848

```
