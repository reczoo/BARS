## EDCN_movielenslatest_x1

A hands-on guide to run the EDCN model on the MovielensLatest_x1 dataset.

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
  fuxictr: 1.2.1
  ```

### Dataset
Dataset ID: [MovielensLatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/MovieLens#MovielensLatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [EDCN](https://github.com/xue-pai/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/EDCN.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [EDCN_movielenslatest_x1_tuner_config_02](./EDCN_movielenslatest_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd EDCN_movielenslatest_x1
    nohup python run_expid.py --config ./EDCN_movielenslatest_x1_tuner_config_02 --expid EDCN_movielenslatest_x1_004_174d2777 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.963164 | 0.266564  |


### Logs
```python
2022-05-28 15:47:47,272 P22511 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "bridge_type": "hadamard_product",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "ReLU",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "EDCN",
    "model_id": "EDCN_movielenslatest_x1_004_174d2777",
    "model_root": "./Movielens/EDCN_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_cross_layers": "2",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "temperature": "5",
    "test_data": "../data/Movielens/MovielensLatest_x1/test.csv",
    "train_data": "../data/Movielens/MovielensLatest_x1/train.csv",
    "use_hdf5": "True",
    "use_regulation_module": "True",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-05-28 15:47:47,273 P22511 INFO Set up feature encoder...
2022-05-28 15:47:47,273 P22511 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-05-28 15:47:47,273 P22511 INFO Loading data...
2022-05-28 15:47:47,276 P22511 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-05-28 15:47:47,305 P22511 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-05-28 15:47:47,314 P22511 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-05-28 15:47:47,314 P22511 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-05-28 15:47:47,314 P22511 INFO Loading train data done.
2022-05-28 15:47:51,290 P22511 INFO Total number of parameters: 904713.
2022-05-28 15:47:51,290 P22511 INFO Start training: 343 batches/epoch
2022-05-28 15:47:51,290 P22511 INFO ************ Epoch=1 start ************
2022-05-28 15:48:18,405 P22511 INFO [Metrics] AUC: 0.928540 - logloss: 0.299863
2022-05-28 15:48:18,405 P22511 INFO Save best model: monitor(max): 0.928540
2022-05-28 15:48:18,414 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:48:18,467 P22511 INFO Train loss: 0.429728
2022-05-28 15:48:18,467 P22511 INFO ************ Epoch=1 end ************
2022-05-28 15:48:45,196 P22511 INFO [Metrics] AUC: 0.932674 - logloss: 0.292298
2022-05-28 15:48:45,197 P22511 INFO Save best model: monitor(max): 0.932674
2022-05-28 15:48:45,206 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:48:45,255 P22511 INFO Train loss: 0.376689
2022-05-28 15:48:45,256 P22511 INFO ************ Epoch=2 end ************
2022-05-28 15:49:12,229 P22511 INFO [Metrics] AUC: 0.935830 - logloss: 0.285578
2022-05-28 15:49:12,230 P22511 INFO Save best model: monitor(max): 0.935830
2022-05-28 15:49:12,240 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:49:12,300 P22511 INFO Train loss: 0.374297
2022-05-28 15:49:12,300 P22511 INFO ************ Epoch=3 end ************
2022-05-28 15:49:38,900 P22511 INFO [Metrics] AUC: 0.937644 - logloss: 0.281143
2022-05-28 15:49:38,901 P22511 INFO Save best model: monitor(max): 0.937644
2022-05-28 15:49:38,907 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:49:38,955 P22511 INFO Train loss: 0.374402
2022-05-28 15:49:38,955 P22511 INFO ************ Epoch=4 end ************
2022-05-28 15:50:04,579 P22511 INFO [Metrics] AUC: 0.938891 - logloss: 0.278628
2022-05-28 15:50:04,580 P22511 INFO Save best model: monitor(max): 0.938891
2022-05-28 15:50:04,586 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:50:04,631 P22511 INFO Train loss: 0.375352
2022-05-28 15:50:04,632 P22511 INFO ************ Epoch=5 end ************
2022-05-28 15:50:30,665 P22511 INFO [Metrics] AUC: 0.940276 - logloss: 0.275588
2022-05-28 15:50:30,665 P22511 INFO Save best model: monitor(max): 0.940276
2022-05-28 15:50:30,672 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:50:30,719 P22511 INFO Train loss: 0.374859
2022-05-28 15:50:30,719 P22511 INFO ************ Epoch=6 end ************
2022-05-28 15:50:55,154 P22511 INFO [Metrics] AUC: 0.941133 - logloss: 0.273871
2022-05-28 15:50:55,155 P22511 INFO Save best model: monitor(max): 0.941133
2022-05-28 15:50:55,162 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:50:55,212 P22511 INFO Train loss: 0.374758
2022-05-28 15:50:55,212 P22511 INFO ************ Epoch=7 end ************
2022-05-28 15:51:16,323 P22511 INFO [Metrics] AUC: 0.941995 - logloss: 0.271908
2022-05-28 15:51:16,324 P22511 INFO Save best model: monitor(max): 0.941995
2022-05-28 15:51:16,333 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:51:16,372 P22511 INFO Train loss: 0.374387
2022-05-28 15:51:16,372 P22511 INFO ************ Epoch=8 end ************
2022-05-28 15:51:36,982 P22511 INFO [Metrics] AUC: 0.942815 - logloss: 0.270266
2022-05-28 15:51:36,983 P22511 INFO Save best model: monitor(max): 0.942815
2022-05-28 15:51:36,992 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:51:37,039 P22511 INFO Train loss: 0.373571
2022-05-28 15:51:37,040 P22511 INFO ************ Epoch=9 end ************
2022-05-28 15:51:54,948 P22511 INFO [Metrics] AUC: 0.942825 - logloss: 0.270406
2022-05-28 15:51:54,949 P22511 INFO Save best model: monitor(max): 0.942825
2022-05-28 15:51:54,959 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:51:55,016 P22511 INFO Train loss: 0.373253
2022-05-28 15:51:55,016 P22511 INFO ************ Epoch=10 end ************
2022-05-28 15:52:10,935 P22511 INFO [Metrics] AUC: 0.943509 - logloss: 0.268577
2022-05-28 15:52:10,936 P22511 INFO Save best model: monitor(max): 0.943509
2022-05-28 15:52:10,942 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:52:10,976 P22511 INFO Train loss: 0.372041
2022-05-28 15:52:10,977 P22511 INFO ************ Epoch=11 end ************
2022-05-28 15:52:27,122 P22511 INFO [Metrics] AUC: 0.943707 - logloss: 0.267621
2022-05-28 15:52:27,123 P22511 INFO Save best model: monitor(max): 0.943707
2022-05-28 15:52:27,129 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:52:27,163 P22511 INFO Train loss: 0.371287
2022-05-28 15:52:27,163 P22511 INFO ************ Epoch=12 end ************
2022-05-28 15:52:43,527 P22511 INFO [Metrics] AUC: 0.944389 - logloss: 0.266514
2022-05-28 15:52:43,528 P22511 INFO Save best model: monitor(max): 0.944389
2022-05-28 15:52:43,534 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:52:43,592 P22511 INFO Train loss: 0.370861
2022-05-28 15:52:43,592 P22511 INFO ************ Epoch=13 end ************
2022-05-28 15:53:00,272 P22511 INFO [Metrics] AUC: 0.944750 - logloss: 0.265138
2022-05-28 15:53:00,272 P22511 INFO Save best model: monitor(max): 0.944750
2022-05-28 15:53:00,279 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:53:00,329 P22511 INFO Train loss: 0.369923
2022-05-28 15:53:00,329 P22511 INFO ************ Epoch=14 end ************
2022-05-28 15:53:17,025 P22511 INFO [Metrics] AUC: 0.945072 - logloss: 0.265164
2022-05-28 15:53:17,026 P22511 INFO Save best model: monitor(max): 0.945072
2022-05-28 15:53:17,035 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:53:17,092 P22511 INFO Train loss: 0.370235
2022-05-28 15:53:17,092 P22511 INFO ************ Epoch=15 end ************
2022-05-28 15:53:30,075 P22511 INFO [Metrics] AUC: 0.945217 - logloss: 0.264727
2022-05-28 15:53:30,076 P22511 INFO Save best model: monitor(max): 0.945217
2022-05-28 15:53:30,085 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:53:30,134 P22511 INFO Train loss: 0.370115
2022-05-28 15:53:30,135 P22511 INFO ************ Epoch=16 end ************
2022-05-28 15:53:41,945 P22511 INFO [Metrics] AUC: 0.945351 - logloss: 0.264182
2022-05-28 15:53:41,946 P22511 INFO Save best model: monitor(max): 0.945351
2022-05-28 15:53:41,953 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:53:41,998 P22511 INFO Train loss: 0.369875
2022-05-28 15:53:41,998 P22511 INFO ************ Epoch=17 end ************
2022-05-28 15:53:53,544 P22511 INFO [Metrics] AUC: 0.945991 - logloss: 0.262799
2022-05-28 15:53:53,544 P22511 INFO Save best model: monitor(max): 0.945991
2022-05-28 15:53:53,554 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:53:53,601 P22511 INFO Train loss: 0.370336
2022-05-28 15:53:53,601 P22511 INFO ************ Epoch=18 end ************
2022-05-28 15:54:05,460 P22511 INFO [Metrics] AUC: 0.946034 - logloss: 0.262727
2022-05-28 15:54:05,460 P22511 INFO Save best model: monitor(max): 0.946034
2022-05-28 15:54:05,467 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:54:05,517 P22511 INFO Train loss: 0.369664
2022-05-28 15:54:05,517 P22511 INFO ************ Epoch=19 end ************
2022-05-28 15:54:17,049 P22511 INFO [Metrics] AUC: 0.946405 - logloss: 0.261850
2022-05-28 15:54:17,050 P22511 INFO Save best model: monitor(max): 0.946405
2022-05-28 15:54:17,057 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:54:17,105 P22511 INFO Train loss: 0.369559
2022-05-28 15:54:17,105 P22511 INFO ************ Epoch=20 end ************
2022-05-28 15:54:28,750 P22511 INFO [Metrics] AUC: 0.946755 - logloss: 0.261641
2022-05-28 15:54:28,751 P22511 INFO Save best model: monitor(max): 0.946755
2022-05-28 15:54:28,758 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:54:28,796 P22511 INFO Train loss: 0.369123
2022-05-28 15:54:28,797 P22511 INFO ************ Epoch=21 end ************
2022-05-28 15:54:40,641 P22511 INFO [Metrics] AUC: 0.947024 - logloss: 0.260763
2022-05-28 15:54:40,642 P22511 INFO Save best model: monitor(max): 0.947024
2022-05-28 15:54:40,649 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:54:40,705 P22511 INFO Train loss: 0.368719
2022-05-28 15:54:40,705 P22511 INFO ************ Epoch=22 end ************
2022-05-28 15:54:52,565 P22511 INFO [Metrics] AUC: 0.947184 - logloss: 0.259562
2022-05-28 15:54:52,566 P22511 INFO Save best model: monitor(max): 0.947184
2022-05-28 15:54:52,573 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:54:52,620 P22511 INFO Train loss: 0.367365
2022-05-28 15:54:52,620 P22511 INFO ************ Epoch=23 end ************
2022-05-28 15:55:04,607 P22511 INFO [Metrics] AUC: 0.947595 - logloss: 0.258567
2022-05-28 15:55:04,608 P22511 INFO Save best model: monitor(max): 0.947595
2022-05-28 15:55:04,615 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:55:04,679 P22511 INFO Train loss: 0.367872
2022-05-28 15:55:04,679 P22511 INFO ************ Epoch=24 end ************
2022-05-28 15:55:16,784 P22511 INFO [Metrics] AUC: 0.947587 - logloss: 0.259106
2022-05-28 15:55:16,785 P22511 INFO Monitor(max) STOP: 0.947587 !
2022-05-28 15:55:16,785 P22511 INFO Reduce learning rate on plateau: 0.000100
2022-05-28 15:55:16,785 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:55:16,836 P22511 INFO Train loss: 0.367041
2022-05-28 15:55:16,836 P22511 INFO ************ Epoch=25 end ************
2022-05-28 15:55:28,706 P22511 INFO [Metrics] AUC: 0.957487 - logloss: 0.235871
2022-05-28 15:55:28,707 P22511 INFO Save best model: monitor(max): 0.957487
2022-05-28 15:55:28,716 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:55:28,779 P22511 INFO Train loss: 0.301114
2022-05-28 15:55:28,780 P22511 INFO ************ Epoch=26 end ************
2022-05-28 15:55:40,759 P22511 INFO [Metrics] AUC: 0.960400 - logloss: 0.230553
2022-05-28 15:55:40,760 P22511 INFO Save best model: monitor(max): 0.960400
2022-05-28 15:55:40,768 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:55:40,828 P22511 INFO Train loss: 0.246520
2022-05-28 15:55:40,828 P22511 INFO ************ Epoch=27 end ************
2022-05-28 15:55:52,835 P22511 INFO [Metrics] AUC: 0.961115 - logloss: 0.232660
2022-05-28 15:55:52,835 P22511 INFO Save best model: monitor(max): 0.961115
2022-05-28 15:55:52,844 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:55:52,901 P22511 INFO Train loss: 0.220978
2022-05-28 15:55:52,901 P22511 INFO ************ Epoch=28 end ************
2022-05-28 15:56:05,329 P22511 INFO [Metrics] AUC: 0.961501 - logloss: 0.235838
2022-05-28 15:56:05,329 P22511 INFO Save best model: monitor(max): 0.961501
2022-05-28 15:56:05,339 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:56:05,385 P22511 INFO Train loss: 0.206129
2022-05-28 15:56:05,385 P22511 INFO ************ Epoch=29 end ************
2022-05-28 15:56:17,317 P22511 INFO [Metrics] AUC: 0.961468 - logloss: 0.240652
2022-05-28 15:56:17,317 P22511 INFO Monitor(max) STOP: 0.961468 !
2022-05-28 15:56:17,318 P22511 INFO Reduce learning rate on plateau: 0.000010
2022-05-28 15:56:17,318 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:56:17,377 P22511 INFO Train loss: 0.194864
2022-05-28 15:56:17,378 P22511 INFO ************ Epoch=30 end ************
2022-05-28 15:56:29,353 P22511 INFO [Metrics] AUC: 0.961821 - logloss: 0.243676
2022-05-28 15:56:29,354 P22511 INFO Save best model: monitor(max): 0.961821
2022-05-28 15:56:29,363 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:56:29,417 P22511 INFO Train loss: 0.170423
2022-05-28 15:56:29,417 P22511 INFO ************ Epoch=31 end ************
2022-05-28 15:56:41,705 P22511 INFO [Metrics] AUC: 0.962163 - logloss: 0.245806
2022-05-28 15:56:41,706 P22511 INFO Save best model: monitor(max): 0.962163
2022-05-28 15:56:41,716 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:56:41,768 P22511 INFO Train loss: 0.162935
2022-05-28 15:56:41,768 P22511 INFO ************ Epoch=32 end ************
2022-05-28 15:56:53,821 P22511 INFO [Metrics] AUC: 0.962439 - logloss: 0.247695
2022-05-28 15:56:53,822 P22511 INFO Save best model: monitor(max): 0.962439
2022-05-28 15:56:53,831 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:56:53,865 P22511 INFO Train loss: 0.157856
2022-05-28 15:56:53,865 P22511 INFO ************ Epoch=33 end ************
2022-05-28 15:57:05,840 P22511 INFO [Metrics] AUC: 0.962619 - logloss: 0.249953
2022-05-28 15:57:05,841 P22511 INFO Save best model: monitor(max): 0.962619
2022-05-28 15:57:05,851 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:57:05,902 P22511 INFO Train loss: 0.152890
2022-05-28 15:57:05,902 P22511 INFO ************ Epoch=34 end ************
2022-05-28 15:57:17,716 P22511 INFO [Metrics] AUC: 0.962784 - logloss: 0.252142
2022-05-28 15:57:17,717 P22511 INFO Save best model: monitor(max): 0.962784
2022-05-28 15:57:17,726 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:57:17,763 P22511 INFO Train loss: 0.149161
2022-05-28 15:57:17,764 P22511 INFO ************ Epoch=35 end ************
2022-05-28 15:57:29,961 P22511 INFO [Metrics] AUC: 0.962887 - logloss: 0.253554
2022-05-28 15:57:29,962 P22511 INFO Save best model: monitor(max): 0.962887
2022-05-28 15:57:29,972 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:57:30,027 P22511 INFO Train loss: 0.145720
2022-05-28 15:57:30,027 P22511 INFO ************ Epoch=36 end ************
2022-05-28 15:57:39,243 P22511 INFO [Metrics] AUC: 0.962960 - logloss: 0.255774
2022-05-28 15:57:39,244 P22511 INFO Save best model: monitor(max): 0.962960
2022-05-28 15:57:39,251 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:57:39,297 P22511 INFO Train loss: 0.142715
2022-05-28 15:57:39,297 P22511 INFO ************ Epoch=37 end ************
2022-05-28 15:57:51,788 P22511 INFO [Metrics] AUC: 0.963115 - logloss: 0.257655
2022-05-28 15:57:51,789 P22511 INFO Save best model: monitor(max): 0.963115
2022-05-28 15:57:51,798 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:57:51,860 P22511 INFO Train loss: 0.140109
2022-05-28 15:57:51,860 P22511 INFO ************ Epoch=38 end ************
2022-05-28 15:58:04,108 P22511 INFO [Metrics] AUC: 0.963133 - logloss: 0.259409
2022-05-28 15:58:04,109 P22511 INFO Save best model: monitor(max): 0.963133
2022-05-28 15:58:04,118 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:58:04,175 P22511 INFO Train loss: 0.137792
2022-05-28 15:58:04,175 P22511 INFO ************ Epoch=39 end ************
2022-05-28 15:58:15,781 P22511 INFO [Metrics] AUC: 0.963173 - logloss: 0.261547
2022-05-28 15:58:15,781 P22511 INFO Save best model: monitor(max): 0.963173
2022-05-28 15:58:15,788 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:58:15,823 P22511 INFO Train loss: 0.135698
2022-05-28 15:58:15,823 P22511 INFO ************ Epoch=40 end ************
2022-05-28 15:58:27,674 P22511 INFO [Metrics] AUC: 0.963219 - logloss: 0.263066
2022-05-28 15:58:27,675 P22511 INFO Save best model: monitor(max): 0.963219
2022-05-28 15:58:27,682 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:58:27,758 P22511 INFO Train loss: 0.133266
2022-05-28 15:58:27,759 P22511 INFO ************ Epoch=41 end ************
2022-05-28 15:58:39,708 P22511 INFO [Metrics] AUC: 0.963204 - logloss: 0.265473
2022-05-28 15:58:39,709 P22511 INFO Monitor(max) STOP: 0.963204 !
2022-05-28 15:58:39,709 P22511 INFO Reduce learning rate on plateau: 0.000001
2022-05-28 15:58:39,709 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:58:39,746 P22511 INFO Train loss: 0.131429
2022-05-28 15:58:39,746 P22511 INFO ************ Epoch=42 end ************
2022-05-28 15:58:51,158 P22511 INFO [Metrics] AUC: 0.963229 - logloss: 0.265166
2022-05-28 15:58:51,159 P22511 INFO Save best model: monitor(max): 0.963229
2022-05-28 15:58:51,169 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:58:51,214 P22511 INFO Train loss: 0.128284
2022-05-28 15:58:51,214 P22511 INFO ************ Epoch=43 end ************
2022-05-28 15:59:01,270 P22511 INFO [Metrics] AUC: 0.963277 - logloss: 0.265796
2022-05-28 15:59:01,271 P22511 INFO Save best model: monitor(max): 0.963277
2022-05-28 15:59:01,280 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:59:01,326 P22511 INFO Train loss: 0.127886
2022-05-28 15:59:01,326 P22511 INFO ************ Epoch=44 end ************
2022-05-28 15:59:14,591 P22511 INFO [Metrics] AUC: 0.963270 - logloss: 0.266192
2022-05-28 15:59:14,592 P22511 INFO Monitor(max) STOP: 0.963270 !
2022-05-28 15:59:14,592 P22511 INFO Reduce learning rate on plateau: 0.000001
2022-05-28 15:59:14,592 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:59:14,648 P22511 INFO Train loss: 0.127562
2022-05-28 15:59:14,648 P22511 INFO ************ Epoch=45 end ************
2022-05-28 15:59:26,948 P22511 INFO [Metrics] AUC: 0.963283 - logloss: 0.266772
2022-05-28 15:59:26,949 P22511 INFO Save best model: monitor(max): 0.963283
2022-05-28 15:59:26,958 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:59:27,006 P22511 INFO Train loss: 0.127666
2022-05-28 15:59:27,006 P22511 INFO ************ Epoch=46 end ************
2022-05-28 15:59:38,782 P22511 INFO [Metrics] AUC: 0.963323 - logloss: 0.266032
2022-05-28 15:59:38,783 P22511 INFO Save best model: monitor(max): 0.963323
2022-05-28 15:59:38,790 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:59:38,835 P22511 INFO Train loss: 0.127174
2022-05-28 15:59:38,835 P22511 INFO ************ Epoch=47 end ************
2022-05-28 15:59:50,873 P22511 INFO [Metrics] AUC: 0.963312 - logloss: 0.267036
2022-05-28 15:59:50,874 P22511 INFO Monitor(max) STOP: 0.963312 !
2022-05-28 15:59:50,874 P22511 INFO Reduce learning rate on plateau: 0.000001
2022-05-28 15:59:50,874 P22511 INFO --- 343/343 batches finished ---
2022-05-28 15:59:50,916 P22511 INFO Train loss: 0.127124
2022-05-28 15:59:50,917 P22511 INFO ************ Epoch=48 end ************
2022-05-28 16:00:03,058 P22511 INFO [Metrics] AUC: 0.963305 - logloss: 0.267034
2022-05-28 16:00:03,059 P22511 INFO Monitor(max) STOP: 0.963305 !
2022-05-28 16:00:03,059 P22511 INFO Reduce learning rate on plateau: 0.000001
2022-05-28 16:00:03,059 P22511 INFO Early stopping at epoch=49
2022-05-28 16:00:03,059 P22511 INFO --- 343/343 batches finished ---
2022-05-28 16:00:03,115 P22511 INFO Train loss: 0.126800
2022-05-28 16:00:03,115 P22511 INFO Training finished.
2022-05-28 16:00:03,115 P22511 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Movielens/EDCN_movielenslatest_x1/movielenslatest_x1_cd32d937/EDCN_movielenslatest_x1_004_174d2777.model
2022-05-28 16:00:06,180 P22511 INFO ****** Validation evaluation ******
2022-05-28 16:00:07,679 P22511 INFO [Metrics] AUC: 0.963323 - logloss: 0.266032
2022-05-28 16:00:07,724 P22511 INFO ******** Test evaluation ********
2022-05-28 16:00:07,725 P22511 INFO Loading data...
2022-05-28 16:00:07,725 P22511 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-05-28 16:00:07,729 P22511 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-05-28 16:00:07,730 P22511 INFO Loading test data done.
2022-05-28 16:00:08,627 P22511 INFO [Metrics] AUC: 0.963164 - logloss: 0.266564

```
