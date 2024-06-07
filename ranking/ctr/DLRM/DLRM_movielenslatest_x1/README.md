## DLRM_movielenslatest_x1

A hands-on guide to run the DLRM model on the MovielensLatest_x1 dataset.

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

We use [FuxiCTR-v1.2.1](https://github.com/reczoo/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [DLRM](https://github.com/reczoo/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/DLRM.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DLRM_movielenslatest_x1_tuner_config_04](./DLRM_movielenslatest_x1_tuner_config_04). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DLRM_movielenslatest_x1
    nohup python run_expid.py --config ./DLRM_movielenslatest_x1_tuner_config_04 --expid DLRM_movielenslatest_x1_002_333e0a39 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.969071 | 0.214993  |


### Logs
```python
2022-05-27 17:45:47,993 P56424 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "bottom_mlp_activations": "ReLU",
    "bottom_mlp_dropout": "0",
    "bottom_mlp_units": "None",
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
    "interaction_op": "cat",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DLRM",
    "model_id": "DLRM_movielenslatest_x1_002_333e0a39",
    "model_root": "./Movielens/DLRM_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Movielens/MovielensLatest_x1/test.csv",
    "top_mlp_activations": "ReLU",
    "top_mlp_dropout": "0.2",
    "top_mlp_units": "[400, 400, 400]",
    "train_data": "../data/Movielens/MovielensLatest_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-05-27 17:45:47,994 P56424 INFO Set up feature encoder...
2022-05-27 17:45:47,995 P56424 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-05-27 17:45:47,995 P56424 INFO Loading data...
2022-05-27 17:45:48,004 P56424 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-05-27 17:45:48,040 P56424 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-05-27 17:45:48,053 P56424 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-05-27 17:45:48,053 P56424 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-05-27 17:45:48,053 P56424 INFO Loading train data done.
2022-05-27 17:45:50,966 P56424 INFO Total number of parameters: 1238391.
2022-05-27 17:45:50,966 P56424 INFO Start training: 343 batches/epoch
2022-05-27 17:45:50,966 P56424 INFO ************ Epoch=1 start ************
2022-05-27 17:46:01,289 P56424 INFO [Metrics] AUC: 0.934995 - logloss: 0.291341
2022-05-27 17:46:01,289 P56424 INFO Save best model: monitor(max): 0.934995
2022-05-27 17:46:01,297 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:46:01,354 P56424 INFO Train loss: 0.377448
2022-05-27 17:46:01,354 P56424 INFO ************ Epoch=1 end ************
2022-05-27 17:46:11,537 P56424 INFO [Metrics] AUC: 0.945625 - logloss: 0.265396
2022-05-27 17:46:11,538 P56424 INFO Save best model: monitor(max): 0.945625
2022-05-27 17:46:11,549 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:46:11,595 P56424 INFO Train loss: 0.362417
2022-05-27 17:46:11,595 P56424 INFO ************ Epoch=2 end ************
2022-05-27 17:46:21,815 P56424 INFO [Metrics] AUC: 0.949287 - logloss: 0.253220
2022-05-27 17:46:21,815 P56424 INFO Save best model: monitor(max): 0.949287
2022-05-27 17:46:21,823 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:46:21,858 P56424 INFO Train loss: 0.364127
2022-05-27 17:46:21,858 P56424 INFO ************ Epoch=3 end ************
2022-05-27 17:46:32,344 P56424 INFO [Metrics] AUC: 0.950830 - logloss: 0.248907
2022-05-27 17:46:32,345 P56424 INFO Save best model: monitor(max): 0.950830
2022-05-27 17:46:32,356 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:46:32,417 P56424 INFO Train loss: 0.368697
2022-05-27 17:46:32,417 P56424 INFO ************ Epoch=4 end ************
2022-05-27 17:46:42,475 P56424 INFO [Metrics] AUC: 0.951979 - logloss: 0.244560
2022-05-27 17:46:42,475 P56424 INFO Save best model: monitor(max): 0.951979
2022-05-27 17:46:42,484 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:46:42,533 P56424 INFO Train loss: 0.371929
2022-05-27 17:46:42,533 P56424 INFO ************ Epoch=5 end ************
2022-05-27 17:46:52,494 P56424 INFO [Metrics] AUC: 0.953028 - logloss: 0.246918
2022-05-27 17:46:52,495 P56424 INFO Save best model: monitor(max): 0.953028
2022-05-27 17:46:52,504 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:46:52,541 P56424 INFO Train loss: 0.375755
2022-05-27 17:46:52,541 P56424 INFO ************ Epoch=6 end ************
2022-05-27 17:47:02,352 P56424 INFO [Metrics] AUC: 0.953333 - logloss: 0.242936
2022-05-27 17:47:02,353 P56424 INFO Save best model: monitor(max): 0.953333
2022-05-27 17:47:02,361 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:47:02,408 P56424 INFO Train loss: 0.375239
2022-05-27 17:47:02,408 P56424 INFO ************ Epoch=7 end ************
2022-05-27 17:47:12,267 P56424 INFO [Metrics] AUC: 0.954451 - logloss: 0.242778
2022-05-27 17:47:12,268 P56424 INFO Save best model: monitor(max): 0.954451
2022-05-27 17:47:12,278 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:47:12,320 P56424 INFO Train loss: 0.376126
2022-05-27 17:47:12,320 P56424 INFO ************ Epoch=8 end ************
2022-05-27 17:47:22,250 P56424 INFO [Metrics] AUC: 0.955334 - logloss: 0.239819
2022-05-27 17:47:22,251 P56424 INFO Save best model: monitor(max): 0.955334
2022-05-27 17:47:22,258 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:47:22,322 P56424 INFO Train loss: 0.377568
2022-05-27 17:47:22,322 P56424 INFO ************ Epoch=9 end ************
2022-05-27 17:47:32,017 P56424 INFO [Metrics] AUC: 0.955787 - logloss: 0.234790
2022-05-27 17:47:32,018 P56424 INFO Save best model: monitor(max): 0.955787
2022-05-27 17:47:32,025 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:47:32,067 P56424 INFO Train loss: 0.376616
2022-05-27 17:47:32,067 P56424 INFO ************ Epoch=10 end ************
2022-05-27 17:47:42,021 P56424 INFO [Metrics] AUC: 0.956218 - logloss: 0.235922
2022-05-27 17:47:42,022 P56424 INFO Save best model: monitor(max): 0.956218
2022-05-27 17:47:42,033 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:47:42,082 P56424 INFO Train loss: 0.376828
2022-05-27 17:47:42,082 P56424 INFO ************ Epoch=11 end ************
2022-05-27 17:47:48,606 P56424 INFO [Metrics] AUC: 0.956553 - logloss: 0.236383
2022-05-27 17:47:48,607 P56424 INFO Save best model: monitor(max): 0.956553
2022-05-27 17:47:48,615 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:47:48,667 P56424 INFO Train loss: 0.378037
2022-05-27 17:47:48,667 P56424 INFO ************ Epoch=12 end ************
2022-05-27 17:47:54,978 P56424 INFO [Metrics] AUC: 0.956843 - logloss: 0.235801
2022-05-27 17:47:54,979 P56424 INFO Save best model: monitor(max): 0.956843
2022-05-27 17:47:54,990 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:47:55,040 P56424 INFO Train loss: 0.377164
2022-05-27 17:47:55,041 P56424 INFO ************ Epoch=13 end ************
2022-05-27 17:48:01,359 P56424 INFO [Metrics] AUC: 0.957092 - logloss: 0.231618
2022-05-27 17:48:01,361 P56424 INFO Save best model: monitor(max): 0.957092
2022-05-27 17:48:01,371 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:48:01,409 P56424 INFO Train loss: 0.376691
2022-05-27 17:48:01,409 P56424 INFO ************ Epoch=14 end ************
2022-05-27 17:48:07,426 P56424 INFO [Metrics] AUC: 0.956775 - logloss: 0.231541
2022-05-27 17:48:07,426 P56424 INFO Monitor(max) STOP: 0.956775 !
2022-05-27 17:48:07,426 P56424 INFO Reduce learning rate on plateau: 0.000100
2022-05-27 17:48:07,427 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:48:07,469 P56424 INFO Train loss: 0.376863
2022-05-27 17:48:07,470 P56424 INFO ************ Epoch=15 end ************
2022-05-27 17:48:13,584 P56424 INFO [Metrics] AUC: 0.967864 - logloss: 0.206055
2022-05-27 17:48:13,585 P56424 INFO Save best model: monitor(max): 0.967864
2022-05-27 17:48:13,596 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:48:13,649 P56424 INFO Train loss: 0.273575
2022-05-27 17:48:13,649 P56424 INFO ************ Epoch=16 end ************
2022-05-27 17:48:19,686 P56424 INFO [Metrics] AUC: 0.969119 - logloss: 0.214729
2022-05-27 17:48:19,687 P56424 INFO Save best model: monitor(max): 0.969119
2022-05-27 17:48:19,695 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:48:19,745 P56424 INFO Train loss: 0.185297
2022-05-27 17:48:19,745 P56424 INFO ************ Epoch=17 end ************
2022-05-27 17:48:25,743 P56424 INFO [Metrics] AUC: 0.967576 - logloss: 0.240492
2022-05-27 17:48:25,744 P56424 INFO Monitor(max) STOP: 0.967576 !
2022-05-27 17:48:25,744 P56424 INFO Reduce learning rate on plateau: 0.000010
2022-05-27 17:48:25,744 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:48:25,795 P56424 INFO Train loss: 0.140979
2022-05-27 17:48:25,795 P56424 INFO ************ Epoch=18 end ************
2022-05-27 17:48:31,748 P56424 INFO [Metrics] AUC: 0.967095 - logloss: 0.253392
2022-05-27 17:48:31,748 P56424 INFO Monitor(max) STOP: 0.967095 !
2022-05-27 17:48:31,749 P56424 INFO Reduce learning rate on plateau: 0.000001
2022-05-27 17:48:31,749 P56424 INFO Early stopping at epoch=19
2022-05-27 17:48:31,749 P56424 INFO --- 343/343 batches finished ---
2022-05-27 17:48:31,795 P56424 INFO Train loss: 0.109826
2022-05-27 17:48:31,795 P56424 INFO Training finished.
2022-05-27 17:48:31,795 P56424 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Movielens/DLRM_movielenslatest_x1/movielenslatest_x1_cd32d937/DLRM_movielenslatest_x1_002_333e0a39.model
2022-05-27 17:48:34,455 P56424 INFO ****** Validation evaluation ******
2022-05-27 17:48:35,720 P56424 INFO [Metrics] AUC: 0.969119 - logloss: 0.214729
2022-05-27 17:48:35,752 P56424 INFO ******** Test evaluation ********
2022-05-27 17:48:35,752 P56424 INFO Loading data...
2022-05-27 17:48:35,753 P56424 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-05-27 17:48:35,756 P56424 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-05-27 17:48:35,757 P56424 INFO Loading test data done.
2022-05-27 17:48:36,507 P56424 INFO [Metrics] AUC: 0.969071 - logloss: 0.214993

```
