## DCNv2_movielenslatest_x1

A hands-on guide to run the DCNv2 model on the Movielenslatest_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DCNv2](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DCNv2.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DCNv2_movielenslatest_x1_tuner_config_01](./DCNv2_movielenslatest_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd DCNv2_movielenslatest_x1
   nohup python run_expid.py --config ./DCNv2_movielenslatest_x1_tuner_config_01 --expid DCNv2_movielenslatest_x1_016_98ea1c72 --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

Total 5 runs:

| Runs | AUC              | logloss          |
|:----:|:----------------:|:----------------:|
| 1    | 0.969098         | 0.214736         |
| 2    | 0.968205         | 0.217647         |
| 3    | 0.968267         | 0.216530         |
| 4    | 0.968876         | 0.214814         |
| 5    | 0.968838         | 0.216954         |
| Avg  | 0.968657         | 0.216136         |
| Std  | &#177;0.00035542 | &#177;0.00116749 |

### Logs

```python
2022-11-01 20:06:41,981 P19226 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "dnn_activations": "relu",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
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
    "model_id": "DCNv2_movielenslatest_x1_016_98ea1c72",
    "model_root": "./Movielens/DCN_movielenslatest_x1/",
    "model_structure": "parallel",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_cross_layers": "5",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_dnn_hidden_units": "[400, 400, 400]",
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
2022-11-01 20:06:41,982 P19226 INFO Set up feature encoder...
2022-11-01 20:06:41,982 P19226 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-11-01 20:06:41,982 P19226 INFO Loading data...
2022-11-01 20:06:41,984 P19226 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-11-01 20:06:42,009 P19226 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-11-01 20:06:42,016 P19226 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-11-01 20:06:42,016 P19226 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-11-01 20:06:42,016 P19226 INFO Loading train data done.
2022-11-01 20:06:45,222 P19226 INFO Total number of parameters: 1243071.
2022-11-01 20:06:45,223 P19226 INFO Start training: 343 batches/epoch
2022-11-01 20:06:45,223 P19226 INFO ************ Epoch=1 start ************
2022-11-01 20:06:56,390 P19226 INFO [Metrics] AUC: 0.935047 - logloss: 0.293483
2022-11-01 20:06:56,391 P19226 INFO Save best model: monitor(max): 0.935047
2022-11-01 20:06:56,401 P19226 INFO --- 343/343 batches finished ---
2022-11-01 20:06:56,442 P19226 INFO Train loss: 0.382016
2022-11-01 20:06:56,443 P19226 INFO ************ Epoch=1 end ************
2022-11-01 20:07:07,574 P19226 INFO [Metrics] AUC: 0.945515 - logloss: 0.264626
2022-11-01 20:07:07,575 P19226 INFO Save best model: monitor(max): 0.945515
2022-11-01 20:07:07,586 P19226 INFO --- 343/343 batches finished ---
2022-11-01 20:07:07,633 P19226 INFO Train loss: 0.371308
2022-11-01 20:07:07,633 P19226 INFO ************ Epoch=2 end ************
2022-11-01 20:07:18,636 P19226 INFO [Metrics] AUC: 0.948241 - logloss: 0.258100
2022-11-01 20:07:18,637 P19226 INFO Save best model: monitor(max): 0.948241
2022-11-01 20:07:18,648 P19226 INFO --- 343/343 batches finished ---
2022-11-01 20:07:18,692 P19226 INFO Train loss: 0.370473
2022-11-01 20:07:18,692 P19226 INFO ************ Epoch=3 end ************
2022-11-01 20:07:29,756 P19226 INFO [Metrics] AUC: 0.950849 - logloss: 0.251910
2022-11-01 20:07:29,757 P19226 INFO Save best model: monitor(max): 0.950849
2022-11-01 20:07:29,765 P19226 INFO --- 343/343 batches finished ---
2022-11-01 20:07:29,815 P19226 INFO Train loss: 0.372997
2022-11-01 20:07:29,815 P19226 INFO ************ Epoch=4 end ************
2022-11-01 20:07:40,774 P19226 INFO [Metrics] AUC: 0.952387 - logloss: 0.247171
2022-11-01 20:07:40,775 P19226 INFO Save best model: monitor(max): 0.952387
2022-11-01 20:07:40,786 P19226 INFO --- 343/343 batches finished ---
2022-11-01 20:07:40,844 P19226 INFO Train loss: 0.375214
2022-11-01 20:07:40,844 P19226 INFO ************ Epoch=5 end ************
2022-11-01 20:07:51,655 P19226 INFO [Metrics] AUC: 0.953951 - logloss: 0.239795
2022-11-01 20:07:51,656 P19226 INFO Save best model: monitor(max): 0.953951
2022-11-01 20:07:51,664 P19226 INFO --- 343/343 batches finished ---
2022-11-01 20:07:51,716 P19226 INFO Train loss: 0.376620
2022-11-01 20:07:51,716 P19226 INFO ************ Epoch=6 end ************
2022-11-01 20:08:02,515 P19226 INFO [Metrics] AUC: 0.954787 - logloss: 0.238054
2022-11-01 20:08:02,515 P19226 INFO Save best model: monitor(max): 0.954787
2022-11-01 20:08:02,523 P19226 INFO --- 343/343 batches finished ---
2022-11-01 20:08:02,566 P19226 INFO Train loss: 0.377080
2022-11-01 20:08:02,566 P19226 INFO ************ Epoch=7 end ************
2022-11-01 20:08:13,569 P19226 INFO [Metrics] AUC: 0.955095 - logloss: 0.237576
2022-11-01 20:08:13,570 P19226 INFO Save best model: monitor(max): 0.955095
2022-11-01 20:08:13,580 P19226 INFO --- 343/343 batches finished ---
2022-11-01 20:08:13,619 P19226 INFO Train loss: 0.377740
2022-11-01 20:08:13,619 P19226 INFO ************ Epoch=8 end ************
2022-11-01 20:08:24,690 P19226 INFO [Metrics] AUC: 0.955781 - logloss: 0.236796
2022-11-01 20:08:24,691 P19226 INFO Save best model: monitor(max): 0.955781
2022-11-01 20:08:24,701 P19226 INFO --- 343/343 batches finished ---
2022-11-01 20:08:24,747 P19226 INFO Train loss: 0.377764
2022-11-01 20:08:24,747 P19226 INFO ************ Epoch=9 end ************
2022-11-01 20:08:35,753 P19226 INFO [Metrics] AUC: 0.956402 - logloss: 0.233103
2022-11-01 20:08:35,754 P19226 INFO Save best model: monitor(max): 0.956402
2022-11-01 20:08:35,765 P19226 INFO --- 343/343 batches finished ---
2022-11-01 20:08:35,818 P19226 INFO Train loss: 0.379035
2022-11-01 20:08:35,818 P19226 INFO ************ Epoch=10 end ************
2022-11-01 20:08:45,044 P19226 INFO [Metrics] AUC: 0.956446 - logloss: 0.233880
2022-11-01 20:08:45,044 P19226 INFO Save best model: monitor(max): 0.956446
2022-11-01 20:08:45,053 P19226 INFO --- 343/343 batches finished ---
2022-11-01 20:08:45,097 P19226 INFO Train loss: 0.379735
2022-11-01 20:08:45,097 P19226 INFO ************ Epoch=11 end ************
2022-11-01 20:08:52,274 P19226 INFO [Metrics] AUC: 0.956138 - logloss: 0.234126
2022-11-01 20:08:52,275 P19226 INFO Monitor(max) STOP: 0.956138 !
2022-11-01 20:08:52,275 P19226 INFO Reduce learning rate on plateau: 0.000100
2022-11-01 20:08:52,275 P19226 INFO --- 343/343 batches finished ---
2022-11-01 20:08:52,322 P19226 INFO Train loss: 0.379465
2022-11-01 20:08:52,322 P19226 INFO ************ Epoch=12 end ************
2022-11-01 20:08:59,322 P19226 INFO [Metrics] AUC: 0.967827 - logloss: 0.205812
2022-11-01 20:08:59,323 P19226 INFO Save best model: monitor(max): 0.967827
2022-11-01 20:08:59,331 P19226 INFO --- 343/343 batches finished ---
2022-11-01 20:08:59,371 P19226 INFO Train loss: 0.277900
2022-11-01 20:08:59,371 P19226 INFO ************ Epoch=13 end ************
2022-11-01 20:09:06,405 P19226 INFO [Metrics] AUC: 0.969117 - logloss: 0.215212
2022-11-01 20:09:06,406 P19226 INFO Save best model: monitor(max): 0.969117
2022-11-01 20:09:06,422 P19226 INFO --- 343/343 batches finished ---
2022-11-01 20:09:06,465 P19226 INFO Train loss: 0.189714
2022-11-01 20:09:06,465 P19226 INFO ************ Epoch=14 end ************
2022-11-01 20:09:13,949 P19226 INFO [Metrics] AUC: 0.967712 - logloss: 0.238754
2022-11-01 20:09:13,950 P19226 INFO Monitor(max) STOP: 0.967712 !
2022-11-01 20:09:13,950 P19226 INFO Reduce learning rate on plateau: 0.000010
2022-11-01 20:09:13,950 P19226 INFO --- 343/343 batches finished ---
2022-11-01 20:09:13,995 P19226 INFO Train loss: 0.143900
2022-11-01 20:09:13,995 P19226 INFO ************ Epoch=15 end ************
2022-11-01 20:09:21,254 P19226 INFO [Metrics] AUC: 0.967278 - logloss: 0.251545
2022-11-01 20:09:21,255 P19226 INFO Monitor(max) STOP: 0.967278 !
2022-11-01 20:09:21,255 P19226 INFO Reduce learning rate on plateau: 0.000001
2022-11-01 20:09:21,255 P19226 INFO Early stopping at epoch=16
2022-11-01 20:09:21,255 P19226 INFO --- 343/343 batches finished ---
2022-11-01 20:09:21,307 P19226 INFO Train loss: 0.111731
2022-11-01 20:09:21,307 P19226 INFO Training finished.
2022-11-01 20:09:21,307 P19226 INFO Load best model: /home/FuxiCTR/benchmarks/Movielens/DCN_movielenslatest_x1/movielenslatest_x1_cd32d937/DCNv2_movielenslatest_x1_016_98ea1c72.model
2022-11-01 20:09:21,323 P19226 INFO ****** Validation evaluation ******
2022-11-01 20:09:22,568 P19226 INFO [Metrics] AUC: 0.969117 - logloss: 0.215212
2022-11-01 20:09:22,636 P19226 INFO ******** Test evaluation ********
2022-11-01 20:09:22,636 P19226 INFO Loading data...
2022-11-01 20:09:22,637 P19226 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-11-01 20:09:22,643 P19226 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-11-01 20:09:22,643 P19226 INFO Loading test data done.
2022-11-01 20:09:23,273 P19226 INFO [Metrics] AUC: 0.969098 - logloss: 0.214736
```
