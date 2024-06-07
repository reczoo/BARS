## DCN_movielenslatest_x1

A hands-on guide to run the DCN model on the Movielenslatest_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DCN](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DCN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DCN_movielenslatest_x1_tuner_config_01](./DCN_movielenslatest_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd DCN_movielenslatest_x1
   nohup python run_expid.py --config ./DCN_movielenslatest_x1_tuner_config_01 --expid DCN_movielenslatest_x1_017_4810b636 --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

Total 5 runs:

| Runs | AUC              | logloss          |
|:----:|:----------------:|:----------------:|
| 1    | 0.968719         | 0.215987         |
| 2    | 0.967358         | 0.214489         |
| 3    | 0.968355         | 0.213700         |
| 4    | 0.968357         | 0.208969         |
| 5    | 0.968990         | 0.215492         |
| Avg  | 0.968356         | 0.213727         |
| Std  | &#177;0.00055312 | &#177;0.00250770 |

### Logs

```python
2022-10-29 15:05:06,575 P24143 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "crossing_layers": "4",
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
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCN",
    "model_id": "DCN_movielenslatest_x1_017_4810b636",
    "model_root": "./Movielens/DCN_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
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
    "train_data": "../data/Movielens/MovielensLatest_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-10-29 15:05:06,576 P24143 INFO Set up feature encoder...
2022-10-29 15:05:06,576 P24143 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-10-29 15:05:06,576 P24143 INFO Loading data...
2022-10-29 15:05:06,578 P24143 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-10-29 15:05:06,604 P24143 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-10-29 15:05:06,612 P24143 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-10-29 15:05:06,612 P24143 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-10-29 15:05:06,612 P24143 INFO Loading train data done.
2022-10-29 15:05:09,805 P24143 INFO Total number of parameters: 1238661.
2022-10-29 15:05:09,806 P24143 INFO Start training: 343 batches/epoch
2022-10-29 15:05:09,806 P24143 INFO ************ Epoch=1 start ************
2022-10-29 15:05:20,319 P24143 INFO [Metrics] AUC: 0.936315 - logloss: 0.286690
2022-10-29 15:05:20,320 P24143 INFO Save best model: monitor(max): 0.936315
2022-10-29 15:05:20,330 P24143 INFO --- 343/343 batches finished ---
2022-10-29 15:05:20,372 P24143 INFO Train loss: 0.380873
2022-10-29 15:05:20,372 P24143 INFO ************ Epoch=1 end ************
2022-10-29 15:05:31,115 P24143 INFO [Metrics] AUC: 0.946622 - logloss: 0.262927
2022-10-29 15:05:31,116 P24143 INFO Save best model: monitor(max): 0.946622
2022-10-29 15:05:31,127 P24143 INFO --- 343/343 batches finished ---
2022-10-29 15:05:31,173 P24143 INFO Train loss: 0.363778
2022-10-29 15:05:31,173 P24143 INFO ************ Epoch=2 end ************
2022-10-29 15:05:53,073 P24143 INFO [Metrics] AUC: 0.949533 - logloss: 0.257413
2022-10-29 15:05:53,074 P24143 INFO Save best model: monitor(max): 0.949533
2022-10-29 15:05:53,085 P24143 INFO --- 343/343 batches finished ---
2022-10-29 15:05:53,130 P24143 INFO Train loss: 0.365301
2022-10-29 15:05:53,130 P24143 INFO ************ Epoch=3 end ************
2022-10-29 15:06:14,820 P24143 INFO [Metrics] AUC: 0.951675 - logloss: 0.246543
2022-10-29 15:06:14,821 P24143 INFO Save best model: monitor(max): 0.951675
2022-10-29 15:06:14,829 P24143 INFO --- 343/343 batches finished ---
2022-10-29 15:06:14,871 P24143 INFO Train loss: 0.369061
2022-10-29 15:06:14,871 P24143 INFO ************ Epoch=4 end ************
2022-10-29 15:06:36,416 P24143 INFO [Metrics] AUC: 0.952749 - logloss: 0.245054
2022-10-29 15:06:36,417 P24143 INFO Save best model: monitor(max): 0.952749
2022-10-29 15:06:36,425 P24143 INFO --- 343/343 batches finished ---
2022-10-29 15:06:36,457 P24143 INFO Train loss: 0.373783
2022-10-29 15:06:36,457 P24143 INFO ************ Epoch=5 end ************
2022-10-29 15:06:58,478 P24143 INFO [Metrics] AUC: 0.953439 - logloss: 0.245455
2022-10-29 15:06:58,479 P24143 INFO Save best model: monitor(max): 0.953439
2022-10-29 15:06:58,489 P24143 INFO --- 343/343 batches finished ---
2022-10-29 15:06:58,541 P24143 INFO Train loss: 0.374946
2022-10-29 15:06:58,542 P24143 INFO ************ Epoch=6 end ************
2022-10-29 15:07:09,877 P24143 INFO [Metrics] AUC: 0.954142 - logloss: 0.239931
2022-10-29 15:07:09,877 P24143 INFO Save best model: monitor(max): 0.954142
2022-10-29 15:07:09,889 P24143 INFO --- 343/343 batches finished ---
2022-10-29 15:07:09,948 P24143 INFO Train loss: 0.375498
2022-10-29 15:07:09,948 P24143 INFO ************ Epoch=7 end ************
2022-10-29 15:07:28,602 P24143 INFO [Metrics] AUC: 0.954636 - logloss: 0.237659
2022-10-29 15:07:28,603 P24143 INFO Save best model: monitor(max): 0.954636
2022-10-29 15:07:28,614 P24143 INFO --- 343/343 batches finished ---
2022-10-29 15:07:28,654 P24143 INFO Train loss: 0.376406
2022-10-29 15:07:28,654 P24143 INFO ************ Epoch=8 end ************
2022-10-29 15:07:49,526 P24143 INFO [Metrics] AUC: 0.955091 - logloss: 0.242453
2022-10-29 15:07:49,527 P24143 INFO Save best model: monitor(max): 0.955091
2022-10-29 15:07:49,535 P24143 INFO --- 343/343 batches finished ---
2022-10-29 15:07:49,579 P24143 INFO Train loss: 0.375937
2022-10-29 15:07:49,579 P24143 INFO ************ Epoch=9 end ************
2022-10-29 15:08:11,017 P24143 INFO [Metrics] AUC: 0.955416 - logloss: 0.238261
2022-10-29 15:08:11,018 P24143 INFO Save best model: monitor(max): 0.955416
2022-10-29 15:08:11,029 P24143 INFO --- 343/343 batches finished ---
2022-10-29 15:08:11,077 P24143 INFO Train loss: 0.376665
2022-10-29 15:08:11,078 P24143 INFO ************ Epoch=10 end ************
2022-10-29 15:08:27,364 P24143 INFO [Metrics] AUC: 0.955511 - logloss: 0.236343
2022-10-29 15:08:27,365 P24143 INFO Save best model: monitor(max): 0.955511
2022-10-29 15:08:27,376 P24143 INFO --- 343/343 batches finished ---
2022-10-29 15:08:27,431 P24143 INFO Train loss: 0.375203
2022-10-29 15:08:27,431 P24143 INFO ************ Epoch=11 end ************
2022-10-29 15:08:44,493 P24143 INFO [Metrics] AUC: 0.956691 - logloss: 0.232197
2022-10-29 15:08:44,493 P24143 INFO Save best model: monitor(max): 0.956691
2022-10-29 15:08:44,502 P24143 INFO --- 343/343 batches finished ---
2022-10-29 15:08:44,541 P24143 INFO Train loss: 0.376660
2022-10-29 15:08:44,541 P24143 INFO ************ Epoch=12 end ************
2022-10-29 15:09:05,556 P24143 INFO [Metrics] AUC: 0.956410 - logloss: 0.233713
2022-10-29 15:09:05,557 P24143 INFO Monitor(max) STOP: 0.956410 !
2022-10-29 15:09:05,557 P24143 INFO Reduce learning rate on plateau: 0.000100
2022-10-29 15:09:05,557 P24143 INFO --- 343/343 batches finished ---
2022-10-29 15:09:05,606 P24143 INFO Train loss: 0.376230
2022-10-29 15:09:05,606 P24143 INFO ************ Epoch=13 end ************
2022-10-29 15:09:26,717 P24143 INFO [Metrics] AUC: 0.967366 - logloss: 0.207615
2022-10-29 15:09:26,718 P24143 INFO Save best model: monitor(max): 0.967366
2022-10-29 15:09:26,729 P24143 INFO --- 343/343 batches finished ---
2022-10-29 15:09:26,768 P24143 INFO Train loss: 0.273167
2022-10-29 15:09:26,769 P24143 INFO ************ Epoch=14 end ************
2022-10-29 15:09:47,745 P24143 INFO [Metrics] AUC: 0.968738 - logloss: 0.215786
2022-10-29 15:09:47,746 P24143 INFO Save best model: monitor(max): 0.968738
2022-10-29 15:09:47,754 P24143 INFO --- 343/343 batches finished ---
2022-10-29 15:09:47,805 P24143 INFO Train loss: 0.188569
2022-10-29 15:09:47,806 P24143 INFO ************ Epoch=15 end ************
2022-10-29 15:10:03,422 P24143 INFO [Metrics] AUC: 0.967290 - logloss: 0.242514
2022-10-29 15:10:03,423 P24143 INFO Monitor(max) STOP: 0.967290 !
2022-10-29 15:10:03,424 P24143 INFO Reduce learning rate on plateau: 0.000010
2022-10-29 15:10:03,424 P24143 INFO --- 343/343 batches finished ---
2022-10-29 15:10:03,481 P24143 INFO Train loss: 0.144130
2022-10-29 15:10:03,482 P24143 INFO ************ Epoch=16 end ************
2022-10-29 15:10:12,527 P24143 INFO [Metrics] AUC: 0.966740 - logloss: 0.254775
2022-10-29 15:10:12,528 P24143 INFO Monitor(max) STOP: 0.966740 !
2022-10-29 15:10:12,528 P24143 INFO Reduce learning rate on plateau: 0.000001
2022-10-29 15:10:12,528 P24143 INFO Early stopping at epoch=17
2022-10-29 15:10:12,528 P24143 INFO --- 343/343 batches finished ---
2022-10-29 15:10:12,576 P24143 INFO Train loss: 0.111467
2022-10-29 15:10:12,577 P24143 INFO Training finished.
2022-10-29 15:10:12,577 P24143 INFO Load best model: /home/benchmarks/Movielens/DCN_movielenslatest_x1/movielenslatest_x1_cd32d937/DCN_movielenslatest_x1_017_4810b636.model
2022-10-29 15:10:15,634 P24143 INFO ****** Validation evaluation ******
2022-10-29 15:10:16,927 P24143 INFO [Metrics] AUC: 0.968738 - logloss: 0.215786
2022-10-29 15:10:16,988 P24143 INFO ******** Test evaluation ********
2022-10-29 15:10:16,988 P24143 INFO Loading data...
2022-10-29 15:10:16,989 P24143 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-10-29 15:10:16,994 P24143 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-10-29 15:10:16,994 P24143 INFO Loading test data done.
2022-10-29 15:10:17,893 P24143 INFO [Metrics] AUC: 0.968719 - logloss: 0.215987
```
