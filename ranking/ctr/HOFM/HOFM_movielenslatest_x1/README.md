## HOFM_movielenslatest_x1

A hands-on guide to run the HOFM model on the Movielenslatest_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [HOFM](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/HOFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HOFM_movielenslatest_x1_tuner_config_03](./HOFM_movielenslatest_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HOFM_movielenslatest_x1
    nohup python run_expid.py --config ./HOFM_movielenslatest_x1_tuner_config_03 --expid HOFM_movielenslatest_x1_004_c58c682e --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.945534 | 0.270525  |


### Logs
```python
2022-01-26 14:34:23,175 P19242 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "embedding_dim": "10",
    "epochs": "200",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "HOFM",
    "model_id": "HOFM_movielenslatest_x1_004_c58c682e",
    "model_root": "./Movielens/HOFM_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "order": "3",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "2e-06",
    "reuse_embedding": "True",
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
2022-01-26 14:34:23,176 P19242 INFO Set up feature encoder...
2022-01-26 14:34:23,176 P19242 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-01-26 14:34:23,176 P19242 INFO Loading data...
2022-01-26 14:34:23,178 P19242 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-01-26 14:34:23,207 P19242 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-01-26 14:34:23,216 P19242 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-01-26 14:34:23,216 P19242 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-01-26 14:34:23,217 P19242 INFO Loading train data done.
2022-01-26 14:34:26,661 P19242 INFO Total number of parameters: 992630.
2022-01-26 14:34:26,662 P19242 INFO Start training: 343 batches/epoch
2022-01-26 14:34:26,662 P19242 INFO ************ Epoch=1 start ************
2022-01-26 14:34:36,579 P19242 INFO [Metrics] AUC: 0.898497 - logloss: 0.489864
2022-01-26 14:34:36,579 P19242 INFO Save best model: monitor(max): 0.898497
2022-01-26 14:34:36,585 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:34:36,623 P19242 INFO Train loss: 0.601725
2022-01-26 14:34:36,623 P19242 INFO ************ Epoch=1 end ************
2022-01-26 14:34:46,838 P19242 INFO [Metrics] AUC: 0.920264 - logloss: 0.359485
2022-01-26 14:34:46,839 P19242 INFO Save best model: monitor(max): 0.920264
2022-01-26 14:34:46,845 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:34:46,886 P19242 INFO Train loss: 0.412975
2022-01-26 14:34:46,887 P19242 INFO ************ Epoch=2 end ************
2022-01-26 14:34:57,069 P19242 INFO [Metrics] AUC: 0.928555 - logloss: 0.316492
2022-01-26 14:34:57,069 P19242 INFO Save best model: monitor(max): 0.928555
2022-01-26 14:34:57,075 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:34:57,111 P19242 INFO Train loss: 0.332614
2022-01-26 14:34:57,111 P19242 INFO ************ Epoch=3 end ************
2022-01-26 14:35:07,014 P19242 INFO [Metrics] AUC: 0.932792 - logloss: 0.299567
2022-01-26 14:35:07,014 P19242 INFO Save best model: monitor(max): 0.932792
2022-01-26 14:35:07,020 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:35:07,059 P19242 INFO Train loss: 0.302773
2022-01-26 14:35:07,059 P19242 INFO ************ Epoch=4 end ************
2022-01-26 14:35:16,988 P19242 INFO [Metrics] AUC: 0.935436 - logloss: 0.290967
2022-01-26 14:35:16,989 P19242 INFO Save best model: monitor(max): 0.935436
2022-01-26 14:35:16,995 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:35:17,035 P19242 INFO Train loss: 0.287635
2022-01-26 14:35:17,035 P19242 INFO ************ Epoch=5 end ************
2022-01-26 14:35:26,777 P19242 INFO [Metrics] AUC: 0.937323 - logloss: 0.285665
2022-01-26 14:35:26,778 P19242 INFO Save best model: monitor(max): 0.937323
2022-01-26 14:35:26,784 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:35:26,820 P19242 INFO Train loss: 0.277828
2022-01-26 14:35:26,820 P19242 INFO ************ Epoch=6 end ************
2022-01-26 14:35:36,674 P19242 INFO [Metrics] AUC: 0.938751 - logloss: 0.282123
2022-01-26 14:35:36,674 P19242 INFO Save best model: monitor(max): 0.938751
2022-01-26 14:35:36,680 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:35:36,719 P19242 INFO Train loss: 0.270484
2022-01-26 14:35:36,719 P19242 INFO ************ Epoch=7 end ************
2022-01-26 14:35:46,723 P19242 INFO [Metrics] AUC: 0.939909 - logloss: 0.279424
2022-01-26 14:35:46,723 P19242 INFO Save best model: monitor(max): 0.939909
2022-01-26 14:35:46,729 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:35:46,764 P19242 INFO Train loss: 0.264476
2022-01-26 14:35:46,764 P19242 INFO ************ Epoch=8 end ************
2022-01-26 14:35:56,837 P19242 INFO [Metrics] AUC: 0.940894 - logloss: 0.277316
2022-01-26 14:35:56,837 P19242 INFO Save best model: monitor(max): 0.940894
2022-01-26 14:35:56,843 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:35:56,901 P19242 INFO Train loss: 0.259280
2022-01-26 14:35:56,901 P19242 INFO ************ Epoch=9 end ************
2022-01-26 14:36:07,036 P19242 INFO [Metrics] AUC: 0.941735 - logloss: 0.275624
2022-01-26 14:36:07,037 P19242 INFO Save best model: monitor(max): 0.941735
2022-01-26 14:36:07,044 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:36:07,090 P19242 INFO Train loss: 0.254613
2022-01-26 14:36:07,090 P19242 INFO ************ Epoch=10 end ************
2022-01-26 14:36:16,891 P19242 INFO [Metrics] AUC: 0.942429 - logloss: 0.274256
2022-01-26 14:36:16,891 P19242 INFO Save best model: monitor(max): 0.942429
2022-01-26 14:36:16,898 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:36:16,937 P19242 INFO Train loss: 0.250321
2022-01-26 14:36:16,938 P19242 INFO ************ Epoch=11 end ************
2022-01-26 14:36:26,723 P19242 INFO [Metrics] AUC: 0.943019 - logloss: 0.273139
2022-01-26 14:36:26,724 P19242 INFO Save best model: monitor(max): 0.943019
2022-01-26 14:36:26,730 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:36:26,769 P19242 INFO Train loss: 0.246324
2022-01-26 14:36:26,769 P19242 INFO ************ Epoch=12 end ************
2022-01-26 14:36:36,578 P19242 INFO [Metrics] AUC: 0.943548 - logloss: 0.272173
2022-01-26 14:36:36,579 P19242 INFO Save best model: monitor(max): 0.943548
2022-01-26 14:36:36,585 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:36:36,628 P19242 INFO Train loss: 0.242566
2022-01-26 14:36:36,628 P19242 INFO ************ Epoch=13 end ************
2022-01-26 14:36:46,402 P19242 INFO [Metrics] AUC: 0.943960 - logloss: 0.271482
2022-01-26 14:36:46,403 P19242 INFO Save best model: monitor(max): 0.943960
2022-01-26 14:36:46,409 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:36:46,448 P19242 INFO Train loss: 0.239004
2022-01-26 14:36:46,448 P19242 INFO ************ Epoch=14 end ************
2022-01-26 14:36:56,006 P19242 INFO [Metrics] AUC: 0.944340 - logloss: 0.270906
2022-01-26 14:36:56,007 P19242 INFO Save best model: monitor(max): 0.944340
2022-01-26 14:36:56,013 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:36:56,050 P19242 INFO Train loss: 0.235610
2022-01-26 14:36:56,050 P19242 INFO ************ Epoch=15 end ************
2022-01-26 14:37:05,646 P19242 INFO [Metrics] AUC: 0.944638 - logloss: 0.270474
2022-01-26 14:37:05,646 P19242 INFO Save best model: monitor(max): 0.944638
2022-01-26 14:37:05,653 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:37:05,689 P19242 INFO Train loss: 0.232353
2022-01-26 14:37:05,689 P19242 INFO ************ Epoch=16 end ************
2022-01-26 14:37:15,807 P19242 INFO [Metrics] AUC: 0.944863 - logloss: 0.270239
2022-01-26 14:37:15,808 P19242 INFO Save best model: monitor(max): 0.944863
2022-01-26 14:37:15,814 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:37:15,861 P19242 INFO Train loss: 0.229221
2022-01-26 14:37:15,861 P19242 INFO ************ Epoch=17 end ************
2022-01-26 14:37:25,584 P19242 INFO [Metrics] AUC: 0.945063 - logloss: 0.270064
2022-01-26 14:37:25,585 P19242 INFO Save best model: monitor(max): 0.945063
2022-01-26 14:37:25,592 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:37:25,626 P19242 INFO Train loss: 0.226215
2022-01-26 14:37:25,626 P19242 INFO ************ Epoch=18 end ************
2022-01-26 14:37:35,233 P19242 INFO [Metrics] AUC: 0.945204 - logloss: 0.270066
2022-01-26 14:37:35,233 P19242 INFO Save best model: monitor(max): 0.945204
2022-01-26 14:37:35,239 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:37:35,283 P19242 INFO Train loss: 0.223299
2022-01-26 14:37:35,283 P19242 INFO ************ Epoch=19 end ************
2022-01-26 14:37:45,040 P19242 INFO [Metrics] AUC: 0.945306 - logloss: 0.270174
2022-01-26 14:37:45,040 P19242 INFO Save best model: monitor(max): 0.945306
2022-01-26 14:37:45,046 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:37:45,090 P19242 INFO Train loss: 0.220490
2022-01-26 14:37:45,091 P19242 INFO ************ Epoch=20 end ************
2022-01-26 14:37:54,893 P19242 INFO [Metrics] AUC: 0.945375 - logloss: 0.270428
2022-01-26 14:37:54,893 P19242 INFO Save best model: monitor(max): 0.945375
2022-01-26 14:37:54,901 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:37:54,940 P19242 INFO Train loss: 0.217767
2022-01-26 14:37:54,940 P19242 INFO ************ Epoch=21 end ************
2022-01-26 14:38:04,557 P19242 INFO [Metrics] AUC: 0.945403 - logloss: 0.270747
2022-01-26 14:38:04,557 P19242 INFO Save best model: monitor(max): 0.945403
2022-01-26 14:38:04,563 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:38:04,603 P19242 INFO Train loss: 0.215147
2022-01-26 14:38:04,603 P19242 INFO ************ Epoch=22 end ************
2022-01-26 14:38:14,011 P19242 INFO [Metrics] AUC: 0.945406 - logloss: 0.271189
2022-01-26 14:38:14,012 P19242 INFO Save best model: monitor(max): 0.945406
2022-01-26 14:38:14,018 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:38:14,053 P19242 INFO Train loss: 0.212622
2022-01-26 14:38:14,053 P19242 INFO ************ Epoch=23 end ************
2022-01-26 14:38:21,454 P19242 INFO [Metrics] AUC: 0.945372 - logloss: 0.271703
2022-01-26 14:38:21,455 P19242 INFO Monitor(max) STOP: 0.945372 !
2022-01-26 14:38:21,455 P19242 INFO Reduce learning rate on plateau: 0.000100
2022-01-26 14:38:21,455 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:38:21,501 P19242 INFO Train loss: 0.210182
2022-01-26 14:38:21,501 P19242 INFO ************ Epoch=24 end ************
2022-01-26 14:38:28,604 P19242 INFO [Metrics] AUC: 0.945379 - logloss: 0.271707
2022-01-26 14:38:28,604 P19242 INFO Monitor(max) STOP: 0.945379 !
2022-01-26 14:38:28,604 P19242 INFO Reduce learning rate on plateau: 0.000010
2022-01-26 14:38:28,605 P19242 INFO Early stopping at epoch=25
2022-01-26 14:38:28,605 P19242 INFO --- 343/343 batches finished ---
2022-01-26 14:38:28,656 P19242 INFO Train loss: 0.204007
2022-01-26 14:38:28,656 P19242 INFO Training finished.
2022-01-26 14:38:28,656 P19242 INFO Load best model: /home/XXX/benchmarks/Movielens/HOFM_movielenslatest_x1/movielenslatest_x1_cd32d937/HOFM_movielenslatest_x1_004_c58c682e.model
2022-01-26 14:38:31,693 P19242 INFO ****** Validation evaluation ******
2022-01-26 14:38:33,025 P19242 INFO [Metrics] AUC: 0.945406 - logloss: 0.271189
2022-01-26 14:38:33,071 P19242 INFO ******** Test evaluation ********
2022-01-26 14:38:33,071 P19242 INFO Loading data...
2022-01-26 14:38:33,072 P19242 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-01-26 14:38:33,076 P19242 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-01-26 14:38:33,076 P19242 INFO Loading test data done.
2022-01-26 14:38:33,835 P19242 INFO [Metrics] AUC: 0.945534 - logloss: 0.270525

```
