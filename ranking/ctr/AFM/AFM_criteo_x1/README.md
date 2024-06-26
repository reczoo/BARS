## AFM_criteo_x1

A hands-on guide to run the AFM model on the Criteo_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) Gold 6278C CPU @ 2.60GHz
  GPU: Tesla V100 32G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 10.2
  python: 3.6.4
  pytorch: 1.0.0
  pandas: 0.22.0
  numpy: 1.19.2
  scipy: 1.5.4
  sklearn: 0.22.1
  pyyaml: 5.4.1
  h5py: 2.8.0
  tqdm: 4.60.0
  fuxictr: 1.1.0
  ```

### Dataset
Dataset ID: [Criteo_x1](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [AFM](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/AFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFM_criteo_x1_tuner_config_02](./AFM_criteo_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFM_criteo_x1
    nohup python run_expid.py --config ./AFM_criteo_x1_tuner_config_02 --expid AFM_criteo_x1_004_954c0ecc --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.804436 | 0.446977  |


### Logs
```python
2022-01-27 09:42:57,551 P81292 INFO {
    "attention_dim": "32",
    "attention_dropout": "[0, 0]",
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-06",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "3",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AFM",
    "model_id": "AFM_criteo_x1_004_954c0ecc",
    "model_root": "./Criteo/AFM_criteo_x1/",
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
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_attention": "True",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-01-27 09:42:57,552 P81292 INFO Set up feature encoder...
2022-01-27 09:42:57,552 P81292 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-01-27 09:42:57,552 P81292 INFO Loading data...
2022-01-27 09:42:57,554 P81292 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-01-27 09:43:02,302 P81292 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-01-27 09:43:03,503 P81292 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-01-27 09:43:03,503 P81292 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-01-27 09:43:03,503 P81292 INFO Loading train data done.
2022-01-27 09:43:08,200 P81292 INFO Total number of parameters: 22949871.
2022-01-27 09:43:08,201 P81292 INFO Start training: 8058 batches/epoch
2022-01-27 09:43:08,201 P81292 INFO ************ Epoch=1 start ************
2022-01-27 09:54:24,259 P81292 INFO [Metrics] AUC: 0.792392 - logloss: 0.457434
2022-01-27 09:54:24,260 P81292 INFO Save best model: monitor(max): 0.792392
2022-01-27 09:54:24,536 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 09:54:24,580 P81292 INFO Train loss: 0.467648
2022-01-27 09:54:24,581 P81292 INFO ************ Epoch=1 end ************
2022-01-27 10:05:40,760 P81292 INFO [Metrics] AUC: 0.794542 - logloss: 0.455529
2022-01-27 10:05:40,761 P81292 INFO Save best model: monitor(max): 0.794542
2022-01-27 10:05:40,863 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 10:05:40,908 P81292 INFO Train loss: 0.458616
2022-01-27 10:05:40,908 P81292 INFO ************ Epoch=2 end ************
2022-01-27 10:16:58,594 P81292 INFO [Metrics] AUC: 0.796130 - logloss: 0.454182
2022-01-27 10:16:58,595 P81292 INFO Save best model: monitor(max): 0.796130
2022-01-27 10:16:58,699 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 10:16:58,743 P81292 INFO Train loss: 0.457069
2022-01-27 10:16:58,743 P81292 INFO ************ Epoch=3 end ************
2022-01-27 10:28:07,154 P81292 INFO [Metrics] AUC: 0.797042 - logloss: 0.453433
2022-01-27 10:28:07,155 P81292 INFO Save best model: monitor(max): 0.797042
2022-01-27 10:28:07,255 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 10:28:07,301 P81292 INFO Train loss: 0.456030
2022-01-27 10:28:07,301 P81292 INFO ************ Epoch=4 end ************
2022-01-27 10:39:14,448 P81292 INFO [Metrics] AUC: 0.797710 - logloss: 0.452792
2022-01-27 10:39:14,450 P81292 INFO Save best model: monitor(max): 0.797710
2022-01-27 10:39:14,547 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 10:39:14,596 P81292 INFO Train loss: 0.455302
2022-01-27 10:39:14,596 P81292 INFO ************ Epoch=5 end ************
2022-01-27 10:50:20,807 P81292 INFO [Metrics] AUC: 0.798328 - logloss: 0.452289
2022-01-27 10:50:20,809 P81292 INFO Save best model: monitor(max): 0.798328
2022-01-27 10:50:20,917 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 10:50:20,966 P81292 INFO Train loss: 0.454742
2022-01-27 10:50:20,966 P81292 INFO ************ Epoch=6 end ************
2022-01-27 11:01:29,055 P81292 INFO [Metrics] AUC: 0.798690 - logloss: 0.452054
2022-01-27 11:01:29,056 P81292 INFO Save best model: monitor(max): 0.798690
2022-01-27 11:01:29,173 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 11:01:29,214 P81292 INFO Train loss: 0.454265
2022-01-27 11:01:29,214 P81292 INFO ************ Epoch=7 end ************
2022-01-27 11:12:35,335 P81292 INFO [Metrics] AUC: 0.799245 - logloss: 0.451536
2022-01-27 11:12:35,336 P81292 INFO Save best model: monitor(max): 0.799245
2022-01-27 11:12:35,442 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 11:12:35,491 P81292 INFO Train loss: 0.453850
2022-01-27 11:12:35,491 P81292 INFO ************ Epoch=8 end ************
2022-01-27 11:23:44,476 P81292 INFO [Metrics] AUC: 0.799517 - logloss: 0.451304
2022-01-27 11:23:44,478 P81292 INFO Save best model: monitor(max): 0.799517
2022-01-27 11:23:44,594 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 11:23:44,645 P81292 INFO Train loss: 0.453516
2022-01-27 11:23:44,645 P81292 INFO ************ Epoch=9 end ************
2022-01-27 11:34:52,959 P81292 INFO [Metrics] AUC: 0.799859 - logloss: 0.451006
2022-01-27 11:34:52,960 P81292 INFO Save best model: monitor(max): 0.799859
2022-01-27 11:34:53,068 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 11:34:53,110 P81292 INFO Train loss: 0.453267
2022-01-27 11:34:53,111 P81292 INFO ************ Epoch=10 end ************
2022-01-27 11:45:59,963 P81292 INFO [Metrics] AUC: 0.800102 - logloss: 0.450715
2022-01-27 11:45:59,964 P81292 INFO Save best model: monitor(max): 0.800102
2022-01-27 11:46:00,062 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 11:46:00,110 P81292 INFO Train loss: 0.453081
2022-01-27 11:46:00,110 P81292 INFO ************ Epoch=11 end ************
2022-01-27 11:57:06,983 P81292 INFO [Metrics] AUC: 0.800308 - logloss: 0.450595
2022-01-27 11:57:06,984 P81292 INFO Save best model: monitor(max): 0.800308
2022-01-27 11:57:07,091 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 11:57:07,141 P81292 INFO Train loss: 0.452883
2022-01-27 11:57:07,141 P81292 INFO ************ Epoch=12 end ************
2022-01-27 12:08:14,388 P81292 INFO [Metrics] AUC: 0.800463 - logloss: 0.450443
2022-01-27 12:08:14,389 P81292 INFO Save best model: monitor(max): 0.800463
2022-01-27 12:08:14,495 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 12:08:14,546 P81292 INFO Train loss: 0.452738
2022-01-27 12:08:14,546 P81292 INFO ************ Epoch=13 end ************
2022-01-27 12:19:22,739 P81292 INFO [Metrics] AUC: 0.800602 - logloss: 0.450319
2022-01-27 12:19:22,741 P81292 INFO Save best model: monitor(max): 0.800602
2022-01-27 12:19:22,848 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 12:19:22,891 P81292 INFO Train loss: 0.452623
2022-01-27 12:19:22,892 P81292 INFO ************ Epoch=14 end ************
2022-01-27 12:30:29,999 P81292 INFO [Metrics] AUC: 0.800738 - logloss: 0.450289
2022-01-27 12:30:30,001 P81292 INFO Save best model: monitor(max): 0.800738
2022-01-27 12:30:30,099 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 12:30:30,145 P81292 INFO Train loss: 0.452500
2022-01-27 12:30:30,146 P81292 INFO ************ Epoch=15 end ************
2022-01-27 12:41:36,011 P81292 INFO [Metrics] AUC: 0.800962 - logloss: 0.449996
2022-01-27 12:41:36,012 P81292 INFO Save best model: monitor(max): 0.800962
2022-01-27 12:41:36,115 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 12:41:36,163 P81292 INFO Train loss: 0.452402
2022-01-27 12:41:36,163 P81292 INFO ************ Epoch=16 end ************
2022-01-27 12:52:43,827 P81292 INFO [Metrics] AUC: 0.801140 - logloss: 0.449882
2022-01-27 12:52:43,828 P81292 INFO Save best model: monitor(max): 0.801140
2022-01-27 12:52:43,929 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 12:52:43,971 P81292 INFO Train loss: 0.452327
2022-01-27 12:52:43,971 P81292 INFO ************ Epoch=17 end ************
2022-01-27 13:03:50,557 P81292 INFO [Metrics] AUC: 0.801249 - logloss: 0.449810
2022-01-27 13:03:50,558 P81292 INFO Save best model: monitor(max): 0.801249
2022-01-27 13:03:50,665 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 13:03:50,714 P81292 INFO Train loss: 0.452248
2022-01-27 13:03:50,714 P81292 INFO ************ Epoch=18 end ************
2022-01-27 13:14:55,352 P81292 INFO [Metrics] AUC: 0.801330 - logloss: 0.449733
2022-01-27 13:14:55,353 P81292 INFO Save best model: monitor(max): 0.801330
2022-01-27 13:14:55,462 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 13:14:55,508 P81292 INFO Train loss: 0.452123
2022-01-27 13:14:55,508 P81292 INFO ************ Epoch=19 end ************
2022-01-27 13:26:03,167 P81292 INFO [Metrics] AUC: 0.801559 - logloss: 0.449549
2022-01-27 13:26:03,168 P81292 INFO Save best model: monitor(max): 0.801559
2022-01-27 13:26:03,271 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 13:26:03,322 P81292 INFO Train loss: 0.451985
2022-01-27 13:26:03,322 P81292 INFO ************ Epoch=20 end ************
2022-01-27 13:37:09,708 P81292 INFO [Metrics] AUC: 0.801713 - logloss: 0.449438
2022-01-27 13:37:09,709 P81292 INFO Save best model: monitor(max): 0.801713
2022-01-27 13:37:09,807 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 13:37:09,857 P81292 INFO Train loss: 0.451851
2022-01-27 13:37:09,857 P81292 INFO ************ Epoch=21 end ************
2022-01-27 13:48:14,924 P81292 INFO [Metrics] AUC: 0.801993 - logloss: 0.449183
2022-01-27 13:48:14,925 P81292 INFO Save best model: monitor(max): 0.801993
2022-01-27 13:48:15,024 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 13:48:15,068 P81292 INFO Train loss: 0.451784
2022-01-27 13:48:15,068 P81292 INFO ************ Epoch=22 end ************
2022-01-27 13:59:22,336 P81292 INFO [Metrics] AUC: 0.802019 - logloss: 0.449106
2022-01-27 13:59:22,338 P81292 INFO Save best model: monitor(max): 0.802019
2022-01-27 13:59:22,440 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 13:59:22,487 P81292 INFO Train loss: 0.451731
2022-01-27 13:59:22,488 P81292 INFO ************ Epoch=23 end ************
2022-01-27 14:10:29,035 P81292 INFO [Metrics] AUC: 0.802032 - logloss: 0.449103
2022-01-27 14:10:29,037 P81292 INFO Save best model: monitor(max): 0.802032
2022-01-27 14:10:29,135 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 14:10:29,179 P81292 INFO Train loss: 0.451692
2022-01-27 14:10:29,180 P81292 INFO ************ Epoch=24 end ************
2022-01-27 14:21:36,494 P81292 INFO [Metrics] AUC: 0.802123 - logloss: 0.449080
2022-01-27 14:21:36,495 P81292 INFO Save best model: monitor(max): 0.802123
2022-01-27 14:21:36,607 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 14:21:36,649 P81292 INFO Train loss: 0.451659
2022-01-27 14:21:36,649 P81292 INFO ************ Epoch=25 end ************
2022-01-27 14:32:41,082 P81292 INFO [Metrics] AUC: 0.802120 - logloss: 0.449022
2022-01-27 14:32:41,083 P81292 INFO Monitor(max) STOP: 0.802120 !
2022-01-27 14:32:41,083 P81292 INFO Reduce learning rate on plateau: 0.000100
2022-01-27 14:32:41,083 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 14:32:41,125 P81292 INFO Train loss: 0.451597
2022-01-27 14:32:41,125 P81292 INFO ************ Epoch=26 end ************
2022-01-27 14:43:46,830 P81292 INFO [Metrics] AUC: 0.803880 - logloss: 0.447525
2022-01-27 14:43:46,831 P81292 INFO Save best model: monitor(max): 0.803880
2022-01-27 14:43:46,937 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 14:43:46,986 P81292 INFO Train loss: 0.445793
2022-01-27 14:43:46,986 P81292 INFO ************ Epoch=27 end ************
2022-01-27 14:54:52,196 P81292 INFO [Metrics] AUC: 0.804066 - logloss: 0.447390
2022-01-27 14:54:52,197 P81292 INFO Save best model: monitor(max): 0.804066
2022-01-27 14:54:52,304 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 14:54:52,346 P81292 INFO Train loss: 0.443912
2022-01-27 14:54:52,346 P81292 INFO ************ Epoch=28 end ************
2022-01-27 15:05:56,643 P81292 INFO [Metrics] AUC: 0.804069 - logloss: 0.447414
2022-01-27 15:05:56,644 P81292 INFO Save best model: monitor(max): 0.804069
2022-01-27 15:05:56,742 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 15:05:56,794 P81292 INFO Train loss: 0.442954
2022-01-27 15:05:56,794 P81292 INFO ************ Epoch=29 end ************
2022-01-27 15:16:59,050 P81292 INFO [Metrics] AUC: 0.804019 - logloss: 0.447461
2022-01-27 15:16:59,051 P81292 INFO Monitor(max) STOP: 0.804019 !
2022-01-27 15:16:59,051 P81292 INFO Reduce learning rate on plateau: 0.000010
2022-01-27 15:16:59,051 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 15:16:59,093 P81292 INFO Train loss: 0.442253
2022-01-27 15:16:59,094 P81292 INFO ************ Epoch=30 end ************
2022-01-27 15:28:03,819 P81292 INFO [Metrics] AUC: 0.803923 - logloss: 0.447627
2022-01-27 15:28:03,820 P81292 INFO Monitor(max) STOP: 0.803923 !
2022-01-27 15:28:03,820 P81292 INFO Reduce learning rate on plateau: 0.000001
2022-01-27 15:28:03,820 P81292 INFO Early stopping at epoch=31
2022-01-27 15:28:03,820 P81292 INFO --- 8058/8058 batches finished ---
2022-01-27 15:28:03,862 P81292 INFO Train loss: 0.440558
2022-01-27 15:28:03,862 P81292 INFO Training finished.
2022-01-27 15:28:03,862 P81292 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/AFM_criteo_x1/criteo_x1_7b681156/AFM_criteo_x1_004_954c0ecc.model
2022-01-27 15:28:08,876 P81292 INFO ****** Validation evaluation ******
2022-01-27 15:28:44,897 P81292 INFO [Metrics] AUC: 0.804069 - logloss: 0.447414
2022-01-27 15:28:44,983 P81292 INFO ******** Test evaluation ********
2022-01-27 15:28:44,984 P81292 INFO Loading data...
2022-01-27 15:28:44,984 P81292 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-01-27 15:28:45,763 P81292 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-01-27 15:28:45,763 P81292 INFO Loading test data done.
2022-01-27 15:29:04,865 P81292 INFO [Metrics] AUC: 0.804436 - logloss: 0.446977

```
