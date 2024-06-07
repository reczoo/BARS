## SAM_criteo_x1

A hands-on guide to run the SAM model on the Criteo_x1 dataset.

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
  fuxictr: 1.2.1

  ```

### Dataset
Dataset ID: [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo#Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.1](https://github.com/reczoo/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [SAM](https://github.com/reczoo/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/SAM.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [SAM_criteo_x1_tuner_config_03](./SAM_criteo_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd SAM_criteo_x1
    nohup python run_expid.py --config ./SAM_criteo_x1_tuner_config_03 --expid SAM_criteo_x1_012_55e25f89 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.813098 | 0.438855  |


### Logs
```python
2022-05-31 10:38:41,600 P69477 INFO {
    "aggregation": "weighted_pooling",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "3",
    "interaction_type": "SAM3A",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "SAM",
    "model_id": "SAM_criteo_x1_012_55e25f89",
    "model_root": "./Criteo/SAM_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_interaction_layers": "5",
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
    "use_hdf5": "True",
    "use_residual": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-05-31 10:38:41,601 P69477 INFO Set up feature encoder...
2022-05-31 10:38:41,601 P69477 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-05-31 10:38:41,601 P69477 INFO Loading data...
2022-05-31 10:38:41,603 P69477 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-05-31 10:38:46,548 P69477 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-05-31 10:38:47,776 P69477 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-05-31 10:38:47,776 P69477 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-05-31 10:38:47,776 P69477 INFO Loading train data done.
2022-05-31 10:38:53,726 P69477 INFO Total number of parameters: 20940260.
2022-05-31 10:38:53,726 P69477 INFO Start training: 8058 batches/epoch
2022-05-31 10:38:53,726 P69477 INFO ************ Epoch=1 start ************
2022-05-31 11:00:13,564 P69477 INFO [Metrics] AUC: 0.801662 - logloss: 0.449289
2022-05-31 11:00:13,565 P69477 INFO Save best model: monitor(max): 0.801662
2022-05-31 11:00:13,852 P69477 INFO --- 8058/8058 batches finished ---
2022-05-31 11:00:13,900 P69477 INFO Train loss: 0.485520
2022-05-31 11:00:13,900 P69477 INFO ************ Epoch=1 end ************
2022-05-31 11:21:30,684 P69477 INFO [Metrics] AUC: 0.805122 - logloss: 0.446276
2022-05-31 11:21:30,686 P69477 INFO Save best model: monitor(max): 0.805122
2022-05-31 11:21:30,787 P69477 INFO --- 8058/8058 batches finished ---
2022-05-31 11:21:30,824 P69477 INFO Train loss: 0.457815
2022-05-31 11:21:30,825 P69477 INFO ************ Epoch=2 end ************
2022-05-31 11:42:45,886 P69477 INFO [Metrics] AUC: 0.806512 - logloss: 0.444994
2022-05-31 11:42:45,887 P69477 INFO Save best model: monitor(max): 0.806512
2022-05-31 11:42:45,987 P69477 INFO --- 8058/8058 batches finished ---
2022-05-31 11:42:46,025 P69477 INFO Train loss: 0.455675
2022-05-31 11:42:46,026 P69477 INFO ************ Epoch=3 end ************
2022-05-31 12:04:03,454 P69477 INFO [Metrics] AUC: 0.807335 - logloss: 0.444205
2022-05-31 12:04:03,455 P69477 INFO Save best model: monitor(max): 0.807335
2022-05-31 12:04:03,558 P69477 INFO --- 8058/8058 batches finished ---
2022-05-31 12:04:03,598 P69477 INFO Train loss: 0.454515
2022-05-31 12:04:03,598 P69477 INFO ************ Epoch=4 end ************
2022-05-31 12:25:20,165 P69477 INFO [Metrics] AUC: 0.807930 - logloss: 0.443640
2022-05-31 12:25:20,166 P69477 INFO Save best model: monitor(max): 0.807930
2022-05-31 12:25:20,256 P69477 INFO --- 8058/8058 batches finished ---
2022-05-31 12:25:20,297 P69477 INFO Train loss: 0.453823
2022-05-31 12:25:20,297 P69477 INFO ************ Epoch=5 end ************
2022-05-31 12:46:33,776 P69477 INFO [Metrics] AUC: 0.808102 - logloss: 0.443469
2022-05-31 12:46:33,778 P69477 INFO Save best model: monitor(max): 0.808102
2022-05-31 12:46:33,869 P69477 INFO --- 8058/8058 batches finished ---
2022-05-31 12:46:33,911 P69477 INFO Train loss: 0.453343
2022-05-31 12:46:33,911 P69477 INFO ************ Epoch=6 end ************
2022-05-31 13:07:43,227 P69477 INFO [Metrics] AUC: 0.808464 - logloss: 0.443094
2022-05-31 13:07:43,228 P69477 INFO Save best model: monitor(max): 0.808464
2022-05-31 13:07:43,325 P69477 INFO --- 8058/8058 batches finished ---
2022-05-31 13:07:43,363 P69477 INFO Train loss: 0.452970
2022-05-31 13:07:43,364 P69477 INFO ************ Epoch=7 end ************
2022-05-31 13:28:50,201 P69477 INFO [Metrics] AUC: 0.808635 - logloss: 0.442933
2022-05-31 13:28:50,203 P69477 INFO Save best model: monitor(max): 0.808635
2022-05-31 13:28:50,307 P69477 INFO --- 8058/8058 batches finished ---
2022-05-31 13:28:50,349 P69477 INFO Train loss: 0.452673
2022-05-31 13:28:50,349 P69477 INFO ************ Epoch=8 end ************
2022-05-31 13:49:57,226 P69477 INFO [Metrics] AUC: 0.808637 - logloss: 0.442973
2022-05-31 13:49:57,227 P69477 INFO Save best model: monitor(max): 0.808637
2022-05-31 13:49:57,342 P69477 INFO --- 8058/8058 batches finished ---
2022-05-31 13:49:57,379 P69477 INFO Train loss: 0.452444
2022-05-31 13:49:57,379 P69477 INFO ************ Epoch=9 end ************
2022-05-31 14:11:03,305 P69477 INFO [Metrics] AUC: 0.808924 - logloss: 0.442759
2022-05-31 14:11:03,306 P69477 INFO Save best model: monitor(max): 0.808924
2022-05-31 14:11:03,396 P69477 INFO --- 8058/8058 batches finished ---
2022-05-31 14:11:03,435 P69477 INFO Train loss: 0.452228
2022-05-31 14:11:03,435 P69477 INFO ************ Epoch=10 end ************
2022-05-31 14:32:07,419 P69477 INFO [Metrics] AUC: 0.808976 - logloss: 0.442699
2022-05-31 14:32:07,421 P69477 INFO Save best model: monitor(max): 0.808976
2022-05-31 14:32:07,510 P69477 INFO --- 8058/8058 batches finished ---
2022-05-31 14:32:07,548 P69477 INFO Train loss: 0.452068
2022-05-31 14:32:07,548 P69477 INFO ************ Epoch=11 end ************
2022-05-31 14:53:11,037 P69477 INFO [Metrics] AUC: 0.809002 - logloss: 0.442607
2022-05-31 14:53:11,039 P69477 INFO Save best model: monitor(max): 0.809002
2022-05-31 14:53:11,136 P69477 INFO --- 8058/8058 batches finished ---
2022-05-31 14:53:11,175 P69477 INFO Train loss: 0.451914
2022-05-31 14:53:11,175 P69477 INFO ************ Epoch=12 end ************
2022-05-31 15:14:14,645 P69477 INFO [Metrics] AUC: 0.808964 - logloss: 0.442691
2022-05-31 15:14:14,647 P69477 INFO Monitor(max) STOP: 0.808964 !
2022-05-31 15:14:14,647 P69477 INFO Reduce learning rate on plateau: 0.000100
2022-05-31 15:14:14,647 P69477 INFO --- 8058/8058 batches finished ---
2022-05-31 15:14:14,689 P69477 INFO Train loss: 0.451778
2022-05-31 15:14:14,690 P69477 INFO ************ Epoch=13 end ************
2022-05-31 15:35:17,365 P69477 INFO [Metrics] AUC: 0.812336 - logloss: 0.439588
2022-05-31 15:35:17,366 P69477 INFO Save best model: monitor(max): 0.812336
2022-05-31 15:35:17,456 P69477 INFO --- 8058/8058 batches finished ---
2022-05-31 15:35:17,494 P69477 INFO Train loss: 0.441169
2022-05-31 15:35:17,494 P69477 INFO ************ Epoch=14 end ************
2022-05-31 15:56:20,198 P69477 INFO [Metrics] AUC: 0.812733 - logloss: 0.439322
2022-05-31 15:56:20,199 P69477 INFO Save best model: monitor(max): 0.812733
2022-05-31 15:56:20,287 P69477 INFO --- 8058/8058 batches finished ---
2022-05-31 15:56:20,327 P69477 INFO Train loss: 0.437275
2022-05-31 15:56:20,327 P69477 INFO ************ Epoch=15 end ************
2022-05-31 16:17:22,768 P69477 INFO [Metrics] AUC: 0.812786 - logloss: 0.439343
2022-05-31 16:17:22,769 P69477 INFO Save best model: monitor(max): 0.812786
2022-05-31 16:17:22,856 P69477 INFO --- 8058/8058 batches finished ---
2022-05-31 16:17:22,897 P69477 INFO Train loss: 0.435485
2022-05-31 16:17:22,897 P69477 INFO ************ Epoch=16 end ************
2022-05-31 16:38:24,290 P69477 INFO [Metrics] AUC: 0.812564 - logloss: 0.439558
2022-05-31 16:38:24,291 P69477 INFO Monitor(max) STOP: 0.812564 !
2022-05-31 16:38:24,291 P69477 INFO Reduce learning rate on plateau: 0.000010
2022-05-31 16:38:24,291 P69477 INFO --- 8058/8058 batches finished ---
2022-05-31 16:38:24,329 P69477 INFO Train loss: 0.434076
2022-05-31 16:38:24,329 P69477 INFO ************ Epoch=17 end ************
2022-05-31 16:59:25,511 P69477 INFO [Metrics] AUC: 0.811779 - logloss: 0.440918
2022-05-31 16:59:25,512 P69477 INFO Monitor(max) STOP: 0.811779 !
2022-05-31 16:59:25,512 P69477 INFO Reduce learning rate on plateau: 0.000001
2022-05-31 16:59:25,512 P69477 INFO Early stopping at epoch=18
2022-05-31 16:59:25,512 P69477 INFO --- 8058/8058 batches finished ---
2022-05-31 16:59:25,555 P69477 INFO Train loss: 0.429269
2022-05-31 16:59:25,555 P69477 INFO Training finished.
2022-05-31 16:59:25,555 P69477 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/SAM_criteo_x1/criteo_x1_7b681156/SAM_criteo_x1_012_55e25f89.model
2022-05-31 16:59:29,433 P69477 INFO ****** Validation evaluation ******
2022-05-31 17:00:07,391 P69477 INFO [Metrics] AUC: 0.812786 - logloss: 0.439343
2022-05-31 17:00:07,467 P69477 INFO ******** Test evaluation ********
2022-05-31 17:00:07,468 P69477 INFO Loading data...
2022-05-31 17:00:07,468 P69477 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-05-31 17:00:08,300 P69477 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-05-31 17:00:08,301 P69477 INFO Loading test data done.
2022-05-31 17:00:30,908 P69477 INFO [Metrics] AUC: 0.813098 - logloss: 0.438855

```
