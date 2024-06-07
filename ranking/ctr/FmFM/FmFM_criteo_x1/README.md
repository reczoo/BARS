## FmFM_criteo_x1

A hands-on guide to run the FmFM model on the Criteo_x1 dataset.

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
Dataset ID: [Criteo_x1](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [FmFM](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FmFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FmFM_criteo_x1_tuner_config_01](./FmFM_criteo_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FmFM_criteo_x1
    nohup python run_expid.py --config ./FmFM_criteo_x1_tuner_config_01 --expid FmFM_criteo_x1_001_9ba4938b --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.805556 | 0.446269  |


### Logs
```python
2022-01-20 23:48:53,264 P53419 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "embedding_dim": "10",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "field_interaction_type": "matrixed",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FmFM",
    "model_id": "FmFM_criteo_x1_001_9ba4938b",
    "model_root": "./Criteo/FmFM_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-06",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-20 23:48:53,265 P53419 INFO Set up feature encoder...
2022-01-20 23:48:53,265 P53419 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-01-20 23:48:53,265 P53419 INFO Loading data...
2022-01-20 23:48:53,267 P53419 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-01-20 23:48:57,847 P53419 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-01-20 23:48:58,967 P53419 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-01-20 23:48:58,967 P53419 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-01-20 23:48:58,968 P53419 INFO Loading train data done.
2022-01-20 23:49:03,873 P53419 INFO Total number of parameters: 23023577.
2022-01-20 23:49:03,874 P53419 INFO Start training: 8058 batches/epoch
2022-01-20 23:49:03,874 P53419 INFO ************ Epoch=1 start ************
2022-01-21 01:24:38,039 P53419 INFO [Metrics] AUC: 0.801020 - logloss: 0.450297
2022-01-21 01:24:38,040 P53419 INFO Save best model: monitor(max): 0.801020
2022-01-21 01:24:38,118 P53419 INFO --- 8058/8058 batches finished ---
2022-01-21 01:24:38,155 P53419 INFO Train loss: 0.459855
2022-01-21 01:24:38,155 P53419 INFO ************ Epoch=1 end ************
2022-01-21 02:59:15,537 P53419 INFO [Metrics] AUC: 0.803175 - logloss: 0.448222
2022-01-21 02:59:15,538 P53419 INFO Save best model: monitor(max): 0.803175
2022-01-21 02:59:15,705 P53419 INFO --- 8058/8058 batches finished ---
2022-01-21 02:59:15,742 P53419 INFO Train loss: 0.453507
2022-01-21 02:59:15,743 P53419 INFO ************ Epoch=2 end ************
2022-01-21 04:16:58,382 P53419 INFO [Metrics] AUC: 0.803717 - logloss: 0.447803
2022-01-21 04:16:58,383 P53419 INFO Save best model: monitor(max): 0.803717
2022-01-21 04:16:58,527 P53419 INFO --- 8058/8058 batches finished ---
2022-01-21 04:16:58,565 P53419 INFO Train loss: 0.452075
2022-01-21 04:16:58,565 P53419 INFO ************ Epoch=3 end ************
2022-01-21 04:42:20,234 P53419 INFO [Metrics] AUC: 0.804106 - logloss: 0.447411
2022-01-21 04:42:20,235 P53419 INFO Save best model: monitor(max): 0.804106
2022-01-21 04:42:20,378 P53419 INFO --- 8058/8058 batches finished ---
2022-01-21 04:42:20,416 P53419 INFO Train loss: 0.451413
2022-01-21 04:42:20,416 P53419 INFO ************ Epoch=4 end ************
2022-01-21 05:07:40,789 P53419 INFO [Metrics] AUC: 0.804139 - logloss: 0.447340
2022-01-21 05:07:40,790 P53419 INFO Save best model: monitor(max): 0.804139
2022-01-21 05:07:40,945 P53419 INFO --- 8058/8058 batches finished ---
2022-01-21 05:07:40,984 P53419 INFO Train loss: 0.451063
2022-01-21 05:07:40,984 P53419 INFO ************ Epoch=5 end ************
2022-01-21 05:32:59,392 P53419 INFO [Metrics] AUC: 0.804243 - logloss: 0.447393
2022-01-21 05:32:59,394 P53419 INFO Save best model: monitor(max): 0.804243
2022-01-21 05:32:59,548 P53419 INFO --- 8058/8058 batches finished ---
2022-01-21 05:32:59,587 P53419 INFO Train loss: 0.450819
2022-01-21 05:32:59,587 P53419 INFO ************ Epoch=6 end ************
2022-01-21 05:58:15,746 P53419 INFO [Metrics] AUC: 0.804027 - logloss: 0.447486
2022-01-21 05:58:15,747 P53419 INFO Monitor(max) STOP: 0.804027 !
2022-01-21 05:58:15,747 P53419 INFO Reduce learning rate on plateau: 0.000100
2022-01-21 05:58:15,748 P53419 INFO --- 8058/8058 batches finished ---
2022-01-21 05:58:15,786 P53419 INFO Train loss: 0.450671
2022-01-21 05:58:15,787 P53419 INFO ************ Epoch=7 end ************
2022-01-21 06:23:35,275 P53419 INFO [Metrics] AUC: 0.805341 - logloss: 0.446566
2022-01-21 06:23:35,276 P53419 INFO Save best model: monitor(max): 0.805341
2022-01-21 06:23:35,428 P53419 INFO --- 8058/8058 batches finished ---
2022-01-21 06:23:35,469 P53419 INFO Train loss: 0.439070
2022-01-21 06:23:35,469 P53419 INFO ************ Epoch=8 end ************
2022-01-21 06:48:49,521 P53419 INFO [Metrics] AUC: 0.804869 - logloss: 0.447152
2022-01-21 06:48:49,522 P53419 INFO Monitor(max) STOP: 0.804869 !
2022-01-21 06:48:49,522 P53419 INFO Reduce learning rate on plateau: 0.000010
2022-01-21 06:48:49,522 P53419 INFO --- 8058/8058 batches finished ---
2022-01-21 06:48:49,561 P53419 INFO Train loss: 0.436120
2022-01-21 06:48:49,561 P53419 INFO ************ Epoch=9 end ************
2022-01-21 07:14:15,882 P53419 INFO [Metrics] AUC: 0.804760 - logloss: 0.447286
2022-01-21 07:14:15,883 P53419 INFO Monitor(max) STOP: 0.804760 !
2022-01-21 07:14:15,883 P53419 INFO Reduce learning rate on plateau: 0.000001
2022-01-21 07:14:15,883 P53419 INFO Early stopping at epoch=10
2022-01-21 07:14:15,883 P53419 INFO --- 8058/8058 batches finished ---
2022-01-21 07:14:15,921 P53419 INFO Train loss: 0.432668
2022-01-21 07:14:15,921 P53419 INFO Training finished.
2022-01-21 07:14:15,921 P53419 INFO Load best model: /home/XXX/FuxiCTR_github/benchmarks/Criteo/FmFM_criteo_x1/criteo_x1_7b681156/FmFM_criteo_x1_001_9ba4938b.model
2022-01-21 07:14:16,041 P53419 INFO ****** Validation evaluation ******
2022-01-21 07:15:53,741 P53419 INFO [Metrics] AUC: 0.805341 - logloss: 0.446566
2022-01-21 07:15:53,803 P53419 INFO ******** Test evaluation ********
2022-01-21 07:15:53,803 P53419 INFO Loading data...
2022-01-21 07:15:53,804 P53419 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-01-21 07:15:54,439 P53419 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-01-21 07:15:54,439 P53419 INFO Loading test data done.
2022-01-21 07:16:48,772 P53419 INFO [Metrics] AUC: 0.805556 - logloss: 0.446269

```
