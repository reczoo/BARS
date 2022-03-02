## DeepFM_criteo_x1

A hands-on guide to run the DeepFM model on the Criteo_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

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
Dataset ID: [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [DeepFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DeepFM_criteo_x1_tuner_config_01](./DeepFM_criteo_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DeepFM_criteo_x1
    nohup python run_expid.py --config ./DeepFM_criteo_x1_tuner_config_01 --expid DeepFM_criteo_x1_001_4b788fed --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.813666 | 0.438147  |
| 2 | 0.813822 | 0.438006  |
| 3 | 0.813899 | 0.438007  |
| 4 | 0.813869 | 0.438005  |
| 5 | 0.813806 | 0.438002  |
| | | | 
| Avg | 0.813812 | 0.438033 |
| Std | &#177;0.00008034 | &#177;0.00005682 |


### Logs
```python
2022-02-08 10:20:09,764 P798 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DeepFM",
    "model_id": "DeepFM_criteo_x1_001_4b788fed",
    "model_root": "./Criteo/DeepFM_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
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
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-02-08 10:20:09,765 P798 INFO Set up feature encoder...
2022-02-08 10:20:09,765 P798 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-02-08 10:20:09,765 P798 INFO Loading data...
2022-02-08 10:20:09,766 P798 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-02-08 10:20:18,466 P798 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-02-08 10:20:19,599 P798 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-02-08 10:20:19,599 P798 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-02-08 10:20:19,599 P798 INFO Loading train data done.
2022-02-08 10:20:23,609 P798 INFO Total number of parameters: 23429477.
2022-02-08 10:20:23,609 P798 INFO Start training: 8058 batches/epoch
2022-02-08 10:20:23,610 P798 INFO ************ Epoch=1 start ************
2022-02-08 10:42:22,460 P798 INFO [Metrics] AUC: 0.802869 - logloss: 0.448263
2022-02-08 10:42:22,461 P798 INFO Save best model: monitor(max): 0.802869
2022-02-08 10:42:22,557 P798 INFO --- 8058/8058 batches finished ---
2022-02-08 10:42:22,602 P798 INFO Train loss: 0.464138
2022-02-08 10:42:22,602 P798 INFO ************ Epoch=1 end ************
2022-02-08 11:04:20,777 P798 INFO [Metrics] AUC: 0.805084 - logloss: 0.446305
2022-02-08 11:04:20,778 P798 INFO Save best model: monitor(max): 0.805084
2022-02-08 11:04:20,893 P798 INFO --- 8058/8058 batches finished ---
2022-02-08 11:04:20,929 P798 INFO Train loss: 0.458914
2022-02-08 11:04:20,930 P798 INFO ************ Epoch=2 end ************
2022-02-08 11:26:18,561 P798 INFO [Metrics] AUC: 0.806479 - logloss: 0.444904
2022-02-08 11:26:18,562 P798 INFO Save best model: monitor(max): 0.806479
2022-02-08 11:26:18,709 P798 INFO --- 8058/8058 batches finished ---
2022-02-08 11:26:18,774 P798 INFO Train loss: 0.457416
2022-02-08 11:26:18,774 P798 INFO ************ Epoch=3 end ************
2022-02-08 11:48:16,373 P798 INFO [Metrics] AUC: 0.807006 - logloss: 0.444409
2022-02-08 11:48:16,375 P798 INFO Save best model: monitor(max): 0.807006
2022-02-08 11:48:16,497 P798 INFO --- 8058/8058 batches finished ---
2022-02-08 11:48:16,533 P798 INFO Train loss: 0.456598
2022-02-08 11:48:16,534 P798 INFO ************ Epoch=4 end ************
2022-02-08 12:10:19,584 P798 INFO [Metrics] AUC: 0.807590 - logloss: 0.443927
2022-02-08 12:10:19,586 P798 INFO Save best model: monitor(max): 0.807590
2022-02-08 12:10:19,721 P798 INFO --- 8058/8058 batches finished ---
2022-02-08 12:10:19,757 P798 INFO Train loss: 0.456098
2022-02-08 12:10:19,757 P798 INFO ************ Epoch=5 end ************
2022-02-08 12:32:21,382 P798 INFO [Metrics] AUC: 0.808003 - logloss: 0.443573
2022-02-08 12:32:21,384 P798 INFO Save best model: monitor(max): 0.808003
2022-02-08 12:32:21,530 P798 INFO --- 8058/8058 batches finished ---
2022-02-08 12:32:21,567 P798 INFO Train loss: 0.455716
2022-02-08 12:32:21,567 P798 INFO ************ Epoch=6 end ************
2022-02-08 12:54:22,225 P798 INFO [Metrics] AUC: 0.808162 - logloss: 0.443370
2022-02-08 12:54:22,226 P798 INFO Save best model: monitor(max): 0.808162
2022-02-08 12:54:22,344 P798 INFO --- 8058/8058 batches finished ---
2022-02-08 12:54:22,381 P798 INFO Train loss: 0.455413
2022-02-08 12:54:22,381 P798 INFO ************ Epoch=7 end ************
2022-02-08 13:16:21,246 P798 INFO [Metrics] AUC: 0.808426 - logloss: 0.443266
2022-02-08 13:16:21,248 P798 INFO Save best model: monitor(max): 0.808426
2022-02-08 13:16:21,353 P798 INFO --- 8058/8058 batches finished ---
2022-02-08 13:16:21,390 P798 INFO Train loss: 0.455165
2022-02-08 13:16:21,390 P798 INFO ************ Epoch=8 end ************
2022-02-08 13:38:19,261 P798 INFO [Metrics] AUC: 0.808594 - logloss: 0.442986
2022-02-08 13:38:19,262 P798 INFO Save best model: monitor(max): 0.808594
2022-02-08 13:38:19,388 P798 INFO --- 8058/8058 batches finished ---
2022-02-08 13:38:19,426 P798 INFO Train loss: 0.455004
2022-02-08 13:38:19,427 P798 INFO ************ Epoch=9 end ************
2022-02-08 14:00:16,396 P798 INFO [Metrics] AUC: 0.808572 - logloss: 0.442988
2022-02-08 14:00:16,397 P798 INFO Monitor(max) STOP: 0.808572 !
2022-02-08 14:00:16,397 P798 INFO Reduce learning rate on plateau: 0.000100
2022-02-08 14:00:16,397 P798 INFO --- 8058/8058 batches finished ---
2022-02-08 14:00:16,434 P798 INFO Train loss: 0.454814
2022-02-08 14:00:16,434 P798 INFO ************ Epoch=10 end ************
2022-02-08 14:22:07,823 P798 INFO [Metrics] AUC: 0.812669 - logloss: 0.439206
2022-02-08 14:22:07,825 P798 INFO Save best model: monitor(max): 0.812669
2022-02-08 14:22:07,938 P798 INFO --- 8058/8058 batches finished ---
2022-02-08 14:22:07,976 P798 INFO Train loss: 0.442939
2022-02-08 14:22:07,976 P798 INFO ************ Epoch=11 end ************
2022-02-08 14:43:53,806 P798 INFO [Metrics] AUC: 0.813221 - logloss: 0.438715
2022-02-08 14:43:53,808 P798 INFO Save best model: monitor(max): 0.813221
2022-02-08 14:43:53,943 P798 INFO --- 8058/8058 batches finished ---
2022-02-08 14:43:53,981 P798 INFO Train loss: 0.438472
2022-02-08 14:43:53,981 P798 INFO ************ Epoch=12 end ************
2022-02-08 15:05:38,099 P798 INFO [Metrics] AUC: 0.813348 - logloss: 0.438602
2022-02-08 15:05:38,100 P798 INFO Save best model: monitor(max): 0.813348
2022-02-08 15:05:38,212 P798 INFO --- 8058/8058 batches finished ---
2022-02-08 15:05:38,257 P798 INFO Train loss: 0.436654
2022-02-08 15:05:38,257 P798 INFO ************ Epoch=13 end ************
2022-02-08 15:27:21,314 P798 INFO [Metrics] AUC: 0.813228 - logloss: 0.438742
2022-02-08 15:27:21,315 P798 INFO Monitor(max) STOP: 0.813228 !
2022-02-08 15:27:21,316 P798 INFO Reduce learning rate on plateau: 0.000010
2022-02-08 15:27:21,316 P798 INFO --- 8058/8058 batches finished ---
2022-02-08 15:27:21,353 P798 INFO Train loss: 0.435279
2022-02-08 15:27:21,353 P798 INFO ************ Epoch=14 end ************
2022-02-08 15:49:04,461 P798 INFO [Metrics] AUC: 0.812800 - logloss: 0.439494
2022-02-08 15:49:04,462 P798 INFO Monitor(max) STOP: 0.812800 !
2022-02-08 15:49:04,463 P798 INFO Reduce learning rate on plateau: 0.000001
2022-02-08 15:49:04,463 P798 INFO Early stopping at epoch=15
2022-02-08 15:49:04,463 P798 INFO --- 8058/8058 batches finished ---
2022-02-08 15:49:04,500 P798 INFO Train loss: 0.430705
2022-02-08 15:49:04,501 P798 INFO Training finished.
2022-02-08 15:49:04,501 P798 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/DeepFM_criteo_x1/criteo_x1_7b681156/DeepFM_criteo_x1_001_4b788fed.model
2022-02-08 15:49:04,605 P798 INFO ****** Validation evaluation ******
2022-02-08 15:49:33,776 P798 INFO [Metrics] AUC: 0.813348 - logloss: 0.438602
2022-02-08 15:49:33,856 P798 INFO ******** Test evaluation ********
2022-02-08 15:49:33,857 P798 INFO Loading data...
2022-02-08 15:49:33,857 P798 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-02-08 15:49:34,637 P798 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-02-08 15:49:34,637 P798 INFO Loading test data done.
2022-02-08 15:49:51,762 P798 INFO [Metrics] AUC: 0.813666 - logloss: 0.438147

```
