## AOANet_criteo_x1

A hands-on guide to run the AOANet model on the Criteo_x1 dataset.

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
  fuxictr: 1.2.1

  ```

### Dataset
Dataset ID: [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo#Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [AOANet](https://github.com/xue-pai/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/AOANet.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AOANet_criteo_x1_tuner_config_03](./AOANet_criteo_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AOANet_criteo_x1
    nohup python run_expid.py --config ./AOANet_criteo_x1_tuner_config_03 --expid AOANet_criteo_x1_005_faa15d7f --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.814127 | 0.437825  |


### Logs
```python
2022-06-01 14:01:59,484 P56037 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "dnn_hidden_activations": "ReLU",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "4",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AOANet",
    "model_id": "AOANet_criteo_x1_005_faa15d7f",
    "model_root": "./Criteo/AOANet_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_interaction_layers": "1",
    "num_subspaces": "8",
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
2022-06-01 14:01:59,484 P56037 INFO Set up feature encoder...
2022-06-01 14:01:59,484 P56037 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-06-01 14:01:59,485 P56037 INFO Loading data...
2022-06-01 14:01:59,485 P56037 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-06-01 14:02:04,219 P56037 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-06-01 14:02:05,389 P56037 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-06-01 14:02:05,389 P56037 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-06-01 14:02:05,389 P56037 INFO Loading train data done.
2022-06-01 14:02:09,801 P56037 INFO Total number of parameters: 21356289.
2022-06-01 14:02:09,802 P56037 INFO Start training: 8058 batches/epoch
2022-06-01 14:02:09,802 P56037 INFO ************ Epoch=1 start ************
2022-06-01 14:22:25,613 P56037 INFO [Metrics] AUC: 0.804139 - logloss: 0.447060
2022-06-01 14:22:25,615 P56037 INFO Save best model: monitor(max): 0.804139
2022-06-01 14:22:25,859 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 14:22:25,898 P56037 INFO Train loss: 0.461745
2022-06-01 14:22:25,899 P56037 INFO ************ Epoch=1 end ************
2022-06-01 14:42:40,739 P56037 INFO [Metrics] AUC: 0.806324 - logloss: 0.445000
2022-06-01 14:42:40,740 P56037 INFO Save best model: monitor(max): 0.806324
2022-06-01 14:42:40,852 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 14:42:40,897 P56037 INFO Train loss: 0.455928
2022-06-01 14:42:40,897 P56037 INFO ************ Epoch=2 end ************
2022-06-01 15:02:54,868 P56037 INFO [Metrics] AUC: 0.807531 - logloss: 0.443970
2022-06-01 15:02:54,869 P56037 INFO Save best model: monitor(max): 0.807531
2022-06-01 15:02:54,967 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 15:02:55,010 P56037 INFO Train loss: 0.454417
2022-06-01 15:02:55,010 P56037 INFO ************ Epoch=3 end ************
2022-06-01 15:23:10,668 P56037 INFO [Metrics] AUC: 0.808247 - logloss: 0.443340
2022-06-01 15:23:10,669 P56037 INFO Save best model: monitor(max): 0.808247
2022-06-01 15:23:10,780 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 15:23:10,817 P56037 INFO Train loss: 0.453657
2022-06-01 15:23:10,817 P56037 INFO ************ Epoch=4 end ************
2022-06-01 15:43:29,411 P56037 INFO [Metrics] AUC: 0.808612 - logloss: 0.442935
2022-06-01 15:43:29,413 P56037 INFO Save best model: monitor(max): 0.808612
2022-06-01 15:43:29,504 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 15:43:29,540 P56037 INFO Train loss: 0.453190
2022-06-01 15:43:29,540 P56037 INFO ************ Epoch=5 end ************
2022-06-01 16:03:45,367 P56037 INFO [Metrics] AUC: 0.808954 - logloss: 0.442598
2022-06-01 16:03:45,368 P56037 INFO Save best model: monitor(max): 0.808954
2022-06-01 16:03:45,468 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 16:03:45,507 P56037 INFO Train loss: 0.452838
2022-06-01 16:03:45,507 P56037 INFO ************ Epoch=6 end ************
2022-06-01 16:24:03,460 P56037 INFO [Metrics] AUC: 0.809303 - logloss: 0.442277
2022-06-01 16:24:03,461 P56037 INFO Save best model: monitor(max): 0.809303
2022-06-01 16:24:03,559 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 16:24:03,606 P56037 INFO Train loss: 0.452568
2022-06-01 16:24:03,606 P56037 INFO ************ Epoch=7 end ************
2022-06-01 16:44:21,524 P56037 INFO [Metrics] AUC: 0.809538 - logloss: 0.442141
2022-06-01 16:44:21,525 P56037 INFO Save best model: monitor(max): 0.809538
2022-06-01 16:44:21,625 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 16:44:21,666 P56037 INFO Train loss: 0.452320
2022-06-01 16:44:21,666 P56037 INFO ************ Epoch=8 end ************
2022-06-01 17:04:42,446 P56037 INFO [Metrics] AUC: 0.809621 - logloss: 0.442050
2022-06-01 17:04:42,447 P56037 INFO Save best model: monitor(max): 0.809621
2022-06-01 17:04:42,547 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 17:04:42,589 P56037 INFO Train loss: 0.452143
2022-06-01 17:04:42,589 P56037 INFO ************ Epoch=9 end ************
2022-06-01 17:25:04,632 P56037 INFO [Metrics] AUC: 0.809706 - logloss: 0.441934
2022-06-01 17:25:04,633 P56037 INFO Save best model: monitor(max): 0.809706
2022-06-01 17:25:04,729 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 17:25:04,774 P56037 INFO Train loss: 0.451989
2022-06-01 17:25:04,774 P56037 INFO ************ Epoch=10 end ************
2022-06-01 17:45:22,004 P56037 INFO [Metrics] AUC: 0.809965 - logloss: 0.441689
2022-06-01 17:45:22,006 P56037 INFO Save best model: monitor(max): 0.809965
2022-06-01 17:45:22,099 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 17:45:22,144 P56037 INFO Train loss: 0.451828
2022-06-01 17:45:22,144 P56037 INFO ************ Epoch=11 end ************
2022-06-01 18:05:44,276 P56037 INFO [Metrics] AUC: 0.810023 - logloss: 0.441629
2022-06-01 18:05:44,277 P56037 INFO Save best model: monitor(max): 0.810023
2022-06-01 18:05:44,367 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 18:05:44,409 P56037 INFO Train loss: 0.451707
2022-06-01 18:05:44,410 P56037 INFO ************ Epoch=12 end ************
2022-06-01 18:25:57,765 P56037 INFO [Metrics] AUC: 0.809893 - logloss: 0.441754
2022-06-01 18:25:57,767 P56037 INFO Monitor(max) STOP: 0.809893 !
2022-06-01 18:25:57,767 P56037 INFO Reduce learning rate on plateau: 0.000100
2022-06-01 18:25:57,767 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 18:25:57,813 P56037 INFO Train loss: 0.451602
2022-06-01 18:25:57,814 P56037 INFO ************ Epoch=13 end ************
2022-06-01 18:46:18,280 P56037 INFO [Metrics] AUC: 0.813067 - logloss: 0.438869
2022-06-01 18:46:18,281 P56037 INFO Save best model: monitor(max): 0.813067
2022-06-01 18:46:18,375 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 18:46:18,423 P56037 INFO Train loss: 0.441052
2022-06-01 18:46:18,423 P56037 INFO ************ Epoch=14 end ************
2022-06-01 19:06:37,259 P56037 INFO [Metrics] AUC: 0.813573 - logloss: 0.438453
2022-06-01 19:06:37,260 P56037 INFO Save best model: monitor(max): 0.813573
2022-06-01 19:06:37,359 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 19:06:37,406 P56037 INFO Train loss: 0.437117
2022-06-01 19:06:37,406 P56037 INFO ************ Epoch=15 end ************
2022-06-01 19:26:55,152 P56037 INFO [Metrics] AUC: 0.813653 - logloss: 0.438349
2022-06-01 19:26:55,154 P56037 INFO Save best model: monitor(max): 0.813653
2022-06-01 19:26:55,266 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 19:26:55,316 P56037 INFO Train loss: 0.435369
2022-06-01 19:26:55,316 P56037 INFO ************ Epoch=16 end ************
2022-06-01 19:47:12,665 P56037 INFO [Metrics] AUC: 0.813704 - logloss: 0.438395
2022-06-01 19:47:12,666 P56037 INFO Save best model: monitor(max): 0.813704
2022-06-01 19:47:12,763 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 19:47:12,803 P56037 INFO Train loss: 0.434067
2022-06-01 19:47:12,804 P56037 INFO ************ Epoch=17 end ************
2022-06-01 20:07:27,152 P56037 INFO [Metrics] AUC: 0.813549 - logloss: 0.438590
2022-06-01 20:07:27,153 P56037 INFO Monitor(max) STOP: 0.813549 !
2022-06-01 20:07:27,153 P56037 INFO Reduce learning rate on plateau: 0.000010
2022-06-01 20:07:27,153 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 20:07:27,196 P56037 INFO Train loss: 0.432977
2022-06-01 20:07:27,196 P56037 INFO ************ Epoch=18 end ************
2022-06-01 20:27:40,295 P56037 INFO [Metrics] AUC: 0.813055 - logloss: 0.439342
2022-06-01 20:27:40,296 P56037 INFO Monitor(max) STOP: 0.813055 !
2022-06-01 20:27:40,296 P56037 INFO Reduce learning rate on plateau: 0.000001
2022-06-01 20:27:40,296 P56037 INFO Early stopping at epoch=19
2022-06-01 20:27:40,296 P56037 INFO --- 8058/8058 batches finished ---
2022-06-01 20:27:40,335 P56037 INFO Train loss: 0.428634
2022-06-01 20:27:40,335 P56037 INFO Training finished.
2022-06-01 20:27:40,335 P56037 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/AOANet_criteo_x1/criteo_x1_7b681156/AOANet_criteo_x1_005_faa15d7f.model
2022-06-01 20:27:45,030 P56037 INFO ****** Validation evaluation ******
2022-06-01 20:28:58,807 P56037 INFO [Metrics] AUC: 0.813704 - logloss: 0.438395
2022-06-01 20:28:58,887 P56037 INFO ******** Test evaluation ********
2022-06-01 20:28:58,887 P56037 INFO Loading data...
2022-06-01 20:28:58,888 P56037 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-06-01 20:28:59,677 P56037 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-06-01 20:28:59,677 P56037 INFO Loading test data done.
2022-06-01 20:29:40,641 P56037 INFO [Metrics] AUC: 0.814127 - logloss: 0.437825

```
