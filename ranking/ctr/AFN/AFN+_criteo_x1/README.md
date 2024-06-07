## AFN+_criteo_x1

A hands-on guide to run the AFN model on the Criteo_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [AFN](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/AFN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFN+_criteo_x1_tuner_config_04](./AFN+_criteo_x1_tuner_config_04). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFN+_criteo_x1
    nohup python run_expid.py --config ./AFN+_criteo_x1_tuner_config_04 --expid AFN_criteo_x1_002_32798d82 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.814254 | 0.437659  |
| 2 | 0.814225 | 0.437707  |
| 3 | 0.814111 | 0.437815  |
| 4 | 0.814108 | 0.437753  |
| 5 | 0.814155 | 0.437715  |
| Avg | 0.814171 | 0.437730 |
| Std | &#177;0.00005938 | &#177;0.00005206 |


### Logs
```python
2022-01-24 13:10:25,499 P834 INFO {
    "afn_activations": "relu",
    "afn_dropout": "0.4",
    "afn_hidden_units": "[200]",
    "batch_norm": "True",
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_dropout": "0.1",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "ensemble_dnn": "True",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "logarithmic_neurons": "100",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AFN",
    "model_id": "AFN_criteo_x1_002_32798d82",
    "model_root": "./Criteo/AFN_criteo_x1/",
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
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-01-24 13:10:25,500 P834 INFO Set up feature encoder...
2022-01-24 13:10:25,500 P834 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-01-24 13:10:25,500 P834 INFO Loading data...
2022-01-24 13:10:25,502 P834 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-01-24 13:10:30,972 P834 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-01-24 13:10:32,181 P834 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-01-24 13:10:32,182 P834 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-01-24 13:10:32,182 P834 INFO Loading train data done.
2022-01-24 13:10:37,310 P834 INFO Total number of parameters: 42411303.
2022-01-24 13:10:37,311 P834 INFO Start training: 8058 batches/epoch
2022-01-24 13:10:37,311 P834 INFO ************ Epoch=1 start ************
2022-01-24 13:22:26,208 P834 INFO [Metrics] AUC: 0.804410 - logloss: 0.447085
2022-01-24 13:22:26,209 P834 INFO Save best model: monitor(max): 0.804410
2022-01-24 13:22:26,519 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 13:22:26,561 P834 INFO Train loss: 0.461620
2022-01-24 13:22:26,561 P834 INFO ************ Epoch=1 end ************
2022-01-24 13:34:08,415 P834 INFO [Metrics] AUC: 0.806814 - logloss: 0.444616
2022-01-24 13:34:08,416 P834 INFO Save best model: monitor(max): 0.806814
2022-01-24 13:34:08,643 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 13:34:08,692 P834 INFO Train loss: 0.456468
2022-01-24 13:34:08,692 P834 INFO ************ Epoch=2 end ************
2022-01-24 13:45:50,576 P834 INFO [Metrics] AUC: 0.807506 - logloss: 0.444031
2022-01-24 13:45:50,578 P834 INFO Save best model: monitor(max): 0.807506
2022-01-24 13:45:50,779 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 13:45:50,822 P834 INFO Train loss: 0.455156
2022-01-24 13:45:50,823 P834 INFO ************ Epoch=3 end ************
2022-01-24 13:57:40,961 P834 INFO [Metrics] AUC: 0.808265 - logloss: 0.443286
2022-01-24 13:57:40,963 P834 INFO Save best model: monitor(max): 0.808265
2022-01-24 13:57:41,199 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 13:57:41,248 P834 INFO Train loss: 0.454502
2022-01-24 13:57:41,248 P834 INFO ************ Epoch=4 end ************
2022-01-24 14:09:24,194 P834 INFO [Metrics] AUC: 0.808750 - logloss: 0.442830
2022-01-24 14:09:24,196 P834 INFO Save best model: monitor(max): 0.808750
2022-01-24 14:09:24,393 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 14:09:24,444 P834 INFO Train loss: 0.454094
2022-01-24 14:09:24,444 P834 INFO ************ Epoch=5 end ************
2022-01-24 14:21:06,364 P834 INFO [Metrics] AUC: 0.808998 - logloss: 0.442558
2022-01-24 14:21:06,365 P834 INFO Save best model: monitor(max): 0.808998
2022-01-24 14:21:06,572 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 14:21:06,615 P834 INFO Train loss: 0.453780
2022-01-24 14:21:06,615 P834 INFO ************ Epoch=6 end ************
2022-01-24 14:33:03,773 P834 INFO [Metrics] AUC: 0.809381 - logloss: 0.442336
2022-01-24 14:33:03,775 P834 INFO Save best model: monitor(max): 0.809381
2022-01-24 14:33:03,982 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 14:33:04,032 P834 INFO Train loss: 0.453530
2022-01-24 14:33:04,033 P834 INFO ************ Epoch=7 end ************
2022-01-24 14:44:58,819 P834 INFO [Metrics] AUC: 0.809465 - logloss: 0.442119
2022-01-24 14:44:58,820 P834 INFO Save best model: monitor(max): 0.809465
2022-01-24 14:44:59,026 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 14:44:59,064 P834 INFO Train loss: 0.453326
2022-01-24 14:44:59,065 P834 INFO ************ Epoch=8 end ************
2022-01-24 14:56:52,256 P834 INFO [Metrics] AUC: 0.809770 - logloss: 0.441904
2022-01-24 14:56:52,257 P834 INFO Save best model: monitor(max): 0.809770
2022-01-24 14:56:52,476 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 14:56:52,527 P834 INFO Train loss: 0.453166
2022-01-24 14:56:52,528 P834 INFO ************ Epoch=9 end ************
2022-01-24 15:08:47,732 P834 INFO [Metrics] AUC: 0.809788 - logloss: 0.441898
2022-01-24 15:08:47,733 P834 INFO Save best model: monitor(max): 0.809788
2022-01-24 15:08:47,948 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 15:08:47,998 P834 INFO Train loss: 0.452976
2022-01-24 15:08:47,998 P834 INFO ************ Epoch=10 end ************
2022-01-24 15:20:42,347 P834 INFO [Metrics] AUC: 0.809925 - logloss: 0.441788
2022-01-24 15:20:42,348 P834 INFO Save best model: monitor(max): 0.809925
2022-01-24 15:20:42,572 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 15:20:42,624 P834 INFO Train loss: 0.452874
2022-01-24 15:20:42,624 P834 INFO ************ Epoch=11 end ************
2022-01-24 15:32:38,014 P834 INFO [Metrics] AUC: 0.810107 - logloss: 0.441583
2022-01-24 15:32:38,016 P834 INFO Save best model: monitor(max): 0.810107
2022-01-24 15:32:38,229 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 15:32:38,278 P834 INFO Train loss: 0.452733
2022-01-24 15:32:38,279 P834 INFO ************ Epoch=12 end ************
2022-01-24 15:44:35,301 P834 INFO [Metrics] AUC: 0.810139 - logloss: 0.441490
2022-01-24 15:44:35,303 P834 INFO Save best model: monitor(max): 0.810139
2022-01-24 15:44:35,534 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 15:44:35,586 P834 INFO Train loss: 0.452630
2022-01-24 15:44:35,586 P834 INFO ************ Epoch=13 end ************
2022-01-24 15:56:31,960 P834 INFO [Metrics] AUC: 0.810247 - logloss: 0.441481
2022-01-24 15:56:31,962 P834 INFO Save best model: monitor(max): 0.810247
2022-01-24 15:56:32,145 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 15:56:32,193 P834 INFO Train loss: 0.452489
2022-01-24 15:56:32,193 P834 INFO ************ Epoch=14 end ************
2022-01-24 16:08:28,078 P834 INFO [Metrics] AUC: 0.810251 - logloss: 0.441376
2022-01-24 16:08:28,079 P834 INFO Save best model: monitor(max): 0.810251
2022-01-24 16:08:28,303 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 16:08:28,344 P834 INFO Train loss: 0.452416
2022-01-24 16:08:28,345 P834 INFO ************ Epoch=15 end ************
2022-01-24 16:20:16,912 P834 INFO [Metrics] AUC: 0.810344 - logloss: 0.441294
2022-01-24 16:20:16,914 P834 INFO Save best model: monitor(max): 0.810344
2022-01-24 16:20:17,127 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 16:20:17,172 P834 INFO Train loss: 0.452327
2022-01-24 16:20:17,172 P834 INFO ************ Epoch=16 end ************
2022-01-24 16:32:01,702 P834 INFO [Metrics] AUC: 0.810442 - logloss: 0.441196
2022-01-24 16:32:01,703 P834 INFO Save best model: monitor(max): 0.810442
2022-01-24 16:32:01,874 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 16:32:01,919 P834 INFO Train loss: 0.452237
2022-01-24 16:32:01,919 P834 INFO ************ Epoch=17 end ************
2022-01-24 16:43:45,620 P834 INFO [Metrics] AUC: 0.810375 - logloss: 0.441286
2022-01-24 16:43:45,622 P834 INFO Monitor(max) STOP: 0.810375 !
2022-01-24 16:43:45,622 P834 INFO Reduce learning rate on plateau: 0.000100
2022-01-24 16:43:45,622 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 16:43:45,663 P834 INFO Train loss: 0.452175
2022-01-24 16:43:45,663 P834 INFO ************ Epoch=18 end ************
2022-01-24 16:55:27,399 P834 INFO [Metrics] AUC: 0.813388 - logloss: 0.438497
2022-01-24 16:55:27,401 P834 INFO Save best model: monitor(max): 0.813388
2022-01-24 16:55:27,608 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 16:55:27,653 P834 INFO Train loss: 0.441259
2022-01-24 16:55:27,654 P834 INFO ************ Epoch=19 end ************
2022-01-24 17:07:07,390 P834 INFO [Metrics] AUC: 0.813815 - logloss: 0.438160
2022-01-24 17:07:07,391 P834 INFO Save best model: monitor(max): 0.813815
2022-01-24 17:07:07,567 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 17:07:07,619 P834 INFO Train loss: 0.436640
2022-01-24 17:07:07,619 P834 INFO ************ Epoch=20 end ************
2022-01-24 17:18:44,903 P834 INFO [Metrics] AUC: 0.813844 - logloss: 0.438177
2022-01-24 17:18:44,905 P834 INFO Save best model: monitor(max): 0.813844
2022-01-24 17:18:45,102 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 17:18:45,150 P834 INFO Train loss: 0.434655
2022-01-24 17:18:45,151 P834 INFO ************ Epoch=21 end ************
2022-01-24 17:30:26,887 P834 INFO [Metrics] AUC: 0.813769 - logloss: 0.438312
2022-01-24 17:30:26,889 P834 INFO Monitor(max) STOP: 0.813769 !
2022-01-24 17:30:26,889 P834 INFO Reduce learning rate on plateau: 0.000010
2022-01-24 17:30:26,889 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 17:30:26,932 P834 INFO Train loss: 0.433159
2022-01-24 17:30:26,933 P834 INFO ************ Epoch=22 end ************
2022-01-24 17:42:08,973 P834 INFO [Metrics] AUC: 0.813352 - logloss: 0.439013
2022-01-24 17:42:08,975 P834 INFO Monitor(max) STOP: 0.813352 !
2022-01-24 17:42:08,975 P834 INFO Reduce learning rate on plateau: 0.000001
2022-01-24 17:42:08,975 P834 INFO Early stopping at epoch=23
2022-01-24 17:42:08,975 P834 INFO --- 8058/8058 batches finished ---
2022-01-24 17:42:09,023 P834 INFO Train loss: 0.428682
2022-01-24 17:42:09,023 P834 INFO Training finished.
2022-01-24 17:42:09,023 P834 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/AFN_criteo_x1/criteo_x1_7b681156/AFN_criteo_x1_002_32798d82.model
2022-01-24 17:42:11,873 P834 INFO ****** Validation evaluation ******
2022-01-24 17:42:39,401 P834 INFO [Metrics] AUC: 0.813844 - logloss: 0.438177
2022-01-24 17:42:39,489 P834 INFO ******** Test evaluation ********
2022-01-24 17:42:39,489 P834 INFO Loading data...
2022-01-24 17:42:39,489 P834 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-01-24 17:42:40,297 P834 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-01-24 17:42:40,297 P834 INFO Loading test data done.
2022-01-24 17:42:56,713 P834 INFO [Metrics] AUC: 0.814254 - logloss: 0.437659

```
