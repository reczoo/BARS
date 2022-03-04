## DESTINE_criteo_x1

A hands-on guide to run the DESTINE model on the Criteo_x1 dataset.

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
  fuxictr: 1.1.1

  ```

### Dataset
Dataset ID: [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.1](https://github.com/xue-pai/FuxiCTR/tree/v1.1.1) for this experiment. See the model code: [DESTINE](https://github.com/xue-pai/FuxiCTR/blob/v1.1.1/fuxictr/pytorch/models/DESTINE.py).

Running steps:

1. Download [FuxiCTR-v1.1.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DESTINE_criteo_x1_tuner_config_03](./DESTINE_criteo_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DESTINE_criteo_x1
    nohup python run_expid.py --config ./DESTINE_criteo_x1_tuner_config_03 --expid DESTINE_criteo_x1_001_767ae9b5 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.813902 | 0.437963  |


### Logs
```python
2022-02-19 22:27:10,753 P93627 INFO {
    "att_dropout": "0.2",
    "attention_dim": "128",
    "attention_layers": "4",
    "batch_norm": "True",
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DESTINE",
    "model_id": "DESTINE_criteo_x1_001_767ae9b5",
    "model_root": "./Criteo/DESTINE_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_heads": "1",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "relu_before_att": "False",
    "residual_mode": "each_layer",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_hdf5": "True",
    "use_scale": "True",
    "use_wide": "False",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-02-19 22:27:10,753 P93627 INFO Set up feature encoder...
2022-02-19 22:27:10,754 P93627 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-02-19 22:27:10,754 P93627 INFO Loading data...
2022-02-19 22:27:10,755 P93627 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-02-19 22:27:15,357 P93627 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-02-19 22:27:16,442 P93627 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-02-19 22:27:16,443 P93627 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-02-19 22:27:16,443 P93627 INFO Loading train data done.
2022-02-19 22:27:22,866 P93627 INFO Total number of parameters: 21552328.
2022-02-19 22:27:22,866 P93627 INFO Start training: 8058 batches/epoch
2022-02-19 22:27:22,866 P93627 INFO ************ Epoch=1 start ************
2022-02-19 23:11:54,508 P93627 INFO [Metrics] AUC: 0.803344 - logloss: 0.447766
2022-02-19 23:11:54,510 P93627 INFO Save best model: monitor(max): 0.803344
2022-02-19 23:11:54,595 P93627 INFO --- 8058/8058 batches finished ---
2022-02-19 23:11:54,635 P93627 INFO Train loss: 0.462952
2022-02-19 23:11:54,635 P93627 INFO ************ Epoch=1 end ************
2022-02-19 23:56:00,543 P93627 INFO [Metrics] AUC: 0.805963 - logloss: 0.445436
2022-02-19 23:56:00,544 P93627 INFO Save best model: monitor(max): 0.805963
2022-02-19 23:56:00,636 P93627 INFO --- 8058/8058 batches finished ---
2022-02-19 23:56:00,674 P93627 INFO Train loss: 0.457172
2022-02-19 23:56:00,674 P93627 INFO ************ Epoch=2 end ************
2022-02-20 00:40:09,814 P93627 INFO [Metrics] AUC: 0.806749 - logloss: 0.444793
2022-02-20 00:40:09,816 P93627 INFO Save best model: monitor(max): 0.806749
2022-02-20 00:40:09,915 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 00:40:09,963 P93627 INFO Train loss: 0.455748
2022-02-20 00:40:09,963 P93627 INFO ************ Epoch=3 end ************
2022-02-20 01:24:22,949 P93627 INFO [Metrics] AUC: 0.807500 - logloss: 0.444022
2022-02-20 01:24:22,951 P93627 INFO Save best model: monitor(max): 0.807500
2022-02-20 01:24:23,058 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 01:24:23,100 P93627 INFO Train loss: 0.455628
2022-02-20 01:24:23,101 P93627 INFO ************ Epoch=4 end ************
2022-02-20 02:08:36,558 P93627 INFO [Metrics] AUC: 0.807848 - logloss: 0.443700
2022-02-20 02:08:36,559 P93627 INFO Save best model: monitor(max): 0.807848
2022-02-20 02:08:36,678 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 02:08:36,722 P93627 INFO Train loss: 0.454984
2022-02-20 02:08:36,722 P93627 INFO ************ Epoch=5 end ************
2022-02-20 02:52:51,922 P93627 INFO [Metrics] AUC: 0.808030 - logloss: 0.443541
2022-02-20 02:52:51,923 P93627 INFO Save best model: monitor(max): 0.808030
2022-02-20 02:52:52,018 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 02:52:52,063 P93627 INFO Train loss: 0.454353
2022-02-20 02:52:52,063 P93627 INFO ************ Epoch=6 end ************
2022-02-20 03:37:10,285 P93627 INFO [Metrics] AUC: 0.808462 - logloss: 0.443134
2022-02-20 03:37:10,287 P93627 INFO Save best model: monitor(max): 0.808462
2022-02-20 03:37:10,380 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 03:37:10,424 P93627 INFO Train loss: 0.456018
2022-02-20 03:37:10,424 P93627 INFO ************ Epoch=7 end ************
2022-02-20 03:58:08,361 P93627 INFO [Metrics] AUC: 0.808616 - logloss: 0.443024
2022-02-20 03:58:08,362 P93627 INFO Save best model: monitor(max): 0.808616
2022-02-20 03:58:08,455 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 03:58:08,500 P93627 INFO Train loss: 0.453858
2022-02-20 03:58:08,501 P93627 INFO ************ Epoch=8 end ************
2022-02-20 04:15:27,202 P93627 INFO [Metrics] AUC: 0.808647 - logloss: 0.443148
2022-02-20 04:15:27,204 P93627 INFO Save best model: monitor(max): 0.808647
2022-02-20 04:15:27,331 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 04:15:27,372 P93627 INFO Train loss: 0.453799
2022-02-20 04:15:27,373 P93627 INFO ************ Epoch=9 end ************
2022-02-20 04:32:43,603 P93627 INFO [Metrics] AUC: 0.808831 - logloss: 0.442722
2022-02-20 04:32:43,605 P93627 INFO Save best model: monitor(max): 0.808831
2022-02-20 04:32:43,720 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 04:32:43,764 P93627 INFO Train loss: 0.453778
2022-02-20 04:32:43,764 P93627 INFO ************ Epoch=10 end ************
2022-02-20 04:50:01,357 P93627 INFO [Metrics] AUC: 0.809015 - logloss: 0.442769
2022-02-20 04:50:01,358 P93627 INFO Save best model: monitor(max): 0.809015
2022-02-20 04:50:01,474 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 04:50:01,515 P93627 INFO Train loss: 0.455944
2022-02-20 04:50:01,515 P93627 INFO ************ Epoch=11 end ************
2022-02-20 05:07:17,093 P93627 INFO [Metrics] AUC: 0.809135 - logloss: 0.442438
2022-02-20 05:07:17,094 P93627 INFO Save best model: monitor(max): 0.809135
2022-02-20 05:07:17,200 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 05:07:17,251 P93627 INFO Train loss: 0.453327
2022-02-20 05:07:17,252 P93627 INFO ************ Epoch=12 end ************
2022-02-20 05:24:33,653 P93627 INFO [Metrics] AUC: 0.809296 - logloss: 0.442361
2022-02-20 05:24:33,655 P93627 INFO Save best model: monitor(max): 0.809296
2022-02-20 05:24:33,767 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 05:24:33,809 P93627 INFO Train loss: 0.453476
2022-02-20 05:24:33,810 P93627 INFO ************ Epoch=13 end ************
2022-02-20 05:41:46,558 P93627 INFO [Metrics] AUC: 0.809351 - logloss: 0.442458
2022-02-20 05:41:46,560 P93627 INFO Save best model: monitor(max): 0.809351
2022-02-20 05:41:46,653 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 05:41:46,694 P93627 INFO Train loss: 0.455356
2022-02-20 05:41:46,694 P93627 INFO ************ Epoch=14 end ************
2022-02-20 05:58:56,445 P93627 INFO [Metrics] AUC: 0.809201 - logloss: 0.442453
2022-02-20 05:58:56,446 P93627 INFO Monitor(max) STOP: 0.809201 !
2022-02-20 05:58:56,447 P93627 INFO Reduce learning rate on plateau: 0.000100
2022-02-20 05:58:56,447 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 05:58:56,491 P93627 INFO Train loss: 0.453815
2022-02-20 05:58:56,491 P93627 INFO ************ Epoch=15 end ************
2022-02-20 06:16:08,206 P93627 INFO [Metrics] AUC: 0.812664 - logloss: 0.439228
2022-02-20 06:16:08,207 P93627 INFO Save best model: monitor(max): 0.812664
2022-02-20 06:16:08,301 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 06:16:08,346 P93627 INFO Train loss: 0.443312
2022-02-20 06:16:08,346 P93627 INFO ************ Epoch=16 end ************
2022-02-20 06:33:20,907 P93627 INFO [Metrics] AUC: 0.813302 - logloss: 0.438645
2022-02-20 06:33:20,909 P93627 INFO Save best model: monitor(max): 0.813302
2022-02-20 06:33:21,007 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 06:33:21,052 P93627 INFO Train loss: 0.439494
2022-02-20 06:33:21,053 P93627 INFO ************ Epoch=17 end ************
2022-02-20 06:50:40,307 P93627 INFO [Metrics] AUC: 0.813489 - logloss: 0.438535
2022-02-20 06:50:40,308 P93627 INFO Save best model: monitor(max): 0.813489
2022-02-20 06:50:40,399 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 06:50:40,437 P93627 INFO Train loss: 0.437800
2022-02-20 06:50:40,437 P93627 INFO ************ Epoch=18 end ************
2022-02-20 07:08:00,226 P93627 INFO [Metrics] AUC: 0.813539 - logloss: 0.438458
2022-02-20 07:08:00,227 P93627 INFO Save best model: monitor(max): 0.813539
2022-02-20 07:08:00,326 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 07:08:00,363 P93627 INFO Train loss: 0.436530
2022-02-20 07:08:00,364 P93627 INFO ************ Epoch=19 end ************
2022-02-20 07:25:13,965 P93627 INFO [Metrics] AUC: 0.813449 - logloss: 0.438534
2022-02-20 07:25:13,967 P93627 INFO Monitor(max) STOP: 0.813449 !
2022-02-20 07:25:13,967 P93627 INFO Reduce learning rate on plateau: 0.000010
2022-02-20 07:25:13,967 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 07:25:14,015 P93627 INFO Train loss: 0.435437
2022-02-20 07:25:14,015 P93627 INFO ************ Epoch=20 end ************
2022-02-20 07:42:32,954 P93627 INFO [Metrics] AUC: 0.813020 - logloss: 0.439159
2022-02-20 07:42:32,956 P93627 INFO Monitor(max) STOP: 0.813020 !
2022-02-20 07:42:32,956 P93627 INFO Reduce learning rate on plateau: 0.000001
2022-02-20 07:42:32,956 P93627 INFO Early stopping at epoch=21
2022-02-20 07:42:32,957 P93627 INFO --- 8058/8058 batches finished ---
2022-02-20 07:42:33,003 P93627 INFO Train loss: 0.431329
2022-02-20 07:42:33,004 P93627 INFO Training finished.
2022-02-20 07:42:33,004 P93627 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/DESTINE_criteo_x1/criteo_x1_7b681156/DESTINE_criteo_x1_001_767ae9b5.model
2022-02-20 07:42:33,097 P93627 INFO ****** Validation evaluation ******
2022-02-20 07:43:33,623 P93627 INFO [Metrics] AUC: 0.813539 - logloss: 0.438458
2022-02-20 07:43:33,720 P93627 INFO ******** Test evaluation ********
2022-02-20 07:43:33,720 P93627 INFO Loading data...
2022-02-20 07:43:33,720 P93627 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-02-20 07:43:34,521 P93627 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-02-20 07:43:34,521 P93627 INFO Loading test data done.
2022-02-20 07:44:07,960 P93627 INFO [Metrics] AUC: 0.813902 - logloss: 0.437963

```
