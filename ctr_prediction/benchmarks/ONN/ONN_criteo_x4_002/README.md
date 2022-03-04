## ONN_criteo_x4_002

A hands-on guide to run the ONN model on the Criteo_x4_002 dataset.

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
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [Criteo_x4_002](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x4_002). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [ONN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/ONN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [ONN_criteo_x4_tuner_config_01](./ONN_criteo_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd ONN_criteo_x4_002
    nohup python run_expid.py --config ./ONN_criteo_x4_tuner_config_01 --expid ONN_criteo_x4_106_7a261a01 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.438053 | 0.814149  |


### Logs
```python
2020-01-21 06:58:29,586 P16076 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "dataset_id": "criteo_x4_001_be98441d",
    "embedding_dim": "2",
    "embedding_dropout": "0",
    "embedding_regularizer": "l2(1.e-5)",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000, 1000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "ONN",
    "model_id": "ONN_criteo_x4_106_8677e86c",
    "model_root": "./Criteo/ONN_criteo/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "3",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "test_data": "../data/Criteo/criteo_x4_001_be98441d/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_be98441d/train.h5",
    "valid_data": "../data/Criteo/criteo_x4_001_be98441d/valid.h5",
    "version": "pytorch",
    "gpu": "0"
}
2020-01-21 06:58:29,587 P16076 INFO Set up feature encoder...
2020-01-21 06:58:29,587 P16076 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-01-21 06:58:29,587 P16076 INFO Loading data...
2020-01-21 06:58:29,589 P16076 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-01-21 06:58:35,199 P16076 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-01-21 06:58:37,575 P16076 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-01-21 06:58:37,785 P16076 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-01-21 06:58:37,786 P16076 INFO Loading train data done.
2020-01-21 06:58:50,759 P16076 INFO **** Start training: 3668 batches/epoch ****
2020-01-21 09:12:57,306 P16076 INFO [Metrics] logloss: 0.443518 - AUC: 0.808186
2020-01-21 09:12:57,364 P16076 INFO Save best model: monitor(max): 0.364668
2020-01-21 09:12:59,028 P16076 INFO --- 3668/3668 batches finished ---
2020-01-21 09:12:59,114 P16076 INFO Train loss: 0.462550
2020-01-21 09:12:59,115 P16076 INFO ************ Epoch=1 end ************
2020-01-21 11:26:20,409 P16076 INFO [Metrics] logloss: 0.442059 - AUC: 0.809987
2020-01-21 11:26:20,467 P16076 INFO Save best model: monitor(max): 0.367928
2020-01-21 11:26:22,701 P16076 INFO --- 3668/3668 batches finished ---
2020-01-21 11:26:22,791 P16076 INFO Train loss: 0.458383
2020-01-21 11:26:22,792 P16076 INFO ************ Epoch=2 end ************
2020-01-21 13:40:10,837 P16076 INFO [Metrics] logloss: 0.441561 - AUC: 0.810515
2020-01-21 13:40:10,895 P16076 INFO Save best model: monitor(max): 0.368954
2020-01-21 13:40:13,082 P16076 INFO --- 3668/3668 batches finished ---
2020-01-21 13:40:13,172 P16076 INFO Train loss: 0.457330
2020-01-21 13:40:13,173 P16076 INFO ************ Epoch=3 end ************
2020-01-21 15:53:32,280 P16076 INFO [Metrics] logloss: 0.441023 - AUC: 0.810956
2020-01-21 15:53:32,337 P16076 INFO Save best model: monitor(max): 0.369932
2020-01-21 15:53:34,555 P16076 INFO --- 3668/3668 batches finished ---
2020-01-21 15:53:34,647 P16076 INFO Train loss: 0.456840
2020-01-21 15:53:34,648 P16076 INFO ************ Epoch=4 end ************
2020-01-21 18:06:53,372 P16076 INFO [Metrics] logloss: 0.440630 - AUC: 0.811345
2020-01-21 18:06:53,428 P16076 INFO Save best model: monitor(max): 0.370714
2020-01-21 18:06:55,665 P16076 INFO --- 3668/3668 batches finished ---
2020-01-21 18:06:55,754 P16076 INFO Train loss: 0.456591
2020-01-21 18:06:55,755 P16076 INFO ************ Epoch=5 end ************
2020-01-21 20:20:02,570 P16076 INFO [Metrics] logloss: 0.440326 - AUC: 0.811498
2020-01-21 20:20:02,629 P16076 INFO Save best model: monitor(max): 0.371172
2020-01-21 20:20:04,889 P16076 INFO --- 3668/3668 batches finished ---
2020-01-21 20:20:04,980 P16076 INFO Train loss: 0.456443
2020-01-21 20:20:04,981 P16076 INFO ************ Epoch=6 end ************
2020-01-21 22:33:26,565 P16076 INFO [Metrics] logloss: 0.440371 - AUC: 0.811512
2020-01-21 22:33:26,624 P16076 INFO Monitor(max) STOP: 0.371141 !
2020-01-21 22:33:26,624 P16076 INFO Reduce learning rate on plateau: 0.000100
2020-01-21 22:33:26,624 P16076 INFO --- 3668/3668 batches finished ---
2020-01-21 22:33:26,716 P16076 INFO Train loss: 0.456441
2020-01-21 22:33:26,718 P16076 INFO ************ Epoch=7 end ************
2020-01-22 00:46:18,280 P16076 INFO [Metrics] logloss: 0.438513 - AUC: 0.813661
2020-01-22 00:46:18,363 P16076 INFO Save best model: monitor(max): 0.375149
2020-01-22 00:46:20,660 P16076 INFO --- 3668/3668 batches finished ---
2020-01-22 00:46:20,751 P16076 INFO Train loss: 0.438992
2020-01-22 00:46:20,751 P16076 INFO ************ Epoch=8 end ************
2020-01-22 02:59:10,359 P16076 INFO [Metrics] logloss: 0.439353 - AUC: 0.813081
2020-01-22 02:59:10,455 P16076 INFO Monitor(max) STOP: 0.373729 !
2020-01-22 02:59:10,456 P16076 INFO Reduce learning rate on plateau: 0.000010
2020-01-22 02:59:10,456 P16076 INFO --- 3668/3668 batches finished ---
2020-01-22 02:59:10,554 P16076 INFO Train loss: 0.432104
2020-01-22 02:59:10,555 P16076 INFO ************ Epoch=9 end ************
2020-01-22 05:12:46,508 P16076 INFO [Metrics] logloss: 0.443001 - AUC: 0.810447
2020-01-22 05:12:46,581 P16076 INFO Monitor(max) STOP: 0.367446 !
2020-01-22 05:12:46,581 P16076 INFO Reduce learning rate on plateau: 0.000001
2020-01-22 05:12:46,581 P16076 INFO --- 3668/3668 batches finished ---
2020-01-22 05:12:46,668 P16076 INFO Train loss: 0.421968
2020-01-22 05:12:46,668 P16076 INFO ************ Epoch=10 end ************
2020-01-22 07:27:07,040 P16076 INFO [Metrics] logloss: 0.443565 - AUC: 0.810159
2020-01-22 07:27:07,101 P16076 INFO Monitor(max) STOP: 0.366594 !
2020-01-22 07:27:07,102 P16076 INFO Reduce learning rate on plateau: 0.000001
2020-01-22 07:27:07,102 P16076 INFO Early stopping at epoch=11
2020-01-22 07:27:07,102 P16076 INFO --- 3668/3668 batches finished ---
2020-01-22 07:27:07,227 P16076 INFO Train loss: 0.419710
2020-01-22 07:27:07,227 P16076 INFO Training finished.
2020-01-22 07:27:07,227 P16076 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/ONN_criteo/criteo_x4_001_be98441d/ONN_criteo_x4_106_8677e86c_criteo_x4_001_be98441d_model.ckpt
2020-01-22 07:27:09,794 P16076 INFO ****** Train/validation evaluation ******
2020-01-22 07:42:03,799 P16076 INFO [Metrics] logloss: 0.419401 - AUC: 0.833183
2020-01-22 07:43:52,955 P16076 INFO [Metrics] logloss: 0.438513 - AUC: 0.813661
2020-01-22 07:43:53,171 P16076 INFO ******** Test evaluation ********
2020-01-22 07:43:53,172 P16076 INFO Loading data...
2020-01-22 07:43:53,172 P16076 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-01-22 07:43:53,912 P16076 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-01-22 07:43:53,912 P16076 INFO Loading test data done.
2020-01-22 07:45:42,293 P16076 INFO [Metrics] logloss: 0.438053 - AUC: 0.814149

```
