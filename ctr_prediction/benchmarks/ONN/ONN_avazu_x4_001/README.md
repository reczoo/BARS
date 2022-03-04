## ONN_avazu_x4_001

A hands-on guide to run the ONN model on the Avazu_x4_001 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
  GPU: Tesla P100 16G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 10.0
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [Avazu_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [ONN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/ONN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [ONN_avazu_x4_tuner_config_01](./ONN_avazu_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd ONN_avazu_x4_001
    nohup python run_expid.py --config ./ONN_avazu_x4_tuner_config_01 --expid ONN_avazu_x4_006_614049da --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.368328 | 0.799150  |


### Logs
```python
2020-07-17 20:42:09,450 P33536 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "embedding_dim": "8",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000, 1000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "ONN",
    "model_id": "ONN_avazu_x4_3bbbc4c9_006_d669ec93",
    "model_root": "./Avazu/ONN_avazu/min2/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/avazu_x4_3bbbc4c9/test.h5",
    "train_data": "../data/Avazu/avazu_x4_3bbbc4c9/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/avazu_x4_3bbbc4c9/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-17 20:42:09,451 P33536 INFO Set up feature encoder...
2020-07-17 20:42:09,451 P33536 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-07-17 20:42:37,582 P33536 INFO Total number of parameters: 723660201.
2020-07-17 20:42:37,583 P33536 INFO Loading data...
2020-07-17 20:42:37,587 P33536 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-07-17 20:42:44,520 P33536 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-07-17 20:42:46,077 P33536 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-07-17 20:42:46,186 P33536 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-07-17 20:42:46,187 P33536 INFO Loading train data done.
2020-07-17 20:43:19,816 P33536 INFO Start training: 3235 batches/epoch
2020-07-17 20:43:19,816 P33536 INFO ************ Epoch=1 start ************
2020-07-17 22:44:36,166 P33536 INFO [Metrics] logloss: 0.368429 - AUC: 0.798979
2020-07-17 22:44:36,167 P33536 INFO Save best model: monitor(max): 0.430550
2020-07-17 22:44:38,918 P33536 INFO --- 3235/3235 batches finished ---
2020-07-17 22:44:39,066 P33536 INFO Train loss: 0.377100
2020-07-17 22:44:39,066 P33536 INFO ************ Epoch=1 end ************
2020-07-18 00:46:03,776 P33536 INFO [Metrics] logloss: 0.387986 - AUC: 0.787972
2020-07-18 00:46:03,781 P33536 INFO Monitor(max) STOP: 0.399986 !
2020-07-18 00:46:03,781 P33536 INFO Reduce learning rate on plateau: 0.000100
2020-07-18 00:46:03,781 P33536 INFO --- 3235/3235 batches finished ---
2020-07-18 00:46:03,935 P33536 INFO Train loss: 0.306951
2020-07-18 00:46:03,935 P33536 INFO ************ Epoch=2 end ************
2020-07-18 02:47:23,378 P33536 INFO [Metrics] logloss: 0.477172 - AUC: 0.768023
2020-07-18 02:47:23,382 P33536 INFO Monitor(max) STOP: 0.290850 !
2020-07-18 02:47:23,382 P33536 INFO Reduce learning rate on plateau: 0.000010
2020-07-18 02:47:23,382 P33536 INFO Early stopping at epoch=3
2020-07-18 02:47:23,382 P33536 INFO --- 3235/3235 batches finished ---
2020-07-18 02:47:23,532 P33536 INFO Train loss: 0.251349
2020-07-18 02:47:23,532 P33536 INFO Training finished.
2020-07-18 02:47:23,532 P33536 INFO Load best model: /home/XXX/benchmarks/Avazu/ONN_avazu/min2/avazu_x4_3bbbc4c9/ONN_avazu_x4_3bbbc4c9_006_d669ec93_model.ckpt
2020-07-18 02:47:27,255 P33536 INFO ****** Train/validation evaluation ******
2020-07-18 02:52:42,613 P33536 INFO [Metrics] logloss: 0.322398 - AUC: 0.866123
2020-07-18 02:53:13,833 P33536 INFO [Metrics] logloss: 0.368429 - AUC: 0.798979
2020-07-18 02:53:13,896 P33536 INFO ******** Test evaluation ********
2020-07-18 02:53:13,896 P33536 INFO Loading data...
2020-07-18 02:53:13,896 P33536 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-07-18 02:53:14,520 P33536 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-07-18 02:53:14,521 P33536 INFO Loading test data done.
2020-07-18 02:53:45,602 P33536 INFO [Metrics] logloss: 0.368328 - AUC: 0.799150

```
