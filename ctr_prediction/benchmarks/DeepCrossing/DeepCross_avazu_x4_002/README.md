## DeepCross_avazu_x4_002

A hands-on guide to run the DeepCrossing model on the Avazu_x4_002 dataset.

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
Dataset ID: [Avazu_x4_002](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x4_002). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [DeepCrossing](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DeepCrossing.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DeepCrossing_avazu_x4_tuner_config_01](./DeepCrossing_avazu_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DeepCross_avazu_x4_002
    nohup python run_expid.py --config ./DeepCrossing_avazu_x4_tuner_config_01 --expid DeepCrossing_avazu_x4_040_aa3259c8 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.370019 | 0.796245  |


### Logs
```python
2020-01-22 23:29:42,958 P18322 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "dnn_activations": "relu",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DeepCrossing",
    "model_id": "DeepCrossing_criteo_x4_040_4cbdb15a",
    "model_root": "./Avazu/DeepCrossing_avazu/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "3",
    "pickle_feature_encoder": "True",
    "residual_blocks": "[1000, 1000, 1000, 1000]",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "use_residual": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "version": "pytorch",
    "gpu": "1"
}
2020-01-22 23:29:42,959 P18322 INFO Set up feature encoder...
2020-01-22 23:29:42,959 P18322 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-01-22 23:29:42,959 P18322 INFO Loading data...
2020-01-22 23:29:42,961 P18322 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-01-22 23:29:45,653 P18322 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-01-22 23:29:46,894 P18322 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-01-22 23:29:47,006 P18322 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-01-22 23:29:47,006 P18322 INFO Loading train data done.
2020-01-22 23:29:57,916 P18322 INFO **** Start training: 3235 batches/epoch ****
2020-01-22 23:42:10,306 P18322 INFO [Metrics] logloss: 0.370044 - AUC: 0.796185
2020-01-22 23:42:10,366 P18322 INFO Save best model: monitor(max): 0.426141
2020-01-22 23:42:11,619 P18322 INFO --- 3235/3235 batches finished ---
2020-01-22 23:42:11,659 P18322 INFO Train loss: 0.380016
2020-01-22 23:42:11,659 P18322 INFO ************ Epoch=1 end ************
2020-01-22 23:54:25,867 P18322 INFO [Metrics] logloss: 0.428020 - AUC: 0.765297
2020-01-22 23:54:25,927 P18322 INFO Monitor(max) STOP: 0.337277 !
2020-01-22 23:54:25,928 P18322 INFO Reduce learning rate on plateau: 0.000100
2020-01-22 23:54:25,928 P18322 INFO --- 3235/3235 batches finished ---
2020-01-22 23:54:26,009 P18322 INFO Train loss: 0.285363
2020-01-22 23:54:26,009 P18322 INFO ************ Epoch=2 end ************
2020-01-23 00:06:39,466 P18322 INFO [Metrics] logloss: 0.550587 - AUC: 0.754590
2020-01-23 00:06:39,530 P18322 INFO Monitor(max) STOP: 0.204003 !
2020-01-23 00:06:39,530 P18322 INFO Reduce learning rate on plateau: 0.000010
2020-01-23 00:06:39,530 P18322 INFO --- 3235/3235 batches finished ---
2020-01-23 00:06:39,610 P18322 INFO Train loss: 0.246087
2020-01-23 00:06:39,610 P18322 INFO ************ Epoch=3 end ************
2020-01-23 00:18:52,812 P18322 INFO [Metrics] logloss: 0.642999 - AUC: 0.747964
2020-01-23 00:18:52,868 P18322 INFO Monitor(max) STOP: 0.104965 !
2020-01-23 00:18:52,869 P18322 INFO Reduce learning rate on plateau: 0.000001
2020-01-23 00:18:52,869 P18322 INFO Early stopping at epoch=4
2020-01-23 00:18:52,869 P18322 INFO --- 3235/3235 batches finished ---
2020-01-23 00:18:52,952 P18322 INFO Train loss: 0.233712
2020-01-23 00:18:52,952 P18322 INFO Training finished.
2020-01-23 00:18:52,952 P18322 INFO Load best model: /home/XXX/benchmarks/Avazu/DeepCrossing_avazu/avazu_x4_001_d45ad60e/DeepCrossing_criteo_x4_040_4cbdb15a_avazu_x4_001_d45ad60e_model.ckpt
2020-01-23 00:18:54,729 P18322 INFO ****** Train/validation evaluation ******
2020-01-23 00:23:02,427 P18322 INFO [Metrics] logloss: 0.321607 - AUC: 0.868086
2020-01-23 00:23:33,251 P18322 INFO [Metrics] logloss: 0.370044 - AUC: 0.796185
2020-01-23 00:23:33,371 P18322 INFO ******** Test evaluation ********
2020-01-23 00:23:33,371 P18322 INFO Loading data...
2020-01-23 00:23:33,371 P18322 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-01-23 00:23:34,159 P18322 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-01-23 00:23:34,159 P18322 INFO Loading test data done.
2020-01-23 00:24:04,490 P18322 INFO [Metrics] logloss: 0.370019 - AUC: 0.796245

```
