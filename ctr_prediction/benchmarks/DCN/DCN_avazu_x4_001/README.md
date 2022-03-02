## DCN_avazu_x4_001

A hands-on guide to run the DCN model on the Avazu_x4_001 dataset.

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
Dataset ID: [Avazu_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [DCN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DCN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DCN_avazu_x4_tuner_config_02](./DCN_avazu_x4_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DCN_avazu_x4_001
    nohup python run_expid.py --config ./DCN_avazu_x4_tuner_config_02 --expid DCN_avazu_x4_018_8f445da6 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.371936 | 0.793061  |


### Logs
```python
2020-06-13 17:44:00,674 P4075 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "crossing_layers": "3",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[2000, 2000, 2000]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-08",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DCN",
    "model_id": "DCN_avazu_x4_3bbbc4c9_018_b2ab697a",
    "model_root": "./Avazu/DCN_avazu/min2/",
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
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-13 17:44:00,677 P4075 INFO Set up feature encoder...
2020-06-13 17:44:00,677 P4075 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-06-13 17:44:00,677 P4075 INFO Loading data...
2020-06-13 17:44:00,681 P4075 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-06-13 17:44:04,530 P4075 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-06-13 17:44:06,164 P4075 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-13 17:44:06,269 P4075 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-13 17:44:06,270 P4075 INFO Loading train data done.
2020-06-13 17:44:11,963 P4075 INFO Start training: 3235 batches/epoch
2020-06-13 17:44:11,964 P4075 INFO ************ Epoch=1 start ************
2020-06-13 17:50:38,489 P4075 INFO [Metrics] logloss: 0.372058 - AUC: 0.792786
2020-06-13 17:50:38,489 P4075 INFO Save best model: monitor(max): 0.420728
2020-06-13 17:50:38,791 P4075 INFO --- 3235/3235 batches finished ---
2020-06-13 17:50:38,850 P4075 INFO Train loss: 0.380479
2020-06-13 17:50:38,850 P4075 INFO ************ Epoch=1 end ************
2020-06-13 17:57:03,463 P4075 INFO [Metrics] logloss: 0.379733 - AUC: 0.788047
2020-06-13 17:57:03,468 P4075 INFO Monitor(max) STOP: 0.408314 !
2020-06-13 17:57:03,468 P4075 INFO Reduce learning rate on plateau: 0.000100
2020-06-13 17:57:03,468 P4075 INFO --- 3235/3235 batches finished ---
2020-06-13 17:57:03,522 P4075 INFO Train loss: 0.334254
2020-06-13 17:57:03,522 P4075 INFO ************ Epoch=2 end ************
2020-06-13 18:03:31,323 P4075 INFO [Metrics] logloss: 0.423422 - AUC: 0.776726
2020-06-13 18:03:31,329 P4075 INFO Monitor(max) STOP: 0.353304 !
2020-06-13 18:03:31,329 P4075 INFO Reduce learning rate on plateau: 0.000010
2020-06-13 18:03:31,329 P4075 INFO Early stopping at epoch=3
2020-06-13 18:03:31,329 P4075 INFO --- 3235/3235 batches finished ---
2020-06-13 18:03:31,387 P4075 INFO Train loss: 0.294529
2020-06-13 18:03:31,387 P4075 INFO Training finished.
2020-06-13 18:03:31,387 P4075 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Avazu/DCN_avazu/min2/avazu_x4_3bbbc4c9/DCN_avazu_x4_3bbbc4c9_018_b2ab697a_model.ckpt
2020-06-13 18:03:31,891 P4075 INFO ****** Train/validation evaluation ******
2020-06-13 18:06:58,280 P4075 INFO [Metrics] logloss: 0.339480 - AUC: 0.843293
2020-06-13 18:07:21,422 P4075 INFO [Metrics] logloss: 0.372058 - AUC: 0.792786
2020-06-13 18:07:21,501 P4075 INFO ******** Test evaluation ********
2020-06-13 18:07:21,501 P4075 INFO Loading data...
2020-06-13 18:07:21,501 P4075 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-06-13 18:07:22,092 P4075 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-13 18:07:22,092 P4075 INFO Loading test data done.
2020-06-13 18:07:44,456 P4075 INFO [Metrics] logloss: 0.371936 - AUC: 0.793061

```
