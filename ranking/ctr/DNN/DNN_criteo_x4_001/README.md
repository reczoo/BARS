## DNN_criteo_x4_001

A hands-on guide to run the DNN model on the Criteo_x4_001 dataset.

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
Dataset ID: [Criteo_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [DNN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DNN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DNN_criteo_x4_tuner_config_02](./DNN_criteo_x4_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DNN_criteo_x4_001
    nohup python run_expid.py --config ./DNN_criteo_x4_tuner_config_02 --expid DNN_criteo_x4_024_673e1651 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.438015 | 0.814042  |


### Logs
```python
2020-06-20 02:41:04,514 P5385 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000, 1000, 1000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DNN",
    "model_id": "DNN_criteo_x4_5c863b0f_024_8ccab1b0",
    "model_root": "./Criteo/DNN_criteo/min10/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/criteo_x4_5c863b0f/test.h5",
    "train_data": "../data/Criteo/criteo_x4_5c863b0f/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-20 02:41:04,516 P5385 INFO Set up feature encoder...
2020-06-20 02:41:04,516 P5385 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-06-20 02:41:04,517 P5385 INFO Loading data...
2020-06-20 02:41:04,522 P5385 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-06-20 02:41:10,946 P5385 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-06-20 02:41:12,742 P5385 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-06-20 02:41:12,865 P5385 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-20 02:41:12,865 P5385 INFO Loading train data done.
2020-06-20 02:41:17,331 P5385 INFO Start training: 3668 batches/epoch
2020-06-20 02:41:17,332 P5385 INFO ************ Epoch=1 start ************
2020-06-20 02:47:44,380 P5385 INFO [Metrics] logloss: 0.446030 - AUC: 0.805207
2020-06-20 02:47:44,383 P5385 INFO Save best model: monitor(max): 0.359177
2020-06-20 02:47:44,917 P5385 INFO --- 3668/3668 batches finished ---
2020-06-20 02:47:44,963 P5385 INFO Train loss: 0.460513
2020-06-20 02:47:44,963 P5385 INFO ************ Epoch=1 end ************
2020-06-20 02:54:10,435 P5385 INFO [Metrics] logloss: 0.444357 - AUC: 0.807660
2020-06-20 02:54:10,436 P5385 INFO Save best model: monitor(max): 0.363303
2020-06-20 02:54:10,536 P5385 INFO --- 3668/3668 batches finished ---
2020-06-20 02:54:10,584 P5385 INFO Train loss: 0.454956
2020-06-20 02:54:10,584 P5385 INFO ************ Epoch=2 end ************
2020-06-20 03:00:33,777 P5385 INFO [Metrics] logloss: 0.442931 - AUC: 0.808673
2020-06-20 03:00:33,779 P5385 INFO Save best model: monitor(max): 0.365742
2020-06-20 03:00:33,940 P5385 INFO --- 3668/3668 batches finished ---
2020-06-20 03:00:33,986 P5385 INFO Train loss: 0.453290
2020-06-20 03:00:33,986 P5385 INFO ************ Epoch=3 end ************
2020-06-20 03:06:56,779 P5385 INFO [Metrics] logloss: 0.442486 - AUC: 0.809373
2020-06-20 03:06:56,781 P5385 INFO Save best model: monitor(max): 0.366887
2020-06-20 03:06:56,874 P5385 INFO --- 3668/3668 batches finished ---
2020-06-20 03:06:56,944 P5385 INFO Train loss: 0.452358
2020-06-20 03:06:56,944 P5385 INFO ************ Epoch=4 end ************
2020-06-20 03:13:20,071 P5385 INFO [Metrics] logloss: 0.442034 - AUC: 0.809866
2020-06-20 03:13:20,072 P5385 INFO Save best model: monitor(max): 0.367832
2020-06-20 03:13:20,181 P5385 INFO --- 3668/3668 batches finished ---
2020-06-20 03:13:20,229 P5385 INFO Train loss: 0.451755
2020-06-20 03:13:20,229 P5385 INFO ************ Epoch=5 end ************
2020-06-20 03:19:44,774 P5385 INFO [Metrics] logloss: 0.441591 - AUC: 0.810247
2020-06-20 03:19:44,775 P5385 INFO Save best model: monitor(max): 0.368656
2020-06-20 03:19:44,878 P5385 INFO --- 3668/3668 batches finished ---
2020-06-20 03:19:44,941 P5385 INFO Train loss: 0.451312
2020-06-20 03:19:44,941 P5385 INFO ************ Epoch=6 end ************
2020-06-20 03:26:07,601 P5385 INFO [Metrics] logloss: 0.442325 - AUC: 0.810288
2020-06-20 03:26:07,602 P5385 INFO Monitor(max) STOP: 0.367963 !
2020-06-20 03:26:07,602 P5385 INFO Reduce learning rate on plateau: 0.000100
2020-06-20 03:26:07,603 P5385 INFO --- 3668/3668 batches finished ---
2020-06-20 03:26:07,650 P5385 INFO Train loss: 0.450905
2020-06-20 03:26:07,650 P5385 INFO ************ Epoch=7 end ************
2020-06-20 03:32:34,662 P5385 INFO [Metrics] logloss: 0.438804 - AUC: 0.813236
2020-06-20 03:32:34,664 P5385 INFO Save best model: monitor(max): 0.374432
2020-06-20 03:32:34,816 P5385 INFO --- 3668/3668 batches finished ---
2020-06-20 03:32:34,880 P5385 INFO Train loss: 0.440567
2020-06-20 03:32:34,881 P5385 INFO ************ Epoch=8 end ************
2020-06-20 03:39:01,317 P5385 INFO [Metrics] logloss: 0.438478 - AUC: 0.813617
2020-06-20 03:39:01,318 P5385 INFO Save best model: monitor(max): 0.375139
2020-06-20 03:39:01,406 P5385 INFO --- 3668/3668 batches finished ---
2020-06-20 03:39:01,469 P5385 INFO Train loss: 0.436241
2020-06-20 03:39:01,470 P5385 INFO ************ Epoch=9 end ************
2020-06-20 03:45:25,525 P5385 INFO [Metrics] logloss: 0.438367 - AUC: 0.813674
2020-06-20 03:45:25,526 P5385 INFO Save best model: monitor(max): 0.375307
2020-06-20 03:45:25,633 P5385 INFO --- 3668/3668 batches finished ---
2020-06-20 03:45:25,680 P5385 INFO Train loss: 0.434088
2020-06-20 03:45:25,681 P5385 INFO ************ Epoch=10 end ************
2020-06-20 03:51:49,621 P5385 INFO [Metrics] logloss: 0.438418 - AUC: 0.813556
2020-06-20 03:51:49,622 P5385 INFO Monitor(max) STOP: 0.375137 !
2020-06-20 03:51:49,622 P5385 INFO Reduce learning rate on plateau: 0.000010
2020-06-20 03:51:49,622 P5385 INFO --- 3668/3668 batches finished ---
2020-06-20 03:51:49,672 P5385 INFO Train loss: 0.432383
2020-06-20 03:51:49,672 P5385 INFO ************ Epoch=11 end ************
2020-06-20 03:58:14,130 P5385 INFO [Metrics] logloss: 0.438773 - AUC: 0.813179
2020-06-20 03:58:14,131 P5385 INFO Monitor(max) STOP: 0.374406 !
2020-06-20 03:58:14,131 P5385 INFO Reduce learning rate on plateau: 0.000001
2020-06-20 03:58:14,131 P5385 INFO Early stopping at epoch=12
2020-06-20 03:58:14,131 P5385 INFO --- 3668/3668 batches finished ---
2020-06-20 03:58:14,180 P5385 INFO Train loss: 0.428424
2020-06-20 03:58:14,180 P5385 INFO Training finished.
2020-06-20 03:58:14,180 P5385 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/DNN_criteo/min10/criteo_x4_5c863b0f/DNN_criteo_x4_5c863b0f_024_8ccab1b0_model.ckpt
2020-06-20 03:58:14,320 P5385 INFO ****** Train/validation evaluation ******
2020-06-20 04:01:58,693 P5385 INFO [Metrics] logloss: 0.423383 - AUC: 0.830613
2020-06-20 04:02:25,393 P5385 INFO [Metrics] logloss: 0.438367 - AUC: 0.813674
2020-06-20 04:02:25,470 P5385 INFO ******** Test evaluation ********
2020-06-20 04:02:25,471 P5385 INFO Loading data...
2020-06-20 04:02:25,471 P5385 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-06-20 04:02:26,484 P5385 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-20 04:02:26,484 P5385 INFO Loading test data done.
2020-06-20 04:02:55,118 P5385 INFO [Metrics] logloss: 0.438015 - AUC: 0.814042

```
