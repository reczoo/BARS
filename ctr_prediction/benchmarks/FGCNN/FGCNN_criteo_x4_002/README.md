## FGCNN_criteo_x4_002

A hands-on guide to run the FGCNN model on the Criteo_x4_002 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FGCNN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FGCNN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FGCNN_criteo_x4_tuner_config_02](./FGCNN_criteo_x4_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FGCNN_criteo_x4_002
    nohup python run_expid.py --config ./FGCNN_criteo_x4_tuner_config_02 --expid FGCNN_criteo_x4_004_50ff06b8 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.438142 | 0.814181  |


### Logs
```python
2020-02-10 14:23:37,371 P594 INFO {
    "batch_size": "2000",
    "channels": "[38, 40, 42, 44]",
    "conv_activation": "Tanh",
    "conv_batch_norm": "False",
    "dataset_id": "criteo_x4_001_be98441d",
    "dnn_activations": "ReLU",
    "dnn_batch_norm": "True",
    "dnn_hidden_units": "[4096, 2048, 1024]",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "embedding_regularizer": "l2(1.e-6)",
    "epochs": "100",
    "every_x_epochs": "1",
    "kernel_heights": "9",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FGCNN",
    "model_id": "FGCNN_criteo_x4_004_4e919cfc",
    "model_root": "./Criteo/FGCNN_criteo/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "pooling_sizes": "2",
    "recombined_channels": "3",
    "save_best_only": "True",
    "seed": "2019",
    "share_embedding": "False",
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
    "gpu": "2"
}
2020-02-10 14:23:37,373 P594 INFO Set up feature encoder...
2020-02-10 14:23:37,373 P594 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-02-10 14:23:37,374 P594 INFO Loading data...
2020-02-10 14:23:37,386 P594 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-02-10 14:23:42,086 P594 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-02-10 14:23:43,780 P594 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-02-10 14:23:43,917 P594 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-10 14:23:43,918 P594 INFO Loading train data done.
2020-02-10 14:23:56,295 P594 INFO **** Start training: 18337 batches/epoch ****
2020-02-10 17:35:02,786 P594 INFO [Metrics] logloss: 0.445651 - AUC: 0.806810
2020-02-10 17:35:02,896 P594 INFO Save best model: monitor(max): 0.361159
2020-02-10 17:35:04,086 P594 INFO --- 18337/18337 batches finished ---
2020-02-10 17:35:04,147 P594 INFO Train loss: 0.458280
2020-02-10 17:35:04,147 P594 INFO ************ Epoch=1 end ************
2020-02-10 20:45:24,640 P594 INFO [Metrics] logloss: 0.443401 - AUC: 0.809772
2020-02-10 20:45:24,741 P594 INFO Save best model: monitor(max): 0.366371
2020-02-10 20:45:26,072 P594 INFO --- 18337/18337 batches finished ---
2020-02-10 20:45:26,122 P594 INFO Train loss: 0.451120
2020-02-10 20:45:26,123 P594 INFO ************ Epoch=2 end ************
2020-02-10 23:56:00,475 P594 INFO [Metrics] logloss: 0.440915 - AUC: 0.811097
2020-02-10 23:56:00,594 P594 INFO Save best model: monitor(max): 0.370182
2020-02-10 23:56:01,952 P594 INFO --- 18337/18337 batches finished ---
2020-02-10 23:56:02,021 P594 INFO Train loss: 0.449226
2020-02-10 23:56:02,022 P594 INFO ************ Epoch=3 end ************
2020-02-11 03:04:52,574 P594 INFO [Metrics] logloss: 0.440278 - AUC: 0.811545
2020-02-11 03:04:52,663 P594 INFO Save best model: monitor(max): 0.371267
2020-02-11 03:04:54,129 P594 INFO --- 18337/18337 batches finished ---
2020-02-11 03:04:54,182 P594 INFO Train loss: 0.448041
2020-02-11 03:04:54,182 P594 INFO ************ Epoch=4 end ************
2020-02-11 06:13:11,382 P594 INFO [Metrics] logloss: 0.440607 - AUC: 0.811679
2020-02-11 06:13:11,471 P594 INFO Monitor(max) STOP: 0.371071 !
2020-02-11 06:13:11,472 P594 INFO Reduce learning rate on plateau: 0.000100
2020-02-11 06:13:11,472 P594 INFO --- 18337/18337 batches finished ---
2020-02-11 06:13:11,540 P594 INFO Train loss: 0.447033
2020-02-11 06:13:11,540 P594 INFO ************ Epoch=5 end ************
2020-02-11 09:19:47,277 P594 INFO [Metrics] logloss: 0.438523 - AUC: 0.813785
2020-02-11 09:19:47,374 P594 INFO Save best model: monitor(max): 0.375262
2020-02-11 09:19:48,808 P594 INFO --- 18337/18337 batches finished ---
2020-02-11 09:19:48,879 P594 INFO Train loss: 0.433939
2020-02-11 09:19:48,879 P594 INFO ************ Epoch=6 end ************
2020-02-11 12:28:16,146 P594 INFO [Metrics] logloss: 0.439474 - AUC: 0.812964
2020-02-11 12:28:16,231 P594 INFO Monitor(max) STOP: 0.373490 !
2020-02-11 12:28:16,232 P594 INFO Reduce learning rate on plateau: 0.000010
2020-02-11 12:28:16,232 P594 INFO --- 18337/18337 batches finished ---
2020-02-11 12:28:16,288 P594 INFO Train loss: 0.428639
2020-02-11 12:28:16,288 P594 INFO ************ Epoch=7 end ************
2020-02-11 15:42:25,505 P594 INFO [Metrics] logloss: 0.442128 - AUC: 0.811074
2020-02-11 15:42:25,647 P594 INFO Monitor(max) STOP: 0.368946 !
2020-02-11 15:42:25,647 P594 INFO Reduce learning rate on plateau: 0.000001
2020-02-11 15:42:25,647 P594 INFO Early stopping at epoch=8
2020-02-11 15:42:25,648 P594 INFO --- 18337/18337 batches finished ---
2020-02-11 15:42:25,706 P594 INFO Train loss: 0.421770
2020-02-11 15:42:25,707 P594 INFO Training finished.
2020-02-11 15:42:25,707 P594 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/FGCNN_criteo/criteo_x4_001_be98441d/FGCNN_criteo_x4_004_4e919cfc_criteo_x4_001_be98441d_model.ckpt
2020-02-11 15:42:27,169 P594 INFO ****** Train/validation evaluation ******
2020-02-11 16:41:41,319 P594 INFO [Metrics] logloss: 0.420347 - AUC: 0.832032
2020-02-11 16:44:44,762 P594 INFO [Metrics] logloss: 0.438523 - AUC: 0.813785
2020-02-11 16:44:45,080 P594 INFO ******** Test evaluation ********
2020-02-11 16:44:45,080 P594 INFO Loading data...
2020-02-11 16:44:45,080 P594 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-02-11 16:44:46,283 P594 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-11 16:44:46,283 P594 INFO Loading test data done.
2020-02-11 16:47:47,015 P594 INFO [Metrics] logloss: 0.438142 - AUC: 0.814181

```
