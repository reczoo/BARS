## FiGNN_criteo_x4_002

A hands-on guide to run the FiGNN model on the Criteo_x4_002 dataset.

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
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [Criteo_x4](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x4). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FiGNN](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FiGNN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FiGNN_criteo_x4_tuner_config_03](./FiGNN_criteo_x4_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FiGNN_criteo_x4_002
    nohup python run_expid.py --config ./FiGNN_criteo_x4_tuner_config_03 --expid FiGNN_criteo_x4_005_033c41f1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.437928 | 0.814073  |


### Logs
```python
2020-05-30 09:29:48,527 P3357 INFO {
    "batch_size": "5000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_001_be98441d",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "gnn_layers": "3",
    "gpu": "0",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FiGNN",
    "model_id": "FiGNN_criteo_x4_005_f8fc849a",
    "model_root": "./Criteo/FiGNN_criteo/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reuse_graph_layer": "False",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/criteo_x4_001_be98441d/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_be98441d/train.h5",
    "use_gru": "True",
    "use_hdf5": "True",
    "use_residual": "True",
    "valid_data": "../data/Criteo/criteo_x4_001_be98441d/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-05-30 09:29:48,529 P3357 INFO Set up feature encoder...
2020-05-30 09:29:48,529 P3357 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-05-30 09:29:48,530 P3357 INFO Loading data...
2020-05-30 09:29:48,535 P3357 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-05-30 09:29:55,226 P3357 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-05-30 09:29:57,582 P3357 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-05-30 09:29:57,712 P3357 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-05-30 09:29:57,712 P3357 INFO Loading train data done.
2020-05-30 09:30:08,818 P3357 INFO **** Start training: 7335 batches/epoch ****
2020-05-30 10:55:22,464 P3357 INFO [Metrics] logloss: 0.447425 - AUC: 0.803576
2020-05-30 10:55:22,490 P3357 INFO Save best model: monitor(max): 0.356151
2020-05-30 10:55:23,548 P3357 INFO --- 7335/7335 batches finished ---
2020-05-30 10:55:23,608 P3357 INFO Train loss: 0.465154
2020-05-30 10:55:23,608 P3357 INFO ************ Epoch=1 end ************
2020-05-30 12:20:31,431 P3357 INFO [Metrics] logloss: 0.445109 - AUC: 0.806202
2020-05-30 12:20:31,433 P3357 INFO Save best model: monitor(max): 0.361093
2020-05-30 12:20:33,182 P3357 INFO --- 7335/7335 batches finished ---
2020-05-30 12:20:33,243 P3357 INFO Train loss: 0.457918
2020-05-30 12:20:33,244 P3357 INFO ************ Epoch=2 end ************
2020-05-30 13:45:37,085 P3357 INFO [Metrics] logloss: 0.444072 - AUC: 0.807350
2020-05-30 13:45:37,086 P3357 INFO Save best model: monitor(max): 0.363278
2020-05-30 13:45:38,909 P3357 INFO --- 7335/7335 batches finished ---
2020-05-30 13:45:38,994 P3357 INFO Train loss: 0.456019
2020-05-30 13:45:38,994 P3357 INFO ************ Epoch=3 end ************
2020-05-30 15:10:43,253 P3357 INFO [Metrics] logloss: 0.443703 - AUC: 0.807840
2020-05-30 15:10:43,260 P3357 INFO Save best model: monitor(max): 0.364137
2020-05-30 15:10:45,044 P3357 INFO --- 7335/7335 batches finished ---
2020-05-30 15:10:45,107 P3357 INFO Train loss: 0.455349
2020-05-30 15:10:45,107 P3357 INFO ************ Epoch=4 end ************
2020-05-30 16:35:49,503 P3357 INFO [Metrics] logloss: 0.442861 - AUC: 0.808600
2020-05-30 16:35:49,524 P3357 INFO Save best model: monitor(max): 0.365739
2020-05-30 16:35:51,350 P3357 INFO --- 7335/7335 batches finished ---
2020-05-30 16:35:51,430 P3357 INFO Train loss: 0.454972
2020-05-30 16:35:51,431 P3357 INFO ************ Epoch=5 end ************
2020-05-30 18:00:54,588 P3357 INFO [Metrics] logloss: 0.442469 - AUC: 0.809008
2020-05-30 18:00:54,595 P3357 INFO Save best model: monitor(max): 0.366539
2020-05-30 18:00:56,437 P3357 INFO --- 7335/7335 batches finished ---
2020-05-30 18:00:56,501 P3357 INFO Train loss: 0.454689
2020-05-30 18:00:56,501 P3357 INFO ************ Epoch=6 end ************
2020-05-30 19:26:00,906 P3357 INFO [Metrics] logloss: 0.442173 - AUC: 0.809342
2020-05-30 19:26:00,907 P3357 INFO Save best model: monitor(max): 0.367169
2020-05-30 19:26:02,640 P3357 INFO --- 7335/7335 batches finished ---
2020-05-30 19:26:02,694 P3357 INFO Train loss: 0.454470
2020-05-30 19:26:02,694 P3357 INFO ************ Epoch=7 end ************
2020-05-30 20:51:06,887 P3357 INFO [Metrics] logloss: 0.441799 - AUC: 0.809723
2020-05-30 20:51:06,906 P3357 INFO Save best model: monitor(max): 0.367924
2020-05-30 20:51:08,748 P3357 INFO --- 7335/7335 batches finished ---
2020-05-30 20:51:08,814 P3357 INFO Train loss: 0.454295
2020-05-30 20:51:08,814 P3357 INFO ************ Epoch=8 end ************
2020-05-30 22:16:13,948 P3357 INFO [Metrics] logloss: 0.441767 - AUC: 0.809769
2020-05-30 22:16:13,954 P3357 INFO Save best model: monitor(max): 0.368003
2020-05-30 22:16:15,807 P3357 INFO --- 7335/7335 batches finished ---
2020-05-30 22:16:15,869 P3357 INFO Train loss: 0.454183
2020-05-30 22:16:15,869 P3357 INFO ************ Epoch=9 end ************
2020-05-30 23:41:20,621 P3357 INFO [Metrics] logloss: 0.441543 - AUC: 0.810019
2020-05-30 23:41:20,630 P3357 INFO Save best model: monitor(max): 0.368476
2020-05-30 23:41:22,455 P3357 INFO --- 7335/7335 batches finished ---
2020-05-30 23:41:22,529 P3357 INFO Train loss: 0.454097
2020-05-30 23:41:22,529 P3357 INFO ************ Epoch=10 end ************
2020-05-31 01:06:38,054 P3357 INFO [Metrics] logloss: 0.441706 - AUC: 0.809786
2020-05-31 01:06:38,063 P3357 INFO Monitor(max) STOP: 0.368080 !
2020-05-31 01:06:38,063 P3357 INFO Reduce learning rate on plateau: 0.000100
2020-05-31 01:06:38,064 P3357 INFO --- 7335/7335 batches finished ---
2020-05-31 01:06:38,136 P3357 INFO Train loss: 0.454054
2020-05-31 01:06:38,136 P3357 INFO ************ Epoch=11 end ************
2020-05-31 02:31:48,554 P3357 INFO [Metrics] logloss: 0.438655 - AUC: 0.813246
2020-05-31 02:31:48,555 P3357 INFO Save best model: monitor(max): 0.374591
2020-05-31 02:31:50,386 P3357 INFO --- 7335/7335 batches finished ---
2020-05-31 02:31:50,442 P3357 INFO Train loss: 0.439997
2020-05-31 02:31:50,442 P3357 INFO ************ Epoch=12 end ************
2020-05-31 03:56:55,735 P3357 INFO [Metrics] logloss: 0.438380 - AUC: 0.813590
2020-05-31 03:56:55,741 P3357 INFO Save best model: monitor(max): 0.375210
2020-05-31 03:56:57,570 P3357 INFO --- 7335/7335 batches finished ---
2020-05-31 03:56:57,647 P3357 INFO Train loss: 0.435671
2020-05-31 03:56:57,647 P3357 INFO ************ Epoch=13 end ************
2020-05-31 05:22:03,852 P3357 INFO [Metrics] logloss: 0.438612 - AUC: 0.813537
2020-05-31 05:22:03,858 P3357 INFO Monitor(max) STOP: 0.374925 !
2020-05-31 05:22:03,858 P3357 INFO Reduce learning rate on plateau: 0.000010
2020-05-31 05:22:03,858 P3357 INFO --- 7335/7335 batches finished ---
2020-05-31 05:22:03,914 P3357 INFO Train loss: 0.433661
2020-05-31 05:22:03,915 P3357 INFO ************ Epoch=14 end ************
2020-05-31 06:47:09,721 P3357 INFO [Metrics] logloss: 0.439795 - AUC: 0.812721
2020-05-31 06:47:09,735 P3357 INFO Monitor(max) STOP: 0.372926 !
2020-05-31 06:47:09,735 P3357 INFO Reduce learning rate on plateau: 0.000001
2020-05-31 06:47:09,735 P3357 INFO Early stopping at epoch=15
2020-05-31 06:47:09,735 P3357 INFO --- 7335/7335 batches finished ---
2020-05-31 06:47:09,797 P3357 INFO Train loss: 0.427705
2020-05-31 06:47:09,797 P3357 INFO Training finished.
2020-05-31 06:47:09,797 P3357 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/FiGNN_criteo/criteo_x4_001_be98441d/FiGNN_criteo_x4_005_f8fc849a_model.ckpt
2020-05-31 06:47:11,512 P3357 INFO ****** Train/validation evaluation ******
2020-05-31 06:56:51,442 P3357 INFO [Metrics] logloss: 0.424182 - AUC: 0.828340
2020-05-31 06:58:03,224 P3357 INFO [Metrics] logloss: 0.438380 - AUC: 0.813590
2020-05-31 06:58:03,379 P3357 INFO ******** Test evaluation ********
2020-05-31 06:58:03,379 P3357 INFO Loading data...
2020-05-31 06:58:03,380 P3357 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-05-31 06:58:04,478 P3357 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-05-31 06:58:04,478 P3357 INFO Loading test data done.
2020-05-31 06:59:16,326 P3357 INFO [Metrics] logloss: 0.437928 - AUC: 0.814073

```
