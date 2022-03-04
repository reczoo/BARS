## CIN_criteo_x4_001

A hands-on guide to run the xDeepFM model on the Criteo_x4_001 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [xDeepFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/xDeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CIN_criteo_x4_tuner_config_01](./CIN_criteo_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CIN_criteo_x4_001
    nohup python run_expid.py --config ./CIN_criteo_x4_tuner_config_01 --expid xDeepFM_criteo_x4_007_f93ff06e --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.439382 | 0.812719  |


### Logs
```python
2020-07-13 06:50:58,601 P2861 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "cin_layer_units": "[64, 64, 64, 64]",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "dnn_hidden_units": "[]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "hidden_activations": "relu",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "xDeepFM",
    "model_id": "xDeepFM_criteo_x4_5c863b0f_007_52d2bf35",
    "model_root": "./Criteo/CIN_criteo/min10/",
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
    "test_data": "../data/Criteo/criteo_x4_5c863b0f/test.h5",
    "train_data": "../data/Criteo/criteo_x4_5c863b0f/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-07-13 06:50:58,603 P2861 INFO Set up feature encoder...
2020-07-13 06:50:58,603 P2861 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-07-13 06:50:58,604 P2861 INFO Loading data...
2020-07-13 06:50:58,608 P2861 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-07-13 06:51:05,089 P2861 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-07-13 06:51:07,342 P2861 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-07-13 06:51:07,466 P2861 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-13 06:51:07,466 P2861 INFO Loading train data done.
2020-07-13 07:20:37,209 P2861 INFO Start training: 3668 batches/epoch
2020-07-13 07:20:37,210 P2861 INFO ************ Epoch=1 start ************
2020-07-13 07:46:50,215 P2861 INFO [Metrics] logloss: 0.449751 - AUC: 0.801592
2020-07-13 07:46:50,217 P2861 INFO Save best model: monitor(max): 0.351841
2020-07-13 07:46:50,351 P2861 INFO --- 3668/3668 batches finished ---
2020-07-13 07:46:50,406 P2861 INFO Train loss: 0.462524
2020-07-13 07:46:50,407 P2861 INFO ************ Epoch=1 end ************
2020-07-13 08:08:27,159 P2861 INFO [Metrics] logloss: 0.445865 - AUC: 0.805937
2020-07-13 08:08:27,160 P2861 INFO Save best model: monitor(max): 0.360072
2020-07-13 08:08:27,254 P2861 INFO --- 3668/3668 batches finished ---
2020-07-13 08:08:27,308 P2861 INFO Train loss: 0.456292
2020-07-13 08:08:27,308 P2861 INFO ************ Epoch=2 end ************
2020-07-13 08:29:44,442 P2861 INFO [Metrics] logloss: 0.443937 - AUC: 0.807481
2020-07-13 08:29:44,443 P2861 INFO Save best model: monitor(max): 0.363543
2020-07-13 08:29:44,558 P2861 INFO --- 3668/3668 batches finished ---
2020-07-13 08:29:44,614 P2861 INFO Train loss: 0.454156
2020-07-13 08:29:44,615 P2861 INFO ************ Epoch=3 end ************
2020-07-13 08:51:39,215 P2861 INFO [Metrics] logloss: 0.443192 - AUC: 0.808313
2020-07-13 08:51:39,216 P2861 INFO Save best model: monitor(max): 0.365121
2020-07-13 08:51:39,308 P2861 INFO --- 3668/3668 batches finished ---
2020-07-13 08:51:39,363 P2861 INFO Train loss: 0.453154
2020-07-13 08:51:39,363 P2861 INFO ************ Epoch=4 end ************
2020-07-13 09:12:48,009 P2861 INFO [Metrics] logloss: 0.442828 - AUC: 0.808660
2020-07-13 09:12:48,010 P2861 INFO Save best model: monitor(max): 0.365832
2020-07-13 09:12:48,088 P2861 INFO --- 3668/3668 batches finished ---
2020-07-13 09:12:48,145 P2861 INFO Train loss: 0.452577
2020-07-13 09:12:48,145 P2861 INFO ************ Epoch=5 end ************
2020-07-13 09:34:03,310 P2861 INFO [Metrics] logloss: 0.442679 - AUC: 0.809022
2020-07-13 09:34:03,311 P2861 INFO Save best model: monitor(max): 0.366342
2020-07-13 09:34:03,395 P2861 INFO --- 3668/3668 batches finished ---
2020-07-13 09:34:03,454 P2861 INFO Train loss: 0.452167
2020-07-13 09:34:03,454 P2861 INFO ************ Epoch=6 end ************
2020-07-13 09:55:03,258 P2861 INFO [Metrics] logloss: 0.442309 - AUC: 0.809296
2020-07-13 09:55:03,260 P2861 INFO Save best model: monitor(max): 0.366987
2020-07-13 09:55:03,371 P2861 INFO --- 3668/3668 batches finished ---
2020-07-13 09:55:03,433 P2861 INFO Train loss: 0.451833
2020-07-13 09:55:03,433 P2861 INFO ************ Epoch=7 end ************
2020-07-13 10:15:33,640 P2861 INFO [Metrics] logloss: 0.442350 - AUC: 0.809513
2020-07-13 10:15:33,642 P2861 INFO Save best model: monitor(max): 0.367163
2020-07-13 10:15:33,766 P2861 INFO --- 3668/3668 batches finished ---
2020-07-13 10:15:33,865 P2861 INFO Train loss: 0.451554
2020-07-13 10:15:33,865 P2861 INFO ************ Epoch=8 end ************
2020-07-13 10:36:09,292 P2861 INFO [Metrics] logloss: 0.442208 - AUC: 0.809669
2020-07-13 10:36:09,294 P2861 INFO Save best model: monitor(max): 0.367461
2020-07-13 10:36:09,370 P2861 INFO --- 3668/3668 batches finished ---
2020-07-13 10:36:09,429 P2861 INFO Train loss: 0.451301
2020-07-13 10:36:09,429 P2861 INFO ************ Epoch=9 end ************
2020-07-13 10:57:28,103 P2861 INFO [Metrics] logloss: 0.442226 - AUC: 0.809806
2020-07-13 10:57:28,104 P2861 INFO Save best model: monitor(max): 0.367580
2020-07-13 10:57:28,177 P2861 INFO --- 3668/3668 batches finished ---
2020-07-13 10:57:28,238 P2861 INFO Train loss: 0.451091
2020-07-13 10:57:28,238 P2861 INFO ************ Epoch=10 end ************
2020-07-13 11:18:27,088 P2861 INFO [Metrics] logloss: 0.441804 - AUC: 0.809921
2020-07-13 11:18:27,089 P2861 INFO Save best model: monitor(max): 0.368117
2020-07-13 11:18:27,164 P2861 INFO --- 3668/3668 batches finished ---
2020-07-13 11:18:27,221 P2861 INFO Train loss: 0.450885
2020-07-13 11:18:27,221 P2861 INFO ************ Epoch=11 end ************
2020-07-13 11:38:54,011 P2861 INFO [Metrics] logloss: 0.441904 - AUC: 0.809810
2020-07-13 11:38:54,013 P2861 INFO Monitor(max) STOP: 0.367906 !
2020-07-13 11:38:54,013 P2861 INFO Reduce learning rate on plateau: 0.000100
2020-07-13 11:38:54,013 P2861 INFO --- 3668/3668 batches finished ---
2020-07-13 11:38:54,077 P2861 INFO Train loss: 0.450721
2020-07-13 11:38:54,077 P2861 INFO ************ Epoch=12 end ************
2020-07-13 11:59:26,045 P2861 INFO [Metrics] logloss: 0.439707 - AUC: 0.812359
2020-07-13 11:59:26,046 P2861 INFO Save best model: monitor(max): 0.372652
2020-07-13 11:59:26,121 P2861 INFO --- 3668/3668 batches finished ---
2020-07-13 11:59:26,183 P2861 INFO Train loss: 0.439356
2020-07-13 11:59:26,183 P2861 INFO ************ Epoch=13 end ************
2020-07-13 12:19:49,608 P2861 INFO [Metrics] logloss: 0.439849 - AUC: 0.812351
2020-07-13 12:19:49,609 P2861 INFO Monitor(max) STOP: 0.372502 !
2020-07-13 12:19:49,609 P2861 INFO Reduce learning rate on plateau: 0.000010
2020-07-13 12:19:49,609 P2861 INFO --- 3668/3668 batches finished ---
2020-07-13 12:19:49,669 P2861 INFO Train loss: 0.435021
2020-07-13 12:19:49,669 P2861 INFO ************ Epoch=14 end ************
2020-07-13 12:40:14,771 P2861 INFO [Metrics] logloss: 0.440779 - AUC: 0.811742
2020-07-13 12:40:14,772 P2861 INFO Monitor(max) STOP: 0.370964 !
2020-07-13 12:40:14,772 P2861 INFO Reduce learning rate on plateau: 0.000001
2020-07-13 12:40:14,778 P2861 INFO Early stopping at epoch=15
2020-07-13 12:40:14,779 P2861 INFO --- 3668/3668 batches finished ---
2020-07-13 12:40:14,852 P2861 INFO Train loss: 0.430462
2020-07-13 12:40:14,852 P2861 INFO Training finished.
2020-07-13 12:40:14,852 P2861 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/CIN_criteo/min10/criteo_x4_5c863b0f/xDeepFM_criteo_x4_5c863b0f_007_52d2bf35_model.ckpt
2020-07-13 12:40:15,118 P2861 INFO ****** Train/validation evaluation ******
2020-07-13 12:44:00,792 P2861 INFO [Metrics] logloss: 0.426205 - AUC: 0.826317
2020-07-13 12:44:27,269 P2861 INFO [Metrics] logloss: 0.439707 - AUC: 0.812359
2020-07-13 12:44:27,354 P2861 INFO ******** Test evaluation ********
2020-07-13 12:44:27,354 P2861 INFO Loading data...
2020-07-13 12:44:27,354 P2861 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-07-13 12:44:28,334 P2861 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-13 12:44:28,334 P2861 INFO Loading test data done.
2020-07-13 12:44:58,411 P2861 INFO [Metrics] logloss: 0.439382 - AUC: 0.812719

```
