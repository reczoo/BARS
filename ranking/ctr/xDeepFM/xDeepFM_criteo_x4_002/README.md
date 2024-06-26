## xDeepFM_criteo_x4_002

A hands-on guide to run the xDeepFM model on the Criteo_x4_002 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

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
Dataset ID: [Criteo_x4](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x4). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [xDeepFM](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/xDeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [xDeepFM_criteo_x4_tuner_config_03](./xDeepFM_criteo_x4_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd xDeepFM_criteo_x4_002
    nohup python run_expid.py --config ./xDeepFM_criteo_x4_tuner_config_03 --expid xDeepFM_criteo_x4_003_c601dd6b --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.437532 | 0.814415  |


### Logs
```python
2020-02-23 04:57:34,710 P12767 INFO {
    "batch_norm": "False",
    "batch_size": "5000",
    "cin_layer_units": "[39, 39, 39, 39]",
    "dataset_id": "criteo_x4_001_be98441d",
    "dnn_hidden_units": "[1000, 1000, 1000, 1000]",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "l2(1.e-5)",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "xDeepFM",
    "model_id": "xDeepFM_criteo_x4_003_364b7497",
    "model_root": "./Criteo/xDeepFM_criteo/",
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
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "test_data": "../data/Criteo/criteo_x4_001_be98441d/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_be98441d/train.h5",
    "valid_data": "../data/Criteo/criteo_x4_001_be98441d/valid.h5",
    "version": "pytorch",
    "gpu": "1"
}
2020-02-23 04:57:34,711 P12767 INFO Set up feature encoder...
2020-02-23 04:57:34,711 P12767 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-02-23 04:57:34,711 P12767 INFO Loading data...
2020-02-23 04:57:34,713 P12767 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-02-23 04:57:38,662 P12767 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-02-23 04:57:40,765 P12767 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-02-23 04:57:40,934 P12767 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-23 04:57:40,934 P12767 INFO Loading train data done.
2020-02-23 04:57:50,456 P12767 INFO **** Start training: 7335 batches/epoch ****
2020-02-23 06:06:22,501 P12767 INFO [Metrics] logloss: 0.445846 - AUC: 0.805487
2020-02-23 06:06:22,612 P12767 INFO Save best model: monitor(max): 0.359640
2020-02-23 06:06:23,512 P12767 INFO --- 7335/7335 batches finished ---
2020-02-23 06:06:23,558 P12767 INFO Train loss: 0.468782
2020-02-23 06:06:23,558 P12767 INFO ************ Epoch=1 end ************
2020-02-23 07:14:53,774 P12767 INFO [Metrics] logloss: 0.443996 - AUC: 0.807363
2020-02-23 07:14:53,900 P12767 INFO Save best model: monitor(max): 0.363367
2020-02-23 07:14:55,323 P12767 INFO --- 7335/7335 batches finished ---
2020-02-23 07:14:55,387 P12767 INFO Train loss: 0.463594
2020-02-23 07:14:55,387 P12767 INFO ************ Epoch=2 end ************
2020-02-23 08:23:25,248 P12767 INFO [Metrics] logloss: 0.443081 - AUC: 0.808419
2020-02-23 08:23:25,360 P12767 INFO Save best model: monitor(max): 0.365338
2020-02-23 08:23:26,768 P12767 INFO --- 7335/7335 batches finished ---
2020-02-23 08:23:26,829 P12767 INFO Train loss: 0.461758
2020-02-23 08:23:26,829 P12767 INFO ************ Epoch=3 end ************
2020-02-23 09:31:57,563 P12767 INFO [Metrics] logloss: 0.442748 - AUC: 0.808755
2020-02-23 09:31:57,850 P12767 INFO Save best model: monitor(max): 0.366007
2020-02-23 09:31:59,304 P12767 INFO --- 7335/7335 batches finished ---
2020-02-23 09:31:59,366 P12767 INFO Train loss: 0.461054
2020-02-23 09:31:59,366 P12767 INFO ************ Epoch=4 end ************
2020-02-23 10:40:31,999 P12767 INFO [Metrics] logloss: 0.442446 - AUC: 0.809009
2020-02-23 10:40:32,165 P12767 INFO Save best model: monitor(max): 0.366564
2020-02-23 10:40:33,609 P12767 INFO --- 7335/7335 batches finished ---
2020-02-23 10:40:33,673 P12767 INFO Train loss: 0.460710
2020-02-23 10:40:33,673 P12767 INFO ************ Epoch=5 end ************
2020-02-23 11:49:04,635 P12767 INFO [Metrics] logloss: 0.442684 - AUC: 0.809005
2020-02-23 11:49:04,749 P12767 INFO Monitor(max) STOP: 0.366321 !
2020-02-23 11:49:04,749 P12767 INFO Reduce learning rate on plateau: 0.000100
2020-02-23 11:49:04,749 P12767 INFO --- 7335/7335 batches finished ---
2020-02-23 11:49:04,834 P12767 INFO Train loss: 0.460522
2020-02-23 11:49:04,834 P12767 INFO ************ Epoch=6 end ************
2020-02-23 12:57:34,660 P12767 INFO [Metrics] logloss: 0.438374 - AUC: 0.813458
2020-02-23 12:57:34,821 P12767 INFO Save best model: monitor(max): 0.375084
2020-02-23 12:57:36,279 P12767 INFO --- 7335/7335 batches finished ---
2020-02-23 12:57:36,344 P12767 INFO Train loss: 0.443042
2020-02-23 12:57:36,344 P12767 INFO ************ Epoch=7 end ************
2020-02-23 14:06:06,279 P12767 INFO [Metrics] logloss: 0.437944 - AUC: 0.813975
2020-02-23 14:06:06,379 P12767 INFO Save best model: monitor(max): 0.376030
2020-02-23 14:06:07,812 P12767 INFO --- 7335/7335 batches finished ---
2020-02-23 14:06:07,877 P12767 INFO Train loss: 0.437806
2020-02-23 14:06:07,877 P12767 INFO ************ Epoch=8 end ************
2020-02-23 15:14:33,397 P12767 INFO [Metrics] logloss: 0.438081 - AUC: 0.813920
2020-02-23 15:14:33,514 P12767 INFO Monitor(max) STOP: 0.375839 !
2020-02-23 15:14:33,514 P12767 INFO Reduce learning rate on plateau: 0.000010
2020-02-23 15:14:33,514 P12767 INFO --- 7335/7335 batches finished ---
2020-02-23 15:14:33,604 P12767 INFO Train loss: 0.435448
2020-02-23 15:14:33,604 P12767 INFO ************ Epoch=9 end ************
2020-02-23 16:23:01,179 P12767 INFO [Metrics] logloss: 0.439213 - AUC: 0.813104
2020-02-23 16:23:01,289 P12767 INFO Monitor(max) STOP: 0.373891 !
2020-02-23 16:23:01,289 P12767 INFO Reduce learning rate on plateau: 0.000001
2020-02-23 16:23:01,289 P12767 INFO Early stopping at epoch=10
2020-02-23 16:23:01,290 P12767 INFO --- 7335/7335 batches finished ---
2020-02-23 16:23:01,358 P12767 INFO Train loss: 0.427982
2020-02-23 16:23:01,358 P12767 INFO Training finished.
2020-02-23 16:23:01,358 P12767 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Criteo/xDeepFM_criteo/criteo_x4_001_be98441d/xDeepFM_criteo_x4_003_364b7497_criteo_x4_001_be98441d_model.ckpt
2020-02-23 16:23:02,629 P12767 INFO ****** Train/validation evaluation ******
2020-02-23 16:34:40,069 P12767 INFO [Metrics] logloss: 0.423823 - AUC: 0.828838
2020-02-23 16:36:07,533 P12767 INFO [Metrics] logloss: 0.437944 - AUC: 0.813975
2020-02-23 16:36:07,754 P12767 INFO ******** Test evaluation ********
2020-02-23 16:36:07,754 P12767 INFO Loading data...
2020-02-23 16:36:07,754 P12767 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-02-23 16:36:08,901 P12767 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-23 16:36:08,901 P12767 INFO Loading test data done.
2020-02-23 16:37:32,921 P12767 INFO [Metrics] logloss: 0.437532 - AUC: 0.814415

```
