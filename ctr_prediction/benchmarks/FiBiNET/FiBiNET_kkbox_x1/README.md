## FiBiNET_kkbox_x1

A hands-on guide to run the FiBiNET model on the KKBox_x1 dataset.

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
Dataset ID: [KKBox_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/KKBox/README.md#KKBox_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FiBiNET](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FiBiNET.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FiBiNET_kkbox_x1_tuner_config_01](./FiBiNET_kkbox_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FiBiNET_kkbox_x1
    nohup python run_expid.py --config ./FiBiNET_kkbox_x1_tuner_config_01 --expid FiBiNET_kkbox_x1_045_c0d62749 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.480985 | 0.851949  |


### Logs
```python
2020-04-21 15:35:43,576 P32287 INFO {
    "batch_norm": "True",
    "batch_size": "10000",
    "bilinear_type": "field_interaction",
    "dataset_id": "kkbox_x1_001_c5c9c6e3",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[500, 500, 500]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FiBiNET",
    "model_id": "FiBiNET_kkbox_x1_045_b7a619a8",
    "model_root": "./KKBox/FiBiNET_kkbox/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "3",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/KKBox/",
    "test_data": "../data/KKBox/kkbox_x1_001_c5c9c6e3/test.h5",
    "train_data": "../data/KKBox/kkbox_x1_001_c5c9c6e3/train.h5",
    "valid_data": "../data/KKBox/kkbox_x1_001_c5c9c6e3/valid.h5",
    "version": "pytorch",
    "gpu": "1"
}
2020-04-21 15:35:43,577 P32287 INFO Set up feature encoder...
2020-04-21 15:35:43,577 P32287 INFO Load feature_map from json: ../data/KKBox/kkbox_x1_001_c5c9c6e3/feature_map.json
2020-04-21 15:35:43,577 P32287 INFO Loading data...
2020-04-21 15:35:43,579 P32287 INFO Loading data from h5: ../data/KKBox/kkbox_x1_001_c5c9c6e3/train.h5
2020-04-21 15:35:49,349 P32287 INFO Loading data from h5: ../data/KKBox/kkbox_x1_001_c5c9c6e3/valid.h5
2020-04-21 15:35:50,407 P32287 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2020-04-21 15:35:50,432 P32287 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-04-21 15:35:50,432 P32287 INFO Loading train data done.
2020-04-21 15:35:59,247 P32287 INFO **** Start training: 591 batches/epoch ****
2020-04-21 15:41:35,172 P32287 INFO [Metrics] logloss: 0.568993 - AUC: 0.791011
2020-04-21 15:41:35,192 P32287 INFO Save best model: monitor(max): 0.222018
2020-04-21 15:41:35,309 P32287 INFO --- 591/591 batches finished ---
2020-04-21 15:41:35,362 P32287 INFO Train loss: 0.627501
2020-04-21 15:41:35,363 P32287 INFO ************ Epoch=1 end ************
2020-04-21 15:47:13,026 P32287 INFO [Metrics] logloss: 0.559805 - AUC: 0.809693
2020-04-21 15:47:13,042 P32287 INFO Save best model: monitor(max): 0.249888
2020-04-21 15:47:13,197 P32287 INFO --- 591/591 batches finished ---
2020-04-21 15:47:13,262 P32287 INFO Train loss: 0.618906
2020-04-21 15:47:13,263 P32287 INFO ************ Epoch=2 end ************
2020-04-21 15:52:49,008 P32287 INFO [Metrics] logloss: 0.530006 - AUC: 0.815781
2020-04-21 15:52:49,021 P32287 INFO Save best model: monitor(max): 0.285776
2020-04-21 15:52:49,186 P32287 INFO --- 591/591 batches finished ---
2020-04-21 15:52:49,246 P32287 INFO Train loss: 0.615401
2020-04-21 15:52:49,247 P32287 INFO ************ Epoch=3 end ************
2020-04-21 15:58:24,700 P32287 INFO [Metrics] logloss: 0.518333 - AUC: 0.819974
2020-04-21 15:58:24,721 P32287 INFO Save best model: monitor(max): 0.301641
2020-04-21 15:58:24,895 P32287 INFO --- 591/591 batches finished ---
2020-04-21 15:58:24,967 P32287 INFO Train loss: 0.612245
2020-04-21 15:58:24,967 P32287 INFO ************ Epoch=4 end ************
2020-04-21 16:03:06,412 P32287 INFO [Metrics] logloss: 0.514964 - AUC: 0.823078
2020-04-21 16:03:06,428 P32287 INFO Save best model: monitor(max): 0.308114
2020-04-21 16:03:07,046 P32287 INFO --- 591/591 batches finished ---
2020-04-21 16:03:07,113 P32287 INFO Train loss: 0.609025
2020-04-21 16:03:07,113 P32287 INFO ************ Epoch=5 end ************
2020-04-21 16:08:45,867 P32287 INFO [Metrics] logloss: 0.517819 - AUC: 0.820944
2020-04-21 16:08:45,878 P32287 INFO Monitor(max) STOP: 0.303125 !
2020-04-21 16:08:45,879 P32287 INFO Reduce learning rate on plateau: 0.000100
2020-04-21 16:08:45,879 P32287 INFO --- 591/591 batches finished ---
2020-04-21 16:08:45,953 P32287 INFO Train loss: 0.606825
2020-04-21 16:08:45,953 P32287 INFO ************ Epoch=6 end ************
2020-04-21 16:14:29,209 P32287 INFO [Metrics] logloss: 0.480030 - AUC: 0.850358
2020-04-21 16:14:29,227 P32287 INFO Save best model: monitor(max): 0.370328
2020-04-21 16:14:29,388 P32287 INFO --- 591/591 batches finished ---
2020-04-21 16:14:29,448 P32287 INFO Train loss: 0.518184
2020-04-21 16:14:29,448 P32287 INFO ************ Epoch=7 end ************
2020-04-21 16:20:04,474 P32287 INFO [Metrics] logloss: 0.480844 - AUC: 0.852028
2020-04-21 16:20:04,491 P32287 INFO Save best model: monitor(max): 0.371184
2020-04-21 16:20:05,301 P32287 INFO --- 591/591 batches finished ---
2020-04-21 16:20:05,359 P32287 INFO Train loss: 0.468223
2020-04-21 16:20:05,359 P32287 INFO ************ Epoch=8 end ************
2020-04-21 16:25:36,236 P32287 INFO [Metrics] logloss: 0.492292 - AUC: 0.848590
2020-04-21 16:25:36,250 P32287 INFO Monitor(max) STOP: 0.356298 !
2020-04-21 16:25:36,251 P32287 INFO Reduce learning rate on plateau: 0.000010
2020-04-21 16:25:36,251 P32287 INFO --- 591/591 batches finished ---
2020-04-21 16:25:36,328 P32287 INFO Train loss: 0.440227
2020-04-21 16:25:36,329 P32287 INFO ************ Epoch=9 end ************
2020-04-21 16:31:09,655 P32287 INFO [Metrics] logloss: 0.539707 - AUC: 0.842771
2020-04-21 16:31:09,678 P32287 INFO Monitor(max) STOP: 0.303063 !
2020-04-21 16:31:09,678 P32287 INFO Reduce learning rate on plateau: 0.000001
2020-04-21 16:31:09,678 P32287 INFO Early stopping at epoch=10
2020-04-21 16:31:09,678 P32287 INFO --- 591/591 batches finished ---
2020-04-21 16:31:09,743 P32287 INFO Train loss: 0.364568
2020-04-21 16:31:09,744 P32287 INFO Training finished.
2020-04-21 16:31:09,744 P32287 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/KKBox/FiBiNET_kkbox/kkbox_x1_001_c5c9c6e3/FiBiNET_kkbox_x1_045_b7a619a8_kkbox_x1_001_c5c9c6e3_model.ckpt
2020-04-21 16:31:10,325 P32287 INFO ****** Train/validation evaluation ******
2020-04-21 16:33:09,864 P32287 INFO [Metrics] logloss: 0.366245 - AUC: 0.919799
2020-04-21 16:33:22,282 P32287 INFO [Metrics] logloss: 0.480844 - AUC: 0.852028
2020-04-21 16:33:22,406 P32287 INFO ******** Test evaluation ********
2020-04-21 16:33:22,406 P32287 INFO Loading data...
2020-04-21 16:33:22,406 P32287 INFO Loading data from h5: ../data/KKBox/kkbox_x1_001_c5c9c6e3/test.h5
2020-04-21 16:33:22,471 P32287 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-04-21 16:33:22,471 P32287 INFO Loading test data done.
2020-04-21 16:33:34,522 P32287 INFO [Metrics] logloss: 0.480985 - AUC: 0.851949

```
