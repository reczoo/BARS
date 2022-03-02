## CIN_kkbox_x1

A hands-on guide to run the xDeepFM model on the Kkbox_x1 dataset.

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
Dataset ID: [Kkbox_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Kkbox/README.md#Kkbox_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [xDeepFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/xDeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CIN_kkbox_x1_tuner_config_02](./CIN_kkbox_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CIN_kkbox_x1
    nohup python run_expid.py --config ./CIN_kkbox_x1_tuner_config_02 --expid xDeepFM_kkbox_x1_022_117199f0 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.490878 | 0.842620  |


### Logs
```python
2020-04-29 05:17:18,308 P22608 INFO {
    "batch_norm": "False",
    "batch_size": "5000",
    "cin_layer_units": "[78, 78]",
    "dataset_id": "kkbox_x1_001_c5c9c6e3",
    "dnn_hidden_units": "[]",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "xDeepFM",
    "model_id": "xDeepFM_kkbox_x1_022_d115d58a",
    "model_root": "./KKBox/CIN_kkbox/",
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
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/KKBox/",
    "test_data": "../data/KKBox/kkbox_x1_001_c5c9c6e3/test.h5",
    "train_data": "../data/KKBox/kkbox_x1_001_c5c9c6e3/train.h5",
    "valid_data": "../data/KKBox/kkbox_x1_001_c5c9c6e3/valid.h5",
    "version": "pytorch",
    "gpu": "0"
}
2020-04-29 05:17:18,309 P22608 INFO Set up feature encoder...
2020-04-29 05:17:18,309 P22608 INFO Load feature_map from json: ../data/KKBox/kkbox_x1_001_c5c9c6e3/feature_map.json
2020-04-29 05:17:18,309 P22608 INFO Loading data...
2020-04-29 05:17:18,311 P22608 INFO Loading data from h5: ../data/KKBox/kkbox_x1_001_c5c9c6e3/train.h5
2020-04-29 05:17:18,598 P22608 INFO Loading data from h5: ../data/KKBox/kkbox_x1_001_c5c9c6e3/valid.h5
2020-04-29 05:17:18,790 P22608 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2020-04-29 05:17:18,810 P22608 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-04-29 05:17:18,810 P22608 INFO Loading train data done.
2020-04-29 05:17:22,691 P22608 INFO **** Start training: 1181 batches/epoch ****
2020-04-29 05:26:56,265 P22608 INFO [Metrics] logloss: 0.570631 - AUC: 0.781371
2020-04-29 05:26:56,281 P22608 INFO Save best model: monitor(max): 0.210740
2020-04-29 05:26:56,324 P22608 INFO --- 1181/1181 batches finished ---
2020-04-29 05:26:56,362 P22608 INFO Train loss: 0.592051
2020-04-29 05:26:56,363 P22608 INFO ************ Epoch=1 end ************
2020-04-29 05:36:27,847 P22608 INFO [Metrics] logloss: 0.528563 - AUC: 0.812455
2020-04-29 05:36:27,862 P22608 INFO Save best model: monitor(max): 0.283893
2020-04-29 05:36:27,920 P22608 INFO --- 1181/1181 batches finished ---
2020-04-29 05:36:27,989 P22608 INFO Train loss: 0.558708
2020-04-29 05:36:27,990 P22608 INFO ************ Epoch=2 end ************
2020-04-29 05:45:59,970 P22608 INFO [Metrics] logloss: 0.500528 - AUC: 0.834321
2020-04-29 05:45:59,984 P22608 INFO Save best model: monitor(max): 0.333793
2020-04-29 05:46:00,051 P22608 INFO --- 1181/1181 batches finished ---
2020-04-29 05:46:00,122 P22608 INFO Train loss: 0.531992
2020-04-29 05:46:00,123 P22608 INFO ************ Epoch=3 end ************
2020-04-29 05:55:33,363 P22608 INFO [Metrics] logloss: 0.490362 - AUC: 0.842968
2020-04-29 05:55:33,381 P22608 INFO Save best model: monitor(max): 0.352606
2020-04-29 05:55:33,443 P22608 INFO --- 1181/1181 batches finished ---
2020-04-29 05:55:33,493 P22608 INFO Train loss: 0.507425
2020-04-29 05:55:33,493 P22608 INFO ************ Epoch=4 end ************
2020-04-29 06:05:06,034 P22608 INFO [Metrics] logloss: 0.500369 - AUC: 0.839405
2020-04-29 06:05:06,060 P22608 INFO Monitor(max) STOP: 0.339036 !
2020-04-29 06:05:06,060 P22608 INFO Reduce learning rate on plateau: 0.000100
2020-04-29 06:05:06,060 P22608 INFO --- 1181/1181 batches finished ---
2020-04-29 06:05:06,105 P22608 INFO Train loss: 0.479181
2020-04-29 06:05:06,106 P22608 INFO ************ Epoch=5 end ************
2020-04-29 06:14:39,634 P22608 INFO [Metrics] logloss: 0.638060 - AUC: 0.821444
2020-04-29 06:14:39,649 P22608 INFO Monitor(max) STOP: 0.183383 !
2020-04-29 06:14:39,649 P22608 INFO Reduce learning rate on plateau: 0.000010
2020-04-29 06:14:39,649 P22608 INFO Early stopping at epoch=6
2020-04-29 06:14:39,649 P22608 INFO --- 1181/1181 batches finished ---
2020-04-29 06:14:39,696 P22608 INFO Train loss: 0.326641
2020-04-29 06:14:39,696 P22608 INFO Training finished.
2020-04-29 06:14:39,697 P22608 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/KKBox/CIN_kkbox/kkbox_x1_001_c5c9c6e3/xDeepFM_kkbox_x1_022_d115d58a_kkbox_x1_001_c5c9c6e3_model.ckpt
2020-04-29 06:14:39,773 P22608 INFO ****** Train/validation evaluation ******
2020-04-29 06:15:58,568 P22608 INFO [Metrics] logloss: 0.393448 - AUC: 0.909427
2020-04-29 06:16:08,778 P22608 INFO [Metrics] logloss: 0.490362 - AUC: 0.842968
2020-04-29 06:16:08,888 P22608 INFO ******** Test evaluation ********
2020-04-29 06:16:08,888 P22608 INFO Loading data...
2020-04-29 06:16:08,888 P22608 INFO Loading data from h5: ../data/KKBox/kkbox_x1_001_c5c9c6e3/test.h5
2020-04-29 06:16:08,949 P22608 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-04-29 06:16:08,949 P22608 INFO Loading test data done.
2020-04-29 06:16:18,845 P22608 INFO [Metrics] logloss: 0.490878 - AUC: 0.842620

```
