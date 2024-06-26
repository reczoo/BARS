## xDeepFM_criteo_x4_001

A hands-on guide to run the xDeepFM model on the Criteo_x4_001 dataset.

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
    cd xDeepFM_criteo_x4_001
    nohup python run_expid.py --config ./xDeepFM_criteo_x4_tuner_config_03 --expid xDeepFM_criteo_x4_001_d509bfbe --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.437602 | 0.814331  |


### Logs
```python
2020-07-23 10:09:55,049 P46308 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "cin_layer_units": "[64, 64, 64]",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "dnn_hidden_units": "[1000, 1000, 1000, 1000, 1000]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.0001",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "hidden_activations": "relu",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "xDeepFM",
    "model_id": "xDeepFM_criteo_x4_5c863b0f_001_109e6738",
    "model_root": "./Criteo/xDeepFM_criteo/min10/",
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
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-23 10:09:55,050 P46308 INFO Set up feature encoder...
2020-07-23 10:09:55,050 P46308 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-07-23 10:09:55,050 P46308 INFO Loading data...
2020-07-23 10:09:55,052 P46308 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-07-23 10:10:00,783 P46308 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-07-23 10:10:03,326 P46308 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-07-23 10:10:03,585 P46308 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-23 10:10:03,585 P46308 INFO Loading train data done.
2020-07-23 10:10:06,726 P46308 INFO **** Start training: 3668 batches/epoch ****
2020-07-23 10:41:53,709 P46308 INFO [Metrics] logloss: 0.449413 - AUC: 0.801274
2020-07-23 10:41:53,714 P46308 INFO Save best model: monitor(max): 0.351861
2020-07-23 10:41:53,794 P46308 INFO --- 3668/3668 batches finished ---
2020-07-23 10:41:53,838 P46308 INFO Train loss: 0.469505
2020-07-23 10:41:53,838 P46308 INFO ************ Epoch=1 end ************
2020-07-23 11:13:42,453 P46308 INFO [Metrics] logloss: 0.447325 - AUC: 0.803567
2020-07-23 11:13:42,454 P46308 INFO Save best model: monitor(max): 0.356242
2020-07-23 11:13:42,600 P46308 INFO --- 3668/3668 batches finished ---
2020-07-23 11:13:42,667 P46308 INFO Train loss: 0.464738
2020-07-23 11:13:42,667 P46308 INFO ************ Epoch=2 end ************
2020-07-23 11:45:17,699 P46308 INFO [Metrics] logloss: 0.446611 - AUC: 0.804425
2020-07-23 11:45:17,700 P46308 INFO Save best model: monitor(max): 0.357815
2020-07-23 11:45:17,832 P46308 INFO --- 3668/3668 batches finished ---
2020-07-23 11:45:17,879 P46308 INFO Train loss: 0.463496
2020-07-23 11:45:17,879 P46308 INFO ************ Epoch=3 end ************
2020-07-23 12:16:56,917 P46308 INFO [Metrics] logloss: 0.445949 - AUC: 0.805071
2020-07-23 12:16:56,919 P46308 INFO Save best model: monitor(max): 0.359122
2020-07-23 12:16:57,068 P46308 INFO --- 3668/3668 batches finished ---
2020-07-23 12:16:57,120 P46308 INFO Train loss: 0.462881
2020-07-23 12:16:57,120 P46308 INFO ************ Epoch=4 end ************
2020-07-23 12:48:44,272 P46308 INFO [Metrics] logloss: 0.445947 - AUC: 0.805224
2020-07-23 12:48:44,273 P46308 INFO Save best model: monitor(max): 0.359277
2020-07-23 12:48:44,402 P46308 INFO --- 3668/3668 batches finished ---
2020-07-23 12:48:44,449 P46308 INFO Train loss: 0.462495
2020-07-23 12:48:44,449 P46308 INFO ************ Epoch=5 end ************
2020-07-23 13:20:32,975 P46308 INFO [Metrics] logloss: 0.445809 - AUC: 0.805316
2020-07-23 13:20:32,976 P46308 INFO Save best model: monitor(max): 0.359506
2020-07-23 13:20:33,095 P46308 INFO --- 3668/3668 batches finished ---
2020-07-23 13:20:33,144 P46308 INFO Train loss: 0.462169
2020-07-23 13:20:33,144 P46308 INFO ************ Epoch=6 end ************
2020-07-23 13:52:19,314 P46308 INFO [Metrics] logloss: 0.445125 - AUC: 0.806012
2020-07-23 13:52:19,315 P46308 INFO Save best model: monitor(max): 0.360887
2020-07-23 13:52:19,432 P46308 INFO --- 3668/3668 batches finished ---
2020-07-23 13:52:19,479 P46308 INFO Train loss: 0.461898
2020-07-23 13:52:19,479 P46308 INFO ************ Epoch=7 end ************
2020-07-23 14:24:05,176 P46308 INFO [Metrics] logloss: 0.445543 - AUC: 0.805697
2020-07-23 14:24:05,177 P46308 INFO Monitor(max) STOP: 0.360154 !
2020-07-23 14:24:05,177 P46308 INFO Reduce learning rate on plateau: 0.000100
2020-07-23 14:24:05,177 P46308 INFO --- 3668/3668 batches finished ---
2020-07-23 14:24:05,225 P46308 INFO Train loss: 0.461684
2020-07-23 14:24:05,225 P46308 INFO ************ Epoch=8 end ************
2020-07-23 14:55:50,775 P46308 INFO [Metrics] logloss: 0.439846 - AUC: 0.811851
2020-07-23 14:55:50,776 P46308 INFO Save best model: monitor(max): 0.372006
2020-07-23 14:55:50,910 P46308 INFO --- 3668/3668 batches finished ---
2020-07-23 14:55:50,957 P46308 INFO Train loss: 0.446375
2020-07-23 14:55:50,957 P46308 INFO ************ Epoch=9 end ************
2020-07-23 15:27:35,105 P46308 INFO [Metrics] logloss: 0.438979 - AUC: 0.812797
2020-07-23 15:27:35,106 P46308 INFO Save best model: monitor(max): 0.373817
2020-07-23 15:27:35,219 P46308 INFO --- 3668/3668 batches finished ---
2020-07-23 15:27:35,266 P46308 INFO Train loss: 0.441690
2020-07-23 15:27:35,266 P46308 INFO ************ Epoch=10 end ************
2020-07-23 15:59:19,614 P46308 INFO [Metrics] logloss: 0.438631 - AUC: 0.813197
2020-07-23 15:59:19,616 P46308 INFO Save best model: monitor(max): 0.374566
2020-07-23 15:59:19,736 P46308 INFO --- 3668/3668 batches finished ---
2020-07-23 15:59:19,783 P46308 INFO Train loss: 0.440110
2020-07-23 15:59:19,783 P46308 INFO ************ Epoch=11 end ************
2020-07-23 16:31:05,993 P46308 INFO [Metrics] logloss: 0.438444 - AUC: 0.813406
2020-07-23 16:31:05,994 P46308 INFO Save best model: monitor(max): 0.374962
2020-07-23 16:31:06,125 P46308 INFO --- 3668/3668 batches finished ---
2020-07-23 16:31:06,172 P46308 INFO Train loss: 0.439075
2020-07-23 16:31:06,172 P46308 INFO ************ Epoch=12 end ************
2020-07-23 17:02:48,335 P46308 INFO [Metrics] logloss: 0.438449 - AUC: 0.813403
2020-07-23 17:02:48,336 P46308 INFO Monitor(max) STOP: 0.374954 !
2020-07-23 17:02:48,336 P46308 INFO Reduce learning rate on plateau: 0.000010
2020-07-23 17:02:48,336 P46308 INFO --- 3668/3668 batches finished ---
2020-07-23 17:02:48,382 P46308 INFO Train loss: 0.438304
2020-07-23 17:02:48,382 P46308 INFO ************ Epoch=13 end ************
2020-07-23 17:34:28,496 P46308 INFO [Metrics] logloss: 0.438136 - AUC: 0.813737
2020-07-23 17:34:28,497 P46308 INFO Save best model: monitor(max): 0.375600
2020-07-23 17:34:28,628 P46308 INFO --- 3668/3668 batches finished ---
2020-07-23 17:34:28,675 P46308 INFO Train loss: 0.433134
2020-07-23 17:34:28,675 P46308 INFO ************ Epoch=14 end ************
2020-07-23 18:06:09,267 P46308 INFO [Metrics] logloss: 0.438161 - AUC: 0.813744
2020-07-23 18:06:09,267 P46308 INFO Monitor(max) STOP: 0.375583 !
2020-07-23 18:06:09,268 P46308 INFO Reduce learning rate on plateau: 0.000001
2020-07-23 18:06:09,268 P46308 INFO --- 3668/3668 batches finished ---
2020-07-23 18:06:09,318 P46308 INFO Train loss: 0.432006
2020-07-23 18:06:09,318 P46308 INFO ************ Epoch=15 end ************
2020-07-23 18:37:42,694 P46308 INFO [Metrics] logloss: 0.438209 - AUC: 0.813711
2020-07-23 18:37:42,695 P46308 INFO Monitor(max) STOP: 0.375502 !
2020-07-23 18:37:42,695 P46308 INFO Reduce learning rate on plateau: 0.000001
2020-07-23 18:37:42,695 P46308 INFO Early stopping at epoch=16
2020-07-23 18:37:42,696 P46308 INFO --- 3668/3668 batches finished ---
2020-07-23 18:37:42,745 P46308 INFO Train loss: 0.430824
2020-07-23 18:37:42,745 P46308 INFO Training finished.
2020-07-23 18:37:42,745 P46308 INFO Load best model: /home/XXX/benchmarks/Criteo/xDeepFM_criteo/min10/criteo_x4_5c863b0f/xDeepFM_criteo_x4_5c863b0f_001_109e6738_model.ckpt
2020-07-23 18:37:42,852 P46308 INFO ****** Train/validation evaluation ******
2020-07-23 18:38:24,166 P46308 INFO [Metrics] logloss: 0.438136 - AUC: 0.813737
2020-07-23 18:38:24,259 P46308 INFO ******** Test evaluation ********
2020-07-23 18:38:24,260 P46308 INFO Loading data...
2020-07-23 18:38:24,260 P46308 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-07-23 18:38:25,027 P46308 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-23 18:38:25,027 P46308 INFO Loading test data done.
2020-07-23 18:39:07,011 P46308 INFO [Metrics] logloss: 0.437602 - AUC: 0.814331

```
