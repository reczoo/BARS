## IPNN_criteo_x4_001

A hands-on guide to run the PNN model on the Criteo_x4_001 dataset.

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

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [PNN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/PNN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [PNN_criteo_x4_tuner_config_02](./PNN_criteo_x4_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd IPNN_criteo_x4_001
    nohup python run_expid.py --config ./PNN_criteo_x4_tuner_config_02 --expid PNN_criteo_x4_002_69a1df28 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.437840 | 0.814212  |


### Logs
```python
2020-06-22 14:09:58,034 P586 INFO {
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
    "hidden_units": "[1000, 1000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "PNN",
    "model_id": "PNN_criteo_x4_5c863b0f_002_5659f418",
    "model_root": "./Criteo/PNN_criteo/min10/",
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
2020-06-22 14:09:58,036 P586 INFO Set up feature encoder...
2020-06-22 14:09:58,036 P586 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-06-22 14:09:58,037 P586 INFO Loading data...
2020-06-22 14:09:58,043 P586 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-06-22 14:10:03,065 P586 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-06-22 14:10:04,784 P586 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-06-22 14:10:04,916 P586 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-22 14:10:04,917 P586 INFO Loading train data done.
2020-06-22 14:10:13,859 P586 INFO Start training: 3668 batches/epoch
2020-06-22 14:10:13,859 P586 INFO ************ Epoch=1 start ************
2020-06-22 14:17:57,766 P586 INFO [Metrics] logloss: 0.444612 - AUC: 0.807052
2020-06-22 14:17:57,770 P586 INFO Save best model: monitor(max): 0.362440
2020-06-22 14:17:58,280 P586 INFO --- 3668/3668 batches finished ---
2020-06-22 14:17:58,326 P586 INFO Train loss: 0.459273
2020-06-22 14:17:58,326 P586 INFO ************ Epoch=1 end ************
2020-06-22 14:25:47,721 P586 INFO [Metrics] logloss: 0.442330 - AUC: 0.809245
2020-06-22 14:25:47,722 P586 INFO Save best model: monitor(max): 0.366916
2020-06-22 14:25:47,816 P586 INFO --- 3668/3668 batches finished ---
2020-06-22 14:25:47,881 P586 INFO Train loss: 0.453597
2020-06-22 14:25:47,881 P586 INFO ************ Epoch=2 end ************
2020-06-22 14:33:33,287 P586 INFO [Metrics] logloss: 0.441757 - AUC: 0.809909
2020-06-22 14:33:33,288 P586 INFO Save best model: monitor(max): 0.368152
2020-06-22 14:33:33,387 P586 INFO --- 3668/3668 batches finished ---
2020-06-22 14:33:33,455 P586 INFO Train loss: 0.451965
2020-06-22 14:33:33,455 P586 INFO ************ Epoch=3 end ************
2020-06-22 14:41:18,671 P586 INFO [Metrics] logloss: 0.441041 - AUC: 0.810586
2020-06-22 14:41:18,673 P586 INFO Save best model: monitor(max): 0.369545
2020-06-22 14:41:18,751 P586 INFO --- 3668/3668 batches finished ---
2020-06-22 14:41:18,815 P586 INFO Train loss: 0.451008
2020-06-22 14:41:18,815 P586 INFO ************ Epoch=4 end ************
2020-06-22 14:49:02,187 P586 INFO [Metrics] logloss: 0.441144 - AUC: 0.810541
2020-06-22 14:49:02,188 P586 INFO Monitor(max) STOP: 0.369396 !
2020-06-22 14:49:02,188 P586 INFO Reduce learning rate on plateau: 0.000100
2020-06-22 14:49:02,188 P586 INFO --- 3668/3668 batches finished ---
2020-06-22 14:49:02,254 P586 INFO Train loss: 0.450321
2020-06-22 14:49:02,254 P586 INFO ************ Epoch=5 end ************
2020-06-22 14:56:41,525 P586 INFO [Metrics] logloss: 0.438305 - AUC: 0.813645
2020-06-22 14:56:41,526 P586 INFO Save best model: monitor(max): 0.375340
2020-06-22 14:56:41,599 P586 INFO --- 3668/3668 batches finished ---
2020-06-22 14:56:41,672 P586 INFO Train loss: 0.438375
2020-06-22 14:56:41,672 P586 INFO ************ Epoch=6 end ************
2020-06-22 15:04:24,600 P586 INFO [Metrics] logloss: 0.438156 - AUC: 0.813853
2020-06-22 15:04:24,602 P586 INFO Save best model: monitor(max): 0.375697
2020-06-22 15:04:24,678 P586 INFO --- 3668/3668 batches finished ---
2020-06-22 15:04:24,766 P586 INFO Train loss: 0.433592
2020-06-22 15:04:24,767 P586 INFO ************ Epoch=7 end ************
2020-06-22 15:12:06,190 P586 INFO [Metrics] logloss: 0.438732 - AUC: 0.813552
2020-06-22 15:12:06,191 P586 INFO Monitor(max) STOP: 0.374821 !
2020-06-22 15:12:06,191 P586 INFO Reduce learning rate on plateau: 0.000010
2020-06-22 15:12:06,191 P586 INFO --- 3668/3668 batches finished ---
2020-06-22 15:12:06,257 P586 INFO Train loss: 0.431038
2020-06-22 15:12:06,258 P586 INFO ************ Epoch=8 end ************
2020-06-22 15:19:51,731 P586 INFO [Metrics] logloss: 0.439788 - AUC: 0.812862
2020-06-22 15:19:51,732 P586 INFO Monitor(max) STOP: 0.373074 !
2020-06-22 15:19:51,732 P586 INFO Reduce learning rate on plateau: 0.000001
2020-06-22 15:19:51,732 P586 INFO Early stopping at epoch=9
2020-06-22 15:19:51,732 P586 INFO --- 3668/3668 batches finished ---
2020-06-22 15:19:51,781 P586 INFO Train loss: 0.425809
2020-06-22 15:19:51,781 P586 INFO Training finished.
2020-06-22 15:19:51,781 P586 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/PNN_criteo/min10/criteo_x4_5c863b0f/PNN_criteo_x4_5c863b0f_002_5659f418_model.ckpt
2020-06-22 15:19:51,894 P586 INFO ****** Train/validation evaluation ******
2020-06-22 15:23:18,029 P586 INFO [Metrics] logloss: 0.420888 - AUC: 0.831734
2020-06-22 15:23:43,911 P586 INFO [Metrics] logloss: 0.438156 - AUC: 0.813853
2020-06-22 15:23:43,986 P586 INFO ******** Test evaluation ********
2020-06-22 15:23:43,986 P586 INFO Loading data...
2020-06-22 15:23:43,986 P586 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-06-22 15:23:44,962 P586 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-22 15:23:44,962 P586 INFO Loading test data done.
2020-06-22 15:24:09,842 P586 INFO [Metrics] logloss: 0.437840 - AUC: 0.814212

```
