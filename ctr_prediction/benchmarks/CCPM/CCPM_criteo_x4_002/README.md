## CCPM_criteo_x4_002

A hands-on guide to run the CCPM model on the Criteo_x4_002 dataset.

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

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [CCPM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/CCPM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CCPM_criteo_x4_tuner_config_04](./CCPM_criteo_x4_tuner_config_04). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CCPM_criteo_x4_002
    nohup python run_expid.py --config ./CCPM_criteo_x4_tuner_config_04 --expid CCPM_criteo_x4_011_3f46109b --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.443958 | 0.807705  |


### Logs
```python
2020-06-01 06:58:00,613 P2721 INFO {
    "activation": "Tanh",
    "batch_size": "5000",
    "channels": "[64, 128, 256]",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_001_be98441d",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-08",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "kernel_heights": "[7, 7, 7]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "CCPM",
    "model_id": "CCPM_criteo_x4_011_dd841ea9",
    "model_root": "./Criteo/CCPM_criteo/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/criteo_x4_001_be98441d/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_be98441d/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_001_be98441d/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-01 06:58:00,614 P2721 INFO Set up feature encoder...
2020-06-01 06:58:00,614 P2721 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-06-01 06:58:00,614 P2721 INFO Loading data...
2020-06-01 06:58:00,616 P2721 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-06-01 06:58:05,253 P2721 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-06-01 06:58:07,272 P2721 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-06-01 06:58:07,482 P2721 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-01 06:58:07,482 P2721 INFO Loading train data done.
2020-06-01 06:58:15,449 P2721 INFO **** Start training: 7335 batches/epoch ****
2020-06-01 12:28:26,744 P2721 INFO [Metrics] logloss: 0.444192 - AUC: 0.807421
2020-06-01 12:28:26,745 P2721 INFO Save best model: monitor(max): 0.363229
2020-06-01 12:28:28,275 P2721 INFO --- 7335/7335 batches finished ---
2020-06-01 12:28:28,361 P2721 INFO Train loss: 0.451764
2020-06-01 12:28:28,361 P2721 INFO ************ Epoch=1 end ************
2020-06-01 17:53:56,079 P2721 INFO [Metrics] logloss: 0.447041 - AUC: 0.805364
2020-06-01 17:53:56,080 P2721 INFO Monitor(max) STOP: 0.358323 !
2020-06-01 17:53:56,080 P2721 INFO Reduce learning rate on plateau: 0.000100
2020-06-01 17:53:56,080 P2721 INFO --- 7335/7335 batches finished ---
2020-06-01 17:53:56,143 P2721 INFO Train loss: 0.427768
2020-06-01 17:53:56,143 P2721 INFO ************ Epoch=2 end ************
2020-06-01 23:20:11,399 P2721 INFO [Metrics] logloss: 0.474700 - AUC: 0.789096
2020-06-01 23:20:11,400 P2721 INFO Monitor(max) STOP: 0.314395 !
2020-06-01 23:20:11,400 P2721 INFO Reduce learning rate on plateau: 0.000010
2020-06-01 23:20:11,400 P2721 INFO Early stopping at epoch=3
2020-06-01 23:20:11,400 P2721 INFO --- 7335/7335 batches finished ---
2020-06-01 23:20:11,465 P2721 INFO Train loss: 0.378270
2020-06-01 23:20:11,466 P2721 INFO Training finished.
2020-06-01 23:20:11,466 P2721 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/CCPM_criteo/criteo_x4_001_be98441d/CCPM_criteo_x4_011_dd841ea9_model.ckpt
2020-06-01 23:20:12,492 P2721 INFO ****** Train/validation evaluation ******
2020-06-02 02:49:49,827 P2721 INFO [Metrics] logloss: 0.424894 - AUC: 0.827891
2020-06-02 03:16:01,100 P2721 INFO [Metrics] logloss: 0.444192 - AUC: 0.807421
2020-06-02 03:16:01,251 P2721 INFO ******** Test evaluation ********
2020-06-02 03:16:01,251 P2721 INFO Loading data...
2020-06-02 03:16:01,251 P2721 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-06-02 03:16:01,993 P2721 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-02 03:16:01,993 P2721 INFO Loading test data done.
2020-06-02 03:42:14,575 P2721 INFO [Metrics] logloss: 0.443958 - AUC: 0.807705

```
