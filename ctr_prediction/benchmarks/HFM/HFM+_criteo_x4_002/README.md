## HFM+_criteo_x4_002

A hands-on guide to run the HFM model on the Criteo_x4_002 dataset.

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

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [HFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/HFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HFM+_criteo_x4_tuner_config_08](./HFM+_criteo_x4_tuner_config_08). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HFM+_criteo_x4_002
    nohup python run_expid.py --config ./HFM+_criteo_x4_tuner_config_08 --expid HFM_criteo_x4_001_0d1cce33 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.439100 | 0.812733  |


### Logs
```python
2020-06-08 12:01:49,408 P864 INFO {
    "batch_norm": "False",
    "batch_size": "5000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_001_be98441d",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "5e-07",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000, 1000]",
    "interaction_type": "circular_convolution",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "HFM",
    "model_id": "HFM_criteo_x4_001_bab575ed",
    "model_root": "./Criteo/HFM_criteo/",
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
    "test_data": "../data/Criteo/criteo_x4_001_be98441d/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_be98441d/train.h5",
    "use_dnn": "True",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_001_be98441d/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-08 12:01:49,410 P864 INFO Set up feature encoder...
2020-06-08 12:01:49,411 P864 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-06-08 12:01:49,411 P864 INFO Loading data...
2020-06-08 12:01:49,422 P864 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-06-08 12:01:54,411 P864 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-06-08 12:01:56,482 P864 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-06-08 12:01:56,607 P864 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-08 12:01:56,607 P864 INFO Loading train data done.
2020-06-08 12:02:13,745 P864 INFO **** Start training: 7335 batches/epoch ****
2020-06-08 13:02:15,146 P864 INFO [Metrics] logloss: 0.441174 - AUC: 0.810458
2020-06-08 13:02:15,147 P864 INFO Save best model: monitor(max): 0.369283
2020-06-08 13:02:16,958 P864 INFO --- 7335/7335 batches finished ---
2020-06-08 13:02:17,009 P864 INFO Train loss: 0.453844
2020-06-08 13:02:17,009 P864 INFO ************ Epoch=1 end ************
2020-06-08 14:02:16,876 P864 INFO [Metrics] logloss: 0.439476 - AUC: 0.812352
2020-06-08 14:02:16,878 P864 INFO Save best model: monitor(max): 0.372876
2020-06-08 14:02:18,980 P864 INFO --- 7335/7335 batches finished ---
2020-06-08 14:02:19,042 P864 INFO Train loss: 0.446491
2020-06-08 14:02:19,042 P864 INFO ************ Epoch=2 end ************
2020-06-08 15:02:14,854 P864 INFO [Metrics] logloss: 0.440138 - AUC: 0.811659
2020-06-08 15:02:14,856 P864 INFO Monitor(max) STOP: 0.371520 !
2020-06-08 15:02:14,856 P864 INFO Reduce learning rate on plateau: 0.000100
2020-06-08 15:02:14,856 P864 INFO --- 7335/7335 batches finished ---
2020-06-08 15:02:14,904 P864 INFO Train loss: 0.442799
2020-06-08 15:02:14,905 P864 INFO ************ Epoch=3 end ************
2020-06-08 16:02:13,211 P864 INFO [Metrics] logloss: 0.467184 - AUC: 0.795545
2020-06-08 16:02:13,213 P864 INFO Monitor(max) STOP: 0.328361 !
2020-06-08 16:02:13,213 P864 INFO Reduce learning rate on plateau: 0.000010
2020-06-08 16:02:13,213 P864 INFO Early stopping at epoch=4
2020-06-08 16:02:13,213 P864 INFO --- 7335/7335 batches finished ---
2020-06-08 16:02:13,262 P864 INFO Train loss: 0.400821
2020-06-08 16:02:13,262 P864 INFO Training finished.
2020-06-08 16:02:13,262 P864 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/HFM_criteo/criteo_x4_001_be98441d/HFM_criteo_x4_001_bab575ed_model.ckpt
2020-06-08 16:02:15,155 P864 INFO ****** Train/validation evaluation ******
2020-06-08 16:20:48,766 P864 INFO [Metrics] logloss: 0.421891 - AUC: 0.831043
2020-06-08 16:23:07,148 P864 INFO [Metrics] logloss: 0.439476 - AUC: 0.812352
2020-06-08 16:23:07,261 P864 INFO ******** Test evaluation ********
2020-06-08 16:23:07,261 P864 INFO Loading data...
2020-06-08 16:23:07,261 P864 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-06-08 16:23:08,361 P864 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-08 16:23:08,361 P864 INFO Loading test data done.
2020-06-08 16:25:26,862 P864 INFO [Metrics] logloss: 0.439100 - AUC: 0.812733

```
