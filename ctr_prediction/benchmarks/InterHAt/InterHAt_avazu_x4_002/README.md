## InterHAt_avazu_x4_002

A hands-on guide to run the InterHAt model on the Avazu_x4_002 dataset.

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
Dataset ID: [Avazu_x4_002](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x4_002). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [InterHAt](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/InterHAt.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [InterHAt_avazu_x4_tuner_config_05](./InterHAt_avazu_x4_tuner_config_05). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd InterHAt_avazu_x4_002
    nohup python run_expid.py --config ./InterHAt_avazu_x4_tuner_config_05 --expid InterHAt_avazu_x4_009_34e90a84 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.372247 | 0.792731  |


### Logs
```python
2020-06-05 21:27:51,037 P32957 INFO {
    "attention_dim": "40",
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "embedding_dim": "40",
    "embedding_regularizer": "1e-08",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_dim": "500",
    "hidden_units": "[]",
    "layer_norm": "True",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "InterHAt",
    "model_id": "InterHAt_avazu_x4_001_d45ad60e_009_bfbac078",
    "model_root": "./Avazu/InterHAt_avazu/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "1",
    "optimizer": "adam",
    "order": "4",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "use_hdf5": "True",
    "use_residual": "True",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-06-05 21:27:51,038 P32957 INFO Set up feature encoder...
2020-06-05 21:27:51,038 P32957 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-06-05 21:27:51,038 P32957 INFO Loading data...
2020-06-05 21:27:51,040 P32957 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-06-05 21:27:53,724 P32957 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-06-05 21:27:55,056 P32957 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-05 21:27:55,170 P32957 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-05 21:27:55,170 P32957 INFO Loading train data done.
2020-06-05 21:28:05,958 P32957 INFO **** Start training: 3235 batches/epoch ****
2020-06-05 21:46:11,924 P32957 INFO [Metrics] logloss: 0.372400 - AUC: 0.792452
2020-06-05 21:46:11,927 P32957 INFO Save best model: monitor(max): 0.420051
2020-06-05 21:46:13,354 P32957 INFO --- 3235/3235 batches finished ---
2020-06-05 21:46:13,389 P32957 INFO Train loss: 0.382025
2020-06-05 21:46:13,389 P32957 INFO ************ Epoch=1 end ************
2020-06-05 22:04:19,873 P32957 INFO [Metrics] logloss: 0.421522 - AUC: 0.764044
2020-06-05 22:04:19,876 P32957 INFO Monitor(max) STOP: 0.342522 !
2020-06-05 22:04:19,876 P32957 INFO Reduce learning rate on plateau: 0.000100
2020-06-05 22:04:19,876 P32957 INFO --- 3235/3235 batches finished ---
2020-06-05 22:04:19,912 P32957 INFO Train loss: 0.288440
2020-06-05 22:04:19,912 P32957 INFO ************ Epoch=2 end ************
2020-06-05 22:22:25,507 P32957 INFO [Metrics] logloss: 0.577693 - AUC: 0.727019
2020-06-05 22:22:25,510 P32957 INFO Monitor(max) STOP: 0.149325 !
2020-06-05 22:22:25,510 P32957 INFO Reduce learning rate on plateau: 0.000010
2020-06-05 22:22:25,510 P32957 INFO Early stopping at epoch=3
2020-06-05 22:22:25,511 P32957 INFO --- 3235/3235 batches finished ---
2020-06-05 22:22:25,543 P32957 INFO Train loss: 0.252509
2020-06-05 22:22:25,544 P32957 INFO Training finished.
2020-06-05 22:22:25,544 P32957 INFO Load best model: /home/XXX/benchmarks/Avazu/InterHAt_avazu/avazu_x4_001_d45ad60e/InterHAt_avazu_x4_001_d45ad60e_009_bfbac078_model.ckpt
2020-06-05 22:22:28,999 P32957 INFO ****** Train/validation evaluation ******
2020-06-05 22:23:04,710 P32957 INFO [Metrics] logloss: 0.372400 - AUC: 0.792452
2020-06-05 22:23:04,842 P32957 INFO ******** Test evaluation ********
2020-06-05 22:23:04,842 P32957 INFO Loading data...
2020-06-05 22:23:04,842 P32957 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-06-05 22:23:05,312 P32957 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-05 22:23:05,312 P32957 INFO Loading test data done.
2020-06-05 22:23:41,402 P32957 INFO [Metrics] logloss: 0.372247 - AUC: 0.792731

```
