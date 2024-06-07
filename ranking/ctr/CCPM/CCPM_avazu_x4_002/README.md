## CCPM_avazu_x4_002

A hands-on guide to run the CCPM model on the Avazu_x4_002 dataset.

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
Dataset ID: [Avazu_x4](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x4). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [CCPM](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/CCPM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CCPM_avazu_x4_tuner_config_01](./CCPM_avazu_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CCPM_avazu_x4_002
    nohup python run_expid.py --config ./CCPM_avazu_x4_tuner_config_01 --expid CCPM_avazu_x4_025_e2254c14 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.372100 | 0.793210  |


### Logs
```python
2020-05-12 08:32:12,386 P6185 INFO {
    "activation": "Tanh",
    "batch_size": "10000",
    "channels": "[16, 32, 64]",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_001_74410863",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "kernel_heights": "[7, 5, 3]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "CCPM",
    "model_id": "CCPM_avazu_x4_025_a3bc05b9",
    "model_root": "./Avazu/CCPM_avazu/",
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
    "test_data": "../data/Avazu/avazu_x4_001_74410863/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_74410863/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/avazu_x4_001_74410863/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-05-12 08:32:12,389 P6185 INFO Set up feature encoder...
2020-05-12 08:32:12,389 P6185 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_74410863/feature_map.json
2020-05-12 08:32:12,389 P6185 INFO Loading data...
2020-05-12 08:32:12,434 P6185 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_74410863/train.h5
2020-05-12 08:32:16,290 P6185 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_74410863/valid.h5
2020-05-12 08:32:17,632 P6185 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-05-12 08:32:17,742 P6185 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-05-12 08:32:17,743 P6185 INFO Loading train data done.
2020-05-12 08:32:31,507 P6185 INFO **** Start training: 3235 batches/epoch ****
2020-05-12 09:31:16,216 P6185 INFO [Metrics] logloss: 0.372102 - AUC: 0.793156
2020-05-12 09:31:16,224 P6185 INFO Save best model: monitor(max): 0.421054
2020-05-12 09:31:17,847 P6185 INFO --- 3235/3235 batches finished ---
2020-05-12 09:31:17,900 P6185 INFO Train loss: 0.384212
2020-05-12 09:31:17,900 P6185 INFO ************ Epoch=1 end ************
2020-05-12 10:29:22,954 P6185 INFO [Metrics] logloss: 0.387160 - AUC: 0.781956
2020-05-12 10:29:22,960 P6185 INFO Monitor(max) STOP: 0.394796 !
2020-05-12 10:29:22,960 P6185 INFO Reduce learning rate on plateau: 0.000100
2020-05-12 10:29:22,960 P6185 INFO --- 3235/3235 batches finished ---
2020-05-12 10:29:23,018 P6185 INFO Train loss: 0.311386
2020-05-12 10:29:23,018 P6185 INFO ************ Epoch=2 end ************
2020-05-12 11:27:31,848 P6185 INFO [Metrics] logloss: 0.439963 - AUC: 0.763997
2020-05-12 11:27:31,860 P6185 INFO Monitor(max) STOP: 0.324034 !
2020-05-12 11:27:31,860 P6185 INFO Reduce learning rate on plateau: 0.000010
2020-05-12 11:27:31,860 P6185 INFO Early stopping at epoch=3
2020-05-12 11:27:31,860 P6185 INFO --- 3235/3235 batches finished ---
2020-05-12 11:27:31,914 P6185 INFO Train loss: 0.243214
2020-05-12 11:27:31,915 P6185 INFO Training finished.
2020-05-12 11:27:31,915 P6185 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Avazu/CCPM_avazu/avazu_x4_001_74410863/CCPM_avazu_x4_025_a3bc05b9_model.ckpt
2020-05-12 11:27:33,958 P6185 INFO ****** Train/validation evaluation ******
2020-05-12 11:59:06,294 P6185 INFO [Metrics] logloss: 0.326900 - AUC: 0.859540
2020-05-12 12:03:02,097 P6185 INFO [Metrics] logloss: 0.372102 - AUC: 0.793156
2020-05-12 12:03:02,216 P6185 INFO ******** Test evaluation ********
2020-05-12 12:03:02,216 P6185 INFO Loading data...
2020-05-12 12:03:02,216 P6185 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_74410863/test.h5
2020-05-12 12:03:02,931 P6185 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-05-12 12:03:02,932 P6185 INFO Loading test data done.
2020-05-12 12:06:58,854 P6185 INFO [Metrics] logloss: 0.372100 - AUC: 0.793210

```
