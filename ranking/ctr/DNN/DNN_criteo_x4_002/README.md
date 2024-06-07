## DNN_criteo_x4_002

A hands-on guide to run the DNN model on the Criteo_x4_002 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [DNN](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DNN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DNN_criteo_x4_tuner_config_01](./DNN_criteo_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DNN_criteo_x4_002
    nohup python run_expid.py --config ./DNN_criteo_x4_tuner_config_01 --expid DNN_criteo_x4_031_90b5a63c --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.440708 | 0.811195  |


### Logs
```python
2019-11-20 04:12:38,679 P11216 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "dataset_id": "criteo_x4_001_0d63c1a1",
    "dropout_rates": "0",
    "embedding_dim": "40",
    "embedding_regularizer": "l2(1.e-7)",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000, 1000]",
    "kernel_regularizer": "0",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DNN",
    "model_dir": "./Criteo/",
    "model_id": "DNN_criteo_x4_031_b43c210e",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "3",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "classification",
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'na_value': 0, 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'preprocess': 'convert_to_bucket', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'na_value': '', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "label_col": "{'dtype': 'float', 'name': 'Label'}",
    "min_categr_count": "2",
    "test_data": "../data/Criteo/Criteo_x4/test.csv",
    "train_data": "../data/Criteo/Criteo_x4/train.csv",
    "valid_data": "../data/Criteo/Criteo_x4/valid.csv",
    "version": "pytorch",
    "device": "1"
}
2019-11-20 04:12:38,680 P11216 INFO Set up feature encoder...
2019-11-20 04:12:53,607 P11216 INFO Load feature encoder cache from ./Criteo/criteo_x4_001_0d63c1a1/feature_encoder.pkl
2019-11-20 04:12:53,608 P11216 INFO Loading data...
2019-11-20 04:12:53,763 P11216 INFO Loading data from ./Criteo/criteo_x4_001_0d63c1a1/train.hdf5
2019-11-20 04:12:58,517 P11216 INFO Loading data from ./Criteo/criteo_x4_001_0d63c1a1/valid.hdf5
2019-11-20 04:13:00,254 P11216 INFO Train samples: total/36672493 - pos/9396350 - neg/27276143 - ratio/25.62%
2019-11-20 04:13:00,390 P11216 INFO Validation samples: total/4584062 - pos/1174544 - neg/3409518 - ratio/25.62%
2019-11-20 04:13:00,390 P11216 INFO Loading train data done.
2019-11-20 04:13:08,597 P11216 INFO **** Start training: 3668 batches/epoch ****
2019-11-20 04:26:26,609 P11216 INFO [Metrics] logloss: 0.442383 - AUC: 0.809219
2019-11-20 04:26:26,712 P11216 INFO Save best model: monitor(max): 0.366836
2019-11-20 04:26:27,875 P11216 INFO ******* 3668/3668 batches finished *******
2019-11-20 04:26:28,070 P11216 INFO [Train] loss: 0.451823
2019-11-20 04:26:28,070 P11216 INFO ************ Epoch=1 end ************
2019-11-20 04:39:44,415 P11216 INFO [Metrics] logloss: 0.441049 - AUC: 0.810782
2019-11-20 04:39:44,499 P11216 INFO Save best model: monitor(max): 0.369734
2019-11-20 04:39:46,274 P11216 INFO ******* 3668/3668 batches finished *******
2019-11-20 04:39:46,483 P11216 INFO [Train] loss: 0.440024
2019-11-20 04:39:46,483 P11216 INFO ************ Epoch=2 end ************
2019-11-20 04:53:05,299 P11216 INFO [Metrics] logloss: 0.447181 - AUC: 0.805238
2019-11-20 04:53:05,368 P11216 INFO Monitor(max) STOP: 0.358057 !!!
2019-11-20 04:53:05,369 P11216 INFO Reduce learning rate on plateau: 0.000100
2019-11-20 04:53:05,369 P11216 INFO ******* 3668/3668 batches finished *******
2019-11-20 04:53:05,583 P11216 INFO [Train] loss: 0.418077
2019-11-20 04:53:05,583 P11216 INFO ************ Epoch=3 end ************
2019-11-20 05:06:22,036 P11216 INFO [Metrics] logloss: 0.516321 - AUC: 0.781888
2019-11-20 05:06:22,103 P11216 INFO Monitor(max) STOP: 0.265567 !!!
2019-11-20 05:06:22,103 P11216 INFO Reduce learning rate on plateau: 0.000010
2019-11-20 05:06:22,103 P11216 INFO ******* 3668/3668 batches finished *******
2019-11-20 05:06:22,315 P11216 INFO [Train] loss: 0.364469
2019-11-20 05:06:22,315 P11216 INFO ************ Epoch=4 end ************
2019-11-20 05:19:38,152 P11216 INFO [Metrics] logloss: 0.553721 - AUC: 0.775120
2019-11-20 05:19:38,236 P11216 INFO Monitor(max) STOP: 0.221399 !!!
2019-11-20 05:19:38,236 P11216 INFO Reduce learning rate on plateau: 0.000001
2019-11-20 05:19:38,237 P11216 INFO Early stopping at epoch=5
2019-11-20 05:19:38,237 P11216 INFO ******* 3668/3668 batches finished *******
2019-11-20 05:19:38,380 P11216 INFO [Train] loss: 0.348458
2019-11-20 05:19:38,381 P11216 INFO Training finished.
2019-11-20 05:19:38,893 P11216 INFO ****** Train/validation evaluation ******
2019-11-20 05:27:26,291 P11216 INFO [Metrics] logloss: 0.409351 - AUC: 0.843586
2019-11-20 05:28:27,143 P11216 INFO [Metrics] logloss: 0.441049 - AUC: 0.810782
2019-11-20 05:28:27,654 P11216 INFO ******** Test evaluation ********
2019-11-20 05:28:27,654 P11216 INFO Loading data...
2019-11-20 05:28:27,654 P11216 INFO Loading data from ./Criteo/criteo_x4_001_0d63c1a1/test.hdf5
2019-11-20 05:28:28,404 P11216 INFO Test samples: total/4584062 - pos/1174544 - neg/3409518 - ratio/25.62%
2019-11-20 05:28:28,404 P11216 INFO Loading test data done.
2019-11-20 05:29:32,589 P11216 INFO [Metrics] logloss: 0.440708 - AUC: 0.811195

```
