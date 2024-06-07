## InterHAt_avazu_x4_001

A hands-on guide to run the InterHAt model on the Avazu_x4_001 dataset.

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
Dataset ID: [Avazu_x4](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x4). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [InterHAt](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/InterHAt.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [InterHAt_avazu_x4_tuner_config_02](./InterHAt_avazu_x4_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd InterHAt_avazu_x4_001
    nohup python run_expid.py --config ./InterHAt_avazu_x4_tuner_config_02 --expid InterHAt_avazu_x4_027_db66045d --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.374911 | 0.788180  |


### Logs
```python
2022-03-02 16:25:40,995 P3366 INFO {
    "attention_dim": "64",
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_regularizer": "1e-08",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'str', 'name': 'id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_dim": "500",
    "hidden_units": "[]",
    "label_col": "{'dtype': 'float', 'name': 'click'}",
    "layer_norm": "True",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "2",
    "model": "InterHAt",
    "model_id": "InterHAt_avazu_x4_027_db66045d",
    "model_root": "./Avazu/InterHAt_avazu_x4_001/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "1",
    "optimizer": "adam",
    "order": "3",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x4/test.csv",
    "train_data": "../data/Avazu/Avazu_x4/train.csv",
    "use_hdf5": "True",
    "use_residual": "True",
    "valid_data": "../data/Avazu/Avazu_x4/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2022-03-02 16:25:40,996 P3366 INFO Set up feature encoder...
2022-03-02 16:25:40,996 P3366 INFO Load feature_encoder from pickle: ../data/Avazu/avazu_x4_3bbbc4c9/feature_encoder.pkl
2022-03-02 16:25:43,812 P3366 INFO Total number of parameters: 60072293.
2022-03-02 16:25:43,813 P3366 INFO Loading data...
2022-03-02 16:25:43,816 P3366 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2022-03-02 16:25:46,944 P3366 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2022-03-02 16:25:48,461 P3366 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2022-03-02 16:25:48,634 P3366 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2022-03-02 16:25:48,635 P3366 INFO Loading train data done.
2022-03-02 16:25:51,362 P3366 INFO Start training: 3235 batches/epoch
2022-03-02 16:25:51,362 P3366 INFO ************ Epoch=1 start ************
2022-03-02 16:37:43,332 P3366 INFO [Metrics] logloss: 0.375034 - AUC: 0.787974
2022-03-02 16:37:43,338 P3366 INFO Save best model: monitor(max): 0.412940
2022-03-02 16:37:43,586 P3366 INFO --- 3235/3235 batches finished ---
2022-03-02 16:37:43,634 P3366 INFO Train loss: 0.382204
2022-03-02 16:37:43,634 P3366 INFO ************ Epoch=1 end ************
2022-03-02 16:49:34,575 P3366 INFO [Metrics] logloss: 0.380792 - AUC: 0.785530
2022-03-02 16:49:34,580 P3366 INFO Monitor(max) STOP: 0.404738 !
2022-03-02 16:49:34,580 P3366 INFO Reduce learning rate on plateau: 0.000100
2022-03-02 16:49:34,580 P3366 INFO --- 3235/3235 batches finished ---
2022-03-02 16:49:34,623 P3366 INFO Train loss: 0.339688
2022-03-02 16:49:34,623 P3366 INFO ************ Epoch=2 end ************
2022-03-02 17:01:24,053 P3366 INFO [Metrics] logloss: 0.425954 - AUC: 0.773329
2022-03-02 17:01:24,058 P3366 INFO Monitor(max) STOP: 0.347374 !
2022-03-02 17:01:24,059 P3366 INFO Reduce learning rate on plateau: 0.000010
2022-03-02 17:01:24,059 P3366 INFO Early stopping at epoch=3
2022-03-02 17:01:24,059 P3366 INFO --- 3235/3235 batches finished ---
2022-03-02 17:01:24,106 P3366 INFO Train loss: 0.302766
2022-03-02 17:01:24,106 P3366 INFO Training finished.
2022-03-02 17:01:24,106 P3366 INFO Load best model: /home/XXX/FuxiCTR_v1.0/benchmarks/Avazu/InterHAt_avazu_x4_001/avazu_x4_3bbbc4c9/InterHAt_avazu_x4_027_db66045d_model.ckpt
2022-03-02 17:01:24,449 P3366 INFO ****** Validation evaluation ******
2022-03-02 17:01:53,890 P3366 INFO [Metrics] logloss: 0.375034 - AUC: 0.787974
2022-03-02 17:01:53,964 P3366 INFO ******** Test evaluation ********
2022-03-02 17:01:53,964 P3366 INFO Loading data...
2022-03-02 17:01:53,965 P3366 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2022-03-02 17:01:54,473 P3366 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2022-03-02 17:01:54,473 P3366 INFO Loading test data done.
2022-03-02 17:02:24,226 P3366 INFO [Metrics] logloss: 0.374911 - AUC: 0.788180

```
