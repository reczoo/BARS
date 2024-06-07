## DCNv2_avazu_x4_001

A hands-on guide to run the DCNv2 model on the Avazu_x4 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)


| [Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs) |
|:-----------------------------:|:-----------:|:--------:|:--------:|-------|
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
  fuxictr: 1.1.1

  ```

### Dataset
Please refer to [Avazu_x4]([Avazu_x4](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x4)) to get the dataset details.

### Code

We use the [DCNv2](https://github.com/reczoo/FuxiCTR/tree/v1.1.1/model_zoo/DCNv2) model code from [FuxiCTR-v1.1.1](https://github.com/reczoo/FuxiCTR/tree/v1.1.1) for this experiment.

Running steps:

1. Download [FuxiCTR-v1.1.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.1.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==1.1.1
    ```

2. Create a data directory and put the downloaded data files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DCNv2_avazu_x4_tuner_config_02](./DCNv2_avazu_x4_tuner_config_02). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/DCNv2
    nohup python run_expid.py --config YOUR_PATH/DCNv2/DCNv2_avazu_x4_tuner_config_02 --expid DCNv2_avazu_x4_011_19794dc6 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.793146 | 0.371865  |


### Logs
```python
2022-05-17 08:01:02,992 P30071 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "dnn_activations": "relu",
    "embedding_dim": "16",
    "embedding_regularizer": "1e-09",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'str', 'name': 'id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'type': 'categorical'}]",
    "gpu": "2",
    "label_col": "{'dtype': 'float', 'name': 'click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "low_rank": "32",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "2",
    "model": "DCNv2",
    "model_id": "DCNv2_avazu_x4_011_19794dc6",
    "model_root": "./Avazu/DCNv2_avazu_x4_001/",
    "model_structure": "parallel",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_cross_layers": "4",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_dnn_hidden_units": "[2000, 2000, 2000, 2000]",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "stacked_dnn_hidden_units": "[500, 500, 500]",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x4/test.csv",
    "train_data": "../data/Avazu/Avazu_x4/train.csv",
    "use_hdf5": "True",
    "use_low_rank_mixture": "False",
    "valid_data": "../data/Avazu/Avazu_x4/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-05-17 08:01:02,992 P30071 INFO Set up feature encoder...
2022-05-17 08:01:02,992 P30071 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2022-05-17 08:01:02,993 P30071 INFO Loading data...
2022-05-17 08:01:02,993 P30071 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2022-05-17 08:01:05,999 P30071 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2022-05-17 08:01:06,425 P30071 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%, blocks/1
2022-05-17 08:01:06,425 P30071 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%, blocks/1
2022-05-17 08:01:06,425 P30071 INFO Loading train data done.
2022-05-17 08:01:13,233 P30071 INFO Total number of parameters: 73385345.
2022-05-17 08:01:13,234 P30071 INFO Start training: 3235 batches/epoch
2022-05-17 08:01:13,234 P30071 INFO ************ Epoch=1 start ************
2022-05-17 08:19:08,450 P30071 INFO [Metrics] AUC: 0.792978 - logloss: 0.371967
2022-05-17 08:19:08,453 P30071 INFO Save best model: monitor(max): 0.421011
2022-05-17 08:19:08,971 P30071 INFO --- 3235/3235 batches finished ---
2022-05-17 08:19:09,016 P30071 INFO Train loss: 0.380317
2022-05-17 08:19:09,016 P30071 INFO ************ Epoch=1 end ************
2022-05-17 08:37:02,790 P30071 INFO [Metrics] AUC: 0.788575 - logloss: 0.383554
2022-05-17 08:37:02,792 P30071 INFO Monitor(max) STOP: 0.405021 !
2022-05-17 08:37:02,792 P30071 INFO Reduce learning rate on plateau: 0.000100
2022-05-17 08:37:02,792 P30071 INFO --- 3235/3235 batches finished ---
2022-05-17 08:37:02,829 P30071 INFO Train loss: 0.332264
2022-05-17 08:37:02,829 P30071 INFO ************ Epoch=2 end ************
2022-05-17 08:54:55,819 P30071 INFO [Metrics] AUC: 0.776121 - logloss: 0.427745
2022-05-17 08:54:55,822 P30071 INFO Monitor(max) STOP: 0.348377 !
2022-05-17 08:54:55,822 P30071 INFO Reduce learning rate on plateau: 0.000010
2022-05-17 08:54:55,822 P30071 INFO Early stopping at epoch=3
2022-05-17 08:54:55,822 P30071 INFO --- 3235/3235 batches finished ---
2022-05-17 08:54:55,858 P30071 INFO Train loss: 0.291580
2022-05-17 08:54:55,858 P30071 INFO Training finished.
2022-05-17 08:54:55,859 P30071 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/DCNv2_avazu_x4_001/avazu_x4_3bbbc4c9/DCNv2_avazu_x4_011_19794dc6.model
2022-05-17 08:55:02,455 P30071 INFO ****** Validation evaluation ******
2022-05-17 08:55:24,972 P30071 INFO [Metrics] AUC: 0.792978 - logloss: 0.371967
2022-05-17 08:55:25,040 P30071 INFO ******** Test evaluation ********
2022-05-17 08:55:25,040 P30071 INFO Loading data...
2022-05-17 08:55:25,041 P30071 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2022-05-17 08:55:25,405 P30071 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%, blocks/1
2022-05-17 08:55:25,405 P30071 INFO Loading test data done.
2022-05-17 08:55:47,592 P30071 INFO [Metrics] AUC: 0.793146 - logloss: 0.371865

```
