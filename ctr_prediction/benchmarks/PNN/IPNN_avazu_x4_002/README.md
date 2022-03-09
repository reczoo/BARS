## IPNN_avazu_x4_002

A hands-on guide to run the PNN model on the Avazu_x4_002 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [PNN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/PNN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [PNN_avazu_x4_tuner_config_01](./PNN_avazu_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd IPNN_avazu_x4_002
    nohup python run_expid.py --config ./PNN_avazu_x4_tuner_config_01 --expid PNN_avazu_x4_013_1893e8ce --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.368566 | 0.798799  |


### Logs
```python
2022-03-03 17:43:39,076 P33455 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_d102865a",
    "debug": "False",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'str', 'name': 'id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "1",
    "model": "PNN",
    "model_id": "PNN_avazu_x4_013_1893e8ce",
    "model_root": "./Avazu/PNN_avazu_x4_002/",
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
    "test_data": "../data/Avazu/Avazu_x4/test.csv",
    "train_data": "../data/Avazu/Avazu_x4/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x4/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2022-03-03 17:43:39,077 P33455 INFO Set up feature encoder...
2022-03-03 17:43:39,077 P33455 INFO Load feature_encoder from pickle: ../data/Avazu/avazu_x4_d102865a/feature_encoder.pkl
2022-03-03 17:43:49,973 P33455 INFO Total number of parameters: 335752601.
2022-03-03 17:43:49,974 P33455 INFO Loading data...
2022-03-03 17:43:49,978 P33455 INFO Loading data from h5: ../data/Avazu/avazu_x4_d102865a/train.h5
2022-03-03 17:44:01,855 P33455 INFO Loading data from h5: ../data/Avazu/avazu_x4_d102865a/valid.h5
2022-03-03 17:44:03,708 P33455 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2022-03-03 17:44:03,823 P33455 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2022-03-03 17:44:03,823 P33455 INFO Loading train data done.
2022-03-03 17:44:06,940 P33455 INFO Start training: 3235 batches/epoch
2022-03-03 17:44:06,941 P33455 INFO ************ Epoch=1 start ************
2022-03-03 17:54:20,321 P33455 INFO [Metrics] logloss: 0.368706 - AUC: 0.798507
2022-03-03 17:54:20,326 P33455 INFO Save best model: monitor(max): 0.429801
2022-03-03 17:54:23,176 P33455 INFO --- 3235/3235 batches finished ---
2022-03-03 17:54:23,348 P33455 INFO Train loss: 0.379465
2022-03-03 17:54:23,348 P33455 INFO ************ Epoch=1 end ************
2022-03-03 18:04:38,444 P33455 INFO [Metrics] logloss: 0.410905 - AUC: 0.772694
2022-03-03 18:04:38,448 P33455 INFO Monitor(max) STOP: 0.361789 !
2022-03-03 18:04:38,448 P33455 INFO Reduce learning rate on plateau: 0.000100
2022-03-03 18:04:38,448 P33455 INFO --- 3235/3235 batches finished ---
2022-03-03 18:04:38,638 P33455 INFO Train loss: 0.282284
2022-03-03 18:04:38,638 P33455 INFO ************ Epoch=2 end ************
2022-03-03 18:14:51,204 P33455 INFO [Metrics] logloss: 0.500300 - AUC: 0.756840
2022-03-03 18:14:51,209 P33455 INFO Monitor(max) STOP: 0.256540 !
2022-03-03 18:14:51,209 P33455 INFO Reduce learning rate on plateau: 0.000010
2022-03-03 18:14:51,209 P33455 INFO Early stopping at epoch=3
2022-03-03 18:14:51,209 P33455 INFO --- 3235/3235 batches finished ---
2022-03-03 18:14:51,418 P33455 INFO Train loss: 0.224136
2022-03-03 18:14:51,419 P33455 INFO Training finished.
2022-03-03 18:14:51,419 P33455 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Avazu/PNN_avazu_x4_002/avazu_x4_d102865a/PNN_avazu_x4_013_1893e8ce_model.ckpt
2022-03-03 18:14:53,149 P33455 INFO ****** Validation evaluation ******
2022-03-03 18:15:16,878 P33455 INFO [Metrics] logloss: 0.368706 - AUC: 0.798507
2022-03-03 18:15:17,665 P33455 INFO ******** Test evaluation ********
2022-03-03 18:15:17,665 P33455 INFO Loading data...
2022-03-03 18:15:17,665 P33455 INFO Loading data from h5: ../data/Avazu/avazu_x4_d102865a/test.h5
2022-03-03 18:15:18,197 P33455 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2022-03-03 18:15:18,197 P33455 INFO Loading test data done.
2022-03-03 18:15:42,055 P33455 INFO [Metrics] logloss: 0.368566 - AUC: 0.798799

```
