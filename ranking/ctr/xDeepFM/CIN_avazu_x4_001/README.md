## CIN_avazu_x4_001

A hands-on guide to run the xDeepFM model on the Avazu_x4_001 dataset.

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
Dataset ID: [Avazu_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu#Avazu_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [xDeepFM](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/xDeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CIN_avazu_x4_tuner_config_03](./CIN_avazu_x4_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CIN_avazu_x4_001
    nohup python run_expid.py --config ./CIN_avazu_x4_tuner_config_03 --expid xDeepFM_avazu_x4_002_9d81123c --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.374203 | 0.789414  |


### Logs
```python
2022-03-03 20:58:43,770 P35704 INFO {
    "batch_norm": "False",
    "batch_size": "5000",
    "cin_layer_units": "[200, 200, 200, 200, 200, 200]",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "dnn_hidden_units": "[]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'str', 'name': 'id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "label_col": "{'dtype': 'float', 'name': 'click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "2",
    "model": "xDeepFM",
    "model_id": "xDeepFM_avazu_x4_002_9d81123c",
    "model_root": "./Avazu/CIN_avazu_x4_001/",
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
2022-03-03 20:58:43,770 P35704 INFO Set up feature encoder...
2022-03-03 20:58:43,771 P35704 INFO Reading file: ../data/Avazu/Avazu_x4/train.csv
2022-03-03 21:00:46,649 P35704 INFO Preprocess feature columns...
2022-03-03 21:06:57,042 P35704 INFO Fit feature encoder...
2022-03-03 21:06:57,042 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}
2022-03-03 21:09:40,227 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C1', 'type': 'categorical'}
2022-03-03 21:09:45,232 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'banner_pos', 'type': 'categorical'}
2022-03-03 21:09:49,729 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'site_id', 'type': 'categorical'}
2022-03-03 21:09:54,889 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'site_domain', 'type': 'categorical'}
2022-03-03 21:10:00,030 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'site_category', 'type': 'categorical'}
2022-03-03 21:10:04,711 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'app_id', 'type': 'categorical'}
2022-03-03 21:10:09,811 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'app_domain', 'type': 'categorical'}
2022-03-03 21:10:15,126 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'app_category', 'type': 'categorical'}
2022-03-03 21:10:20,030 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_id', 'type': 'categorical'}
2022-03-03 21:10:28,226 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_ip', 'type': 'categorical'}
2022-03-03 21:10:47,662 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_model', 'type': 'categorical'}
2022-03-03 21:10:53,551 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_type', 'type': 'categorical'}
2022-03-03 21:10:58,246 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_conn_type', 'type': 'categorical'}
2022-03-03 21:11:02,985 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C14', 'type': 'categorical'}
2022-03-03 21:11:08,253 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C15', 'type': 'categorical'}
2022-03-03 21:11:13,309 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C16', 'type': 'categorical'}
2022-03-03 21:11:18,510 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C17', 'type': 'categorical'}
2022-03-03 21:11:24,126 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C18', 'type': 'categorical'}
2022-03-03 21:11:29,004 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C19', 'type': 'categorical'}
2022-03-03 21:11:34,279 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C20', 'type': 'categorical'}
2022-03-03 21:11:39,679 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C21', 'type': 'categorical'}
2022-03-03 21:11:44,932 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'type': 'categorical'}
2022-03-03 21:14:20,245 P35704 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'type': 'categorical'}
2022-03-03 21:16:54,381 P35704 INFO Set feature index...
2022-03-03 21:16:54,381 P35704 INFO Pickle feature_encode: ../data/Avazu/avazu_x4_3bbbc4c9/feature_encoder.pkl
2022-03-03 21:16:56,350 P35704 INFO Save feature_map to json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2022-03-03 21:16:56,351 P35704 INFO Set feature encoder done.
2022-03-03 21:17:00,916 P35704 INFO Total number of parameters: 68684176.
2022-03-03 21:17:00,917 P35704 INFO Loading data...
2022-03-03 21:17:00,921 P35704 INFO Reading file: ../data/Avazu/Avazu_x4/train.csv
2022-03-03 21:19:03,004 P35704 INFO Preprocess feature columns...
2022-03-03 21:25:10,892 P35704 INFO Transform feature columns...
2022-03-03 21:29:45,064 P35704 INFO Saving data to h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2022-03-03 21:29:54,915 P35704 INFO Reading file: ../data/Avazu/Avazu_x4/valid.csv
2022-03-03 21:30:09,107 P35704 INFO Preprocess feature columns...
2022-03-03 21:30:54,899 P35704 INFO Transform feature columns...
2022-03-03 21:31:29,987 P35704 INFO Saving data to h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2022-03-03 21:31:34,250 P35704 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2022-03-03 21:31:34,384 P35704 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2022-03-03 21:31:34,384 P35704 INFO Loading train data done.
2022-03-03 21:31:36,959 P35704 INFO Start training: 6469 batches/epoch
2022-03-03 21:31:36,959 P35704 INFO ************ Epoch=1 start ************
2022-03-03 22:50:10,610 P35704 INFO [Metrics] logloss: 0.374284 - AUC: 0.789217
2022-03-03 22:50:10,610 P35704 INFO Save best model: monitor(max): 0.414933
2022-03-03 22:50:10,809 P35704 INFO --- 6469/6469 batches finished ---
2022-03-03 22:50:11,208 P35704 INFO Train loss: 0.382470
2022-03-03 22:50:11,208 P35704 INFO ************ Epoch=1 end ************
2022-03-04 00:08:41,385 P35704 INFO [Metrics] logloss: 0.377057 - AUC: 0.789571
2022-03-04 00:08:41,386 P35704 INFO Monitor(max) STOP: 0.412515 !
2022-03-04 00:08:41,386 P35704 INFO Reduce learning rate on plateau: 0.000100
2022-03-04 00:08:41,386 P35704 INFO --- 6469/6469 batches finished ---
2022-03-04 00:08:41,853 P35704 INFO Train loss: 0.343310
2022-03-04 00:08:41,853 P35704 INFO ************ Epoch=2 end ************
2022-03-04 01:27:09,746 P35704 INFO [Metrics] logloss: 0.402184 - AUC: 0.778495
2022-03-04 01:27:09,747 P35704 INFO Monitor(max) STOP: 0.376310 !
2022-03-04 01:27:09,747 P35704 INFO Reduce learning rate on plateau: 0.000010
2022-03-04 01:27:09,747 P35704 INFO Early stopping at epoch=3
2022-03-04 01:27:09,747 P35704 INFO --- 6469/6469 batches finished ---
2022-03-04 01:27:10,164 P35704 INFO Train loss: 0.300144
2022-03-04 01:27:10,164 P35704 INFO Training finished.
2022-03-04 01:27:10,164 P35704 INFO Load best model: /home/XXX/FuxiCTR_v1.0/benchmarks/Avazu/CIN_avazu_x4_001/avazu_x4_3bbbc4c9/xDeepFM_avazu_x4_002_9d81123c_model.ckpt
2022-03-04 01:27:10,488 P35704 INFO ****** Validation evaluation ******
2022-03-04 01:29:41,097 P35704 INFO [Metrics] logloss: 0.374284 - AUC: 0.789217
2022-03-04 01:29:42,216 P35704 INFO ******** Test evaluation ********
2022-03-04 01:29:42,216 P35704 INFO Loading data...
2022-03-04 01:29:42,216 P35704 INFO Reading file: ../data/Avazu/Avazu_x4/test.csv
2022-03-04 01:29:56,884 P35704 INFO Preprocess feature columns...
2022-03-04 01:30:41,679 P35704 INFO Transform feature columns...
2022-03-04 01:31:14,381 P35704 INFO Saving data to h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2022-03-04 01:31:15,594 P35704 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2022-03-04 01:31:15,594 P35704 INFO Loading test data done.
2022-03-04 01:33:46,218 P35704 INFO [Metrics] logloss: 0.374203 - AUC: 0.789414

```
