## MaskNet_criteo_x1

A hands-on guide to run the MaskNet model on the Criteo_x1 dataset.

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
  fuxictr: 1.2.1

  ```

### Dataset
Dataset ID: [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo#Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [MaskNet](https://github.com/xue-pai/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/MaskNet.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [MaskNet_criteo_x1_tuner_config_03](./MaskNet_criteo_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd MaskNet_criteo_x1
    nohup python run_expid.py --config ./MaskNet_criteo_x1_tuner_config_03 --expid MaskNet_criteo_x1_008_7071fa3f --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.813909 | 0.438021  |


### Logs
```python
2022-05-26 10:26:49,285 P41671 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "dnn_hidden_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "emb_layernorm": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "7",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "MaskNet",
    "model_id": "MaskNet_criteo_x1_008_7071fa3f",
    "model_root": "./Criteo/MaskNet_criteo_x1/",
    "model_type": "SerialMaskNet",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_layernorm": "True",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_block_dim": "64",
    "parallel_num_blocks": "1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "0.2",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-05-26 10:26:49,285 P41671 INFO Set up feature encoder...
2022-05-26 10:26:49,286 P41671 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-05-26 10:26:49,286 P41671 INFO Loading data...
2022-05-26 10:26:49,287 P41671 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-05-26 10:26:53,809 P41671 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-05-26 10:26:55,000 P41671 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-05-26 10:26:55,000 P41671 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-05-26 10:26:55,000 P41671 INFO Loading train data done.
2022-05-26 10:27:01,230 P41671 INFO Total number of parameters: 21530629.
2022-05-26 10:27:01,230 P41671 INFO Start training: 8058 batches/epoch
2022-05-26 10:27:01,231 P41671 INFO ************ Epoch=1 start ************
2022-05-26 10:40:48,842 P41671 INFO [Metrics] AUC: 0.804949 - logloss: 0.446448
2022-05-26 10:40:48,843 P41671 INFO Save best model: monitor(max): 0.804949
2022-05-26 10:40:49,121 P41671 INFO --- 8058/8058 batches finished ---
2022-05-26 10:40:49,172 P41671 INFO Train loss: 0.461097
2022-05-26 10:40:49,173 P41671 INFO ************ Epoch=1 end ************
2022-05-26 10:54:37,393 P41671 INFO [Metrics] AUC: 0.807359 - logloss: 0.444166
2022-05-26 10:54:37,394 P41671 INFO Save best model: monitor(max): 0.807359
2022-05-26 10:54:37,502 P41671 INFO --- 8058/8058 batches finished ---
2022-05-26 10:54:37,549 P41671 INFO Train loss: 0.455280
2022-05-26 10:54:37,550 P41671 INFO ************ Epoch=2 end ************
2022-05-26 11:08:24,750 P41671 INFO [Metrics] AUC: 0.808440 - logloss: 0.443105
2022-05-26 11:08:24,752 P41671 INFO Save best model: monitor(max): 0.808440
2022-05-26 11:08:24,865 P41671 INFO --- 8058/8058 batches finished ---
2022-05-26 11:08:24,915 P41671 INFO Train loss: 0.453643
2022-05-26 11:08:24,915 P41671 INFO ************ Epoch=3 end ************
2022-05-26 11:22:11,843 P41671 INFO [Metrics] AUC: 0.808830 - logloss: 0.442825
2022-05-26 11:22:11,844 P41671 INFO Save best model: monitor(max): 0.808830
2022-05-26 11:22:11,944 P41671 INFO --- 8058/8058 batches finished ---
2022-05-26 11:22:11,992 P41671 INFO Train loss: 0.452815
2022-05-26 11:22:11,992 P41671 INFO ************ Epoch=4 end ************
2022-05-26 11:35:55,981 P41671 INFO [Metrics] AUC: 0.809381 - logloss: 0.442204
2022-05-26 11:35:55,982 P41671 INFO Save best model: monitor(max): 0.809381
2022-05-26 11:35:56,086 P41671 INFO --- 8058/8058 batches finished ---
2022-05-26 11:35:56,133 P41671 INFO Train loss: 0.452272
2022-05-26 11:35:56,133 P41671 INFO ************ Epoch=5 end ************
2022-05-26 11:49:40,423 P41671 INFO [Metrics] AUC: 0.809543 - logloss: 0.442238
2022-05-26 11:49:40,425 P41671 INFO Save best model: monitor(max): 0.809543
2022-05-26 11:49:40,527 P41671 INFO --- 8058/8058 batches finished ---
2022-05-26 11:49:40,572 P41671 INFO Train loss: 0.451902
2022-05-26 11:49:40,573 P41671 INFO ************ Epoch=6 end ************
2022-05-26 12:03:27,171 P41671 INFO [Metrics] AUC: 0.809817 - logloss: 0.442100
2022-05-26 12:03:27,172 P41671 INFO Save best model: monitor(max): 0.809817
2022-05-26 12:03:27,285 P41671 INFO --- 8058/8058 batches finished ---
2022-05-26 12:03:27,334 P41671 INFO Train loss: 0.451610
2022-05-26 12:03:27,334 P41671 INFO ************ Epoch=7 end ************
2022-05-26 12:17:09,478 P41671 INFO [Metrics] AUC: 0.810088 - logloss: 0.441912
2022-05-26 12:17:09,479 P41671 INFO Save best model: monitor(max): 0.810088
2022-05-26 12:17:09,586 P41671 INFO --- 8058/8058 batches finished ---
2022-05-26 12:17:09,656 P41671 INFO Train loss: 0.451344
2022-05-26 12:17:09,657 P41671 INFO ************ Epoch=8 end ************
2022-05-26 12:30:54,589 P41671 INFO [Metrics] AUC: 0.810064 - logloss: 0.441637
2022-05-26 12:30:54,590 P41671 INFO Monitor(max) STOP: 0.810064 !
2022-05-26 12:30:54,590 P41671 INFO Reduce learning rate on plateau: 0.000100
2022-05-26 12:30:54,590 P41671 INFO --- 8058/8058 batches finished ---
2022-05-26 12:30:54,640 P41671 INFO Train loss: 0.451133
2022-05-26 12:30:54,640 P41671 INFO ************ Epoch=9 end ************
2022-05-26 12:44:40,776 P41671 INFO [Metrics] AUC: 0.813097 - logloss: 0.438987
2022-05-26 12:44:40,778 P41671 INFO Save best model: monitor(max): 0.813097
2022-05-26 12:44:40,880 P41671 INFO --- 8058/8058 batches finished ---
2022-05-26 12:44:40,930 P41671 INFO Train loss: 0.439918
2022-05-26 12:44:40,930 P41671 INFO ************ Epoch=10 end ************
2022-05-26 12:58:25,755 P41671 INFO [Metrics] AUC: 0.813501 - logloss: 0.438563
2022-05-26 12:58:25,756 P41671 INFO Save best model: monitor(max): 0.813501
2022-05-26 12:58:25,861 P41671 INFO --- 8058/8058 batches finished ---
2022-05-26 12:58:25,910 P41671 INFO Train loss: 0.435416
2022-05-26 12:58:25,911 P41671 INFO ************ Epoch=11 end ************
2022-05-26 13:12:14,783 P41671 INFO [Metrics] AUC: 0.813307 - logloss: 0.438698
2022-05-26 13:12:14,784 P41671 INFO Monitor(max) STOP: 0.813307 !
2022-05-26 13:12:14,784 P41671 INFO Reduce learning rate on plateau: 0.000010
2022-05-26 13:12:14,784 P41671 INFO --- 8058/8058 batches finished ---
2022-05-26 13:12:14,837 P41671 INFO Train loss: 0.433191
2022-05-26 13:12:14,838 P41671 INFO ************ Epoch=12 end ************
2022-05-26 13:25:57,223 P41671 INFO [Metrics] AUC: 0.812605 - logloss: 0.439862
2022-05-26 13:25:57,225 P41671 INFO Monitor(max) STOP: 0.812605 !
2022-05-26 13:25:57,225 P41671 INFO Reduce learning rate on plateau: 0.000001
2022-05-26 13:25:57,225 P41671 INFO Early stopping at epoch=13
2022-05-26 13:25:57,225 P41671 INFO --- 8058/8058 batches finished ---
2022-05-26 13:25:57,273 P41671 INFO Train loss: 0.428167
2022-05-26 13:25:57,273 P41671 INFO Training finished.
2022-05-26 13:25:57,273 P41671 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/MaskNet_criteo_x1/criteo_x1_7b681156/MaskNet_criteo_x1_008_7071fa3f.model
2022-05-26 13:26:01,557 P41671 INFO ****** Validation evaluation ******
2022-05-26 13:26:28,850 P41671 INFO [Metrics] AUC: 0.813501 - logloss: 0.438563
2022-05-26 13:26:28,931 P41671 INFO ******** Test evaluation ********
2022-05-26 13:26:28,931 P41671 INFO Loading data...
2022-05-26 13:26:28,932 P41671 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-05-26 13:26:29,653 P41671 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-05-26 13:26:29,654 P41671 INFO Loading test data done.
2022-05-26 13:26:45,633 P41671 INFO [Metrics] AUC: 0.813909 - logloss: 0.438021

```
