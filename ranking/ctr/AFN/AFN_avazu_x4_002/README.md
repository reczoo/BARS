## AFN_avazu_x4_002

A hands-on guide to run the AFN model on the Avazu_x4_002 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [AFN](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/AFN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFN_avazu_x4_tuner_config_03](./AFN_avazu_x4_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFN_avazu_x4_002
    nohup python run_expid.py --config ./AFN_avazu_x4_tuner_config_03 --expid AFN_avazu_x4_012_00ce5d6c --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.371986 | 0.793615  |


### Logs
```python
2020-05-17 06:41:56,475 P34306 INFO {
    "afn_activations": "relu",
    "afn_dropout": "0.2",
    "afn_hidden_units": "[1000, 1000]",
    "batch_norm": "True",
    "batch_size": "2000",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "dnn_activations": "relu",
    "dnn_dropout": "0",
    "dnn_hidden_units": "[]",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "ensemble_dnn": "False",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "logarithmic_neurons": "1600",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AFN",
    "model_id": "AFN_avazu_x4_012_c764a0b0",
    "model_root": "./Avazu/AFN_avazu/",
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
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "version": "pytorch",
    "gpu": "0"
}
2020-05-17 06:41:56,476 P34306 INFO Set up feature encoder...
2020-05-17 06:41:56,476 P34306 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-05-17 06:41:56,476 P34306 INFO Loading data...
2020-05-17 06:41:56,478 P34306 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-05-17 06:42:19,655 P34306 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-05-17 06:42:31,390 P34306 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-05-17 06:42:31,623 P34306 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-05-17 06:42:31,624 P34306 INFO Loading train data done.
2020-05-17 06:43:04,634 P34306 INFO **** Start training: 16172 batches/epoch ****
2020-05-17 07:58:28,430 P34306 INFO [Metrics] logloss: 0.377286 - AUC: 0.784229
2020-05-17 07:58:28,550 P34306 INFO Save best model: monitor(max): 0.406944
2020-05-17 07:58:39,475 P34306 INFO --- 16172/16172 batches finished ---
2020-05-17 07:58:39,559 P34306 INFO Train loss: 0.387296
2020-05-17 07:58:39,559 P34306 INFO ************ Epoch=1 end ************
2020-05-17 09:14:08,387 P34306 INFO [Metrics] logloss: 0.372118 - AUC: 0.793429
2020-05-17 09:14:08,526 P34306 INFO Save best model: monitor(max): 0.421311
2020-05-17 09:14:19,553 P34306 INFO --- 16172/16172 batches finished ---
2020-05-17 09:14:19,673 P34306 INFO Train loss: 0.358897
2020-05-17 09:14:19,673 P34306 INFO ************ Epoch=2 end ************
2020-05-17 10:29:47,309 P34306 INFO [Metrics] logloss: 0.433932 - AUC: 0.741570
2020-05-17 10:29:47,410 P34306 INFO Monitor(max) STOP: 0.307638 !
2020-05-17 10:29:47,410 P34306 INFO Reduce learning rate on plateau: 0.000100
2020-05-17 10:29:47,410 P34306 INFO --- 16172/16172 batches finished ---
2020-05-17 10:29:47,534 P34306 INFO Train loss: 0.284858
2020-05-17 10:29:47,534 P34306 INFO ************ Epoch=3 end ************
2020-05-17 11:45:20,574 P34306 INFO [Metrics] logloss: 0.518884 - AUC: 0.741191
2020-05-17 11:45:20,745 P34306 INFO Monitor(max) STOP: 0.222307 !
2020-05-17 11:45:20,745 P34306 INFO Reduce learning rate on plateau: 0.000010
2020-05-17 11:45:20,745 P34306 INFO Early stopping at epoch=4
2020-05-17 11:45:20,745 P34306 INFO --- 16172/16172 batches finished ---
2020-05-17 11:45:20,884 P34306 INFO Train loss: 0.262990
2020-05-17 11:45:20,884 P34306 INFO Training finished.
2020-05-17 11:45:20,884 P34306 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Avazu/AFN_avazu/avazu_x4_001_d45ad60e/AFN_avazu_x4_012_c764a0b0_avazu_x4_001_d45ad60e_model.ckpt
2020-05-17 11:45:46,766 P34306 INFO ****** Train/validation evaluation ******
2020-05-17 12:08:17,889 P34306 INFO [Metrics] logloss: 0.316834 - AUC: 0.869911
2020-05-17 12:11:05,378 P34306 INFO [Metrics] logloss: 0.372118 - AUC: 0.793429
2020-05-17 12:11:05,813 P34306 INFO ******** Test evaluation ********
2020-05-17 12:11:05,813 P34306 INFO Loading data...
2020-05-17 12:11:05,813 P34306 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-05-17 12:11:06,437 P34306 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-05-17 12:11:06,437 P34306 INFO Loading test data done.
2020-05-17 12:13:51,955 P34306 INFO [Metrics] logloss: 0.371986 - AUC: 0.793615

```
