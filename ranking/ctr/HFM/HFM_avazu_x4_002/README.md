## HFM_avazu_x4_002

A hands-on guide to run the HFM model on the Avazu_x4_002 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [HFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/HFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HFM_avazu_x4_tuner_config_01](./HFM_avazu_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HFM_avazu_x4_002
    nohup python run_expid.py --config ./HFM_avazu_x4_tuner_config_01 --expid HFM_avazu_x4_004_1109463b --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.373484 | 0.790978  |


### Logs
```python
2020-05-08 12:08:25,094 P14940 INFO {
    "batch_norm": "False",
    "batch_size": "5000",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[64, 64, 64]",
    "interaction_type": "circular_correlation",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "HFM",
    "model_id": "HFM_avazu_x4_004_97481a37",
    "model_root": "./Avazu/HFM_avazu/",
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
    "use_dnn": "False",
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "version": "pytorch",
    "gpu": "1"
}
2020-05-08 12:08:25,095 P14940 INFO Set up feature encoder...
2020-05-08 12:08:25,095 P14940 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-05-08 12:08:25,095 P14940 INFO Loading data...
2020-05-08 12:08:25,097 P14940 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-05-08 12:08:27,374 P14940 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-05-08 12:08:28,878 P14940 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-05-08 12:08:28,996 P14940 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-05-08 12:08:28,997 P14940 INFO Loading train data done.
2020-05-08 12:08:41,267 P14940 INFO **** Start training: 6469 batches/epoch ****
2020-05-08 12:41:37,076 P14940 INFO [Metrics] logloss: 0.373503 - AUC: 0.790895
2020-05-08 12:41:37,163 P14940 INFO Save best model: monitor(max): 0.417392
2020-05-08 12:41:38,495 P14940 INFO --- 6469/6469 batches finished ---
2020-05-08 12:41:38,539 P14940 INFO Train loss: 0.382030
2020-05-08 12:41:38,539 P14940 INFO ************ Epoch=1 end ************
2020-05-08 13:14:33,647 P14940 INFO [Metrics] logloss: 0.388359 - AUC: 0.783456
2020-05-08 13:14:33,723 P14940 INFO Monitor(max) STOP: 0.395097 !
2020-05-08 13:14:33,723 P14940 INFO Reduce learning rate on plateau: 0.000100
2020-05-08 13:14:33,723 P14940 INFO --- 6469/6469 batches finished ---
2020-05-08 13:14:33,779 P14940 INFO Train loss: 0.319082
2020-05-08 13:14:33,779 P14940 INFO ************ Epoch=2 end ************
2020-05-08 13:47:32,587 P14940 INFO [Metrics] logloss: 0.435102 - AUC: 0.767151
2020-05-08 13:47:32,874 P14940 INFO Monitor(max) STOP: 0.332050 !
2020-05-08 13:47:32,874 P14940 INFO Reduce learning rate on plateau: 0.000010
2020-05-08 13:47:32,874 P14940 INFO Early stopping at epoch=3
2020-05-08 13:47:32,874 P14940 INFO --- 6469/6469 batches finished ---
2020-05-08 13:47:32,928 P14940 INFO Train loss: 0.255623
2020-05-08 13:47:32,929 P14940 INFO Training finished.
2020-05-08 13:47:32,929 P14940 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Avazu/HFM_avazu/avazu_x4_001_d45ad60e/HFM_avazu_x4_004_97481a37_avazu_x4_001_d45ad60e_model.ckpt
2020-05-08 13:47:36,382 P14940 INFO ****** Train/validation evaluation ******
2020-05-08 13:55:02,889 P14940 INFO [Metrics] logloss: 0.332298 - AUC: 0.851987
2020-05-08 13:55:59,635 P14940 INFO [Metrics] logloss: 0.373503 - AUC: 0.790895
2020-05-08 13:55:59,856 P14940 INFO ******** Test evaluation ********
2020-05-08 13:55:59,856 P14940 INFO Loading data...
2020-05-08 13:55:59,856 P14940 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-05-08 13:56:00,380 P14940 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-05-08 13:56:00,380 P14940 INFO Loading test data done.
2020-05-08 13:56:55,627 P14940 INFO [Metrics] logloss: 0.373484 - AUC: 0.790978

```
