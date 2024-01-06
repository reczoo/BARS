## AFN+_avazu_x4_002

A hands-on guide to run the AFN model on the Avazu_x4_002 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [AFN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/AFN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFN_avazu_x4_tuner_config_07](./AFN_avazu_x4_tuner_config_07). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFN+_avazu_x4_002
    nohup python run_expid.py --config ./AFN_avazu_x4_tuner_config_07 --expid AFN_avazu_x4_003_24bfb45b --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.370024 | 0.796542  |


### Logs
```python
2020-02-22 05:02:10,768 P7379 INFO {
    "afn_activations": "relu",
    "afn_dropout": "0",
    "afn_hidden_units": "[1000, 1000, 1000]",
    "batch_norm": "True",
    "batch_size": "10000",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "dnn_activations": "relu",
    "dnn_dropout": "0",
    "dnn_hidden_units": "[1000, 1000, 1000]",
    "embedding_dim": "20",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-08",
    "ensemble_dnn": "True",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "logarithmic_neurons": "1200",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AFN",
    "model_id": "AFN_avazu_x4_003_8443ba13",
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
2020-02-22 05:02:10,769 P7379 INFO Set up feature encoder...
2020-02-22 05:02:10,769 P7379 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-02-22 05:02:10,769 P7379 INFO Loading data...
2020-02-22 05:02:10,771 P7379 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-02-22 05:02:13,008 P7379 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-02-22 05:02:14,252 P7379 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-02-22 05:02:14,372 P7379 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-02-22 05:02:14,372 P7379 INFO Loading train data done.
2020-02-22 05:02:26,414 P7379 INFO **** Start training: 3235 batches/epoch ****
2020-02-22 05:37:13,353 P7379 INFO [Metrics] logloss: 0.370202 - AUC: 0.796254
2020-02-22 05:37:13,426 P7379 INFO Save best model: monitor(max): 0.426052
2020-02-22 05:37:14,768 P7379 INFO --- 3235/3235 batches finished ---
2020-02-22 05:37:14,808 P7379 INFO Train loss: 0.380281
2020-02-22 05:37:14,808 P7379 INFO ************ Epoch=1 end ************
2020-02-22 06:12:00,843 P7379 INFO [Metrics] logloss: 0.403646 - AUC: 0.777398
2020-02-22 06:12:00,924 P7379 INFO Monitor(max) STOP: 0.373752 !
2020-02-22 06:12:00,924 P7379 INFO Reduce learning rate on plateau: 0.000100
2020-02-22 06:12:00,924 P7379 INFO --- 3235/3235 batches finished ---
2020-02-22 06:12:00,985 P7379 INFO Train loss: 0.288434
2020-02-22 06:12:00,985 P7379 INFO ************ Epoch=2 end ************
2020-02-22 06:46:46,857 P7379 INFO [Metrics] logloss: 0.477018 - AUC: 0.764034
2020-02-22 06:46:46,930 P7379 INFO Monitor(max) STOP: 0.287016 !
2020-02-22 06:46:46,930 P7379 INFO Reduce learning rate on plateau: 0.000010
2020-02-22 06:46:46,930 P7379 INFO Early stopping at epoch=3
2020-02-22 06:46:46,930 P7379 INFO --- 3235/3235 batches finished ---
2020-02-22 06:46:46,988 P7379 INFO Train loss: 0.245259
2020-02-22 06:46:46,988 P7379 INFO Training finished.
2020-02-22 06:46:46,988 P7379 INFO Load best model: /home/hispace/container/data/zhujieming/FuxiCTR/benchmarks/Avazu/AFN_avazu/avazu_x4_001_d45ad60e/AFN_avazu_x4_003_8443ba13_avazu_x4_001_d45ad60e_model.ckpt
2020-02-22 06:46:48,811 P7379 INFO ****** Train/validation evaluation ******
2020-02-22 06:56:41,125 P7379 INFO [Metrics] logloss: 0.319393 - AUC: 0.873740
2020-02-22 06:57:55,863 P7379 INFO [Metrics] logloss: 0.370202 - AUC: 0.796254
2020-02-22 06:57:55,989 P7379 INFO ******** Test evaluation ********
2020-02-22 06:57:55,989 P7379 INFO Loading data...
2020-02-22 06:57:55,989 P7379 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-02-22 06:57:56,531 P7379 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-02-22 06:57:56,531 P7379 INFO Loading test data done.
2020-02-22 06:59:11,017 P7379 INFO [Metrics] logloss: 0.370024 - AUC: 0.796542

```
