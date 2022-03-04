## AFN+_avazu_x4_001

A hands-on guide to run the AFN model on the Avazu_x4_001 dataset.

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
Dataset ID: [Avazu_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [AFN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/AFN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFN_avazu_x4_tuner_config_11](./AFN_avazu_x4_tuner_config_11). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFN+_avazu_x4_001
    nohup python run_expid.py --config ./AFN_avazu_x4_tuner_config_11 --expid AFN_avazu_x4_003_9466021e --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.372589 | 0.792929  |


### Logs
```python
2020-07-15 19:27:16,774 P9162 INFO {
    "afn_activations": "relu",
    "afn_dropout": "0",
    "afn_hidden_units": "[1000, 1000]",
    "batch_norm": "True",
    "batch_size": "2000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_dropout": "0",
    "dnn_hidden_units": "[1000, 1000]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-08",
    "ensemble_dnn": "True",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "learning_rate": "0.001",
    "logarithmic_neurons": "1200",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AFN",
    "model_id": "AFN_avazu_x4_3bbbc4c9_003_59b37b70",
    "model_root": "./Avazu/AFN_avazu/min2/",
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
    "test_data": "../data/Avazu/avazu_x4_3bbbc4c9/test.h5",
    "train_data": "../data/Avazu/avazu_x4_3bbbc4c9/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/avazu_x4_3bbbc4c9/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-15 19:27:16,775 P9162 INFO Set up feature encoder...
2020-07-15 19:27:16,775 P9162 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-07-15 19:27:19,933 P9162 INFO Total number of parameters: 141660453.
2020-07-15 19:27:19,933 P9162 INFO Loading data...
2020-07-15 19:27:19,935 P9162 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-07-15 19:27:22,730 P9162 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-07-15 19:27:23,953 P9162 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-07-15 19:27:24,064 P9162 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-07-15 19:27:24,064 P9162 INFO Loading train data done.
2020-07-15 19:27:28,742 P9162 INFO Start training: 16172 batches/epoch
2020-07-15 19:27:28,742 P9162 INFO ************ Epoch=1 start ************
2020-07-15 20:03:04,342 P9162 INFO [Metrics] logloss: 0.373648 - AUC: 0.790299
2020-07-15 20:03:04,343 P9162 INFO Save best model: monitor(max): 0.416651
2020-07-15 20:03:04,887 P9162 INFO --- 16172/16172 batches finished ---
2020-07-15 20:03:04,924 P9162 INFO Train loss: 0.380791
2020-07-15 20:03:04,924 P9162 INFO ************ Epoch=1 end ************
2020-07-15 20:39:10,562 P9162 INFO [Metrics] logloss: 0.372653 - AUC: 0.792778
2020-07-15 20:39:10,566 P9162 INFO Save best model: monitor(max): 0.420125
2020-07-15 20:39:11,711 P9162 INFO --- 16172/16172 batches finished ---
2020-07-15 20:39:11,749 P9162 INFO Train loss: 0.350438
2020-07-15 20:39:11,749 P9162 INFO ************ Epoch=2 end ************
2020-07-15 21:15:01,438 P9162 INFO [Metrics] logloss: 0.375738 - AUC: 0.791107
2020-07-15 21:15:01,440 P9162 INFO Monitor(max) STOP: 0.415369 !
2020-07-15 21:15:01,440 P9162 INFO Reduce learning rate on plateau: 0.000100
2020-07-15 21:15:01,441 P9162 INFO --- 16172/16172 batches finished ---
2020-07-15 21:15:01,480 P9162 INFO Train loss: 0.336072
2020-07-15 21:15:01,480 P9162 INFO ************ Epoch=3 end ************
2020-07-15 21:50:47,278 P9162 INFO [Metrics] logloss: 0.412006 - AUC: 0.775337
2020-07-15 21:50:47,292 P9162 INFO Monitor(max) STOP: 0.363331 !
2020-07-15 21:50:47,292 P9162 INFO Reduce learning rate on plateau: 0.000010
2020-07-15 21:50:47,292 P9162 INFO Early stopping at epoch=4
2020-07-15 21:50:47,292 P9162 INFO --- 16172/16172 batches finished ---
2020-07-15 21:50:47,333 P9162 INFO Train loss: 0.299461
2020-07-15 21:50:47,334 P9162 INFO Training finished.
2020-07-15 21:50:47,334 P9162 INFO Load best model: /home/XXX/benchmarks/Avazu/AFN_avazu/min2/avazu_x4_3bbbc4c9/AFN_avazu_x4_3bbbc4c9_003_59b37b70_model.ckpt
2020-07-15 21:50:48,542 P9162 INFO ****** Train/validation evaluation ******
2020-07-15 22:00:20,270 P9162 INFO [Metrics] logloss: 0.327129 - AUC: 0.855984
2020-07-15 22:01:26,125 P9162 INFO [Metrics] logloss: 0.372653 - AUC: 0.792778
2020-07-15 22:01:26,172 P9162 INFO ******** Test evaluation ********
2020-07-15 22:01:26,172 P9162 INFO Loading data...
2020-07-15 22:01:26,172 P9162 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-07-15 22:01:27,500 P9162 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-07-15 22:01:27,500 P9162 INFO Loading test data done.
2020-07-15 22:02:34,136 P9162 INFO [Metrics] logloss: 0.372589 - AUC: 0.792929

```
