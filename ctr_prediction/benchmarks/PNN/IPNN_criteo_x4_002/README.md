## IPNN_criteo_x4_002

A hands-on guide to run the PNN model on the Criteo_x4_002 dataset.

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
Dataset ID: [Criteo_x4_002](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x4_002). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [PNN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/PNN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [PNN_criteo_x4_tuner_config_04](./PNN_criteo_x4_tuner_config_04). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd IPNN_criteo_x4_002
    nohup python run_expid.py --config ./PNN_criteo_x4_tuner_config_04 --expid PNN_criteo_x4_005_2c82efe5 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.438836 | 0.813244  |


### Logs
```python
2020-03-01 10:18:59,329 P5677 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "dataset_id": "criteo_x4_001_be98441d",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "l2(5.e-7)",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[5000, 5000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "PNN",
    "model_id": "PNN_criteo_x4_005_c8ef60d9",
    "model_root": "./Criteo/PNN_criteo/",
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
    "use_hdf5": "True",
    "verbose": "1",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "test_data": "../data/Criteo/criteo_x4_001_be98441d/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_be98441d/train.h5",
    "valid_data": "../data/Criteo/criteo_x4_001_be98441d/valid.h5",
    "version": "pytorch",
    "gpu": "0"
}
2020-03-01 10:18:59,330 P5677 INFO Set up feature encoder...
2020-03-01 10:18:59,330 P5677 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-03-01 10:18:59,330 P5677 INFO Loading data...
2020-03-01 10:18:59,333 P5677 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-03-01 10:19:04,293 P5677 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-03-01 10:19:06,237 P5677 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-03-01 10:19:06,449 P5677 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-03-01 10:19:06,449 P5677 INFO Loading train data done.
2020-03-01 10:19:15,513 P5677 INFO **** Start training: 3668 batches/epoch ****
2020-03-01 10:50:59,211 P5677 INFO [Metrics] logloss: 0.440974 - AUC: 0.810732
2020-03-01 10:50:59,317 P5677 INFO Save best model: monitor(max): 0.369758
2020-03-01 10:51:00,438 P5677 INFO --- 3668/3668 batches finished ---
2020-03-01 10:51:00,492 P5677 INFO Train loss: 0.454230
2020-03-01 10:51:00,492 P5677 INFO ************ Epoch=1 end ************
2020-03-01 11:22:40,281 P5677 INFO [Metrics] logloss: 0.439248 - AUC: 0.812742
2020-03-01 11:22:40,393 P5677 INFO Save best model: monitor(max): 0.373495
2020-03-01 11:22:42,527 P5677 INFO --- 3668/3668 batches finished ---
2020-03-01 11:22:42,605 P5677 INFO Train loss: 0.449633
2020-03-01 11:22:42,605 P5677 INFO ************ Epoch=2 end ************
2020-03-01 11:54:21,347 P5677 INFO [Metrics] logloss: 0.443771 - AUC: 0.808374
2020-03-01 11:54:21,418 P5677 INFO Monitor(max) STOP: 0.364603 !
2020-03-01 11:54:21,419 P5677 INFO Reduce learning rate on plateau: 0.000100
2020-03-01 11:54:21,419 P5677 INFO --- 3668/3668 batches finished ---
2020-03-01 11:54:21,493 P5677 INFO Train loss: 0.441672
2020-03-01 11:54:21,493 P5677 INFO ************ Epoch=3 end ************
2020-03-01 12:25:57,528 P5677 INFO [Metrics] logloss: 0.526544 - AUC: 0.776310
2020-03-01 12:25:57,612 P5677 INFO Monitor(max) STOP: 0.249766 !
2020-03-01 12:25:57,612 P5677 INFO Reduce learning rate on plateau: 0.000010
2020-03-01 12:25:57,612 P5677 INFO Early stopping at epoch=4
2020-03-01 12:25:57,612 P5677 INFO --- 3668/3668 batches finished ---
2020-03-01 12:25:57,727 P5677 INFO Train loss: 0.357623
2020-03-01 12:25:57,727 P5677 INFO Training finished.
2020-03-01 12:25:57,727 P5677 INFO Load best model: /home/XXX/benchmarks/Criteo/PNN_criteo/criteo_x4_001_be98441d/PNN_criteo_x4_005_c8ef60d9_model.ckpt
2020-03-01 12:25:59,057 P5677 INFO ****** Train/validation evaluation ******
2020-03-01 12:36:02,713 P5677 INFO [Metrics] logloss: 0.408069 - AUC: 0.846062
2020-03-01 12:37:18,713 P5677 INFO [Metrics] logloss: 0.439248 - AUC: 0.812742
2020-03-01 12:37:18,928 P5677 INFO ******** Test evaluation ********
2020-03-01 12:37:18,928 P5677 INFO Loading data...
2020-03-01 12:37:18,928 P5677 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-03-01 12:37:19,770 P5677 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-03-01 12:37:19,770 P5677 INFO Loading test data done.
2020-03-01 12:38:34,305 P5677 INFO [Metrics] logloss: 0.438836 - AUC: 0.813244

```
