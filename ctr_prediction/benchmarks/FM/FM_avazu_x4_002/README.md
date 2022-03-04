## FM_avazu_x4_002

A hands-on guide to run the FM model on the Avazu_x4_002 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FM_avazu_x4_tuner_config_04](./FM_avazu_x4_tuner_config_04). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FM_avazu_x4_002
    nohup python run_expid.py --config ./FM_avazu_x4_tuner_config_04 --expid FM_avazu_x4_003_6dd622eb --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.373561 | 0.790887  |


### Logs
```python
2020-02-23 14:54:49,630 P28545 INFO {
    "batch_size": "5000",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FM",
    "model_id": "FM_avazu_x4_003_509e8ea6",
    "model_root": "./Avazu/FM_avazu/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "0",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "verbose": "1",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "version": "pytorch",
    "gpu": "0"
}
2020-02-23 14:54:49,631 P28545 INFO Set up feature encoder...
2020-02-23 14:54:49,631 P28545 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-02-23 14:54:49,632 P28545 INFO Loading data...
2020-02-23 14:54:49,634 P28545 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-02-23 14:54:52,575 P28545 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-02-23 14:54:54,131 P28545 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-02-23 14:54:54,302 P28545 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-02-23 14:54:54,302 P28545 INFO Loading train data done.
2020-02-23 14:55:05,283 P28545 INFO **** Start training: 6469 batches/epoch ****
2020-02-23 15:12:54,365 P28545 INFO [Metrics] logloss: 0.373535 - AUC: 0.790776
2020-02-23 15:12:54,456 P28545 INFO Save best model: monitor(max): 0.417241
2020-02-23 15:12:55,856 P28545 INFO --- 6469/6469 batches finished ---
2020-02-23 15:12:55,910 P28545 INFO Train loss: 0.383358
2020-02-23 15:12:55,910 P28545 INFO ************ Epoch=1 end ************
2020-02-23 15:30:45,511 P28545 INFO [Metrics] logloss: 0.383542 - AUC: 0.786443
2020-02-23 15:30:45,602 P28545 INFO Monitor(max) STOP: 0.402901 !
2020-02-23 15:30:45,602 P28545 INFO Reduce learning rate on plateau: 0.000100
2020-02-23 15:30:45,602 P28545 INFO --- 6469/6469 batches finished ---
2020-02-23 15:30:45,671 P28545 INFO Train loss: 0.325166
2020-02-23 15:30:45,671 P28545 INFO ************ Epoch=2 end ************
2020-02-23 15:48:34,738 P28545 INFO [Metrics] logloss: 0.402431 - AUC: 0.778739
2020-02-23 15:48:34,845 P28545 INFO Monitor(max) STOP: 0.376308 !
2020-02-23 15:48:34,845 P28545 INFO Reduce learning rate on plateau: 0.000010
2020-02-23 15:48:34,845 P28545 INFO Early stopping at epoch=3
2020-02-23 15:48:34,845 P28545 INFO --- 6469/6469 batches finished ---
2020-02-23 15:48:34,953 P28545 INFO Train loss: 0.270158
2020-02-23 15:48:34,953 P28545 INFO Training finished.
2020-02-23 15:48:34,953 P28545 INFO Load best model: /home/XXX/benchmarks/Avazu/FM_avazu/avazu_x4_001_d45ad60e/FM_avazu_x4_003_509e8ea6_avazu_x4_001_d45ad60e_model.ckpt
2020-02-23 15:48:36,859 P28545 INFO ****** Train/validation evaluation ******
2020-02-23 15:53:43,890 P28545 INFO [Metrics] logloss: 0.331818 - AUC: 0.853327
2020-02-23 15:54:22,893 P28545 INFO [Metrics] logloss: 0.373535 - AUC: 0.790776
2020-02-23 15:54:23,067 P28545 INFO ******** Test evaluation ********
2020-02-23 15:54:23,067 P28545 INFO Loading data...
2020-02-23 15:54:23,067 P28545 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-02-23 15:54:23,944 P28545 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-02-23 15:54:23,944 P28545 INFO Loading test data done.
2020-02-23 15:55:00,616 P28545 INFO [Metrics] logloss: 0.373561 - AUC: 0.790887

```
