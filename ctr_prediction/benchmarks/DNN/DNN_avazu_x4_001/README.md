## DNN_avazu_x4_001

A hands-on guide to run the DNN model on the Avazu_x4_001 dataset.

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
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [Avazu_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [DNN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DNN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DNN_avazu_x4_tuner_config_01](./DNN_avazu_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DNN_avazu_x4_001
    nohup python run_expid.py --config ./DNN_avazu_x4_tuner_config_01 --expid DNN_avazu_x4_021_144a901f --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.372169 | 0.792761  |


### Logs
```python
2020-06-13 16:37:49,535 P3995 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-08",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000, 1000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DNN",
    "model_id": "DNN_avazu_x4_3bbbc4c9_021_a28dd0ed",
    "model_root": "./Avazu/DNN_avazu/min2/",
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
    "test_data": "../data/Avazu/avazu_x4_3bbbc4c9/test.h5",
    "train_data": "../data/Avazu/avazu_x4_3bbbc4c9/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/avazu_x4_3bbbc4c9/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-13 16:37:49,536 P3995 INFO Set up feature encoder...
2020-06-13 16:37:49,536 P3995 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-06-13 16:37:49,536 P3995 INFO Loading data...
2020-06-13 16:37:49,538 P3995 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-06-13 16:37:52,794 P3995 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-06-13 16:37:54,067 P3995 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-13 16:37:54,169 P3995 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-13 16:37:54,169 P3995 INFO Loading train data done.
2020-06-13 16:37:59,796 P3995 INFO Start training: 3235 batches/epoch
2020-06-13 16:37:59,797 P3995 INFO ************ Epoch=1 start ************
2020-06-13 16:43:06,119 P3995 INFO [Metrics] logloss: 0.372318 - AUC: 0.792504
2020-06-13 16:43:06,121 P3995 INFO Save best model: monitor(max): 0.420187
2020-06-13 16:43:06,886 P3995 INFO --- 3235/3235 batches finished ---
2020-06-13 16:43:06,961 P3995 INFO Train loss: 0.380502
2020-06-13 16:43:06,961 P3995 INFO ************ Epoch=1 end ************
2020-06-13 16:48:08,875 P3995 INFO [Metrics] logloss: 0.381076 - AUC: 0.787331
2020-06-13 16:48:08,879 P3995 INFO Monitor(max) STOP: 0.406255 !
2020-06-13 16:48:08,879 P3995 INFO Reduce learning rate on plateau: 0.000100
2020-06-13 16:48:08,879 P3995 INFO --- 3235/3235 batches finished ---
2020-06-13 16:48:08,949 P3995 INFO Train loss: 0.334775
2020-06-13 16:48:08,949 P3995 INFO ************ Epoch=2 end ************
2020-06-13 16:53:12,322 P3995 INFO [Metrics] logloss: 0.423471 - AUC: 0.776052
2020-06-13 16:53:12,327 P3995 INFO Monitor(max) STOP: 0.352581 !
2020-06-13 16:53:12,327 P3995 INFO Reduce learning rate on plateau: 0.000010
2020-06-13 16:53:12,327 P3995 INFO Early stopping at epoch=3
2020-06-13 16:53:12,327 P3995 INFO --- 3235/3235 batches finished ---
2020-06-13 16:53:12,394 P3995 INFO Train loss: 0.294342
2020-06-13 16:53:12,395 P3995 INFO Training finished.
2020-06-13 16:53:12,395 P3995 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Avazu/DNN_avazu/min2/avazu_x4_3bbbc4c9/DNN_avazu_x4_3bbbc4c9_021_a28dd0ed_model.ckpt
2020-06-13 16:53:12,727 P3995 INFO ****** Train/validation evaluation ******
2020-06-13 16:56:31,508 P3995 INFO [Metrics] logloss: 0.337775 - AUC: 0.844945
2020-06-13 16:56:55,775 P3995 INFO [Metrics] logloss: 0.372318 - AUC: 0.792504
2020-06-13 16:56:55,846 P3995 INFO ******** Test evaluation ********
2020-06-13 16:56:55,846 P3995 INFO Loading data...
2020-06-13 16:56:55,846 P3995 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-06-13 16:56:56,301 P3995 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-13 16:56:56,301 P3995 INFO Loading test data done.
2020-06-13 16:57:19,781 P3995 INFO [Metrics] logloss: 0.372169 - AUC: 0.792761

```
