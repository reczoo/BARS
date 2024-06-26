## xDeepFM_avazu_x4_001

A hands-on guide to run the xDeepFM model on the Avazu_x4_001 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

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
Dataset ID: [Avazu_x4](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x4). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [xDeepFM](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/xDeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [xDeepFM_avazu_x4_tuner_config_03](./xDeepFM_avazu_x4_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd xDeepFM_avazu_x4_001
    nohup python run_expid.py --config ./xDeepFM_avazu_x4_tuner_config_03 --expid xDeepFM_avazu_x4_001_579eef11 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.371780 | 0.793283  |


### Logs
```python
2020-06-14 21:28:50,584 P660 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "cin_layer_units": "[276]",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "dnn_hidden_units": "[500, 500, 500]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "hidden_activations": "relu",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "xDeepFM",
    "model_id": "xDeepFM_avazu_x4_3bbbc4c9_001_5e656a3d",
    "model_root": "./Avazu/xDeepFM_avazu/min2/",
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
2020-06-14 21:28:50,587 P660 INFO Set up feature encoder...
2020-06-14 21:28:50,587 P660 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-06-14 21:28:50,588 P660 INFO Loading data...
2020-06-14 21:28:50,593 P660 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-06-14 21:28:53,183 P660 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-06-14 21:28:54,479 P660 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-14 21:28:54,618 P660 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-14 21:28:54,618 P660 INFO Loading train data done.
2020-06-14 21:29:03,158 P660 INFO Start training: 3235 batches/epoch
2020-06-14 21:29:03,158 P660 INFO ************ Epoch=1 start ************
2020-06-14 21:38:14,282 P660 INFO [Metrics] logloss: 0.371895 - AUC: 0.793132
2020-06-14 21:38:14,283 P660 INFO Save best model: monitor(max): 0.421237
2020-06-14 21:38:14,626 P660 INFO --- 3235/3235 batches finished ---
2020-06-14 21:38:14,669 P660 INFO Train loss: 0.380561
2020-06-14 21:38:14,670 P660 INFO ************ Epoch=1 end ************
2020-06-14 21:47:26,949 P660 INFO [Metrics] logloss: 0.380666 - AUC: 0.788586
2020-06-14 21:47:26,952 P660 INFO Monitor(max) STOP: 0.407919 !
2020-06-14 21:47:26,952 P660 INFO Reduce learning rate on plateau: 0.000100
2020-06-14 21:47:26,952 P660 INFO --- 3235/3235 batches finished ---
2020-06-14 21:47:26,992 P660 INFO Train loss: 0.331775
2020-06-14 21:47:26,993 P660 INFO ************ Epoch=2 end ************
2020-06-14 21:56:37,720 P660 INFO [Metrics] logloss: 0.424659 - AUC: 0.775277
2020-06-14 21:56:37,724 P660 INFO Monitor(max) STOP: 0.350618 !
2020-06-14 21:56:37,724 P660 INFO Reduce learning rate on plateau: 0.000010
2020-06-14 21:56:37,727 P660 INFO Early stopping at epoch=3
2020-06-14 21:56:37,728 P660 INFO --- 3235/3235 batches finished ---
2020-06-14 21:56:37,776 P660 INFO Train loss: 0.285276
2020-06-14 21:56:37,777 P660 INFO Training finished.
2020-06-14 21:56:37,777 P660 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Avazu/xDeepFM_avazu/min2/avazu_x4_3bbbc4c9/xDeepFM_avazu_x4_3bbbc4c9_001_5e656a3d_model.ckpt
2020-06-14 21:56:38,255 P660 INFO ****** Train/validation evaluation ******
2020-06-14 21:59:49,037 P660 INFO [Metrics] logloss: 0.338127 - AUC: 0.845617
2020-06-14 22:00:11,516 P660 INFO [Metrics] logloss: 0.371895 - AUC: 0.793132
2020-06-14 22:00:11,592 P660 INFO ******** Test evaluation ********
2020-06-14 22:00:11,592 P660 INFO Loading data...
2020-06-14 22:00:11,592 P660 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-06-14 22:00:12,305 P660 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-14 22:00:12,305 P660 INFO Loading test data done.
2020-06-14 22:00:36,785 P660 INFO [Metrics] logloss: 0.371780 - AUC: 0.793283

```
