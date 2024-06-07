## FiBiNET_criteo_x4_002

A hands-on guide to run the FiBiNET model on the Criteo_x4_002 dataset.

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
Dataset ID: [Criteo_x4](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x4). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FiBiNET](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FiBiNET.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FiBiNET_criteo_x4_tuner_config_03](./FiBiNET_criteo_x4_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FiBiNET_criteo_x4_002
    nohup python run_expid.py --config ./FiBiNET_criteo_x4_tuner_config_03 --expid FiBiNET_criteo_x4_012_faef5b65 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.438573 | 0.813350  |


### Logs
```python
2020-02-27 23:21:30,831 P1862 INFO {
    "batch_norm": "False",
    "batch_size": "5000",
    "bilinear_type": "field_interaction",
    "dataset_id": "criteo_x4_001_be98441d",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "l2(1.e-6)",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[4096, 2048, 1024, 512]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FiBiNET",
    "model_id": "FiBiNET_criteo_x4_012_90314955",
    "model_root": "./Criteo/FiBiNET_criteo/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "3",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "test_data": "../data/Criteo/criteo_x4_001_be98441d/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_be98441d/train.h5",
    "valid_data": "../data/Criteo/criteo_x4_001_be98441d/valid.h5",
    "version": "pytorch",
    "gpu": "2"
}
2020-02-27 23:21:30,834 P1862 INFO Set up feature encoder...
2020-02-27 23:21:30,834 P1862 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-02-27 23:21:30,834 P1862 INFO Loading data...
2020-02-27 23:21:30,838 P1862 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-02-27 23:21:36,939 P1862 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-02-27 23:21:38,683 P1862 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-02-27 23:21:38,809 P1862 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-27 23:21:38,809 P1862 INFO Loading train data done.
2020-02-27 23:21:54,025 P1862 INFO **** Start training: 7335 batches/epoch ****
2020-02-28 01:43:27,077 P1862 INFO [Metrics] logloss: 0.440543 - AUC: 0.811105
2020-02-28 01:43:27,187 P1862 INFO Save best model: monitor(max): 0.370562
2020-02-28 01:43:30,186 P1862 INFO --- 7335/7335 batches finished ---
2020-02-28 01:43:30,247 P1862 INFO Train loss: 0.455204
2020-02-28 01:43:30,250 P1862 INFO ************ Epoch=1 end ************
2020-02-28 04:04:20,459 P1862 INFO [Metrics] logloss: 0.438986 - AUC: 0.812933
2020-02-28 04:04:20,563 P1862 INFO Save best model: monitor(max): 0.373946
2020-02-28 04:04:24,632 P1862 INFO --- 7335/7335 batches finished ---
2020-02-28 04:04:24,709 P1862 INFO Train loss: 0.448318
2020-02-28 04:04:24,711 P1862 INFO ************ Epoch=2 end ************
2020-02-28 06:24:23,362 P1862 INFO [Metrics] logloss: 0.443736 - AUC: 0.809294
2020-02-28 06:24:23,437 P1862 INFO Monitor(max) STOP: 0.365557 !
2020-02-28 06:24:23,438 P1862 INFO Reduce learning rate on plateau: 0.000100
2020-02-28 06:24:23,438 P1862 INFO --- 7335/7335 batches finished ---
2020-02-28 06:24:23,513 P1862 INFO Train loss: 0.442901
2020-02-28 06:24:23,514 P1862 INFO ************ Epoch=3 end ************
2020-02-28 08:44:06,302 P1862 INFO [Metrics] logloss: 0.532712 - AUC: 0.772564
2020-02-28 08:44:06,376 P1862 INFO Monitor(max) STOP: 0.239852 !
2020-02-28 08:44:06,376 P1862 INFO Reduce learning rate on plateau: 0.000010
2020-02-28 08:44:06,376 P1862 INFO Early stopping at epoch=4
2020-02-28 08:44:06,376 P1862 INFO --- 7335/7335 batches finished ---
2020-02-28 08:44:06,453 P1862 INFO Train loss: 0.348021
2020-02-28 08:44:06,454 P1862 INFO Training finished.
2020-02-28 08:44:06,454 P1862 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/FiBiNET_criteo/criteo_x4_001_be98441d/FiBiNET_criteo_x4_012_90314955_criteo_x4_001_be98441d_model.ckpt
2020-02-28 08:44:09,581 P1862 INFO ****** Train/validation evaluation ******
2020-02-28 09:30:14,627 P1862 INFO [Metrics] logloss: 0.420199 - AUC: 0.832668
2020-02-28 09:35:55,150 P1862 INFO [Metrics] logloss: 0.438986 - AUC: 0.812933
2020-02-28 09:35:55,339 P1862 INFO ******** Test evaluation ********
2020-02-28 09:35:55,339 P1862 INFO Loading data...
2020-02-28 09:35:55,340 P1862 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-02-28 09:35:56,258 P1862 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-28 09:35:56,258 P1862 INFO Loading test data done.
2020-02-28 09:41:36,705 P1862 INFO [Metrics] logloss: 0.438573 - AUC: 0.813350

```
