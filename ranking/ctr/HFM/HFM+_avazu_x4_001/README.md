## HFM+_avazu_x4_001

A hands-on guide to run the HFM model on the Avazu_x4_001 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [HFM](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/HFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HFM+_avazu_x4_tuner_config_12](./HFM+_avazu_x4_tuner_config_12). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HFM+_avazu_x4_001
    nohup python run_expid.py --config ./HFM+_avazu_x4_tuner_config_12 --expid HFM_avazu_x4_005_8f97f15f --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.371402 | 0.794399  |


### Logs
```python
2020-07-18 10:19:44,523 P32560 INFO {
    "batch_norm": "False",
    "batch_size": "5000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[500, 500]",
    "interaction_type": "circular_correlation",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "HFM",
    "model_id": "HFM_avazu_x4_3bbbc4c9_005_5c1c93b7",
    "model_root": "./Avazu/HFM_avazu/min2/",
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
    "use_dnn": "True",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/avazu_x4_3bbbc4c9/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-18 10:19:44,524 P32560 INFO Set up feature encoder...
2020-07-18 10:19:44,524 P32560 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-07-18 10:19:44,524 P32560 INFO Loading data...
2020-07-18 10:19:44,526 P32560 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-07-18 10:19:47,137 P32560 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-07-18 10:19:48,399 P32560 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-07-18 10:19:48,511 P32560 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-07-18 10:19:48,511 P32560 INFO Loading train data done.
2020-07-18 10:19:52,743 P32560 INFO **** Start training: 6469 batches/epoch ****
2020-07-18 10:38:05,916 P32560 INFO [Metrics] logloss: 0.371482 - AUC: 0.794246
2020-07-18 10:38:05,920 P32560 INFO Save best model: monitor(max): 0.422764
2020-07-18 10:38:06,174 P32560 INFO --- 6469/6469 batches finished ---
2020-07-18 10:38:06,223 P32560 INFO Train loss: 0.378967
2020-07-18 10:38:06,223 P32560 INFO ************ Epoch=1 end ************
2020-07-18 10:56:17,990 P32560 INFO [Metrics] logloss: 0.375630 - AUC: 0.791751
2020-07-18 10:56:17,991 P32560 INFO Monitor(max) STOP: 0.416121 !
2020-07-18 10:56:17,991 P32560 INFO Reduce learning rate on plateau: 0.000100
2020-07-18 10:56:17,991 P32560 INFO --- 6469/6469 batches finished ---
2020-07-18 10:56:18,040 P32560 INFO Train loss: 0.331182
2020-07-18 10:56:18,040 P32560 INFO ************ Epoch=2 end ************
2020-07-18 11:14:29,967 P32560 INFO [Metrics] logloss: 0.416799 - AUC: 0.776919
2020-07-18 11:14:29,968 P32560 INFO Monitor(max) STOP: 0.360120 !
2020-07-18 11:14:29,968 P32560 INFO Reduce learning rate on plateau: 0.000010
2020-07-18 11:14:29,968 P32560 INFO Early stopping at epoch=3
2020-07-18 11:14:29,968 P32560 INFO --- 6469/6469 batches finished ---
2020-07-18 11:14:30,018 P32560 INFO Train loss: 0.285853
2020-07-18 11:14:30,018 P32560 INFO Training finished.
2020-07-18 11:14:30,018 P32560 INFO Load best model: /home/XXX/benchmarks/Avazu/HFM_avazu/min2/avazu_x4_3bbbc4c9/HFM_avazu_x4_3bbbc4c9_005_5c1c93b7_model.ckpt
2020-07-18 11:14:30,353 P32560 INFO ****** Train/validation evaluation ******
2020-07-18 11:15:05,068 P32560 INFO [Metrics] logloss: 0.371482 - AUC: 0.794246
2020-07-18 11:15:05,195 P32560 INFO ******** Test evaluation ********
2020-07-18 11:15:05,195 P32560 INFO Loading data...
2020-07-18 11:15:05,195 P32560 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-07-18 11:15:05,639 P32560 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-07-18 11:15:05,639 P32560 INFO Loading test data done.
2020-07-18 11:15:40,068 P32560 INFO [Metrics] logloss: 0.371402 - AUC: 0.794399

```
