## AutoInt_criteo_x4_001

A hands-on guide to run the AutoInt model on the Criteo_x4_001 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [AutoInt](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt_criteo_x4_tuner_config_11](./AutoInt_criteo_x4_tuner_config_11). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AutoInt_criteo_x4_001
    nohup python run_expid.py --config ./AutoInt_criteo_x4_tuner_config_11 --expid AutoInt_criteo_x4_024_e721f8ce --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.439914 | 0.811918  |


### Logs
```python
2020-07-06 02:42:45,733 P8874 INFO {
    "attention_dim": "64",
    "attention_layers": "5",
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AutoInt",
    "model_id": "AutoInt_criteo_x4_5c863b0f_024_bc8399d3",
    "model_root": "./Criteo/AutoInt_criteo/min10/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "1",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/criteo_x4_5c863b0f/test.h5",
    "train_data": "../data/Criteo/criteo_x4_5c863b0f/train.h5",
    "use_hdf5": "True",
    "use_residual": "True",
    "use_scale": "False",
    "use_wide": "False",
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-07-06 02:42:45,734 P8874 INFO Set up feature encoder...
2020-07-06 02:42:45,734 P8874 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-07-06 02:42:45,735 P8874 INFO Loading data...
2020-07-06 02:42:45,736 P8874 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-07-06 02:42:50,882 P8874 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-07-06 02:42:52,835 P8874 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-07-06 02:42:53,052 P8874 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-06 02:42:53,052 P8874 INFO Loading train data done.
2020-07-06 02:42:57,849 P8874 INFO Start training: 3668 batches/epoch
2020-07-06 02:42:57,849 P8874 INFO ************ Epoch=1 start ************
2020-07-06 02:53:54,905 P8874 INFO [Metrics] logloss: 0.451994 - AUC: 0.798588
2020-07-06 02:53:54,906 P8874 INFO Save best model: monitor(max): 0.346594
2020-07-06 02:53:54,994 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 02:53:55,043 P8874 INFO Train loss: 0.465050
2020-07-06 02:53:55,043 P8874 INFO ************ Epoch=1 end ************
2020-07-06 03:04:54,861 P8874 INFO [Metrics] logloss: 0.449976 - AUC: 0.801453
2020-07-06 03:04:54,863 P8874 INFO Save best model: monitor(max): 0.351476
2020-07-06 03:04:54,934 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 03:04:54,996 P8874 INFO Train loss: 0.459169
2020-07-06 03:04:54,996 P8874 INFO ************ Epoch=2 end ************
2020-07-06 03:15:54,253 P8874 INFO [Metrics] logloss: 0.448047 - AUC: 0.802946
2020-07-06 03:15:54,254 P8874 INFO Save best model: monitor(max): 0.354900
2020-07-06 03:15:54,323 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 03:15:54,376 P8874 INFO Train loss: 0.457341
2020-07-06 03:15:54,376 P8874 INFO ************ Epoch=3 end ************
2020-07-06 03:26:53,364 P8874 INFO [Metrics] logloss: 0.447112 - AUC: 0.803969
2020-07-06 03:26:53,366 P8874 INFO Save best model: monitor(max): 0.356857
2020-07-06 03:26:53,435 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 03:26:53,486 P8874 INFO Train loss: 0.456309
2020-07-06 03:26:53,487 P8874 INFO ************ Epoch=4 end ************
2020-07-06 03:37:53,030 P8874 INFO [Metrics] logloss: 0.446800 - AUC: 0.804523
2020-07-06 03:37:53,031 P8874 INFO Save best model: monitor(max): 0.357723
2020-07-06 03:37:53,103 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 03:37:53,158 P8874 INFO Train loss: 0.455488
2020-07-06 03:37:53,158 P8874 INFO ************ Epoch=5 end ************
2020-07-06 03:49:02,979 P8874 INFO [Metrics] logloss: 0.446036 - AUC: 0.805222
2020-07-06 03:49:02,980 P8874 INFO Save best model: monitor(max): 0.359186
2020-07-06 03:49:03,052 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 03:49:03,116 P8874 INFO Train loss: 0.454821
2020-07-06 03:49:03,116 P8874 INFO ************ Epoch=6 end ************
2020-07-06 04:00:08,253 P8874 INFO [Metrics] logloss: 0.445419 - AUC: 0.805807
2020-07-06 04:00:08,255 P8874 INFO Save best model: monitor(max): 0.360388
2020-07-06 04:00:08,328 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 04:00:08,389 P8874 INFO Train loss: 0.454231
2020-07-06 04:00:08,390 P8874 INFO ************ Epoch=7 end ************
2020-07-06 04:11:04,835 P8874 INFO [Metrics] logloss: 0.445130 - AUC: 0.806134
2020-07-06 04:11:04,836 P8874 INFO Save best model: monitor(max): 0.361003
2020-07-06 04:11:04,907 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 04:11:04,961 P8874 INFO Train loss: 0.453733
2020-07-06 04:11:04,961 P8874 INFO ************ Epoch=8 end ************
2020-07-06 04:22:01,224 P8874 INFO [Metrics] logloss: 0.444796 - AUC: 0.806535
2020-07-06 04:22:01,226 P8874 INFO Save best model: monitor(max): 0.361738
2020-07-06 04:22:01,292 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 04:22:01,344 P8874 INFO Train loss: 0.453315
2020-07-06 04:22:01,344 P8874 INFO ************ Epoch=9 end ************
2020-07-06 04:32:57,472 P8874 INFO [Metrics] logloss: 0.444686 - AUC: 0.806669
2020-07-06 04:32:57,473 P8874 INFO Save best model: monitor(max): 0.361983
2020-07-06 04:32:57,540 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 04:32:57,592 P8874 INFO Train loss: 0.452976
2020-07-06 04:32:57,592 P8874 INFO ************ Epoch=10 end ************
2020-07-06 04:43:53,913 P8874 INFO [Metrics] logloss: 0.444554 - AUC: 0.807046
2020-07-06 04:43:53,914 P8874 INFO Save best model: monitor(max): 0.362492
2020-07-06 04:43:53,981 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 04:43:54,036 P8874 INFO Train loss: 0.452664
2020-07-06 04:43:54,036 P8874 INFO ************ Epoch=11 end ************
2020-07-06 04:54:47,618 P8874 INFO [Metrics] logloss: 0.444414 - AUC: 0.807055
2020-07-06 04:54:47,619 P8874 INFO Save best model: monitor(max): 0.362642
2020-07-06 04:54:47,685 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 04:54:47,739 P8874 INFO Train loss: 0.452451
2020-07-06 04:54:47,739 P8874 INFO ************ Epoch=12 end ************
2020-07-06 05:05:43,379 P8874 INFO [Metrics] logloss: 0.443987 - AUC: 0.807439
2020-07-06 05:05:43,380 P8874 INFO Save best model: monitor(max): 0.363453
2020-07-06 05:05:43,447 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 05:05:43,504 P8874 INFO Train loss: 0.452260
2020-07-06 05:05:43,505 P8874 INFO ************ Epoch=13 end ************
2020-07-06 05:16:49,493 P8874 INFO [Metrics] logloss: 0.443976 - AUC: 0.807482
2020-07-06 05:16:49,494 P8874 INFO Save best model: monitor(max): 0.363506
2020-07-06 05:16:49,574 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 05:16:49,637 P8874 INFO Train loss: 0.452102
2020-07-06 05:16:49,637 P8874 INFO ************ Epoch=14 end ************
2020-07-06 05:27:56,976 P8874 INFO [Metrics] logloss: 0.443789 - AUC: 0.807705
2020-07-06 05:27:56,978 P8874 INFO Save best model: monitor(max): 0.363916
2020-07-06 05:27:57,049 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 05:27:57,105 P8874 INFO Train loss: 0.451964
2020-07-06 05:27:57,105 P8874 INFO ************ Epoch=15 end ************
2020-07-06 05:39:02,467 P8874 INFO [Metrics] logloss: 0.443646 - AUC: 0.807709
2020-07-06 05:39:02,468 P8874 INFO Save best model: monitor(max): 0.364063
2020-07-06 05:39:02,538 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 05:39:02,592 P8874 INFO Train loss: 0.451842
2020-07-06 05:39:02,592 P8874 INFO ************ Epoch=16 end ************
2020-07-06 05:50:05,719 P8874 INFO [Metrics] logloss: 0.443764 - AUC: 0.807943
2020-07-06 05:50:05,721 P8874 INFO Save best model: monitor(max): 0.364179
2020-07-06 05:50:05,799 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 05:50:05,861 P8874 INFO Train loss: 0.451744
2020-07-06 05:50:05,861 P8874 INFO ************ Epoch=17 end ************
2020-07-06 06:01:09,551 P8874 INFO [Metrics] logloss: 0.443579 - AUC: 0.807910
2020-07-06 06:01:09,552 P8874 INFO Save best model: monitor(max): 0.364332
2020-07-06 06:01:09,624 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 06:01:09,687 P8874 INFO Train loss: 0.451642
2020-07-06 06:01:09,687 P8874 INFO ************ Epoch=18 end ************
2020-07-06 06:12:10,030 P8874 INFO [Metrics] logloss: 0.443335 - AUC: 0.808107
2020-07-06 06:12:10,031 P8874 INFO Save best model: monitor(max): 0.364772
2020-07-06 06:12:10,101 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 06:12:10,154 P8874 INFO Train loss: 0.451567
2020-07-06 06:12:10,155 P8874 INFO ************ Epoch=19 end ************
2020-07-06 06:23:13,444 P8874 INFO [Metrics] logloss: 0.443503 - AUC: 0.807966
2020-07-06 06:23:13,446 P8874 INFO Monitor(max) STOP: 0.364463 !
2020-07-06 06:23:13,446 P8874 INFO Reduce learning rate on plateau: 0.000100
2020-07-06 06:23:13,446 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 06:23:13,499 P8874 INFO Train loss: 0.451485
2020-07-06 06:23:13,499 P8874 INFO ************ Epoch=20 end ************
2020-07-06 06:34:13,568 P8874 INFO [Metrics] logloss: 0.440639 - AUC: 0.811105
2020-07-06 06:34:13,569 P8874 INFO Save best model: monitor(max): 0.370466
2020-07-06 06:34:13,637 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 06:34:13,711 P8874 INFO Train loss: 0.442984
2020-07-06 06:34:13,712 P8874 INFO ************ Epoch=21 end ************
2020-07-06 06:45:11,523 P8874 INFO [Metrics] logloss: 0.440339 - AUC: 0.811450
2020-07-06 06:45:11,524 P8874 INFO Save best model: monitor(max): 0.371111
2020-07-06 06:45:11,590 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 06:45:11,642 P8874 INFO Train loss: 0.439376
2020-07-06 06:45:11,643 P8874 INFO ************ Epoch=22 end ************
2020-07-06 06:56:06,376 P8874 INFO [Metrics] logloss: 0.440473 - AUC: 0.811317
2020-07-06 06:56:06,377 P8874 INFO Monitor(max) STOP: 0.370844 !
2020-07-06 06:56:06,377 P8874 INFO Reduce learning rate on plateau: 0.000010
2020-07-06 06:56:06,377 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 06:56:06,429 P8874 INFO Train loss: 0.437413
2020-07-06 06:56:06,429 P8874 INFO ************ Epoch=23 end ************
2020-07-06 07:07:00,013 P8874 INFO [Metrics] logloss: 0.441576 - AUC: 0.810606
2020-07-06 07:07:00,013 P8874 INFO Monitor(max) STOP: 0.369029 !
2020-07-06 07:07:00,014 P8874 INFO Reduce learning rate on plateau: 0.000001
2020-07-06 07:07:00,014 P8874 INFO Early stopping at epoch=24
2020-07-06 07:07:00,014 P8874 INFO --- 3668/3668 batches finished ---
2020-07-06 07:07:00,065 P8874 INFO Train loss: 0.432769
2020-07-06 07:07:00,065 P8874 INFO Training finished.
2020-07-06 07:07:00,065 P8874 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/AutoInt_criteo/min10/criteo_x4_5c863b0f/AutoInt_criteo_x4_5c863b0f_024_bc8399d3_model.ckpt
2020-07-06 07:07:00,279 P8874 INFO ****** Train/validation evaluation ******
2020-07-06 07:10:39,286 P8874 INFO [Metrics] logloss: 0.429500 - AUC: 0.822950
2020-07-06 07:11:06,950 P8874 INFO [Metrics] logloss: 0.440339 - AUC: 0.811450
2020-07-06 07:11:07,028 P8874 INFO ******** Test evaluation ********
2020-07-06 07:11:07,029 P8874 INFO Loading data...
2020-07-06 07:11:07,029 P8874 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-07-06 07:11:08,019 P8874 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-06 07:11:08,019 P8874 INFO Loading test data done.
2020-07-06 07:11:33,342 P8874 INFO [Metrics] logloss: 0.439914 - AUC: 0.811918

```
