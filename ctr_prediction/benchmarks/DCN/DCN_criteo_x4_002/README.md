## DCN_criteo_x4_002

A hands-on guide to run the DCN model on the Criteo_x4_002 dataset.

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
Dataset ID: [Criteo_x4_002](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x4_002). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [DCN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DCN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DCN_criteo_x4_tuner_config_04](./DCN_criteo_x4_tuner_config_04). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DCN_criteo_x4_002
    nohup python run_expid.py --config ./DCN_criteo_x4_tuner_config_04 --expid DCN_criteo_x4_001_0a86fe97 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.437832 | 0.814073  |


### Logs
```python
2020-02-01 15:47:32,650 P601 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "crossing_layers": "6",
    "dataset_id": "criteo_x4_001_be98441d",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[2000, 2000, 2000, 2000, 2000, 2000]",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "l2(1.e-5)",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DCN",
    "model_id": "DCN_criteo_x4_001_93ce629f",
    "model_root": "./Criteo/DCN_criteo/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "3",
    "pickle_feature_encoder": "True",
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
    "gpu": "5"
}
2020-02-01 15:47:32,656 P601 INFO Set up feature encoder...
2020-02-01 15:47:32,657 P601 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-02-01 15:47:32,657 P601 INFO Loading data...
2020-02-01 15:47:32,662 P601 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-02-01 15:47:37,728 P601 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-02-01 15:47:39,898 P601 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-02-01 15:47:40,118 P601 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-01 15:47:40,118 P601 INFO Loading train data done.
2020-02-01 15:47:54,349 P601 INFO **** Start training: 3668 batches/epoch ****
2020-02-01 16:02:10,085 P601 INFO [Metrics] logloss: 0.445910 - AUC: 0.805306
2020-02-01 16:02:10,182 P601 INFO Save best model: monitor(max): 0.359396
2020-02-01 16:02:12,058 P601 INFO --- 3668/3668 batches finished ---
2020-02-01 16:02:12,119 P601 INFO Train loss: 0.466411
2020-02-01 16:02:12,119 P601 INFO ************ Epoch=1 end ************
2020-02-01 16:16:22,021 P601 INFO [Metrics] logloss: 0.444249 - AUC: 0.807624
2020-02-01 16:16:22,079 P601 INFO Save best model: monitor(max): 0.363375
2020-02-01 16:16:24,092 P601 INFO --- 3668/3668 batches finished ---
2020-02-01 16:16:24,183 P601 INFO Train loss: 0.460030
2020-02-01 16:16:24,183 P601 INFO ************ Epoch=2 end ************
2020-02-01 16:30:29,945 P601 INFO [Metrics] logloss: 0.442744 - AUC: 0.808852
2020-02-01 16:30:30,003 P601 INFO Save best model: monitor(max): 0.366108
2020-02-01 16:30:31,988 P601 INFO --- 3668/3668 batches finished ---
2020-02-01 16:30:32,056 P601 INFO Train loss: 0.458642
2020-02-01 16:30:32,056 P601 INFO ************ Epoch=3 end ************
2020-02-01 16:44:36,627 P601 INFO [Metrics] logloss: 0.442649 - AUC: 0.809457
2020-02-01 16:44:36,686 P601 INFO Save best model: monitor(max): 0.366809
2020-02-01 16:44:38,610 P601 INFO --- 3668/3668 batches finished ---
2020-02-01 16:44:38,671 P601 INFO Train loss: 0.458100
2020-02-01 16:44:38,671 P601 INFO ************ Epoch=4 end ************
2020-02-01 16:58:43,551 P601 INFO [Metrics] logloss: 0.442050 - AUC: 0.809604
2020-02-01 16:58:43,608 P601 INFO Save best model: monitor(max): 0.367554
2020-02-01 16:58:45,511 P601 INFO --- 3668/3668 batches finished ---
2020-02-01 16:58:45,573 P601 INFO Train loss: 0.457774
2020-02-01 16:58:45,573 P601 INFO ************ Epoch=5 end ************
2020-02-01 17:12:52,835 P601 INFO [Metrics] logloss: 0.441877 - AUC: 0.809939
2020-02-01 17:12:52,892 P601 INFO Save best model: monitor(max): 0.368062
2020-02-01 17:12:54,898 P601 INFO --- 3668/3668 batches finished ---
2020-02-01 17:12:54,961 P601 INFO Train loss: 0.457532
2020-02-01 17:12:54,961 P601 INFO ************ Epoch=6 end ************
2020-02-01 17:27:02,598 P601 INFO [Metrics] logloss: 0.442046 - AUC: 0.809920
2020-02-01 17:27:02,657 P601 INFO Monitor(max) STOP: 0.367874 !
2020-02-01 17:27:02,658 P601 INFO Reduce learning rate on plateau: 0.000100
2020-02-01 17:27:02,658 P601 INFO --- 3668/3668 batches finished ---
2020-02-01 17:27:02,718 P601 INFO Train loss: 0.457455
2020-02-01 17:27:02,718 P601 INFO ************ Epoch=7 end ************
2020-02-01 17:41:11,867 P601 INFO [Metrics] logloss: 0.438543 - AUC: 0.813283
2020-02-01 17:41:11,926 P601 INFO Save best model: monitor(max): 0.374741
2020-02-01 17:41:13,880 P601 INFO --- 3668/3668 batches finished ---
2020-02-01 17:41:13,947 P601 INFO Train loss: 0.439910
2020-02-01 17:41:13,948 P601 INFO ************ Epoch=8 end ************
2020-02-01 17:55:20,964 P601 INFO [Metrics] logloss: 0.438299 - AUC: 0.813539
2020-02-01 17:55:21,020 P601 INFO Save best model: monitor(max): 0.375240
2020-02-01 17:55:23,015 P601 INFO --- 3668/3668 batches finished ---
2020-02-01 17:55:23,083 P601 INFO Train loss: 0.433878
2020-02-01 17:55:23,084 P601 INFO ************ Epoch=9 end ************
2020-02-01 18:09:31,382 P601 INFO [Metrics] logloss: 0.438639 - AUC: 0.813256
2020-02-01 18:09:31,476 P601 INFO Monitor(max) STOP: 0.374617 !
2020-02-01 18:09:31,477 P601 INFO Reduce learning rate on plateau: 0.000010
2020-02-01 18:09:31,477 P601 INFO --- 3668/3668 batches finished ---
2020-02-01 18:09:31,548 P601 INFO Train loss: 0.430699
2020-02-01 18:09:31,548 P601 INFO ************ Epoch=10 end ************
2020-02-01 18:23:40,001 P601 INFO [Metrics] logloss: 0.439557 - AUC: 0.812590
2020-02-01 18:23:40,086 P601 INFO Monitor(max) STOP: 0.373034 !
2020-02-01 18:23:40,086 P601 INFO Reduce learning rate on plateau: 0.000001
2020-02-01 18:23:40,086 P601 INFO --- 3668/3668 batches finished ---
2020-02-01 18:23:40,171 P601 INFO Train loss: 0.423819
2020-02-01 18:23:40,171 P601 INFO ************ Epoch=11 end ************
2020-02-01 18:37:46,956 P601 INFO [Metrics] logloss: 0.439701 - AUC: 0.812441
2020-02-01 18:37:47,014 P601 INFO Monitor(max) STOP: 0.372740 !
2020-02-01 18:37:47,014 P601 INFO Reduce learning rate on plateau: 0.000001
2020-02-01 18:37:47,014 P601 INFO Early stopping at epoch=12
2020-02-01 18:37:47,014 P601 INFO --- 3668/3668 batches finished ---
2020-02-01 18:37:47,086 P601 INFO Train loss: 0.422349
2020-02-01 18:37:47,086 P601 INFO Training finished.
2020-02-01 18:37:47,086 P601 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/DCN_criteo/criteo_x4_001_be98441d/DCN_criteo_x4_001_93ce629f_criteo_x4_001_be98441d_model.ckpt
2020-02-01 18:37:48,870 P601 INFO ****** Train/validation evaluation ******
2020-02-01 18:42:42,785 P601 INFO [Metrics] logloss: 0.419981 - AUC: 0.833309
2020-02-01 18:43:19,555 P601 INFO [Metrics] logloss: 0.438299 - AUC: 0.813539
2020-02-01 18:43:19,744 P601 INFO ******** Test evaluation ********
2020-02-01 18:43:19,744 P601 INFO Loading data...
2020-02-01 18:43:19,744 P601 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-02-01 18:43:20,721 P601 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-01 18:43:20,722 P601 INFO Loading test data done.
2020-02-01 18:43:54,981 P601 INFO [Metrics] logloss: 0.437832 - AUC: 0.814073

```
