## ONN_criteo_x4_001

A hands-on guide to run the ONN model on the Criteo_x4_001 dataset.

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
Dataset ID: [Criteo_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [ONN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/ONN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [ONN_criteo_x4_tuner_config_03](./ONN_criteo_x4_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd ONN_criteo_x4_001
    nohup python run_expid.py --config ./ONN_criteo_x4_tuner_config_03 --expid ONN_criteo_x4_002_e3541fe7 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.437190 | 0.814772  |


### Logs
```python
2020-08-09 23:28:47,581 P587 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "embedding_dim": "8",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "ONN",
    "model_id": "ONN_criteo_x4_5c863b0f_002_313a90ff",
    "model_root": "./Criteo/ONN_criteo/min10/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.4",
    "net_regularizer": "0",
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
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-08-09 23:28:47,583 P587 INFO Set up feature encoder...
2020-08-09 23:28:47,583 P587 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-08-09 23:28:47,583 P587 INFO Loading data...
2020-08-09 23:28:47,588 P587 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-08-09 23:28:52,372 P587 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-08-09 23:28:54,189 P587 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-08-09 23:28:54,315 P587 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-08-09 23:28:54,315 P587 INFO Loading train data done.
2020-08-09 23:29:13,705 P587 INFO Start training: 3668 batches/epoch
2020-08-09 23:29:13,705 P587 INFO ************ Epoch=1 start ************
2020-08-10 02:08:56,236 P587 INFO [Metrics] logloss: 0.445034 - AUC: 0.806660
2020-08-10 02:08:56,238 P587 INFO Save best model: monitor(max): 0.361626
2020-08-10 02:08:57,906 P587 INFO --- 3668/3668 batches finished ---
2020-08-10 02:08:57,976 P587 INFO Train loss: 0.462868
2020-08-10 02:08:57,976 P587 INFO ************ Epoch=1 end ************
2020-08-10 04:48:54,593 P587 INFO [Metrics] logloss: 0.442922 - AUC: 0.808947
2020-08-10 04:48:54,594 P587 INFO Save best model: monitor(max): 0.366025
2020-08-10 04:48:56,825 P587 INFO --- 3668/3668 batches finished ---
2020-08-10 04:48:56,898 P587 INFO Train loss: 0.457419
2020-08-10 04:48:56,899 P587 INFO ************ Epoch=2 end ************
2020-08-10 07:27:36,201 P587 INFO [Metrics] logloss: 0.441819 - AUC: 0.809854
2020-08-10 07:27:36,203 P587 INFO Save best model: monitor(max): 0.368035
2020-08-10 07:27:38,525 P587 INFO --- 3668/3668 batches finished ---
2020-08-10 07:27:38,611 P587 INFO Train loss: 0.455875
2020-08-10 07:27:38,612 P587 INFO ************ Epoch=3 end ************
2020-08-10 10:06:16,180 P587 INFO [Metrics] logloss: 0.441085 - AUC: 0.810665
2020-08-10 10:06:16,182 P587 INFO Save best model: monitor(max): 0.369580
2020-08-10 10:06:18,496 P587 INFO --- 3668/3668 batches finished ---
2020-08-10 10:06:18,585 P587 INFO Train loss: 0.455244
2020-08-10 10:06:18,586 P587 INFO ************ Epoch=4 end ************
2020-08-10 12:44:34,226 P587 INFO [Metrics] logloss: 0.440756 - AUC: 0.811154
2020-08-10 12:44:34,227 P587 INFO Save best model: monitor(max): 0.370398
2020-08-10 12:44:36,355 P587 INFO --- 3668/3668 batches finished ---
2020-08-10 12:44:36,421 P587 INFO Train loss: 0.454807
2020-08-10 12:44:36,422 P587 INFO ************ Epoch=5 end ************
2020-08-10 15:22:58,895 P587 INFO [Metrics] logloss: 0.440627 - AUC: 0.811255
2020-08-10 15:22:58,896 P587 INFO Save best model: monitor(max): 0.370628
2020-08-10 15:23:01,163 P587 INFO --- 3668/3668 batches finished ---
2020-08-10 15:23:01,228 P587 INFO Train loss: 0.454600
2020-08-10 15:23:01,230 P587 INFO ************ Epoch=6 end ************
2020-08-10 18:01:23,084 P587 INFO [Metrics] logloss: 0.440299 - AUC: 0.811419
2020-08-10 18:01:23,085 P587 INFO Save best model: monitor(max): 0.371120
2020-08-10 18:01:25,621 P587 INFO --- 3668/3668 batches finished ---
2020-08-10 18:01:25,690 P587 INFO Train loss: 0.454488
2020-08-10 18:01:25,691 P587 INFO ************ Epoch=7 end ************
2020-08-10 20:39:59,414 P587 INFO [Metrics] logloss: 0.440278 - AUC: 0.811594
2020-08-10 20:39:59,415 P587 INFO Save best model: monitor(max): 0.371317
2020-08-10 20:40:01,660 P587 INFO --- 3668/3668 batches finished ---
2020-08-10 20:40:01,726 P587 INFO Train loss: 0.454390
2020-08-10 20:40:01,728 P587 INFO ************ Epoch=8 end ************
2020-08-10 23:18:01,062 P587 INFO [Metrics] logloss: 0.441116 - AUC: 0.811730
2020-08-10 23:18:01,063 P587 INFO Monitor(max) STOP: 0.370614 !
2020-08-10 23:18:01,063 P587 INFO Reduce learning rate on plateau: 0.000100
2020-08-10 23:18:01,063 P587 INFO --- 3668/3668 batches finished ---
2020-08-10 23:18:01,135 P587 INFO Train loss: 0.454356
2020-08-10 23:18:01,136 P587 INFO ************ Epoch=9 end ************
2020-08-11 01:56:33,069 P587 INFO [Metrics] logloss: 0.437683 - AUC: 0.814184
2020-08-11 01:56:33,071 P587 INFO Save best model: monitor(max): 0.376501
2020-08-11 01:56:35,316 P587 INFO --- 3668/3668 batches finished ---
2020-08-11 01:56:35,385 P587 INFO Train loss: 0.440933
2020-08-11 01:56:35,387 P587 INFO ************ Epoch=10 end ************
2020-08-11 04:34:57,841 P587 INFO [Metrics] logloss: 0.437544 - AUC: 0.814328
2020-08-11 04:34:57,842 P587 INFO Save best model: monitor(max): 0.376785
2020-08-11 04:34:59,944 P587 INFO --- 3668/3668 batches finished ---
2020-08-11 04:35:00,015 P587 INFO Train loss: 0.435983
2020-08-11 04:35:00,016 P587 INFO ************ Epoch=11 end ************
2020-08-11 07:13:25,470 P587 INFO [Metrics] logloss: 0.437974 - AUC: 0.813953
2020-08-11 07:13:25,472 P587 INFO Monitor(max) STOP: 0.375979 !
2020-08-11 07:13:25,472 P587 INFO Reduce learning rate on plateau: 0.000010
2020-08-11 07:13:25,472 P587 INFO --- 3668/3668 batches finished ---
2020-08-11 07:13:25,547 P587 INFO Train loss: 0.433218
2020-08-11 07:13:25,548 P587 INFO ************ Epoch=12 end ************
2020-08-11 09:51:36,945 P587 INFO [Metrics] logloss: 0.439130 - AUC: 0.813035
2020-08-11 09:51:36,946 P587 INFO Monitor(max) STOP: 0.373905 !
2020-08-11 09:51:36,947 P587 INFO Reduce learning rate on plateau: 0.000001
2020-08-11 09:51:36,947 P587 INFO Early stopping at epoch=13
2020-08-11 09:51:36,947 P587 INFO --- 3668/3668 batches finished ---
2020-08-11 09:51:37,023 P587 INFO Train loss: 0.427436
2020-08-11 09:51:37,024 P587 INFO Training finished.
2020-08-11 09:51:37,024 P587 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/ONN_criteo/min10/criteo_x4_5c863b0f/ONN_criteo_x4_5c863b0f_002_313a90ff_model.ckpt
2020-08-11 09:51:39,747 P587 INFO ****** Train/validation evaluation ******
2020-08-11 10:06:38,544 P587 INFO [Metrics] logloss: 0.421248 - AUC: 0.832009
2020-08-11 10:08:25,990 P587 INFO [Metrics] logloss: 0.437544 - AUC: 0.814328
2020-08-11 10:08:26,066 P587 INFO ******** Test evaluation ********
2020-08-11 10:08:26,066 P587 INFO Loading data...
2020-08-11 10:08:26,067 P587 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-08-11 10:08:27,222 P587 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-08-11 10:08:27,223 P587 INFO Loading test data done.
2020-08-11 10:10:15,026 P587 INFO [Metrics] logloss: 0.437190 - AUC: 0.814772

```
