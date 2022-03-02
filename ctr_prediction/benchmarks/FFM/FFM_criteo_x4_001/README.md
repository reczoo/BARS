## FFM_criteo_x4_001

A hands-on guide to run the FFM model on the Criteo_x4_001 dataset.

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

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [FFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FFM_criteo_x4_tuner_config_01](./FFM_criteo_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FFM_criteo_x4_001
    nohup python run_expid.py --config ./FFM_criteo_x4_tuner_config_01 --expid FFM_criteo_x4_010_d090ef3d --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.440745 | 0.811263  |


### Logs
```python
2020-06-26 23:58:44,499 P2560 INFO {
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "embedding_dim": "4",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "3",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FFM",
    "model_id": "FFM_criteo_x4_5c863b0f_010_6f65737a",
    "model_root": "./Criteo/FFM_criteo/min10/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-06",
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
2020-06-26 23:58:44,507 P2560 INFO Set up feature encoder...
2020-06-26 23:58:44,507 P2560 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-06-26 23:58:44,507 P2560 INFO Loading data...
2020-06-26 23:58:44,517 P2560 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-06-26 23:58:51,055 P2560 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-06-26 23:58:52,878 P2560 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-06-26 23:58:53,002 P2560 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-26 23:58:53,002 P2560 INFO Loading train data done.
2020-06-26 23:59:00,484 P2560 INFO Start training: 3668 batches/epoch
2020-06-26 23:59:00,484 P2560 INFO ************ Epoch=1 start ************
2020-06-27 02:15:48,526 P2560 INFO [Metrics] logloss: 0.443578 - AUC: 0.807967
2020-06-27 02:15:48,528 P2560 INFO Save best model: monitor(max): 0.364389
2020-06-27 02:15:49,335 P2560 INFO --- 3668/3668 batches finished ---
2020-06-27 02:15:49,383 P2560 INFO Train loss: 0.457038
2020-06-27 02:15:49,384 P2560 INFO ************ Epoch=1 end ************
2020-06-27 04:32:28,946 P2560 INFO [Metrics] logloss: 0.442052 - AUC: 0.809687
2020-06-27 04:32:28,948 P2560 INFO Save best model: monitor(max): 0.367634
2020-06-27 04:32:30,044 P2560 INFO --- 3668/3668 batches finished ---
2020-06-27 04:32:30,108 P2560 INFO Train loss: 0.450131
2020-06-27 04:32:30,108 P2560 INFO ************ Epoch=2 end ************
2020-06-27 06:48:38,069 P2560 INFO [Metrics] logloss: 0.441775 - AUC: 0.809998
2020-06-27 06:48:38,070 P2560 INFO Save best model: monitor(max): 0.368223
2020-06-27 06:48:39,250 P2560 INFO --- 3668/3668 batches finished ---
2020-06-27 06:48:39,308 P2560 INFO Train loss: 0.448668
2020-06-27 06:48:39,309 P2560 INFO ************ Epoch=3 end ************
2020-06-27 09:04:36,430 P2560 INFO [Metrics] logloss: 0.442046 - AUC: 0.809730
2020-06-27 09:04:36,431 P2560 INFO Monitor(max) STOP: 0.367684 !
2020-06-27 09:04:36,432 P2560 INFO Reduce learning rate on plateau: 0.000100
2020-06-27 09:04:36,432 P2560 INFO --- 3668/3668 batches finished ---
2020-06-27 09:04:36,495 P2560 INFO Train loss: 0.447682
2020-06-27 09:04:36,495 P2560 INFO ************ Epoch=4 end ************
2020-06-27 11:21:23,020 P2560 INFO [Metrics] logloss: 0.441177 - AUC: 0.810739
2020-06-27 11:21:23,027 P2560 INFO Save best model: monitor(max): 0.369561
2020-06-27 11:21:24,149 P2560 INFO --- 3668/3668 batches finished ---
2020-06-27 11:21:24,203 P2560 INFO Train loss: 0.432505
2020-06-27 11:21:24,204 P2560 INFO ************ Epoch=5 end ************
2020-06-27 13:38:04,763 P2560 INFO [Metrics] logloss: 0.441768 - AUC: 0.810203
2020-06-27 13:38:04,763 P2560 INFO Monitor(max) STOP: 0.368435 !
2020-06-27 13:38:04,764 P2560 INFO Reduce learning rate on plateau: 0.000010
2020-06-27 13:38:04,764 P2560 INFO --- 3668/3668 batches finished ---
2020-06-27 13:38:04,811 P2560 INFO Train loss: 0.428703
2020-06-27 13:38:04,812 P2560 INFO ************ Epoch=6 end ************
2020-06-27 15:54:21,631 P2560 INFO [Metrics] logloss: 0.441821 - AUC: 0.810172
2020-06-27 15:54:21,632 P2560 INFO Monitor(max) STOP: 0.368351 !
2020-06-27 15:54:21,632 P2560 INFO Reduce learning rate on plateau: 0.000001
2020-06-27 15:54:21,632 P2560 INFO Early stopping at epoch=7
2020-06-27 15:54:21,632 P2560 INFO --- 3668/3668 batches finished ---
2020-06-27 15:54:21,691 P2560 INFO Train loss: 0.424549
2020-06-27 15:54:21,691 P2560 INFO Training finished.
2020-06-27 15:54:21,691 P2560 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/FFM_criteo/min10/criteo_x4_5c863b0f/FFM_criteo_x4_5c863b0f_010_6f65737a_model.ckpt
2020-06-27 15:54:23,795 P2560 INFO ****** Train/validation evaluation ******
2020-06-27 16:08:12,533 P2560 INFO [Metrics] logloss: 0.416780 - AUC: 0.836701
2020-06-27 16:09:52,183 P2560 INFO [Metrics] logloss: 0.441177 - AUC: 0.810739
2020-06-27 16:09:52,250 P2560 INFO ******** Test evaluation ********
2020-06-27 16:09:52,250 P2560 INFO Loading data...
2020-06-27 16:09:52,250 P2560 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-06-27 16:09:53,287 P2560 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-27 16:09:53,287 P2560 INFO Loading test data done.
2020-06-27 16:11:33,352 P2560 INFO [Metrics] logloss: 0.440745 - AUC: 0.811263

```
