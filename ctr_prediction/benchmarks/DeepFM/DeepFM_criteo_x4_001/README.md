## DeepFM_criteo_x4_001

A hands-on guide to run the DeepFM model on the Criteo_x4_001 dataset.

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

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [DeepFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DeepFM_criteo_x4_tuner_config_01](./DeepFM_criteo_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DeepFM_criteo_x4_001
    nohup python run_expid.py --config ./DeepFM_criteo_x4_tuner_config_01 --expid DeepFM_criteo_x4_024_626165ea --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.437584 | 0.814332  |


### Logs
```python
2020-06-23 11:16:27,135 P6804 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "3",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000, 1000, 1000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DeepFM",
    "model_id": "DeepFM_criteo_x4_5c863b0f_024_c15d5b96",
    "model_root": "./Criteo/DeepFM_criteo/min10/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.2",
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
2020-06-23 11:16:27,137 P6804 INFO Set up feature encoder...
2020-06-23 11:16:27,137 P6804 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-06-23 11:16:27,137 P6804 INFO Loading data...
2020-06-23 11:16:27,145 P6804 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-06-23 11:16:34,331 P6804 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-06-23 11:16:36,796 P6804 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-06-23 11:16:37,005 P6804 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-23 11:16:37,005 P6804 INFO Loading train data done.
2020-06-23 11:16:40,767 P6804 INFO Start training: 3668 batches/epoch
2020-06-23 11:16:40,767 P6804 INFO ************ Epoch=1 start ************
2020-06-23 11:26:02,032 P6804 INFO [Metrics] logloss: 0.446031 - AUC: 0.805445
2020-06-23 11:26:02,035 P6804 INFO Save best model: monitor(max): 0.359414
2020-06-23 11:26:02,281 P6804 INFO --- 3668/3668 batches finished ---
2020-06-23 11:26:02,344 P6804 INFO Train loss: 0.459945
2020-06-23 11:26:02,344 P6804 INFO ************ Epoch=1 end ************
2020-06-23 11:35:19,939 P6804 INFO [Metrics] logloss: 0.444145 - AUC: 0.807468
2020-06-23 11:35:19,940 P6804 INFO Save best model: monitor(max): 0.363323
2020-06-23 11:35:20,073 P6804 INFO --- 3668/3668 batches finished ---
2020-06-23 11:35:20,111 P6804 INFO Train loss: 0.455084
2020-06-23 11:35:20,112 P6804 INFO ************ Epoch=2 end ************
2020-06-23 11:44:30,984 P6804 INFO [Metrics] logloss: 0.442921 - AUC: 0.808500
2020-06-23 11:44:30,985 P6804 INFO Save best model: monitor(max): 0.365580
2020-06-23 11:44:31,080 P6804 INFO --- 3668/3668 batches finished ---
2020-06-23 11:44:31,128 P6804 INFO Train loss: 0.453792
2020-06-23 11:44:31,129 P6804 INFO ************ Epoch=3 end ************
2020-06-23 11:53:40,224 P6804 INFO [Metrics] logloss: 0.442192 - AUC: 0.809317
2020-06-23 11:53:40,225 P6804 INFO Save best model: monitor(max): 0.367125
2020-06-23 11:53:40,316 P6804 INFO --- 3668/3668 batches finished ---
2020-06-23 11:53:40,360 P6804 INFO Train loss: 0.453088
2020-06-23 11:53:40,360 P6804 INFO ************ Epoch=4 end ************
2020-06-23 12:02:50,649 P6804 INFO [Metrics] logloss: 0.442291 - AUC: 0.809407
2020-06-23 12:02:50,650 P6804 INFO Monitor(max) STOP: 0.367116 !
2020-06-23 12:02:50,651 P6804 INFO Reduce learning rate on plateau: 0.000100
2020-06-23 12:02:50,651 P6804 INFO --- 3668/3668 batches finished ---
2020-06-23 12:02:50,688 P6804 INFO Train loss: 0.452561
2020-06-23 12:02:50,689 P6804 INFO ************ Epoch=5 end ************
2020-06-23 12:12:00,176 P6804 INFO [Metrics] logloss: 0.438483 - AUC: 0.813303
2020-06-23 12:12:00,177 P6804 INFO Save best model: monitor(max): 0.374820
2020-06-23 12:12:00,265 P6804 INFO --- 3668/3668 batches finished ---
2020-06-23 12:12:00,303 P6804 INFO Train loss: 0.441384
2020-06-23 12:12:00,304 P6804 INFO ************ Epoch=6 end ************
2020-06-23 12:21:09,263 P6804 INFO [Metrics] logloss: 0.438066 - AUC: 0.813774
2020-06-23 12:21:09,264 P6804 INFO Save best model: monitor(max): 0.375708
2020-06-23 12:21:09,352 P6804 INFO --- 3668/3668 batches finished ---
2020-06-23 12:21:09,391 P6804 INFO Train loss: 0.436778
2020-06-23 12:21:09,391 P6804 INFO ************ Epoch=7 end ************
2020-06-23 12:30:19,897 P6804 INFO [Metrics] logloss: 0.437963 - AUC: 0.813896
2020-06-23 12:30:19,898 P6804 INFO Save best model: monitor(max): 0.375933
2020-06-23 12:30:20,021 P6804 INFO --- 3668/3668 batches finished ---
2020-06-23 12:30:20,079 P6804 INFO Train loss: 0.434601
2020-06-23 12:30:20,079 P6804 INFO ************ Epoch=8 end ************
2020-06-23 12:39:33,389 P6804 INFO [Metrics] logloss: 0.438155 - AUC: 0.813705
2020-06-23 12:39:33,391 P6804 INFO Monitor(max) STOP: 0.375549 !
2020-06-23 12:39:33,391 P6804 INFO Reduce learning rate on plateau: 0.000010
2020-06-23 12:39:33,391 P6804 INFO --- 3668/3668 batches finished ---
2020-06-23 12:39:33,430 P6804 INFO Train loss: 0.432899
2020-06-23 12:39:33,430 P6804 INFO ************ Epoch=9 end ************
2020-06-23 12:48:46,759 P6804 INFO [Metrics] logloss: 0.438718 - AUC: 0.813293
2020-06-23 12:48:46,760 P6804 INFO Monitor(max) STOP: 0.374574 !
2020-06-23 12:48:46,760 P6804 INFO Reduce learning rate on plateau: 0.000001
2020-06-23 12:48:46,760 P6804 INFO Early stopping at epoch=10
2020-06-23 12:48:46,760 P6804 INFO --- 3668/3668 batches finished ---
2020-06-23 12:48:46,799 P6804 INFO Train loss: 0.428494
2020-06-23 12:48:46,799 P6804 INFO Training finished.
2020-06-23 12:48:46,799 P6804 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/DeepFM_criteo/min10/criteo_x4_5c863b0f/DeepFM_criteo_x4_5c863b0f_024_c15d5b96_model.ckpt
2020-06-23 12:48:46,948 P6804 INFO ****** Train/validation evaluation ******
2020-06-23 12:52:21,273 P6804 INFO [Metrics] logloss: 0.423523 - AUC: 0.829712
2020-06-23 12:52:47,165 P6804 INFO [Metrics] logloss: 0.437963 - AUC: 0.813896
2020-06-23 12:52:47,245 P6804 INFO ******** Test evaluation ********
2020-06-23 12:52:47,246 P6804 INFO Loading data...
2020-06-23 12:52:47,246 P6804 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-06-23 12:52:48,066 P6804 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-23 12:52:48,067 P6804 INFO Loading test data done.
2020-06-23 12:53:13,508 P6804 INFO [Metrics] logloss: 0.437584 - AUC: 0.814332

```
