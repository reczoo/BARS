## DeepCross_criteo_x4_002

A hands-on guide to run the DeepCrossing model on the Criteo_x4_002 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [DeepCrossing](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DeepCrossing.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DeepCrossing_criteo_x4_tuner_config_02](./DeepCrossing_criteo_x4_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DeepCross_criteo_x4_002
    nohup python run_expid.py --config ./DeepCrossing_criteo_x4_tuner_config_02 --expid DeepCrossing_criteo_x4_056_0f92ea50 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.438044 | 0.813945  |


### Logs
```python
2020-01-19 13:24:05,959 P10216 INFO {
    "batch_norm": "True",
    "batch_size": "10000",
    "dataset_id": "criteo_x4_001_be98441d",
    "dnn_activations": "relu",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "l2(1.e-5)",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DeepCrossing",
    "model_id": "DeepCrossing_criteo_x4_056_43575e47",
    "model_root": "./Criteo/DeepCrossing_criteo/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "3",
    "pickle_feature_encoder": "True",
    "residual_blocks": "[5000, 5000, 5000, 5000]",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "use_residual": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "test_data": "../data/Criteo/criteo_x4_001_be98441d/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_be98441d/train.h5",
    "valid_data": "../data/Criteo/criteo_x4_001_be98441d/valid.h5",
    "version": "pytorch",
    "gpu": "1"
}
2020-01-19 13:24:05,960 P10216 INFO Set up feature encoder...
2020-01-19 13:24:05,960 P10216 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-01-19 13:24:05,960 P10216 INFO Loading data...
2020-01-19 13:24:05,962 P10216 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-01-19 13:24:10,760 P10216 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-01-19 13:24:12,638 P10216 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-01-19 13:24:12,763 P10216 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-01-19 13:24:12,764 P10216 INFO Loading train data done.
2020-01-19 13:24:21,609 P10216 INFO **** Start training: 3668 batches/epoch ****
2020-01-19 13:50:27,311 P10216 INFO [Metrics] logloss: 0.445331 - AUC: 0.806359
2020-01-19 13:50:27,368 P10216 INFO Save best model: monitor(max): 0.361027
2020-01-19 13:50:28,938 P10216 INFO --- 3668/3668 batches finished ---
2020-01-19 13:50:28,990 P10216 INFO Train loss: 0.468051
2020-01-19 13:50:28,990 P10216 INFO ************ Epoch=1 end ************
2020-01-19 14:16:32,833 P10216 INFO [Metrics] logloss: 0.453513 - AUC: 0.806340
2020-01-19 14:16:32,886 P10216 INFO Monitor(max) STOP: 0.352827 !
2020-01-19 14:16:32,886 P10216 INFO Reduce learning rate on plateau: 0.000100
2020-01-19 14:16:32,886 P10216 INFO --- 3668/3668 batches finished ---
2020-01-19 14:16:32,943 P10216 INFO Train loss: 0.459663
2020-01-19 14:16:32,944 P10216 INFO ************ Epoch=2 end ************
2020-01-19 14:42:37,142 P10216 INFO [Metrics] logloss: 0.438469 - AUC: 0.813427
2020-01-19 14:42:37,196 P10216 INFO Save best model: monitor(max): 0.374957
2020-01-19 14:42:43,580 P10216 INFO --- 3668/3668 batches finished ---
2020-01-19 14:42:43,639 P10216 INFO Train loss: 0.441153
2020-01-19 14:42:43,639 P10216 INFO ************ Epoch=3 end ************
2020-01-19 15:08:48,521 P10216 INFO [Metrics] logloss: 0.438668 - AUC: 0.813559
2020-01-19 15:08:48,589 P10216 INFO Monitor(max) STOP: 0.374891 !
2020-01-19 15:08:48,589 P10216 INFO Reduce learning rate on plateau: 0.000010
2020-01-19 15:08:48,589 P10216 INFO --- 3668/3668 batches finished ---
2020-01-19 15:08:48,689 P10216 INFO Train loss: 0.434235
2020-01-19 15:08:48,689 P10216 INFO ************ Epoch=4 end ************
2020-01-19 15:34:56,469 P10216 INFO [Metrics] logloss: 0.443278 - AUC: 0.810148
2020-01-19 15:34:56,529 P10216 INFO Monitor(max) STOP: 0.366871 !
2020-01-19 15:34:56,529 P10216 INFO Reduce learning rate on plateau: 0.000001
2020-01-19 15:34:56,529 P10216 INFO --- 3668/3668 batches finished ---
2020-01-19 15:34:56,589 P10216 INFO Train loss: 0.422021
2020-01-19 15:34:56,589 P10216 INFO ************ Epoch=5 end ************
2020-01-19 16:01:03,111 P10216 INFO [Metrics] logloss: 0.444817 - AUC: 0.809186
2020-01-19 16:01:03,171 P10216 INFO Monitor(max) STOP: 0.364370 !
2020-01-19 16:01:03,171 P10216 INFO Reduce learning rate on plateau: 0.000001
2020-01-19 16:01:03,172 P10216 INFO Early stopping at epoch=6
2020-01-19 16:01:03,172 P10216 INFO --- 3668/3668 batches finished ---
2020-01-19 16:01:03,232 P10216 INFO Train loss: 0.418519
2020-01-19 16:01:03,233 P10216 INFO Training finished.
2020-01-19 16:01:03,233 P10216 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/DeepCrossing_criteo/criteo_x4_001_be98441d/DeepCrossing_criteo_x4_056_43575e47_criteo_x4_001_be98441d_model.ckpt
2020-01-19 16:01:05,029 P10216 INFO ****** Train/validation evaluation ******
2020-01-19 16:09:09,441 P10216 INFO [Metrics] logloss: 0.424812 - AUC: 0.827930
2020-01-19 16:10:09,827 P10216 INFO [Metrics] logloss: 0.438469 - AUC: 0.813427
2020-01-19 16:10:10,013 P10216 INFO ******** Test evaluation ********
2020-01-19 16:10:10,013 P10216 INFO Loading data...
2020-01-19 16:10:10,013 P10216 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-01-19 16:10:10,770 P10216 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-01-19 16:10:10,770 P10216 INFO Loading test data done.
2020-01-19 16:11:09,848 P10216 INFO [Metrics] logloss: 0.438044 - AUC: 0.813945

```
