## AutoInt+_criteo_x4_002

A hands-on guide to run the AutoInt model on the Criteo_x4_002 dataset.

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

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt+_criteo_x4_tuner_config_03](./AutoInt+_criteo_x4_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AutoInt+_criteo_x4_002
    nohup python run_expid.py --config ./AutoInt+_criteo_x4_tuner_config_03 --expid AutoInt_criteo_x4_003_1a01a590 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.438536 | 0.813363  |


### Logs
```python
2020-03-04 10:53:56,147 P916 INFO {
    "attention_dim": "64",
    "attention_layers": "5",
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_001_be98441d",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[5000, 5000, 5000]",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-06",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AutoInt",
    "model_id": "AutoInt_criteo_x4_003_387daa3b",
    "model_root": "./Criteo/AutoInt_criteo/",
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
    "test_data": "../data/Criteo/criteo_x4_001_be98441d/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_be98441d/train.h5",
    "use_hdf5": "True",
    "use_residual": "True",
    "use_scale": "False",
    "use_wide": "True",
    "valid_data": "../data/Criteo/criteo_x4_001_be98441d/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-03-04 10:53:56,156 P916 INFO Set up feature encoder...
2020-03-04 10:53:56,156 P916 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-03-04 10:53:56,157 P916 INFO Loading data...
2020-03-04 10:53:56,162 P916 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-03-04 10:54:03,479 P916 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-03-04 10:54:06,120 P916 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-03-04 10:54:06,292 P916 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-03-04 10:54:06,292 P916 INFO Loading train data done.
2020-03-04 10:54:18,719 P916 INFO **** Start training: 3668 batches/epoch ****
2020-03-04 11:29:02,998 P916 INFO [Metrics] logloss: 0.443227 - AUC: 0.809029
2020-03-04 11:29:03,087 P916 INFO Save best model: monitor(max): 0.365802
2020-03-04 11:29:04,751 P916 INFO --- 3668/3668 batches finished ---
2020-03-04 11:29:04,823 P916 INFO Train loss: 0.455882
2020-03-04 11:29:04,823 P916 INFO ************ Epoch=1 end ************
2020-03-04 12:03:48,613 P916 INFO [Metrics] logloss: 0.439977 - AUC: 0.811790
2020-03-04 12:03:48,680 P916 INFO Save best model: monitor(max): 0.371813
2020-03-04 12:03:51,185 P916 INFO --- 3668/3668 batches finished ---
2020-03-04 12:03:51,279 P916 INFO Train loss: 0.448881
2020-03-04 12:03:51,280 P916 INFO ************ Epoch=2 end ************
2020-03-04 12:38:34,860 P916 INFO [Metrics] logloss: 0.438952 - AUC: 0.812871
2020-03-04 12:38:34,922 P916 INFO Save best model: monitor(max): 0.373920
2020-03-04 12:38:37,879 P916 INFO --- 3668/3668 batches finished ---
2020-03-04 12:38:37,977 P916 INFO Train loss: 0.446646
2020-03-04 12:38:37,977 P916 INFO ************ Epoch=3 end ************
2020-03-04 13:13:24,344 P916 INFO [Metrics] logloss: 0.439069 - AUC: 0.812850
2020-03-04 13:13:24,418 P916 INFO Monitor(max) STOP: 0.373781 !
2020-03-04 13:13:24,418 P916 INFO Reduce learning rate on plateau: 0.000100
2020-03-04 13:13:24,418 P916 INFO --- 3668/3668 batches finished ---
2020-03-04 13:13:24,503 P916 INFO Train loss: 0.444938
2020-03-04 13:13:24,504 P916 INFO ************ Epoch=4 end ************
2020-03-04 13:48:09,808 P916 INFO [Metrics] logloss: 0.448477 - AUC: 0.806974
2020-03-04 13:48:09,876 P916 INFO Monitor(max) STOP: 0.358496 !
2020-03-04 13:48:09,876 P916 INFO Reduce learning rate on plateau: 0.000010
2020-03-04 13:48:09,878 P916 INFO Early stopping at epoch=5
2020-03-04 13:48:09,879 P916 INFO --- 3668/3668 batches finished ---
2020-03-04 13:48:09,967 P916 INFO Train loss: 0.417321
2020-03-04 13:48:09,968 P916 INFO Training finished.
2020-03-04 13:48:09,968 P916 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/AutoInt_criteo/criteo_x4_001_be98441d/AutoInt_criteo_x4_003_387daa3b_criteo_x4_001_be98441d_model.ckpt
2020-03-04 13:48:12,342 P916 INFO ****** Train/validation evaluation ******
2020-03-04 13:57:02,907 P916 INFO [Metrics] logloss: 0.422754 - AUC: 0.830164
2020-03-04 13:58:08,030 P916 INFO [Metrics] logloss: 0.438952 - AUC: 0.812871
2020-03-04 13:58:08,230 P916 INFO ******** Test evaluation ********
2020-03-04 13:58:08,231 P916 INFO Loading data...
2020-03-04 13:58:08,231 P916 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-03-04 13:58:09,770 P916 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-03-04 13:58:09,770 P916 INFO Loading test data done.
2020-03-04 13:59:14,259 P916 INFO [Metrics] logloss: 0.438536 - AUC: 0.813363

```
