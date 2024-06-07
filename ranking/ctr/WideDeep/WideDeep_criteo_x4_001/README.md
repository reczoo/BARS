## WideDeep_criteo_x4_001

A hands-on guide to run the WideDeep model on the Criteo_x4_001 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [WideDeep](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/WideDeep.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [WideDeep_criteo_x4_tuner_config_01](./WideDeep_criteo_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd WideDeep_criteo_x4_001
    nohup python run_expid.py --config ./WideDeep_criteo_x4_tuner_config_01 --expid WideDeep_criteo_x4_024_ba0036f7 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.437693 | 0.814232  |


### Logs
```python
2020-06-23 09:12:46,266 P5673 INFO {
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
    "model": "WideDeep",
    "model_id": "WideDeep_criteo_x4_5c863b0f_024_5c11bdab",
    "model_root": "./Criteo/WideDeep_criteo/min10/",
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
2020-06-23 09:12:46,268 P5673 INFO Set up feature encoder...
2020-06-23 09:12:46,268 P5673 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-06-23 09:12:46,269 P5673 INFO Loading data...
2020-06-23 09:12:46,277 P5673 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-06-23 09:12:52,654 P5673 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-06-23 09:12:54,785 P5673 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-06-23 09:12:54,995 P5673 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-23 09:12:54,995 P5673 INFO Loading train data done.
2020-06-23 09:12:59,510 P5673 INFO Start training: 3668 batches/epoch
2020-06-23 09:12:59,511 P5673 INFO ************ Epoch=1 start ************
2020-06-23 09:22:27,515 P5673 INFO [Metrics] logloss: 0.446271 - AUC: 0.805271
2020-06-23 09:22:27,519 P5673 INFO Save best model: monitor(max): 0.359000
2020-06-23 09:22:28,040 P5673 INFO --- 3668/3668 batches finished ---
2020-06-23 09:22:28,099 P5673 INFO Train loss: 0.460048
2020-06-23 09:22:28,099 P5673 INFO ************ Epoch=1 end ************
2020-06-23 09:31:51,609 P5673 INFO [Metrics] logloss: 0.444134 - AUC: 0.807536
2020-06-23 09:31:51,610 P5673 INFO Save best model: monitor(max): 0.363402
2020-06-23 09:31:51,732 P5673 INFO --- 3668/3668 batches finished ---
2020-06-23 09:31:51,786 P5673 INFO Train loss: 0.454757
2020-06-23 09:31:51,787 P5673 INFO ************ Epoch=2 end ************
2020-06-23 09:41:08,919 P5673 INFO [Metrics] logloss: 0.442728 - AUC: 0.808684
2020-06-23 09:41:08,920 P5673 INFO Save best model: monitor(max): 0.365956
2020-06-23 09:41:09,010 P5673 INFO --- 3668/3668 batches finished ---
2020-06-23 09:41:09,068 P5673 INFO Train loss: 0.453255
2020-06-23 09:41:09,069 P5673 INFO ************ Epoch=3 end ************
2020-06-23 09:50:23,782 P5673 INFO [Metrics] logloss: 0.442151 - AUC: 0.809343
2020-06-23 09:50:23,783 P5673 INFO Save best model: monitor(max): 0.367192
2020-06-23 09:50:23,896 P5673 INFO --- 3668/3668 batches finished ---
2020-06-23 09:50:23,953 P5673 INFO Train loss: 0.452460
2020-06-23 09:50:23,953 P5673 INFO ************ Epoch=4 end ************
2020-06-23 09:59:39,234 P5673 INFO [Metrics] logloss: 0.442114 - AUC: 0.809630
2020-06-23 09:59:39,235 P5673 INFO Save best model: monitor(max): 0.367516
2020-06-23 09:59:39,358 P5673 INFO --- 3668/3668 batches finished ---
2020-06-23 09:59:39,415 P5673 INFO Train loss: 0.451888
2020-06-23 09:59:39,415 P5673 INFO ************ Epoch=5 end ************
2020-06-23 10:08:59,633 P5673 INFO [Metrics] logloss: 0.441625 - AUC: 0.809936
2020-06-23 10:08:59,634 P5673 INFO Save best model: monitor(max): 0.368310
2020-06-23 10:08:59,727 P5673 INFO --- 3668/3668 batches finished ---
2020-06-23 10:08:59,780 P5673 INFO Train loss: 0.451477
2020-06-23 10:08:59,781 P5673 INFO ************ Epoch=6 end ************
2020-06-23 10:18:15,624 P5673 INFO [Metrics] logloss: 0.442120 - AUC: 0.809860
2020-06-23 10:18:15,626 P5673 INFO Monitor(max) STOP: 0.367740 !
2020-06-23 10:18:15,626 P5673 INFO Reduce learning rate on plateau: 0.000100
2020-06-23 10:18:15,626 P5673 INFO --- 3668/3668 batches finished ---
2020-06-23 10:18:15,682 P5673 INFO Train loss: 0.451120
2020-06-23 10:18:15,682 P5673 INFO ************ Epoch=7 end ************
2020-06-23 10:27:32,145 P5673 INFO [Metrics] logloss: 0.438490 - AUC: 0.813390
2020-06-23 10:27:32,146 P5673 INFO Save best model: monitor(max): 0.374900
2020-06-23 10:27:32,252 P5673 INFO --- 3668/3668 batches finished ---
2020-06-23 10:27:32,309 P5673 INFO Train loss: 0.440507
2020-06-23 10:27:32,309 P5673 INFO ************ Epoch=8 end ************
2020-06-23 10:36:52,936 P5673 INFO [Metrics] logloss: 0.438103 - AUC: 0.813777
2020-06-23 10:36:52,937 P5673 INFO Save best model: monitor(max): 0.375674
2020-06-23 10:36:53,055 P5673 INFO --- 3668/3668 batches finished ---
2020-06-23 10:36:53,116 P5673 INFO Train loss: 0.435992
2020-06-23 10:36:53,116 P5673 INFO ************ Epoch=9 end ************
2020-06-23 10:46:11,243 P5673 INFO [Metrics] logloss: 0.438061 - AUC: 0.813809
2020-06-23 10:46:11,245 P5673 INFO Save best model: monitor(max): 0.375747
2020-06-23 10:46:11,353 P5673 INFO --- 3668/3668 batches finished ---
2020-06-23 10:46:11,411 P5673 INFO Train loss: 0.433762
2020-06-23 10:46:11,411 P5673 INFO ************ Epoch=10 end ************
2020-06-23 10:55:27,178 P5673 INFO [Metrics] logloss: 0.438205 - AUC: 0.813628
2020-06-23 10:55:27,179 P5673 INFO Monitor(max) STOP: 0.375423 !
2020-06-23 10:55:27,180 P5673 INFO Reduce learning rate on plateau: 0.000010
2020-06-23 10:55:27,180 P5673 INFO --- 3668/3668 batches finished ---
2020-06-23 10:55:27,236 P5673 INFO Train loss: 0.432017
2020-06-23 10:55:27,236 P5673 INFO ************ Epoch=11 end ************
2020-06-23 11:04:40,876 P5673 INFO [Metrics] logloss: 0.438640 - AUC: 0.813245
2020-06-23 11:04:40,878 P5673 INFO Monitor(max) STOP: 0.374606 !
2020-06-23 11:04:40,878 P5673 INFO Reduce learning rate on plateau: 0.000001
2020-06-23 11:04:40,878 P5673 INFO Early stopping at epoch=12
2020-06-23 11:04:40,878 P5673 INFO --- 3668/3668 batches finished ---
2020-06-23 11:04:40,936 P5673 INFO Train loss: 0.427875
2020-06-23 11:04:40,937 P5673 INFO Training finished.
2020-06-23 11:04:40,937 P5673 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/WideDeep_criteo/min10/criteo_x4_5c863b0f/WideDeep_criteo_x4_5c863b0f_024_5c11bdab_model.ckpt
2020-06-23 11:04:41,135 P5673 INFO ****** Train/validation evaluation ******
2020-06-23 11:08:14,955 P5673 INFO [Metrics] logloss: 0.422796 - AUC: 0.830626
2020-06-23 11:08:39,413 P5673 INFO [Metrics] logloss: 0.438061 - AUC: 0.813809
2020-06-23 11:08:39,502 P5673 INFO ******** Test evaluation ********
2020-06-23 11:08:39,502 P5673 INFO Loading data...
2020-06-23 11:08:39,502 P5673 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-06-23 11:08:40,505 P5673 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-23 11:08:40,506 P5673 INFO Loading test data done.
2020-06-23 11:09:03,847 P5673 INFO [Metrics] logloss: 0.437693 - AUC: 0.814232

```
