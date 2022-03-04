## HFM+_criteo_x4_001

A hands-on guide to run the HFM model on the Criteo_x4_001 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

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
Dataset ID: [Criteo_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [HFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/HFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HFM+_criteo_x4_tuner_config_11](./HFM+_criteo_x4_tuner_config_11). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HFM+_criteo_x4_001
    nohup python run_expid.py --config ./HFM+_criteo_x4_tuner_config_11 --expid HFM_criteo_x4_006_2abdb8e2 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.439178 | 0.812710  |


### Logs
```python
2020-07-27 00:58:57,647 P13799 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-06",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000]",
    "interaction_type": "circular_correlation",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "HFM",
    "model_id": "HFM_criteo_x4_5c863b0f_006_e5c408ac",
    "model_root": "./Criteo/HFM_criteo/min10/",
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
    "test_data": "../data/Criteo/criteo_x4_5c863b0f/test.h5",
    "train_data": "../data/Criteo/criteo_x4_5c863b0f/train.h5",
    "use_dnn": "True",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-27 00:58:57,647 P13799 INFO Set up feature encoder...
2020-07-27 00:58:57,647 P13799 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-07-27 00:58:57,648 P13799 INFO Loading data...
2020-07-27 00:58:57,650 P13799 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-07-27 00:59:03,009 P13799 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-07-27 00:59:05,045 P13799 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-07-27 00:59:05,190 P13799 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-27 00:59:05,190 P13799 INFO Loading train data done.
2020-07-27 00:59:08,558 P13799 INFO **** Start training: 3668 batches/epoch ****
2020-07-27 01:39:44,038 P13799 INFO [Metrics] logloss: 0.441114 - AUC: 0.810581
2020-07-27 01:39:44,039 P13799 INFO Save best model: monitor(max): 0.369467
2020-07-27 01:39:44,150 P13799 INFO --- 3668/3668 batches finished ---
2020-07-27 01:39:44,198 P13799 INFO Train loss: 0.451195
2020-07-27 01:39:44,198 P13799 INFO ************ Epoch=1 end ************
2020-07-27 02:20:19,589 P13799 INFO [Metrics] logloss: 0.439555 - AUC: 0.812258
2020-07-27 02:20:19,590 P13799 INFO Save best model: monitor(max): 0.372703
2020-07-27 02:20:19,794 P13799 INFO --- 3668/3668 batches finished ---
2020-07-27 02:20:19,844 P13799 INFO Train loss: 0.442213
2020-07-27 02:20:19,844 P13799 INFO ************ Epoch=2 end ************
2020-07-27 03:00:55,418 P13799 INFO [Metrics] logloss: 0.441363 - AUC: 0.810539
2020-07-27 03:00:55,419 P13799 INFO Monitor(max) STOP: 0.369176 !
2020-07-27 03:00:55,419 P13799 INFO Reduce learning rate on plateau: 0.000100
2020-07-27 03:00:55,419 P13799 INFO --- 3668/3668 batches finished ---
2020-07-27 03:00:55,471 P13799 INFO Train loss: 0.436734
2020-07-27 03:00:55,471 P13799 INFO ************ Epoch=3 end ************
2020-07-27 03:41:36,566 P13799 INFO [Metrics] logloss: 0.476991 - AUC: 0.790653
2020-07-27 03:41:36,567 P13799 INFO Monitor(max) STOP: 0.313663 !
2020-07-27 03:41:36,567 P13799 INFO Reduce learning rate on plateau: 0.000010
2020-07-27 03:41:36,567 P13799 INFO Early stopping at epoch=4
2020-07-27 03:41:36,567 P13799 INFO --- 3668/3668 batches finished ---
2020-07-27 03:41:36,617 P13799 INFO Train loss: 0.393167
2020-07-27 03:41:36,617 P13799 INFO Training finished.
2020-07-27 03:41:36,617 P13799 INFO Load best model: /home/XXX/benchmarks/Criteo/HFM_criteo/min10/criteo_x4_5c863b0f/HFM_criteo_x4_5c863b0f_006_e5c408ac_model.ckpt
2020-07-27 03:41:36,777 P13799 INFO ****** Train/validation evaluation ******
2020-07-27 03:43:07,618 P13799 INFO [Metrics] logloss: 0.439555 - AUC: 0.812258
2020-07-27 03:43:07,733 P13799 INFO ******** Test evaluation ********
2020-07-27 03:43:07,733 P13799 INFO Loading data...
2020-07-27 03:43:07,733 P13799 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-07-27 03:43:08,529 P13799 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-27 03:43:08,530 P13799 INFO Loading test data done.
2020-07-27 03:44:39,219 P13799 INFO [Metrics] logloss: 0.439178 - AUC: 0.812710

```
