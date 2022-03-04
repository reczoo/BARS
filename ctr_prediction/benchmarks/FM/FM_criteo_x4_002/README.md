## FM_criteo_x4_002

A hands-on guide to run the FM model on the Criteo_x4_002 dataset.

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
Dataset ID: [Criteo_x4_002](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x4_002). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FM_criteo_x4_tuner_config_02](./FM_criteo_x4_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FM_criteo_x4_002
    nohup python run_expid.py --config ./FM_criteo_x4_tuner_config_02 --expid FM_criteo_x4_003_608b63aa --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.444521 | 0.807768  |


### Logs
```python
2020-02-23 18:05:28,658 P16598 INFO {
    "batch_size": "10000",
    "dataset_id": "criteo_x4_001_be98441d",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FM",
    "model_id": "FM_criteo_x4_003_a2962a05",
    "model_root": "./Criteo/AFN_criteo/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "l2(1.e-6)",
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
    "gpu": "1"
}
2020-02-23 18:05:28,658 P16598 INFO Set up feature encoder...
2020-02-23 18:05:28,659 P16598 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-02-23 18:05:28,659 P16598 INFO Loading data...
2020-02-23 18:05:28,661 P16598 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-02-23 18:05:46,789 P16598 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-02-23 18:05:54,828 P16598 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-02-23 18:05:55,137 P16598 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-23 18:05:55,137 P16598 INFO Loading train data done.
2020-02-23 18:06:09,458 P16598 INFO **** Start training: 3668 batches/epoch ****
2020-02-23 18:23:57,202 P16598 INFO [Metrics] logloss: 0.449224 - AUC: 0.802329
2020-02-23 18:23:57,307 P16598 INFO Save best model: monitor(max): 0.353105
2020-02-23 18:24:13,604 P16598 INFO --- 3668/3668 batches finished ---
2020-02-23 18:24:13,723 P16598 INFO Train loss: 0.466898
2020-02-23 18:24:13,723 P16598 INFO ************ Epoch=1 end ************
2020-02-23 18:42:04,757 P16598 INFO [Metrics] logloss: 0.448104 - AUC: 0.803612
2020-02-23 18:42:04,864 P16598 INFO Save best model: monitor(max): 0.355508
2020-02-23 18:42:23,156 P16598 INFO --- 3668/3668 batches finished ---
2020-02-23 18:42:23,311 P16598 INFO Train loss: 0.459605
2020-02-23 18:42:23,311 P16598 INFO ************ Epoch=2 end ************
2020-02-23 19:00:09,776 P16598 INFO [Metrics] logloss: 0.447962 - AUC: 0.803838
2020-02-23 19:00:09,887 P16598 INFO Save best model: monitor(max): 0.355876
2020-02-23 19:00:20,143 P16598 INFO --- 3668/3668 batches finished ---
2020-02-23 19:00:20,297 P16598 INFO Train loss: 0.457737
2020-02-23 19:00:20,297 P16598 INFO ************ Epoch=3 end ************
2020-02-23 19:18:14,355 P16598 INFO [Metrics] logloss: 0.448193 - AUC: 0.803793
2020-02-23 19:18:14,473 P16598 INFO Monitor(max) STOP: 0.355601 !
2020-02-23 19:18:14,473 P16598 INFO Reduce learning rate on plateau: 0.000100
2020-02-23 19:18:14,473 P16598 INFO --- 3668/3668 batches finished ---
2020-02-23 19:18:14,653 P16598 INFO Train loss: 0.456941
2020-02-23 19:18:14,654 P16598 INFO ************ Epoch=4 end ************
2020-02-23 19:35:56,547 P16598 INFO [Metrics] logloss: 0.444802 - AUC: 0.807371
2020-02-23 19:35:56,623 P16598 INFO Save best model: monitor(max): 0.362569
2020-02-23 19:36:07,101 P16598 INFO --- 3668/3668 batches finished ---
2020-02-23 19:36:07,256 P16598 INFO Train loss: 0.433009
2020-02-23 19:36:07,256 P16598 INFO ************ Epoch=5 end ************
2020-02-23 19:53:55,119 P16598 INFO [Metrics] logloss: 0.444909 - AUC: 0.807468
2020-02-23 19:53:55,230 P16598 INFO Monitor(max) STOP: 0.362559 !
2020-02-23 19:53:55,231 P16598 INFO Reduce learning rate on plateau: 0.000010
2020-02-23 19:53:55,231 P16598 INFO --- 3668/3668 batches finished ---
2020-02-23 19:53:55,381 P16598 INFO Train loss: 0.426765
2020-02-23 19:53:55,381 P16598 INFO ************ Epoch=6 end ************
2020-02-23 20:11:34,878 P16598 INFO [Metrics] logloss: 0.444926 - AUC: 0.807486
2020-02-23 20:11:34,983 P16598 INFO Monitor(max) STOP: 0.362560 !
2020-02-23 20:11:34,983 P16598 INFO Reduce learning rate on plateau: 0.000001
2020-02-23 20:11:34,983 P16598 INFO Early stopping at epoch=7
2020-02-23 20:11:34,983 P16598 INFO --- 3668/3668 batches finished ---
2020-02-23 20:11:35,163 P16598 INFO Train loss: 0.420552
2020-02-23 20:11:35,164 P16598 INFO Training finished.
2020-02-23 20:11:35,164 P16598 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Criteo/AFN_criteo/criteo_x4_001_be98441d/FM_criteo_x4_003_a2962a05_criteo_x4_001_be98441d_model.ckpt
2020-02-23 20:11:56,924 P16598 INFO ****** Train/validation evaluation ******
2020-02-23 20:17:53,970 P16598 INFO [Metrics] logloss: 0.410629 - AUC: 0.841832
2020-02-23 20:18:31,992 P16598 INFO [Metrics] logloss: 0.444802 - AUC: 0.807371
2020-02-23 20:18:32,489 P16598 INFO ******** Test evaluation ********
2020-02-23 20:18:32,489 P16598 INFO Loading data...
2020-02-23 20:18:32,489 P16598 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-02-23 20:18:33,403 P16598 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-23 20:18:33,404 P16598 INFO Loading test data done.
2020-02-23 20:19:10,860 P16598 INFO [Metrics] logloss: 0.444521 - AUC: 0.807768

```
