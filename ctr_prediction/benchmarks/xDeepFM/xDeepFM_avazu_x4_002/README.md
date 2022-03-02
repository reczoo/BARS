## xDeepFM_avazu_x4_002

A hands-on guide to run the xDeepFM model on the Avazu_x4_002 dataset.

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
Dataset ID: [Avazu_x4_002](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x4_002). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [xDeepFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/xDeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [xDeepFM_avazu_x4_tuner_config_08](./xDeepFM_avazu_x4_tuner_config_08). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd xDeepFM_avazu_x4_002
    nohup python run_expid.py --config ./xDeepFM_avazu_x4_tuner_config_08 --expid xDeepFM_avazu_x4_008_e76be77b --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.369703 | 0.796744  |


### Logs
```python
2020-03-23 02:17:23,276 P1433 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "cin_layer_units": "[50, 50]",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "dnn_hidden_units": "[500, 500]",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "hidden_activations": "relu",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "xDeepFM",
    "model_id": "xDeepFM_avazu_x4_008_3aefa952",
    "model_root": "./Avazu/xDeepFM_avazu/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "1",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-03-23 02:17:23,278 P1433 INFO Set up feature encoder...
2020-03-23 02:17:23,278 P1433 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-03-23 02:17:23,279 P1433 INFO Loading data...
2020-03-23 02:17:23,288 P1433 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-03-23 02:17:26,818 P1433 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-03-23 02:17:28,141 P1433 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-03-23 02:17:28,243 P1433 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-03-23 02:17:28,243 P1433 INFO Loading train data done.
2020-03-23 02:17:40,642 P1433 INFO **** Start training: 3235 batches/epoch ****
2020-03-23 02:36:53,471 P1433 INFO [Metrics] logloss: 0.369772 - AUC: 0.796629
2020-03-23 02:36:53,473 P1433 INFO Save best model: monitor(max): 0.426857
2020-03-23 02:36:55,560 P1433 INFO --- 3235/3235 batches finished ---
2020-03-23 02:36:55,621 P1433 INFO Train loss: 0.380686
2020-03-23 02:36:55,621 P1433 INFO ************ Epoch=1 end ************
2020-03-23 02:56:05,356 P1433 INFO [Metrics] logloss: 0.392353 - AUC: 0.780802
2020-03-23 02:56:05,360 P1433 INFO Monitor(max) STOP: 0.388449 !
2020-03-23 02:56:05,360 P1433 INFO Reduce learning rate on plateau: 0.000100
2020-03-23 02:56:05,360 P1433 INFO Early stopping at epoch=2
2020-03-23 02:56:05,360 P1433 INFO --- 3235/3235 batches finished ---
2020-03-23 02:56:05,415 P1433 INFO Train loss: 0.287833
2020-03-23 02:56:05,416 P1433 INFO Training finished.
2020-03-23 02:56:05,416 P1433 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Avazu/xDeepFM_avazu/avazu_x4_001_d45ad60e/xDeepFM_avazu_x4_008_3aefa952_model.ckpt
2020-03-23 02:56:07,298 P1433 INFO ****** Train/validation evaluation ******
2020-03-23 02:59:36,028 P1433 INFO [Metrics] logloss: 0.317834 - AUC: 0.871821
2020-03-23 03:00:00,499 P1433 INFO [Metrics] logloss: 0.369772 - AUC: 0.796629
2020-03-23 03:00:00,584 P1433 INFO ******** Test evaluation ********
2020-03-23 03:00:00,584 P1433 INFO Loading data...
2020-03-23 03:00:00,584 P1433 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-03-23 03:00:01,158 P1433 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-03-23 03:00:01,158 P1433 INFO Loading test data done.
2020-03-23 03:00:25,817 P1433 INFO [Metrics] logloss: 0.369703 - AUC: 0.796744

```
