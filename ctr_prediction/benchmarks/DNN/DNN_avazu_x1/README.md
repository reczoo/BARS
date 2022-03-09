## DNN_avazu_x1

A hands-on guide to run the DNN model on the Avazu_x1 dataset.

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
  fuxictr: 1.1.0

  ```

### Dataset
Dataset ID: [Avazu_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DNN](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DNN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DNN_avazu_x1_tuner_config_seeds](./DNN_avazu_x1_tuner_config_seeds). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DNN_avazu_x1
    nohup python run_expid.py --config ./DNN_avazu_x1_tuner_config_seeds --expid DNN_avazu_x1_001_3da2d674 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.763019 | 0.368178  |
| 2 | 0.763126 | 0.367452  |
| 3 | 0.763342 | 0.367366  |
| 4 | 0.763145 | 0.367455  |
| 5 | 0.763870 | 0.367038  |
| Avg | 0.763300 | 0.367498 |
| Std | &#177;0.00030329 | &#177;0.00037293 |


### Logs
```python
2022-02-08 09:57:44,716 P50417 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DNN",
    "model_id": "DNN_avazu_x1_001_3da2d674",
    "model_root": "./Avazu/DNN_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-02-08 09:57:44,717 P50417 INFO Set up feature encoder...
2022-02-08 09:57:44,717 P50417 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-02-08 09:57:44,717 P50417 INFO Loading data...
2022-02-08 09:57:44,718 P50417 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-02-08 09:57:46,771 P50417 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-02-08 09:57:47,098 P50417 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-02-08 09:57:47,098 P50417 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-02-08 09:57:47,098 P50417 INFO Loading train data done.
2022-02-08 09:57:51,209 P50417 INFO Total number of parameters: 13805192.
2022-02-08 09:57:51,209 P50417 INFO Start training: 6910 batches/epoch
2022-02-08 09:57:51,209 P50417 INFO ************ Epoch=1 start ************
2022-02-08 10:02:20,462 P50417 INFO [Metrics] AUC: 0.742239 - logloss: 0.398721
2022-02-08 10:02:20,465 P50417 INFO Save best model: monitor(max): 0.742239
2022-02-08 10:02:20,538 P50417 INFO --- 6910/6910 batches finished ---
2022-02-08 10:02:20,587 P50417 INFO Train loss: 0.427262
2022-02-08 10:02:20,587 P50417 INFO ************ Epoch=1 end ************
2022-02-08 10:06:46,211 P50417 INFO [Metrics] AUC: 0.739514 - logloss: 0.401270
2022-02-08 10:06:46,214 P50417 INFO Monitor(max) STOP: 0.739514 !
2022-02-08 10:06:46,214 P50417 INFO Reduce learning rate on plateau: 0.000100
2022-02-08 10:06:46,214 P50417 INFO --- 6910/6910 batches finished ---
2022-02-08 10:06:46,262 P50417 INFO Train loss: 0.427740
2022-02-08 10:06:46,262 P50417 INFO ************ Epoch=2 end ************
2022-02-08 10:11:10,596 P50417 INFO [Metrics] AUC: 0.743622 - logloss: 0.398500
2022-02-08 10:11:10,599 P50417 INFO Save best model: monitor(max): 0.743622
2022-02-08 10:11:10,667 P50417 INFO --- 6910/6910 batches finished ---
2022-02-08 10:11:10,713 P50417 INFO Train loss: 0.404701
2022-02-08 10:11:10,713 P50417 INFO ************ Epoch=3 end ************
2022-02-08 10:15:34,346 P50417 INFO [Metrics] AUC: 0.745833 - logloss: 0.396748
2022-02-08 10:15:34,349 P50417 INFO Save best model: monitor(max): 0.745833
2022-02-08 10:15:34,421 P50417 INFO --- 6910/6910 batches finished ---
2022-02-08 10:15:34,472 P50417 INFO Train loss: 0.405991
2022-02-08 10:15:34,472 P50417 INFO ************ Epoch=4 end ************
2022-02-08 10:19:57,705 P50417 INFO [Metrics] AUC: 0.745333 - logloss: 0.396666
2022-02-08 10:19:57,708 P50417 INFO Monitor(max) STOP: 0.745333 !
2022-02-08 10:19:57,708 P50417 INFO Reduce learning rate on plateau: 0.000010
2022-02-08 10:19:57,708 P50417 INFO --- 6910/6910 batches finished ---
2022-02-08 10:19:57,759 P50417 INFO Train loss: 0.406534
2022-02-08 10:19:57,759 P50417 INFO ************ Epoch=5 end ************
2022-02-08 10:24:22,342 P50417 INFO [Metrics] AUC: 0.745134 - logloss: 0.397022
2022-02-08 10:24:22,345 P50417 INFO Monitor(max) STOP: 0.745134 !
2022-02-08 10:24:22,345 P50417 INFO Reduce learning rate on plateau: 0.000001
2022-02-08 10:24:22,345 P50417 INFO Early stopping at epoch=6
2022-02-08 10:24:22,345 P50417 INFO --- 6910/6910 batches finished ---
2022-02-08 10:24:22,395 P50417 INFO Train loss: 0.393572
2022-02-08 10:24:22,395 P50417 INFO Training finished.
2022-02-08 10:24:22,395 P50417 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/DNN_avazu_x1/avazu_x1_3fb65689/DNN_avazu_x1_001_3da2d674.model
2022-02-08 10:24:22,686 P50417 INFO ****** Validation evaluation ******
2022-02-08 10:24:34,114 P50417 INFO [Metrics] AUC: 0.745833 - logloss: 0.396748
2022-02-08 10:24:34,195 P50417 INFO ******** Test evaluation ********
2022-02-08 10:24:34,195 P50417 INFO Loading data...
2022-02-08 10:24:34,195 P50417 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-02-08 10:24:34,880 P50417 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-02-08 10:24:34,881 P50417 INFO Loading test data done.
2022-02-08 10:25:00,299 P50417 INFO [Metrics] AUC: 0.763019 - logloss: 0.368178

```
