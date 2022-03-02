## NFM_avazu_x1

A hands-on guide to run the NFM model on the Avazu_x1 dataset.

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

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [NFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/NFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [NFM_avazu_x1_tuner_config_02](./NFM_avazu_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd NFM_avazu_x1
    nohup python run_expid.py --config ./NFM_avazu_x1_tuner_config_02 --expid NFM_avazu_x1_005_1fe17989 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.762700 | 0.367681  |


### Logs
```python
2022-01-26 18:15:29,579 P52995 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.0005",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "4",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "NFM",
    "model_id": "NFM_avazu_x1_005_1fe17989",
    "model_root": "./Avazu/NFM_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
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
    "verbose": "1",
    "version": "pytorch"
}
2022-01-26 18:15:29,580 P52995 INFO Set up feature encoder...
2022-01-26 18:15:29,580 P52995 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-01-26 18:15:29,580 P52995 INFO Loading data...
2022-01-26 18:15:29,581 P52995 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-01-26 18:15:31,889 P52995 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-01-26 18:15:32,248 P52995 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-01-26 18:15:32,248 P52995 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-01-26 18:15:32,248 P52995 INFO Loading train data done.
2022-01-26 18:15:37,284 P52995 INFO Total number of parameters: 14610190.
2022-01-26 18:15:37,284 P52995 INFO Start training: 6910 batches/epoch
2022-01-26 18:15:37,284 P52995 INFO ************ Epoch=1 start ************
2022-01-26 18:22:24,983 P52995 INFO [Metrics] AUC: 0.743028 - logloss: 0.400330
2022-01-26 18:22:24,985 P52995 INFO Save best model: monitor(max): 0.743028
2022-01-26 18:22:25,218 P52995 INFO --- 6910/6910 batches finished ---
2022-01-26 18:22:25,257 P52995 INFO Train loss: 0.416401
2022-01-26 18:22:25,257 P52995 INFO ************ Epoch=1 end ************
2022-01-26 18:29:13,406 P52995 INFO [Metrics] AUC: 0.744131 - logloss: 0.397345
2022-01-26 18:29:13,408 P52995 INFO Save best model: monitor(max): 0.744131
2022-01-26 18:29:13,490 P52995 INFO --- 6910/6910 batches finished ---
2022-01-26 18:29:13,525 P52995 INFO Train loss: 0.414699
2022-01-26 18:29:13,526 P52995 INFO ************ Epoch=2 end ************
2022-01-26 18:35:59,477 P52995 INFO [Metrics] AUC: 0.743657 - logloss: 0.397869
2022-01-26 18:35:59,479 P52995 INFO Monitor(max) STOP: 0.743657 !
2022-01-26 18:35:59,479 P52995 INFO Reduce learning rate on plateau: 0.000100
2022-01-26 18:35:59,479 P52995 INFO --- 6910/6910 batches finished ---
2022-01-26 18:35:59,533 P52995 INFO Train loss: 0.414380
2022-01-26 18:35:59,533 P52995 INFO ************ Epoch=3 end ************
2022-01-26 18:42:45,946 P52995 INFO [Metrics] AUC: 0.746480 - logloss: 0.396401
2022-01-26 18:42:45,949 P52995 INFO Save best model: monitor(max): 0.746480
2022-01-26 18:42:46,027 P52995 INFO --- 6910/6910 batches finished ---
2022-01-26 18:42:46,076 P52995 INFO Train loss: 0.400584
2022-01-26 18:42:46,076 P52995 INFO ************ Epoch=4 end ************
2022-01-26 18:49:34,547 P52995 INFO [Metrics] AUC: 0.743148 - logloss: 0.398356
2022-01-26 18:49:34,550 P52995 INFO Monitor(max) STOP: 0.743148 !
2022-01-26 18:49:34,550 P52995 INFO Reduce learning rate on plateau: 0.000010
2022-01-26 18:49:34,550 P52995 INFO --- 6910/6910 batches finished ---
2022-01-26 18:49:34,600 P52995 INFO Train loss: 0.397758
2022-01-26 18:49:34,600 P52995 INFO ************ Epoch=5 end ************
2022-01-26 18:56:12,348 P52995 INFO [Metrics] AUC: 0.739637 - logloss: 0.402627
2022-01-26 18:56:12,352 P52995 INFO Monitor(max) STOP: 0.739637 !
2022-01-26 18:56:12,352 P52995 INFO Reduce learning rate on plateau: 0.000001
2022-01-26 18:56:12,352 P52995 INFO Early stopping at epoch=6
2022-01-26 18:56:12,353 P52995 INFO --- 6910/6910 batches finished ---
2022-01-26 18:56:12,405 P52995 INFO Train loss: 0.391244
2022-01-26 18:56:12,406 P52995 INFO Training finished.
2022-01-26 18:56:12,406 P52995 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/NFM_avazu_x1/avazu_x1_3fb65689/NFM_avazu_x1_005_1fe17989.model
2022-01-26 18:56:17,956 P52995 INFO ****** Validation evaluation ******
2022-01-26 18:56:28,967 P52995 INFO [Metrics] AUC: 0.746480 - logloss: 0.396401
2022-01-26 18:56:29,036 P52995 INFO ******** Test evaluation ********
2022-01-26 18:56:29,036 P52995 INFO Loading data...
2022-01-26 18:56:29,037 P52995 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-01-26 18:56:29,849 P52995 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-01-26 18:56:29,849 P52995 INFO Loading test data done.
2022-01-26 18:56:53,332 P52995 INFO [Metrics] AUC: 0.762700 - logloss: 0.367681

```
