## AutoInt+_avazu_x1

A hands-on guide to run the AutoInt model on the Avazu_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [AutoInt](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt+_avazu_x1_tuner_config_03](./AutoInt+_avazu_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AutoInt+_avazu_x1
    nohup python run_expid.py --config ./AutoInt+_avazu_x1_tuner_config_03 --expid AutoInt_avazu_x1_004_4fe23ce2 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.765027 | 0.366462  |
| 2 | 0.763587 | 0.367470  |
| 3 | 0.764457 | 0.366959  |
| 4 | 0.764991 | 0.366918  |
| 5 | 0.764395 | 0.367214  |
| Avg | 0.764491 | 0.367005 |
| Std | &#177;0.00052247 | &#177;0.00033599 |


### Logs
```python
2022-01-22 09:32:29,244 P804 INFO {
    "attention_dim": "128",
    "attention_layers": "5",
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "3",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "layer_norm": "True",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AutoInt",
    "model_id": "AutoInt_avazu_x1_004_4fe23ce2",
    "model_root": "./Avazu/AutoInt_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_heads": "2",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_hdf5": "True",
    "use_residual": "True",
    "use_scale": "True",
    "use_wide": "False",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-22 09:32:29,245 P804 INFO Set up feature encoder...
2022-01-22 09:32:29,245 P804 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-01-22 09:32:29,245 P804 INFO Loading data...
2022-01-22 09:32:29,246 P804 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-01-22 09:32:31,964 P804 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-01-22 09:32:32,369 P804 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-01-22 09:32:32,369 P804 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-01-22 09:32:32,369 P804 INFO Loading train data done.
2022-01-22 09:32:40,231 P804 INFO Total number of parameters: 14202856.
2022-01-22 09:32:40,232 P804 INFO Start training: 6910 batches/epoch
2022-01-22 09:32:40,232 P804 INFO ************ Epoch=1 start ************
2022-01-22 10:18:42,753 P804 INFO [Metrics] AUC: 0.742664 - logloss: 0.398270
2022-01-22 10:18:42,756 P804 INFO Save best model: monitor(max): 0.742664
2022-01-22 10:18:43,074 P804 INFO --- 6910/6910 batches finished ---
2022-01-22 10:18:43,106 P804 INFO Train loss: 0.445123
2022-01-22 10:18:43,106 P804 INFO ************ Epoch=1 end ************
2022-01-22 11:04:44,294 P804 INFO [Metrics] AUC: 0.743205 - logloss: 0.397928
2022-01-22 11:04:44,298 P804 INFO Save best model: monitor(max): 0.743205
2022-01-22 11:04:44,371 P804 INFO --- 6910/6910 batches finished ---
2022-01-22 11:04:44,408 P804 INFO Train loss: 0.434130
2022-01-22 11:04:44,408 P804 INFO ************ Epoch=2 end ************
2022-01-22 11:50:41,370 P804 INFO [Metrics] AUC: 0.742893 - logloss: 0.397601
2022-01-22 11:50:41,373 P804 INFO Monitor(max) STOP: 0.742893 !
2022-01-22 11:50:41,373 P804 INFO Reduce learning rate on plateau: 0.000100
2022-01-22 11:50:41,373 P804 INFO --- 6910/6910 batches finished ---
2022-01-22 11:50:41,406 P804 INFO Train loss: 0.431313
2022-01-22 11:50:41,407 P804 INFO ************ Epoch=3 end ************
2022-01-22 12:36:37,843 P804 INFO [Metrics] AUC: 0.746804 - logloss: 0.395706
2022-01-22 12:36:37,847 P804 INFO Save best model: monitor(max): 0.746804
2022-01-22 12:36:37,918 P804 INFO --- 6910/6910 batches finished ---
2022-01-22 12:36:37,962 P804 INFO Train loss: 0.404546
2022-01-22 12:36:37,962 P804 INFO ************ Epoch=4 end ************
2022-01-22 13:22:36,277 P804 INFO [Metrics] AUC: 0.746342 - logloss: 0.396460
2022-01-22 13:22:36,280 P804 INFO Monitor(max) STOP: 0.746342 !
2022-01-22 13:22:36,280 P804 INFO Reduce learning rate on plateau: 0.000010
2022-01-22 13:22:36,280 P804 INFO --- 6910/6910 batches finished ---
2022-01-22 13:22:36,323 P804 INFO Train loss: 0.404421
2022-01-22 13:22:36,324 P804 INFO ************ Epoch=5 end ************
2022-01-22 14:09:04,618 P804 INFO [Metrics] AUC: 0.743823 - logloss: 0.397819
2022-01-22 14:09:04,621 P804 INFO Monitor(max) STOP: 0.743823 !
2022-01-22 14:09:04,621 P804 INFO Reduce learning rate on plateau: 0.000001
2022-01-22 14:09:04,621 P804 INFO Early stopping at epoch=6
2022-01-22 14:09:04,622 P804 INFO --- 6910/6910 batches finished ---
2022-01-22 14:09:04,667 P804 INFO Train loss: 0.392363
2022-01-22 14:09:04,667 P804 INFO Training finished.
2022-01-22 14:09:04,667 P804 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/AutoInt_avazu_x1/avazu_x1_3fb65689/AutoInt_avazu_x1_004_4fe23ce2.model
2022-01-22 14:09:09,243 P804 INFO ****** Validation evaluation ******
2022-01-22 14:10:17,834 P804 INFO [Metrics] AUC: 0.746804 - logloss: 0.395706
2022-01-22 14:10:17,898 P804 INFO ******** Test evaluation ********
2022-01-22 14:10:17,898 P804 INFO Loading data...
2022-01-22 14:10:17,899 P804 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-01-22 14:10:18,577 P804 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-01-22 14:10:18,577 P804 INFO Loading test data done.
2022-01-22 14:12:35,635 P804 INFO [Metrics] AUC: 0.765027 - logloss: 0.366462

```
