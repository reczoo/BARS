## AutoInt_avazu_x1

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

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [AutoInt](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt_avazu_x1_tuner_config_02](./AutoInt_avazu_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AutoInt_avazu_x1
    nohup python run_expid.py --config ./AutoInt_avazu_x1_tuner_config_02 --expid AutoInt_avazu_x1_004_37380bc5 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.761814 | 0.368381  |


### Logs
```python
2022-01-21 07:10:38,002 P800 INFO {
    "attention_dim": "32",
    "attention_layers": "4",
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
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
    "model_id": "AutoInt_avazu_x1_004_37380bc5",
    "model_root": "./Avazu/AutoInt_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "1",
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
2022-01-21 07:10:38,003 P800 INFO Set up feature encoder...
2022-01-21 07:10:38,003 P800 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-01-21 07:10:38,003 P800 INFO Loading data...
2022-01-21 07:10:38,004 P800 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-01-21 07:10:40,406 P800 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-01-21 07:10:40,823 P800 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-01-21 07:10:40,823 P800 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-01-21 07:10:40,823 P800 INFO Loading train data done.
2022-01-21 07:10:46,922 P800 INFO Total number of parameters: 12997447.
2022-01-21 07:10:46,923 P800 INFO Start training: 6910 batches/epoch
2022-01-21 07:10:46,923 P800 INFO ************ Epoch=1 start ************
2022-01-21 07:27:43,783 P800 INFO [Metrics] AUC: 0.738708 - logloss: 0.399528
2022-01-21 07:27:43,785 P800 INFO Save best model: monitor(max): 0.738708
2022-01-21 07:27:44,032 P800 INFO --- 6910/6910 batches finished ---
2022-01-21 07:27:44,067 P800 INFO Train loss: 0.470325
2022-01-21 07:27:44,068 P800 INFO ************ Epoch=1 end ************
2022-01-21 07:44:36,591 P800 INFO [Metrics] AUC: 0.741134 - logloss: 0.398834
2022-01-21 07:44:36,595 P800 INFO Save best model: monitor(max): 0.741134
2022-01-21 07:44:36,668 P800 INFO --- 6910/6910 batches finished ---
2022-01-21 07:44:36,712 P800 INFO Train loss: 0.479222
2022-01-21 07:44:36,712 P800 INFO ************ Epoch=2 end ************
2022-01-21 08:01:25,274 P800 INFO [Metrics] AUC: 0.740770 - logloss: 0.398918
2022-01-21 08:01:25,276 P800 INFO Monitor(max) STOP: 0.740770 !
2022-01-21 08:01:25,276 P800 INFO Reduce learning rate on plateau: 0.000100
2022-01-21 08:01:25,276 P800 INFO --- 6910/6910 batches finished ---
2022-01-21 08:01:25,313 P800 INFO Train loss: 0.490323
2022-01-21 08:01:25,313 P800 INFO ************ Epoch=3 end ************
2022-01-21 08:18:05,938 P800 INFO [Metrics] AUC: 0.747576 - logloss: 0.395330
2022-01-21 08:18:05,942 P800 INFO Save best model: monitor(max): 0.747576
2022-01-21 08:18:06,013 P800 INFO --- 6910/6910 batches finished ---
2022-01-21 08:18:06,051 P800 INFO Train loss: 0.418341
2022-01-21 08:18:06,051 P800 INFO ************ Epoch=4 end ************
2022-01-21 08:34:46,240 P800 INFO [Metrics] AUC: 0.746884 - logloss: 0.395974
2022-01-21 08:34:46,242 P800 INFO Monitor(max) STOP: 0.746884 !
2022-01-21 08:34:46,242 P800 INFO Reduce learning rate on plateau: 0.000010
2022-01-21 08:34:46,242 P800 INFO --- 6910/6910 batches finished ---
2022-01-21 08:34:46,279 P800 INFO Train loss: 0.424649
2022-01-21 08:34:46,279 P800 INFO ************ Epoch=5 end ************
2022-01-21 08:51:27,728 P800 INFO [Metrics] AUC: 0.743268 - logloss: 0.398538
2022-01-21 08:51:27,731 P800 INFO Monitor(max) STOP: 0.743268 !
2022-01-21 08:51:27,731 P800 INFO Reduce learning rate on plateau: 0.000001
2022-01-21 08:51:27,731 P800 INFO Early stopping at epoch=6
2022-01-21 08:51:27,731 P800 INFO --- 6910/6910 batches finished ---
2022-01-21 08:51:27,768 P800 INFO Train loss: 0.398064
2022-01-21 08:51:27,768 P800 INFO Training finished.
2022-01-21 08:51:27,768 P800 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/AutoInt_avazu_x1/avazu_x1_3fb65689/AutoInt_avazu_x1_004_37380bc5.model
2022-01-21 08:51:34,067 P800 INFO ****** Validation evaluation ******
2022-01-21 08:52:05,597 P800 INFO [Metrics] AUC: 0.747576 - logloss: 0.395330
2022-01-21 08:52:05,677 P800 INFO ******** Test evaluation ********
2022-01-21 08:52:05,677 P800 INFO Loading data...
2022-01-21 08:52:05,677 P800 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-01-21 08:52:06,322 P800 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-01-21 08:52:06,322 P800 INFO Loading test data done.
2022-01-21 08:52:58,311 P800 INFO [Metrics] AUC: 0.761814 - logloss: 0.368381

```
