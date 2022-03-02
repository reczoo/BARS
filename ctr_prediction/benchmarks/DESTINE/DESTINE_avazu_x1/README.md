## DESTINE_avazu_x1

A hands-on guide to run the DESTINE model on the Avazu_x1 dataset.

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
  fuxictr: 1.1.1

  ```

### Dataset
Dataset ID: [Avazu_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.1](fuxictr_url) for this experiment. See model code: [DESTINE](https://github.com/xue-pai/FuxiCTR/blob/v1.1.1/fuxictr/pytorch/models/DESTINE.py).

Running steps:

1. Download [FuxiCTR-v1.1.1](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DESTINE_avazu_x1_tuner_config_02](./DESTINE_avazu_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DESTINE_avazu_x1
    nohup python run_expid.py --config ./DESTINE_avazu_x1_tuner_config_02 --expid DESTINE_avazu_x1_016_61d63533 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.766051 | 0.366124  |


### Logs
```python
2022-02-19 08:00:44,920 P80226 INFO {
    "att_dropout": "0",
    "attention_dim": "64",
    "attention_layers": "3",
    "batch_norm": "True",
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.1",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "7",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DESTINE",
    "model_id": "DESTINE_avazu_x1_016_61d63533",
    "model_root": "./Avazu/DESTINE_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_heads": "1",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "relu_before_att": "False",
    "residual_mode": "each_layer",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_hdf5": "True",
    "use_scale": "True",
    "use_wide": "False",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-02-19 08:00:44,920 P80226 INFO Set up feature encoder...
2022-02-19 08:00:44,921 P80226 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-02-19 08:00:44,921 P80226 INFO Loading data...
2022-02-19 08:00:44,922 P80226 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-02-19 08:00:47,509 P80226 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-02-19 08:00:47,823 P80226 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-02-19 08:00:47,824 P80226 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-02-19 08:00:47,824 P80226 INFO Loading train data done.
2022-02-19 08:00:53,834 P80226 INFO Total number of parameters: 13435637.
2022-02-19 08:00:53,834 P80226 INFO Start training: 6910 batches/epoch
2022-02-19 08:00:53,834 P80226 INFO ************ Epoch=1 start ************
2022-02-19 08:18:52,569 P80226 INFO [Metrics] AUC: 0.736735 - logloss: 0.401595
2022-02-19 08:18:52,572 P80226 INFO Save best model: monitor(max): 0.736735
2022-02-19 08:18:52,830 P80226 INFO --- 6910/6910 batches finished ---
2022-02-19 08:18:52,866 P80226 INFO Train loss: 0.446233
2022-02-19 08:18:52,866 P80226 INFO ************ Epoch=1 end ************
2022-02-19 08:36:53,559 P80226 INFO [Metrics] AUC: 0.735627 - logloss: 0.401445
2022-02-19 08:36:53,561 P80226 INFO Monitor(max) STOP: 0.735627 !
2022-02-19 08:36:53,561 P80226 INFO Reduce learning rate on plateau: 0.000100
2022-02-19 08:36:53,561 P80226 INFO --- 6910/6910 batches finished ---
2022-02-19 08:36:53,598 P80226 INFO Train loss: 0.443054
2022-02-19 08:36:53,598 P80226 INFO ************ Epoch=2 end ************
2022-02-19 08:54:54,482 P80226 INFO [Metrics] AUC: 0.744797 - logloss: 0.397422
2022-02-19 08:54:54,484 P80226 INFO Save best model: monitor(max): 0.744797
2022-02-19 08:54:54,567 P80226 INFO --- 6910/6910 batches finished ---
2022-02-19 08:54:54,607 P80226 INFO Train loss: 0.412016
2022-02-19 08:54:54,607 P80226 INFO ************ Epoch=3 end ************
2022-02-19 09:12:55,380 P80226 INFO [Metrics] AUC: 0.746245 - logloss: 0.396002
2022-02-19 09:12:55,382 P80226 INFO Save best model: monitor(max): 0.746245
2022-02-19 09:12:55,452 P80226 INFO --- 6910/6910 batches finished ---
2022-02-19 09:12:55,498 P80226 INFO Train loss: 0.413840
2022-02-19 09:12:55,498 P80226 INFO ************ Epoch=4 end ************
2022-02-19 09:30:55,617 P80226 INFO [Metrics] AUC: 0.747488 - logloss: 0.395961
2022-02-19 09:30:55,620 P80226 INFO Save best model: monitor(max): 0.747488
2022-02-19 09:30:55,684 P80226 INFO --- 6910/6910 batches finished ---
2022-02-19 09:30:55,720 P80226 INFO Train loss: 0.414996
2022-02-19 09:30:55,721 P80226 INFO ************ Epoch=5 end ************
2022-02-19 09:48:54,945 P80226 INFO [Metrics] AUC: 0.746340 - logloss: 0.396135
2022-02-19 09:48:54,948 P80226 INFO Monitor(max) STOP: 0.746340 !
2022-02-19 09:48:54,948 P80226 INFO Reduce learning rate on plateau: 0.000010
2022-02-19 09:48:54,948 P80226 INFO --- 6910/6910 batches finished ---
2022-02-19 09:48:54,984 P80226 INFO Train loss: 0.415613
2022-02-19 09:48:54,984 P80226 INFO ************ Epoch=6 end ************
2022-02-19 10:06:53,515 P80226 INFO [Metrics] AUC: 0.748386 - logloss: 0.395397
2022-02-19 10:06:53,517 P80226 INFO Save best model: monitor(max): 0.748386
2022-02-19 10:06:53,582 P80226 INFO --- 6910/6910 batches finished ---
2022-02-19 10:06:53,624 P80226 INFO Train loss: 0.398358
2022-02-19 10:06:53,625 P80226 INFO ************ Epoch=7 end ************
2022-02-19 10:24:50,750 P80226 INFO [Metrics] AUC: 0.747080 - logloss: 0.396037
2022-02-19 10:24:50,753 P80226 INFO Monitor(max) STOP: 0.747080 !
2022-02-19 10:24:50,753 P80226 INFO Reduce learning rate on plateau: 0.000001
2022-02-19 10:24:50,753 P80226 INFO --- 6910/6910 batches finished ---
2022-02-19 10:24:50,791 P80226 INFO Train loss: 0.396583
2022-02-19 10:24:50,791 P80226 INFO ************ Epoch=8 end ************
2022-02-19 10:42:48,257 P80226 INFO [Metrics] AUC: 0.745653 - logloss: 0.397259
2022-02-19 10:42:48,259 P80226 INFO Monitor(max) STOP: 0.745653 !
2022-02-19 10:42:48,259 P80226 INFO Reduce learning rate on plateau: 0.000001
2022-02-19 10:42:48,259 P80226 INFO Early stopping at epoch=9
2022-02-19 10:42:48,259 P80226 INFO --- 6910/6910 batches finished ---
2022-02-19 10:42:48,297 P80226 INFO Train loss: 0.390062
2022-02-19 10:42:48,298 P80226 INFO Training finished.
2022-02-19 10:42:48,298 P80226 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/DESTINE_avazu_x1/avazu_x1_3fb65689/DESTINE_avazu_x1_016_61d63533.model
2022-02-19 10:42:50,933 P80226 INFO ****** Validation evaluation ******
2022-02-19 10:43:11,722 P80226 INFO [Metrics] AUC: 0.748386 - logloss: 0.395397
2022-02-19 10:43:11,813 P80226 INFO ******** Test evaluation ********
2022-02-19 10:43:11,813 P80226 INFO Loading data...
2022-02-19 10:43:11,813 P80226 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-02-19 10:43:12,660 P80226 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-02-19 10:43:12,660 P80226 INFO Loading test data done.
2022-02-19 10:43:55,194 P80226 INFO [Metrics] AUC: 0.766051 - logloss: 0.366124

```
