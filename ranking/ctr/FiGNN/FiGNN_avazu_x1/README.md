## FiGNN_avazu_x1

A hands-on guide to run the FiGNN model on the Avazu_x1 dataset.

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
  fuxictr: 1.1.0

  ```

### Dataset
Dataset ID: [Avazu_x1](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [FiGNN](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FiGNN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FiGNN_avazu_x1_tuner_config_02](./FiGNN_avazu_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FiGNN_avazu_x1
    nohup python run_expid.py --config ./FiGNN_avazu_x1_tuner_config_02 --expid FiGNN_avazu_x1_015_9c5d8df1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.762188 | 0.368129  |


### Logs
```python
2022-01-26 14:07:11,120 P11492 INFO {
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.0001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gnn_layers": "8",
    "gpu": "6",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FiGNN",
    "model_id": "FiGNN_avazu_x1_015_9c5d8df1",
    "model_root": "./Avazu/FiGNN_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reuse_graph_layer": "False",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_gru": "False",
    "use_hdf5": "True",
    "use_residual": "True",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-01-26 14:07:11,120 P11492 INFO Set up feature encoder...
2022-01-26 14:07:11,121 P11492 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-01-26 14:07:11,121 P11492 INFO Loading data...
2022-01-26 14:07:11,121 P11492 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-01-26 14:07:14,034 P11492 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-01-26 14:07:14,395 P11492 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-01-26 14:07:14,396 P11492 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-01-26 14:07:14,396 P11492 INFO Loading train data done.
2022-01-26 14:07:22,331 P11492 INFO Total number of parameters: 13026140.
2022-01-26 14:07:22,331 P11492 INFO Start training: 6910 batches/epoch
2022-01-26 14:07:22,332 P11492 INFO ************ Epoch=1 start ************
2022-01-26 14:41:49,479 P11492 INFO [Metrics] AUC: 0.745805 - logloss: 0.397408
2022-01-26 14:41:49,482 P11492 INFO Save best model: monitor(max): 0.745805
2022-01-26 14:41:49,741 P11492 INFO --- 6910/6910 batches finished ---
2022-01-26 14:41:49,786 P11492 INFO Train loss: 0.409241
2022-01-26 14:41:49,786 P11492 INFO ************ Epoch=1 end ************
2022-01-26 15:16:17,198 P11492 INFO [Metrics] AUC: 0.747006 - logloss: 0.396994
2022-01-26 15:16:17,201 P11492 INFO Save best model: monitor(max): 0.747006
2022-01-26 15:16:17,265 P11492 INFO --- 6910/6910 batches finished ---
2022-01-26 15:16:17,308 P11492 INFO Train loss: 0.406820
2022-01-26 15:16:17,308 P11492 INFO ************ Epoch=2 end ************
2022-01-26 15:50:45,106 P11492 INFO [Metrics] AUC: 0.744692 - logloss: 0.396886
2022-01-26 15:50:45,108 P11492 INFO Monitor(max) STOP: 0.744692 !
2022-01-26 15:50:45,108 P11492 INFO Reduce learning rate on plateau: 0.000100
2022-01-26 15:50:45,108 P11492 INFO --- 6910/6910 batches finished ---
2022-01-26 15:50:45,150 P11492 INFO Train loss: 0.406138
2022-01-26 15:50:45,150 P11492 INFO ************ Epoch=3 end ************
2022-01-26 16:25:05,277 P11492 INFO [Metrics] AUC: 0.743396 - logloss: 0.399163
2022-01-26 16:25:05,279 P11492 INFO Monitor(max) STOP: 0.743396 !
2022-01-26 16:25:05,279 P11492 INFO Reduce learning rate on plateau: 0.000010
2022-01-26 16:25:05,279 P11492 INFO Early stopping at epoch=4
2022-01-26 16:25:05,279 P11492 INFO --- 6910/6910 batches finished ---
2022-01-26 16:25:05,322 P11492 INFO Train loss: 0.391849
2022-01-26 16:25:05,322 P11492 INFO Training finished.
2022-01-26 16:25:05,322 P11492 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/FiGNN_avazu_x1/avazu_x1_3fb65689/FiGNN_avazu_x1_015_9c5d8df1.model
2022-01-26 16:25:12,557 P11492 INFO ****** Validation evaluation ******
2022-01-26 16:25:46,032 P11492 INFO [Metrics] AUC: 0.747006 - logloss: 0.396994
2022-01-26 16:25:46,103 P11492 INFO ******** Test evaluation ********
2022-01-26 16:25:46,103 P11492 INFO Loading data...
2022-01-26 16:25:46,104 P11492 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-01-26 16:25:46,808 P11492 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-01-26 16:25:46,808 P11492 INFO Loading test data done.
2022-01-26 16:26:56,045 P11492 INFO [Metrics] AUC: 0.762188 - logloss: 0.368129

```
