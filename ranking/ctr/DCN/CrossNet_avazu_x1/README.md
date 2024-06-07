## CrossNet_avazu_x1

A hands-on guide to run the DCN model on the Avazu_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DCN](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DCN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CrossNet_avazu_x1_tuner_config_01](./CrossNet_avazu_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CrossNet_avazu_x1
    nohup python run_expid.py --config ./CrossNet_avazu_x1_tuner_config_01 --expid DCN_avazu_x1_002_ffba88fe --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.759728 | 0.368561  |


### Logs
```python
2022-01-20 21:36:49,114 P51136 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "crossing_layers": "5",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "None",
    "embedding_dim": "10",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCN",
    "model_id": "DCN_avazu_x1_002_ffba88fe",
    "model_root": "./Avazu/DCN_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
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
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-20 21:36:49,115 P51136 INFO Set up feature encoder...
2022-01-20 21:36:49,115 P51136 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-01-20 21:36:49,115 P51136 INFO Loading data...
2022-01-20 21:36:49,117 P51136 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-01-20 21:36:51,340 P51136 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-01-20 21:36:51,668 P51136 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-01-20 21:36:51,668 P51136 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-01-20 21:36:51,668 P51136 INFO Loading train data done.
2022-01-20 21:36:56,455 P51136 INFO Total number of parameters: 12988411.
2022-01-20 21:36:56,456 P51136 INFO Start training: 6910 batches/epoch
2022-01-20 21:36:56,456 P51136 INFO ************ Epoch=1 start ************
2022-01-20 21:53:43,305 P51136 INFO [Metrics] AUC: 0.733988 - logloss: 0.402023
2022-01-20 21:53:43,306 P51136 INFO Save best model: monitor(max): 0.733988
2022-01-20 21:53:43,620 P51136 INFO --- 6910/6910 batches finished ---
2022-01-20 21:53:43,666 P51136 INFO Train loss: 0.417243
2022-01-20 21:53:43,667 P51136 INFO ************ Epoch=1 end ************
2022-01-20 22:10:28,360 P51136 INFO [Metrics] AUC: 0.735150 - logloss: 0.401390
2022-01-20 22:10:28,363 P51136 INFO Save best model: monitor(max): 0.735150
2022-01-20 22:10:28,447 P51136 INFO --- 6910/6910 batches finished ---
2022-01-20 22:10:28,497 P51136 INFO Train loss: 0.416061
2022-01-20 22:10:28,497 P51136 INFO ************ Epoch=2 end ************
2022-01-20 22:27:13,091 P51136 INFO [Metrics] AUC: 0.736402 - logloss: 0.400723
2022-01-20 22:27:13,095 P51136 INFO Save best model: monitor(max): 0.736402
2022-01-20 22:27:13,202 P51136 INFO --- 6910/6910 batches finished ---
2022-01-20 22:27:13,260 P51136 INFO Train loss: 0.416372
2022-01-20 22:27:13,261 P51136 INFO ************ Epoch=3 end ************
2022-01-20 22:43:51,651 P51136 INFO [Metrics] AUC: 0.738760 - logloss: 0.399420
2022-01-20 22:43:51,653 P51136 INFO Save best model: monitor(max): 0.738760
2022-01-20 22:43:51,739 P51136 INFO --- 6910/6910 batches finished ---
2022-01-20 22:43:51,791 P51136 INFO Train loss: 0.417951
2022-01-20 22:43:51,791 P51136 INFO ************ Epoch=4 end ************
2022-01-20 23:00:27,163 P51136 INFO [Metrics] AUC: 0.739578 - logloss: 0.399224
2022-01-20 23:00:27,165 P51136 INFO Save best model: monitor(max): 0.739578
2022-01-20 23:00:27,268 P51136 INFO --- 6910/6910 batches finished ---
2022-01-20 23:00:27,319 P51136 INFO Train loss: 0.418651
2022-01-20 23:00:27,319 P51136 INFO ************ Epoch=5 end ************
2022-01-20 23:17:03,245 P51136 INFO [Metrics] AUC: 0.739309 - logloss: 0.399810
2022-01-20 23:17:03,248 P51136 INFO Monitor(max) STOP: 0.739309 !
2022-01-20 23:17:03,248 P51136 INFO Reduce learning rate on plateau: 0.000100
2022-01-20 23:17:03,248 P51136 INFO --- 6910/6910 batches finished ---
2022-01-20 23:17:03,307 P51136 INFO Train loss: 0.418550
2022-01-20 23:17:03,307 P51136 INFO ************ Epoch=6 end ************
2022-01-20 23:33:38,026 P51136 INFO [Metrics] AUC: 0.743454 - logloss: 0.397401
2022-01-20 23:33:38,028 P51136 INFO Save best model: monitor(max): 0.743454
2022-01-20 23:33:38,133 P51136 INFO --- 6910/6910 batches finished ---
2022-01-20 23:33:38,183 P51136 INFO Train loss: 0.401881
2022-01-20 23:33:38,183 P51136 INFO ************ Epoch=7 end ************
2022-01-20 23:50:14,749 P51136 INFO [Metrics] AUC: 0.741686 - logloss: 0.398580
2022-01-20 23:50:14,751 P51136 INFO Monitor(max) STOP: 0.741686 !
2022-01-20 23:50:14,751 P51136 INFO Reduce learning rate on plateau: 0.000010
2022-01-20 23:50:14,751 P51136 INFO --- 6910/6910 batches finished ---
2022-01-20 23:50:14,821 P51136 INFO Train loss: 0.398635
2022-01-20 23:50:14,821 P51136 INFO ************ Epoch=8 end ************
2022-01-21 00:06:19,206 P51136 INFO [Metrics] AUC: 0.740144 - logloss: 0.400035
2022-01-21 00:06:19,208 P51136 INFO Monitor(max) STOP: 0.740144 !
2022-01-21 00:06:19,208 P51136 INFO Reduce learning rate on plateau: 0.000001
2022-01-21 00:06:19,209 P51136 INFO Early stopping at epoch=9
2022-01-21 00:06:19,209 P51136 INFO --- 6910/6910 batches finished ---
2022-01-21 00:06:19,277 P51136 INFO Train loss: 0.390773
2022-01-21 00:06:19,277 P51136 INFO Training finished.
2022-01-21 00:06:19,277 P51136 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Avazu/DCN_avazu_x1/avazu_x1_3fb65689/DCN_avazu_x1_002_ffba88fe.model
2022-01-21 00:06:22,799 P51136 INFO ****** Validation evaluation ******
2022-01-21 00:06:35,937 P51136 INFO [Metrics] AUC: 0.743454 - logloss: 0.397401
2022-01-21 00:06:35,992 P51136 INFO ******** Test evaluation ********
2022-01-21 00:06:35,992 P51136 INFO Loading data...
2022-01-21 00:06:35,993 P51136 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-01-21 00:06:36,703 P51136 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-01-21 00:06:36,703 P51136 INFO Loading test data done.
2022-01-21 00:07:02,459 P51136 INFO [Metrics] AUC: 0.759728 - logloss: 0.368561

```
