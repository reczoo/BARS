## AOANet_avazu_x1

A hands-on guide to run the AOANet model on the Avazu_x1 dataset.

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
  fuxictr: 1.2.1

  ```

### Dataset
Dataset ID: [Avazu_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu#Avazu_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [AOANet](https://github.com/xue-pai/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/AOANet.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AOANet_avazu_x1_tuner_config_01](./AOANet_avazu_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AOANet_avazu_x1
    nohup python run_expid.py --config ./AOANet_avazu_x1_tuner_config_01 --expid AOANet_avazu_x1_004_a663f0bb --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.765390 | 0.366352  |


### Logs
```python
2022-05-31 10:44:41,754 P20282 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "dnn_hidden_activations": "ReLU",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.1",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "3",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AOANet",
    "model_id": "AOANet_avazu_x1_004_a663f0bb",
    "model_root": "./Avazu/AOANet_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_interaction_layers": "1",
    "num_subspaces": "2",
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
2022-05-31 10:44:41,754 P20282 INFO Set up feature encoder...
2022-05-31 10:44:41,755 P20282 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-05-31 10:44:41,755 P20282 INFO Loading data...
2022-05-31 10:44:41,756 P20282 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-05-31 10:44:44,227 P20282 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-05-31 10:44:44,604 P20282 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-05-31 10:44:44,604 P20282 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-05-31 10:44:44,604 P20282 INFO Loading train data done.
2022-05-31 10:44:50,693 P20282 INFO Total number of parameters: 13399199.
2022-05-31 10:44:50,694 P20282 INFO Start training: 6910 batches/epoch
2022-05-31 10:44:50,694 P20282 INFO ************ Epoch=1 start ************
2022-05-31 11:04:18,922 P20282 INFO [Metrics] AUC: 0.731240 - logloss: 0.404159
2022-05-31 11:04:18,924 P20282 INFO Save best model: monitor(max): 0.731240
2022-05-31 11:04:19,187 P20282 INFO --- 6910/6910 batches finished ---
2022-05-31 11:04:19,231 P20282 INFO Train loss: 0.446839
2022-05-31 11:04:19,231 P20282 INFO ************ Epoch=1 end ************
2022-05-31 11:23:43,789 P20282 INFO [Metrics] AUC: 0.736532 - logloss: 0.402076
2022-05-31 11:23:43,792 P20282 INFO Save best model: monitor(max): 0.736532
2022-05-31 11:23:43,862 P20282 INFO --- 6910/6910 batches finished ---
2022-05-31 11:23:43,901 P20282 INFO Train loss: 0.443322
2022-05-31 11:23:43,901 P20282 INFO ************ Epoch=2 end ************
2022-05-31 11:43:06,150 P20282 INFO [Metrics] AUC: 0.734590 - logloss: 0.401921
2022-05-31 11:43:06,152 P20282 INFO Monitor(max) STOP: 0.734590 !
2022-05-31 11:43:06,152 P20282 INFO Reduce learning rate on plateau: 0.000100
2022-05-31 11:43:06,152 P20282 INFO --- 6910/6910 batches finished ---
2022-05-31 11:43:06,197 P20282 INFO Train loss: 0.442913
2022-05-31 11:43:06,198 P20282 INFO ************ Epoch=3 end ************
2022-05-31 12:02:24,338 P20282 INFO [Metrics] AUC: 0.745001 - logloss: 0.397410
2022-05-31 12:02:24,340 P20282 INFO Save best model: monitor(max): 0.745001
2022-05-31 12:02:24,404 P20282 INFO --- 6910/6910 batches finished ---
2022-05-31 12:02:24,452 P20282 INFO Train loss: 0.411588
2022-05-31 12:02:24,452 P20282 INFO ************ Epoch=4 end ************
2022-05-31 12:21:41,677 P20282 INFO [Metrics] AUC: 0.745250 - logloss: 0.396864
2022-05-31 12:21:41,680 P20282 INFO Save best model: monitor(max): 0.745250
2022-05-31 12:21:41,754 P20282 INFO --- 6910/6910 batches finished ---
2022-05-31 12:21:41,795 P20282 INFO Train loss: 0.413135
2022-05-31 12:21:41,795 P20282 INFO ************ Epoch=5 end ************
2022-05-31 12:40:57,943 P20282 INFO [Metrics] AUC: 0.742879 - logloss: 0.398163
2022-05-31 12:40:57,945 P20282 INFO Monitor(max) STOP: 0.742879 !
2022-05-31 12:40:57,945 P20282 INFO Reduce learning rate on plateau: 0.000010
2022-05-31 12:40:57,945 P20282 INFO --- 6910/6910 batches finished ---
2022-05-31 12:40:57,992 P20282 INFO Train loss: 0.414100
2022-05-31 12:40:57,992 P20282 INFO ************ Epoch=6 end ************
2022-05-31 13:00:13,211 P20282 INFO [Metrics] AUC: 0.747211 - logloss: 0.396400
2022-05-31 13:00:13,213 P20282 INFO Save best model: monitor(max): 0.747211
2022-05-31 13:00:13,278 P20282 INFO --- 6910/6910 batches finished ---
2022-05-31 13:00:13,322 P20282 INFO Train loss: 0.398612
2022-05-31 13:00:13,322 P20282 INFO ************ Epoch=7 end ************
2022-05-31 13:19:29,209 P20282 INFO [Metrics] AUC: 0.744972 - logloss: 0.397386
2022-05-31 13:19:29,211 P20282 INFO Monitor(max) STOP: 0.744972 !
2022-05-31 13:19:29,211 P20282 INFO Reduce learning rate on plateau: 0.000001
2022-05-31 13:19:29,211 P20282 INFO --- 6910/6910 batches finished ---
2022-05-31 13:19:29,258 P20282 INFO Train loss: 0.397166
2022-05-31 13:19:29,258 P20282 INFO ************ Epoch=8 end ************
2022-05-31 13:38:41,141 P20282 INFO [Metrics] AUC: 0.742032 - logloss: 0.398859
2022-05-31 13:38:41,144 P20282 INFO Monitor(max) STOP: 0.742032 !
2022-05-31 13:38:41,144 P20282 INFO Reduce learning rate on plateau: 0.000001
2022-05-31 13:38:41,144 P20282 INFO Early stopping at epoch=9
2022-05-31 13:38:41,144 P20282 INFO --- 6910/6910 batches finished ---
2022-05-31 13:38:41,191 P20282 INFO Train loss: 0.390920
2022-05-31 13:38:41,191 P20282 INFO Training finished.
2022-05-31 13:38:41,191 P20282 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/AOANet_avazu_x1/avazu_x1_3fb65689/AOANet_avazu_x1_004_a663f0bb.model
2022-05-31 13:38:48,274 P20282 INFO ****** Validation evaluation ******
2022-05-31 13:39:14,341 P20282 INFO [Metrics] AUC: 0.747211 - logloss: 0.396400
2022-05-31 13:39:14,419 P20282 INFO ******** Test evaluation ********
2022-05-31 13:39:14,419 P20282 INFO Loading data...
2022-05-31 13:39:14,420 P20282 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-05-31 13:39:15,250 P20282 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-05-31 13:39:15,251 P20282 INFO Loading test data done.
2022-05-31 13:40:05,573 P20282 INFO [Metrics] AUC: 0.765390 - logloss: 0.366352

```
