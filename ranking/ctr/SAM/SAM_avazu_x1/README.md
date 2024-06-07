## SAM_avazu_x1

A hands-on guide to run the SAM model on the Avazu_x1 dataset.

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
  fuxictr: 1.2.1

  ```

### Dataset
Dataset ID: [Avazu_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu#Avazu_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.1](https://github.com/reczoo/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [SAM](https://github.com/reczoo/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/SAM.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [SAM_avazu_x1_tuner_config_01](./SAM_avazu_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd SAM_avazu_x1
    nohup python run_expid.py --config ./SAM_avazu_x1_tuner_config_01 --expid SAM_avazu_x1_012_1cecba8c --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.763179 | 0.367202  |


### Logs
```python
2022-05-28 11:54:50,811 P80006 INFO {
    "aggregation": "concat",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "3",
    "interaction_type": "SAM3A",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "SAM",
    "model_id": "SAM_avazu_x1_012_1cecba8c",
    "model_root": "./Avazu/SAM_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_interaction_layers": "4",
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
    "use_residual": "True",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-05-28 11:54:50,812 P80006 INFO Set up feature encoder...
2022-05-28 11:54:50,812 P80006 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-05-28 11:54:50,812 P80006 INFO Loading data...
2022-05-28 11:54:50,813 P80006 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-05-28 11:54:53,353 P80006 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-05-28 11:54:53,740 P80006 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-05-28 11:54:53,740 P80006 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-05-28 11:54:53,740 P80006 INFO Loading train data done.
2022-05-28 11:55:00,146 P80006 INFO Total number of parameters: 13006371.
2022-05-28 11:55:00,146 P80006 INFO Start training: 6910 batches/epoch
2022-05-28 11:55:00,146 P80006 INFO ************ Epoch=1 start ************
2022-05-28 12:05:05,187 P80006 INFO [Metrics] AUC: 0.736598 - logloss: 0.403137
2022-05-28 12:05:05,191 P80006 INFO Save best model: monitor(max): 0.736598
2022-05-28 12:05:05,477 P80006 INFO --- 6910/6910 batches finished ---
2022-05-28 12:05:05,520 P80006 INFO Train loss: 0.432452
2022-05-28 12:05:05,520 P80006 INFO ************ Epoch=1 end ************
2022-05-28 12:15:07,944 P80006 INFO [Metrics] AUC: 0.738395 - logloss: 0.400699
2022-05-28 12:15:07,946 P80006 INFO Save best model: monitor(max): 0.738395
2022-05-28 12:15:08,009 P80006 INFO --- 6910/6910 batches finished ---
2022-05-28 12:15:08,055 P80006 INFO Train loss: 0.428933
2022-05-28 12:15:08,055 P80006 INFO ************ Epoch=2 end ************
2022-05-28 12:25:10,134 P80006 INFO [Metrics] AUC: 0.736692 - logloss: 0.400102
2022-05-28 12:25:10,137 P80006 INFO Monitor(max) STOP: 0.736692 !
2022-05-28 12:25:10,137 P80006 INFO Reduce learning rate on plateau: 0.000100
2022-05-28 12:25:10,137 P80006 INFO --- 6910/6910 batches finished ---
2022-05-28 12:25:10,190 P80006 INFO Train loss: 0.429232
2022-05-28 12:25:10,190 P80006 INFO ************ Epoch=3 end ************
2022-05-28 12:35:10,171 P80006 INFO [Metrics] AUC: 0.744484 - logloss: 0.396703
2022-05-28 12:35:10,173 P80006 INFO Save best model: monitor(max): 0.744484
2022-05-28 12:35:10,236 P80006 INFO --- 6910/6910 batches finished ---
2022-05-28 12:35:10,282 P80006 INFO Train loss: 0.406561
2022-05-28 12:35:10,282 P80006 INFO ************ Epoch=4 end ************
2022-05-28 12:45:08,190 P80006 INFO [Metrics] AUC: 0.745908 - logloss: 0.396222
2022-05-28 12:45:08,193 P80006 INFO Save best model: monitor(max): 0.745908
2022-05-28 12:45:08,260 P80006 INFO --- 6910/6910 batches finished ---
2022-05-28 12:45:08,308 P80006 INFO Train loss: 0.406881
2022-05-28 12:45:08,309 P80006 INFO ************ Epoch=5 end ************
2022-05-28 12:55:05,921 P80006 INFO [Metrics] AUC: 0.745192 - logloss: 0.396466
2022-05-28 12:55:05,923 P80006 INFO Monitor(max) STOP: 0.745192 !
2022-05-28 12:55:05,923 P80006 INFO Reduce learning rate on plateau: 0.000010
2022-05-28 12:55:05,923 P80006 INFO --- 6910/6910 batches finished ---
2022-05-28 12:55:05,970 P80006 INFO Train loss: 0.407527
2022-05-28 12:55:05,970 P80006 INFO ************ Epoch=6 end ************
2022-05-28 13:05:01,874 P80006 INFO [Metrics] AUC: 0.747762 - logloss: 0.395219
2022-05-28 13:05:01,876 P80006 INFO Save best model: monitor(max): 0.747762
2022-05-28 13:05:01,948 P80006 INFO --- 6910/6910 batches finished ---
2022-05-28 13:05:02,009 P80006 INFO Train loss: 0.398262
2022-05-28 13:05:02,009 P80006 INFO ************ Epoch=7 end ************
2022-05-28 13:14:42,182 P80006 INFO [Metrics] AUC: 0.745087 - logloss: 0.396685
2022-05-28 13:14:42,184 P80006 INFO Monitor(max) STOP: 0.745087 !
2022-05-28 13:14:42,184 P80006 INFO Reduce learning rate on plateau: 0.000001
2022-05-28 13:14:42,184 P80006 INFO --- 6910/6910 batches finished ---
2022-05-28 13:14:42,238 P80006 INFO Train loss: 0.394946
2022-05-28 13:14:42,238 P80006 INFO ************ Epoch=8 end ************
2022-05-28 13:24:50,543 P80006 INFO [Metrics] AUC: 0.744029 - logloss: 0.397360
2022-05-28 13:24:50,545 P80006 INFO Monitor(max) STOP: 0.744029 !
2022-05-28 13:24:50,545 P80006 INFO Reduce learning rate on plateau: 0.000001
2022-05-28 13:24:50,545 P80006 INFO Early stopping at epoch=9
2022-05-28 13:24:50,545 P80006 INFO --- 6910/6910 batches finished ---
2022-05-28 13:24:50,600 P80006 INFO Train loss: 0.390888
2022-05-28 13:24:50,600 P80006 INFO Training finished.
2022-05-28 13:24:50,600 P80006 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/SAM_avazu_x1/avazu_x1_3fb65689/SAM_avazu_x1_012_1cecba8c.model
2022-05-28 13:24:54,920 P80006 INFO ****** Validation evaluation ******
2022-05-28 13:25:08,572 P80006 INFO [Metrics] AUC: 0.747762 - logloss: 0.395219
2022-05-28 13:25:08,651 P80006 INFO ******** Test evaluation ********
2022-05-28 13:25:08,651 P80006 INFO Loading data...
2022-05-28 13:25:08,652 P80006 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-05-28 13:25:09,526 P80006 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-05-28 13:25:09,526 P80006 INFO Loading test data done.
2022-05-28 13:25:37,331 P80006 INFO [Metrics] AUC: 0.763179 - logloss: 0.367202

```
