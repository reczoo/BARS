## AFN+_avazu_x1

A hands-on guide to run the AFN model on the Avazu_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [AFN](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/AFN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFN+_avazu_x1_tuner_config_01](./AFN+_avazu_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFN+_avazu_x1
    nohup python run_expid.py --config ./AFN+_avazu_x1_tuner_config_01 --expid AFN_avazu_x1_013_9015d34c --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.763985 | 0.367093  |
| 2 | 0.764438 | 0.366959  |
| 3 | 0.763496 | 0.367529  |
| 4 | 0.764488 | 0.367240  |
| 5 | 0.764320 | 0.366922  |
| | | | 
| Avg | 0.764145 | 0.367149 |
| Std | &#177;0.00036903 | &#177;0.00022066 |


### Logs
```python
2022-01-24 07:32:10,806 P45727 INFO {
    "afn_activations": "relu",
    "afn_dropout": "0.2",
    "afn_hidden_units": "[1200]",
    "batch_norm": "True",
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_dropout": "0",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "ensemble_dnn": "True",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "logarithmic_neurons": "1500",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AFN",
    "model_id": "AFN_avazu_x1_013_9015d34c",
    "model_root": "./Avazu/AFN_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
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
2022-01-24 07:32:10,807 P45727 INFO Set up feature encoder...
2022-01-24 07:32:10,807 P45727 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-01-24 07:32:10,807 P45727 INFO Loading data...
2022-01-24 07:32:10,810 P45727 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-01-24 07:32:13,582 P45727 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-01-24 07:32:13,997 P45727 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-01-24 07:32:13,997 P45727 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-01-24 07:32:13,997 P45727 INFO Loading train data done.
2022-01-24 07:32:19,127 P45727 INFO Total number of parameters: 44424829.
2022-01-24 07:32:19,127 P45727 INFO Start training: 6910 batches/epoch
2022-01-24 07:32:19,128 P45727 INFO ************ Epoch=1 start ************
2022-01-24 08:32:48,826 P45727 INFO [Metrics] AUC: 0.737307 - logloss: 0.400300
2022-01-24 08:32:48,830 P45727 INFO Save best model: monitor(max): 0.737307
2022-01-24 08:32:49,010 P45727 INFO --- 6910/6910 batches finished ---
2022-01-24 08:32:49,047 P45727 INFO Train loss: 0.550978
2022-01-24 08:32:49,047 P45727 INFO ************ Epoch=1 end ************
2022-01-24 09:33:14,194 P45727 INFO [Metrics] AUC: 0.743247 - logloss: 0.397953
2022-01-24 09:33:14,197 P45727 INFO Save best model: monitor(max): 0.743247
2022-01-24 09:33:14,473 P45727 INFO --- 6910/6910 batches finished ---
2022-01-24 09:33:14,508 P45727 INFO Train loss: 0.572475
2022-01-24 09:33:14,508 P45727 INFO ************ Epoch=2 end ************
2022-01-24 10:34:22,135 P45727 INFO [Metrics] AUC: 0.739823 - logloss: 0.398981
2022-01-24 10:34:22,141 P45727 INFO Monitor(max) STOP: 0.739823 !
2022-01-24 10:34:22,141 P45727 INFO Reduce learning rate on plateau: 0.000100
2022-01-24 10:34:22,141 P45727 INFO --- 6910/6910 batches finished ---
2022-01-24 10:34:22,186 P45727 INFO Train loss: 0.540273
2022-01-24 10:34:22,186 P45727 INFO ************ Epoch=3 end ************
2022-01-24 11:36:41,921 P45727 INFO [Metrics] AUC: 0.747597 - logloss: 0.395265
2022-01-24 11:36:41,925 P45727 INFO Save best model: monitor(max): 0.747597
2022-01-24 11:36:42,226 P45727 INFO --- 6910/6910 batches finished ---
2022-01-24 11:36:42,262 P45727 INFO Train loss: 0.419225
2022-01-24 11:36:42,262 P45727 INFO ************ Epoch=4 end ************
2022-01-24 12:39:59,290 P45727 INFO [Metrics] AUC: 0.747635 - logloss: 0.395563
2022-01-24 12:39:59,295 P45727 INFO Save best model: monitor(max): 0.747635
2022-01-24 12:39:59,601 P45727 INFO --- 6910/6910 batches finished ---
2022-01-24 12:39:59,637 P45727 INFO Train loss: 0.407993
2022-01-24 12:39:59,638 P45727 INFO ************ Epoch=5 end ************
2022-01-24 13:45:21,155 P45727 INFO [Metrics] AUC: 0.744961 - logloss: 0.396460
2022-01-24 13:45:21,160 P45727 INFO Monitor(max) STOP: 0.744961 !
2022-01-24 13:45:21,160 P45727 INFO Reduce learning rate on plateau: 0.000010
2022-01-24 13:45:21,160 P45727 INFO --- 6910/6910 batches finished ---
2022-01-24 13:45:21,203 P45727 INFO Train loss: 0.409913
2022-01-24 13:45:21,203 P45727 INFO ************ Epoch=6 end ************
2022-01-24 14:51:01,732 P45727 INFO [Metrics] AUC: 0.743107 - logloss: 0.397600
2022-01-24 14:51:01,737 P45727 INFO Monitor(max) STOP: 0.743107 !
2022-01-24 14:51:01,737 P45727 INFO Reduce learning rate on plateau: 0.000001
2022-01-24 14:51:01,737 P45727 INFO Early stopping at epoch=7
2022-01-24 14:51:01,737 P45727 INFO --- 6910/6910 batches finished ---
2022-01-24 14:51:01,784 P45727 INFO Train loss: 0.392615
2022-01-24 14:51:01,784 P45727 INFO Training finished.
2022-01-24 14:51:01,784 P45727 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Avazu/AFN_avazu_x1/avazu_x1_3fb65689/AFN_avazu_x1_013_9015d34c.model
2022-01-24 14:51:06,867 P45727 INFO ****** Validation evaluation ******
2022-01-24 14:52:46,459 P45727 INFO [Metrics] AUC: 0.747635 - logloss: 0.395563
2022-01-24 14:52:46,514 P45727 INFO ******** Test evaluation ********
2022-01-24 14:52:46,514 P45727 INFO Loading data...
2022-01-24 14:52:46,515 P45727 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-01-24 14:52:47,240 P45727 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-01-24 14:52:47,240 P45727 INFO Loading test data done.
2022-01-24 14:55:17,183 P45727 INFO [Metrics] AUC: 0.764824 - logloss: 0.367235

```
