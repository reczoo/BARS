## xDeepFM_avazu_x1

A guide to benchmark xDeepFM on [Avazu_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x1).

Author: [zhujiem](https://github.com/zhujiem)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
  GPU: Tesla P100 16G
  RAM: 755G
  ```
+ Software

  ```python
  CUDA: 10.0.130
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  ```

### Dataset

To reproduce the dataset splitting, please follow the details of [Avazu_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x1) to get data ready.

### Code

We use [FuxiCTR v1.1](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. The model implementation can be found [here](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/xDeepFM.py).

1. Install FuxiCTR and all the dependencies. 
   ```bash
   pip install fuxictr==1.1.*
   ```
   
2. Put the downloaded dataset in `../data/Avazu/Avazu_x1`. 

3. The dataset_config and model_config files are available in the sub-folder [xDeepFM_avazu_x1_tuner_config_01](./xDeepFM_avazu_x1_tuner_config_01).

   Note that in this setting, we follow the AFN work to fix embedding_dim=10, batch_size=4096, and MLP_hidden_units=[400, 400, 400] to make fair comparisons. Other hyper-parameters are tuned via grid search.

4. Run the following script to start.

  ```bash
  cd BARS/ctr_prediction/benchmarks/xDeepFM/xDeepFM_avazu_x1
  nohup python run_expid.py --version pytorch --config xDeepFM_avazu_x1_tuner_config_01 --expid xDeepFM_avazu_x1_013_667f616d --gpu 0 > run.log & 
  tail -f run.log
  ```

### Results
```python
[Metrics] AUC: 0.764282 - logloss: 0.366885
```

### Logs
```python
2021-12-30 16:56:06,772 P48107 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "cin_layer_units": "[32, 32, 32]",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "xDeepFM",
    "model_id": "xDeepFM_avazu_x1_013_667f616d",
    "model_root": "./Avazu/xDeepFM_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
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
2021-12-30 16:56:06,773 P48107 INFO Set up feature encoder...
2021-12-30 16:56:06,773 P48107 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2021-12-30 16:56:06,773 P48107 INFO Loading data...
2021-12-30 16:56:06,775 P48107 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2021-12-30 16:56:09,297 P48107 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2021-12-30 16:56:09,652 P48107 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2021-12-30 16:56:09,653 P48107 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2021-12-30 16:56:09,653 P48107 INFO Loading train data done.
2021-12-30 16:56:12,331 P48107 INFO Total number of parameters: 14754927.
2021-12-30 16:56:12,331 P48107 INFO Start training: 6910 batches/epoch
2021-12-30 16:56:12,331 P48107 INFO ************ Epoch=1 start ************
2021-12-30 17:10:29,229 P48107 INFO [Metrics] AUC: 0.743193 - logloss: 0.400409
2021-12-30 17:10:29,231 P48107 INFO Save best model: monitor(max): 0.743193
2021-12-30 17:10:29,296 P48107 INFO --- 6910/6910 batches finished ---
2021-12-30 17:10:29,334 P48107 INFO Train loss: 0.420883
2021-12-30 17:10:29,335 P48107 INFO ************ Epoch=1 end ************
2021-12-30 17:24:46,193 P48107 INFO [Metrics] AUC: 0.740728 - logloss: 0.398915
2021-12-30 17:24:46,197 P48107 INFO Monitor(max) STOP: 0.740728 !
2021-12-30 17:24:46,198 P48107 INFO Reduce learning rate on plateau: 0.000100
2021-12-30 17:24:46,198 P48107 INFO --- 6910/6910 batches finished ---
2021-12-30 17:24:46,237 P48107 INFO Train loss: 0.421738
2021-12-30 17:24:46,237 P48107 INFO ************ Epoch=2 end ************
2021-12-30 17:39:03,143 P48107 INFO [Metrics] AUC: 0.747070 - logloss: 0.396057
2021-12-30 17:39:03,147 P48107 INFO Save best model: monitor(max): 0.747070
2021-12-30 17:39:03,232 P48107 INFO --- 6910/6910 batches finished ---
2021-12-30 17:39:03,272 P48107 INFO Train loss: 0.398934
2021-12-30 17:39:03,272 P48107 INFO ************ Epoch=3 end ************
2021-12-30 17:53:19,966 P48107 INFO [Metrics] AUC: 0.744084 - logloss: 0.398645
2021-12-30 17:53:19,973 P48107 INFO Monitor(max) STOP: 0.744084 !
2021-12-30 17:53:19,973 P48107 INFO Reduce learning rate on plateau: 0.000010
2021-12-30 17:53:19,973 P48107 INFO --- 6910/6910 batches finished ---
2021-12-30 17:53:20,013 P48107 INFO Train loss: 0.395411
2021-12-30 17:53:20,013 P48107 INFO ************ Epoch=4 end ************
2021-12-30 18:07:39,495 P48107 INFO [Metrics] AUC: 0.736562 - logloss: 0.402197
2021-12-30 18:07:39,499 P48107 INFO Monitor(max) STOP: 0.736562 !
2021-12-30 18:07:39,499 P48107 INFO Reduce learning rate on plateau: 0.000001
2021-12-30 18:07:39,499 P48107 INFO Early stopping at epoch=5
2021-12-30 18:07:39,499 P48107 INFO --- 6910/6910 batches finished ---
2021-12-30 18:07:39,539 P48107 INFO Train loss: 0.384235
2021-12-30 18:07:39,539 P48107 INFO Training finished.
2021-12-30 18:07:39,540 P48107 INFO Load best model: /home/ma-user/work/github/benchmarks/Avazu/xDeepFM_avazu_x1/avazu_x1_3fb65689/xDeepFM_avazu_x1_013_667f616d.model
2021-12-30 18:07:39,615 P48107 INFO ****** Validation evaluation ******
2021-12-30 18:07:54,651 P48107 INFO [Metrics] AUC: 0.747070 - logloss: 0.396057
2021-12-30 18:07:54,696 P48107 INFO ******** Test evaluation ********
2021-12-30 18:07:54,696 P48107 INFO Loading data...
2021-12-30 18:07:54,696 P48107 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2021-12-30 18:07:55,477 P48107 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2021-12-30 18:07:55,477 P48107 INFO Loading test data done.
2021-12-30 18:08:26,152 P48107 INFO [Metrics] AUC: 0.764282 - logloss: 0.366885

```
