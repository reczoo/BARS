## DeepIM_avazu_x1

A guide to benchmark DeepIM on [Avazu_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x1).

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

We use [FuxiCTR v1.1](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. The model implementation can be found [here](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DeepIM.py).

1. Install FuxiCTR and all the dependencies. 
   ```bash
   pip install fuxictr==1.1.*
   ```
   
2. Put the downloaded dataset in `../data/Avazu/Avazu_x1`. 

3. The dataset_config and model_config files are available in the sub-folder [DeepIM_avazu_x1_tuner_config_03](./DeepIM_avazu_x1_tuner_config_03).

   Note that in this setting, we follow the AFN work to fix embedding_dim=10, batch_size=4096, and MLP_hidden_units=[400, 400, 400] to make fair comparisons. Other hyper-parameters are tuned via grid search.

4. Run the following script to start.

  ```bash
  cd BARS/ctr_prediction/benchmarks/DeepIM/DeepIM_avazu_x1
  nohup python run_expid.py --version pytorch --config DeepIM_avazu_x1_tuner_config_03 --expid DeepIM_avazu_x1_002_81e1625e --gpu 0 > run.log & 
  tail -f run.log
  ```

### Results
```python
[Metrics] AUC: 0.765154 - logloss: 0.366802
```

### Logs
```python
2021-12-30 07:25:10,274 P42296 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.005",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "im_batch_norm": "False",
    "im_order": "5",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DeepIM",
    "model_id": "DeepIM_avazu_x1_002_81e1625e",
    "model_root": "./Avazu/DeepIM_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_batch_norm": "True",
    "net_dropout": "0.1",
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
2021-12-30 07:25:10,275 P42296 INFO Set up feature encoder...
2021-12-30 07:25:10,275 P42296 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2021-12-30 07:25:10,275 P42296 INFO Loading data...
2021-12-30 07:25:10,277 P42296 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2021-12-30 07:25:12,981 P42296 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2021-12-30 07:25:13,365 P42296 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2021-12-30 07:25:13,365 P42296 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2021-12-30 07:25:13,366 P42296 INFO Loading train data done.
2021-12-30 07:25:15,996 P42296 INFO Total number of parameters: 13398042.
2021-12-30 07:25:15,997 P42296 INFO Start training: 6910 batches/epoch
2021-12-30 07:25:15,997 P42296 INFO ************ Epoch=1 start ************
2021-12-30 07:31:39,772 P42296 INFO [Metrics] AUC: 0.743980 - logloss: 0.397792
2021-12-30 07:31:39,775 P42296 INFO Save best model: monitor(max): 0.743980
2021-12-30 07:31:39,982 P42296 INFO --- 6910/6910 batches finished ---
2021-12-30 07:31:40,025 P42296 INFO Train loss: 0.427627
2021-12-30 07:31:40,025 P42296 INFO ************ Epoch=1 end ************
2021-12-30 07:37:59,754 P42296 INFO [Metrics] AUC: 0.743622 - logloss: 0.397643
2021-12-30 07:37:59,758 P42296 INFO Monitor(max) STOP: 0.743622 !
2021-12-30 07:37:59,758 P42296 INFO Reduce learning rate on plateau: 0.000100
2021-12-30 07:37:59,758 P42296 INFO --- 6910/6910 batches finished ---
2021-12-30 07:37:59,805 P42296 INFO Train loss: 0.426824
2021-12-30 07:37:59,805 P42296 INFO ************ Epoch=2 end ************
2021-12-30 07:44:23,566 P42296 INFO [Metrics] AUC: 0.747350 - logloss: 0.395801
2021-12-30 07:44:23,570 P42296 INFO Save best model: monitor(max): 0.747350
2021-12-30 07:44:23,651 P42296 INFO --- 6910/6910 batches finished ---
2021-12-30 07:44:23,697 P42296 INFO Train loss: 0.402395
2021-12-30 07:44:23,697 P42296 INFO ************ Epoch=3 end ************
2021-12-30 07:50:45,670 P42296 INFO [Metrics] AUC: 0.746814 - logloss: 0.395883
2021-12-30 07:50:45,673 P42296 INFO Monitor(max) STOP: 0.746814 !
2021-12-30 07:50:45,673 P42296 INFO Reduce learning rate on plateau: 0.000010
2021-12-30 07:50:45,673 P42296 INFO --- 6910/6910 batches finished ---
2021-12-30 07:50:45,721 P42296 INFO Train loss: 0.401848
2021-12-30 07:50:45,721 P42296 INFO ************ Epoch=4 end ************
2021-12-30 07:57:07,932 P42296 INFO [Metrics] AUC: 0.742981 - logloss: 0.398504
2021-12-30 07:57:07,936 P42296 INFO Monitor(max) STOP: 0.742981 !
2021-12-30 07:57:07,936 P42296 INFO Reduce learning rate on plateau: 0.000001
2021-12-30 07:57:07,936 P42296 INFO Early stopping at epoch=5
2021-12-30 07:57:07,936 P42296 INFO --- 6910/6910 batches finished ---
2021-12-30 07:57:07,982 P42296 INFO Train loss: 0.389260
2021-12-30 07:57:07,982 P42296 INFO Training finished.
2021-12-30 07:57:07,983 P42296 INFO Load best model: /home/ma-user/work/github/benchmarks/Avazu/DeepIM_avazu_x1/avazu_x1_3fb65689/DeepIM_avazu_x1_002_81e1625e.model
2021-12-30 07:57:11,268 P42296 INFO ****** Validation evaluation ******
2021-12-30 07:57:22,715 P42296 INFO [Metrics] AUC: 0.747350 - logloss: 0.395801
2021-12-30 07:57:22,751 P42296 INFO ******** Test evaluation ********
2021-12-30 07:57:22,751 P42296 INFO Loading data...
2021-12-30 07:57:22,752 P42296 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2021-12-30 07:57:23,465 P42296 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2021-12-30 07:57:23,465 P42296 INFO Loading test data done.
2021-12-30 07:57:46,876 P42296 INFO [Metrics] AUC: 0.765154 - logloss: 0.366802

```
