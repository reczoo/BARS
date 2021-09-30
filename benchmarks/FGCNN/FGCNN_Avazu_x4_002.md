## FGCNN_Avazu_x4_002

A notebook to benchmark FGCNN on Avazu_x4_002 dataset.

Author: [XUEPAI Team](https://github.com/xue-pai)


### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
  RAM: 500G+
  ```
+ Software

  ```python
  python: 3.6.5
  pandas: 1.0.0
  numpy: 1.18.1
  ```

### Dataset
In this setting, we preprocess the data split by removing the id field that is useless for CTR prediction. In addition, we transform the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we filter infrequent features by setting the threshold min_category_count=1 and replace them with a default <OOV> token. Note that we found that min_category_count=1 performs the best, which is surprising.

We fix embedding_dim=40 following the existing FGCNN work.
### Code
1. Install FuxiCTR
  
    Install FuxiCTR via `pip install fuxictr==1.0` to get all dependencies ready. Then download [the FuxiCTR repository](https://github.com/huawei-noah/benchmark/archive/53e314461c19dbc7f462b42bf0f0bfae020dc398.zip) to your local path.

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/Avazu/Avazu_x4/split_avazu_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [FGCNN_avazu_x4_tuner_config_02.yaml](./FGCNN_avazu_x4_tuner_config_02.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/002/FGCNN_avazu_x4_tuner_config_02.yaml --tag 043 --gpu 0
  ```
  

### Results
```python
[Metrics] logloss: 0.369611 - AUC: 0.797052
```


### Logs
```python
2020-02-05 11:35:33,006 P7393 INFO {
    "batch_size": "2000",
    "channels": "[14, 16, 18, 20]",
    "conv_activation": "Tanh",
    "conv_batch_norm": "True",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "dnn_activations": "ReLU",
    "dnn_batch_norm": "False",
    "dnn_hidden_units": "[4096, 2048, 1024, 512]",
    "embedding_dim": "20",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "kernel_heights": "[7, 7, 7, 7]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FGCNN",
    "model_id": "FGCNN_avazu_x4_043_daeedeac",
    "model_root": "./Avazu/FGCNN_avazu/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "pooling_sizes": "[2, 2, 2, 2]",
    "recombined_channels": "[3, 3, 3, 3]",
    "save_best_only": "True",
    "seed": "2019",
    "share_embedding": "False",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "version": "pytorch",
    "gpu": "0"
}
2020-02-05 11:35:33,007 P7393 INFO Set up feature encoder...
2020-02-05 11:35:33,007 P7393 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-02-05 11:35:33,007 P7393 INFO Loading data...
2020-02-05 11:35:33,009 P7393 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-02-05 11:35:35,397 P7393 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-02-05 11:35:37,052 P7393 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-02-05 11:35:37,223 P7393 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-02-05 11:35:37,223 P7393 INFO Loading train data done.
2020-02-05 11:35:50,692 P7393 INFO **** Start training: 16172 batches/epoch ****
2020-02-05 13:46:20,116 P7393 INFO [Metrics] logloss: 0.369694 - AUC: 0.796808
2020-02-05 13:46:20,233 P7393 INFO Save best model: monitor(max): 0.427113
2020-02-05 13:46:21,883 P7393 INFO --- 16172/16172 batches finished ---
2020-02-05 13:46:21,926 P7393 INFO Train loss: 0.379204
2020-02-05 13:46:21,926 P7393 INFO ************ Epoch=1 end ************
2020-02-05 15:56:44,409 P7393 INFO [Metrics] logloss: 0.403008 - AUC: 0.776663
2020-02-05 15:56:44,518 P7393 INFO Monitor(max) STOP: 0.373656 !
2020-02-05 15:56:44,518 P7393 INFO Reduce learning rate on plateau: 0.000100
2020-02-05 15:56:44,518 P7393 INFO --- 16172/16172 batches finished ---
2020-02-05 15:56:44,613 P7393 INFO Train loss: 0.289722
2020-02-05 15:56:44,613 P7393 INFO ************ Epoch=2 end ************
2020-02-05 18:08:25,709 P7393 INFO [Metrics] logloss: 0.515838 - AUC: 0.752300
2020-02-05 18:08:25,851 P7393 INFO Monitor(max) STOP: 0.236461 !
2020-02-05 18:08:25,851 P7393 INFO Reduce learning rate on plateau: 0.000010
2020-02-05 18:08:25,852 P7393 INFO Early stopping at epoch=3
2020-02-05 18:08:25,852 P7393 INFO --- 16172/16172 batches finished ---
2020-02-05 18:08:25,932 P7393 INFO Train loss: 0.253901
2020-02-05 18:08:25,932 P7393 INFO Training finished.
2020-02-05 18:08:25,932 P7393 INFO Load best model: /home/hispace/container/data/xxx/FuxiCTR/benchmarks/Avazu/FGCNN_avazu/avazu_x4_001_d45ad60e/FGCNN_avazu_x4_043_daeedeac_avazu_x4_001_d45ad60e_model.ckpt
2020-02-05 18:08:27,875 P7393 INFO ****** Train/validation evaluation ******
2020-02-05 18:53:29,424 P7393 INFO [Metrics] logloss: 0.322127 - AUC: 0.865346
2020-02-05 18:58:20,206 P7393 INFO [Metrics] logloss: 0.369694 - AUC: 0.796808
2020-02-05 18:58:20,374 P7393 INFO ******** Test evaluation ********
2020-02-05 18:58:20,374 P7393 INFO Loading data...
2020-02-05 18:58:20,374 P7393 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-02-05 18:58:20,835 P7393 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-02-05 18:58:20,835 P7393 INFO Loading test data done.
2020-02-05 19:03:48,382 P7393 INFO [Metrics] logloss: 0.369611 - AUC: 0.797052

```
