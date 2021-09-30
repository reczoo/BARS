## DeepCross_Avazu_x4_001

A notebook to benchmark DeepCross on Avazu_x4_001 dataset.

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
In this setting, we preprocess the data split by removing the id field that is useless for CTR prediction. In addition, we transform the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we filter infrequent features by setting the threshold min_category_count=2 (performs well) and replace them with a default <OOV> token. Note that we do not follow the exact preprocessing steps in AutoInt, because the authors neither remove the useless id field nor specially preprocess the timestamp field.

To make a fair comparison, we fix embedding_dim=16 as with AutoInt.


### Code
1. Install FuxiCTR
  
    Install FuxiCTR via `pip install fuxictr==1.0` to get all dependencies ready. Then download [the FuxiCTR repository](https://github.com/huawei-noah/benchmark/archive/53e314461c19dbc7f462b42bf0f0bfae020dc398.zip) to your local path.

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/Avazu/Avazu_x4/split_avazu_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [DeepCross_avazu_x4_tuner_config_01.yaml](./DeepCross_avazu_x4_tuner_config_01.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/DeepCross_avazu_x4_tuner_config_01.yaml --tag 005 --gpu 0
  ```


### Results
```python
[Metrics] logloss: 0.372066 - AUC: 0.792953
```


### Logs
```python
2020-06-15 12:18:22,188 P1284 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "dnn_activations": "relu",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DeepCrossing",
    "model_id": "DeepCrossing_avazu_x4_3bbbc4c9_005_2335faa3",
    "model_root": "./Avazu/DeepCrossing_avazu/min2/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "residual_blocks": "[1000, 1000, 1000, 1000]",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/avazu_x4_3bbbc4c9/test.h5",
    "train_data": "../data/Avazu/avazu_x4_3bbbc4c9/train.h5",
    "use_hdf5": "True",
    "use_residual": "True",
    "valid_data": "../data/Avazu/avazu_x4_3bbbc4c9/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-15 12:18:22,190 P1284 INFO Set up feature encoder...
2020-06-15 12:18:22,190 P1284 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-06-15 12:18:22,191 P1284 INFO Loading data...
2020-06-15 12:18:22,197 P1284 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-06-15 12:18:26,152 P1284 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-06-15 12:18:27,616 P1284 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-15 12:18:27,720 P1284 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-15 12:18:27,720 P1284 INFO Loading train data done.
2020-06-15 12:18:36,419 P1284 INFO Start training: 3235 batches/epoch
2020-06-15 12:18:36,419 P1284 INFO ************ Epoch=1 start ************
2020-06-15 12:23:17,725 P1284 INFO [Metrics] logloss: 0.372169 - AUC: 0.792766
2020-06-15 12:23:17,726 P1284 INFO Save best model: monitor(max): 0.420597
2020-06-15 12:23:18,643 P1284 INFO --- 3235/3235 batches finished ---
2020-06-15 12:23:18,698 P1284 INFO Train loss: 0.380008
2020-06-15 12:23:18,699 P1284 INFO ************ Epoch=1 end ************
2020-06-15 12:27:59,660 P1284 INFO [Metrics] logloss: 0.379739 - AUC: 0.789453
2020-06-15 12:27:59,664 P1284 INFO Monitor(max) STOP: 0.409715 !
2020-06-15 12:27:59,664 P1284 INFO Reduce learning rate on plateau: 0.000100
2020-06-15 12:27:59,664 P1284 INFO --- 3235/3235 batches finished ---
2020-06-15 12:27:59,715 P1284 INFO Train loss: 0.332881
2020-06-15 12:27:59,715 P1284 INFO ************ Epoch=2 end ************
2020-06-15 12:32:42,078 P1284 INFO [Metrics] logloss: 0.426269 - AUC: 0.776080
2020-06-15 12:32:42,081 P1284 INFO Monitor(max) STOP: 0.349810 !
2020-06-15 12:32:42,082 P1284 INFO Reduce learning rate on plateau: 0.000010
2020-06-15 12:32:42,082 P1284 INFO Early stopping at epoch=3
2020-06-15 12:32:42,082 P1284 INFO --- 3235/3235 batches finished ---
2020-06-15 12:32:42,132 P1284 INFO Train loss: 0.292803
2020-06-15 12:32:42,132 P1284 INFO Training finished.
2020-06-15 12:32:42,132 P1284 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Avazu/DeepCrossing_avazu/min2/avazu_x4_3bbbc4c9/DeepCrossing_avazu_x4_3bbbc4c9_005_2335faa3_model.ckpt
2020-06-15 12:32:42,633 P1284 INFO ****** Train/validation evaluation ******
2020-06-15 12:35:53,736 P1284 INFO [Metrics] logloss: 0.340192 - AUC: 0.844455
2020-06-15 12:36:15,854 P1284 INFO [Metrics] logloss: 0.372169 - AUC: 0.792766
2020-06-15 12:36:15,923 P1284 INFO ******** Test evaluation ********
2020-06-15 12:36:15,923 P1284 INFO Loading data...
2020-06-15 12:36:15,923 P1284 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-06-15 12:36:16,594 P1284 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-15 12:36:16,594 P1284 INFO Loading test data done.
2020-06-15 12:36:41,391 P1284 INFO [Metrics] logloss: 0.372066 - AUC: 0.792953


```
