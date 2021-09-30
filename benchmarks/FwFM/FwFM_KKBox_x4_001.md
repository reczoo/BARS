## FwFM_KKBox_x4_001

A notebook to benchmark FwFM on KKBox_x4_001 dataset.

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
In this setting, For all categorical fields, we replace infrequent features with a default <OOV> token by setting the threshold min_category_count=10.

To make a fair comparison, we fix embedding_dim=128, which performs well.


### Code
1. Install FuxiCTR
  
    Install FuxiCTR via `pip install fuxictr==1.0` to get all dependencies ready. Then download [the FuxiCTR repository](https://github.com/huawei-noah/benchmark/archive/53e314461c19dbc7f462b42bf0f0bfae020dc398.zip) to your local path.

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/KKBox/KKBox_x4/split_kkbox_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [FwFM_kkbox_x4_tuner_config_01.yaml](./FwFM_kkbox_x4_tuner_config_01.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/FwFM_kkbox_x4_tuner_config_01.yaml --tag 007 --gpu 0
  ```




### Results
```python
[Metrics] logloss: 0.496596 - AUC: 0.840832
```


### Logs
```python
2020-04-18 10:31:54,253 P1527 INFO {
    "batch_size": "10000",
    "dataset_id": "kkbox_x4_001_c5c9c6e3",
    "embedding_dim": "128",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "linear_type": "LW",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FwFM",
    "model_id": "FwFM_kkbox_x4_007_64d90caa",
    "model_root": "./KKBox/FwFM_kkbox/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-05",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/KKBox/",
    "test_data": "../data/KKBox/kkbox_x4_001_c5c9c6e3/test.h5",
    "train_data": "../data/KKBox/kkbox_x4_001_c5c9c6e3/train.h5",
    "valid_data": "../data/KKBox/kkbox_x4_001_c5c9c6e3/valid.h5",
    "version": "pytorch",
    "gpu": "1"
}
2020-04-18 10:31:54,253 P1527 INFO Set up feature encoder...
2020-04-18 10:31:54,254 P1527 INFO Load feature_map from json: ../data/KKBox/kkbox_x4_001_c5c9c6e3/feature_map.json
2020-04-18 10:31:54,254 P1527 INFO Loading data...
2020-04-18 10:31:54,256 P1527 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/train.h5
2020-04-18 10:31:54,547 P1527 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/valid.h5
2020-04-18 10:31:54,760 P1527 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2020-04-18 10:31:54,803 P1527 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-04-18 10:31:54,803 P1527 INFO Loading train data done.
2020-04-18 10:31:59,158 P1527 INFO **** Start training: 591 batches/epoch ****
2020-04-18 10:34:46,422 P1527 INFO [Metrics] logloss: 0.555895 - AUC: 0.786161
2020-04-18 10:34:46,433 P1527 INFO Save best model: monitor(max): 0.230266
2020-04-18 10:34:46,472 P1527 INFO --- 591/591 batches finished ---
2020-04-18 10:34:46,525 P1527 INFO Train loss: 0.594317
2020-04-18 10:34:46,525 P1527 INFO ************ Epoch=1 end ************
2020-04-18 10:37:33,936 P1527 INFO [Metrics] logloss: 0.532167 - AUC: 0.808056
2020-04-18 10:37:33,948 P1527 INFO Save best model: monitor(max): 0.275889
2020-04-18 10:37:34,104 P1527 INFO --- 591/591 batches finished ---
2020-04-18 10:37:34,143 P1527 INFO Train loss: 0.557647
2020-04-18 10:37:34,143 P1527 INFO ************ Epoch=2 end ************
2020-04-18 10:40:20,526 P1527 INFO [Metrics] logloss: 0.520204 - AUC: 0.818234
2020-04-18 10:40:20,543 P1527 INFO Save best model: monitor(max): 0.298031
2020-04-18 10:40:20,707 P1527 INFO --- 591/591 batches finished ---
2020-04-18 10:40:20,765 P1527 INFO Train loss: 0.540568
2020-04-18 10:40:20,765 P1527 INFO ************ Epoch=3 end ************
2020-04-18 10:42:55,715 P1527 INFO [Metrics] logloss: 0.510927 - AUC: 0.825829
2020-04-18 10:42:55,727 P1527 INFO Save best model: monitor(max): 0.314902
2020-04-18 10:42:55,883 P1527 INFO --- 591/591 batches finished ---
2020-04-18 10:42:55,923 P1527 INFO Train loss: 0.528381
2020-04-18 10:42:55,923 P1527 INFO ************ Epoch=4 end ************
2020-04-18 10:45:42,708 P1527 INFO [Metrics] logloss: 0.504873 - AUC: 0.830895
2020-04-18 10:45:42,720 P1527 INFO Save best model: monitor(max): 0.326021
2020-04-18 10:45:42,879 P1527 INFO --- 591/591 batches finished ---
2020-04-18 10:45:42,921 P1527 INFO Train loss: 0.517947
2020-04-18 10:45:42,921 P1527 INFO ************ Epoch=5 end ************
2020-04-18 10:48:29,930 P1527 INFO [Metrics] logloss: 0.500143 - AUC: 0.834796
2020-04-18 10:48:29,948 P1527 INFO Save best model: monitor(max): 0.334653
2020-04-18 10:48:30,019 P1527 INFO --- 591/591 batches finished ---
2020-04-18 10:48:30,078 P1527 INFO Train loss: 0.508340
2020-04-18 10:48:30,079 P1527 INFO ************ Epoch=6 end ************
2020-04-18 10:51:17,531 P1527 INFO [Metrics] logloss: 0.497459 - AUC: 0.837212
2020-04-18 10:51:17,549 P1527 INFO Save best model: monitor(max): 0.339753
2020-04-18 10:51:17,618 P1527 INFO --- 591/591 batches finished ---
2020-04-18 10:51:17,681 P1527 INFO Train loss: 0.498749
2020-04-18 10:51:17,681 P1527 INFO ************ Epoch=7 end ************
2020-04-18 10:53:53,969 P1527 INFO [Metrics] logloss: 0.497186 - AUC: 0.838217
2020-04-18 10:53:53,986 P1527 INFO Save best model: monitor(max): 0.341031
2020-04-18 10:53:54,050 P1527 INFO --- 591/591 batches finished ---
2020-04-18 10:53:54,110 P1527 INFO Train loss: 0.488816
2020-04-18 10:53:54,110 P1527 INFO ************ Epoch=8 end ************
2020-04-18 10:56:41,765 P1527 INFO [Metrics] logloss: 0.498310 - AUC: 0.838705
2020-04-18 10:56:41,783 P1527 INFO Monitor(max) STOP: 0.340396 !
2020-04-18 10:56:41,783 P1527 INFO Reduce learning rate on plateau: 0.000100
2020-04-18 10:56:41,783 P1527 INFO --- 591/591 batches finished ---
2020-04-18 10:56:41,842 P1527 INFO Train loss: 0.478387
2020-04-18 10:56:41,843 P1527 INFO ************ Epoch=9 end ************
2020-04-18 10:59:29,135 P1527 INFO [Metrics] logloss: 0.496420 - AUC: 0.840947
2020-04-18 10:59:29,153 P1527 INFO Save best model: monitor(max): 0.344527
2020-04-18 10:59:29,314 P1527 INFO --- 591/591 batches finished ---
2020-04-18 10:59:29,360 P1527 INFO Train loss: 0.432088
2020-04-18 10:59:29,360 P1527 INFO ************ Epoch=10 end ************
2020-04-18 11:02:15,943 P1527 INFO [Metrics] logloss: 0.497015 - AUC: 0.841293
2020-04-18 11:02:15,956 P1527 INFO Monitor(max) STOP: 0.344277 !
2020-04-18 11:02:15,957 P1527 INFO Reduce learning rate on plateau: 0.000010
2020-04-18 11:02:15,957 P1527 INFO --- 591/591 batches finished ---
2020-04-18 11:02:16,015 P1527 INFO Train loss: 0.424985
2020-04-18 11:02:16,016 P1527 INFO ************ Epoch=11 end ************
2020-04-18 11:04:56,753 P1527 INFO [Metrics] logloss: 0.497151 - AUC: 0.841330
2020-04-18 11:04:56,770 P1527 INFO Monitor(max) STOP: 0.344180 !
2020-04-18 11:04:56,771 P1527 INFO Reduce learning rate on plateau: 0.000001
2020-04-18 11:04:56,771 P1527 INFO Early stopping at epoch=12
2020-04-18 11:04:56,771 P1527 INFO --- 591/591 batches finished ---
2020-04-18 11:04:56,821 P1527 INFO Train loss: 0.416980
2020-04-18 11:04:56,821 P1527 INFO Training finished.
2020-04-18 11:04:56,821 P1527 INFO Load best model: /home/hispace/container/data/xxx/FuxiCTR/benchmarks/KKBox/FwFM_kkbox/kkbox_x4_001_c5c9c6e3/FwFM_kkbox_x4_007_64d90caa_kkbox_x4_001_c5c9c6e3_model.ckpt
2020-04-18 11:04:57,142 P1527 INFO ****** Train/validation evaluation ******
2020-04-18 11:06:15,107 P1527 INFO [Metrics] logloss: 0.372385 - AUC: 0.919078
2020-04-18 11:06:25,051 P1527 INFO [Metrics] logloss: 0.496420 - AUC: 0.840947
2020-04-18 11:06:25,174 P1527 INFO ******** Test evaluation ********
2020-04-18 11:06:25,174 P1527 INFO Loading data...
2020-04-18 11:06:25,174 P1527 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/test.h5
2020-04-18 11:06:25,269 P1527 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-04-18 11:06:25,269 P1527 INFO Loading test data done.
2020-04-18 11:06:35,504 P1527 INFO [Metrics] logloss: 0.496596 - AUC: 0.840832



```
