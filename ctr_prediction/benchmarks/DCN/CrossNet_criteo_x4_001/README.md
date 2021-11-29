## CrossNet_Criteo_x4_001

A notebook to benchmark AFN on CrossNet_x4_001 dataset.

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
This dataset split follows the setting in the AutoInt work. Specifically, we randomly split the data into 8:1:1 as the training set, validation set, and test set, respectively. To make it exactly reproducible, we reuse the code provided by AutoInt and control the random seed (i.e., seed=2018) for splitting.

### Code
1. Install FuxiCTR
  
    Install FuxiCTR via `pip install fuxictr==1.0` to get all dependencies ready. Then download [the FuxiCTR repository](https://github.com/huawei-noah/benchmark/archive/53e314461c19dbc7f462b42bf0f0bfae020dc398.zip) to your local path.

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/Criteo/Criteo_x4/split_criteo_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [CrossNet_criteo_x4_tuner_config_13.yaml](./CrossNet_criteo_x4_tuner_config_13.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/CrossNet_criteo_x4_tuner_config_13.yaml --tag 019 --gpu 0
  ```



### Results
```python
[Metrics] logloss: 0.445791 - AUC: 0.805756
```


### Logs
```python
2021-09-09 01:48:21,492 P5328 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "crossing_layers": "8",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_9ea3bdfc",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "None",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DCN",
    "model_id": "DCN_criteo_x4_9ea3bdfc_019_2683e720",
    "model_root": "./Criteo/CrossNet_criteo_x4_001/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/criteo_x4_9ea3bdfc/test.h5",
    "train_data": "../data/Criteo/criteo_x4_9ea3bdfc/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_9ea3bdfc/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2021-09-09 01:48:21,493 P5328 INFO Set up feature encoder...
2021-09-09 01:48:21,493 P5328 INFO Load feature_map from json: ../data/Criteo/criteo_x4_9ea3bdfc/feature_map.json
2021-09-09 01:48:21,818 P5328 INFO Total number of parameters: 14581937.
2021-09-09 01:48:21,818 P5328 INFO Loading data...
2021-09-09 01:48:21,821 P5328 INFO Loading data from h5: ../data/Criteo/criteo_x4_9ea3bdfc/train.h5
2021-09-09 01:48:27,659 P5328 INFO Loading data from h5: ../data/Criteo/criteo_x4_9ea3bdfc/valid.h5
2021-09-09 01:48:29,833 P5328 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2021-09-09 01:48:29,956 P5328 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2021-09-09 01:48:29,956 P5328 INFO Loading train data done.
2021-09-09 01:48:33,030 P5328 INFO Start training: 3668 batches/epoch
2021-09-09 01:48:33,030 P5328 INFO ************ Epoch=1 start ************
2021-09-09 01:54:17,337 P5328 INFO [Metrics] logloss: 0.458623 - AUC: 0.791963
2021-09-09 01:54:17,338 P5328 INFO Save best model: monitor(max): 0.333339
2021-09-09 01:54:17,529 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 01:54:17,559 P5328 INFO Train loss: 0.466079
2021-09-09 01:54:17,559 P5328 INFO ************ Epoch=1 end ************
2021-09-09 02:00:03,207 P5328 INFO [Metrics] logloss: 0.452876 - AUC: 0.797554
2021-09-09 02:00:03,211 P5328 INFO Save best model: monitor(max): 0.344678
2021-09-09 02:00:03,274 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 02:00:03,316 P5328 INFO Train loss: 0.460098
2021-09-09 02:00:03,316 P5328 INFO ************ Epoch=2 end ************
2021-09-09 02:05:47,628 P5328 INFO [Metrics] logloss: 0.452030 - AUC: 0.798566
2021-09-09 02:05:47,631 P5328 INFO Save best model: monitor(max): 0.346536
2021-09-09 02:05:47,695 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 02:05:47,739 P5328 INFO Train loss: 0.457809
2021-09-09 02:05:47,739 P5328 INFO ************ Epoch=3 end ************
2021-09-09 02:11:31,160 P5328 INFO [Metrics] logloss: 0.451567 - AUC: 0.799339
2021-09-09 02:11:31,162 P5328 INFO Save best model: monitor(max): 0.347771
2021-09-09 02:11:31,224 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 02:11:31,256 P5328 INFO Train loss: 0.457189
2021-09-09 02:11:31,256 P5328 INFO ************ Epoch=4 end ************
2021-09-09 02:17:14,284 P5328 INFO [Metrics] logloss: 0.450927 - AUC: 0.799837
2021-09-09 02:17:14,287 P5328 INFO Save best model: monitor(max): 0.348909
2021-09-09 02:17:14,348 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 02:17:14,379 P5328 INFO Train loss: 0.456812
2021-09-09 02:17:14,379 P5328 INFO ************ Epoch=5 end ************
2021-09-09 02:22:58,558 P5328 INFO [Metrics] logloss: 0.450809 - AUC: 0.800043
2021-09-09 02:22:58,560 P5328 INFO Save best model: monitor(max): 0.349235
2021-09-09 02:22:58,619 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 02:22:58,651 P5328 INFO Train loss: 0.456476
2021-09-09 02:22:58,651 P5328 INFO ************ Epoch=6 end ************
2021-09-09 02:28:42,568 P5328 INFO [Metrics] logloss: 0.450543 - AUC: 0.800371
2021-09-09 02:28:42,569 P5328 INFO Save best model: monitor(max): 0.349828
2021-09-09 02:28:42,631 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 02:28:42,670 P5328 INFO Train loss: 0.456244
2021-09-09 02:28:42,671 P5328 INFO ************ Epoch=7 end ************
2021-09-09 02:34:29,136 P5328 INFO [Metrics] logloss: 0.450127 - AUC: 0.800721
2021-09-09 02:34:29,137 P5328 INFO Save best model: monitor(max): 0.350594
2021-09-09 02:34:29,200 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 02:34:29,241 P5328 INFO Train loss: 0.456028
2021-09-09 02:34:29,241 P5328 INFO ************ Epoch=8 end ************
2021-09-09 02:40:19,769 P5328 INFO [Metrics] logloss: 0.449966 - AUC: 0.800832
2021-09-09 02:40:19,771 P5328 INFO Save best model: monitor(max): 0.350866
2021-09-09 02:40:19,838 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 02:40:19,877 P5328 INFO Train loss: 0.455840
2021-09-09 02:40:19,877 P5328 INFO ************ Epoch=9 end ************
2021-09-09 02:46:03,858 P5328 INFO [Metrics] logloss: 0.449815 - AUC: 0.801137
2021-09-09 02:46:03,861 P5328 INFO Save best model: monitor(max): 0.351321
2021-09-09 02:46:03,924 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 02:46:03,963 P5328 INFO Train loss: 0.455670
2021-09-09 02:46:03,963 P5328 INFO ************ Epoch=10 end ************
2021-09-09 02:51:47,079 P5328 INFO [Metrics] logloss: 0.449570 - AUC: 0.801344
2021-09-09 02:51:47,081 P5328 INFO Save best model: monitor(max): 0.351773
2021-09-09 02:51:47,143 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 02:51:47,175 P5328 INFO Train loss: 0.455534
2021-09-09 02:51:47,176 P5328 INFO ************ Epoch=11 end ************
2021-09-09 02:57:30,445 P5328 INFO [Metrics] logloss: 0.449525 - AUC: 0.801554
2021-09-09 02:57:30,446 P5328 INFO Save best model: monitor(max): 0.352029
2021-09-09 02:57:30,505 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 02:57:30,545 P5328 INFO Train loss: 0.455421
2021-09-09 02:57:30,546 P5328 INFO ************ Epoch=12 end ************
2021-09-09 03:03:14,849 P5328 INFO [Metrics] logloss: 0.449469 - AUC: 0.801492
2021-09-09 03:03:14,851 P5328 INFO Monitor(max) STOP: 0.352023 !
2021-09-09 03:03:14,851 P5328 INFO Reduce learning rate on plateau: 0.000100
2021-09-09 03:03:14,851 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 03:03:14,885 P5328 INFO Train loss: 0.455300
2021-09-09 03:03:14,885 P5328 INFO ************ Epoch=13 end ************
2021-09-09 03:08:58,593 P5328 INFO [Metrics] logloss: 0.446740 - AUC: 0.804486
2021-09-09 03:08:58,594 P5328 INFO Save best model: monitor(max): 0.357746
2021-09-09 03:08:58,653 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 03:08:58,685 P5328 INFO Train loss: 0.448421
2021-09-09 03:08:58,685 P5328 INFO ************ Epoch=14 end ************
2021-09-09 03:14:44,160 P5328 INFO [Metrics] logloss: 0.446391 - AUC: 0.804974
2021-09-09 03:14:44,164 P5328 INFO Save best model: monitor(max): 0.358583
2021-09-09 03:14:44,235 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 03:14:44,269 P5328 INFO Train loss: 0.446174
2021-09-09 03:14:44,269 P5328 INFO ************ Epoch=15 end ************
2021-09-09 03:20:28,785 P5328 INFO [Metrics] logloss: 0.446119 - AUC: 0.805211
2021-09-09 03:20:28,788 P5328 INFO Save best model: monitor(max): 0.359092
2021-09-09 03:20:28,856 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 03:20:28,888 P5328 INFO Train loss: 0.445136
2021-09-09 03:20:28,888 P5328 INFO ************ Epoch=16 end ************
2021-09-09 03:26:12,058 P5328 INFO [Metrics] logloss: 0.446092 - AUC: 0.805343
2021-09-09 03:26:12,059 P5328 INFO Save best model: monitor(max): 0.359251
2021-09-09 03:26:12,124 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 03:26:12,164 P5328 INFO Train loss: 0.444364
2021-09-09 03:26:12,164 P5328 INFO ************ Epoch=17 end ************
2021-09-09 03:31:56,217 P5328 INFO [Metrics] logloss: 0.446087 - AUC: 0.805373
2021-09-09 03:31:56,220 P5328 INFO Save best model: monitor(max): 0.359286
2021-09-09 03:31:56,286 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 03:31:56,318 P5328 INFO Train loss: 0.443731
2021-09-09 03:31:56,318 P5328 INFO ************ Epoch=18 end ************
2021-09-09 03:37:41,144 P5328 INFO [Metrics] logloss: 0.446072 - AUC: 0.805355
2021-09-09 03:37:41,147 P5328 INFO Monitor(max) STOP: 0.359284 !
2021-09-09 03:37:41,147 P5328 INFO Reduce learning rate on plateau: 0.000010
2021-09-09 03:37:41,147 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 03:37:41,180 P5328 INFO Train loss: 0.443202
2021-09-09 03:37:41,180 P5328 INFO ************ Epoch=19 end ************
2021-09-09 03:43:25,692 P5328 INFO [Metrics] logloss: 0.446078 - AUC: 0.805390
2021-09-09 03:43:25,693 P5328 INFO Save best model: monitor(max): 0.359312
2021-09-09 03:43:25,758 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 03:43:25,789 P5328 INFO Train loss: 0.440875
2021-09-09 03:43:25,790 P5328 INFO ************ Epoch=20 end ************
2021-09-09 03:49:11,903 P5328 INFO [Metrics] logloss: 0.446179 - AUC: 0.805324
2021-09-09 03:49:11,906 P5328 INFO Monitor(max) STOP: 0.359145 !
2021-09-09 03:49:11,906 P5328 INFO Reduce learning rate on plateau: 0.000001
2021-09-09 03:49:11,906 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 03:49:11,951 P5328 INFO Train loss: 0.440623
2021-09-09 03:49:11,951 P5328 INFO ************ Epoch=21 end ************
2021-09-09 03:54:57,568 P5328 INFO [Metrics] logloss: 0.446175 - AUC: 0.805327
2021-09-09 03:54:57,569 P5328 INFO Monitor(max) STOP: 0.359152 !
2021-09-09 03:54:57,569 P5328 INFO Reduce learning rate on plateau: 0.000001
2021-09-09 03:54:57,569 P5328 INFO Early stopping at epoch=22
2021-09-09 03:54:57,569 P5328 INFO --- 3668/3668 batches finished ---
2021-09-09 03:54:57,600 P5328 INFO Train loss: 0.440268
2021-09-09 03:54:57,600 P5328 INFO Training finished.
2021-09-09 03:54:57,600 P5328 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/CrossNet_criteo_x4_001/criteo_x4_9ea3bdfc/DCN_criteo_x4_9ea3bdfc_019_2683e720_model.ckpt
2021-09-09 03:54:57,671 P5328 INFO ****** Train/validation evaluation ******
2021-09-09 03:55:22,640 P5328 INFO [Metrics] logloss: 0.446078 - AUC: 0.805390
2021-09-09 03:55:22,718 P5328 INFO ******** Test evaluation ********
2021-09-09 03:55:22,718 P5328 INFO Loading data...
2021-09-09 03:55:22,718 P5328 INFO Loading data from h5: ../data/Criteo/criteo_x4_9ea3bdfc/test.h5
2021-09-09 03:55:23,632 P5328 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2021-09-09 03:55:23,632 P5328 INFO Loading test data done.
2021-09-09 03:55:49,555 P5328 INFO [Metrics] logloss: 0.445791 - AUC: 0.805756



```
