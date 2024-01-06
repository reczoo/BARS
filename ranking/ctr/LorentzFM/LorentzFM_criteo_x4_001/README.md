## LorentzFM_criteo_x4_001

A hands-on guide to run the LorentzFM model on the Criteo_x4_001 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
  GPU: Tesla P100 16G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 10.0
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [Criteo_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [LorentzFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/LorentzFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [LorentzFM_criteo_x4_tuner_config_02](./LorentzFM_criteo_x4_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd LorentzFM_criteo_x4_001
    nohup python run_expid.py --config ./LorentzFM_criteo_x4_tuner_config_02 --expid LorentzFM_criteo_x4_001_342b3588 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.443359 | 0.808346  |


### Logs
```python
2020-07-20 11:53:48,435 P24362 INFO {
    "batch_size": "5000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "LorentzFM",
    "model_id": "LorentzFM_criteo_x4_5c863b0f_001_86f261de",
    "model_root": "./Criteo/LorentzFM_criteo/min10/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "5e-06",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/criteo_x4_5c863b0f/test.h5",
    "train_data": "../data/Criteo/criteo_x4_5c863b0f/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-20 11:53:48,436 P24362 INFO Set up feature encoder...
2020-07-20 11:53:48,436 P24362 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-07-20 11:53:48,816 P24362 INFO Total number of parameters: 14571328.
2020-07-20 11:53:48,816 P24362 INFO Loading data...
2020-07-20 11:53:48,818 P24362 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-07-20 11:54:07,283 P24362 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-07-20 11:54:10,113 P24362 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-07-20 11:54:10,249 P24362 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-20 11:54:10,249 P24362 INFO Loading train data done.
2020-07-20 11:54:40,853 P24362 INFO Start training: 7335 batches/epoch
2020-07-20 11:54:40,853 P24362 INFO ************ Epoch=1 start ************
2020-07-20 12:04:28,101 P24362 INFO [Metrics] logloss: 0.451777 - AUC: 0.798958
2020-07-20 12:04:28,108 P24362 INFO Save best model: monitor(max): 0.347181
2020-07-20 12:04:28,162 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 12:04:28,311 P24362 INFO Train loss: 0.466994
2020-07-20 12:04:28,312 P24362 INFO ************ Epoch=1 end ************
2020-07-20 12:14:11,245 P24362 INFO [Metrics] logloss: 0.450239 - AUC: 0.800610
2020-07-20 12:14:11,247 P24362 INFO Save best model: monitor(max): 0.350371
2020-07-20 12:14:11,341 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 12:14:11,484 P24362 INFO Train loss: 0.460094
2020-07-20 12:14:11,484 P24362 INFO ************ Epoch=2 end ************
2020-07-20 12:23:59,695 P24362 INFO [Metrics] logloss: 0.449882 - AUC: 0.801075
2020-07-20 12:23:59,698 P24362 INFO Save best model: monitor(max): 0.351193
2020-07-20 12:23:59,775 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 12:23:59,950 P24362 INFO Train loss: 0.459285
2020-07-20 12:23:59,950 P24362 INFO ************ Epoch=3 end ************
2020-07-20 12:33:51,570 P24362 INFO [Metrics] logloss: 0.449631 - AUC: 0.801316
2020-07-20 12:33:51,574 P24362 INFO Save best model: monitor(max): 0.351685
2020-07-20 12:33:51,654 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 12:33:51,856 P24362 INFO Train loss: 0.459047
2020-07-20 12:33:51,857 P24362 INFO ************ Epoch=4 end ************
2020-07-20 12:43:38,881 P24362 INFO [Metrics] logloss: 0.449728 - AUC: 0.801316
2020-07-20 12:43:38,885 P24362 INFO Monitor(max) STOP: 0.351588 !
2020-07-20 12:43:38,885 P24362 INFO Reduce learning rate on plateau: 0.000100
2020-07-20 12:43:38,886 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 12:43:39,043 P24362 INFO Train loss: 0.458943
2020-07-20 12:43:39,043 P24362 INFO ************ Epoch=5 end ************
2020-07-20 12:53:31,112 P24362 INFO [Metrics] logloss: 0.445256 - AUC: 0.806247
2020-07-20 12:53:31,116 P24362 INFO Save best model: monitor(max): 0.360991
2020-07-20 12:53:31,197 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 12:53:31,402 P24362 INFO Train loss: 0.448758
2020-07-20 12:53:31,402 P24362 INFO ************ Epoch=6 end ************
2020-07-20 13:03:22,546 P24362 INFO [Metrics] logloss: 0.444681 - AUC: 0.806881
2020-07-20 13:03:22,551 P24362 INFO Save best model: monitor(max): 0.362200
2020-07-20 13:03:22,634 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 13:03:22,789 P24362 INFO Train loss: 0.445523
2020-07-20 13:03:22,789 P24362 INFO ************ Epoch=7 end ************
2020-07-20 13:13:10,231 P24362 INFO [Metrics] logloss: 0.444454 - AUC: 0.807126
2020-07-20 13:13:10,233 P24362 INFO Save best model: monitor(max): 0.362672
2020-07-20 13:13:10,321 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 13:13:10,486 P24362 INFO Train loss: 0.444661
2020-07-20 13:13:10,486 P24362 INFO ************ Epoch=8 end ************
2020-07-20 13:22:57,614 P24362 INFO [Metrics] logloss: 0.444325 - AUC: 0.807263
2020-07-20 13:22:57,617 P24362 INFO Save best model: monitor(max): 0.362937
2020-07-20 13:22:57,698 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 13:22:57,861 P24362 INFO Train loss: 0.444214
2020-07-20 13:22:57,861 P24362 INFO ************ Epoch=9 end ************
2020-07-20 13:32:43,332 P24362 INFO [Metrics] logloss: 0.444212 - AUC: 0.807391
2020-07-20 13:32:43,334 P24362 INFO Save best model: monitor(max): 0.363179
2020-07-20 13:32:43,414 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 13:32:43,591 P24362 INFO Train loss: 0.443921
2020-07-20 13:32:43,591 P24362 INFO ************ Epoch=10 end ************
2020-07-20 13:42:35,877 P24362 INFO [Metrics] logloss: 0.444151 - AUC: 0.807450
2020-07-20 13:42:35,878 P24362 INFO Save best model: monitor(max): 0.363299
2020-07-20 13:42:35,959 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 13:42:36,213 P24362 INFO Train loss: 0.443686
2020-07-20 13:42:36,213 P24362 INFO ************ Epoch=11 end ************
2020-07-20 13:52:20,989 P24362 INFO [Metrics] logloss: 0.444097 - AUC: 0.807512
2020-07-20 13:52:20,991 P24362 INFO Save best model: monitor(max): 0.363415
2020-07-20 13:52:21,071 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 13:52:21,244 P24362 INFO Train loss: 0.443504
2020-07-20 13:52:21,244 P24362 INFO ************ Epoch=12 end ************
2020-07-20 14:01:57,713 P24362 INFO [Metrics] logloss: 0.444076 - AUC: 0.807551
2020-07-20 14:01:57,715 P24362 INFO Save best model: monitor(max): 0.363475
2020-07-20 14:01:57,793 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 14:01:58,005 P24362 INFO Train loss: 0.443345
2020-07-20 14:01:58,005 P24362 INFO ************ Epoch=13 end ************
2020-07-20 14:11:40,352 P24362 INFO [Metrics] logloss: 0.444028 - AUC: 0.807585
2020-07-20 14:11:40,356 P24362 INFO Save best model: monitor(max): 0.363556
2020-07-20 14:11:40,435 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 14:11:40,602 P24362 INFO Train loss: 0.443207
2020-07-20 14:11:40,602 P24362 INFO ************ Epoch=14 end ************
2020-07-20 14:21:27,745 P24362 INFO [Metrics] logloss: 0.443998 - AUC: 0.807623
2020-07-20 14:21:27,749 P24362 INFO Save best model: monitor(max): 0.363625
2020-07-20 14:21:27,830 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 14:21:27,990 P24362 INFO Train loss: 0.443086
2020-07-20 14:21:27,990 P24362 INFO ************ Epoch=15 end ************
2020-07-20 14:31:11,188 P24362 INFO [Metrics] logloss: 0.444008 - AUC: 0.807617
2020-07-20 14:31:11,191 P24362 INFO Monitor(max) STOP: 0.363609 !
2020-07-20 14:31:11,192 P24362 INFO Reduce learning rate on plateau: 0.000010
2020-07-20 14:31:11,192 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 14:31:11,369 P24362 INFO Train loss: 0.442975
2020-07-20 14:31:11,369 P24362 INFO ************ Epoch=16 end ************
2020-07-20 14:40:54,028 P24362 INFO [Metrics] logloss: 0.443796 - AUC: 0.807845
2020-07-20 14:40:54,029 P24362 INFO Save best model: monitor(max): 0.364049
2020-07-20 14:40:54,109 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 14:40:54,296 P24362 INFO Train loss: 0.440561
2020-07-20 14:40:54,296 P24362 INFO ************ Epoch=17 end ************
2020-07-20 14:50:34,922 P24362 INFO [Metrics] logloss: 0.443763 - AUC: 0.807877
2020-07-20 14:50:34,924 P24362 INFO Save best model: monitor(max): 0.364114
2020-07-20 14:50:35,014 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 14:50:35,187 P24362 INFO Train loss: 0.440435
2020-07-20 14:50:35,188 P24362 INFO ************ Epoch=18 end ************
2020-07-20 15:00:15,216 P24362 INFO [Metrics] logloss: 0.443758 - AUC: 0.807883
2020-07-20 15:00:15,218 P24362 INFO Save best model: monitor(max): 0.364125
2020-07-20 15:00:15,300 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 15:00:15,466 P24362 INFO Train loss: 0.440377
2020-07-20 15:00:15,466 P24362 INFO ************ Epoch=19 end ************
2020-07-20 15:09:50,558 P24362 INFO [Metrics] logloss: 0.443746 - AUC: 0.807896
2020-07-20 15:09:50,560 P24362 INFO Save best model: monitor(max): 0.364150
2020-07-20 15:09:50,644 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 15:09:50,813 P24362 INFO Train loss: 0.440337
2020-07-20 15:09:50,813 P24362 INFO ************ Epoch=20 end ************
2020-07-20 15:19:32,607 P24362 INFO [Metrics] logloss: 0.443745 - AUC: 0.807896
2020-07-20 15:19:32,609 P24362 INFO Monitor(max) STOP: 0.364151 !
2020-07-20 15:19:32,609 P24362 INFO Reduce learning rate on plateau: 0.000001
2020-07-20 15:19:32,609 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 15:19:32,790 P24362 INFO Train loss: 0.440304
2020-07-20 15:19:32,790 P24362 INFO ************ Epoch=21 end ************
2020-07-20 15:29:20,321 P24362 INFO [Metrics] logloss: 0.443743 - AUC: 0.807899
2020-07-20 15:29:20,326 P24362 INFO Save best model: monitor(max): 0.364156
2020-07-20 15:29:20,410 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 15:29:20,631 P24362 INFO Train loss: 0.439976
2020-07-20 15:29:20,631 P24362 INFO ************ Epoch=22 end ************
2020-07-20 15:39:02,568 P24362 INFO [Metrics] logloss: 0.443743 - AUC: 0.807900
2020-07-20 15:39:02,576 P24362 INFO Monitor(max) STOP: 0.364157 !
2020-07-20 15:39:02,576 P24362 INFO Reduce learning rate on plateau: 0.000001
2020-07-20 15:39:02,576 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 15:39:02,775 P24362 INFO Train loss: 0.439971
2020-07-20 15:39:02,775 P24362 INFO ************ Epoch=23 end ************
2020-07-20 15:48:42,776 P24362 INFO [Metrics] logloss: 0.443742 - AUC: 0.807901
2020-07-20 15:48:42,786 P24362 INFO Save best model: monitor(max): 0.364159
2020-07-20 15:48:42,868 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 15:48:43,053 P24362 INFO Train loss: 0.439968
2020-07-20 15:48:43,053 P24362 INFO ************ Epoch=24 end ************
2020-07-20 15:58:29,956 P24362 INFO [Metrics] logloss: 0.443742 - AUC: 0.807900
2020-07-20 15:58:29,961 P24362 INFO Monitor(max) STOP: 0.364158 !
2020-07-20 15:58:29,961 P24362 INFO Reduce learning rate on plateau: 0.000001
2020-07-20 15:58:29,961 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 15:58:30,148 P24362 INFO Train loss: 0.439966
2020-07-20 15:58:30,148 P24362 INFO ************ Epoch=25 end ************
2020-07-20 16:08:14,711 P24362 INFO [Metrics] logloss: 0.443743 - AUC: 0.807900
2020-07-20 16:08:14,714 P24362 INFO Monitor(max) STOP: 0.364158 !
2020-07-20 16:08:14,714 P24362 INFO Reduce learning rate on plateau: 0.000001
2020-07-20 16:08:14,715 P24362 INFO Early stopping at epoch=26
2020-07-20 16:08:14,715 P24362 INFO --- 7335/7335 batches finished ---
2020-07-20 16:08:14,899 P24362 INFO Train loss: 0.439962
2020-07-20 16:08:14,899 P24362 INFO Training finished.
2020-07-20 16:08:14,899 P24362 INFO Load best model: /home/XXX/benchmarks/Criteo/LorentzFM_criteo/min10/criteo_x4_5c863b0f/LorentzFM_criteo_x4_5c863b0f_001_86f261de_model.ckpt
2020-07-20 16:08:14,975 P24362 INFO ****** Train/validation evaluation ******
2020-07-20 16:13:02,390 P24362 INFO [Metrics] logloss: 0.434386 - AUC: 0.818361
2020-07-20 16:13:40,980 P24362 INFO [Metrics] logloss: 0.443742 - AUC: 0.807901
2020-07-20 16:13:41,040 P24362 INFO ******** Test evaluation ********
2020-07-20 16:13:41,041 P24362 INFO Loading data...
2020-07-20 16:13:41,041 P24362 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-07-20 16:13:47,709 P24362 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-20 16:13:47,710 P24362 INFO Loading test data done.
2020-07-20 16:14:15,852 P24362 INFO [Metrics] logloss: 0.443359 - AUC: 0.808346

```
