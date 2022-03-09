## NFM_criteo_x4_001

A hands-on guide to run the NFM model on the Criteo_x4_001 dataset.

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
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [Criteo_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [NFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/NFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [NFM_criteo_x4_tuner_config_01](./NFM_criteo_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd NFM_criteo_x4_001
    nohup python run_expid.py --config ./NFM_criteo_x4_tuner_config_01 --expid NFM_criteo_x4_017_2326425e --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.442406 | 0.809330  |


### Logs
```python
2020-06-26 19:26:18,193 P5322 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "NFM",
    "model_id": "NFM_criteo_x4_5c863b0f_017_2a1a6375",
    "model_root": "./Criteo/NFM_criteo/min10/",
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
    "test_data": "../data/Criteo/criteo_x4_5c863b0f/test.h5",
    "train_data": "../data/Criteo/criteo_x4_5c863b0f/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-26 19:26:18,196 P5322 INFO Set up feature encoder...
2020-06-26 19:26:18,196 P5322 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-06-26 19:26:18,196 P5322 INFO Loading data...
2020-06-26 19:26:18,203 P5322 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-06-26 19:26:24,959 P5322 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-06-26 19:26:26,945 P5322 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-06-26 19:26:27,149 P5322 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-26 19:26:27,149 P5322 INFO Loading train data done.
2020-06-26 19:26:33,615 P5322 INFO Start training: 3668 batches/epoch
2020-06-26 19:26:33,615 P5322 INFO ************ Epoch=1 start ************
2020-06-26 19:35:58,342 P5322 INFO [Metrics] logloss: 0.450581 - AUC: 0.800343
2020-06-26 19:35:58,344 P5322 INFO Save best model: monitor(max): 0.349761
2020-06-26 19:35:58,451 P5322 INFO --- 3668/3668 batches finished ---
2020-06-26 19:35:58,514 P5322 INFO Train loss: 0.463911
2020-06-26 19:35:58,514 P5322 INFO ************ Epoch=1 end ************
2020-06-26 19:45:25,847 P5322 INFO [Metrics] logloss: 0.448986 - AUC: 0.801912
2020-06-26 19:45:25,849 P5322 INFO Save best model: monitor(max): 0.352925
2020-06-26 19:45:25,952 P5322 INFO --- 3668/3668 batches finished ---
2020-06-26 19:45:26,004 P5322 INFO Train loss: 0.458657
2020-06-26 19:45:26,004 P5322 INFO ************ Epoch=2 end ************
2020-06-26 19:54:42,441 P5322 INFO [Metrics] logloss: 0.448367 - AUC: 0.802552
2020-06-26 19:54:42,443 P5322 INFO Save best model: monitor(max): 0.354185
2020-06-26 19:54:42,546 P5322 INFO --- 3668/3668 batches finished ---
2020-06-26 19:54:42,600 P5322 INFO Train loss: 0.457931
2020-06-26 19:54:42,600 P5322 INFO ************ Epoch=3 end ************
2020-06-26 20:04:05,673 P5322 INFO [Metrics] logloss: 0.448088 - AUC: 0.802803
2020-06-26 20:04:05,675 P5322 INFO Save best model: monitor(max): 0.354716
2020-06-26 20:04:05,804 P5322 INFO --- 3668/3668 batches finished ---
2020-06-26 20:04:05,857 P5322 INFO Train loss: 0.457486
2020-06-26 20:04:05,857 P5322 INFO ************ Epoch=4 end ************
2020-06-26 20:13:24,635 P5322 INFO [Metrics] logloss: 0.448057 - AUC: 0.802972
2020-06-26 20:13:24,636 P5322 INFO Save best model: monitor(max): 0.354915
2020-06-26 20:13:24,720 P5322 INFO --- 3668/3668 batches finished ---
2020-06-26 20:13:24,773 P5322 INFO Train loss: 0.457194
2020-06-26 20:13:24,774 P5322 INFO ************ Epoch=5 end ************
2020-06-26 20:22:39,429 P5322 INFO [Metrics] logloss: 0.447898 - AUC: 0.803111
2020-06-26 20:22:39,431 P5322 INFO Save best model: monitor(max): 0.355213
2020-06-26 20:22:39,527 P5322 INFO --- 3668/3668 batches finished ---
2020-06-26 20:22:39,581 P5322 INFO Train loss: 0.456946
2020-06-26 20:22:39,581 P5322 INFO ************ Epoch=6 end ************
2020-06-26 20:31:51,905 P5322 INFO [Metrics] logloss: 0.447687 - AUC: 0.803342
2020-06-26 20:31:51,906 P5322 INFO Save best model: monitor(max): 0.355655
2020-06-26 20:31:52,034 P5322 INFO --- 3668/3668 batches finished ---
2020-06-26 20:31:52,088 P5322 INFO Train loss: 0.456792
2020-06-26 20:31:52,089 P5322 INFO ************ Epoch=7 end ************
2020-06-26 20:41:07,882 P5322 INFO [Metrics] logloss: 0.447737 - AUC: 0.803444
2020-06-26 20:41:07,883 P5322 INFO Save best model: monitor(max): 0.355707
2020-06-26 20:41:07,966 P5322 INFO --- 3668/3668 batches finished ---
2020-06-26 20:41:08,022 P5322 INFO Train loss: 0.456648
2020-06-26 20:41:08,022 P5322 INFO ************ Epoch=8 end ************
2020-06-26 20:50:25,313 P5322 INFO [Metrics] logloss: 0.447476 - AUC: 0.803669
2020-06-26 20:50:25,315 P5322 INFO Save best model: monitor(max): 0.356194
2020-06-26 20:50:25,451 P5322 INFO --- 3668/3668 batches finished ---
2020-06-26 20:50:25,504 P5322 INFO Train loss: 0.456555
2020-06-26 20:50:25,505 P5322 INFO ************ Epoch=9 end ************
2020-06-26 20:59:52,213 P5322 INFO [Metrics] logloss: 0.447318 - AUC: 0.803741
2020-06-26 20:59:52,214 P5322 INFO Save best model: monitor(max): 0.356423
2020-06-26 20:59:52,301 P5322 INFO --- 3668/3668 batches finished ---
2020-06-26 20:59:52,355 P5322 INFO Train loss: 0.456466
2020-06-26 20:59:52,355 P5322 INFO ************ Epoch=10 end ************
2020-06-26 21:09:13,101 P5322 INFO [Metrics] logloss: 0.447334 - AUC: 0.803708
2020-06-26 21:09:13,103 P5322 INFO Monitor(max) STOP: 0.356374 !
2020-06-26 21:09:13,103 P5322 INFO Reduce learning rate on plateau: 0.000100
2020-06-26 21:09:13,103 P5322 INFO --- 3668/3668 batches finished ---
2020-06-26 21:09:13,167 P5322 INFO Train loss: 0.456403
2020-06-26 21:09:13,167 P5322 INFO ************ Epoch=11 end ************
2020-06-26 21:18:35,577 P5322 INFO [Metrics] logloss: 0.443734 - AUC: 0.807665
2020-06-26 21:18:35,579 P5322 INFO Save best model: monitor(max): 0.363931
2020-06-26 21:18:35,696 P5322 INFO --- 3668/3668 batches finished ---
2020-06-26 21:18:35,752 P5322 INFO Train loss: 0.447550
2020-06-26 21:18:35,752 P5322 INFO ************ Epoch=12 end ************
2020-06-26 21:27:57,166 P5322 INFO [Metrics] logloss: 0.443171 - AUC: 0.808348
2020-06-26 21:27:57,168 P5322 INFO Save best model: monitor(max): 0.365177
2020-06-26 21:27:57,268 P5322 INFO --- 3668/3668 batches finished ---
2020-06-26 21:27:57,321 P5322 INFO Train loss: 0.443962
2020-06-26 21:27:57,321 P5322 INFO ************ Epoch=13 end ************
2020-06-26 21:37:24,032 P5322 INFO [Metrics] logloss: 0.442977 - AUC: 0.808605
2020-06-26 21:37:24,034 P5322 INFO Save best model: monitor(max): 0.365628
2020-06-26 21:37:24,155 P5322 INFO --- 3668/3668 batches finished ---
2020-06-26 21:37:24,198 P5322 INFO Train loss: 0.442490
2020-06-26 21:37:24,199 P5322 INFO ************ Epoch=14 end ************
2020-06-26 21:46:50,961 P5322 INFO [Metrics] logloss: 0.442807 - AUC: 0.808753
2020-06-26 21:46:50,963 P5322 INFO Save best model: monitor(max): 0.365947
2020-06-26 21:46:51,075 P5322 INFO --- 3668/3668 batches finished ---
2020-06-26 21:46:51,127 P5322 INFO Train loss: 0.441435
2020-06-26 21:46:51,127 P5322 INFO ************ Epoch=15 end ************
2020-06-26 21:56:15,347 P5322 INFO [Metrics] logloss: 0.442787 - AUC: 0.808803
2020-06-26 21:56:15,348 P5322 INFO Save best model: monitor(max): 0.366016
2020-06-26 21:56:15,450 P5322 INFO --- 3668/3668 batches finished ---
2020-06-26 21:56:15,505 P5322 INFO Train loss: 0.440546
2020-06-26 21:56:15,505 P5322 INFO ************ Epoch=16 end ************
2020-06-26 22:05:33,332 P5322 INFO [Metrics] logloss: 0.442849 - AUC: 0.808726
2020-06-26 22:05:33,333 P5322 INFO Monitor(max) STOP: 0.365878 !
2020-06-26 22:05:33,333 P5322 INFO Reduce learning rate on plateau: 0.000010
2020-06-26 22:05:33,333 P5322 INFO --- 3668/3668 batches finished ---
2020-06-26 22:05:33,386 P5322 INFO Train loss: 0.439771
2020-06-26 22:05:33,386 P5322 INFO ************ Epoch=17 end ************
2020-06-26 22:14:56,116 P5322 INFO [Metrics] logloss: 0.443414 - AUC: 0.808481
2020-06-26 22:14:56,118 P5322 INFO Monitor(max) STOP: 0.365067 !
2020-06-26 22:14:56,118 P5322 INFO Reduce learning rate on plateau: 0.000001
2020-06-26 22:14:56,118 P5322 INFO Early stopping at epoch=18
2020-06-26 22:14:56,118 P5322 INFO --- 3668/3668 batches finished ---
2020-06-26 22:14:56,171 P5322 INFO Train loss: 0.435969
2020-06-26 22:14:56,172 P5322 INFO Training finished.
2020-06-26 22:14:56,172 P5322 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/NFM_criteo/min10/criteo_x4_5c863b0f/NFM_criteo_x4_5c863b0f_017_2a1a6375_model.ckpt
2020-06-26 22:14:56,316 P5322 INFO ****** Train/validation evaluation ******
2020-06-26 22:18:51,530 P5322 INFO [Metrics] logloss: 0.431636 - AUC: 0.820899
2020-06-26 22:19:19,599 P5322 INFO [Metrics] logloss: 0.442787 - AUC: 0.808803
2020-06-26 22:19:19,683 P5322 INFO ******** Test evaluation ********
2020-06-26 22:19:19,683 P5322 INFO Loading data...
2020-06-26 22:19:19,683 P5322 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-06-26 22:19:20,643 P5322 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-26 22:19:20,643 P5322 INFO Loading test data done.
2020-06-26 22:19:47,394 P5322 INFO [Metrics] logloss: 0.442406 - AUC: 0.809330

```
