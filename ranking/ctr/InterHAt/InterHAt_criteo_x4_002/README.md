## InterHAt_criteo_x4_002

A hands-on guide to run the InterHAt model on the Criteo_x4_002 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

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
Dataset ID: [Criteo_x4](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x4). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [InterHAt](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/InterHAt.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [InterHAt_criteo_x4_tuner_config_11](./InterHAt_criteo_x4_tuner_config_11). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd InterHAt_criteo_x4_002
    nohup python run_expid.py --config ./InterHAt_criteo_x4_tuner_config_11 --expid InterHAt_criteo_x4_004_3fdd2b78 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.440131 | 0.811671  |


### Logs
```python
2020-06-04 02:29:22,808 P47589 INFO {
    "attention_dim": "40",
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_001_be98441d",
    "debug": "False",
    "embedding_dim": "40",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_dim": "500",
    "hidden_units": "[100, 100]",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "InterHAt",
    "model_id": "InterHAt_criteo_x4_001_be98441d_004_e0bea572",
    "model_root": "./Criteo/InterHAt_criteo/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "1",
    "optimizer": "adam",
    "order": "2",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/criteo_x4_001_be98441d/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_be98441d/train.h5",
    "use_hdf5": "True",
    "use_residual": "False",
    "valid_data": "../data/Criteo/criteo_x4_001_be98441d/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-06-04 02:29:22,808 P47589 INFO Set up feature encoder...
2020-06-04 02:29:22,809 P47589 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-06-04 02:29:27,809 P47589 INFO Total number of parameters: 222091161.
2020-06-04 02:29:27,810 P47589 INFO Loading data...
2020-06-04 02:29:27,814 P47589 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-06-04 02:29:35,648 P47589 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-06-04 02:29:38,448 P47589 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-06-04 02:29:38,744 P47589 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-04 02:29:38,744 P47589 INFO Loading train data done.
2020-06-04 02:29:41,976 P47589 INFO **** Start training: 3668 batches/epoch ****
2020-06-04 02:47:34,948 P47589 INFO [Metrics] logloss: 0.451784 - AUC: 0.798741
2020-06-04 02:47:34,949 P47589 INFO Save best model: monitor(max): 0.346957
2020-06-04 02:47:35,898 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 02:47:36,072 P47589 INFO Train loss: 0.473296
2020-06-04 02:47:36,072 P47589 INFO ************ Epoch=1 end ************
2020-06-04 03:05:23,307 P47589 INFO [Metrics] logloss: 0.449445 - AUC: 0.801639
2020-06-04 03:05:23,308 P47589 INFO Save best model: monitor(max): 0.352194
2020-06-04 03:05:25,257 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 03:05:25,444 P47589 INFO Train loss: 0.466529
2020-06-04 03:05:25,444 P47589 INFO ************ Epoch=2 end ************
2020-06-04 03:23:15,051 P47589 INFO [Metrics] logloss: 0.447903 - AUC: 0.803147
2020-06-04 03:23:15,052 P47589 INFO Save best model: monitor(max): 0.355244
2020-06-04 03:23:16,996 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 03:23:17,175 P47589 INFO Train loss: 0.464874
2020-06-04 03:23:17,175 P47589 INFO ************ Epoch=3 end ************
2020-06-04 03:41:07,098 P47589 INFO [Metrics] logloss: 0.447272 - AUC: 0.803737
2020-06-04 03:41:07,099 P47589 INFO Save best model: monitor(max): 0.356465
2020-06-04 03:41:08,233 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 03:41:08,434 P47589 INFO Train loss: 0.464133
2020-06-04 03:41:08,434 P47589 INFO ************ Epoch=4 end ************
2020-06-04 03:58:57,740 P47589 INFO [Metrics] logloss: 0.447010 - AUC: 0.804099
2020-06-04 03:58:57,742 P47589 INFO Save best model: monitor(max): 0.357089
2020-06-04 03:58:58,972 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 03:58:59,152 P47589 INFO Train loss: 0.463901
2020-06-04 03:58:59,152 P47589 INFO ************ Epoch=5 end ************
2020-06-04 04:16:48,833 P47589 INFO [Metrics] logloss: 0.446687 - AUC: 0.804397
2020-06-04 04:16:48,835 P47589 INFO Save best model: monitor(max): 0.357710
2020-06-04 04:16:50,855 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 04:16:51,048 P47589 INFO Train loss: 0.463803
2020-06-04 04:16:51,048 P47589 INFO ************ Epoch=6 end ************
2020-06-04 04:34:38,653 P47589 INFO [Metrics] logloss: 0.446799 - AUC: 0.804588
2020-06-04 04:34:38,654 P47589 INFO Save best model: monitor(max): 0.357789
2020-06-04 04:34:40,654 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 04:34:40,884 P47589 INFO Train loss: 0.463735
2020-06-04 04:34:40,884 P47589 INFO ************ Epoch=7 end ************
2020-06-04 04:52:28,930 P47589 INFO [Metrics] logloss: 0.446379 - AUC: 0.804877
2020-06-04 04:52:28,931 P47589 INFO Save best model: monitor(max): 0.358498
2020-06-04 04:52:34,577 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 04:52:34,766 P47589 INFO Train loss: 0.463669
2020-06-04 04:52:34,766 P47589 INFO ************ Epoch=8 end ************
2020-06-04 05:10:23,056 P47589 INFO [Metrics] logloss: 0.445852 - AUC: 0.805229
2020-06-04 05:10:23,058 P47589 INFO Save best model: monitor(max): 0.359378
2020-06-04 05:10:25,020 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 05:10:25,238 P47589 INFO Train loss: 0.463641
2020-06-04 05:10:25,238 P47589 INFO ************ Epoch=9 end ************
2020-06-04 05:28:14,276 P47589 INFO [Metrics] logloss: 0.445845 - AUC: 0.805295
2020-06-04 05:28:14,277 P47589 INFO Save best model: monitor(max): 0.359450
2020-06-04 05:28:16,205 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 05:28:16,390 P47589 INFO Train loss: 0.463578
2020-06-04 05:28:16,390 P47589 INFO ************ Epoch=10 end ************
2020-06-04 05:46:13,274 P47589 INFO [Metrics] logloss: 0.445898 - AUC: 0.805362
2020-06-04 05:46:13,276 P47589 INFO Save best model: monitor(max): 0.359464
2020-06-04 05:46:15,241 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 05:46:15,430 P47589 INFO Train loss: 0.463520
2020-06-04 05:46:15,430 P47589 INFO ************ Epoch=11 end ************
2020-06-04 06:04:02,354 P47589 INFO [Metrics] logloss: 0.445781 - AUC: 0.805531
2020-06-04 06:04:02,355 P47589 INFO Save best model: monitor(max): 0.359750
2020-06-04 06:04:04,328 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 06:04:04,514 P47589 INFO Train loss: 0.463520
2020-06-04 06:04:04,514 P47589 INFO ************ Epoch=12 end ************
2020-06-04 06:21:51,695 P47589 INFO [Metrics] logloss: 0.445469 - AUC: 0.805881
2020-06-04 06:21:51,697 P47589 INFO Save best model: monitor(max): 0.360412
2020-06-04 06:21:53,634 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 06:21:53,825 P47589 INFO Train loss: 0.463531
2020-06-04 06:21:53,825 P47589 INFO ************ Epoch=13 end ************
2020-06-04 06:39:40,142 P47589 INFO [Metrics] logloss: 0.445311 - AUC: 0.805820
2020-06-04 06:39:40,143 P47589 INFO Save best model: monitor(max): 0.360510
2020-06-04 06:39:41,241 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 06:39:41,425 P47589 INFO Train loss: 0.463465
2020-06-04 06:39:41,425 P47589 INFO ************ Epoch=14 end ************
2020-06-04 06:57:33,578 P47589 INFO [Metrics] logloss: 0.445093 - AUC: 0.806157
2020-06-04 06:57:33,579 P47589 INFO Save best model: monitor(max): 0.361064
2020-06-04 06:57:35,533 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 06:57:35,715 P47589 INFO Train loss: 0.463455
2020-06-04 06:57:35,716 P47589 INFO ************ Epoch=15 end ************
2020-06-04 07:15:22,619 P47589 INFO [Metrics] logloss: 0.445135 - AUC: 0.806275
2020-06-04 07:15:22,620 P47589 INFO Save best model: monitor(max): 0.361140
2020-06-04 07:15:24,586 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 07:15:24,775 P47589 INFO Train loss: 0.463382
2020-06-04 07:15:24,775 P47589 INFO ************ Epoch=16 end ************
2020-06-04 07:33:18,535 P47589 INFO [Metrics] logloss: 0.444941 - AUC: 0.806418
2020-06-04 07:33:18,538 P47589 INFO Save best model: monitor(max): 0.361477
2020-06-04 07:33:20,553 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 07:33:20,749 P47589 INFO Train loss: 0.463362
2020-06-04 07:33:20,749 P47589 INFO ************ Epoch=17 end ************
2020-06-04 07:51:07,283 P47589 INFO [Metrics] logloss: 0.444668 - AUC: 0.806535
2020-06-04 07:51:07,285 P47589 INFO Save best model: monitor(max): 0.361867
2020-06-04 07:51:08,400 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 07:51:08,587 P47589 INFO Train loss: 0.463303
2020-06-04 07:51:08,588 P47589 INFO ************ Epoch=18 end ************
2020-06-04 08:09:04,588 P47589 INFO [Metrics] logloss: 0.444771 - AUC: 0.806484
2020-06-04 08:09:04,592 P47589 INFO Monitor(max) STOP: 0.361713 !
2020-06-04 08:09:04,592 P47589 INFO Reduce learning rate on plateau: 0.000100
2020-06-04 08:09:04,592 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 08:09:04,774 P47589 INFO Train loss: 0.463233
2020-06-04 08:09:04,774 P47589 INFO ************ Epoch=19 end ************
2020-06-04 08:26:54,431 P47589 INFO [Metrics] logloss: 0.441030 - AUC: 0.810590
2020-06-04 08:26:54,432 P47589 INFO Save best model: monitor(max): 0.369559
2020-06-04 08:26:56,444 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 08:26:56,624 P47589 INFO Train loss: 0.446907
2020-06-04 08:26:56,624 P47589 INFO ************ Epoch=20 end ************
2020-06-04 08:44:43,775 P47589 INFO [Metrics] logloss: 0.440507 - AUC: 0.811239
2020-06-04 08:44:43,777 P47589 INFO Save best model: monitor(max): 0.370732
2020-06-04 08:44:44,895 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 08:44:45,072 P47589 INFO Train loss: 0.442111
2020-06-04 08:44:45,072 P47589 INFO ************ Epoch=21 end ************
2020-06-04 09:02:32,121 P47589 INFO [Metrics] logloss: 0.440603 - AUC: 0.811281
2020-06-04 09:02:32,122 P47589 INFO Monitor(max) STOP: 0.370678 !
2020-06-04 09:02:32,122 P47589 INFO Reduce learning rate on plateau: 0.000010
2020-06-04 09:02:32,122 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 09:02:32,300 P47589 INFO Train loss: 0.440101
2020-06-04 09:02:32,300 P47589 INFO ************ Epoch=22 end ************
2020-06-04 09:20:28,861 P47589 INFO [Metrics] logloss: 0.441387 - AUC: 0.810820
2020-06-04 09:20:28,866 P47589 INFO Monitor(max) STOP: 0.369434 !
2020-06-04 09:20:28,867 P47589 INFO Reduce learning rate on plateau: 0.000001
2020-06-04 09:20:28,867 P47589 INFO Early stopping at epoch=23
2020-06-04 09:20:28,867 P47589 INFO --- 3668/3668 batches finished ---
2020-06-04 09:20:29,055 P47589 INFO Train loss: 0.433332
2020-06-04 09:20:29,055 P47589 INFO Training finished.
2020-06-04 09:20:29,055 P47589 INFO Load best model: /home/XXX/benchmarks/Criteo/InterHAt_criteo/criteo_x4_001_be98441d/InterHAt_criteo_x4_001_be98441d_004_e0bea572_model.ckpt
2020-06-04 09:20:30,475 P47589 INFO ****** Train/validation evaluation ******
2020-06-04 09:21:08,891 P47589 INFO [Metrics] logloss: 0.440507 - AUC: 0.811239
2020-06-04 09:21:08,978 P47589 INFO ******** Test evaluation ********
2020-06-04 09:21:08,979 P47589 INFO Loading data...
2020-06-04 09:21:08,979 P47589 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-06-04 09:21:10,414 P47589 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-04 09:21:10,415 P47589 INFO Loading test data done.
2020-06-04 09:22:00,413 P47589 INFO [Metrics] logloss: 0.440131 - AUC: 0.811671

```
