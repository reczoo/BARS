## LorentzFM_criteo_x4_002

A hands-on guide to run the LorentzFM model on the Criteo_x4_002 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [LorentzFM](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/LorentzFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [LorentzFM_criteo_x4_tuner_config_02](./LorentzFM_criteo_x4_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd LorentzFM_criteo_x4_002
    nohup python run_expid.py --config ./LorentzFM_criteo_x4_tuner_config_02 --expid LorentzFM_criteo_x4_002_6d2e078f --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.441322 | 0.810550  |


### Logs
```python
2020-02-19 18:11:41,455 P7007 INFO {
    "batch_size": "5000",
    "dataset_id": "criteo_x4_001_be98441d",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "LorentzFM",
    "model_id": "LorentzFM_criteo_x4_002_235e40ca",
    "model_root": "./Criteo/LorentzFM_criteo/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "l2(5.e-6)",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "test_data": "../data/Criteo/criteo_x4_001_be98441d/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_be98441d/train.h5",
    "valid_data": "../data/Criteo/criteo_x4_001_be98441d/valid.h5",
    "version": "pytorch",
    "gpu": "0"
}
2020-02-19 18:11:41,456 P7007 INFO Set up feature encoder...
2020-02-19 18:11:41,456 P7007 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-02-19 18:11:41,456 P7007 INFO Loading data...
2020-02-19 18:11:41,458 P7007 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-02-19 18:11:45,509 P7007 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-02-19 18:11:47,482 P7007 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-02-19 18:11:47,627 P7007 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-19 18:11:47,627 P7007 INFO Loading train data done.
2020-02-19 18:11:56,898 P7007 INFO **** Start training: 7335 batches/epoch ****
2020-02-19 20:24:10,691 P7007 INFO [Metrics] logloss: 0.450986 - AUC: 0.799810
2020-02-19 20:24:10,788 P7007 INFO Save best model: monitor(max): 0.348824
2020-02-19 20:24:11,586 P7007 INFO --- 7335/7335 batches finished ---
2020-02-19 20:24:11,630 P7007 INFO Train loss: 0.476465
2020-02-19 20:24:11,631 P7007 INFO ************ Epoch=1 end ************
2020-02-19 22:36:17,724 P7007 INFO [Metrics] logloss: 0.449760 - AUC: 0.801173
2020-02-19 22:36:17,805 P7007 INFO Save best model: monitor(max): 0.351412
2020-02-19 22:36:19,109 P7007 INFO --- 7335/7335 batches finished ---
2020-02-19 22:36:19,147 P7007 INFO Train loss: 0.472777
2020-02-19 22:36:19,147 P7007 INFO ************ Epoch=2 end ************
2020-02-20 00:48:27,866 P7007 INFO [Metrics] logloss: 0.449769 - AUC: 0.801205
2020-02-20 00:48:27,945 P7007 INFO Save best model: monitor(max): 0.351436
2020-02-20 00:48:29,245 P7007 INFO --- 7335/7335 batches finished ---
2020-02-20 00:48:29,291 P7007 INFO Train loss: 0.472417
2020-02-20 00:48:29,291 P7007 INFO ************ Epoch=3 end ************
2020-02-20 03:00:35,445 P7007 INFO [Metrics] logloss: 0.449535 - AUC: 0.801420
2020-02-20 03:00:35,556 P7007 INFO Save best model: monitor(max): 0.351884
2020-02-20 03:00:36,860 P7007 INFO --- 7335/7335 batches finished ---
2020-02-20 03:00:36,904 P7007 INFO Train loss: 0.472349
2020-02-20 03:00:36,904 P7007 INFO ************ Epoch=4 end ************
2020-02-20 05:13:32,034 P7007 INFO [Metrics] logloss: 0.449672 - AUC: 0.801332
2020-02-20 05:13:32,172 P7007 INFO Monitor(max) STOP: 0.351660 !
2020-02-20 05:13:32,173 P7007 INFO Reduce learning rate on plateau: 0.000100
2020-02-20 05:13:32,173 P7007 INFO --- 7335/7335 batches finished ---
2020-02-20 05:13:32,219 P7007 INFO Train loss: 0.472290
2020-02-20 05:13:32,219 P7007 INFO ************ Epoch=5 end ************
2020-02-20 07:25:45,015 P7007 INFO [Metrics] logloss: 0.443569 - AUC: 0.808027
2020-02-20 07:25:45,123 P7007 INFO Save best model: monitor(max): 0.364459
2020-02-20 07:25:46,438 P7007 INFO --- 7335/7335 batches finished ---
2020-02-20 07:25:46,482 P7007 INFO Train loss: 0.449719
2020-02-20 07:25:46,482 P7007 INFO ************ Epoch=6 end ************
2020-02-20 09:37:50,317 P7007 INFO [Metrics] logloss: 0.442784 - AUC: 0.808912
2020-02-20 09:37:50,390 P7007 INFO Save best model: monitor(max): 0.366128
2020-02-20 09:37:51,701 P7007 INFO --- 7335/7335 batches finished ---
2020-02-20 09:37:51,746 P7007 INFO Train loss: 0.443678
2020-02-20 09:37:51,747 P7007 INFO ************ Epoch=7 end ************
2020-02-20 11:50:02,601 P7007 INFO [Metrics] logloss: 0.442421 - AUC: 0.809303
2020-02-20 11:50:02,718 P7007 INFO Save best model: monitor(max): 0.366882
2020-02-20 11:50:04,067 P7007 INFO --- 7335/7335 batches finished ---
2020-02-20 11:50:04,105 P7007 INFO Train loss: 0.442275
2020-02-20 11:50:04,105 P7007 INFO ************ Epoch=8 end ************
2020-02-20 14:02:11,029 P7007 INFO [Metrics] logloss: 0.442260 - AUC: 0.809489
2020-02-20 14:02:11,154 P7007 INFO Save best model: monitor(max): 0.367229
2020-02-20 14:02:12,492 P7007 INFO --- 7335/7335 batches finished ---
2020-02-20 14:02:12,537 P7007 INFO Train loss: 0.441506
2020-02-20 14:02:12,537 P7007 INFO ************ Epoch=9 end ************
2020-02-20 16:14:15,946 P7007 INFO [Metrics] logloss: 0.442200 - AUC: 0.809543
2020-02-20 16:14:16,019 P7007 INFO Save best model: monitor(max): 0.367343
2020-02-20 16:14:17,317 P7007 INFO --- 7335/7335 batches finished ---
2020-02-20 16:14:17,356 P7007 INFO Train loss: 0.440976
2020-02-20 16:14:17,356 P7007 INFO ************ Epoch=10 end ************
2020-02-20 18:26:11,476 P7007 INFO [Metrics] logloss: 0.442091 - AUC: 0.809676
2020-02-20 18:26:11,555 P7007 INFO Save best model: monitor(max): 0.367584
2020-02-20 18:26:12,875 P7007 INFO --- 7335/7335 batches finished ---
2020-02-20 18:26:12,913 P7007 INFO Train loss: 0.440572
2020-02-20 18:26:12,913 P7007 INFO ************ Epoch=11 end ************
2020-02-20 20:38:07,858 P7007 INFO [Metrics] logloss: 0.442079 - AUC: 0.809680
2020-02-20 20:38:07,950 P7007 INFO Save best model: monitor(max): 0.367601
2020-02-20 20:38:09,254 P7007 INFO --- 7335/7335 batches finished ---
2020-02-20 20:38:09,298 P7007 INFO Train loss: 0.440239
2020-02-20 20:38:09,298 P7007 INFO ************ Epoch=12 end ************
2020-02-20 22:50:07,283 P7007 INFO [Metrics] logloss: 0.442029 - AUC: 0.809755
2020-02-20 22:50:07,361 P7007 INFO Save best model: monitor(max): 0.367726
2020-02-20 22:50:08,692 P7007 INFO --- 7335/7335 batches finished ---
2020-02-20 22:50:08,729 P7007 INFO Train loss: 0.439967
2020-02-20 22:50:08,730 P7007 INFO ************ Epoch=13 end ************
2020-02-21 01:02:01,504 P7007 INFO [Metrics] logloss: 0.441999 - AUC: 0.809768
2020-02-21 01:02:01,599 P7007 INFO Save best model: monitor(max): 0.367769
2020-02-21 01:02:02,909 P7007 INFO --- 7335/7335 batches finished ---
2020-02-21 01:02:02,949 P7007 INFO Train loss: 0.439718
2020-02-21 01:02:02,949 P7007 INFO ************ Epoch=14 end ************
2020-02-21 03:14:09,138 P7007 INFO [Metrics] logloss: 0.442040 - AUC: 0.809776
2020-02-21 03:14:09,325 P7007 INFO Monitor(max) STOP: 0.367737 !
2020-02-21 03:14:09,325 P7007 INFO Reduce learning rate on plateau: 0.000010
2020-02-21 03:14:09,325 P7007 INFO --- 7335/7335 batches finished ---
2020-02-21 03:14:09,363 P7007 INFO Train loss: 0.439504
2020-02-21 03:14:09,363 P7007 INFO ************ Epoch=15 end ************
2020-02-21 05:26:35,960 P7007 INFO [Metrics] logloss: 0.441699 - AUC: 0.810104
2020-02-21 05:26:36,074 P7007 INFO Save best model: monitor(max): 0.368405
2020-02-21 05:26:37,423 P7007 INFO --- 7335/7335 batches finished ---
2020-02-21 05:26:37,466 P7007 INFO Train loss: 0.434319
2020-02-21 05:26:37,466 P7007 INFO ************ Epoch=16 end ************
2020-02-21 07:38:47,426 P7007 INFO [Metrics] logloss: 0.441672 - AUC: 0.810132
2020-02-21 07:38:47,513 P7007 INFO Save best model: monitor(max): 0.368460
2020-02-21 07:38:48,776 P7007 INFO --- 7335/7335 batches finished ---
2020-02-21 07:38:48,815 P7007 INFO Train loss: 0.434028
2020-02-21 07:38:48,815 P7007 INFO ************ Epoch=17 end ************
2020-02-21 09:50:54,723 P7007 INFO [Metrics] logloss: 0.441670 - AUC: 0.810138
2020-02-21 09:50:54,953 P7007 INFO Save best model: monitor(max): 0.368467
2020-02-21 09:50:56,261 P7007 INFO --- 7335/7335 batches finished ---
2020-02-21 09:50:56,301 P7007 INFO Train loss: 0.433946
2020-02-21 09:50:56,301 P7007 INFO ************ Epoch=18 end ************
2020-02-21 12:03:02,026 P7007 INFO [Metrics] logloss: 0.441678 - AUC: 0.810139
2020-02-21 12:03:02,175 P7007 INFO Monitor(max) STOP: 0.368461 !
2020-02-21 12:03:02,175 P7007 INFO Reduce learning rate on plateau: 0.000001
2020-02-21 12:03:02,175 P7007 INFO --- 7335/7335 batches finished ---
2020-02-21 12:03:02,215 P7007 INFO Train loss: 0.433892
2020-02-21 12:03:02,215 P7007 INFO ************ Epoch=19 end ************
2020-02-21 14:15:08,000 P7007 INFO [Metrics] logloss: 0.441670 - AUC: 0.810142
2020-02-21 14:15:08,144 P7007 INFO Save best model: monitor(max): 0.368472
2020-02-21 14:15:09,454 P7007 INFO --- 7335/7335 batches finished ---
2020-02-21 14:15:09,496 P7007 INFO Train loss: 0.433092
2020-02-21 14:15:09,496 P7007 INFO ************ Epoch=20 end ************
2020-02-21 16:27:15,913 P7007 INFO [Metrics] logloss: 0.441670 - AUC: 0.810143
2020-02-21 16:27:16,035 P7007 INFO Monitor(max) STOP: 0.368473 !
2020-02-21 16:27:16,035 P7007 INFO Reduce learning rate on plateau: 0.000001
2020-02-21 16:27:16,035 P7007 INFO --- 7335/7335 batches finished ---
2020-02-21 16:27:16,099 P7007 INFO Train loss: 0.433093
2020-02-21 16:27:16,099 P7007 INFO ************ Epoch=21 end ************
2020-02-21 18:39:34,370 P7007 INFO [Metrics] logloss: 0.441670 - AUC: 0.810143
2020-02-21 18:39:34,472 P7007 INFO Monitor(max) STOP: 0.368473 !
2020-02-21 18:39:34,473 P7007 INFO Reduce learning rate on plateau: 0.000001
2020-02-21 18:39:34,473 P7007 INFO Early stopping at epoch=22
2020-02-21 18:39:34,473 P7007 INFO --- 7335/7335 batches finished ---
2020-02-21 18:39:34,534 P7007 INFO Train loss: 0.433096
2020-02-21 18:39:34,535 P7007 INFO Training finished.
2020-02-21 18:39:34,535 P7007 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Criteo/LorentzFM_criteo/criteo_x4_001_be98441d/LorentzFM_criteo_x4_002_235e40ca_criteo_x4_001_be98441d_model.ckpt
2020-02-21 18:39:35,731 P7007 INFO ****** Train/validation evaluation ******
2020-02-21 19:20:34,100 P7007 INFO [Metrics] logloss: 0.425428 - AUC: 0.827736
2020-02-21 19:25:36,493 P7007 INFO [Metrics] logloss: 0.441670 - AUC: 0.810142
2020-02-21 19:25:36,745 P7007 INFO ******** Test evaluation ********
2020-02-21 19:25:36,745 P7007 INFO Loading data...
2020-02-21 19:25:36,745 P7007 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-02-21 19:25:37,472 P7007 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-21 19:25:37,472 P7007 INFO Loading test data done.
2020-02-21 19:30:31,810 P7007 INFO [Metrics] logloss: 0.441322 - AUC: 0.810550

```
