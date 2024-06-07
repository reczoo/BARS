## LR_criteo_x4_001

A hands-on guide to run the LR model on the Criteo_x4_001 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

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
Dataset ID: [Criteo_x4](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x4). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [LR](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/LR.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [LR_criteo_x4_tuner_config_01](./LR_criteo_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd LR_criteo_x4_001
    nohup python run_expid.py --config ./LR_criteo_x4_tuner_config_01 --expid LR_criteo_x4_003_76cc7982 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.456754 | 0.793359  |


### Logs
```python
2021-09-08 11:32:08,692 P512 INFO {
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_9ea3bdfc",
    "debug": "False",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "2",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "LR",
    "model_id": "LR_criteo_x4_9ea3bdfc_003_0a7da2fd",
    "model_root": "./Criteo/LR_criteo_x4_001/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-07",
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
2021-09-08 11:32:08,693 P512 INFO Set up feature encoder...
2021-09-08 11:32:08,693 P512 INFO Load feature_map from json: ../data/Criteo/criteo_x4_9ea3bdfc/feature_map.json
2021-09-08 11:32:08,955 P512 INFO Total number of parameters: 910709.
2021-09-08 11:32:08,955 P512 INFO Loading data...
2021-09-08 11:32:08,957 P512 INFO Loading data from h5: ../data/Criteo/criteo_x4_9ea3bdfc/train.h5
2021-09-08 11:32:13,955 P512 INFO Loading data from h5: ../data/Criteo/criteo_x4_9ea3bdfc/valid.h5
2021-09-08 11:32:15,498 P512 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2021-09-08 11:32:15,620 P512 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2021-09-08 11:32:15,620 P512 INFO Loading train data done.
2021-09-08 11:32:18,759 P512 INFO Start training: 3668 batches/epoch
2021-09-08 11:32:18,759 P512 INFO ************ Epoch=1 start ************
2021-09-08 11:36:31,394 P512 INFO [Metrics] logloss: 0.458613 - AUC: 0.791112
2021-09-08 11:36:31,396 P512 INFO Save best model: monitor(max): 0.332498
2021-09-08 11:36:31,564 P512 INFO --- 3668/3668 batches finished ---
2021-09-08 11:36:31,600 P512 INFO Train loss: 0.466917
2021-09-08 11:36:31,601 P512 INFO ************ Epoch=1 end ************
2021-09-08 11:40:43,383 P512 INFO [Metrics] logloss: 0.457557 - AUC: 0.792313
2021-09-08 11:40:43,384 P512 INFO Save best model: monitor(max): 0.334756
2021-09-08 11:40:43,391 P512 INFO --- 3668/3668 batches finished ---
2021-09-08 11:40:43,450 P512 INFO Train loss: 0.456525
2021-09-08 11:40:43,450 P512 INFO ************ Epoch=2 end ************
2021-09-08 11:44:56,868 P512 INFO [Metrics] logloss: 0.457449 - AUC: 0.792475
2021-09-08 11:44:56,869 P512 INFO Save best model: monitor(max): 0.335026
2021-09-08 11:44:56,876 P512 INFO --- 3668/3668 batches finished ---
2021-09-08 11:44:56,922 P512 INFO Train loss: 0.455342
2021-09-08 11:44:56,922 P512 INFO ************ Epoch=3 end ************
2021-09-08 11:49:06,145 P512 INFO [Metrics] logloss: 0.457373 - AUC: 0.792578
2021-09-08 11:49:06,146 P512 INFO Save best model: monitor(max): 0.335205
2021-09-08 11:49:06,153 P512 INFO --- 3668/3668 batches finished ---
2021-09-08 11:49:06,192 P512 INFO Train loss: 0.454908
2021-09-08 11:49:06,192 P512 INFO ************ Epoch=4 end ************
2021-09-08 11:53:15,218 P512 INFO [Metrics] logloss: 0.457389 - AUC: 0.792544
2021-09-08 11:53:15,220 P512 INFO Monitor(max) STOP: 0.335156 !
2021-09-08 11:53:15,220 P512 INFO Reduce learning rate on plateau: 0.000100
2021-09-08 11:53:15,220 P512 INFO --- 3668/3668 batches finished ---
2021-09-08 11:53:15,260 P512 INFO Train loss: 0.454700
2021-09-08 11:53:15,260 P512 INFO ************ Epoch=5 end ************
2021-09-08 11:57:24,331 P512 INFO [Metrics] logloss: 0.457121 - AUC: 0.792863
2021-09-08 11:57:24,332 P512 INFO Save best model: monitor(max): 0.335742
2021-09-08 11:57:24,340 P512 INFO --- 3668/3668 batches finished ---
2021-09-08 11:57:24,379 P512 INFO Train loss: 0.452973
2021-09-08 11:57:24,379 P512 INFO ************ Epoch=6 end ************
2021-09-08 12:01:30,753 P512 INFO [Metrics] logloss: 0.457080 - AUC: 0.792916
2021-09-08 12:01:30,755 P512 INFO Save best model: monitor(max): 0.335835
2021-09-08 12:01:30,761 P512 INFO --- 3668/3668 batches finished ---
2021-09-08 12:01:30,803 P512 INFO Train loss: 0.452851
2021-09-08 12:01:30,803 P512 INFO ************ Epoch=7 end ************
2021-09-08 12:05:39,946 P512 INFO [Metrics] logloss: 0.457072 - AUC: 0.792937
2021-09-08 12:05:39,948 P512 INFO Save best model: monitor(max): 0.335865
2021-09-08 12:05:39,954 P512 INFO --- 3668/3668 batches finished ---
2021-09-08 12:05:39,995 P512 INFO Train loss: 0.452814
2021-09-08 12:05:39,995 P512 INFO ************ Epoch=8 end ************
2021-09-08 12:09:50,335 P512 INFO [Metrics] logloss: 0.457064 - AUC: 0.792936
2021-09-08 12:09:50,337 P512 INFO Save best model: monitor(max): 0.335872
2021-09-08 12:09:50,345 P512 INFO --- 3668/3668 batches finished ---
2021-09-08 12:09:50,392 P512 INFO Train loss: 0.452793
2021-09-08 12:09:50,392 P512 INFO ************ Epoch=9 end ************
2021-09-08 12:14:02,614 P512 INFO [Metrics] logloss: 0.457059 - AUC: 0.792945
2021-09-08 12:14:02,616 P512 INFO Save best model: monitor(max): 0.335887
2021-09-08 12:14:02,623 P512 INFO --- 3668/3668 batches finished ---
2021-09-08 12:14:02,666 P512 INFO Train loss: 0.452776
2021-09-08 12:14:02,666 P512 INFO ************ Epoch=10 end ************
2021-09-08 12:18:11,883 P512 INFO [Metrics] logloss: 0.457051 - AUC: 0.792944
2021-09-08 12:18:11,884 P512 INFO Save best model: monitor(max): 0.335894
2021-09-08 12:18:11,891 P512 INFO --- 3668/3668 batches finished ---
2021-09-08 12:18:11,929 P512 INFO Train loss: 0.452757
2021-09-08 12:18:11,929 P512 INFO ************ Epoch=11 end ************
2021-09-08 12:22:20,598 P512 INFO [Metrics] logloss: 0.457054 - AUC: 0.792952
2021-09-08 12:22:20,599 P512 INFO Save best model: monitor(max): 0.335898
2021-09-08 12:22:20,606 P512 INFO --- 3668/3668 batches finished ---
2021-09-08 12:22:20,648 P512 INFO Train loss: 0.452755
2021-09-08 12:22:20,649 P512 INFO ************ Epoch=12 end ************
2021-09-08 12:26:24,541 P512 INFO [Metrics] logloss: 0.457059 - AUC: 0.792943
2021-09-08 12:26:24,543 P512 INFO Monitor(max) STOP: 0.335884 !
2021-09-08 12:26:24,543 P512 INFO Reduce learning rate on plateau: 0.000010
2021-09-08 12:26:24,543 P512 INFO --- 3668/3668 batches finished ---
2021-09-08 12:26:24,590 P512 INFO Train loss: 0.452744
2021-09-08 12:26:24,590 P512 INFO ************ Epoch=13 end ************
2021-09-08 12:30:28,964 P512 INFO [Metrics] logloss: 0.457051 - AUC: 0.792950
2021-09-08 12:30:28,966 P512 INFO Monitor(max) STOP: 0.335899 !
2021-09-08 12:30:28,966 P512 INFO Reduce learning rate on plateau: 0.000001
2021-09-08 12:30:28,966 P512 INFO Early stopping at epoch=14
2021-09-08 12:30:28,966 P512 INFO --- 3668/3668 batches finished ---
2021-09-08 12:30:29,006 P512 INFO Train loss: 0.452529
2021-09-08 12:30:29,006 P512 INFO Training finished.
2021-09-08 12:30:29,007 P512 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/LR_criteo_x4_001/criteo_x4_9ea3bdfc/LR_criteo_x4_9ea3bdfc_003_0a7da2fd_model.ckpt
2021-09-08 12:30:29,027 P512 INFO ****** Train/validation evaluation ******
2021-09-08 12:30:52,751 P512 INFO [Metrics] logloss: 0.457054 - AUC: 0.792952
2021-09-08 12:30:52,822 P512 INFO ******** Test evaluation ********
2021-09-08 12:30:52,822 P512 INFO Loading data...
2021-09-08 12:30:52,822 P512 INFO Loading data from h5: ../data/Criteo/criteo_x4_9ea3bdfc/test.h5
2021-09-08 12:30:53,715 P512 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2021-09-08 12:30:53,715 P512 INFO Loading test data done.
2021-09-08 12:31:16,538 P512 INFO [Metrics] logloss: 0.456754 - AUC: 0.793359

```
