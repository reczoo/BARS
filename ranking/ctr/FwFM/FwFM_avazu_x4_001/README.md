## FwFM_avazu_x4_001

A hands-on guide to run the FwFM model on the Avazu_x4_001 dataset.

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
Dataset ID: [Avazu_x4](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x4). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FwFM](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FwFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FwFM_avazu_x4_tuner_config_01](./FwFM_avazu_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FwFM_avazu_x4_001
    nohup python run_expid.py --config ./FwFM_avazu_x4_tuner_config_01 --expid FwFM_avazu_x4_012_3a5f1b67 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.374422 | 0.790681  |


### Logs
```python
2020-06-15 13:30:17,919 P4137 INFO {
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "embedding_dim": "16",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "learning_rate": "0.001",
    "linear_type": "FiLV",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FwFM",
    "model_id": "FwFM_avazu_x4_3bbbc4c9_012_57c9a942",
    "model_root": "./Avazu/FwFM_avazu/min2/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-06",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/avazu_x4_3bbbc4c9/test.h5",
    "train_data": "../data/Avazu/avazu_x4_3bbbc4c9/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/avazu_x4_3bbbc4c9/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-15 13:30:17,931 P4137 INFO Set up feature encoder...
2020-06-15 13:30:17,931 P4137 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-06-15 13:30:17,931 P4137 INFO Loading data...
2020-06-15 13:30:17,937 P4137 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-06-15 13:30:21,515 P4137 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-06-15 13:30:23,104 P4137 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-15 13:30:23,278 P4137 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-15 13:30:23,279 P4137 INFO Loading train data done.
2020-06-15 13:30:29,637 P4137 INFO Start training: 3235 batches/epoch
2020-06-15 13:30:29,637 P4137 INFO ************ Epoch=1 start ************
2020-06-15 13:34:43,544 P4137 INFO [Metrics] logloss: 0.379760 - AUC: 0.780668
2020-06-15 13:34:43,546 P4137 INFO Save best model: monitor(max): 0.400908
2020-06-15 13:34:44,013 P4137 INFO --- 3235/3235 batches finished ---
2020-06-15 13:34:44,058 P4137 INFO Train loss: 0.396854
2020-06-15 13:34:44,059 P4137 INFO ************ Epoch=1 end ************
2020-06-15 13:38:56,929 P4137 INFO [Metrics] logloss: 0.374645 - AUC: 0.789315
2020-06-15 13:38:56,932 P4137 INFO Save best model: monitor(max): 0.414671
2020-06-15 13:38:57,408 P4137 INFO --- 3235/3235 batches finished ---
2020-06-15 13:38:57,474 P4137 INFO Train loss: 0.381845
2020-06-15 13:38:57,474 P4137 INFO ************ Epoch=2 end ************
2020-06-15 13:43:09,744 P4137 INFO [Metrics] logloss: 0.374374 - AUC: 0.790695
2020-06-15 13:43:09,747 P4137 INFO Save best model: monitor(max): 0.416321
2020-06-15 13:43:10,230 P4137 INFO --- 3235/3235 batches finished ---
2020-06-15 13:43:10,279 P4137 INFO Train loss: 0.369504
2020-06-15 13:43:10,279 P4137 INFO ************ Epoch=3 end ************
2020-06-15 13:47:21,541 P4137 INFO [Metrics] logloss: 0.375582 - AUC: 0.789973
2020-06-15 13:47:21,543 P4137 INFO Monitor(max) STOP: 0.414390 !
2020-06-15 13:47:21,544 P4137 INFO Reduce learning rate on plateau: 0.000100
2020-06-15 13:47:21,544 P4137 INFO --- 3235/3235 batches finished ---
2020-06-15 13:47:21,594 P4137 INFO Train loss: 0.361119
2020-06-15 13:47:21,594 P4137 INFO ************ Epoch=4 end ************
2020-06-15 13:51:32,092 P4137 INFO [Metrics] logloss: 0.385940 - AUC: 0.785312
2020-06-15 13:51:32,096 P4137 INFO Monitor(max) STOP: 0.399372 !
2020-06-15 13:51:32,096 P4137 INFO Reduce learning rate on plateau: 0.000010
2020-06-15 13:51:32,096 P4137 INFO Early stopping at epoch=5
2020-06-15 13:51:32,096 P4137 INFO --- 3235/3235 batches finished ---
2020-06-15 13:51:32,144 P4137 INFO Train loss: 0.326130
2020-06-15 13:51:32,144 P4137 INFO Training finished.
2020-06-15 13:51:32,144 P4137 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Avazu/FwFM_avazu/min2/avazu_x4_3bbbc4c9/FwFM_avazu_x4_3bbbc4c9_012_57c9a942_model.ckpt
2020-06-15 13:51:32,490 P4137 INFO ****** Train/validation evaluation ******
2020-06-15 13:54:54,049 P4137 INFO [Metrics] logloss: 0.321964 - AUC: 0.863177
2020-06-15 13:55:17,115 P4137 INFO [Metrics] logloss: 0.374374 - AUC: 0.790695
2020-06-15 13:55:17,200 P4137 INFO ******** Test evaluation ********
2020-06-15 13:55:17,200 P4137 INFO Loading data...
2020-06-15 13:55:17,200 P4137 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-06-15 13:55:17,793 P4137 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-15 13:55:17,793 P4137 INFO Loading test data done.
2020-06-15 13:55:41,332 P4137 INFO [Metrics] logloss: 0.374422 - AUC: 0.790681

```
