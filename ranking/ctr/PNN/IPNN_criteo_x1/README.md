## IPNN_criteo_x1

A hands-on guide to run the PNN model on the Criteo_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
  GPU: Tesla P100 16G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 11.4
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  fuxictr: 1.1.0
  ```

### Dataset
Dataset ID: [Criteo_x1](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [PNN](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/PNN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [PNN_criteo_x1_tuner_config_01](./PNN_criteo_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd IPNN_criteo_x1
    nohup python run_expid.py --config ./PNN_criteo_x1_tuner_config_01 --expid PNN_criteo_x1_007_491aff75 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.813664 | 0.438297  |


### Logs
```python
2020-12-25 17:40:48,024 P33296 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_ace9c1b9",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "PNN",
    "model_id": "PNN_criteo_x1_007_2cb55658",
    "model_root": "./Criteo/PNN_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "3",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2020-12-25 17:40:48,025 P33296 INFO Set up feature encoder...
2020-12-25 17:40:48,025 P33296 INFO Load feature_encoder from pickle: ../data/Criteo/criteo_x1_ace9c1b9/feature_encoder.pkl
2020-12-25 17:40:49,592 P33296 INFO Total number of parameters: 21639561.
2020-12-25 17:40:49,593 P33296 INFO Loading data...
2020-12-25 17:40:49,595 P33296 INFO Loading data from h5: ../data/Criteo/criteo_x1_ace9c1b9/train.h5
2020-12-25 17:40:54,202 P33296 INFO Loading data from h5: ../data/Criteo/criteo_x1_ace9c1b9/valid.h5
2020-12-25 17:40:55,702 P33296 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2020-12-25 17:40:55,702 P33296 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2020-12-25 17:40:55,702 P33296 INFO Loading train data done.
2020-12-25 17:40:58,488 P33296 INFO Start training: 8058 batches/epoch
2020-12-25 17:40:58,488 P33296 INFO ************ Epoch=1 start ************
2020-12-25 17:49:24,077 P33296 INFO [Metrics] AUC: 0.804696 - logloss: 0.446617
2020-12-25 17:49:24,079 P33296 INFO Save best model: monitor(max): 0.804696
2020-12-25 17:49:24,161 P33296 INFO --- 8058/8058 batches finished ---
2020-12-25 17:49:24,237 P33296 INFO Train loss: 0.461818
2020-12-25 17:49:24,237 P33296 INFO ************ Epoch=1 end ************
2020-12-25 17:57:49,054 P33296 INFO [Metrics] AUC: 0.807003 - logloss: 0.444455
2020-12-25 17:57:49,055 P33296 INFO Save best model: monitor(max): 0.807003
2020-12-25 17:57:49,190 P33296 INFO --- 8058/8058 batches finished ---
2020-12-25 17:57:49,275 P33296 INFO Train loss: 0.456325
2020-12-25 17:57:49,275 P33296 INFO ************ Epoch=2 end ************
2020-12-25 18:06:10,828 P33296 INFO [Metrics] AUC: 0.807938 - logloss: 0.443643
2020-12-25 18:06:10,829 P33296 INFO Save best model: monitor(max): 0.807938
2020-12-25 18:06:10,964 P33296 INFO --- 8058/8058 batches finished ---
2020-12-25 18:06:11,051 P33296 INFO Train loss: 0.454747
2020-12-25 18:06:11,051 P33296 INFO ************ Epoch=3 end ************
2020-12-25 18:14:33,483 P33296 INFO [Metrics] AUC: 0.808534 - logloss: 0.443059
2020-12-25 18:14:33,484 P33296 INFO Save best model: monitor(max): 0.808534
2020-12-25 18:14:33,642 P33296 INFO --- 8058/8058 batches finished ---
2020-12-25 18:14:33,728 P33296 INFO Train loss: 0.453918
2020-12-25 18:14:33,728 P33296 INFO ************ Epoch=4 end ************
2020-12-25 18:22:57,796 P33296 INFO [Metrics] AUC: 0.808795 - logloss: 0.442821
2020-12-25 18:22:57,797 P33296 INFO Save best model: monitor(max): 0.808795
2020-12-25 18:22:57,949 P33296 INFO --- 8058/8058 batches finished ---
2020-12-25 18:22:58,037 P33296 INFO Train loss: 0.453344
2020-12-25 18:22:58,037 P33296 INFO ************ Epoch=5 end ************
2020-12-25 18:31:22,708 P33296 INFO [Metrics] AUC: 0.809132 - logloss: 0.442756
2020-12-25 18:31:22,709 P33296 INFO Save best model: monitor(max): 0.809132
2020-12-25 18:31:22,843 P33296 INFO --- 8058/8058 batches finished ---
2020-12-25 18:31:22,942 P33296 INFO Train loss: 0.452950
2020-12-25 18:31:22,942 P33296 INFO ************ Epoch=6 end ************
2020-12-25 18:39:45,534 P33296 INFO [Metrics] AUC: 0.809443 - logloss: 0.442192
2020-12-25 18:39:45,536 P33296 INFO Save best model: monitor(max): 0.809443
2020-12-25 18:39:45,692 P33296 INFO --- 8058/8058 batches finished ---
2020-12-25 18:39:45,779 P33296 INFO Train loss: 0.452626
2020-12-25 18:39:45,779 P33296 INFO ************ Epoch=7 end ************
2020-12-25 18:48:08,579 P33296 INFO [Metrics] AUC: 0.809424 - logloss: 0.442210
2020-12-25 18:48:08,580 P33296 INFO Monitor(max) STOP: 0.809424 !
2020-12-25 18:48:08,581 P33296 INFO Reduce learning rate on plateau: 0.000100
2020-12-25 18:48:08,581 P33296 INFO --- 8058/8058 batches finished ---
2020-12-25 18:48:08,666 P33296 INFO Train loss: 0.452345
2020-12-25 18:48:08,666 P33296 INFO ************ Epoch=8 end ************
2020-12-25 18:56:31,825 P33296 INFO [Metrics] AUC: 0.812772 - logloss: 0.439155
2020-12-25 18:56:31,826 P33296 INFO Save best model: monitor(max): 0.812772
2020-12-25 18:56:31,960 P33296 INFO --- 8058/8058 batches finished ---
2020-12-25 18:56:32,043 P33296 INFO Train loss: 0.441234
2020-12-25 18:56:32,043 P33296 INFO ************ Epoch=9 end ************
2020-12-25 19:04:55,785 P33296 INFO [Metrics] AUC: 0.813280 - logloss: 0.438684
2020-12-25 19:04:55,787 P33296 INFO Save best model: monitor(max): 0.813280
2020-12-25 19:04:55,933 P33296 INFO --- 8058/8058 batches finished ---
2020-12-25 19:04:56,017 P33296 INFO Train loss: 0.437012
2020-12-25 19:04:56,017 P33296 INFO ************ Epoch=10 end ************
2020-12-25 19:13:19,534 P33296 INFO [Metrics] AUC: 0.813369 - logloss: 0.438737
2020-12-25 19:13:19,536 P33296 INFO Save best model: monitor(max): 0.813369
2020-12-25 19:13:19,661 P33296 INFO --- 8058/8058 batches finished ---
2020-12-25 19:13:19,750 P33296 INFO Train loss: 0.435124
2020-12-25 19:13:19,750 P33296 INFO ************ Epoch=11 end ************
2020-12-25 19:21:44,538 P33296 INFO [Metrics] AUC: 0.813228 - logloss: 0.438880
2020-12-25 19:21:44,540 P33296 INFO Monitor(max) STOP: 0.813228 !
2020-12-25 19:21:44,540 P33296 INFO Reduce learning rate on plateau: 0.000010
2020-12-25 19:21:44,540 P33296 INFO --- 8058/8058 batches finished ---
2020-12-25 19:21:44,626 P33296 INFO Train loss: 0.433690
2020-12-25 19:21:44,626 P33296 INFO ************ Epoch=12 end ************
2020-12-25 19:30:10,506 P33296 INFO [Metrics] AUC: 0.812710 - logloss: 0.439834
2020-12-25 19:30:10,508 P33296 INFO Monitor(max) STOP: 0.812710 !
2020-12-25 19:30:10,508 P33296 INFO Reduce learning rate on plateau: 0.000001
2020-12-25 19:30:10,508 P33296 INFO --- 8058/8058 batches finished ---
2020-12-25 19:30:10,593 P33296 INFO Train loss: 0.429032
2020-12-25 19:30:10,593 P33296 INFO ************ Epoch=13 end ************
2020-12-25 19:38:35,161 P33296 INFO [Metrics] AUC: 0.812704 - logloss: 0.439865
2020-12-25 19:38:35,162 P33296 INFO Monitor(max) STOP: 0.812704 !
2020-12-25 19:38:35,162 P33296 INFO Reduce learning rate on plateau: 0.000001
2020-12-25 19:38:35,162 P33296 INFO Early stopping at epoch=14
2020-12-25 19:38:35,163 P33296 INFO --- 8058/8058 batches finished ---
2020-12-25 19:38:35,249 P33296 INFO Train loss: 0.428119
2020-12-25 19:38:35,249 P33296 INFO Training finished.
2020-12-25 19:38:35,249 P33296 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Criteo/PNN_criteo_x1/criteo_x1_ace9c1b9/PNN_criteo_x1_007_2cb55658_model.ckpt
2020-12-25 19:38:35,366 P33296 INFO ****** Train/validation evaluation ******
2020-12-25 19:39:06,917 P33296 INFO [Metrics] AUC: 0.813369 - logloss: 0.438737
2020-12-25 19:39:07,027 P33296 INFO ******** Test evaluation ********
2020-12-25 19:39:07,027 P33296 INFO Loading data...
2020-12-25 19:39:07,028 P33296 INFO Loading data from h5: ../data/Criteo/criteo_x1_ace9c1b9/test.h5
2020-12-25 19:39:07,791 P33296 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2020-12-25 19:39:07,791 P33296 INFO Loading test data done.
2020-12-25 19:39:26,213 P33296 INFO [Metrics] AUC: 0.813664 - logloss: 0.438297

```
