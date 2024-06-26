## xDeepFM_criteo_x1

A hands-on guide to run the xDeepFM model on the Criteo_x1 dataset.

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
  fuxictr: 1.1.0

  ```

### Dataset
Dataset ID: [Criteo_x1](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [xDeepFM](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/xDeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [xDeepFM_criteo_x1_tuner_config_03](./xDeepFM_criteo_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd xDeepFM_criteo_x1
    nohup python run_expid.py --config ./xDeepFM_criteo_x1_tuner_config_03 --expid xDeepFM_criteo_x1_001_e08ec7de --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.813901 | 0.438018  |
| 2 | 0.813682 | 0.438202  |
| 3 | 0.813670 | 0.438280  |
| 4 | 0.813826 | 0.438003  |
| 5 | 0.813861 | 0.437991  |
| Avg | 0.813788 | 0.438099 |
| Std | &#177;0.00009455 | &#177;0.00011900 |


### Logs
```python
2022-01-20 19:01:59,319 P648 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "cin_layer_units": "[16, 16]",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "xDeepFM",
    "model_id": "xDeepFM_criteo_x1_001_e08ec7de",
    "model_root": "./Criteo/xDeepFM_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
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
2022-01-20 19:01:59,320 P648 INFO Set up feature encoder...
2022-01-20 19:01:59,320 P648 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-01-20 19:01:59,320 P648 INFO Loading data...
2022-01-20 19:01:59,321 P648 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-01-20 19:02:03,690 P648 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-01-20 19:02:04,862 P648 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-01-20 19:02:04,862 P648 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-01-20 19:02:04,862 P648 INFO Loading train data done.
2022-01-20 19:02:09,484 P648 INFO Total number of parameters: 23461462.
2022-01-20 19:02:09,484 P648 INFO Start training: 8058 batches/epoch
2022-01-20 19:02:09,484 P648 INFO ************ Epoch=1 start ************
2022-01-20 19:14:59,584 P648 INFO [Metrics] AUC: 0.804225 - logloss: 0.447041
2022-01-20 19:14:59,586 P648 INFO Save best model: monitor(max): 0.804225
2022-01-20 19:14:59,679 P648 INFO --- 8058/8058 batches finished ---
2022-01-20 19:14:59,708 P648 INFO Train loss: 0.462615
2022-01-20 19:14:59,708 P648 INFO ************ Epoch=1 end ************
2022-01-20 19:27:40,564 P648 INFO [Metrics] AUC: 0.805953 - logloss: 0.445531
2022-01-20 19:27:40,565 P648 INFO Save best model: monitor(max): 0.805953
2022-01-20 19:27:40,675 P648 INFO --- 8058/8058 batches finished ---
2022-01-20 19:27:40,712 P648 INFO Train loss: 0.458243
2022-01-20 19:27:40,712 P648 INFO ************ Epoch=2 end ************
2022-01-20 19:40:35,275 P648 INFO [Metrics] AUC: 0.806886 - logloss: 0.445106
2022-01-20 19:40:35,277 P648 INFO Save best model: monitor(max): 0.806886
2022-01-20 19:40:35,378 P648 INFO --- 8058/8058 batches finished ---
2022-01-20 19:40:35,413 P648 INFO Train loss: 0.457055
2022-01-20 19:40:35,414 P648 INFO ************ Epoch=3 end ************
2022-01-20 19:53:13,291 P648 INFO [Metrics] AUC: 0.807739 - logloss: 0.443781
2022-01-20 19:53:13,292 P648 INFO Save best model: monitor(max): 0.807739
2022-01-20 19:53:13,399 P648 INFO --- 8058/8058 batches finished ---
2022-01-20 19:53:13,430 P648 INFO Train loss: 0.456303
2022-01-20 19:53:13,430 P648 INFO ************ Epoch=4 end ************
2022-01-20 20:05:53,261 P648 INFO [Metrics] AUC: 0.807945 - logloss: 0.443774
2022-01-20 20:05:53,263 P648 INFO Save best model: monitor(max): 0.807945
2022-01-20 20:05:53,360 P648 INFO --- 8058/8058 batches finished ---
2022-01-20 20:05:53,395 P648 INFO Train loss: 0.455786
2022-01-20 20:05:53,395 P648 INFO ************ Epoch=5 end ************
2022-01-20 20:18:32,496 P648 INFO [Metrics] AUC: 0.808221 - logloss: 0.443299
2022-01-20 20:18:32,497 P648 INFO Save best model: monitor(max): 0.808221
2022-01-20 20:18:32,605 P648 INFO --- 8058/8058 batches finished ---
2022-01-20 20:18:32,635 P648 INFO Train loss: 0.455404
2022-01-20 20:18:32,636 P648 INFO ************ Epoch=6 end ************
2022-01-20 20:31:13,121 P648 INFO [Metrics] AUC: 0.808351 - logloss: 0.443175
2022-01-20 20:31:13,122 P648 INFO Save best model: monitor(max): 0.808351
2022-01-20 20:31:13,221 P648 INFO --- 8058/8058 batches finished ---
2022-01-20 20:31:13,256 P648 INFO Train loss: 0.455091
2022-01-20 20:31:13,256 P648 INFO ************ Epoch=7 end ************
2022-01-20 20:43:53,848 P648 INFO [Metrics] AUC: 0.808433 - logloss: 0.443121
2022-01-20 20:43:53,850 P648 INFO Save best model: monitor(max): 0.808433
2022-01-20 20:43:53,957 P648 INFO --- 8058/8058 batches finished ---
2022-01-20 20:43:53,985 P648 INFO Train loss: 0.454894
2022-01-20 20:43:53,986 P648 INFO ************ Epoch=8 end ************
2022-01-20 20:56:33,533 P648 INFO [Metrics] AUC: 0.808658 - logloss: 0.442974
2022-01-20 20:56:33,535 P648 INFO Save best model: monitor(max): 0.808658
2022-01-20 20:56:33,653 P648 INFO --- 8058/8058 batches finished ---
2022-01-20 20:56:33,681 P648 INFO Train loss: 0.454702
2022-01-20 20:56:33,681 P648 INFO ************ Epoch=9 end ************
2022-01-20 21:09:19,633 P648 INFO [Metrics] AUC: 0.808749 - logloss: 0.443122
2022-01-20 21:09:19,634 P648 INFO Save best model: monitor(max): 0.808749
2022-01-20 21:09:19,765 P648 INFO --- 8058/8058 batches finished ---
2022-01-20 21:09:19,799 P648 INFO Train loss: 0.454540
2022-01-20 21:09:19,799 P648 INFO ************ Epoch=10 end ************
2022-01-20 21:22:00,656 P648 INFO [Metrics] AUC: 0.808979 - logloss: 0.442578
2022-01-20 21:22:00,657 P648 INFO Save best model: monitor(max): 0.808979
2022-01-20 21:22:00,760 P648 INFO --- 8058/8058 batches finished ---
2022-01-20 21:22:00,793 P648 INFO Train loss: 0.454410
2022-01-20 21:22:00,793 P648 INFO ************ Epoch=11 end ************
2022-01-20 21:34:39,387 P648 INFO [Metrics] AUC: 0.808935 - logloss: 0.442636
2022-01-20 21:34:39,388 P648 INFO Monitor(max) STOP: 0.808935 !
2022-01-20 21:34:39,388 P648 INFO Reduce learning rate on plateau: 0.000100
2022-01-20 21:34:39,389 P648 INFO --- 8058/8058 batches finished ---
2022-01-20 21:34:39,421 P648 INFO Train loss: 0.454333
2022-01-20 21:34:39,421 P648 INFO ************ Epoch=12 end ************
2022-01-20 21:47:18,898 P648 INFO [Metrics] AUC: 0.812764 - logloss: 0.439095
2022-01-20 21:47:18,900 P648 INFO Save best model: monitor(max): 0.812764
2022-01-20 21:47:19,006 P648 INFO --- 8058/8058 batches finished ---
2022-01-20 21:47:19,036 P648 INFO Train loss: 0.443009
2022-01-20 21:47:19,036 P648 INFO ************ Epoch=13 end ************
2022-01-20 21:59:56,233 P648 INFO [Metrics] AUC: 0.813290 - logloss: 0.438624
2022-01-20 21:59:56,235 P648 INFO Save best model: monitor(max): 0.813290
2022-01-20 21:59:56,344 P648 INFO --- 8058/8058 batches finished ---
2022-01-20 21:59:56,379 P648 INFO Train loss: 0.438848
2022-01-20 21:59:56,379 P648 INFO ************ Epoch=14 end ************
2022-01-20 22:12:33,683 P648 INFO [Metrics] AUC: 0.813470 - logloss: 0.438456
2022-01-20 22:12:33,684 P648 INFO Save best model: monitor(max): 0.813470
2022-01-20 22:12:33,806 P648 INFO --- 8058/8058 batches finished ---
2022-01-20 22:12:33,838 P648 INFO Train loss: 0.437168
2022-01-20 22:12:33,838 P648 INFO ************ Epoch=15 end ************
2022-01-20 22:25:07,451 P648 INFO [Metrics] AUC: 0.813498 - logloss: 0.438523
2022-01-20 22:25:07,452 P648 INFO Save best model: monitor(max): 0.813498
2022-01-20 22:25:07,553 P648 INFO --- 8058/8058 batches finished ---
2022-01-20 22:25:07,589 P648 INFO Train loss: 0.435895
2022-01-20 22:25:07,589 P648 INFO ************ Epoch=16 end ************
2022-01-20 22:37:41,603 P648 INFO [Metrics] AUC: 0.813384 - logloss: 0.438650
2022-01-20 22:37:41,604 P648 INFO Monitor(max) STOP: 0.813384 !
2022-01-20 22:37:41,604 P648 INFO Reduce learning rate on plateau: 0.000010
2022-01-20 22:37:41,604 P648 INFO --- 8058/8058 batches finished ---
2022-01-20 22:37:41,635 P648 INFO Train loss: 0.434798
2022-01-20 22:37:41,636 P648 INFO ************ Epoch=17 end ************
2022-01-20 22:50:14,129 P648 INFO [Metrics] AUC: 0.813149 - logloss: 0.439090
2022-01-20 22:50:14,131 P648 INFO Monitor(max) STOP: 0.813149 !
2022-01-20 22:50:14,131 P648 INFO Reduce learning rate on plateau: 0.000001
2022-01-20 22:50:14,131 P648 INFO Early stopping at epoch=18
2022-01-20 22:50:14,131 P648 INFO --- 8058/8058 batches finished ---
2022-01-20 22:50:14,160 P648 INFO Train loss: 0.430277
2022-01-20 22:50:14,160 P648 INFO Training finished.
2022-01-20 22:50:14,160 P648 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/xDeepFM_criteo_x1/criteo_x1_7b681156/xDeepFM_criteo_x1_001_e08ec7de.model
2022-01-20 22:50:14,224 P648 INFO ****** Validation evaluation ******
2022-01-20 22:50:43,114 P648 INFO [Metrics] AUC: 0.813498 - logloss: 0.438523
2022-01-20 22:50:43,204 P648 INFO ******** Test evaluation ********
2022-01-20 22:50:43,205 P648 INFO Loading data...
2022-01-20 22:50:43,205 P648 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-01-20 22:50:43,982 P648 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-01-20 22:50:43,982 P648 INFO Loading test data done.
2022-01-20 22:51:01,051 P648 INFO [Metrics] AUC: 0.813901 - logloss: 0.438018

```
