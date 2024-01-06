## DLRM_criteo_x1

A hands-on guide to run the DLRM model on the Criteo_x1 dataset.

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
  fuxictr: 1.2.1

  ```

### Dataset
Dataset ID: [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo#Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [DLRM](https://github.com/xue-pai/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/DLRM.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DLRM_criteo_x1_tuner_config_02](./DLRM_criteo_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DLRM_criteo_x1
    nohup python run_expid.py --config ./DLRM_criteo_x1_tuner_config_02 --expid DLRM_criteo_x1_001_4d897285 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.813804 | 0.438155  |


### Logs
```python
2022-05-29 15:05:50,295 P7851 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "bottom_mlp_activations": "ReLU",
    "bottom_mlp_dropout": "0",
    "bottom_mlp_units": "[400, 400, 400]",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "0",
    "interaction_op": "dot",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DLRM",
    "model_id": "DLRM_criteo_x1_001_4d897285",
    "model_root": "./Criteo/DLRM_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "top_mlp_activations": "ReLU",
    "top_mlp_dropout": "0.2",
    "top_mlp_units": "[400, 400, 400]",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-05-29 15:05:50,295 P7851 INFO Set up feature encoder...
2022-05-29 15:05:50,295 P7851 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-05-29 15:05:50,296 P7851 INFO Loading data...
2022-05-29 15:05:50,297 P7851 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-05-29 15:05:55,450 P7851 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-05-29 15:05:56,805 P7851 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-05-29 15:05:56,805 P7851 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-05-29 15:05:56,805 P7851 INFO Loading train data done.
2022-05-29 15:06:02,843 P7851 INFO Total number of parameters: 21664241.
2022-05-29 15:06:02,843 P7851 INFO Start training: 8058 batches/epoch
2022-05-29 15:06:02,844 P7851 INFO ************ Epoch=1 start ************
2022-05-29 15:21:26,950 P7851 INFO [Metrics] AUC: 0.804187 - logloss: 0.447040
2022-05-29 15:21:26,952 P7851 INFO Save best model: monitor(max): 0.804187
2022-05-29 15:21:27,059 P7851 INFO --- 8058/8058 batches finished ---
2022-05-29 15:21:27,104 P7851 INFO Train loss: 0.461660
2022-05-29 15:21:27,104 P7851 INFO ************ Epoch=1 end ************
2022-05-29 15:36:49,337 P7851 INFO [Metrics] AUC: 0.806524 - logloss: 0.445125
2022-05-29 15:36:49,339 P7851 INFO Save best model: monitor(max): 0.806524
2022-05-29 15:36:49,463 P7851 INFO --- 8058/8058 batches finished ---
2022-05-29 15:36:49,502 P7851 INFO Train loss: 0.456277
2022-05-29 15:36:49,503 P7851 INFO ************ Epoch=2 end ************
2022-05-29 15:52:12,386 P7851 INFO [Metrics] AUC: 0.807615 - logloss: 0.443908
2022-05-29 15:52:12,387 P7851 INFO Save best model: monitor(max): 0.807615
2022-05-29 15:52:12,513 P7851 INFO --- 8058/8058 batches finished ---
2022-05-29 15:52:12,573 P7851 INFO Train loss: 0.454813
2022-05-29 15:52:12,573 P7851 INFO ************ Epoch=3 end ************
2022-05-29 16:07:31,604 P7851 INFO [Metrics] AUC: 0.807956 - logloss: 0.443703
2022-05-29 16:07:31,605 P7851 INFO Save best model: monitor(max): 0.807956
2022-05-29 16:07:31,731 P7851 INFO --- 8058/8058 batches finished ---
2022-05-29 16:07:31,779 P7851 INFO Train loss: 0.454118
2022-05-29 16:07:31,779 P7851 INFO ************ Epoch=4 end ************
2022-05-29 16:22:48,793 P7851 INFO [Metrics] AUC: 0.808651 - logloss: 0.442900
2022-05-29 16:22:48,794 P7851 INFO Save best model: monitor(max): 0.808651
2022-05-29 16:22:48,932 P7851 INFO --- 8058/8058 batches finished ---
2022-05-29 16:22:48,978 P7851 INFO Train loss: 0.453736
2022-05-29 16:22:48,978 P7851 INFO ************ Epoch=5 end ************
2022-05-29 16:38:05,989 P7851 INFO [Metrics] AUC: 0.808663 - logloss: 0.442960
2022-05-29 16:38:05,990 P7851 INFO Save best model: monitor(max): 0.808663
2022-05-29 16:38:06,115 P7851 INFO --- 8058/8058 batches finished ---
2022-05-29 16:38:06,159 P7851 INFO Train loss: 0.453420
2022-05-29 16:38:06,160 P7851 INFO ************ Epoch=6 end ************
2022-05-29 16:53:22,282 P7851 INFO [Metrics] AUC: 0.809122 - logloss: 0.442572
2022-05-29 16:53:22,284 P7851 INFO Save best model: monitor(max): 0.809122
2022-05-29 16:53:22,386 P7851 INFO --- 8058/8058 batches finished ---
2022-05-29 16:53:22,425 P7851 INFO Train loss: 0.453186
2022-05-29 16:53:22,425 P7851 INFO ************ Epoch=7 end ************
2022-05-29 17:08:38,105 P7851 INFO [Metrics] AUC: 0.809201 - logloss: 0.442405
2022-05-29 17:08:38,106 P7851 INFO Save best model: monitor(max): 0.809201
2022-05-29 17:08:38,207 P7851 INFO --- 8058/8058 batches finished ---
2022-05-29 17:08:38,245 P7851 INFO Train loss: 0.453003
2022-05-29 17:08:38,246 P7851 INFO ************ Epoch=8 end ************
2022-05-29 17:23:54,309 P7851 INFO [Metrics] AUC: 0.809330 - logloss: 0.442346
2022-05-29 17:23:54,310 P7851 INFO Save best model: monitor(max): 0.809330
2022-05-29 17:23:54,410 P7851 INFO --- 8058/8058 batches finished ---
2022-05-29 17:23:54,448 P7851 INFO Train loss: 0.452831
2022-05-29 17:23:54,448 P7851 INFO ************ Epoch=9 end ************
2022-05-29 17:39:10,920 P7851 INFO [Metrics] AUC: 0.809664 - logloss: 0.441991
2022-05-29 17:39:10,922 P7851 INFO Save best model: monitor(max): 0.809664
2022-05-29 17:39:11,024 P7851 INFO --- 8058/8058 batches finished ---
2022-05-29 17:39:11,062 P7851 INFO Train loss: 0.452682
2022-05-29 17:39:11,062 P7851 INFO ************ Epoch=10 end ************
2022-05-29 17:54:24,245 P7851 INFO [Metrics] AUC: 0.809675 - logloss: 0.441967
2022-05-29 17:54:24,246 P7851 INFO Save best model: monitor(max): 0.809675
2022-05-29 17:54:24,355 P7851 INFO --- 8058/8058 batches finished ---
2022-05-29 17:54:24,394 P7851 INFO Train loss: 0.452588
2022-05-29 17:54:24,394 P7851 INFO ************ Epoch=11 end ************
2022-05-29 18:09:40,265 P7851 INFO [Metrics] AUC: 0.809607 - logloss: 0.442167
2022-05-29 18:09:40,266 P7851 INFO Monitor(max) STOP: 0.809607 !
2022-05-29 18:09:40,267 P7851 INFO Reduce learning rate on plateau: 0.000100
2022-05-29 18:09:40,267 P7851 INFO --- 8058/8058 batches finished ---
2022-05-29 18:09:40,307 P7851 INFO Train loss: 0.452451
2022-05-29 18:09:40,307 P7851 INFO ************ Epoch=12 end ************
2022-05-29 18:17:07,207 P7851 INFO [Metrics] AUC: 0.812970 - logloss: 0.438972
2022-05-29 18:17:07,208 P7851 INFO Save best model: monitor(max): 0.812970
2022-05-29 18:17:07,302 P7851 INFO --- 8058/8058 batches finished ---
2022-05-29 18:17:07,340 P7851 INFO Train loss: 0.441606
2022-05-29 18:17:07,341 P7851 INFO ************ Epoch=13 end ************
2022-05-29 18:24:30,663 P7851 INFO [Metrics] AUC: 0.813400 - logloss: 0.438582
2022-05-29 18:24:30,665 P7851 INFO Save best model: monitor(max): 0.813400
2022-05-29 18:24:30,772 P7851 INFO --- 8058/8058 batches finished ---
2022-05-29 18:24:30,823 P7851 INFO Train loss: 0.437457
2022-05-29 18:24:30,823 P7851 INFO ************ Epoch=14 end ************
2022-05-29 18:31:50,965 P7851 INFO [Metrics] AUC: 0.813497 - logloss: 0.438596
2022-05-29 18:31:50,966 P7851 INFO Save best model: monitor(max): 0.813497
2022-05-29 18:31:51,061 P7851 INFO --- 8058/8058 batches finished ---
2022-05-29 18:31:51,101 P7851 INFO Train loss: 0.435468
2022-05-29 18:31:51,101 P7851 INFO ************ Epoch=15 end ************
2022-05-29 18:39:13,073 P7851 INFO [Metrics] AUC: 0.813298 - logloss: 0.438926
2022-05-29 18:39:13,075 P7851 INFO Monitor(max) STOP: 0.813298 !
2022-05-29 18:39:13,075 P7851 INFO Reduce learning rate on plateau: 0.000010
2022-05-29 18:39:13,075 P7851 INFO --- 8058/8058 batches finished ---
2022-05-29 18:39:13,115 P7851 INFO Train loss: 0.433854
2022-05-29 18:39:13,115 P7851 INFO ************ Epoch=16 end ************
2022-05-29 18:46:31,049 P7851 INFO [Metrics] AUC: 0.812458 - logloss: 0.440054
2022-05-29 18:46:31,051 P7851 INFO Monitor(max) STOP: 0.812458 !
2022-05-29 18:46:31,051 P7851 INFO Reduce learning rate on plateau: 0.000001
2022-05-29 18:46:31,051 P7851 INFO Early stopping at epoch=17
2022-05-29 18:46:31,051 P7851 INFO --- 8058/8058 batches finished ---
2022-05-29 18:46:31,090 P7851 INFO Train loss: 0.428564
2022-05-29 18:46:31,090 P7851 INFO Training finished.
2022-05-29 18:46:31,090 P7851 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/DLRM_criteo_x1/criteo_x1_7b681156/DLRM_criteo_x1_001_4d897285.model
2022-05-29 18:46:31,168 P7851 INFO ****** Validation evaluation ******
2022-05-29 18:46:55,691 P7851 INFO [Metrics] AUC: 0.813497 - logloss: 0.438596
2022-05-29 18:46:55,768 P7851 INFO ******** Test evaluation ********
2022-05-29 18:46:55,768 P7851 INFO Loading data...
2022-05-29 18:46:55,768 P7851 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-05-29 18:46:56,559 P7851 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-05-29 18:46:56,559 P7851 INFO Loading test data done.
2022-05-29 18:47:11,210 P7851 INFO [Metrics] AUC: 0.813804 - logloss: 0.438155

```
