## AutoInt_criteo_x1

A hands-on guide to run the AutoInt model on the Criteo_x1 dataset.

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
  fuxictr: 1.1.0

  ```

### Dataset
Dataset ID: [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [AutoInt](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt_criteo_x1_tuner_config_02](./AutoInt_criteo_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AutoInt_criteo_x1
    nohup python run_expid.py --config ./AutoInt_criteo_x1_tuner_config_02 --expid AutoInt_criteo_x1_013_ccbbf1d8 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 24 runs:
| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.811981 | 0.439840  |
| 2 | 0.811905 | 0.439829  |
| 3 | 0.811579 | 0.440198  |
| 4 | 0.811557 | 0.440149  |
| 5 | 0.811455 | 0.440230  |
| 6 | 0.811243 | 0.440513  |
| 7 | 0.810873 | 0.440857  |
| 8 | 0.810578 | 0.441131  |
| 9 | 0.810506 | 0.441105  |
| 10 | 0.809768 | 0.441853  |
| 11 | 0.809650 | 0.441965  |
| 12 | 0.809357 | 0.442219  |
| 13 | 0.809302 | 0.442284  |
| 14 | 0.809241 | 0.442273  |
| 15 | 0.807748 | 0.443683  |
| 16 | 0.807171 | 0.444215  |
| 17 | 0.807063 | 0.444298  |
| 18 | 0.806792 | 0.444571  |
| 19 | 0.806303 | 0.445028  |
| 20 | 0.806103 | 0.445175  |
| 21 | 0.805557 | 0.445535  |
| 22 | 0.804361 | 0.446606  |
| 23 | 0.803948 | 0.446969  |
| 24 | 0.801775 | 0.449138  |
| | | | 
| Avg | 0.8085756666666666 | 0.4429026666666665 |
| Std | &#177;0.0027921850861924546 | &#177;0.0025262125251494993 |


### Logs
```python
2022-01-22 09:40:30,138 P846 INFO {
    "attention_dim": "128",
    "attention_layers": "6",
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "4",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AutoInt",
    "model_id": "AutoInt_criteo_x1_013_ccbbf1d8",
    "model_root": "./Criteo/AutoInt_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "2",
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
    "use_residual": "True",
    "use_scale": "False",
    "use_wide": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-22 09:40:30,139 P846 INFO Set up feature encoder...
2022-01-22 09:40:30,139 P846 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-01-22 09:40:30,139 P846 INFO Loading data...
2022-01-22 09:40:30,140 P846 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-01-22 09:40:35,397 P846 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-01-22 09:40:36,573 P846 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-01-22 09:40:36,573 P846 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-01-22 09:40:36,573 P846 INFO Loading train data done.
2022-01-22 09:40:45,239 P846 INFO Total number of parameters: 23952741.
2022-01-22 09:40:45,239 P846 INFO Start training: 8058 batches/epoch
2022-01-22 09:40:45,239 P846 INFO ************ Epoch=1 start ************
2022-01-22 10:35:28,992 P846 INFO [Metrics] AUC: 0.798473 - logloss: 0.452104
2022-01-22 10:35:28,994 P846 INFO Save best model: monitor(max): 0.798473
2022-01-22 10:35:29,308 P846 INFO --- 8058/8058 batches finished ---
2022-01-22 10:35:29,348 P846 INFO Train loss: 0.467688
2022-01-22 10:35:29,348 P846 INFO ************ Epoch=1 end ************
2022-01-22 11:30:08,201 P846 INFO [Metrics] AUC: 0.802019 - logloss: 0.449048
2022-01-22 11:30:08,203 P846 INFO Save best model: monitor(max): 0.802019
2022-01-22 11:30:08,311 P846 INFO --- 8058/8058 batches finished ---
2022-01-22 11:30:08,369 P846 INFO Train loss: 0.462339
2022-01-22 11:30:08,369 P846 INFO ************ Epoch=2 end ************
2022-01-22 12:24:47,038 P846 INFO [Metrics] AUC: 0.803972 - logloss: 0.447137
2022-01-22 12:24:47,040 P846 INFO Save best model: monitor(max): 0.803972
2022-01-22 12:24:47,154 P846 INFO --- 8058/8058 batches finished ---
2022-01-22 12:24:47,200 P846 INFO Train loss: 0.459982
2022-01-22 12:24:47,200 P846 INFO ************ Epoch=3 end ************
2022-01-22 13:19:27,421 P846 INFO [Metrics] AUC: 0.804585 - logloss: 0.446581
2022-01-22 13:19:27,422 P846 INFO Save best model: monitor(max): 0.804585
2022-01-22 13:19:27,534 P846 INFO --- 8058/8058 batches finished ---
2022-01-22 13:19:27,583 P846 INFO Train loss: 0.458843
2022-01-22 13:19:27,583 P846 INFO ************ Epoch=4 end ************
2022-01-22 14:14:07,823 P846 INFO [Metrics] AUC: 0.805493 - logloss: 0.445809
2022-01-22 14:14:07,825 P846 INFO Save best model: monitor(max): 0.805493
2022-01-22 14:14:07,946 P846 INFO --- 8058/8058 batches finished ---
2022-01-22 14:14:07,990 P846 INFO Train loss: 0.458360
2022-01-22 14:14:07,990 P846 INFO ************ Epoch=5 end ************
2022-01-22 15:08:46,567 P846 INFO [Metrics] AUC: 0.805906 - logloss: 0.445856
2022-01-22 15:08:46,569 P846 INFO Save best model: monitor(max): 0.805906
2022-01-22 15:08:46,674 P846 INFO --- 8058/8058 batches finished ---
2022-01-22 15:08:46,724 P846 INFO Train loss: 0.458048
2022-01-22 15:08:46,724 P846 INFO ************ Epoch=6 end ************
2022-01-22 16:03:25,779 P846 INFO [Metrics] AUC: 0.806013 - logloss: 0.445721
2022-01-22 16:03:25,780 P846 INFO Save best model: monitor(max): 0.806013
2022-01-22 16:03:25,899 P846 INFO --- 8058/8058 batches finished ---
2022-01-22 16:03:25,945 P846 INFO Train loss: 0.457921
2022-01-22 16:03:25,946 P846 INFO ************ Epoch=7 end ************
2022-01-22 16:58:02,932 P846 INFO [Metrics] AUC: 0.806363 - logloss: 0.444973
2022-01-22 16:58:02,934 P846 INFO Save best model: monitor(max): 0.806363
2022-01-22 16:58:03,049 P846 INFO --- 8058/8058 batches finished ---
2022-01-22 16:58:03,098 P846 INFO Train loss: 0.457892
2022-01-22 16:58:03,099 P846 INFO ************ Epoch=8 end ************
2022-01-22 17:52:42,480 P846 INFO [Metrics] AUC: 0.806632 - logloss: 0.444690
2022-01-22 17:52:42,482 P846 INFO Save best model: monitor(max): 0.806632
2022-01-22 17:52:42,591 P846 INFO --- 8058/8058 batches finished ---
2022-01-22 17:52:42,641 P846 INFO Train loss: 0.457761
2022-01-22 17:52:42,642 P846 INFO ************ Epoch=9 end ************
2022-01-22 18:47:22,252 P846 INFO [Metrics] AUC: 0.806644 - logloss: 0.444698
2022-01-22 18:47:22,254 P846 INFO Save best model: monitor(max): 0.806644
2022-01-22 18:47:22,367 P846 INFO --- 8058/8058 batches finished ---
2022-01-22 18:47:22,420 P846 INFO Train loss: 0.459089
2022-01-22 18:47:22,421 P846 INFO ************ Epoch=10 end ************
2022-01-22 19:42:01,284 P846 INFO [Metrics] AUC: 0.806733 - logloss: 0.444949
2022-01-22 19:42:01,286 P846 INFO Save best model: monitor(max): 0.806733
2022-01-22 19:42:01,389 P846 INFO --- 8058/8058 batches finished ---
2022-01-22 19:42:01,437 P846 INFO Train loss: 0.458006
2022-01-22 19:42:01,437 P846 INFO ************ Epoch=11 end ************
2022-01-22 20:36:39,089 P846 INFO [Metrics] AUC: 0.806890 - logloss: 0.444523
2022-01-22 20:36:39,090 P846 INFO Save best model: monitor(max): 0.806890
2022-01-22 20:36:39,228 P846 INFO --- 8058/8058 batches finished ---
2022-01-22 20:36:39,276 P846 INFO Train loss: 0.457696
2022-01-22 20:36:39,277 P846 INFO ************ Epoch=12 end ************
2022-01-22 21:31:17,450 P846 INFO [Metrics] AUC: 0.807093 - logloss: 0.444321
2022-01-22 21:31:17,452 P846 INFO Save best model: monitor(max): 0.807093
2022-01-22 21:31:17,555 P846 INFO --- 8058/8058 batches finished ---
2022-01-22 21:31:17,618 P846 INFO Train loss: 0.457722
2022-01-22 21:31:17,618 P846 INFO ************ Epoch=13 end ************
2022-01-22 22:25:55,936 P846 INFO [Metrics] AUC: 0.807269 - logloss: 0.444377
2022-01-22 22:25:55,938 P846 INFO Save best model: monitor(max): 0.807269
2022-01-22 22:25:56,043 P846 INFO --- 8058/8058 batches finished ---
2022-01-22 22:25:56,095 P846 INFO Train loss: 0.457857
2022-01-22 22:25:56,096 P846 INFO ************ Epoch=14 end ************
2022-01-22 23:20:34,820 P846 INFO [Metrics] AUC: 0.807412 - logloss: 0.444048
2022-01-22 23:20:34,822 P846 INFO Save best model: monitor(max): 0.807412
2022-01-22 23:20:34,932 P846 INFO --- 8058/8058 batches finished ---
2022-01-22 23:20:34,984 P846 INFO Train loss: 0.458057
2022-01-22 23:20:34,985 P846 INFO ************ Epoch=15 end ************
2022-01-23 00:15:14,073 P846 INFO [Metrics] AUC: 0.807526 - logloss: 0.444205
2022-01-23 00:15:14,075 P846 INFO Save best model: monitor(max): 0.807526
2022-01-23 00:15:14,179 P846 INFO --- 8058/8058 batches finished ---
2022-01-23 00:15:14,229 P846 INFO Train loss: 0.458265
2022-01-23 00:15:14,230 P846 INFO ************ Epoch=16 end ************
2022-01-23 01:09:50,402 P846 INFO [Metrics] AUC: 0.807278 - logloss: 0.444094
2022-01-23 01:09:50,403 P846 INFO Monitor(max) STOP: 0.807278 !
2022-01-23 01:09:50,403 P846 INFO Reduce learning rate on plateau: 0.000100
2022-01-23 01:09:50,403 P846 INFO --- 8058/8058 batches finished ---
2022-01-23 01:09:50,453 P846 INFO Train loss: 0.458332
2022-01-23 01:09:50,454 P846 INFO ************ Epoch=17 end ************
2022-01-23 02:24:54,747 P846 INFO [Metrics] AUC: 0.811468 - logloss: 0.440358
2022-01-23 02:24:54,749 P846 INFO Save best model: monitor(max): 0.811468
2022-01-23 02:24:54,865 P846 INFO --- 8058/8058 batches finished ---
2022-01-23 02:24:54,917 P846 INFO Train loss: 0.444361
2022-01-23 02:24:54,917 P846 INFO ************ Epoch=18 end ************
2022-01-23 04:12:57,256 P846 INFO [Metrics] AUC: 0.811719 - logloss: 0.440203
2022-01-23 04:12:57,258 P846 INFO Save best model: monitor(max): 0.811719
2022-01-23 04:12:57,374 P846 INFO --- 8058/8058 batches finished ---
2022-01-23 04:12:57,424 P846 INFO Train loss: 0.437278
2022-01-23 04:12:57,424 P846 INFO ************ Epoch=19 end ************
2022-01-23 06:00:49,514 P846 INFO [Metrics] AUC: 0.811415 - logloss: 0.440747
2022-01-23 06:00:49,516 P846 INFO Monitor(max) STOP: 0.811415 !
2022-01-23 06:00:49,516 P846 INFO Reduce learning rate on plateau: 0.000010
2022-01-23 06:00:49,516 P846 INFO --- 8058/8058 batches finished ---
2022-01-23 06:00:49,567 P846 INFO Train loss: 0.433806
2022-01-23 06:00:49,568 P846 INFO ************ Epoch=20 end ************
2022-01-23 07:48:09,205 P846 INFO [Metrics] AUC: 0.807058 - logloss: 0.447093
2022-01-23 07:48:09,206 P846 INFO Monitor(max) STOP: 0.807058 !
2022-01-23 07:48:09,206 P846 INFO Reduce learning rate on plateau: 0.000001
2022-01-23 07:48:09,206 P846 INFO Early stopping at epoch=21
2022-01-23 07:48:09,206 P846 INFO --- 8058/8058 batches finished ---
2022-01-23 07:48:09,255 P846 INFO Train loss: 0.422650
2022-01-23 07:48:09,255 P846 INFO Training finished.
2022-01-23 07:48:09,255 P846 INFO Load best model: /cache/FuxiCTR/benchmarks_modelarts/Criteo/AutoInt_criteo_x1/criteo_x1_7b681156/AutoInt_criteo_x1_013_ccbbf1d8.model
2022-01-23 07:48:15,956 P846 INFO ****** Validation evaluation ******
2022-01-23 07:51:50,119 P846 INFO [Metrics] AUC: 0.811719 - logloss: 0.440203
2022-01-23 07:51:50,206 P846 INFO ******** Test evaluation ********
2022-01-23 07:51:50,207 P846 INFO Loading data...
2022-01-23 07:51:50,207 P846 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-01-23 07:51:51,019 P846 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-01-23 07:51:51,019 P846 INFO Loading test data done.
2022-01-23 07:53:49,975 P846 INFO [Metrics] AUC: 0.811981 - logloss: 0.439840

```
