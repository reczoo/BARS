## HOFM_avazu_x1

A hands-on guide to run the HOFM model on the Avazu_x1 dataset.

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
Dataset ID: [Avazu_x1](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [HOFM](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/HOFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HOFM_avazu_x1_tuner_config_02](./HOFM_avazu_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HOFM_avazu_x1
    nohup python run_expid.py --config ./HOFM_avazu_x1_tuner_config_02 --expid HOFM_avazu_x1_002_11ff3102 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.760137 | 0.368702  |


### Logs
```python
2022-01-26 10:57:38,258 P49084 INFO {
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "embedding_dim": "10",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "HOFM",
    "model_id": "HOFM_avazu_x1_002_11ff3102",
    "model_root": "./Avazu/HOFM_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "order": "3",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "5e-05",
    "reuse_embedding": "False",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-01-26 10:57:38,259 P49084 INFO Set up feature encoder...
2022-01-26 10:57:38,259 P49084 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-01-26 10:57:38,259 P49084 INFO Loading data...
2022-01-26 10:57:38,260 P49084 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-01-26 10:57:40,691 P49084 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-01-26 10:57:41,055 P49084 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-01-26 10:57:41,055 P49084 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-01-26 10:57:41,055 P49084 INFO Loading train data done.
2022-01-26 10:57:44,953 P49084 INFO Total number of parameters: 27270580.
2022-01-26 10:57:44,953 P49084 INFO Start training: 6910 batches/epoch
2022-01-26 10:57:44,953 P49084 INFO ************ Epoch=1 start ************
2022-01-26 11:08:52,604 P49084 INFO [Metrics] AUC: 0.737402 - logloss: 0.402381
2022-01-26 11:08:52,606 P49084 INFO Save best model: monitor(max): 0.737402
2022-01-26 11:08:52,890 P49084 INFO --- 6910/6910 batches finished ---
2022-01-26 11:08:52,923 P49084 INFO Train loss: 0.411907
2022-01-26 11:08:52,923 P49084 INFO ************ Epoch=1 end ************
2022-01-26 11:19:58,863 P49084 INFO [Metrics] AUC: 0.741667 - logloss: 0.400483
2022-01-26 11:19:58,865 P49084 INFO Save best model: monitor(max): 0.741667
2022-01-26 11:19:59,000 P49084 INFO --- 6910/6910 batches finished ---
2022-01-26 11:19:59,040 P49084 INFO Train loss: 0.408836
2022-01-26 11:19:59,040 P49084 INFO ************ Epoch=2 end ************
2022-01-26 11:31:05,301 P49084 INFO [Metrics] AUC: 0.741675 - logloss: 0.402180
2022-01-26 11:31:05,304 P49084 INFO Save best model: monitor(max): 0.741675
2022-01-26 11:31:05,448 P49084 INFO --- 6910/6910 batches finished ---
2022-01-26 11:31:05,483 P49084 INFO Train loss: 0.408053
2022-01-26 11:31:05,483 P49084 INFO ************ Epoch=3 end ************
2022-01-26 11:42:12,213 P49084 INFO [Metrics] AUC: 0.741443 - logloss: 0.400899
2022-01-26 11:42:12,216 P49084 INFO Monitor(max) STOP: 0.741443 !
2022-01-26 11:42:12,216 P49084 INFO Reduce learning rate on plateau: 0.000100
2022-01-26 11:42:12,216 P49084 INFO --- 6910/6910 batches finished ---
2022-01-26 11:42:12,266 P49084 INFO Train loss: 0.407547
2022-01-26 11:42:12,266 P49084 INFO ************ Epoch=4 end ************
2022-01-26 11:53:30,888 P49084 INFO [Metrics] AUC: 0.742307 - logloss: 0.399731
2022-01-26 11:53:30,891 P49084 INFO Save best model: monitor(max): 0.742307
2022-01-26 11:53:31,030 P49084 INFO --- 6910/6910 batches finished ---
2022-01-26 11:53:31,071 P49084 INFO Train loss: 0.399791
2022-01-26 11:53:31,071 P49084 INFO ************ Epoch=5 end ************
2022-01-26 12:04:50,854 P49084 INFO [Metrics] AUC: 0.740517 - logloss: 0.400566
2022-01-26 12:04:50,857 P49084 INFO Monitor(max) STOP: 0.740517 !
2022-01-26 12:04:50,857 P49084 INFO Reduce learning rate on plateau: 0.000010
2022-01-26 12:04:50,857 P49084 INFO --- 6910/6910 batches finished ---
2022-01-26 12:04:50,898 P49084 INFO Train loss: 0.398334
2022-01-26 12:04:50,899 P49084 INFO ************ Epoch=6 end ************
2022-01-26 12:16:10,077 P49084 INFO [Metrics] AUC: 0.740949 - logloss: 0.400192
2022-01-26 12:16:10,081 P49084 INFO Monitor(max) STOP: 0.740949 !
2022-01-26 12:16:10,081 P49084 INFO Reduce learning rate on plateau: 0.000001
2022-01-26 12:16:10,081 P49084 INFO Early stopping at epoch=7
2022-01-26 12:16:10,081 P49084 INFO --- 6910/6910 batches finished ---
2022-01-26 12:16:10,132 P49084 INFO Train loss: 0.396915
2022-01-26 12:16:10,132 P49084 INFO Training finished.
2022-01-26 12:16:10,132 P49084 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/HOFM_avazu_x1/avazu_x1_3fb65689/HOFM_avazu_x1_002_11ff3102.model
2022-01-26 12:16:13,193 P49084 INFO ****** Validation evaluation ******
2022-01-26 12:16:32,000 P49084 INFO [Metrics] AUC: 0.742307 - logloss: 0.399731
2022-01-26 12:16:32,071 P49084 INFO ******** Test evaluation ********
2022-01-26 12:16:32,071 P49084 INFO Loading data...
2022-01-26 12:16:32,071 P49084 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-01-26 12:16:32,900 P49084 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-01-26 12:16:32,901 P49084 INFO Loading test data done.
2022-01-26 12:17:12,224 P49084 INFO [Metrics] AUC: 0.760137 - logloss: 0.368702

```
