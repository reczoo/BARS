## AFM_avazu_x1

A hands-on guide to run the AFM model on the Avazu_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [AFM](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/AFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFM_avazu_x1_tuner_config_03](./AFM_avazu_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFM_avazu_x1
    nohup python run_expid.py --config ./AFM_avazu_x1_tuner_config_03 --expid AFM_avazu_x1_002_4a58edb9 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.757359 | 0.370485  |


### Logs
```python
2022-01-26 17:59:32,046 P17548 INFO {
    "attention_dim": "16",
    "attention_dropout": "[0.2, 0.2]",
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AFM",
    "model_id": "AFM_avazu_x1_002_4a58edb9",
    "model_root": "./Avazu/AFM_avazu_x1/",
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
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_attention": "True",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-01-26 17:59:32,047 P17548 INFO Set up feature encoder...
2022-01-26 17:59:32,047 P17548 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-01-26 17:59:32,048 P17548 INFO Loading data...
2022-01-26 17:59:32,049 P17548 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-01-26 17:59:34,307 P17548 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-01-26 17:59:34,645 P17548 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-01-26 17:59:34,645 P17548 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-01-26 17:59:34,645 P17548 INFO Loading train data done.
2022-01-26 17:59:38,965 P17548 INFO Total number of parameters: 14284792.
2022-01-26 17:59:38,965 P17548 INFO Start training: 6910 batches/epoch
2022-01-26 17:59:38,965 P17548 INFO ************ Epoch=1 start ************
2022-01-26 18:14:48,208 P17548 INFO [Metrics] AUC: 0.736908 - logloss: 0.401249
2022-01-26 18:14:48,211 P17548 INFO Save best model: monitor(max): 0.736908
2022-01-26 18:14:48,442 P17548 INFO --- 6910/6910 batches finished ---
2022-01-26 18:14:48,485 P17548 INFO Train loss: 0.409624
2022-01-26 18:14:48,485 P17548 INFO ************ Epoch=1 end ************
2022-01-26 18:29:55,221 P17548 INFO [Metrics] AUC: 0.737648 - logloss: 0.401328
2022-01-26 18:29:55,222 P17548 INFO Save best model: monitor(max): 0.737648
2022-01-26 18:29:55,306 P17548 INFO --- 6910/6910 batches finished ---
2022-01-26 18:29:55,348 P17548 INFO Train loss: 0.405069
2022-01-26 18:29:55,348 P17548 INFO ************ Epoch=2 end ************
2022-01-26 18:45:02,535 P17548 INFO [Metrics] AUC: 0.737243 - logloss: 0.401850
2022-01-26 18:45:02,537 P17548 INFO Monitor(max) STOP: 0.737243 !
2022-01-26 18:45:02,538 P17548 INFO Reduce learning rate on plateau: 0.000100
2022-01-26 18:45:02,538 P17548 INFO --- 6910/6910 batches finished ---
2022-01-26 18:45:02,580 P17548 INFO Train loss: 0.404246
2022-01-26 18:45:02,580 P17548 INFO ************ Epoch=3 end ************
2022-01-26 19:00:10,246 P17548 INFO [Metrics] AUC: 0.738503 - logloss: 0.401095
2022-01-26 19:00:10,249 P17548 INFO Save best model: monitor(max): 0.738503
2022-01-26 19:00:10,326 P17548 INFO --- 6910/6910 batches finished ---
2022-01-26 19:00:10,369 P17548 INFO Train loss: 0.399396
2022-01-26 19:00:10,369 P17548 INFO ************ Epoch=4 end ************
2022-01-26 19:15:16,886 P17548 INFO [Metrics] AUC: 0.738449 - logloss: 0.401243
2022-01-26 19:15:16,888 P17548 INFO Monitor(max) STOP: 0.738449 !
2022-01-26 19:15:16,888 P17548 INFO Reduce learning rate on plateau: 0.000010
2022-01-26 19:15:16,888 P17548 INFO --- 6910/6910 batches finished ---
2022-01-26 19:15:16,933 P17548 INFO Train loss: 0.397390
2022-01-26 19:15:16,933 P17548 INFO ************ Epoch=5 end ************
2022-01-26 19:22:25,326 P17548 INFO [Metrics] AUC: 0.738388 - logloss: 0.401492
2022-01-26 19:22:25,329 P17548 INFO Monitor(max) STOP: 0.738388 !
2022-01-26 19:22:25,329 P17548 INFO Reduce learning rate on plateau: 0.000001
2022-01-26 19:22:25,329 P17548 INFO Early stopping at epoch=6
2022-01-26 19:22:25,329 P17548 INFO --- 6910/6910 batches finished ---
2022-01-26 19:22:25,373 P17548 INFO Train loss: 0.395477
2022-01-26 19:22:25,373 P17548 INFO Training finished.
2022-01-26 19:22:25,374 P17548 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/AFM_avazu_x1/avazu_x1_3fb65689/AFM_avazu_x1_002_4a58edb9.model
2022-01-26 19:22:28,154 P17548 INFO ****** Validation evaluation ******
2022-01-26 19:22:39,616 P17548 INFO [Metrics] AUC: 0.738503 - logloss: 0.401095
2022-01-26 19:22:39,681 P17548 INFO ******** Test evaluation ********
2022-01-26 19:22:39,682 P17548 INFO Loading data...
2022-01-26 19:22:39,682 P17548 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-01-26 19:22:40,320 P17548 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-01-26 19:22:40,320 P17548 INFO Loading test data done.
2022-01-26 19:23:06,356 P17548 INFO [Metrics] AUC: 0.757359 - logloss: 0.370485

```
