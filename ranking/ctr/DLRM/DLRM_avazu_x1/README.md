## DLRM_avazu_x1

A hands-on guide to run the DLRM model on the Avazu_x1 dataset.

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
  fuxictr: 1.2.1

  ```

### Dataset
Dataset ID: [Avazu_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu#Avazu_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.1](https://github.com/reczoo/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [DLRM](https://github.com/reczoo/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/DLRM.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DLRM_avazu_x1_tuner_config_01](./DLRM_avazu_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DLRM_avazu_x1
    nohup python run_expid.py --config ./DLRM_avazu_x1_tuner_config_01 --expid DLRM_avazu_x1_015_cf6fdabe --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.763934 | 0.366904  |


### Logs
```python
2022-05-27 19:47:35,352 P101383 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "bottom_mlp_activations": "ReLU",
    "bottom_mlp_dropout": "0.3",
    "bottom_mlp_units": "[400, 400, 400]",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "6",
    "interaction_op": "dot",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DLRM",
    "model_id": "DLRM_avazu_x1_015_cf6fdabe",
    "model_root": "./Avazu/DLRM_avazu_x1/",
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
    "top_mlp_activations": "ReLU",
    "top_mlp_dropout": "0.3",
    "top_mlp_units": "[400, 400, 400]",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-05-27 19:47:35,353 P101383 INFO Set up feature encoder...
2022-05-27 19:47:35,353 P101383 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-05-27 19:47:35,353 P101383 INFO Loading data...
2022-05-27 19:47:35,355 P101383 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-05-27 19:47:37,977 P101383 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-05-27 19:47:38,317 P101383 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-05-27 19:47:38,317 P101383 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-05-27 19:47:38,317 P101383 INFO Loading train data done.
2022-05-27 19:47:44,625 P101383 INFO Total number of parameters: 13402391.
2022-05-27 19:47:44,625 P101383 INFO Start training: 6910 batches/epoch
2022-05-27 19:47:44,625 P101383 INFO ************ Epoch=1 start ************
2022-05-27 19:58:31,628 P101383 INFO [Metrics] AUC: 0.731686 - logloss: 0.409691
2022-05-27 19:58:31,631 P101383 INFO Save best model: monitor(max): 0.731686
2022-05-27 19:58:31,853 P101383 INFO --- 6910/6910 batches finished ---
2022-05-27 19:58:31,894 P101383 INFO Train loss: 0.432078
2022-05-27 19:58:31,894 P101383 INFO ************ Epoch=1 end ************
2022-05-27 20:09:15,651 P101383 INFO [Metrics] AUC: 0.732220 - logloss: 0.409117
2022-05-27 20:09:15,653 P101383 INFO Save best model: monitor(max): 0.732220
2022-05-27 20:09:15,726 P101383 INFO --- 6910/6910 batches finished ---
2022-05-27 20:09:15,770 P101383 INFO Train loss: 0.429395
2022-05-27 20:09:15,770 P101383 INFO ************ Epoch=2 end ************
2022-05-27 20:20:01,152 P101383 INFO [Metrics] AUC: 0.733731 - logloss: 0.404299
2022-05-27 20:20:01,154 P101383 INFO Save best model: monitor(max): 0.733731
2022-05-27 20:20:01,233 P101383 INFO --- 6910/6910 batches finished ---
2022-05-27 20:20:01,280 P101383 INFO Train loss: 0.429940
2022-05-27 20:20:01,280 P101383 INFO ************ Epoch=3 end ************
2022-05-27 20:30:46,927 P101383 INFO [Metrics] AUC: 0.735620 - logloss: 0.406103
2022-05-27 20:30:46,930 P101383 INFO Save best model: monitor(max): 0.735620
2022-05-27 20:30:47,008 P101383 INFO --- 6910/6910 batches finished ---
2022-05-27 20:30:47,057 P101383 INFO Train loss: 0.430528
2022-05-27 20:30:47,057 P101383 INFO ************ Epoch=4 end ************
2022-05-27 20:41:32,465 P101383 INFO [Metrics] AUC: 0.733933 - logloss: 0.403524
2022-05-27 20:41:32,468 P101383 INFO Monitor(max) STOP: 0.733933 !
2022-05-27 20:41:32,468 P101383 INFO Reduce learning rate on plateau: 0.000100
2022-05-27 20:41:32,468 P101383 INFO --- 6910/6910 batches finished ---
2022-05-27 20:41:32,514 P101383 INFO Train loss: 0.430711
2022-05-27 20:41:32,515 P101383 INFO ************ Epoch=5 end ************
2022-05-27 20:52:16,107 P101383 INFO [Metrics] AUC: 0.745650 - logloss: 0.396405
2022-05-27 20:52:16,109 P101383 INFO Save best model: monitor(max): 0.745650
2022-05-27 20:52:16,176 P101383 INFO --- 6910/6910 batches finished ---
2022-05-27 20:52:16,216 P101383 INFO Train loss: 0.403793
2022-05-27 20:52:16,216 P101383 INFO ************ Epoch=6 end ************
2022-05-27 21:03:00,204 P101383 INFO [Metrics] AUC: 0.744514 - logloss: 0.396459
2022-05-27 21:03:00,206 P101383 INFO Monitor(max) STOP: 0.744514 !
2022-05-27 21:03:00,206 P101383 INFO Reduce learning rate on plateau: 0.000010
2022-05-27 21:03:00,206 P101383 INFO --- 6910/6910 batches finished ---
2022-05-27 21:03:00,246 P101383 INFO Train loss: 0.403799
2022-05-27 21:03:00,246 P101383 INFO ************ Epoch=7 end ************
2022-05-27 21:08:12,038 P101383 INFO [Metrics] AUC: 0.741671 - logloss: 0.398020
2022-05-27 21:08:12,040 P101383 INFO Monitor(max) STOP: 0.741671 !
2022-05-27 21:08:12,040 P101383 INFO Reduce learning rate on plateau: 0.000001
2022-05-27 21:08:12,041 P101383 INFO Early stopping at epoch=8
2022-05-27 21:08:12,041 P101383 INFO --- 6910/6910 batches finished ---
2022-05-27 21:08:12,077 P101383 INFO Train loss: 0.395465
2022-05-27 21:08:12,077 P101383 INFO Training finished.
2022-05-27 21:08:12,077 P101383 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/DLRM_avazu_x1/avazu_x1_3fb65689/DLRM_avazu_x1_015_cf6fdabe.model
2022-05-27 21:08:15,153 P101383 INFO ****** Validation evaluation ******
2022-05-27 21:08:27,304 P101383 INFO [Metrics] AUC: 0.745650 - logloss: 0.396405
2022-05-27 21:08:27,397 P101383 INFO ******** Test evaluation ********
2022-05-27 21:08:27,397 P101383 INFO Loading data...
2022-05-27 21:08:27,398 P101383 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-05-27 21:08:28,020 P101383 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-05-27 21:08:28,020 P101383 INFO Loading test data done.
2022-05-27 21:08:54,796 P101383 INFO [Metrics] AUC: 0.763934 - logloss: 0.366904

```
