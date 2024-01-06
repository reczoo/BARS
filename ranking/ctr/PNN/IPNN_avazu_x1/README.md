## IPNN_avazu_x1

A hands-on guide to run the PNN model on the Avazu_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

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
Dataset ID: [Avazu_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [PNN](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/PNN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [PNN_avazu_x1_tuner_config_02](./PNN_avazu_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd IPNN_avazu_x1
    nohup python run_expid.py --config ./PNN_avazu_x1_tuner_config_02 --expid PNN_avazu_x1_002_8a91559c --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.763014 | 0.367578  |


### Logs
```python
2020-12-25 07:52:44,333 P38784 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_83355fc7",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "embedding_regularizer": "5e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "PNN",
    "model_id": "PNN_avazu_x1_002_fe13aa74",
    "model_root": "./Avazu/PNN_avazu_x1/",
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
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2020-12-25 07:52:44,334 P38784 INFO Set up feature encoder...
2020-12-25 07:52:44,334 P38784 INFO Load feature_encoder from pickle: ../data/Avazu/avazu_x1_83355fc7/feature_encoder.pkl
2020-12-25 07:52:46,363 P38784 INFO Total number of parameters: 13487991.
2020-12-25 07:52:46,364 P38784 INFO Loading data...
2020-12-25 07:52:46,369 P38784 INFO Loading data from h5: ../data/Avazu/avazu_x1_83355fc7/train.h5
2020-12-25 07:52:49,227 P38784 INFO Loading data from h5: ../data/Avazu/avazu_x1_83355fc7/valid.h5
2020-12-25 07:52:49,603 P38784 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2020-12-25 07:52:49,603 P38784 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2020-12-25 07:52:49,603 P38784 INFO Loading train data done.
2020-12-25 07:52:52,464 P38784 INFO Start training: 6910 batches/epoch
2020-12-25 07:52:52,464 P38784 INFO ************ Epoch=1 start ************
2020-12-25 07:58:27,324 P38784 INFO [Metrics] AUC: 0.746309 - logloss: 0.397688
2020-12-25 07:58:27,327 P38784 INFO Save best model: monitor(max): 0.746309
2020-12-25 07:58:27,384 P38784 INFO --- 6910/6910 batches finished ---
2020-12-25 07:58:27,472 P38784 INFO Train loss: 0.409058
2020-12-25 07:58:27,472 P38784 INFO ************ Epoch=1 end ************
2020-12-25 08:04:01,310 P38784 INFO [Metrics] AUC: 0.746696 - logloss: 0.396429
2020-12-25 08:04:01,312 P38784 INFO Save best model: monitor(max): 0.746696
2020-12-25 08:04:01,395 P38784 INFO --- 6910/6910 batches finished ---
2020-12-25 08:04:01,456 P38784 INFO Train loss: 0.405722
2020-12-25 08:04:01,457 P38784 INFO ************ Epoch=2 end ************
2020-12-25 08:09:35,838 P38784 INFO [Metrics] AUC: 0.746959 - logloss: 0.395730
2020-12-25 08:09:35,841 P38784 INFO Save best model: monitor(max): 0.746959
2020-12-25 08:09:35,921 P38784 INFO --- 6910/6910 batches finished ---
2020-12-25 08:09:36,002 P38784 INFO Train loss: 0.405672
2020-12-25 08:09:36,002 P38784 INFO ************ Epoch=3 end ************
2020-12-25 08:15:11,017 P38784 INFO [Metrics] AUC: 0.744964 - logloss: 0.397028
2020-12-25 08:15:11,020 P38784 INFO Monitor(max) STOP: 0.744964 !
2020-12-25 08:15:11,020 P38784 INFO Reduce learning rate on plateau: 0.000100
2020-12-25 08:15:11,020 P38784 INFO --- 6910/6910 batches finished ---
2020-12-25 08:15:11,088 P38784 INFO Train loss: 0.408494
2020-12-25 08:15:11,088 P38784 INFO ************ Epoch=4 end ************
2020-12-25 08:20:45,712 P38784 INFO [Metrics] AUC: 0.738994 - logloss: 0.400536
2020-12-25 08:20:45,715 P38784 INFO Monitor(max) STOP: 0.738994 !
2020-12-25 08:20:45,715 P38784 INFO Reduce learning rate on plateau: 0.000010
2020-12-25 08:20:45,715 P38784 INFO Early stopping at epoch=5
2020-12-25 08:20:45,715 P38784 INFO --- 6910/6910 batches finished ---
2020-12-25 08:20:45,782 P38784 INFO Train loss: 0.390052
2020-12-25 08:20:45,783 P38784 INFO Training finished.
2020-12-25 08:20:45,783 P38784 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Avazu/PNN_avazu_x1/avazu_x1_83355fc7/PNN_avazu_x1_002_fe13aa74_model.ckpt
2020-12-25 08:20:45,871 P38784 INFO ****** Train/validation evaluation ******
2020-12-25 08:20:58,571 P38784 INFO [Metrics] AUC: 0.746959 - logloss: 0.395730
2020-12-25 08:20:58,658 P38784 INFO ******** Test evaluation ********
2020-12-25 08:20:58,658 P38784 INFO Loading data...
2020-12-25 08:20:58,659 P38784 INFO Loading data from h5: ../data/Avazu/avazu_x1_83355fc7/test.h5
2020-12-25 08:20:59,527 P38784 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2020-12-25 08:20:59,527 P38784 INFO Loading test data done.
2020-12-25 08:21:27,780 P38784 INFO [Metrics] AUC: 0.763014 - logloss: 0.367578

```
