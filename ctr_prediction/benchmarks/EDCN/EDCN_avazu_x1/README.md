## EDCN_avazu_x1

A hands-on guide to run the EDCN model on the Avazu_x1 dataset.

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
Dataset ID: [Avazu_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu#Avazu_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [EDCN](https://github.com/xue-pai/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/EDCN.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [EDCN_avazu_x1_tuner_config_03](./EDCN_avazu_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd EDCN_avazu_x1
    nohup python run_expid.py --config ./EDCN_avazu_x1_tuner_config_03 --expid EDCN_avazu_x1_011_ba8f8c41 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.764742 | 0.366979  |


### Logs
```python
2022-05-27 23:26:25,688 P89283 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "bridge_type": "hadamard_product",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.1",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "2",
    "hidden_activations": "ReLU",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "EDCN",
    "model_id": "EDCN_avazu_x1_011_ba8f8c41",
    "model_root": "./Avazu/EDCN_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_cross_layers": "3",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "temperature": "1",
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_hdf5": "True",
    "use_regulation_module": "False",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-05-27 23:26:25,689 P89283 INFO Set up feature encoder...
2022-05-27 23:26:25,689 P89283 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-05-27 23:26:25,689 P89283 INFO Loading data...
2022-05-27 23:26:25,690 P89283 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-05-27 23:26:28,160 P89283 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-05-27 23:26:28,527 P89283 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-05-27 23:26:28,527 P89283 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-05-27 23:26:28,527 P89283 INFO Loading train data done.
2022-05-27 23:26:35,212 P89283 INFO Total number of parameters: 13136471.
2022-05-27 23:26:35,213 P89283 INFO Start training: 6910 batches/epoch
2022-05-27 23:26:35,213 P89283 INFO ************ Epoch=1 start ************
2022-05-27 23:39:03,599 P89283 INFO [Metrics] AUC: 0.735519 - logloss: 0.402374
2022-05-27 23:39:03,601 P89283 INFO Save best model: monitor(max): 0.735519
2022-05-27 23:39:03,918 P89283 INFO --- 6910/6910 batches finished ---
2022-05-27 23:39:03,961 P89283 INFO Train loss: 0.455109
2022-05-27 23:39:03,961 P89283 INFO ************ Epoch=1 end ************
2022-05-27 23:51:31,906 P89283 INFO [Metrics] AUC: 0.738184 - logloss: 0.401970
2022-05-27 23:51:31,909 P89283 INFO Save best model: monitor(max): 0.738184
2022-05-27 23:51:31,983 P89283 INFO --- 6910/6910 batches finished ---
2022-05-27 23:51:32,029 P89283 INFO Train loss: 0.447705
2022-05-27 23:51:32,029 P89283 INFO ************ Epoch=2 end ************
2022-05-28 00:03:58,572 P89283 INFO [Metrics] AUC: 0.731291 - logloss: 0.404292
2022-05-28 00:03:58,575 P89283 INFO Monitor(max) STOP: 0.731291 !
2022-05-28 00:03:58,575 P89283 INFO Reduce learning rate on plateau: 0.000100
2022-05-28 00:03:58,575 P89283 INFO --- 6910/6910 batches finished ---
2022-05-28 00:03:58,621 P89283 INFO Train loss: 0.446227
2022-05-28 00:03:58,621 P89283 INFO ************ Epoch=3 end ************
2022-05-28 00:16:23,697 P89283 INFO [Metrics] AUC: 0.745270 - logloss: 0.397153
2022-05-28 00:16:23,701 P89283 INFO Save best model: monitor(max): 0.745270
2022-05-28 00:16:23,767 P89283 INFO --- 6910/6910 batches finished ---
2022-05-28 00:16:23,810 P89283 INFO Train loss: 0.412209
2022-05-28 00:16:23,810 P89283 INFO ************ Epoch=4 end ************
2022-05-28 00:28:50,079 P89283 INFO [Metrics] AUC: 0.745660 - logloss: 0.395973
2022-05-28 00:28:50,083 P89283 INFO Save best model: monitor(max): 0.745660
2022-05-28 00:28:50,157 P89283 INFO --- 6910/6910 batches finished ---
2022-05-28 00:28:50,203 P89283 INFO Train loss: 0.415270
2022-05-28 00:28:50,203 P89283 INFO ************ Epoch=5 end ************
2022-05-28 00:41:17,537 P89283 INFO [Metrics] AUC: 0.746937 - logloss: 0.395994
2022-05-28 00:41:17,539 P89283 INFO Save best model: monitor(max): 0.746937
2022-05-28 00:41:17,617 P89283 INFO --- 6910/6910 batches finished ---
2022-05-28 00:41:17,660 P89283 INFO Train loss: 0.416393
2022-05-28 00:41:17,660 P89283 INFO ************ Epoch=6 end ************
2022-05-28 00:53:43,840 P89283 INFO [Metrics] AUC: 0.747215 - logloss: 0.395421
2022-05-28 00:53:43,843 P89283 INFO Save best model: monitor(max): 0.747215
2022-05-28 00:53:43,923 P89283 INFO --- 6910/6910 batches finished ---
2022-05-28 00:53:43,966 P89283 INFO Train loss: 0.416898
2022-05-28 00:53:43,966 P89283 INFO ************ Epoch=7 end ************
2022-05-28 01:06:09,717 P89283 INFO [Metrics] AUC: 0.744703 - logloss: 0.397172
2022-05-28 01:06:09,720 P89283 INFO Monitor(max) STOP: 0.744703 !
2022-05-28 01:06:09,720 P89283 INFO Reduce learning rate on plateau: 0.000010
2022-05-28 01:06:09,720 P89283 INFO --- 6910/6910 batches finished ---
2022-05-28 01:06:09,768 P89283 INFO Train loss: 0.417321
2022-05-28 01:06:09,768 P89283 INFO ************ Epoch=8 end ************
2022-05-28 01:17:26,741 P89283 INFO [Metrics] AUC: 0.748199 - logloss: 0.395193
2022-05-28 01:17:26,744 P89283 INFO Save best model: monitor(max): 0.748199
2022-05-28 01:17:26,809 P89283 INFO --- 6910/6910 batches finished ---
2022-05-28 01:17:26,849 P89283 INFO Train loss: 0.399089
2022-05-28 01:17:26,849 P89283 INFO ************ Epoch=9 end ************
2022-05-28 01:29:30,942 P89283 INFO [Metrics] AUC: 0.746565 - logloss: 0.395911
2022-05-28 01:29:30,946 P89283 INFO Monitor(max) STOP: 0.746565 !
2022-05-28 01:29:30,946 P89283 INFO Reduce learning rate on plateau: 0.000001
2022-05-28 01:29:30,947 P89283 INFO --- 6910/6910 batches finished ---
2022-05-28 01:29:30,996 P89283 INFO Train loss: 0.396774
2022-05-28 01:29:30,996 P89283 INFO ************ Epoch=10 end ************
2022-05-28 01:41:38,928 P89283 INFO [Metrics] AUC: 0.745017 - logloss: 0.396750
2022-05-28 01:41:38,933 P89283 INFO Monitor(max) STOP: 0.745017 !
2022-05-28 01:41:38,933 P89283 INFO Reduce learning rate on plateau: 0.000001
2022-05-28 01:41:38,933 P89283 INFO Early stopping at epoch=11
2022-05-28 01:41:38,933 P89283 INFO --- 6910/6910 batches finished ---
2022-05-28 01:41:38,978 P89283 INFO Train loss: 0.390812
2022-05-28 01:41:38,978 P89283 INFO Training finished.
2022-05-28 01:41:38,978 P89283 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/EDCN_avazu_x1/avazu_x1_3fb65689/EDCN_avazu_x1_011_ba8f8c41.model
2022-05-28 01:41:43,922 P89283 INFO ****** Validation evaluation ******
2022-05-28 01:41:56,691 P89283 INFO [Metrics] AUC: 0.748199 - logloss: 0.395193
2022-05-28 01:41:56,770 P89283 INFO ******** Test evaluation ********
2022-05-28 01:41:56,770 P89283 INFO Loading data...
2022-05-28 01:41:56,770 P89283 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-05-28 01:41:57,537 P89283 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-05-28 01:41:57,537 P89283 INFO Loading test data done.
2022-05-28 01:42:26,708 P89283 INFO [Metrics] AUC: 0.764742 - logloss: 0.366979

```
