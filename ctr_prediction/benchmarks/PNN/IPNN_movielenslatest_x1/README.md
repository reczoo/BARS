## IPNN_movielenslatest_x1

A hands-on guide to run the PNN model on the Movielenslatest_x1 dataset.

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
Dataset ID: [Movielenslatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/MovieLens/README.md#Movielenslatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [PNN](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/PNN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [PNN_movielenslatest_x1_tuner_config_02](./PNN_movielenslatest_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd IPNN_movielenslatest_x1
    nohup python run_expid.py --config ./PNN_movielenslatest_x1_tuner_config_02 --expid PNN_movielenslatest_x1_026_3f8280ec --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.969885 | 0.209455  |


### Logs
```python
2020-12-24 18:50:43,326 P17439 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/MovielensLatest/",
    "dataset_id": "movielenslatest_x1_bcd26aed",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "PNN",
    "model_id": "PNN_movielenslatest_x1_026_8a4a0ebb",
    "model_root": "./MovielensLatest/PNN_movielenslatest_x1/",
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
    "test_data": "../data/MovielensLatest/MovielensLatest_x1/test.csv",
    "train_data": "../data/MovielensLatest/MovielensLatest_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/MovielensLatest/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2020-12-24 18:50:43,327 P17439 INFO Set up feature encoder...
2020-12-24 18:50:43,327 P17439 INFO Load feature_encoder from pickle: ../data/MovielensLatest/movielenslatest_x1_bcd26aed/feature_encoder.pkl
2020-12-24 18:50:43,511 P17439 INFO Total number of parameters: 1237191.
2020-12-24 18:50:43,511 P17439 INFO Loading data...
2020-12-24 18:50:43,513 P17439 INFO Loading data from h5: ../data/MovielensLatest/movielenslatest_x1_bcd26aed/train.h5
2020-12-24 18:50:43,542 P17439 INFO Loading data from h5: ../data/MovielensLatest/movielenslatest_x1_bcd26aed/valid.h5
2020-12-24 18:50:43,551 P17439 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2020-12-24 18:50:43,551 P17439 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2020-12-24 18:50:43,551 P17439 INFO Loading train data done.
2020-12-24 18:50:46,549 P17439 INFO Start training: 343 batches/epoch
2020-12-24 18:50:46,549 P17439 INFO ************ Epoch=1 start ************
2020-12-24 18:50:55,807 P17439 INFO [Metrics] AUC: 0.925541 - logloss: 0.308871
2020-12-24 18:50:55,808 P17439 INFO Save best model: monitor(max): 0.925541
2020-12-24 18:50:55,818 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:50:55,871 P17439 INFO Train loss: 0.408390
2020-12-24 18:50:55,871 P17439 INFO ************ Epoch=1 end ************
2020-12-24 18:51:04,859 P17439 INFO [Metrics] AUC: 0.933338 - logloss: 0.292922
2020-12-24 18:51:04,860 P17439 INFO Save best model: monitor(max): 0.933338
2020-12-24 18:51:04,868 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:51:04,921 P17439 INFO Train loss: 0.361296
2020-12-24 18:51:04,921 P17439 INFO ************ Epoch=2 end ************
2020-12-24 18:51:13,707 P17439 INFO [Metrics] AUC: 0.939390 - logloss: 0.278177
2020-12-24 18:51:13,708 P17439 INFO Save best model: monitor(max): 0.939390
2020-12-24 18:51:13,716 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:51:13,773 P17439 INFO Train loss: 0.351441
2020-12-24 18:51:13,773 P17439 INFO ************ Epoch=3 end ************
2020-12-24 18:51:22,260 P17439 INFO [Metrics] AUC: 0.942252 - logloss: 0.271119
2020-12-24 18:51:22,260 P17439 INFO Save best model: monitor(max): 0.942252
2020-12-24 18:51:22,270 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:51:22,324 P17439 INFO Train loss: 0.351835
2020-12-24 18:51:22,324 P17439 INFO ************ Epoch=4 end ************
2020-12-24 18:51:30,480 P17439 INFO [Metrics] AUC: 0.944522 - logloss: 0.264827
2020-12-24 18:51:30,481 P17439 INFO Save best model: monitor(max): 0.944522
2020-12-24 18:51:30,489 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:51:30,546 P17439 INFO Train loss: 0.356088
2020-12-24 18:51:30,546 P17439 INFO ************ Epoch=5 end ************
2020-12-24 18:51:38,893 P17439 INFO [Metrics] AUC: 0.946338 - logloss: 0.260306
2020-12-24 18:51:38,894 P17439 INFO Save best model: monitor(max): 0.946338
2020-12-24 18:51:38,903 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:51:38,960 P17439 INFO Train loss: 0.360547
2020-12-24 18:51:38,960 P17439 INFO ************ Epoch=6 end ************
2020-12-24 18:51:47,495 P17439 INFO [Metrics] AUC: 0.948253 - logloss: 0.254422
2020-12-24 18:51:47,496 P17439 INFO Save best model: monitor(max): 0.948253
2020-12-24 18:51:47,503 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:51:47,559 P17439 INFO Train loss: 0.362527
2020-12-24 18:51:47,559 P17439 INFO ************ Epoch=7 end ************
2020-12-24 18:51:55,760 P17439 INFO [Metrics] AUC: 0.950380 - logloss: 0.249973
2020-12-24 18:51:55,761 P17439 INFO Save best model: monitor(max): 0.950380
2020-12-24 18:51:55,769 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:51:55,825 P17439 INFO Train loss: 0.364945
2020-12-24 18:51:55,825 P17439 INFO ************ Epoch=8 end ************
2020-12-24 18:52:03,796 P17439 INFO [Metrics] AUC: 0.951209 - logloss: 0.246787
2020-12-24 18:52:03,797 P17439 INFO Save best model: monitor(max): 0.951209
2020-12-24 18:52:03,805 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:52:03,864 P17439 INFO Train loss: 0.365556
2020-12-24 18:52:03,865 P17439 INFO ************ Epoch=9 end ************
2020-12-24 18:52:12,390 P17439 INFO [Metrics] AUC: 0.952677 - logloss: 0.242904
2020-12-24 18:52:12,390 P17439 INFO Save best model: monitor(max): 0.952677
2020-12-24 18:52:12,398 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:52:12,450 P17439 INFO Train loss: 0.367840
2020-12-24 18:52:12,451 P17439 INFO ************ Epoch=10 end ************
2020-12-24 18:52:20,803 P17439 INFO [Metrics] AUC: 0.953329 - logloss: 0.240873
2020-12-24 18:52:20,804 P17439 INFO Save best model: monitor(max): 0.953329
2020-12-24 18:52:20,811 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:52:20,875 P17439 INFO Train loss: 0.368924
2020-12-24 18:52:20,875 P17439 INFO ************ Epoch=11 end ************
2020-12-24 18:52:29,500 P17439 INFO [Metrics] AUC: 0.953981 - logloss: 0.239154
2020-12-24 18:52:29,501 P17439 INFO Save best model: monitor(max): 0.953981
2020-12-24 18:52:29,509 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:52:29,572 P17439 INFO Train loss: 0.369650
2020-12-24 18:52:29,573 P17439 INFO ************ Epoch=12 end ************
2020-12-24 18:52:38,446 P17439 INFO [Metrics] AUC: 0.954732 - logloss: 0.236891
2020-12-24 18:52:38,447 P17439 INFO Save best model: monitor(max): 0.954732
2020-12-24 18:52:38,455 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:52:38,532 P17439 INFO Train loss: 0.369080
2020-12-24 18:52:38,533 P17439 INFO ************ Epoch=13 end ************
2020-12-24 18:52:47,545 P17439 INFO [Metrics] AUC: 0.955165 - logloss: 0.235590
2020-12-24 18:52:47,546 P17439 INFO Save best model: monitor(max): 0.955165
2020-12-24 18:52:47,554 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:52:47,619 P17439 INFO Train loss: 0.371437
2020-12-24 18:52:47,619 P17439 INFO ************ Epoch=14 end ************
2020-12-24 18:52:56,597 P17439 INFO [Metrics] AUC: 0.955646 - logloss: 0.234222
2020-12-24 18:52:56,597 P17439 INFO Save best model: monitor(max): 0.955646
2020-12-24 18:52:56,605 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:52:56,667 P17439 INFO Train loss: 0.371274
2020-12-24 18:52:56,667 P17439 INFO ************ Epoch=15 end ************
2020-12-24 18:53:03,012 P17439 INFO [Metrics] AUC: 0.955688 - logloss: 0.234531
2020-12-24 18:53:03,013 P17439 INFO Save best model: monitor(max): 0.955688
2020-12-24 18:53:03,021 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:53:03,084 P17439 INFO Train loss: 0.371639
2020-12-24 18:53:03,084 P17439 INFO ************ Epoch=16 end ************
2020-12-24 18:53:09,366 P17439 INFO [Metrics] AUC: 0.957040 - logloss: 0.230534
2020-12-24 18:53:09,367 P17439 INFO Save best model: monitor(max): 0.957040
2020-12-24 18:53:09,375 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:53:09,430 P17439 INFO Train loss: 0.372382
2020-12-24 18:53:09,431 P17439 INFO ************ Epoch=17 end ************
2020-12-24 18:53:18,159 P17439 INFO [Metrics] AUC: 0.957153 - logloss: 0.230131
2020-12-24 18:53:18,160 P17439 INFO Save best model: monitor(max): 0.957153
2020-12-24 18:53:18,168 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:53:18,224 P17439 INFO Train loss: 0.373655
2020-12-24 18:53:18,224 P17439 INFO ************ Epoch=18 end ************
2020-12-24 18:53:26,005 P17439 INFO [Metrics] AUC: 0.956912 - logloss: 0.231891
2020-12-24 18:53:26,006 P17439 INFO Monitor(max) STOP: 0.956912 !
2020-12-24 18:53:26,006 P17439 INFO Reduce learning rate on plateau: 0.000100
2020-12-24 18:53:26,006 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:53:26,063 P17439 INFO Train loss: 0.373296
2020-12-24 18:53:26,063 P17439 INFO ************ Epoch=19 end ************
2020-12-24 18:53:34,391 P17439 INFO [Metrics] AUC: 0.967911 - logloss: 0.203577
2020-12-24 18:53:34,392 P17439 INFO Save best model: monitor(max): 0.967911
2020-12-24 18:53:34,400 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:53:34,468 P17439 INFO Train loss: 0.273731
2020-12-24 18:53:34,468 P17439 INFO ************ Epoch=20 end ************
2020-12-24 18:53:42,971 P17439 INFO [Metrics] AUC: 0.969699 - logloss: 0.210082
2020-12-24 18:53:42,971 P17439 INFO Save best model: monitor(max): 0.969699
2020-12-24 18:53:42,979 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:53:43,057 P17439 INFO Train loss: 0.185410
2020-12-24 18:53:43,057 P17439 INFO ************ Epoch=21 end ************
2020-12-24 18:53:51,022 P17439 INFO [Metrics] AUC: 0.969397 - logloss: 0.230294
2020-12-24 18:53:51,022 P17439 INFO Monitor(max) STOP: 0.969397 !
2020-12-24 18:53:51,023 P17439 INFO Reduce learning rate on plateau: 0.000010
2020-12-24 18:53:51,023 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:53:51,078 P17439 INFO Train loss: 0.139387
2020-12-24 18:53:51,078 P17439 INFO ************ Epoch=22 end ************
2020-12-24 18:53:59,361 P17439 INFO [Metrics] AUC: 0.969651 - logloss: 0.259287
2020-12-24 18:53:59,362 P17439 INFO Monitor(max) STOP: 0.969651 !
2020-12-24 18:53:59,362 P17439 INFO Reduce learning rate on plateau: 0.000001
2020-12-24 18:53:59,362 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:53:59,421 P17439 INFO Train loss: 0.106086
2020-12-24 18:53:59,421 P17439 INFO ************ Epoch=23 end ************
2020-12-24 18:54:07,679 P17439 INFO [Metrics] AUC: 0.969633 - logloss: 0.261251
2020-12-24 18:54:07,680 P17439 INFO Monitor(max) STOP: 0.969633 !
2020-12-24 18:54:07,680 P17439 INFO Reduce learning rate on plateau: 0.000001
2020-12-24 18:54:07,680 P17439 INFO Early stopping at epoch=24
2020-12-24 18:54:07,680 P17439 INFO --- 343/343 batches finished ---
2020-12-24 18:54:07,736 P17439 INFO Train loss: 0.098796
2020-12-24 18:54:07,736 P17439 INFO Training finished.
2020-12-24 18:54:07,736 P17439 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/MovielensLatest/PNN_movielenslatest_x1/movielenslatest_x1_bcd26aed/PNN_movielenslatest_x1_026_8a4a0ebb_model.ckpt
2020-12-24 18:54:07,764 P17439 INFO ****** Train/validation evaluation ******
2020-12-24 18:54:09,242 P17439 INFO [Metrics] AUC: 0.969699 - logloss: 0.210082
2020-12-24 18:54:09,358 P17439 INFO ******** Test evaluation ********
2020-12-24 18:54:09,359 P17439 INFO Loading data...
2020-12-24 18:54:09,359 P17439 INFO Loading data from h5: ../data/MovielensLatest/movielenslatest_x1_bcd26aed/test.h5
2020-12-24 18:54:09,363 P17439 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2020-12-24 18:54:09,363 P17439 INFO Loading test data done.
2020-12-24 18:54:10,288 P17439 INFO [Metrics] AUC: 0.969885 - logloss: 0.209455

```
