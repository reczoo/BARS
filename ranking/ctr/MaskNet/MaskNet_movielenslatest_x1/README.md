## MaskNet_movielenslatest_x1

A hands-on guide to run the MaskNet model on the MovielensLatest_x1 dataset.

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
  fuxictr: 1.2.1
  ```

### Dataset
Dataset ID: [MovielensLatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/MovieLens#MovielensLatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [MaskNet](https://github.com/xue-pai/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/MaskNet.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [MaskNet_movielenslatest_x1_tuner_config_03](./MaskNet_movielenslatest_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd MaskNet_movielenslatest_x1
    nohup python run_expid.py --config ./MaskNet_movielenslatest_x1_tuner_config_03 --expid MaskNet_movielenslatest_x1_010_13a8d29c --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.967201 | 0.236404  |


### Logs
```python
2022-05-26 16:22:44,402 P24572 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "dnn_hidden_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "emb_layernorm": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.005",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "MaskNet",
    "model_id": "MaskNet_movielenslatest_x1_010_13a8d29c",
    "model_root": "./Movielens/MaskNet_movielenslatest_x1/",
    "model_type": "SerialMaskNet",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.5",
    "net_layernorm": "True",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_block_dim": "64",
    "parallel_num_blocks": "1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "4",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Movielens/MovielensLatest_x1/test.csv",
    "train_data": "../data/Movielens/MovielensLatest_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-05-26 16:22:44,403 P24572 INFO Set up feature encoder...
2022-05-26 16:22:44,403 P24572 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-05-26 16:22:44,403 P24572 INFO Loading data...
2022-05-26 16:22:44,407 P24572 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-05-26 16:22:44,434 P24572 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-05-26 16:22:44,442 P24572 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-05-26 16:22:44,442 P24572 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-05-26 16:22:44,442 P24572 INFO Loading train data done.
2022-05-26 16:22:48,199 P24572 INFO Total number of parameters: 2624541.
2022-05-26 16:22:48,200 P24572 INFO Start training: 343 batches/epoch
2022-05-26 16:22:48,200 P24572 INFO ************ Epoch=1 start ************
2022-05-26 16:23:06,838 P24572 INFO [Metrics] AUC: 0.939530 - logloss: 0.279642
2022-05-26 16:23:06,839 P24572 INFO Save best model: monitor(max): 0.939530
2022-05-26 16:23:06,855 P24572 INFO --- 343/343 batches finished ---
2022-05-26 16:23:06,936 P24572 INFO Train loss: 0.358909
2022-05-26 16:23:06,936 P24572 INFO ************ Epoch=1 end ************
2022-05-26 16:23:26,245 P24572 INFO [Metrics] AUC: 0.947827 - logloss: 0.259432
2022-05-26 16:23:26,246 P24572 INFO Save best model: monitor(max): 0.947827
2022-05-26 16:23:26,263 P24572 INFO --- 343/343 batches finished ---
2022-05-26 16:23:26,319 P24572 INFO Train loss: 0.335317
2022-05-26 16:23:26,319 P24572 INFO ************ Epoch=2 end ************
2022-05-26 16:23:48,661 P24572 INFO [Metrics] AUC: 0.951542 - logloss: 0.248670
2022-05-26 16:23:48,662 P24572 INFO Save best model: monitor(max): 0.951542
2022-05-26 16:23:48,681 P24572 INFO --- 343/343 batches finished ---
2022-05-26 16:23:48,737 P24572 INFO Train loss: 0.337195
2022-05-26 16:23:48,737 P24572 INFO ************ Epoch=3 end ************
2022-05-26 16:24:10,982 P24572 INFO [Metrics] AUC: 0.954593 - logloss: 0.239815
2022-05-26 16:24:10,983 P24572 INFO Save best model: monitor(max): 0.954593
2022-05-26 16:24:10,996 P24572 INFO --- 343/343 batches finished ---
2022-05-26 16:24:11,033 P24572 INFO Train loss: 0.339654
2022-05-26 16:24:11,033 P24572 INFO ************ Epoch=4 end ************
2022-05-26 16:24:33,108 P24572 INFO [Metrics] AUC: 0.956553 - logloss: 0.234321
2022-05-26 16:24:33,108 P24572 INFO Save best model: monitor(max): 0.956553
2022-05-26 16:24:33,122 P24572 INFO --- 343/343 batches finished ---
2022-05-26 16:24:33,171 P24572 INFO Train loss: 0.341546
2022-05-26 16:24:33,171 P24572 INFO ************ Epoch=5 end ************
2022-05-26 16:24:55,427 P24572 INFO [Metrics] AUC: 0.957810 - logloss: 0.230663
2022-05-26 16:24:55,428 P24572 INFO Save best model: monitor(max): 0.957810
2022-05-26 16:24:55,445 P24572 INFO --- 343/343 batches finished ---
2022-05-26 16:24:55,489 P24572 INFO Train loss: 0.342225
2022-05-26 16:24:55,489 P24572 INFO ************ Epoch=6 end ************
2022-05-26 16:25:17,591 P24572 INFO [Metrics] AUC: 0.959182 - logloss: 0.228512
2022-05-26 16:25:17,591 P24572 INFO Save best model: monitor(max): 0.959182
2022-05-26 16:25:17,607 P24572 INFO --- 343/343 batches finished ---
2022-05-26 16:25:17,654 P24572 INFO Train loss: 0.343592
2022-05-26 16:25:17,654 P24572 INFO ************ Epoch=7 end ************
2022-05-26 16:25:39,861 P24572 INFO [Metrics] AUC: 0.959392 - logloss: 0.226583
2022-05-26 16:25:39,862 P24572 INFO Save best model: monitor(max): 0.959392
2022-05-26 16:25:39,877 P24572 INFO --- 343/343 batches finished ---
2022-05-26 16:25:39,928 P24572 INFO Train loss: 0.342146
2022-05-26 16:25:39,929 P24572 INFO ************ Epoch=8 end ************
2022-05-26 16:26:01,899 P24572 INFO [Metrics] AUC: 0.960518 - logloss: 0.222874
2022-05-26 16:26:01,900 P24572 INFO Save best model: monitor(max): 0.960518
2022-05-26 16:26:01,914 P24572 INFO --- 343/343 batches finished ---
2022-05-26 16:26:01,960 P24572 INFO Train loss: 0.343147
2022-05-26 16:26:01,960 P24572 INFO ************ Epoch=9 end ************
2022-05-26 16:26:23,953 P24572 INFO [Metrics] AUC: 0.959965 - logloss: 0.224976
2022-05-26 16:26:23,954 P24572 INFO Monitor(max) STOP: 0.959965 !
2022-05-26 16:26:23,954 P24572 INFO Reduce learning rate on plateau: 0.000100
2022-05-26 16:26:23,954 P24572 INFO --- 343/343 batches finished ---
2022-05-26 16:26:24,022 P24572 INFO Train loss: 0.342246
2022-05-26 16:26:24,022 P24572 INFO ************ Epoch=10 end ************
2022-05-26 16:26:46,012 P24572 INFO [Metrics] AUC: 0.966597 - logloss: 0.219069
2022-05-26 16:26:46,012 P24572 INFO Save best model: monitor(max): 0.966597
2022-05-26 16:26:46,030 P24572 INFO --- 343/343 batches finished ---
2022-05-26 16:26:46,073 P24572 INFO Train loss: 0.253219
2022-05-26 16:26:46,073 P24572 INFO ************ Epoch=11 end ************
2022-05-26 16:27:08,017 P24572 INFO [Metrics] AUC: 0.967110 - logloss: 0.237895
2022-05-26 16:27:08,018 P24572 INFO Save best model: monitor(max): 0.967110
2022-05-26 16:27:08,036 P24572 INFO --- 343/343 batches finished ---
2022-05-26 16:27:08,112 P24572 INFO Train loss: 0.175022
2022-05-26 16:27:08,112 P24572 INFO ************ Epoch=12 end ************
2022-05-26 16:27:30,239 P24572 INFO [Metrics] AUC: 0.966273 - logloss: 0.270119
2022-05-26 16:27:30,240 P24572 INFO Monitor(max) STOP: 0.966273 !
2022-05-26 16:27:30,240 P24572 INFO Reduce learning rate on plateau: 0.000010
2022-05-26 16:27:30,240 P24572 INFO --- 343/343 batches finished ---
2022-05-26 16:27:30,286 P24572 INFO Train loss: 0.129939
2022-05-26 16:27:30,286 P24572 INFO ************ Epoch=13 end ************
2022-05-26 16:27:52,708 P24572 INFO [Metrics] AUC: 0.965713 - logloss: 0.313920
2022-05-26 16:27:52,709 P24572 INFO Monitor(max) STOP: 0.965713 !
2022-05-26 16:27:52,709 P24572 INFO Reduce learning rate on plateau: 0.000001
2022-05-26 16:27:52,709 P24572 INFO Early stopping at epoch=14
2022-05-26 16:27:52,709 P24572 INFO --- 343/343 batches finished ---
2022-05-26 16:27:52,754 P24572 INFO Train loss: 0.101960
2022-05-26 16:27:52,754 P24572 INFO Training finished.
2022-05-26 16:27:52,755 P24572 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Movielens/MaskNet_movielenslatest_x1/movielenslatest_x1_cd32d937/MaskNet_movielenslatest_x1_010_13a8d29c.model
2022-05-26 16:27:56,149 P24572 INFO ****** Validation evaluation ******
2022-05-26 16:27:58,164 P24572 INFO [Metrics] AUC: 0.967110 - logloss: 0.237895
2022-05-26 16:27:58,308 P24572 INFO ******** Test evaluation ********
2022-05-26 16:27:58,308 P24572 INFO Loading data...
2022-05-26 16:27:58,309 P24572 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-05-26 16:27:58,316 P24572 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-05-26 16:27:58,316 P24572 INFO Loading test data done.
2022-05-26 16:27:59,400 P24572 INFO [Metrics] AUC: 0.967201 - logloss: 0.236404

```
