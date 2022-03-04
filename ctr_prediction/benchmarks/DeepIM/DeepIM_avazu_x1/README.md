## DeepIM_avazu_x1

A hands-on guide to run the DeepIM model on the Avazu_x1 dataset.

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
Dataset ID: [Avazu_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DeepIM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DeepIM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DeepIM_avazu_x1_tuner_config_03](./DeepIM_avazu_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DeepIM_avazu_x1
    nohup python run_expid.py --config ./DeepIM_avazu_x1_tuner_config_03 --expid DeepIM_avazu_x1_001_ce22770f --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.764527 | 0.366952  |
| 2 | 0.764594 | 0.366688  |
| 3 | 0.764754 | 0.366836  |
| 4 | 0.764719 | 0.366887  |
| 5 | 0.764546 | 0.367109  |
| | | | 
| Avg | 0.764628 | 0.366894 |
| Std | &#177;0.00009191 | &#177;0.00013816 |


### Logs
```python
2022-02-08 16:14:11,989 P64571 INFO {
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.005",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "im_batch_norm": "False",
    "im_order": "5",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DeepIM",
    "model_id": "DeepIM_avazu_x1_001_ce22770f",
    "model_root": "./Avazu/DeepIM_avazu_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_batch_norm": "True",
    "net_dropout": "0.1",
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
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-02-08 16:14:11,990 P64571 INFO Set up feature encoder...
2022-02-08 16:14:11,990 P64571 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-02-08 16:14:11,990 P64571 INFO Loading data...
2022-02-08 16:14:11,992 P64571 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-02-08 16:14:14,446 P64571 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-02-08 16:14:14,776 P64571 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-02-08 16:14:14,776 P64571 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-02-08 16:14:14,776 P64571 INFO Loading train data done.
2022-02-08 16:14:18,902 P64571 INFO Total number of parameters: 13398042.
2022-02-08 16:14:18,902 P64571 INFO Start training: 6910 batches/epoch
2022-02-08 16:14:18,902 P64571 INFO ************ Epoch=1 start ************
2022-02-08 16:19:30,210 P64571 INFO [Metrics] AUC: 0.743777 - logloss: 0.397452
2022-02-08 16:19:30,213 P64571 INFO Save best model: monitor(max): 0.743777
2022-02-08 16:19:30,273 P64571 INFO --- 6910/6910 batches finished ---
2022-02-08 16:19:30,314 P64571 INFO Train loss: 0.428072
2022-02-08 16:19:30,314 P64571 INFO ************ Epoch=1 end ************
2022-02-08 16:24:40,462 P64571 INFO [Metrics] AUC: 0.743606 - logloss: 0.397428
2022-02-08 16:24:40,465 P64571 INFO Monitor(max) STOP: 0.743606 !
2022-02-08 16:24:40,465 P64571 INFO Reduce learning rate on plateau: 0.000100
2022-02-08 16:24:40,465 P64571 INFO --- 6910/6910 batches finished ---
2022-02-08 16:24:40,512 P64571 INFO Train loss: 0.426713
2022-02-08 16:24:40,512 P64571 INFO ************ Epoch=2 end ************
2022-02-08 16:29:46,813 P64571 INFO [Metrics] AUC: 0.746659 - logloss: 0.395856
2022-02-08 16:29:46,816 P64571 INFO Save best model: monitor(max): 0.746659
2022-02-08 16:29:46,888 P64571 INFO --- 6910/6910 batches finished ---
2022-02-08 16:29:46,937 P64571 INFO Train loss: 0.402327
2022-02-08 16:29:46,938 P64571 INFO ************ Epoch=3 end ************
2022-02-08 16:34:53,922 P64571 INFO [Metrics] AUC: 0.746947 - logloss: 0.395691
2022-02-08 16:34:53,924 P64571 INFO Save best model: monitor(max): 0.746947
2022-02-08 16:34:53,994 P64571 INFO --- 6910/6910 batches finished ---
2022-02-08 16:34:54,040 P64571 INFO Train loss: 0.401785
2022-02-08 16:34:54,041 P64571 INFO ************ Epoch=4 end ************
2022-02-08 16:40:02,239 P64571 INFO [Metrics] AUC: 0.746606 - logloss: 0.396089
2022-02-08 16:40:02,242 P64571 INFO Monitor(max) STOP: 0.746606 !
2022-02-08 16:40:02,242 P64571 INFO Reduce learning rate on plateau: 0.000010
2022-02-08 16:40:02,242 P64571 INFO --- 6910/6910 batches finished ---
2022-02-08 16:40:02,283 P64571 INFO Train loss: 0.401593
2022-02-08 16:40:02,283 P64571 INFO ************ Epoch=5 end ************
2022-02-08 16:45:10,808 P64571 INFO [Metrics] AUC: 0.741757 - logloss: 0.399125
2022-02-08 16:45:10,812 P64571 INFO Monitor(max) STOP: 0.741757 !
2022-02-08 16:45:10,812 P64571 INFO Reduce learning rate on plateau: 0.000001
2022-02-08 16:45:10,812 P64571 INFO Early stopping at epoch=6
2022-02-08 16:45:10,812 P64571 INFO --- 6910/6910 batches finished ---
2022-02-08 16:45:10,853 P64571 INFO Train loss: 0.388748
2022-02-08 16:45:10,853 P64571 INFO Training finished.
2022-02-08 16:45:10,854 P64571 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/DeepIM_avazu_x1/avazu_x1_3fb65689/DeepIM_avazu_x1_001_ce22770f.model
2022-02-08 16:45:10,914 P64571 INFO ****** Validation evaluation ******
2022-02-08 16:45:22,057 P64571 INFO [Metrics] AUC: 0.746947 - logloss: 0.395691
2022-02-08 16:45:22,123 P64571 INFO ******** Test evaluation ********
2022-02-08 16:45:22,123 P64571 INFO Loading data...
2022-02-08 16:45:22,123 P64571 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-02-08 16:45:22,943 P64571 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-02-08 16:45:22,943 P64571 INFO Loading test data done.
2022-02-08 16:45:46,776 P64571 INFO [Metrics] AUC: 0.764527 - logloss: 0.366952

```
