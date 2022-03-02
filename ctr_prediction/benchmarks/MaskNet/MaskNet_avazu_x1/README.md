## MaskNet_avazu_x1

A hands-on guide to run the MaskNet model on the Avazu_x1 dataset.

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
  fuxictr: 1.1.1

  ```

### Dataset
Dataset ID: [Avazu_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.1](fuxictr_url) for this experiment. See model code: [MaskNet](https://github.com/xue-pai/FuxiCTR/blob/v1.1.1/fuxictr/pytorch/models/MaskNet.py).

Running steps:

1. Download [FuxiCTR-v1.1.1](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [MaskNet_avazu_x1_tuner_config_03](./MaskNet_avazu_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd MaskNet_avazu_x1
    nohup python run_expid.py --config ./MaskNet_avazu_x1_tuner_config_03 --expid MaskNet_avazu_x1_006_85019ef5 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.764946 | 0.367391  |
| 2 | 0.760696 | 0.368801  |
| 3 | 0.757452 | 0.370855  |
| 4 | 0.760375 | 0.368818  |
| 5 | 0.763312 | 0.367983  |
| | | | 
| Avg | 0.761356 | 0.368770 |
| Std | &#177;0.00258276 | &#177;0.00117231 |


### Logs
```python
2022-01-29 21:52:04,290 P70698 INFO {
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "dnn_hidden_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "emb_layernorm": "True",
    "embedding_dim": "10",
    "embedding_regularizer": "0.1",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "5",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "MaskNet",
    "model_id": "MaskNet_avazu_x1_006_85019ef5",
    "model_root": "./Avazu/MaskNet_avazu_x1/",
    "model_type": "SerialMaskNet",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_layernorm": "True",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_block_dim": "64",
    "parallel_num_blocks": "1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "1",
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
2022-01-29 21:52:04,291 P70698 INFO Set up feature encoder...
2022-01-29 21:52:04,291 P70698 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-01-29 21:52:04,291 P70698 INFO Loading data...
2022-01-29 21:52:04,293 P70698 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-01-29 21:52:06,823 P70698 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-01-29 21:52:07,164 P70698 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-01-29 21:52:07,165 P70698 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-01-29 21:52:07,165 P70698 INFO Loading train data done.
2022-01-29 21:52:11,842 P70698 INFO Total number of parameters: 13992071.
2022-01-29 21:52:11,843 P70698 INFO Start training: 6910 batches/epoch
2022-01-29 21:52:11,843 P70698 INFO ************ Epoch=1 start ************
2022-01-29 21:57:07,217 P70698 INFO [Metrics] AUC: 0.726944 - logloss: 0.405138
2022-01-29 21:57:07,221 P70698 INFO Save best model: monitor(max): 0.726944
2022-01-29 21:57:07,426 P70698 INFO --- 6910/6910 batches finished ---
2022-01-29 21:57:07,466 P70698 INFO Train loss: 0.446690
2022-01-29 21:57:07,467 P70698 INFO ************ Epoch=1 end ************
2022-01-29 22:01:59,745 P70698 INFO [Metrics] AUC: 0.736898 - logloss: 0.400907
2022-01-29 22:01:59,747 P70698 INFO Save best model: monitor(max): 0.736898
2022-01-29 22:01:59,818 P70698 INFO --- 6910/6910 batches finished ---
2022-01-29 22:01:59,860 P70698 INFO Train loss: 0.444385
2022-01-29 22:01:59,860 P70698 INFO ************ Epoch=2 end ************
2022-01-29 22:06:53,584 P70698 INFO [Metrics] AUC: 0.735256 - logloss: 0.401581
2022-01-29 22:06:53,586 P70698 INFO Monitor(max) STOP: 0.735256 !
2022-01-29 22:06:53,586 P70698 INFO Reduce learning rate on plateau: 0.000100
2022-01-29 22:06:53,586 P70698 INFO --- 6910/6910 batches finished ---
2022-01-29 22:06:53,628 P70698 INFO Train loss: 0.443157
2022-01-29 22:06:53,628 P70698 INFO ************ Epoch=3 end ************
2022-01-29 22:11:46,543 P70698 INFO [Metrics] AUC: 0.742298 - logloss: 0.397526
2022-01-29 22:11:46,546 P70698 INFO Save best model: monitor(max): 0.742298
2022-01-29 22:11:46,614 P70698 INFO --- 6910/6910 batches finished ---
2022-01-29 22:11:46,658 P70698 INFO Train loss: 0.409985
2022-01-29 22:11:46,658 P70698 INFO ************ Epoch=4 end ************
2022-01-29 22:16:37,416 P70698 INFO [Metrics] AUC: 0.744285 - logloss: 0.396944
2022-01-29 22:16:37,418 P70698 INFO Save best model: monitor(max): 0.744285
2022-01-29 22:16:37,483 P70698 INFO --- 6910/6910 batches finished ---
2022-01-29 22:16:37,525 P70698 INFO Train loss: 0.412193
2022-01-29 22:16:37,525 P70698 INFO ************ Epoch=5 end ************
2022-01-29 22:21:28,898 P70698 INFO [Metrics] AUC: 0.742247 - logloss: 0.397706
2022-01-29 22:21:28,901 P70698 INFO Monitor(max) STOP: 0.742247 !
2022-01-29 22:21:28,901 P70698 INFO Reduce learning rate on plateau: 0.000010
2022-01-29 22:21:28,901 P70698 INFO --- 6910/6910 batches finished ---
2022-01-29 22:21:28,946 P70698 INFO Train loss: 0.413557
2022-01-29 22:21:28,946 P70698 INFO ************ Epoch=6 end ************
2022-01-29 22:26:20,006 P70698 INFO [Metrics] AUC: 0.745233 - logloss: 0.396289
2022-01-29 22:26:20,008 P70698 INFO Save best model: monitor(max): 0.745233
2022-01-29 22:26:20,077 P70698 INFO --- 6910/6910 batches finished ---
2022-01-29 22:26:20,119 P70698 INFO Train loss: 0.397188
2022-01-29 22:26:20,119 P70698 INFO ************ Epoch=7 end ************
2022-01-29 22:31:15,252 P70698 INFO [Metrics] AUC: 0.743739 - logloss: 0.397242
2022-01-29 22:31:15,256 P70698 INFO Monitor(max) STOP: 0.743739 !
2022-01-29 22:31:15,256 P70698 INFO Reduce learning rate on plateau: 0.000001
2022-01-29 22:31:15,256 P70698 INFO --- 6910/6910 batches finished ---
2022-01-29 22:31:15,299 P70698 INFO Train loss: 0.394561
2022-01-29 22:31:15,299 P70698 INFO ************ Epoch=8 end ************
2022-01-29 22:36:07,559 P70698 INFO [Metrics] AUC: 0.738594 - logloss: 0.400480
2022-01-29 22:36:07,561 P70698 INFO Monitor(max) STOP: 0.738594 !
2022-01-29 22:36:07,561 P70698 INFO Reduce learning rate on plateau: 0.000001
2022-01-29 22:36:07,561 P70698 INFO Early stopping at epoch=9
2022-01-29 22:36:07,561 P70698 INFO --- 6910/6910 batches finished ---
2022-01-29 22:36:07,606 P70698 INFO Train loss: 0.387007
2022-01-29 22:36:07,606 P70698 INFO Training finished.
2022-01-29 22:36:07,606 P70698 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/MaskNet_avazu_x1/avazu_x1_3fb65689/MaskNet_avazu_x1_006_85019ef5.model
2022-01-29 22:36:10,751 P70698 INFO ****** Validation evaluation ******
2022-01-29 22:36:22,707 P70698 INFO [Metrics] AUC: 0.745233 - logloss: 0.396289
2022-01-29 22:36:22,802 P70698 INFO ******** Test evaluation ********
2022-01-29 22:36:22,802 P70698 INFO Loading data...
2022-01-29 22:36:22,803 P70698 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-01-29 22:36:23,498 P70698 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-01-29 22:36:23,498 P70698 INFO Loading test data done.
2022-01-29 22:36:48,785 P70698 INFO [Metrics] AUC: 0.764946 - logloss: 0.367391

```
