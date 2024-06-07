## AutoInt_avazu_x4_002

A hands-on guide to run the AutoInt model on the Avazu_x4_002 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
  GPU: Tesla P100 16G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 10.0
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [Avazu_x4](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x4). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [AutoInt](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt_avazu_x4_tuner_config_04](./AutoInt_avazu_x4_tuner_config_04). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AutoInt_avazu_x4_002
    nohup python run_expid.py --config ./AutoInt_avazu_x4_tuner_config_04 --expid AutoInt_avazu_x4_013_3a66ab94 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.372581 | 0.792228  |


### Logs
```python
2020-05-11 18:36:47,739 P26065 INFO {
    "attention_dim": "160",
    "attention_layers": "6",
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AutoInt",
    "model_id": "AutoInt_avazu_x4_013_3a66ab94",
    "model_root": "./Avazu/AutoInt_avazu/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "1",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "use_hdf5": "True",
    "use_residual": "True",
    "use_scale": "False",
    "use_wide": "False",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-05-11 18:36:47,740 P26065 INFO Set up feature encoder...
2020-05-11 18:36:47,740 P26065 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-05-11 18:36:47,741 P26065 INFO Loading data...
2020-05-11 18:36:47,742 P26065 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-05-11 18:36:50,514 P26065 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-05-11 18:36:51,806 P26065 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-05-11 18:36:51,922 P26065 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-05-11 18:36:51,922 P26065 INFO Loading train data done.
2020-05-11 18:37:05,045 P26065 INFO **** Start training: 3235 batches/epoch ****
2020-05-11 18:59:28,106 P26065 INFO [Metrics] logloss: 0.372695 - AUC: 0.791962
2020-05-11 18:59:28,109 P26065 INFO Save best model: monitor(max): 0.419268
2020-05-11 18:59:29,324 P26065 INFO --- 3235/3235 batches finished ---
2020-05-11 18:59:29,357 P26065 INFO Train loss: 0.382909
2020-05-11 18:59:29,357 P26065 INFO ************ Epoch=1 end ************
2020-05-11 19:21:51,275 P26065 INFO [Metrics] logloss: 0.467224 - AUC: 0.756973
2020-05-11 19:21:51,277 P26065 INFO Monitor(max) STOP: 0.289749 !
2020-05-11 19:21:51,278 P26065 INFO Reduce learning rate on plateau: 0.000100
2020-05-11 19:21:51,278 P26065 INFO --- 3235/3235 batches finished ---
2020-05-11 19:21:51,311 P26065 INFO Train loss: 0.281661
2020-05-11 19:21:51,311 P26065 INFO ************ Epoch=2 end ************
2020-05-11 19:44:14,621 P26065 INFO [Metrics] logloss: 0.516160 - AUC: 0.749198
2020-05-11 19:44:14,624 P26065 INFO Monitor(max) STOP: 0.233038 !
2020-05-11 19:44:14,624 P26065 INFO Reduce learning rate on plateau: 0.000010
2020-05-11 19:44:14,624 P26065 INFO Early stopping at epoch=3
2020-05-11 19:44:14,624 P26065 INFO --- 3235/3235 batches finished ---
2020-05-11 19:44:14,657 P26065 INFO Train loss: 0.254104
2020-05-11 19:44:14,657 P26065 INFO Training finished.
2020-05-11 19:44:14,657 P26065 INFO Load best model: /home/XXX/benchmarks/Avazu/AutoInt_avazu/avazu_x4_001_d45ad60e/AutoInt_avazu_x4_013_3a66ab94_model.ckpt
2020-05-11 19:44:16,275 P26065 INFO ****** Train/validation evaluation ******
2020-05-11 19:44:56,883 P26065 INFO [Metrics] logloss: 0.372695 - AUC: 0.791962
2020-05-11 19:44:56,984 P26065 INFO ******** Test evaluation ********
2020-05-11 19:44:56,984 P26065 INFO Loading data...
2020-05-11 19:44:56,984 P26065 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-05-11 19:44:57,417 P26065 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-05-11 19:44:57,418 P26065 INFO Loading test data done.
2020-05-11 19:45:38,162 P26065 INFO [Metrics] logloss: 0.372581 - AUC: 0.792228

```
