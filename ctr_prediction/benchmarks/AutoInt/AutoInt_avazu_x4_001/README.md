## AutoInt_avazu_x4_001

A hands-on guide to run the AutoInt model on the Avazu_x4_001 dataset.

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
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [Avazu_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [AutoInt](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt_avazu_x4_tuner_config_02](./AutoInt_avazu_x4_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AutoInt_avazu_x4_001
    nohup python run_expid.py --config ./AutoInt_avazu_x4_tuner_config_02 --expid AutoInt_avazu_x4_048_8c3b072b --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.374519 | 0.789103  |


### Logs
```python
2020-06-14 10:31:14,719 P14792 INFO {
    "attention_dim": "128",
    "attention_layers": "7",
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-06",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AutoInt",
    "model_id": "AutoInt_avazu_x4_3bbbc4c9_072_112d976f",
    "model_root": "./Avazu/AutoInt_avazu/min2/",
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
    "test_data": "../data/Avazu/avazu_x4_3bbbc4c9/test.h5",
    "train_data": "../data/Avazu/avazu_x4_3bbbc4c9/train.h5",
    "use_hdf5": "True",
    "use_residual": "False",
    "use_scale": "False",
    "use_wide": "True",
    "valid_data": "../data/Avazu/avazu_x4_3bbbc4c9/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-14 10:31:14,721 P14792 INFO Set up feature encoder...
2020-06-14 10:31:14,721 P14792 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-06-14 10:31:14,721 P14792 INFO Loading data...
2020-06-14 10:31:14,744 P14792 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-06-14 10:31:18,677 P14792 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-06-14 10:31:20,442 P14792 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-14 10:31:20,621 P14792 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-14 10:31:20,621 P14792 INFO Loading train data done.
2020-06-14 10:31:28,827 P14792 INFO Start training: 3235 batches/epoch
2020-06-14 10:31:28,827 P14792 INFO ************ Epoch=1 start ************
2020-06-14 10:43:35,262 P14792 INFO [Metrics] logloss: 0.377925 - AUC: 0.783227
2020-06-14 10:43:35,263 P14792 INFO Save best model: monitor(max): 0.405302
2020-06-14 10:43:36,155 P14792 INFO --- 3235/3235 batches finished ---
2020-06-14 10:43:36,213 P14792 INFO Train loss: 0.393201
2020-06-14 10:43:36,213 P14792 INFO ************ Epoch=1 end ************
2020-06-14 10:55:35,535 P14792 INFO [Metrics] logloss: 0.374649 - AUC: 0.788863
2020-06-14 10:55:35,540 P14792 INFO Save best model: monitor(max): 0.414213
2020-06-14 10:55:36,005 P14792 INFO --- 3235/3235 batches finished ---
2020-06-14 10:55:36,066 P14792 INFO Train loss: 0.379079
2020-06-14 10:55:36,067 P14792 INFO ************ Epoch=2 end ************
2020-06-14 11:07:31,028 P14792 INFO [Metrics] logloss: 0.374997 - AUC: 0.789109
2020-06-14 11:07:31,031 P14792 INFO Monitor(max) STOP: 0.414112 !
2020-06-14 11:07:31,031 P14792 INFO Reduce learning rate on plateau: 0.000100
2020-06-14 11:07:31,031 P14792 INFO --- 3235/3235 batches finished ---
2020-06-14 11:07:31,092 P14792 INFO Train loss: 0.367851
2020-06-14 11:07:31,093 P14792 INFO ************ Epoch=3 end ************
2020-06-14 11:19:25,609 P14792 INFO [Metrics] logloss: 0.399006 - AUC: 0.779783
2020-06-14 11:19:25,613 P14792 INFO Monitor(max) STOP: 0.380777 !
2020-06-14 11:19:25,613 P14792 INFO Reduce learning rate on plateau: 0.000010
2020-06-14 11:19:25,614 P14792 INFO Early stopping at epoch=4
2020-06-14 11:19:25,614 P14792 INFO --- 3235/3235 batches finished ---
2020-06-14 11:19:25,674 P14792 INFO Train loss: 0.324028
2020-06-14 11:19:25,674 P14792 INFO Training finished.
2020-06-14 11:19:25,674 P14792 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Avazu/AutoInt_avazu/min2/avazu_x4_3bbbc4c9/AutoInt_avazu_x4_3bbbc4c9_072_112d976f_model.ckpt
2020-06-14 11:19:26,097 P14792 INFO ****** Train/validation evaluation ******
2020-06-14 11:22:58,032 P14792 INFO [Metrics] logloss: 0.339152 - AUC: 0.842612
2020-06-14 11:23:22,259 P14792 INFO [Metrics] logloss: 0.374649 - AUC: 0.788863
2020-06-14 11:23:22,342 P14792 INFO ******** Test evaluation ********
2020-06-14 11:23:22,342 P14792 INFO Loading data...
2020-06-14 11:23:22,342 P14792 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-06-14 11:23:23,028 P14792 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-14 11:23:23,029 P14792 INFO Loading test data done.
2020-06-14 11:23:48,327 P14792 INFO [Metrics] logloss: 0.374519 - AUC: 0.789103

```
