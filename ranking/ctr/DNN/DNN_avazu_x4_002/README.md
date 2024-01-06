## DNN_avazu_x4_002

A hands-on guide to run the DNN model on the Avazu_x4_002 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

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
Dataset ID: [Avazu_x4_002](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x4_002). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [DNN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DNN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DNN_avazu_x4_tuner_config_03](./DNN_avazu_x4_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DNN_avazu_x4_002
    nohup python run_expid.py --config ./DNN_avazu_x4_tuner_config_03 --expid DNN_avazu_x4_003_7fd77485 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.370475 | 0.795887  |


### Logs
```python
2020-02-23 11:01:54,613 P16308 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "l2(1.e-9)",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000, 1000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DNN",
    "model_id": "DNN_avazu_x4_003_58ecaddc",
    "model_root": "./Avazu/DNN_avazu/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "version": "pytorch",
    "gpu": "1"
}
2020-02-23 11:01:54,613 P16308 INFO Set up feature encoder...
2020-02-23 11:01:54,613 P16308 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-02-23 11:01:54,614 P16308 INFO Loading data...
2020-02-23 11:01:54,616 P16308 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-02-23 11:01:57,196 P16308 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-02-23 11:01:58,655 P16308 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-02-23 11:01:58,775 P16308 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-02-23 11:01:58,775 P16308 INFO Loading train data done.
2020-02-23 11:02:28,401 P16308 INFO **** Start training: 3235 batches/epoch ****
2020-02-23 11:14:15,459 P16308 INFO [Metrics] logloss: 0.370621 - AUC: 0.795628
2020-02-23 11:14:15,548 P16308 INFO Save best model: monitor(max): 0.425006
2020-02-23 11:14:28,945 P16308 INFO --- 3235/3235 batches finished ---
2020-02-23 11:14:28,983 P16308 INFO Train loss: 0.380214
2020-02-23 11:14:28,983 P16308 INFO ************ Epoch=1 end ************
2020-02-23 11:26:18,251 P16308 INFO [Metrics] logloss: 0.416595 - AUC: 0.770405
2020-02-23 11:26:18,350 P16308 INFO Monitor(max) STOP: 0.353811 !
2020-02-23 11:26:18,351 P16308 INFO Reduce learning rate on plateau: 0.000100
2020-02-23 11:26:18,351 P16308 INFO --- 3235/3235 batches finished ---
2020-02-23 11:26:18,435 P16308 INFO Train loss: 0.288926
2020-02-23 11:26:18,436 P16308 INFO ************ Epoch=2 end ************
2020-02-23 11:37:59,953 P16308 INFO [Metrics] logloss: 0.500406 - AUC: 0.757558
2020-02-23 11:38:00,052 P16308 INFO Monitor(max) STOP: 0.257152 !
2020-02-23 11:38:00,053 P16308 INFO Reduce learning rate on plateau: 0.000010
2020-02-23 11:38:00,053 P16308 INFO Early stopping at epoch=3
2020-02-23 11:38:00,053 P16308 INFO --- 3235/3235 batches finished ---
2020-02-23 11:38:00,129 P16308 INFO Train loss: 0.248428
2020-02-23 11:38:00,130 P16308 INFO Training finished.
2020-02-23 11:38:00,130 P16308 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Avazu/DNN_avazu/avazu_x4_001_d45ad60e/DNN_avazu_x4_003_58ecaddc_avazu_x4_001_d45ad60e_model.ckpt
2020-02-23 11:38:15,978 P16308 INFO ****** Train/validation evaluation ******
2020-02-23 11:43:19,671 P16308 INFO [Metrics] logloss: 0.321095 - AUC: 0.867518
2020-02-23 11:43:52,928 P16308 INFO [Metrics] logloss: 0.370621 - AUC: 0.795628
2020-02-23 11:43:53,052 P16308 INFO ******** Test evaluation ********
2020-02-23 11:43:53,052 P16308 INFO Loading data...
2020-02-23 11:43:53,052 P16308 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-02-23 11:43:53,655 P16308 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-02-23 11:43:53,655 P16308 INFO Loading test data done.
2020-02-23 11:44:25,250 P16308 INFO [Metrics] logloss: 0.370475 - AUC: 0.795887

```
