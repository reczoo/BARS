## DeepFM_avazu_x4_002

A hands-on guide to run the DeepFM model on the Avazu_x4_002 dataset.

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
Dataset ID: [Avazu_x4_002](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x4_002). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [DeepFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DeepFM_avazu_x4_tuner_config_07](./DeepFM_avazu_x4_tuner_config_07). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DeepFM_avazu_x4_002
    nohup python run_expid.py --config ./DeepFM_avazu_x4_tuner_config_07 --expid DeepFM_avazu_x4_003_f37a42a1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.370178 | 0.796177  |


### Logs
```python
2020-03-03 17:25:15,479 P825 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[3000, 3000, 3000, 3000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DeepFM",
    "model_id": "DeepFM_avazu_x4_003_f11d0986",
    "model_root": "./Avazu/DeepFM_avazu/",
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
    "gpu": "0"
}
2020-03-03 17:25:15,481 P825 INFO Set up feature encoder...
2020-03-03 17:25:15,481 P825 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-03-03 17:25:15,482 P825 INFO Loading data...
2020-03-03 17:25:15,486 P825 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-03-03 17:25:20,154 P825 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-03-03 17:25:22,053 P825 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-03-03 17:25:22,236 P825 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-03-03 17:25:22,236 P825 INFO Loading train data done.
2020-03-03 17:25:36,334 P825 INFO **** Start training: 3235 batches/epoch ****
2020-03-03 17:41:13,475 P825 INFO [Metrics] logloss: 0.370368 - AUC: 0.795802
2020-03-03 17:41:13,570 P825 INFO Save best model: monitor(max): 0.425434
2020-03-03 17:41:15,790 P825 INFO --- 3235/3235 batches finished ---
2020-03-03 17:41:15,867 P825 INFO Train loss: 0.379846
2020-03-03 17:41:15,868 P825 INFO ************ Epoch=1 end ************
2020-03-03 17:56:50,749 P825 INFO [Metrics] logloss: 0.450234 - AUC: 0.761621
2020-03-03 17:56:50,801 P825 INFO Monitor(max) STOP: 0.311386 !
2020-03-03 17:56:50,802 P825 INFO Reduce learning rate on plateau: 0.000100
2020-03-03 17:56:50,802 P825 INFO --- 3235/3235 batches finished ---
2020-03-03 17:56:50,866 P825 INFO Train loss: 0.286010
2020-03-03 17:56:50,866 P825 INFO ************ Epoch=2 end ************
2020-03-03 18:12:24,407 P825 INFO [Metrics] logloss: 0.512873 - AUC: 0.757223
2020-03-03 18:12:24,473 P825 INFO Monitor(max) STOP: 0.244350 !
2020-03-03 18:12:24,473 P825 INFO Reduce learning rate on plateau: 0.000010
2020-03-03 18:12:24,473 P825 INFO Early stopping at epoch=3
2020-03-03 18:12:24,473 P825 INFO --- 3235/3235 batches finished ---
2020-03-03 18:12:24,535 P825 INFO Train loss: 0.243807
2020-03-03 18:12:24,536 P825 INFO Training finished.
2020-03-03 18:12:24,536 P825 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Avazu/DeepFM_avazu/avazu_x4_001_d45ad60e/DeepFM_avazu_x4_003_f11d0986_avazu_x4_001_d45ad60e_model.ckpt
2020-03-03 18:12:27,391 P825 INFO ****** Train/validation evaluation ******
2020-03-03 18:16:58,624 P825 INFO [Metrics] logloss: 0.318769 - AUC: 0.868529
2020-03-03 18:17:32,060 P825 INFO [Metrics] logloss: 0.370368 - AUC: 0.795802
2020-03-03 18:17:32,215 P825 INFO ******** Test evaluation ********
2020-03-03 18:17:32,215 P825 INFO Loading data...
2020-03-03 18:17:32,216 P825 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-03-03 18:17:32,996 P825 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-03-03 18:17:32,997 P825 INFO Loading test data done.
2020-03-03 18:18:06,729 P825 INFO [Metrics] logloss: 0.370178 - AUC: 0.796177

```
