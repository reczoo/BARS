## FwFM_avazu_x4_002

A hands-on guide to run the FwFM model on the Avazu_x4_002 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FwFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FwFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FwFM_avazu_x4_tuner_config_02](./FwFM_avazu_x4_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FwFM_avazu_x4_002
    nohup python run_expid.py --config ./FwFM_avazu_x4_tuner_config_02 --expid FwFM_avazu_x4_003_9c0d0ea9 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.372403 | 0.792534  |


### Logs
```python
2020-03-01 01:21:21,195 P5026 INFO {
    "batch_size": "10000",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "embedding_dim": "40",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "linear_type": "FiLV",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FwFM",
    "model_id": "FwFM_avazu_x4_003_fec49391",
    "model_root": "./Avazu/FwFM_avazu/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "0",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "verbose": "1",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "version": "pytorch",
    "gpu": "1"
}
2020-03-01 01:21:21,196 P5026 INFO Set up feature encoder...
2020-03-01 01:21:21,196 P5026 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-03-01 01:21:21,196 P5026 INFO Loading data...
2020-03-01 01:21:21,199 P5026 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-03-01 01:21:24,042 P5026 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-03-01 01:21:25,302 P5026 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-03-01 01:21:25,412 P5026 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-03-01 01:21:25,412 P5026 INFO Loading train data done.
2020-03-01 01:21:35,921 P5026 INFO **** Start training: 3235 batches/epoch ****
2020-03-01 01:32:09,330 P5026 INFO [Metrics] logloss: 0.372556 - AUC: 0.792171
2020-03-01 01:32:09,423 P5026 INFO Save best model: monitor(max): 0.419615
2020-03-01 01:32:10,709 P5026 INFO --- 3235/3235 batches finished ---
2020-03-01 01:32:10,756 P5026 INFO Train loss: 0.384608
2020-03-01 01:32:10,756 P5026 INFO ************ Epoch=1 end ************
2020-03-01 01:42:41,686 P5026 INFO [Metrics] logloss: 0.390477 - AUC: 0.782471
2020-03-01 01:42:41,777 P5026 INFO Monitor(max) STOP: 0.391994 !
2020-03-01 01:42:41,777 P5026 INFO Reduce learning rate on plateau: 0.000100
2020-03-01 01:42:41,777 P5026 INFO --- 3235/3235 batches finished ---
2020-03-01 01:42:41,873 P5026 INFO Train loss: 0.315592
2020-03-01 01:42:41,873 P5026 INFO ************ Epoch=2 end ************
2020-03-01 01:53:12,393 P5026 INFO [Metrics] logloss: 0.428223 - AUC: 0.768862
2020-03-01 01:53:12,455 P5026 INFO Monitor(max) STOP: 0.340639 !
2020-03-01 01:53:12,455 P5026 INFO Reduce learning rate on plateau: 0.000010
2020-03-01 01:53:12,455 P5026 INFO Early stopping at epoch=3
2020-03-01 01:53:12,455 P5026 INFO --- 3235/3235 batches finished ---
2020-03-01 01:53:12,547 P5026 INFO Train loss: 0.248085
2020-03-01 01:53:12,547 P5026 INFO Training finished.
2020-03-01 01:53:12,547 P5026 INFO Load best model: /home/XXX/benchmarks/Avazu/FwFM_avazu/avazu_x4_001_d45ad60e/FwFM_avazu_x4_003_fec49391_model.ckpt
2020-03-01 01:53:14,038 P5026 INFO ****** Train/validation evaluation ******
2020-03-01 01:58:12,790 P5026 INFO [Metrics] logloss: 0.330077 - AUC: 0.856027
2020-03-01 01:58:49,945 P5026 INFO [Metrics] logloss: 0.372556 - AUC: 0.792171
2020-03-01 01:58:50,071 P5026 INFO ******** Test evaluation ********
2020-03-01 01:58:50,071 P5026 INFO Loading data...
2020-03-01 01:58:50,071 P5026 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-03-01 01:58:50,655 P5026 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-03-01 01:58:50,655 P5026 INFO Loading test data done.
2020-03-01 01:59:27,673 P5026 INFO [Metrics] logloss: 0.372403 - AUC: 0.792534

```
