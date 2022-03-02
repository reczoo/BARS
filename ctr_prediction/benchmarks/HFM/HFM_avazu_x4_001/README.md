## HFM_avazu_x4_001

A hands-on guide to run the HFM model on the Avazu_x4_001 dataset.

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
Dataset ID: [Avazu_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [HFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/HFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HFM_avazu_x4_tuner_config_02](./HFM_avazu_x4_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HFM_avazu_x4_001
    nohup python run_expid.py --config ./HFM_avazu_x4_tuner_config_02 --expid HFM_avazu_x4_009_8053d0ee --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.375680 | 0.787866  |


### Logs
```python
2020-07-17 04:36:03,813 P38814 INFO {
    "batch_norm": "False",
    "batch_size": "5000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-07",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[64, 64, 64]",
    "interaction_type": "circular_correlation",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "HFM",
    "model_id": "HFM_avazu_x4_3bbbc4c9_009_5fa16550",
    "model_root": "./Avazu/HFM_avazu/min2/",
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
    "test_data": "../data/Avazu/avazu_x4_3bbbc4c9/test.h5",
    "train_data": "../data/Avazu/avazu_x4_3bbbc4c9/train.h5",
    "use_dnn": "False",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/avazu_x4_3bbbc4c9/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-17 04:36:03,814 P38814 INFO Set up feature encoder...
2020-07-17 04:36:03,814 P38814 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-07-17 04:36:03,814 P38814 INFO Loading data...
2020-07-17 04:36:03,816 P38814 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-07-17 04:36:06,447 P38814 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-07-17 04:36:08,029 P38814 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-07-17 04:36:08,145 P38814 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-07-17 04:36:08,145 P38814 INFO Loading train data done.
2020-07-17 04:36:12,313 P38814 INFO **** Start training: 6469 batches/epoch ****
2020-07-17 04:54:40,207 P38814 INFO [Metrics] logloss: 0.377465 - AUC: 0.783979
2020-07-17 04:54:40,208 P38814 INFO Save best model: monitor(max): 0.406514
2020-07-17 04:54:40,437 P38814 INFO --- 6469/6469 batches finished ---
2020-07-17 04:54:40,489 P38814 INFO Train loss: 0.387310
2020-07-17 04:54:40,489 P38814 INFO ************ Epoch=1 end ************
2020-07-17 05:13:07,488 P38814 INFO [Metrics] logloss: 0.375697 - AUC: 0.787760
2020-07-17 05:13:07,491 P38814 INFO Save best model: monitor(max): 0.412063
2020-07-17 05:13:07,950 P38814 INFO --- 6469/6469 batches finished ---
2020-07-17 05:13:08,004 P38814 INFO Train loss: 0.367259
2020-07-17 05:13:08,004 P38814 INFO ************ Epoch=2 end ************
2020-07-17 05:31:36,665 P38814 INFO [Metrics] logloss: 0.376670 - AUC: 0.787865
2020-07-17 05:31:36,666 P38814 INFO Monitor(max) STOP: 0.411194 !
2020-07-17 05:31:36,666 P38814 INFO Reduce learning rate on plateau: 0.000100
2020-07-17 05:31:36,666 P38814 INFO --- 6469/6469 batches finished ---
2020-07-17 05:31:36,720 P38814 INFO Train loss: 0.357512
2020-07-17 05:31:36,720 P38814 INFO ************ Epoch=3 end ************
2020-07-17 05:50:05,042 P38814 INFO [Metrics] logloss: 0.388555 - AUC: 0.782669
2020-07-17 05:50:05,046 P38814 INFO Monitor(max) STOP: 0.394113 !
2020-07-17 05:50:05,046 P38814 INFO Reduce learning rate on plateau: 0.000010
2020-07-17 05:50:05,046 P38814 INFO Early stopping at epoch=4
2020-07-17 05:50:05,046 P38814 INFO --- 6469/6469 batches finished ---
2020-07-17 05:50:05,099 P38814 INFO Train loss: 0.327623
2020-07-17 05:50:05,099 P38814 INFO Training finished.
2020-07-17 05:50:05,099 P38814 INFO Load best model: /home/XXX/benchmarks/Avazu/HFM_avazu/min2/avazu_x4_3bbbc4c9/HFM_avazu_x4_3bbbc4c9_009_5fa16550_model.ckpt
2020-07-17 05:50:05,437 P38814 INFO ****** Train/validation evaluation ******
2020-07-17 05:50:42,191 P38814 INFO [Metrics] logloss: 0.375697 - AUC: 0.787760
2020-07-17 05:50:42,300 P38814 INFO ******** Test evaluation ********
2020-07-17 05:50:42,300 P38814 INFO Loading data...
2020-07-17 05:50:42,300 P38814 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-07-17 05:50:42,749 P38814 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-07-17 05:50:42,749 P38814 INFO Loading test data done.
2020-07-17 05:51:19,212 P38814 INFO [Metrics] logloss: 0.375680 - AUC: 0.787866

```
