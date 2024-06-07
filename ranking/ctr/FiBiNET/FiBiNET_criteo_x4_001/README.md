## FiBiNET_criteo_x4_001

A hands-on guide to run the FiBiNET model on the Criteo_x4_001 dataset.

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
Dataset ID: [Criteo_x4](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x4). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FiBiNET](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FiBiNET.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FiBiNET_criteo_x4_tuner_config_05](./FiBiNET_criteo_x4_tuner_config_05). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FiBiNET_criteo_x4_001
    nohup python run_expid.py --config ./FiBiNET_criteo_x4_tuner_config_05 --expid FiBiNET_criteo_x4_004_73513faa --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.438736 | 0.813097  |


### Logs
```python
2021-09-14 04:20:51,248 P23284 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "bilinear_type": "field_interaction",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_9ea3bdfc",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-06",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[2000, 2000, 2000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FiBiNET",
    "model_id": "FiBiNET_criteo_x4_9ea3bdfc_004_73513faa",
    "model_root": "./Criteo/FiBiNET_criteo_x4_001/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "3",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/criteo_x4_9ea3bdfc/test.h5",
    "train_data": "../data/Criteo/criteo_x4_9ea3bdfc/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_9ea3bdfc/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2021-09-14 04:20:51,249 P23284 INFO Set up feature encoder...
2021-09-14 04:20:51,249 P23284 INFO Load feature_map from json: ../data/Criteo/criteo_x4_9ea3bdfc/feature_map.json
2021-09-14 04:20:52,815 P23284 INFO Total number of parameters: 71104747.
2021-09-14 04:20:52,815 P23284 INFO Loading data...
2021-09-14 04:20:52,817 P23284 INFO Loading data from h5: ../data/Criteo/criteo_x4_9ea3bdfc/train.h5
2021-09-14 04:20:57,725 P23284 INFO Loading data from h5: ../data/Criteo/criteo_x4_9ea3bdfc/valid.h5
2021-09-14 04:20:59,979 P23284 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2021-09-14 04:21:00,118 P23284 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2021-09-14 04:21:00,118 P23284 INFO Loading train data done.
2021-09-14 04:21:03,012 P23284 INFO Start training: 3668 batches/epoch
2021-09-14 04:21:03,013 P23284 INFO ************ Epoch=1 start ************
2021-09-14 05:30:58,488 P23284 INFO [Metrics] logloss: 0.441152 - AUC: 0.810529
2021-09-14 05:30:58,492 P23284 INFO Save best model: monitor(max): 0.369377
2021-09-14 05:30:58,792 P23284 INFO --- 3668/3668 batches finished ---
2021-09-14 05:30:58,840 P23284 INFO Train loss: 0.451255
2021-09-14 05:30:58,841 P23284 INFO ************ Epoch=1 end ************
2021-09-14 06:42:19,439 P23284 INFO [Metrics] logloss: 0.439118 - AUC: 0.812614
2021-09-14 06:42:19,440 P23284 INFO Save best model: monitor(max): 0.373495
2021-09-14 06:42:20,087 P23284 INFO --- 3668/3668 batches finished ---
2021-09-14 06:42:20,140 P23284 INFO Train loss: 0.441994
2021-09-14 06:42:20,142 P23284 INFO ************ Epoch=2 end ************
2021-09-14 07:53:44,859 P23284 INFO [Metrics] logloss: 0.441423 - AUC: 0.811169
2021-09-14 07:53:44,861 P23284 INFO Monitor(max) STOP: 0.369746 !
2021-09-14 07:53:44,861 P23284 INFO Reduce learning rate on plateau: 0.000100
2021-09-14 07:53:44,861 P23284 INFO --- 3668/3668 batches finished ---
2021-09-14 07:53:44,930 P23284 INFO Train loss: 0.436822
2021-09-14 07:53:44,932 P23284 INFO ************ Epoch=3 end ************
2021-09-14 09:03:58,529 P23284 INFO [Metrics] logloss: 0.475763 - AUC: 0.791703
2021-09-14 09:03:58,533 P23284 INFO Monitor(max) STOP: 0.315940 !
2021-09-14 09:03:58,533 P23284 INFO Reduce learning rate on plateau: 0.000010
2021-09-14 09:03:58,533 P23284 INFO Early stopping at epoch=4
2021-09-14 09:03:58,533 P23284 INFO --- 3668/3668 batches finished ---
2021-09-14 09:03:58,597 P23284 INFO Train loss: 0.392997
2021-09-14 09:03:58,599 P23284 INFO Training finished.
2021-09-14 09:03:58,600 P23284 INFO Load best model: /home/XXX/benchmarks/Criteo/FiBiNET_criteo_x4_001/criteo_x4_9ea3bdfc/FiBiNET_criteo_x4_9ea3bdfc_004_73513faa_model.ckpt
2021-09-14 09:03:59,431 P23284 INFO ****** Train/validation evaluation ******
2021-09-14 09:06:24,647 P23284 INFO [Metrics] logloss: 0.439118 - AUC: 0.812614
2021-09-14 09:06:24,684 P23284 INFO ******** Test evaluation ********
2021-09-14 09:06:24,684 P23284 INFO Loading data...
2021-09-14 09:06:24,685 P23284 INFO Loading data from h5: ../data/Criteo/criteo_x4_9ea3bdfc/test.h5
2021-09-14 09:06:25,466 P23284 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2021-09-14 09:06:25,466 P23284 INFO Loading test data done.
2021-09-14 09:08:55,523 P23284 INFO [Metrics] logloss: 0.438736 - AUC: 0.813097

```
