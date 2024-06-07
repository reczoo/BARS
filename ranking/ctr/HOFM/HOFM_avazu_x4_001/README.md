## HOFM_avazu_x4_001

A hands-on guide to run the HOFM model on the Avazu_x4_001 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [HOFM](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/HOFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HOFM_avazu_x4_tuner_config_02](./HOFM_avazu_x4_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HOFM_avazu_x4_001
    nohup python run_expid.py --config ./HOFM_avazu_x4_tuner_config_02 --expid HOFM_avazu_x4_001_a9f41af5 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.375365 | 0.789067  |


### Logs
```python
2020-07-16 08:12:48,869 P39109 INFO {
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "embedding_dim": "[16, 16]",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "HOFM",
    "model_id": "HOFM_avazu_x4_3bbbc4c9_001_5b17eb2d",
    "model_root": "./Avazu/HOFM_avazu/min2/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "order": "3",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-07",
    "reuse_embedding": "False",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/avazu_x4_3bbbc4c9/test.h5",
    "train_data": "../data/Avazu/avazu_x4_3bbbc4c9/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/avazu_x4_3bbbc4c9/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-16 08:12:48,870 P39109 INFO Set up feature encoder...
2020-07-16 08:12:48,870 P39109 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-07-16 08:12:48,870 P39109 INFO Loading data...
2020-07-16 08:12:48,873 P39109 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-07-16 08:12:51,872 P39109 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-07-16 08:12:53,237 P39109 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-07-16 08:12:53,353 P39109 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-07-16 08:12:53,353 P39109 INFO Loading train data done.
2020-07-16 08:12:58,921 P39109 INFO **** Start training: 3235 batches/epoch ****
2020-07-16 08:34:02,585 P39109 INFO [Metrics] logloss: 0.376450 - AUC: 0.785862
2020-07-16 08:34:02,591 P39109 INFO Save best model: monitor(max): 0.409412
2020-07-16 08:34:03,082 P39109 INFO --- 3235/3235 batches finished ---
2020-07-16 08:34:03,127 P39109 INFO Train loss: 0.389044
2020-07-16 08:34:03,127 P39109 INFO ************ Epoch=1 end ************
2020-07-16 08:55:02,816 P39109 INFO [Metrics] logloss: 0.375422 - AUC: 0.788964
2020-07-16 08:55:02,821 P39109 INFO Save best model: monitor(max): 0.413542
2020-07-16 08:55:03,911 P39109 INFO --- 3235/3235 batches finished ---
2020-07-16 08:55:03,956 P39109 INFO Train loss: 0.362889
2020-07-16 08:55:03,956 P39109 INFO ************ Epoch=2 end ************
2020-07-16 09:16:04,777 P39109 INFO [Metrics] logloss: 0.377995 - AUC: 0.787760
2020-07-16 09:16:04,783 P39109 INFO Monitor(max) STOP: 0.409765 !
2020-07-16 09:16:04,783 P39109 INFO Reduce learning rate on plateau: 0.000100
2020-07-16 09:16:04,783 P39109 INFO --- 3235/3235 batches finished ---
2020-07-16 09:16:04,830 P39109 INFO Train loss: 0.347860
2020-07-16 09:16:04,830 P39109 INFO ************ Epoch=3 end ************
2020-07-16 09:37:03,272 P39109 INFO [Metrics] logloss: 0.387875 - AUC: 0.784775
2020-07-16 09:37:03,278 P39109 INFO Monitor(max) STOP: 0.396900 !
2020-07-16 09:37:03,278 P39109 INFO Reduce learning rate on plateau: 0.000010
2020-07-16 09:37:03,278 P39109 INFO Early stopping at epoch=4
2020-07-16 09:37:03,278 P39109 INFO --- 3235/3235 batches finished ---
2020-07-16 09:37:03,321 P39109 INFO Train loss: 0.318845
2020-07-16 09:37:03,321 P39109 INFO Training finished.
2020-07-16 09:37:03,321 P39109 INFO Load best model: /home/XXX/benchmarks/Avazu/HOFM_avazu/min2/avazu_x4_3bbbc4c9/HOFM_avazu_x4_3bbbc4c9_001_5b17eb2d_model.ckpt
2020-07-16 09:37:04,005 P39109 INFO ****** Train/validation evaluation ******
2020-07-16 09:37:41,494 P39109 INFO [Metrics] logloss: 0.375422 - AUC: 0.788964
2020-07-16 09:37:41,601 P39109 INFO ******** Test evaluation ********
2020-07-16 09:37:41,601 P39109 INFO Loading data...
2020-07-16 09:37:41,601 P39109 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-07-16 09:37:42,063 P39109 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-07-16 09:37:42,063 P39109 INFO Loading test data done.
2020-07-16 09:38:20,306 P39109 INFO [Metrics] logloss: 0.375365 - AUC: 0.789067

```
