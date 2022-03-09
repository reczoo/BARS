## AFN_kkbox_x1

A hands-on guide to run the AFN model on the KKBox_x1 dataset.

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
Dataset ID: [KKBox_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/KKBox#KKBox_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [AFN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/AFN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFN_kkbox_x1_tuner_config_03](./AFN_kkbox_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFN_kkbox_x1
    nohup python run_expid.py --config ./AFN_kkbox_x1_tuner_config_03 --expid AFN_kkbox_x1_006_f1f7fac0 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.491040 | 0.842551  |


### Logs
```python
2022-03-11 07:56:54,619 P23226 INFO {
    "afn_activations": "relu",
    "afn_dropout": "0.2",
    "afn_hidden_units": "[1000, 1000, 1000]",
    "batch_norm": "True",
    "batch_size": "5000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_dropout": "0",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "5e-06",
    "ensemble_dnn": "False",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "5",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "logarithmic_neurons": "500",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "AFN",
    "model_id": "AFN_kkbox_x1_006_f1f7fac0",
    "model_root": "./KKBox/AFN_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/KKBox/KKBox_x1/test.csv",
    "train_data": "../data/KKBox/KKBox_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/KKBox/KKBox_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2022-03-11 07:56:54,620 P23226 INFO Set up feature encoder...
2022-03-11 07:56:54,620 P23226 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-11 07:56:56,610 P23226 INFO Total number of parameters: 77825143.
2022-03-11 07:56:56,610 P23226 INFO Loading data...
2022-03-11 07:56:56,610 P23226 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-11 07:56:56,990 P23226 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-11 07:56:57,194 P23226 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-11 07:56:57,211 P23226 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-11 07:56:57,211 P23226 INFO Loading train data done.
2022-03-11 07:57:00,840 P23226 INFO Start training: 1181 batches/epoch
2022-03-11 07:57:00,840 P23226 INFO ************ Epoch=1 start ************
2022-03-11 08:02:17,788 P23226 INFO [Metrics] logloss: 0.548030 - AUC: 0.793080
2022-03-11 08:02:17,788 P23226 INFO Save best model: monitor(max): 0.245049
2022-03-11 08:02:18,460 P23226 INFO --- 1181/1181 batches finished ---
2022-03-11 08:02:18,504 P23226 INFO Train loss: 0.583628
2022-03-11 08:02:18,504 P23226 INFO ************ Epoch=1 end ************
2022-03-11 08:07:36,168 P23226 INFO [Metrics] logloss: 0.527768 - AUC: 0.811567
2022-03-11 08:07:36,169 P23226 INFO Save best model: monitor(max): 0.283799
2022-03-11 08:07:36,603 P23226 INFO --- 1181/1181 batches finished ---
2022-03-11 08:07:36,659 P23226 INFO Train loss: 0.547982
2022-03-11 08:07:36,659 P23226 INFO ************ Epoch=2 end ************
2022-03-11 08:12:53,916 P23226 INFO [Metrics] logloss: 0.514424 - AUC: 0.823047
2022-03-11 08:12:53,916 P23226 INFO Save best model: monitor(max): 0.308623
2022-03-11 08:12:54,354 P23226 INFO --- 1181/1181 batches finished ---
2022-03-11 08:12:54,394 P23226 INFO Train loss: 0.532729
2022-03-11 08:12:54,394 P23226 INFO ************ Epoch=3 end ************
2022-03-11 08:18:11,599 P23226 INFO [Metrics] logloss: 0.502505 - AUC: 0.832310
2022-03-11 08:18:11,600 P23226 INFO Save best model: monitor(max): 0.329805
2022-03-11 08:18:12,025 P23226 INFO --- 1181/1181 batches finished ---
2022-03-11 08:18:12,074 P23226 INFO Train loss: 0.518311
2022-03-11 08:18:12,074 P23226 INFO ************ Epoch=4 end ************
2022-03-11 08:23:29,354 P23226 INFO [Metrics] logloss: 0.495220 - AUC: 0.838225
2022-03-11 08:23:29,354 P23226 INFO Save best model: monitor(max): 0.343005
2022-03-11 08:23:29,784 P23226 INFO --- 1181/1181 batches finished ---
2022-03-11 08:23:29,821 P23226 INFO Train loss: 0.504988
2022-03-11 08:23:29,821 P23226 INFO ************ Epoch=5 end ************
2022-03-11 08:28:46,835 P23226 INFO [Metrics] logloss: 0.491771 - AUC: 0.841645
2022-03-11 08:28:46,835 P23226 INFO Save best model: monitor(max): 0.349874
2022-03-11 08:28:47,250 P23226 INFO --- 1181/1181 batches finished ---
2022-03-11 08:28:47,295 P23226 INFO Train loss: 0.492923
2022-03-11 08:28:47,295 P23226 INFO ************ Epoch=6 end ************
2022-03-11 08:34:02,500 P23226 INFO [Metrics] logloss: 0.490689 - AUC: 0.842807
2022-03-11 08:34:02,501 P23226 INFO Save best model: monitor(max): 0.352118
2022-03-11 08:34:02,929 P23226 INFO --- 1181/1181 batches finished ---
2022-03-11 08:34:02,975 P23226 INFO Train loss: 0.481557
2022-03-11 08:34:02,975 P23226 INFO ************ Epoch=7 end ************
2022-03-11 08:39:18,192 P23226 INFO [Metrics] logloss: 0.493114 - AUC: 0.842856
2022-03-11 08:39:18,193 P23226 INFO Monitor(max) STOP: 0.349742 !
2022-03-11 08:39:18,193 P23226 INFO Reduce learning rate on plateau: 0.000100
2022-03-11 08:39:18,193 P23226 INFO --- 1181/1181 batches finished ---
2022-03-11 08:39:18,237 P23226 INFO Train loss: 0.470085
2022-03-11 08:39:18,237 P23226 INFO ************ Epoch=8 end ************
2022-03-11 08:44:33,654 P23226 INFO [Metrics] logloss: 0.530689 - AUC: 0.839448
2022-03-11 08:44:33,655 P23226 INFO Monitor(max) STOP: 0.308758 !
2022-03-11 08:44:33,655 P23226 INFO Reduce learning rate on plateau: 0.000010
2022-03-11 08:44:33,655 P23226 INFO Early stopping at epoch=9
2022-03-11 08:44:33,655 P23226 INFO --- 1181/1181 batches finished ---
2022-03-11 08:44:33,693 P23226 INFO Train loss: 0.410334
2022-03-11 08:44:33,693 P23226 INFO Training finished.
2022-03-11 08:44:33,693 P23226 INFO Load best model: /cache/FuxiCTR/benchmarks/KKBox/AFN_kkbox_x1/kkbox_x1_227d337d/AFN_kkbox_x1_006_f1f7fac0_model.ckpt
2022-03-11 08:44:34,032 P23226 INFO ****** Validation evaluation ******
2022-03-11 08:44:46,480 P23226 INFO [Metrics] logloss: 0.490689 - AUC: 0.842807
2022-03-11 08:44:46,538 P23226 INFO ******** Test evaluation ********
2022-03-11 08:44:46,538 P23226 INFO Loading data...
2022-03-11 08:44:46,538 P23226 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-11 08:44:46,594 P23226 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-11 08:44:46,594 P23226 INFO Loading test data done.
2022-03-11 08:44:58,946 P23226 INFO [Metrics] logloss: 0.491040 - AUC: 0.842551

```
