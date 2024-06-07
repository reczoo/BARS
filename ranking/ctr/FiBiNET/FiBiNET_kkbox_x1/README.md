## FiBiNET_kkbox_x1

A hands-on guide to run the FiBiNET model on the KKBox_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

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

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FiBiNET](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FiBiNET.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FiBiNET_kkbox_x1_tuner_config_03](./FiBiNET_kkbox_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FiBiNET_kkbox_x1
    nohup python run_expid.py --config ./FiBiNET_kkbox_x1_tuner_config_03 --expid FiBiNET_kkbox_x1_007_f8ea597b --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.481384 | 0.849932  |


### Logs
```python
2022-03-11 19:56:25,271 P8190 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "bilinear_type": "field_interaction",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "6",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "FiBiNET",
    "model_id": "FiBiNET_kkbox_x1_007_f8ea597b",
    "model_root": "./KKBox/FiBiNET_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.4",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "3",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/KKBox/KKBox_x1/test.csv",
    "train_data": "../data/KKBox/KKBox_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/KKBox/KKBox_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2022-03-11 19:56:25,272 P8190 INFO Set up feature encoder...
2022-03-11 19:56:25,273 P8190 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-11 19:56:26,946 P8190 INFO Total number of parameters: 34148920.
2022-03-11 19:56:26,946 P8190 INFO Loading data...
2022-03-11 19:56:26,947 P8190 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-11 19:56:27,346 P8190 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-11 19:56:27,541 P8190 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-11 19:56:27,558 P8190 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-11 19:56:27,558 P8190 INFO Loading train data done.
2022-03-11 19:56:32,698 P8190 INFO Start training: 591 batches/epoch
2022-03-11 19:56:32,698 P8190 INFO ************ Epoch=1 start ************
2022-03-11 20:02:03,004 P8190 INFO [Metrics] logloss: 0.555924 - AUC: 0.786392
2022-03-11 20:02:03,007 P8190 INFO Save best model: monitor(max): 0.230468
2022-03-11 20:02:03,430 P8190 INFO --- 591/591 batches finished ---
2022-03-11 20:02:03,476 P8190 INFO Train loss: 0.630415
2022-03-11 20:02:03,477 P8190 INFO ************ Epoch=1 end ************
2022-03-11 20:07:32,404 P8190 INFO [Metrics] logloss: 0.539907 - AUC: 0.801394
2022-03-11 20:07:32,405 P8190 INFO Save best model: monitor(max): 0.261488
2022-03-11 20:07:32,674 P8190 INFO --- 591/591 batches finished ---
2022-03-11 20:07:32,715 P8190 INFO Train loss: 0.602419
2022-03-11 20:07:32,716 P8190 INFO ************ Epoch=2 end ************
2022-03-11 20:13:01,735 P8190 INFO [Metrics] logloss: 0.531643 - AUC: 0.808471
2022-03-11 20:13:01,736 P8190 INFO Save best model: monitor(max): 0.276827
2022-03-11 20:13:01,979 P8190 INFO --- 591/591 batches finished ---
2022-03-11 20:13:02,023 P8190 INFO Train loss: 0.592730
2022-03-11 20:13:02,023 P8190 INFO ************ Epoch=3 end ************
2022-03-11 20:18:31,297 P8190 INFO [Metrics] logloss: 0.526882 - AUC: 0.812571
2022-03-11 20:18:31,298 P8190 INFO Save best model: monitor(max): 0.285689
2022-03-11 20:18:31,556 P8190 INFO --- 591/591 batches finished ---
2022-03-11 20:18:31,597 P8190 INFO Train loss: 0.589278
2022-03-11 20:18:31,598 P8190 INFO ************ Epoch=4 end ************
2022-03-11 20:24:00,722 P8190 INFO [Metrics] logloss: 0.524033 - AUC: 0.815113
2022-03-11 20:24:00,723 P8190 INFO Save best model: monitor(max): 0.291080
2022-03-11 20:24:00,966 P8190 INFO --- 591/591 batches finished ---
2022-03-11 20:24:01,010 P8190 INFO Train loss: 0.587049
2022-03-11 20:24:01,011 P8190 INFO ************ Epoch=5 end ************
2022-03-11 20:29:29,970 P8190 INFO [Metrics] logloss: 0.521065 - AUC: 0.817680
2022-03-11 20:29:29,971 P8190 INFO Save best model: monitor(max): 0.296614
2022-03-11 20:29:30,228 P8190 INFO --- 591/591 batches finished ---
2022-03-11 20:29:30,273 P8190 INFO Train loss: 0.584920
2022-03-11 20:29:30,273 P8190 INFO ************ Epoch=6 end ************
2022-03-11 20:34:59,199 P8190 INFO [Metrics] logloss: 0.519752 - AUC: 0.819808
2022-03-11 20:34:59,200 P8190 INFO Save best model: monitor(max): 0.300056
2022-03-11 20:34:59,467 P8190 INFO --- 591/591 batches finished ---
2022-03-11 20:34:59,512 P8190 INFO Train loss: 0.583726
2022-03-11 20:34:59,513 P8190 INFO ************ Epoch=7 end ************
2022-03-11 20:40:28,507 P8190 INFO [Metrics] logloss: 0.517111 - AUC: 0.820747
2022-03-11 20:40:28,508 P8190 INFO Save best model: monitor(max): 0.303636
2022-03-11 20:40:28,705 P8190 INFO --- 591/591 batches finished ---
2022-03-11 20:40:28,747 P8190 INFO Train loss: 0.582261
2022-03-11 20:40:28,748 P8190 INFO ************ Epoch=8 end ************
2022-03-11 20:45:57,446 P8190 INFO [Metrics] logloss: 0.515471 - AUC: 0.822165
2022-03-11 20:45:57,447 P8190 INFO Save best model: monitor(max): 0.306694
2022-03-11 20:45:57,611 P8190 INFO --- 591/591 batches finished ---
2022-03-11 20:45:57,655 P8190 INFO Train loss: 0.581280
2022-03-11 20:45:57,656 P8190 INFO ************ Epoch=9 end ************
2022-03-11 20:51:11,097 P8190 INFO [Metrics] logloss: 0.514210 - AUC: 0.823336
2022-03-11 20:51:11,098 P8190 INFO Save best model: monitor(max): 0.309126
2022-03-11 20:51:11,266 P8190 INFO --- 591/591 batches finished ---
2022-03-11 20:51:11,311 P8190 INFO Train loss: 0.579849
2022-03-11 20:51:11,311 P8190 INFO ************ Epoch=10 end ************
2022-03-11 20:56:36,875 P8190 INFO [Metrics] logloss: 0.513116 - AUC: 0.824187
2022-03-11 20:56:36,876 P8190 INFO Save best model: monitor(max): 0.311071
2022-03-11 20:56:37,057 P8190 INFO --- 591/591 batches finished ---
2022-03-11 20:56:37,103 P8190 INFO Train loss: 0.578787
2022-03-11 20:56:37,103 P8190 INFO ************ Epoch=11 end ************
2022-03-11 21:02:03,451 P8190 INFO [Metrics] logloss: 0.511919 - AUC: 0.825107
2022-03-11 21:02:03,451 P8190 INFO Save best model: monitor(max): 0.313187
2022-03-11 21:02:03,621 P8190 INFO --- 591/591 batches finished ---
2022-03-11 21:02:03,666 P8190 INFO Train loss: 0.578212
2022-03-11 21:02:03,666 P8190 INFO ************ Epoch=12 end ************
2022-03-11 21:07:30,144 P8190 INFO [Metrics] logloss: 0.510976 - AUC: 0.825969
2022-03-11 21:07:30,145 P8190 INFO Save best model: monitor(max): 0.314993
2022-03-11 21:07:30,312 P8190 INFO --- 591/591 batches finished ---
2022-03-11 21:07:30,356 P8190 INFO Train loss: 0.577297
2022-03-11 21:07:30,356 P8190 INFO ************ Epoch=13 end ************
2022-03-11 21:12:56,096 P8190 INFO [Metrics] logloss: 0.510019 - AUC: 0.826629
2022-03-11 21:12:56,097 P8190 INFO Save best model: monitor(max): 0.316610
2022-03-11 21:12:56,267 P8190 INFO --- 591/591 batches finished ---
2022-03-11 21:12:56,309 P8190 INFO Train loss: 0.576684
2022-03-11 21:12:56,309 P8190 INFO ************ Epoch=14 end ************
2022-03-11 21:18:22,669 P8190 INFO [Metrics] logloss: 0.509420 - AUC: 0.827063
2022-03-11 21:18:22,670 P8190 INFO Save best model: monitor(max): 0.317643
2022-03-11 21:18:22,844 P8190 INFO --- 591/591 batches finished ---
2022-03-11 21:18:22,885 P8190 INFO Train loss: 0.575459
2022-03-11 21:18:22,886 P8190 INFO ************ Epoch=15 end ************
2022-03-11 21:23:49,321 P8190 INFO [Metrics] logloss: 0.508688 - AUC: 0.827770
2022-03-11 21:23:49,321 P8190 INFO Save best model: monitor(max): 0.319082
2022-03-11 21:23:49,505 P8190 INFO --- 591/591 batches finished ---
2022-03-11 21:23:49,548 P8190 INFO Train loss: 0.575403
2022-03-11 21:23:49,549 P8190 INFO ************ Epoch=16 end ************
2022-03-11 21:29:15,803 P8190 INFO [Metrics] logloss: 0.508490 - AUC: 0.828031
2022-03-11 21:29:15,804 P8190 INFO Save best model: monitor(max): 0.319540
2022-03-11 21:29:15,989 P8190 INFO --- 591/591 batches finished ---
2022-03-11 21:29:16,029 P8190 INFO Train loss: 0.574762
2022-03-11 21:29:16,029 P8190 INFO ************ Epoch=17 end ************
2022-03-11 21:34:42,218 P8190 INFO [Metrics] logloss: 0.507740 - AUC: 0.828515
2022-03-11 21:34:42,219 P8190 INFO Save best model: monitor(max): 0.320775
2022-03-11 21:34:42,381 P8190 INFO --- 591/591 batches finished ---
2022-03-11 21:34:42,423 P8190 INFO Train loss: 0.574480
2022-03-11 21:34:42,423 P8190 INFO ************ Epoch=18 end ************
2022-03-11 21:40:08,563 P8190 INFO [Metrics] logloss: 0.507450 - AUC: 0.829142
2022-03-11 21:40:08,564 P8190 INFO Save best model: monitor(max): 0.321692
2022-03-11 21:40:08,816 P8190 INFO --- 591/591 batches finished ---
2022-03-11 21:40:08,856 P8190 INFO Train loss: 0.573746
2022-03-11 21:40:08,857 P8190 INFO ************ Epoch=19 end ************
2022-03-11 21:43:02,839 P8190 INFO [Metrics] logloss: 0.506540 - AUC: 0.829237
2022-03-11 21:43:02,839 P8190 INFO Save best model: monitor(max): 0.322697
2022-03-11 21:43:03,062 P8190 INFO --- 591/591 batches finished ---
2022-03-11 21:43:03,119 P8190 INFO Train loss: 0.573562
2022-03-11 21:43:03,120 P8190 INFO ************ Epoch=20 end ************
2022-03-11 21:45:28,093 P8190 INFO [Metrics] logloss: 0.506903 - AUC: 0.829967
2022-03-11 21:45:28,094 P8190 INFO Save best model: monitor(max): 0.323064
2022-03-11 21:45:28,280 P8190 INFO --- 591/591 batches finished ---
2022-03-11 21:45:28,320 P8190 INFO Train loss: 0.573224
2022-03-11 21:45:28,321 P8190 INFO ************ Epoch=21 end ************
2022-03-11 21:47:53,187 P8190 INFO [Metrics] logloss: 0.505112 - AUC: 0.830550
2022-03-11 21:47:53,188 P8190 INFO Save best model: monitor(max): 0.325438
2022-03-11 21:47:53,377 P8190 INFO --- 591/591 batches finished ---
2022-03-11 21:47:53,416 P8190 INFO Train loss: 0.572749
2022-03-11 21:47:53,416 P8190 INFO ************ Epoch=22 end ************
2022-03-11 21:50:18,155 P8190 INFO [Metrics] logloss: 0.505298 - AUC: 0.830382
2022-03-11 21:50:18,156 P8190 INFO Monitor(max) STOP: 0.325084 !
2022-03-11 21:50:18,156 P8190 INFO Reduce learning rate on plateau: 0.000100
2022-03-11 21:50:18,156 P8190 INFO --- 591/591 batches finished ---
2022-03-11 21:50:18,207 P8190 INFO Train loss: 0.572262
2022-03-11 21:50:18,208 P8190 INFO ************ Epoch=23 end ************
2022-03-11 21:52:43,611 P8190 INFO [Metrics] logloss: 0.486691 - AUC: 0.845322
2022-03-11 21:52:43,612 P8190 INFO Save best model: monitor(max): 0.358631
2022-03-11 21:52:43,821 P8190 INFO --- 591/591 batches finished ---
2022-03-11 21:52:43,860 P8190 INFO Train loss: 0.512430
2022-03-11 21:52:43,861 P8190 INFO ************ Epoch=24 end ************
2022-03-11 21:55:08,746 P8190 INFO [Metrics] logloss: 0.482724 - AUC: 0.848677
2022-03-11 21:55:08,747 P8190 INFO Save best model: monitor(max): 0.365953
2022-03-11 21:55:08,939 P8190 INFO --- 591/591 batches finished ---
2022-03-11 21:55:08,994 P8190 INFO Train loss: 0.483625
2022-03-11 21:55:08,994 P8190 INFO ************ Epoch=25 end ************
2022-03-11 21:57:33,933 P8190 INFO [Metrics] logloss: 0.481706 - AUC: 0.849737
2022-03-11 21:57:33,933 P8190 INFO Save best model: monitor(max): 0.368031
2022-03-11 21:57:34,140 P8190 INFO --- 591/591 batches finished ---
2022-03-11 21:57:34,179 P8190 INFO Train loss: 0.471627
2022-03-11 21:57:34,179 P8190 INFO ************ Epoch=26 end ************
2022-03-11 21:59:58,987 P8190 INFO [Metrics] logloss: 0.482623 - AUC: 0.849886
2022-03-11 21:59:58,987 P8190 INFO Monitor(max) STOP: 0.367263 !
2022-03-11 21:59:58,987 P8190 INFO Reduce learning rate on plateau: 0.000010
2022-03-11 21:59:58,987 P8190 INFO --- 591/591 batches finished ---
2022-03-11 21:59:59,024 P8190 INFO Train loss: 0.463315
2022-03-11 21:59:59,024 P8190 INFO ************ Epoch=27 end ************
2022-03-11 22:02:23,992 P8190 INFO [Metrics] logloss: 0.492403 - AUC: 0.848673
2022-03-11 22:02:23,993 P8190 INFO Monitor(max) STOP: 0.356270 !
2022-03-11 22:02:23,994 P8190 INFO Reduce learning rate on plateau: 0.000001
2022-03-11 22:02:23,994 P8190 INFO Early stopping at epoch=28
2022-03-11 22:02:23,994 P8190 INFO --- 591/591 batches finished ---
2022-03-11 22:02:24,033 P8190 INFO Train loss: 0.435016
2022-03-11 22:02:24,034 P8190 INFO Training finished.
2022-03-11 22:02:24,034 P8190 INFO Load best model: /cache/FuxiCTR/benchmarks/KKBox/FiBiNET_kkbox_x1/kkbox_x1_227d337d/FiBiNET_kkbox_x1_007_f8ea597b_model.ckpt
2022-03-11 22:02:24,261 P8190 INFO ****** Validation evaluation ******
2022-03-11 22:02:29,824 P8190 INFO [Metrics] logloss: 0.481706 - AUC: 0.849737
2022-03-11 22:02:29,878 P8190 INFO ******** Test evaluation ********
2022-03-11 22:02:29,878 P8190 INFO Loading data...
2022-03-11 22:02:29,878 P8190 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-11 22:02:29,938 P8190 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-11 22:02:29,938 P8190 INFO Loading test data done.
2022-03-11 22:02:35,403 P8190 INFO [Metrics] logloss: 0.481384 - AUC: 0.849932

```
