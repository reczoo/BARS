## LorentzFM_kkbox_x1

A hands-on guide to run the LorentzFM model on the KKBox_x1 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [LorentzFM](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/LorentzFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [LorentzFM_kkbox_x1_tuner_config_02](./LorentzFM_kkbox_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd LorentzFM_kkbox_x1
    nohup python run_expid.py --config ./LorentzFM_kkbox_x1_tuner_config_02 --expid LorentzFM_kkbox_x1_003_0455bdef --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.518848 | 0.820207  |


### Logs
```python
2022-03-10 10:17:07,540 P41332 INFO {
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "2",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "LorentzFM",
    "model_id": "LorentzFM_kkbox_x1_003_0455bdef",
    "model_root": "./KKBox/LorentzFM_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-06",
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
2022-03-10 10:17:07,541 P41332 INFO Set up feature encoder...
2022-03-10 10:17:07,542 P41332 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-10 10:17:08,469 P41332 INFO Total number of parameters: 11807616.
2022-03-10 10:17:08,469 P41332 INFO Loading data...
2022-03-10 10:17:08,470 P41332 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-10 10:17:08,832 P41332 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-10 10:17:09,051 P41332 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-10 10:17:09,073 P41332 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-10 10:17:09,073 P41332 INFO Loading train data done.
2022-03-10 10:17:13,055 P41332 INFO Start training: 591 batches/epoch
2022-03-10 10:17:13,056 P41332 INFO ************ Epoch=1 start ************
2022-03-10 10:17:47,611 P41332 INFO [Metrics] logloss: 0.561536 - AUC: 0.781421
2022-03-10 10:17:47,616 P41332 INFO Save best model: monitor(max): 0.219885
2022-03-10 10:17:47,942 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:17:47,981 P41332 INFO Train loss: 0.589643
2022-03-10 10:17:47,981 P41332 INFO ************ Epoch=1 end ************
2022-03-10 10:18:22,008 P41332 INFO [Metrics] logloss: 0.546223 - AUC: 0.795819
2022-03-10 10:18:22,009 P41332 INFO Save best model: monitor(max): 0.249596
2022-03-10 10:18:22,066 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:18:22,108 P41332 INFO Train loss: 0.550875
2022-03-10 10:18:22,108 P41332 INFO ************ Epoch=2 end ************
2022-03-10 10:18:56,662 P41332 INFO [Metrics] logloss: 0.538973 - AUC: 0.802626
2022-03-10 10:18:56,663 P41332 INFO Save best model: monitor(max): 0.263653
2022-03-10 10:18:56,730 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:18:56,772 P41332 INFO Train loss: 0.536165
2022-03-10 10:18:56,772 P41332 INFO ************ Epoch=3 end ************
2022-03-10 10:19:31,345 P41332 INFO [Metrics] logloss: 0.534488 - AUC: 0.806673
2022-03-10 10:19:31,346 P41332 INFO Save best model: monitor(max): 0.272185
2022-03-10 10:19:31,406 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:19:31,447 P41332 INFO Train loss: 0.526991
2022-03-10 10:19:31,447 P41332 INFO ************ Epoch=4 end ************
2022-03-10 10:20:05,741 P41332 INFO [Metrics] logloss: 0.531310 - AUC: 0.809495
2022-03-10 10:20:05,742 P41332 INFO Save best model: monitor(max): 0.278185
2022-03-10 10:20:05,794 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:20:05,847 P41332 INFO Train loss: 0.520549
2022-03-10 10:20:05,847 P41332 INFO ************ Epoch=5 end ************
2022-03-10 10:20:41,958 P41332 INFO [Metrics] logloss: 0.529257 - AUC: 0.811305
2022-03-10 10:20:41,959 P41332 INFO Save best model: monitor(max): 0.282048
2022-03-10 10:20:42,018 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:20:42,064 P41332 INFO Train loss: 0.516020
2022-03-10 10:20:42,064 P41332 INFO ************ Epoch=6 end ************
2022-03-10 10:21:16,199 P41332 INFO [Metrics] logloss: 0.527650 - AUC: 0.812612
2022-03-10 10:21:16,199 P41332 INFO Save best model: monitor(max): 0.284963
2022-03-10 10:21:16,259 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:21:16,297 P41332 INFO Train loss: 0.512699
2022-03-10 10:21:16,297 P41332 INFO ************ Epoch=7 end ************
2022-03-10 10:21:50,137 P41332 INFO [Metrics] logloss: 0.526577 - AUC: 0.813520
2022-03-10 10:21:50,137 P41332 INFO Save best model: monitor(max): 0.286943
2022-03-10 10:21:50,201 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:21:50,238 P41332 INFO Train loss: 0.510222
2022-03-10 10:21:50,238 P41332 INFO ************ Epoch=8 end ************
2022-03-10 10:22:23,579 P41332 INFO [Metrics] logloss: 0.525591 - AUC: 0.814428
2022-03-10 10:22:23,580 P41332 INFO Save best model: monitor(max): 0.288836
2022-03-10 10:22:23,639 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:22:23,675 P41332 INFO Train loss: 0.508269
2022-03-10 10:22:23,675 P41332 INFO ************ Epoch=9 end ************
2022-03-10 10:22:56,807 P41332 INFO [Metrics] logloss: 0.524992 - AUC: 0.814898
2022-03-10 10:22:56,808 P41332 INFO Save best model: monitor(max): 0.289906
2022-03-10 10:22:56,868 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:22:56,905 P41332 INFO Train loss: 0.506742
2022-03-10 10:22:56,905 P41332 INFO ************ Epoch=10 end ************
2022-03-10 10:23:31,662 P41332 INFO [Metrics] logloss: 0.524255 - AUC: 0.815599
2022-03-10 10:23:31,663 P41332 INFO Save best model: monitor(max): 0.291344
2022-03-10 10:23:31,733 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:23:31,780 P41332 INFO Train loss: 0.505529
2022-03-10 10:23:31,781 P41332 INFO ************ Epoch=11 end ************
2022-03-10 10:24:04,648 P41332 INFO [Metrics] logloss: 0.523971 - AUC: 0.815822
2022-03-10 10:24:04,649 P41332 INFO Save best model: monitor(max): 0.291851
2022-03-10 10:24:04,707 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:24:04,744 P41332 INFO Train loss: 0.504482
2022-03-10 10:24:04,745 P41332 INFO ************ Epoch=12 end ************
2022-03-10 10:24:37,700 P41332 INFO [Metrics] logloss: 0.523502 - AUC: 0.816232
2022-03-10 10:24:37,700 P41332 INFO Save best model: monitor(max): 0.292730
2022-03-10 10:24:37,756 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:24:37,793 P41332 INFO Train loss: 0.503722
2022-03-10 10:24:37,793 P41332 INFO ************ Epoch=13 end ************
2022-03-10 10:25:10,469 P41332 INFO [Metrics] logloss: 0.523206 - AUC: 0.816502
2022-03-10 10:25:10,470 P41332 INFO Save best model: monitor(max): 0.293295
2022-03-10 10:25:10,517 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:25:10,555 P41332 INFO Train loss: 0.503005
2022-03-10 10:25:10,555 P41332 INFO ************ Epoch=14 end ************
2022-03-10 10:25:43,610 P41332 INFO [Metrics] logloss: 0.522839 - AUC: 0.816770
2022-03-10 10:25:43,611 P41332 INFO Save best model: monitor(max): 0.293931
2022-03-10 10:25:43,666 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:25:43,703 P41332 INFO Train loss: 0.502446
2022-03-10 10:25:43,703 P41332 INFO ************ Epoch=15 end ************
2022-03-10 10:26:16,525 P41332 INFO [Metrics] logloss: 0.522795 - AUC: 0.816967
2022-03-10 10:26:16,525 P41332 INFO Save best model: monitor(max): 0.294172
2022-03-10 10:26:16,581 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:26:16,617 P41332 INFO Train loss: 0.501886
2022-03-10 10:26:16,618 P41332 INFO ************ Epoch=16 end ************
2022-03-10 10:26:51,200 P41332 INFO [Metrics] logloss: 0.522369 - AUC: 0.817295
2022-03-10 10:26:51,201 P41332 INFO Save best model: monitor(max): 0.294926
2022-03-10 10:26:51,263 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:26:51,307 P41332 INFO Train loss: 0.501423
2022-03-10 10:26:51,307 P41332 INFO ************ Epoch=17 end ************
2022-03-10 10:27:25,630 P41332 INFO [Metrics] logloss: 0.522271 - AUC: 0.817492
2022-03-10 10:27:25,630 P41332 INFO Save best model: monitor(max): 0.295221
2022-03-10 10:27:25,692 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:27:25,735 P41332 INFO Train loss: 0.501100
2022-03-10 10:27:25,735 P41332 INFO ************ Epoch=18 end ************
2022-03-10 10:28:01,813 P41332 INFO [Metrics] logloss: 0.522063 - AUC: 0.817497
2022-03-10 10:28:01,814 P41332 INFO Save best model: monitor(max): 0.295435
2022-03-10 10:28:01,873 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:28:01,910 P41332 INFO Train loss: 0.500716
2022-03-10 10:28:01,910 P41332 INFO ************ Epoch=19 end ************
2022-03-10 10:28:35,032 P41332 INFO [Metrics] logloss: 0.521953 - AUC: 0.817633
2022-03-10 10:28:35,033 P41332 INFO Save best model: monitor(max): 0.295680
2022-03-10 10:28:35,087 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:28:35,123 P41332 INFO Train loss: 0.500465
2022-03-10 10:28:35,123 P41332 INFO ************ Epoch=20 end ************
2022-03-10 10:29:10,412 P41332 INFO [Metrics] logloss: 0.521806 - AUC: 0.817835
2022-03-10 10:29:10,412 P41332 INFO Save best model: monitor(max): 0.296030
2022-03-10 10:29:10,474 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:29:10,520 P41332 INFO Train loss: 0.500156
2022-03-10 10:29:10,521 P41332 INFO ************ Epoch=21 end ************
2022-03-10 10:29:46,792 P41332 INFO [Metrics] logloss: 0.522107 - AUC: 0.817399
2022-03-10 10:29:46,793 P41332 INFO Monitor(max) STOP: 0.295292 !
2022-03-10 10:29:46,793 P41332 INFO Reduce learning rate on plateau: 0.000100
2022-03-10 10:29:46,793 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:29:46,839 P41332 INFO Train loss: 0.499867
2022-03-10 10:29:46,839 P41332 INFO ************ Epoch=22 end ************
2022-03-10 10:30:21,586 P41332 INFO [Metrics] logloss: 0.520255 - AUC: 0.819091
2022-03-10 10:30:21,587 P41332 INFO Save best model: monitor(max): 0.298836
2022-03-10 10:30:21,638 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:30:21,680 P41332 INFO Train loss: 0.487482
2022-03-10 10:30:21,680 P41332 INFO ************ Epoch=23 end ************
2022-03-10 10:30:55,717 P41332 INFO [Metrics] logloss: 0.519822 - AUC: 0.819423
2022-03-10 10:30:55,717 P41332 INFO Save best model: monitor(max): 0.299601
2022-03-10 10:30:55,775 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:30:55,814 P41332 INFO Train loss: 0.486410
2022-03-10 10:30:55,814 P41332 INFO ************ Epoch=24 end ************
2022-03-10 10:31:30,278 P41332 INFO [Metrics] logloss: 0.519670 - AUC: 0.819569
2022-03-10 10:31:30,279 P41332 INFO Save best model: monitor(max): 0.299899
2022-03-10 10:31:30,341 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:31:30,382 P41332 INFO Train loss: 0.485951
2022-03-10 10:31:30,382 P41332 INFO ************ Epoch=25 end ************
2022-03-10 10:32:04,384 P41332 INFO [Metrics] logloss: 0.519545 - AUC: 0.819667
2022-03-10 10:32:04,384 P41332 INFO Save best model: monitor(max): 0.300122
2022-03-10 10:32:04,441 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:32:04,480 P41332 INFO Train loss: 0.485638
2022-03-10 10:32:04,480 P41332 INFO ************ Epoch=26 end ************
2022-03-10 10:32:38,721 P41332 INFO [Metrics] logloss: 0.519545 - AUC: 0.819653
2022-03-10 10:32:38,722 P41332 INFO Monitor(max) STOP: 0.300108 !
2022-03-10 10:32:38,722 P41332 INFO Reduce learning rate on plateau: 0.000010
2022-03-10 10:32:38,722 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:32:38,771 P41332 INFO Train loss: 0.485436
2022-03-10 10:32:38,771 P41332 INFO ************ Epoch=27 end ************
2022-03-10 10:33:14,798 P41332 INFO [Metrics] logloss: 0.519510 - AUC: 0.819676
2022-03-10 10:33:14,799 P41332 INFO Save best model: monitor(max): 0.300166
2022-03-10 10:33:14,869 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:33:14,919 P41332 INFO Train loss: 0.483624
2022-03-10 10:33:14,919 P41332 INFO ************ Epoch=28 end ************
2022-03-10 10:33:47,928 P41332 INFO [Metrics] logloss: 0.519502 - AUC: 0.819681
2022-03-10 10:33:47,929 P41332 INFO Save best model: monitor(max): 0.300179
2022-03-10 10:33:47,985 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:33:48,044 P41332 INFO Train loss: 0.483623
2022-03-10 10:33:48,046 P41332 INFO ************ Epoch=29 end ************
2022-03-10 10:34:20,613 P41332 INFO [Metrics] logloss: 0.519497 - AUC: 0.819686
2022-03-10 10:34:20,614 P41332 INFO Save best model: monitor(max): 0.300189
2022-03-10 10:34:20,671 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:34:20,709 P41332 INFO Train loss: 0.483591
2022-03-10 10:34:20,709 P41332 INFO ************ Epoch=30 end ************
2022-03-10 10:34:54,042 P41332 INFO [Metrics] logloss: 0.519496 - AUC: 0.819684
2022-03-10 10:34:54,043 P41332 INFO Monitor(max) STOP: 0.300188 !
2022-03-10 10:34:54,043 P41332 INFO Reduce learning rate on plateau: 0.000001
2022-03-10 10:34:54,043 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:34:54,081 P41332 INFO Train loss: 0.483554
2022-03-10 10:34:54,081 P41332 INFO ************ Epoch=31 end ************
2022-03-10 10:35:28,417 P41332 INFO [Metrics] logloss: 0.519495 - AUC: 0.819684
2022-03-10 10:35:28,418 P41332 INFO Monitor(max) STOP: 0.300189 !
2022-03-10 10:35:28,418 P41332 INFO Reduce learning rate on plateau: 0.000001
2022-03-10 10:35:28,419 P41332 INFO Early stopping at epoch=32
2022-03-10 10:35:28,419 P41332 INFO --- 591/591 batches finished ---
2022-03-10 10:35:28,460 P41332 INFO Train loss: 0.483399
2022-03-10 10:35:28,460 P41332 INFO Training finished.
2022-03-10 10:35:28,460 P41332 INFO Load best model: /cache/FuxiCTR/benchmarks/KKBox/LorentzFM_kkbox_x1/kkbox_x1_227d337d/LorentzFM_kkbox_x1_003_0455bdef_model.ckpt
2022-03-10 10:35:28,533 P41332 INFO ****** Validation evaluation ******
2022-03-10 10:35:32,906 P41332 INFO [Metrics] logloss: 0.519497 - AUC: 0.819686
2022-03-10 10:35:32,970 P41332 INFO ******** Test evaluation ********
2022-03-10 10:35:32,970 P41332 INFO Loading data...
2022-03-10 10:35:32,970 P41332 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-10 10:35:33,040 P41332 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-10 10:35:33,040 P41332 INFO Loading test data done.
2022-03-10 10:35:37,028 P41332 INFO [Metrics] logloss: 0.518848 - AUC: 0.820207

```
