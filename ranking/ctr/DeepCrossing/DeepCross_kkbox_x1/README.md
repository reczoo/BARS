## DeepCross_kkbox_x1

A hands-on guide to run the DeepCrossing model on the KKBox_x1 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [DeepCrossing](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DeepCrossing.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DeepCrossing_kkbox_x1_tuner_config_02](./DeepCrossing_kkbox_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DeepCross_kkbox_x1
    nohup python run_expid.py --config ./DeepCrossing_kkbox_x1_tuner_config_02 --expid DeepCrossing_kkbox_x1_022_34237390 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.479949 | 0.849486  |


### Logs
```python
2022-03-09 15:46:47,215 P20158 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "dnn_activations": "relu",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.0005",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "5",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "DeepCrossing",
    "model_id": "DeepCrossing_kkbox_x1_022_34237390",
    "model_root": "./KKBox/DeepCrossing_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "residual_blocks": "[1000, 1000, 1000, 1000]",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/KKBox/KKBox_x1/test.csv",
    "train_data": "../data/KKBox/KKBox_x1/train.csv",
    "use_hdf5": "True",
    "use_residual": "True",
    "valid_data": "../data/KKBox/KKBox_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2022-03-09 15:46:47,216 P20158 INFO Set up feature encoder...
2022-03-09 15:46:47,216 P20158 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-09 15:46:49,145 P20158 INFO Total number of parameters: 25131937.
2022-03-09 15:46:49,146 P20158 INFO Loading data...
2022-03-09 15:46:49,146 P20158 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-09 15:46:49,739 P20158 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-09 15:46:50,135 P20158 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-09 15:46:50,160 P20158 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-09 15:46:50,160 P20158 INFO Loading train data done.
2022-03-09 15:46:56,242 P20158 INFO Start training: 591 batches/epoch
2022-03-09 15:46:56,242 P20158 INFO ************ Epoch=1 start ************
2022-03-09 15:49:34,891 P20158 INFO [Metrics] logloss: 0.555760 - AUC: 0.786964
2022-03-09 15:49:34,895 P20158 INFO Save best model: monitor(max): 0.231205
2022-03-09 15:49:35,206 P20158 INFO --- 591/591 batches finished ---
2022-03-09 15:49:35,242 P20158 INFO Train loss: 0.611627
2022-03-09 15:49:35,242 P20158 INFO ************ Epoch=1 end ************
2022-03-09 15:52:13,891 P20158 INFO [Metrics] logloss: 0.542980 - AUC: 0.800824
2022-03-09 15:52:13,892 P20158 INFO Save best model: monitor(max): 0.257843
2022-03-09 15:52:14,006 P20158 INFO --- 591/591 batches finished ---
2022-03-09 15:52:14,039 P20158 INFO Train loss: 0.594397
2022-03-09 15:52:14,039 P20158 INFO ************ Epoch=2 end ************
2022-03-09 15:54:52,800 P20158 INFO [Metrics] logloss: 0.537000 - AUC: 0.809500
2022-03-09 15:54:52,801 P20158 INFO Save best model: monitor(max): 0.272500
2022-03-09 15:54:52,936 P20158 INFO --- 591/591 batches finished ---
2022-03-09 15:54:52,977 P20158 INFO Train loss: 0.585026
2022-03-09 15:54:52,977 P20158 INFO ************ Epoch=3 end ************
2022-03-09 15:57:31,700 P20158 INFO [Metrics] logloss: 0.532009 - AUC: 0.813996
2022-03-09 15:57:31,701 P20158 INFO Save best model: monitor(max): 0.281987
2022-03-09 15:57:31,818 P20158 INFO --- 591/591 batches finished ---
2022-03-09 15:57:31,855 P20158 INFO Train loss: 0.579652
2022-03-09 15:57:31,856 P20158 INFO ************ Epoch=4 end ************
2022-03-09 16:00:10,417 P20158 INFO [Metrics] logloss: 0.524610 - AUC: 0.817397
2022-03-09 16:00:10,418 P20158 INFO Save best model: monitor(max): 0.292787
2022-03-09 16:00:10,516 P20158 INFO --- 591/591 batches finished ---
2022-03-09 16:00:10,551 P20158 INFO Train loss: 0.575900
2022-03-09 16:00:10,551 P20158 INFO ************ Epoch=5 end ************
2022-03-09 16:02:48,696 P20158 INFO [Metrics] logloss: 0.521031 - AUC: 0.820984
2022-03-09 16:02:48,697 P20158 INFO Save best model: monitor(max): 0.299953
2022-03-09 16:02:48,812 P20158 INFO --- 591/591 batches finished ---
2022-03-09 16:02:48,847 P20158 INFO Train loss: 0.573155
2022-03-09 16:02:48,847 P20158 INFO ************ Epoch=6 end ************
2022-03-09 16:05:27,764 P20158 INFO [Metrics] logloss: 0.522458 - AUC: 0.822651
2022-03-09 16:05:27,765 P20158 INFO Save best model: monitor(max): 0.300192
2022-03-09 16:05:27,886 P20158 INFO --- 591/591 batches finished ---
2022-03-09 16:05:27,921 P20158 INFO Train loss: 0.570924
2022-03-09 16:05:27,921 P20158 INFO ************ Epoch=7 end ************
2022-03-09 16:08:06,586 P20158 INFO [Metrics] logloss: 0.518999 - AUC: 0.824785
2022-03-09 16:08:06,587 P20158 INFO Save best model: monitor(max): 0.305786
2022-03-09 16:08:06,734 P20158 INFO --- 591/591 batches finished ---
2022-03-09 16:08:06,770 P20158 INFO Train loss: 0.569023
2022-03-09 16:08:06,770 P20158 INFO ************ Epoch=8 end ************
2022-03-09 16:09:20,504 P20158 INFO [Metrics] logloss: 0.515499 - AUC: 0.826242
2022-03-09 16:09:20,504 P20158 INFO Save best model: monitor(max): 0.310742
2022-03-09 16:09:20,626 P20158 INFO --- 591/591 batches finished ---
2022-03-09 16:09:20,662 P20158 INFO Train loss: 0.567613
2022-03-09 16:09:20,662 P20158 INFO ************ Epoch=9 end ************
2022-03-09 16:10:31,411 P20158 INFO [Metrics] logloss: 0.513836 - AUC: 0.826836
2022-03-09 16:10:31,412 P20158 INFO Save best model: monitor(max): 0.313000
2022-03-09 16:10:31,517 P20158 INFO --- 591/591 batches finished ---
2022-03-09 16:10:31,555 P20158 INFO Train loss: 0.566212
2022-03-09 16:10:31,555 P20158 INFO ************ Epoch=10 end ************
2022-03-09 16:11:42,762 P20158 INFO [Metrics] logloss: 0.512140 - AUC: 0.828521
2022-03-09 16:11:42,763 P20158 INFO Save best model: monitor(max): 0.316381
2022-03-09 16:11:42,893 P20158 INFO --- 591/591 batches finished ---
2022-03-09 16:11:42,932 P20158 INFO Train loss: 0.565040
2022-03-09 16:11:42,932 P20158 INFO ************ Epoch=11 end ************
2022-03-09 16:12:53,946 P20158 INFO [Metrics] logloss: 0.512148 - AUC: 0.829688
2022-03-09 16:12:53,947 P20158 INFO Save best model: monitor(max): 0.317540
2022-03-09 16:12:54,075 P20158 INFO --- 591/591 batches finished ---
2022-03-09 16:12:54,115 P20158 INFO Train loss: 0.563821
2022-03-09 16:12:54,115 P20158 INFO ************ Epoch=12 end ************
2022-03-09 16:14:04,969 P20158 INFO [Metrics] logloss: 0.508696 - AUC: 0.830894
2022-03-09 16:14:04,969 P20158 INFO Save best model: monitor(max): 0.322198
2022-03-09 16:14:05,101 P20158 INFO --- 591/591 batches finished ---
2022-03-09 16:14:05,139 P20158 INFO Train loss: 0.562849
2022-03-09 16:14:05,139 P20158 INFO ************ Epoch=13 end ************
2022-03-09 16:15:15,878 P20158 INFO [Metrics] logloss: 0.512093 - AUC: 0.830865
2022-03-09 16:15:15,879 P20158 INFO Monitor(max) STOP: 0.318772 !
2022-03-09 16:15:15,879 P20158 INFO Reduce learning rate on plateau: 0.000100
2022-03-09 16:15:15,879 P20158 INFO --- 591/591 batches finished ---
2022-03-09 16:15:15,915 P20158 INFO Train loss: 0.561740
2022-03-09 16:15:15,915 P20158 INFO ************ Epoch=14 end ************
2022-03-09 16:16:26,486 P20158 INFO [Metrics] logloss: 0.486521 - AUC: 0.845178
2022-03-09 16:16:26,487 P20158 INFO Save best model: monitor(max): 0.358657
2022-03-09 16:16:26,585 P20158 INFO --- 591/591 batches finished ---
2022-03-09 16:16:26,619 P20158 INFO Train loss: 0.504080
2022-03-09 16:16:26,619 P20158 INFO ************ Epoch=15 end ************
2022-03-09 16:17:37,029 P20158 INFO [Metrics] logloss: 0.481792 - AUC: 0.848009
2022-03-09 16:17:37,030 P20158 INFO Save best model: monitor(max): 0.366217
2022-03-09 16:17:37,147 P20158 INFO --- 591/591 batches finished ---
2022-03-09 16:17:37,181 P20158 INFO Train loss: 0.475689
2022-03-09 16:17:37,181 P20158 INFO ************ Epoch=16 end ************
2022-03-09 16:18:47,864 P20158 INFO [Metrics] logloss: 0.480752 - AUC: 0.848992
2022-03-09 16:18:47,864 P20158 INFO Save best model: monitor(max): 0.368240
2022-03-09 16:18:47,986 P20158 INFO --- 591/591 batches finished ---
2022-03-09 16:18:48,020 P20158 INFO Train loss: 0.462996
2022-03-09 16:18:48,021 P20158 INFO ************ Epoch=17 end ************
2022-03-09 16:19:58,763 P20158 INFO [Metrics] logloss: 0.481771 - AUC: 0.848983
2022-03-09 16:19:58,764 P20158 INFO Monitor(max) STOP: 0.367211 !
2022-03-09 16:19:58,764 P20158 INFO Reduce learning rate on plateau: 0.000010
2022-03-09 16:19:58,764 P20158 INFO --- 591/591 batches finished ---
2022-03-09 16:19:58,797 P20158 INFO Train loss: 0.453432
2022-03-09 16:19:58,798 P20158 INFO ************ Epoch=18 end ************
2022-03-09 16:21:09,593 P20158 INFO [Metrics] logloss: 0.492354 - AUC: 0.847362
2022-03-09 16:21:09,593 P20158 INFO Monitor(max) STOP: 0.355008 !
2022-03-09 16:21:09,594 P20158 INFO Reduce learning rate on plateau: 0.000001
2022-03-09 16:21:09,594 P20158 INFO Early stopping at epoch=19
2022-03-09 16:21:09,594 P20158 INFO --- 591/591 batches finished ---
2022-03-09 16:21:09,627 P20158 INFO Train loss: 0.424854
2022-03-09 16:21:09,628 P20158 INFO Training finished.
2022-03-09 16:21:09,628 P20158 INFO Load best model: /cache/FuxiCTR/benchmarks/KKBox/DeepCrossing_kkbox_x1/kkbox_x1_227d337d/DeepCrossing_kkbox_x1_022_34237390_model.ckpt
2022-03-09 16:21:09,794 P20158 INFO ****** Validation evaluation ******
2022-03-09 16:21:13,619 P20158 INFO [Metrics] logloss: 0.480752 - AUC: 0.848992
2022-03-09 16:21:13,672 P20158 INFO ******** Test evaluation ********
2022-03-09 16:21:13,672 P20158 INFO Loading data...
2022-03-09 16:21:13,672 P20158 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-09 16:21:13,736 P20158 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-09 16:21:13,736 P20158 INFO Loading test data done.
2022-03-09 16:21:17,784 P20158 INFO [Metrics] logloss: 0.479949 - AUC: 0.849486

```
