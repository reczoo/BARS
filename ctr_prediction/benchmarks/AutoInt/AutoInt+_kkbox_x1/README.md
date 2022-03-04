## AutoInt+_kkbox_x1

A hands-on guide to run the AutoInt model on the KKBox_x1 dataset.

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
Dataset ID: [KKBox_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/KKBox/README.md#KKBox_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [AutoInt](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt+_kkbox_x1_tuner_config_06](./AutoInt+_kkbox_x1_tuner_config_06). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AutoInt+_kkbox_x1
    nohup python run_expid.py --config ./AutoInt+_kkbox_x1_tuner_config_06 --expid AutoInt_kkbox_x1_014_44f4cf9f --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.476795 | 0.852967  |


### Logs
```python
2020-05-07 17:55:14,229 P11754 INFO {
    "attention_dim": "256",
    "attention_layers": "4",
    "batch_norm": "False",
    "batch_size": "10000",
    "dataset_id": "kkbox_x1_001_c5c9c6e3",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[5000, 5000]",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.0005",
    "epochs": "100",
    "every_x_epochs": "1",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AutoInt",
    "model_id": "AutoInt_kkbox_x1_014_b479679e",
    "model_root": "./KKBox/AutoInt_kkbox/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.4",
    "net_regularizer": "0",
    "num_heads": "1",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "use_residual": "False",
    "use_scale": "False",
    "use_wide": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/KKBox/",
    "test_data": "../data/KKBox/kkbox_x1_001_c5c9c6e3/test.h5",
    "train_data": "../data/KKBox/kkbox_x1_001_c5c9c6e3/train.h5",
    "valid_data": "../data/KKBox/kkbox_x1_001_c5c9c6e3/valid.h5",
    "version": "pytorch",
    "gpu": "1"
}
2020-05-07 17:55:14,230 P11754 INFO Set up feature encoder...
2020-05-07 17:55:14,230 P11754 INFO Load feature_map from json: ../data/KKBox/kkbox_x1_001_c5c9c6e3/feature_map.json
2020-05-07 17:55:14,231 P11754 INFO Loading data...
2020-05-07 17:55:14,233 P11754 INFO Loading data from h5: ../data/KKBox/kkbox_x1_001_c5c9c6e3/train.h5
2020-05-07 17:55:14,528 P11754 INFO Loading data from h5: ../data/KKBox/kkbox_x1_001_c5c9c6e3/valid.h5
2020-05-07 17:55:14,720 P11754 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2020-05-07 17:55:14,739 P11754 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-05-07 17:55:14,739 P11754 INFO Loading train data done.
2020-05-07 17:55:19,431 P11754 INFO **** Start training: 591 batches/epoch ****
2020-05-07 18:00:21,143 P11754 INFO [Metrics] logloss: 0.556085 - AUC: 0.786012
2020-05-07 18:00:21,156 P11754 INFO Save best model: monitor(max): 0.229927
2020-05-07 18:00:21,323 P11754 INFO --- 591/591 batches finished ---
2020-05-07 18:00:21,365 P11754 INFO Train loss: 0.609588
2020-05-07 18:00:21,365 P11754 INFO ************ Epoch=1 end ************
2020-05-07 18:05:22,308 P11754 INFO [Metrics] logloss: 0.541878 - AUC: 0.799150
2020-05-07 18:05:22,333 P11754 INFO Save best model: monitor(max): 0.257273
2020-05-07 18:05:22,572 P11754 INFO --- 591/591 batches finished ---
2020-05-07 18:05:22,613 P11754 INFO Train loss: 0.592799
2020-05-07 18:05:22,614 P11754 INFO ************ Epoch=2 end ************
2020-05-07 18:10:23,097 P11754 INFO [Metrics] logloss: 0.534817 - AUC: 0.805836
2020-05-07 18:10:23,108 P11754 INFO Save best model: monitor(max): 0.271019
2020-05-07 18:10:23,333 P11754 INFO --- 591/591 batches finished ---
2020-05-07 18:10:23,377 P11754 INFO Train loss: 0.586220
2020-05-07 18:10:23,377 P11754 INFO ************ Epoch=3 end ************
2020-05-07 18:15:23,349 P11754 INFO [Metrics] logloss: 0.529034 - AUC: 0.810754
2020-05-07 18:15:23,361 P11754 INFO Save best model: monitor(max): 0.281720
2020-05-07 18:15:23,604 P11754 INFO --- 591/591 batches finished ---
2020-05-07 18:15:23,646 P11754 INFO Train loss: 0.581400
2020-05-07 18:15:23,646 P11754 INFO ************ Epoch=4 end ************
2020-05-07 18:20:23,633 P11754 INFO [Metrics] logloss: 0.523578 - AUC: 0.815112
2020-05-07 18:20:23,653 P11754 INFO Save best model: monitor(max): 0.291534
2020-05-07 18:20:23,900 P11754 INFO --- 591/591 batches finished ---
2020-05-07 18:20:23,946 P11754 INFO Train loss: 0.578016
2020-05-07 18:20:23,946 P11754 INFO ************ Epoch=5 end ************
2020-05-07 18:25:25,073 P11754 INFO [Metrics] logloss: 0.522761 - AUC: 0.816672
2020-05-07 18:25:25,087 P11754 INFO Save best model: monitor(max): 0.293911
2020-05-07 18:25:25,333 P11754 INFO --- 591/591 batches finished ---
2020-05-07 18:25:25,376 P11754 INFO Train loss: 0.575423
2020-05-07 18:25:25,376 P11754 INFO ************ Epoch=6 end ************
2020-05-07 18:30:25,383 P11754 INFO [Metrics] logloss: 0.517961 - AUC: 0.819904
2020-05-07 18:30:25,397 P11754 INFO Save best model: monitor(max): 0.301943
2020-05-07 18:30:25,635 P11754 INFO --- 591/591 batches finished ---
2020-05-07 18:30:25,679 P11754 INFO Train loss: 0.573978
2020-05-07 18:30:25,679 P11754 INFO ************ Epoch=7 end ************
2020-05-07 18:35:26,882 P11754 INFO [Metrics] logloss: 0.516043 - AUC: 0.821495
2020-05-07 18:35:26,896 P11754 INFO Save best model: monitor(max): 0.305452
2020-05-07 18:35:27,132 P11754 INFO --- 591/591 batches finished ---
2020-05-07 18:35:27,174 P11754 INFO Train loss: 0.572361
2020-05-07 18:35:27,174 P11754 INFO ************ Epoch=8 end ************
2020-05-07 18:40:28,030 P11754 INFO [Metrics] logloss: 0.513819 - AUC: 0.823312
2020-05-07 18:40:28,042 P11754 INFO Save best model: monitor(max): 0.309493
2020-05-07 18:40:28,272 P11754 INFO --- 591/591 batches finished ---
2020-05-07 18:40:28,314 P11754 INFO Train loss: 0.571171
2020-05-07 18:40:28,314 P11754 INFO ************ Epoch=9 end ************
2020-05-07 18:45:29,021 P11754 INFO [Metrics] logloss: 0.511862 - AUC: 0.825129
2020-05-07 18:45:29,035 P11754 INFO Save best model: monitor(max): 0.313267
2020-05-07 18:45:29,287 P11754 INFO --- 591/591 batches finished ---
2020-05-07 18:45:29,329 P11754 INFO Train loss: 0.569827
2020-05-07 18:45:29,329 P11754 INFO ************ Epoch=10 end ************
2020-05-07 18:50:29,169 P11754 INFO [Metrics] logloss: 0.509898 - AUC: 0.826434
2020-05-07 18:50:29,184 P11754 INFO Save best model: monitor(max): 0.316536
2020-05-07 18:50:29,426 P11754 INFO --- 591/591 batches finished ---
2020-05-07 18:50:29,471 P11754 INFO Train loss: 0.568944
2020-05-07 18:50:29,471 P11754 INFO ************ Epoch=11 end ************
2020-05-07 18:55:29,624 P11754 INFO [Metrics] logloss: 0.509201 - AUC: 0.827417
2020-05-07 18:55:29,636 P11754 INFO Save best model: monitor(max): 0.318216
2020-05-07 18:55:29,892 P11754 INFO --- 591/591 batches finished ---
2020-05-07 18:55:29,934 P11754 INFO Train loss: 0.568254
2020-05-07 18:55:29,934 P11754 INFO ************ Epoch=12 end ************
2020-05-07 19:00:31,369 P11754 INFO [Metrics] logloss: 0.507582 - AUC: 0.828332
2020-05-07 19:00:31,381 P11754 INFO Save best model: monitor(max): 0.320750
2020-05-07 19:00:31,640 P11754 INFO --- 591/591 batches finished ---
2020-05-07 19:00:31,683 P11754 INFO Train loss: 0.567266
2020-05-07 19:00:31,683 P11754 INFO ************ Epoch=13 end ************
2020-05-07 19:05:32,852 P11754 INFO [Metrics] logloss: 0.506074 - AUC: 0.829598
2020-05-07 19:05:32,863 P11754 INFO Save best model: monitor(max): 0.323524
2020-05-07 19:05:33,096 P11754 INFO --- 591/591 batches finished ---
2020-05-07 19:05:33,138 P11754 INFO Train loss: 0.566590
2020-05-07 19:05:33,138 P11754 INFO ************ Epoch=14 end ************
2020-05-07 19:10:33,945 P11754 INFO [Metrics] logloss: 0.505632 - AUC: 0.830098
2020-05-07 19:10:33,956 P11754 INFO Save best model: monitor(max): 0.324466
2020-05-07 19:10:34,193 P11754 INFO --- 591/591 batches finished ---
2020-05-07 19:10:34,238 P11754 INFO Train loss: 0.565978
2020-05-07 19:10:34,238 P11754 INFO ************ Epoch=15 end ************
2020-05-07 19:15:34,544 P11754 INFO [Metrics] logloss: 0.505015 - AUC: 0.830449
2020-05-07 19:15:34,555 P11754 INFO Save best model: monitor(max): 0.325434
2020-05-07 19:15:34,806 P11754 INFO --- 591/591 batches finished ---
2020-05-07 19:15:34,848 P11754 INFO Train loss: 0.565262
2020-05-07 19:15:34,848 P11754 INFO ************ Epoch=16 end ************
2020-05-07 19:20:35,721 P11754 INFO [Metrics] logloss: 0.503260 - AUC: 0.831845
2020-05-07 19:20:35,732 P11754 INFO Save best model: monitor(max): 0.328585
2020-05-07 19:20:35,979 P11754 INFO --- 591/591 batches finished ---
2020-05-07 19:20:36,020 P11754 INFO Train loss: 0.564910
2020-05-07 19:20:36,021 P11754 INFO ************ Epoch=17 end ************
2020-05-07 19:25:36,043 P11754 INFO [Metrics] logloss: 0.502107 - AUC: 0.832779
2020-05-07 19:25:36,055 P11754 INFO Save best model: monitor(max): 0.330672
2020-05-07 19:25:36,289 P11754 INFO --- 591/591 batches finished ---
2020-05-07 19:25:36,337 P11754 INFO Train loss: 0.564588
2020-05-07 19:25:36,337 P11754 INFO ************ Epoch=18 end ************
2020-05-07 19:30:36,356 P11754 INFO [Metrics] logloss: 0.501486 - AUC: 0.833211
2020-05-07 19:30:36,372 P11754 INFO Save best model: monitor(max): 0.331725
2020-05-07 19:30:36,633 P11754 INFO --- 591/591 batches finished ---
2020-05-07 19:30:36,674 P11754 INFO Train loss: 0.564006
2020-05-07 19:30:36,674 P11754 INFO ************ Epoch=19 end ************
2020-05-07 19:35:36,916 P11754 INFO [Metrics] logloss: 0.501182 - AUC: 0.833409
2020-05-07 19:35:36,929 P11754 INFO Save best model: monitor(max): 0.332227
2020-05-07 19:35:37,163 P11754 INFO --- 591/591 batches finished ---
2020-05-07 19:35:37,204 P11754 INFO Train loss: 0.563647
2020-05-07 19:35:37,205 P11754 INFO ************ Epoch=20 end ************
2020-05-07 19:40:37,453 P11754 INFO [Metrics] logloss: 0.500131 - AUC: 0.834498
2020-05-07 19:40:37,467 P11754 INFO Save best model: monitor(max): 0.334367
2020-05-07 19:40:37,700 P11754 INFO --- 591/591 batches finished ---
2020-05-07 19:40:37,742 P11754 INFO Train loss: 0.563183
2020-05-07 19:40:37,743 P11754 INFO ************ Epoch=21 end ************
2020-05-07 19:45:37,387 P11754 INFO [Metrics] logloss: 0.499515 - AUC: 0.834645
2020-05-07 19:45:37,403 P11754 INFO Save best model: monitor(max): 0.335129
2020-05-07 19:45:37,644 P11754 INFO --- 591/591 batches finished ---
2020-05-07 19:45:37,686 P11754 INFO Train loss: 0.562736
2020-05-07 19:45:37,686 P11754 INFO ************ Epoch=22 end ************
2020-05-07 19:50:38,227 P11754 INFO [Metrics] logloss: 0.498911 - AUC: 0.835339
2020-05-07 19:50:38,239 P11754 INFO Save best model: monitor(max): 0.336428
2020-05-07 19:50:38,471 P11754 INFO --- 591/591 batches finished ---
2020-05-07 19:50:38,519 P11754 INFO Train loss: 0.562575
2020-05-07 19:50:38,519 P11754 INFO ************ Epoch=23 end ************
2020-05-07 19:55:38,749 P11754 INFO [Metrics] logloss: 0.498712 - AUC: 0.835444
2020-05-07 19:55:38,762 P11754 INFO Save best model: monitor(max): 0.336732
2020-05-07 19:55:38,996 P11754 INFO --- 591/591 batches finished ---
2020-05-07 19:55:39,039 P11754 INFO Train loss: 0.561965
2020-05-07 19:55:39,039 P11754 INFO ************ Epoch=24 end ************
2020-05-07 20:00:38,902 P11754 INFO [Metrics] logloss: 0.497896 - AUC: 0.836135
2020-05-07 20:00:38,914 P11754 INFO Save best model: monitor(max): 0.338239
2020-05-07 20:00:39,158 P11754 INFO --- 591/591 batches finished ---
2020-05-07 20:00:39,200 P11754 INFO Train loss: 0.561570
2020-05-07 20:00:39,201 P11754 INFO ************ Epoch=25 end ************
2020-05-07 20:05:39,451 P11754 INFO [Metrics] logloss: 0.497725 - AUC: 0.836342
2020-05-07 20:05:39,463 P11754 INFO Save best model: monitor(max): 0.338617
2020-05-07 20:05:39,707 P11754 INFO --- 591/591 batches finished ---
2020-05-07 20:05:39,749 P11754 INFO Train loss: 0.561311
2020-05-07 20:05:39,750 P11754 INFO ************ Epoch=26 end ************
2020-05-07 20:10:40,935 P11754 INFO [Metrics] logloss: 0.497348 - AUC: 0.836684
2020-05-07 20:10:40,953 P11754 INFO Save best model: monitor(max): 0.339336
2020-05-07 20:10:41,204 P11754 INFO --- 591/591 batches finished ---
2020-05-07 20:10:41,252 P11754 INFO Train loss: 0.561128
2020-05-07 20:10:41,252 P11754 INFO ************ Epoch=27 end ************
2020-05-07 20:15:40,899 P11754 INFO [Metrics] logloss: 0.496595 - AUC: 0.837092
2020-05-07 20:15:40,911 P11754 INFO Save best model: monitor(max): 0.340496
2020-05-07 20:15:41,151 P11754 INFO --- 591/591 batches finished ---
2020-05-07 20:15:41,192 P11754 INFO Train loss: 0.560717
2020-05-07 20:15:41,192 P11754 INFO ************ Epoch=28 end ************
2020-05-07 20:20:41,012 P11754 INFO [Metrics] logloss: 0.495850 - AUC: 0.837518
2020-05-07 20:20:41,024 P11754 INFO Save best model: monitor(max): 0.341669
2020-05-07 20:20:41,267 P11754 INFO --- 591/591 batches finished ---
2020-05-07 20:20:41,309 P11754 INFO Train loss: 0.560469
2020-05-07 20:20:41,310 P11754 INFO ************ Epoch=29 end ************
2020-05-07 20:25:41,157 P11754 INFO [Metrics] logloss: 0.495428 - AUC: 0.838001
2020-05-07 20:25:41,170 P11754 INFO Save best model: monitor(max): 0.342573
2020-05-07 20:25:41,409 P11754 INFO --- 591/591 batches finished ---
2020-05-07 20:25:41,451 P11754 INFO Train loss: 0.560112
2020-05-07 20:25:41,452 P11754 INFO ************ Epoch=30 end ************
2020-05-07 20:30:41,610 P11754 INFO [Metrics] logloss: 0.495565 - AUC: 0.837902
2020-05-07 20:30:41,624 P11754 INFO Monitor(max) STOP: 0.342338 !
2020-05-07 20:30:41,625 P11754 INFO Reduce learning rate on plateau: 0.000100
2020-05-07 20:30:41,625 P11754 INFO --- 591/591 batches finished ---
2020-05-07 20:30:41,687 P11754 INFO Train loss: 0.559889
2020-05-07 20:30:41,687 P11754 INFO ************ Epoch=31 end ************
2020-05-07 20:35:41,571 P11754 INFO [Metrics] logloss: 0.480726 - AUC: 0.849077
2020-05-07 20:35:41,587 P11754 INFO Save best model: monitor(max): 0.368351
2020-05-07 20:35:41,847 P11754 INFO --- 591/591 batches finished ---
2020-05-07 20:35:41,890 P11754 INFO Train loss: 0.500319
2020-05-07 20:35:41,890 P11754 INFO ************ Epoch=32 end ************
2020-05-07 20:40:42,799 P11754 INFO [Metrics] logloss: 0.477826 - AUC: 0.851450
2020-05-07 20:40:42,811 P11754 INFO Save best model: monitor(max): 0.373624
2020-05-07 20:40:43,053 P11754 INFO --- 591/591 batches finished ---
2020-05-07 20:40:43,097 P11754 INFO Train loss: 0.471905
2020-05-07 20:40:43,097 P11754 INFO ************ Epoch=33 end ************
2020-05-07 20:45:43,573 P11754 INFO [Metrics] logloss: 0.477137 - AUC: 0.852268
2020-05-07 20:45:43,584 P11754 INFO Save best model: monitor(max): 0.375131
2020-05-07 20:45:43,851 P11754 INFO --- 591/591 batches finished ---
2020-05-07 20:45:43,893 P11754 INFO Train loss: 0.461206
2020-05-07 20:45:43,893 P11754 INFO ************ Epoch=34 end ************
2020-05-07 20:50:43,469 P11754 INFO [Metrics] logloss: 0.477322 - AUC: 0.852634
2020-05-07 20:50:43,480 P11754 INFO Save best model: monitor(max): 0.375312
2020-05-07 20:50:43,720 P11754 INFO --- 591/591 batches finished ---
2020-05-07 20:50:43,763 P11754 INFO Train loss: 0.454532
2020-05-07 20:50:43,763 P11754 INFO ************ Epoch=35 end ************
2020-05-07 20:55:43,529 P11754 INFO [Metrics] logloss: 0.478319 - AUC: 0.852659
2020-05-07 20:55:43,541 P11754 INFO Monitor(max) STOP: 0.374340 !
2020-05-07 20:55:43,541 P11754 INFO Reduce learning rate on plateau: 0.000010
2020-05-07 20:55:43,541 P11754 INFO --- 591/591 batches finished ---
2020-05-07 20:55:43,602 P11754 INFO Train loss: 0.449328
2020-05-07 20:55:43,602 P11754 INFO ************ Epoch=36 end ************
2020-05-07 21:00:43,769 P11754 INFO [Metrics] logloss: 0.482406 - AUC: 0.852346
2020-05-07 21:00:43,781 P11754 INFO Monitor(max) STOP: 0.369940 !
2020-05-07 21:00:43,781 P11754 INFO Reduce learning rate on plateau: 0.000001
2020-05-07 21:00:43,781 P11754 INFO Early stopping at epoch=37
2020-05-07 21:00:43,781 P11754 INFO --- 591/591 batches finished ---
2020-05-07 21:00:43,824 P11754 INFO Train loss: 0.430571
2020-05-07 21:00:43,824 P11754 INFO Training finished.
2020-05-07 21:00:43,824 P11754 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/KKBox/AutoInt_kkbox/kkbox_x1_001_c5c9c6e3/AutoInt_kkbox_x1_014_b479679e_kkbox_x1_001_c5c9c6e3_model.ckpt
2020-05-07 21:00:44,049 P11754 INFO ****** Train/validation evaluation ******
2020-05-07 21:02:18,959 P11754 INFO [Metrics] logloss: 0.391159 - AUC: 0.905808
2020-05-07 21:02:31,102 P11754 INFO [Metrics] logloss: 0.477322 - AUC: 0.852634
2020-05-07 21:02:31,191 P11754 INFO ******** Test evaluation ********
2020-05-07 21:02:31,191 P11754 INFO Loading data...
2020-05-07 21:02:31,191 P11754 INFO Loading data from h5: ../data/KKBox/kkbox_x1_001_c5c9c6e3/test.h5
2020-05-07 21:02:31,258 P11754 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-05-07 21:02:31,258 P11754 INFO Loading test data done.
2020-05-07 21:02:43,065 P11754 INFO [Metrics] logloss: 0.476795 - AUC: 0.852967

```
