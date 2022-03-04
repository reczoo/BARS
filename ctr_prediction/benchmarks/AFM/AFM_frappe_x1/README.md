## AFM_frappe_x1

A hands-on guide to run the AFM model on the Frappe_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
  GPU: Tesla P100 16G
  RAM: 503G

  ```

+ Software

  ```python
  CUDA: 10.2
  python: 3.6.4
  pytorch: 1.0.0
  pandas: 0.22.0
  numpy: 1.19.1
  scipy: 1.2.2
  sklearn: 0.19.1
  pyyaml: 6.0
  h5py: 2.8.0
  tqdm: 4.28.1
  fuxictr: 1.1.0
  ```

### Dataset
Dataset ID: [Frappe_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Frappe/README.md#Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [AFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/AFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFM_frappe_x1_tuner_config_02](./AFM_frappe_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFM_frappe_x1
    nohup python run_expid.py --config ./AFM_frappe_x1_tuner_config_02 --expid AFM_frappe_x1_011_2d591f68 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.969733 | 0.226424  |


### Logs
```python
2022-01-26 21:39:33,439 P65841 INFO {
    "attention_dim": "128",
    "attention_dropout": "[0.1, 0.1]",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AFM",
    "model_id": "AFM_frappe_x1_011_2d591f68",
    "model_root": "./Frappe/AFM_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Frappe/Frappe_x1/test.csv",
    "train_data": "../data/Frappe/Frappe_x1/train.csv",
    "use_attention": "True",
    "use_hdf5": "True",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-26 21:39:33,440 P65841 INFO Set up feature encoder...
2022-01-26 21:39:33,440 P65841 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-01-26 21:39:33,441 P65841 INFO Loading data...
2022-01-26 21:39:33,443 P65841 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-01-26 21:39:33,455 P65841 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-01-26 21:39:33,458 P65841 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-01-26 21:39:33,459 P65841 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-01-26 21:39:33,459 P65841 INFO Loading train data done.
2022-01-26 21:39:37,231 P65841 INFO Total number of parameters: 60826.
2022-01-26 21:39:37,232 P65841 INFO Start training: 50 batches/epoch
2022-01-26 21:39:37,232 P65841 INFO ************ Epoch=1 start ************
2022-01-26 21:39:42,747 P65841 INFO [Metrics] AUC: 0.899578 - logloss: 0.631500
2022-01-26 21:39:42,747 P65841 INFO Save best model: monitor(max): 0.899578
2022-01-26 21:39:42,750 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:39:42,799 P65841 INFO Train loss: 0.658715
2022-01-26 21:39:42,799 P65841 INFO ************ Epoch=1 end ************
2022-01-26 21:39:48,326 P65841 INFO [Metrics] AUC: 0.910360 - logloss: 0.610467
2022-01-26 21:39:48,326 P65841 INFO Save best model: monitor(max): 0.910360
2022-01-26 21:39:48,329 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:39:48,370 P65841 INFO Train loss: 0.622058
2022-01-26 21:39:48,371 P65841 INFO ************ Epoch=2 end ************
2022-01-26 21:39:53,905 P65841 INFO [Metrics] AUC: 0.924115 - logloss: 0.564399
2022-01-26 21:39:53,905 P65841 INFO Save best model: monitor(max): 0.924115
2022-01-26 21:39:53,908 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:39:53,956 P65841 INFO Train loss: 0.597455
2022-01-26 21:39:53,956 P65841 INFO ************ Epoch=3 end ************
2022-01-26 21:39:59,442 P65841 INFO [Metrics] AUC: 0.929097 - logloss: 0.477258
2022-01-26 21:39:59,443 P65841 INFO Save best model: monitor(max): 0.929097
2022-01-26 21:39:59,446 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:39:59,490 P65841 INFO Train loss: 0.525879
2022-01-26 21:39:59,490 P65841 INFO ************ Epoch=4 end ************
2022-01-26 21:40:05,028 P65841 INFO [Metrics] AUC: 0.930875 - logloss: 0.378835
2022-01-26 21:40:05,028 P65841 INFO Save best model: monitor(max): 0.930875
2022-01-26 21:40:05,031 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:40:05,072 P65841 INFO Train loss: 0.432085
2022-01-26 21:40:05,072 P65841 INFO ************ Epoch=5 end ************
2022-01-26 21:40:10,565 P65841 INFO [Metrics] AUC: 0.933548 - logloss: 0.323143
2022-01-26 21:40:10,566 P65841 INFO Save best model: monitor(max): 0.933548
2022-01-26 21:40:10,568 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:40:10,615 P65841 INFO Train loss: 0.355494
2022-01-26 21:40:10,615 P65841 INFO ************ Epoch=6 end ************
2022-01-26 21:40:16,073 P65841 INFO [Metrics] AUC: 0.935912 - logloss: 0.303203
2022-01-26 21:40:16,073 P65841 INFO Save best model: monitor(max): 0.935912
2022-01-26 21:40:16,076 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:40:16,120 P65841 INFO Train loss: 0.322466
2022-01-26 21:40:16,121 P65841 INFO ************ Epoch=7 end ************
2022-01-26 21:40:21,657 P65841 INFO [Metrics] AUC: 0.938143 - logloss: 0.293831
2022-01-26 21:40:21,658 P65841 INFO Save best model: monitor(max): 0.938143
2022-01-26 21:40:21,660 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:40:21,701 P65841 INFO Train loss: 0.305230
2022-01-26 21:40:21,701 P65841 INFO ************ Epoch=8 end ************
2022-01-26 21:40:27,275 P65841 INFO [Metrics] AUC: 0.940090 - logloss: 0.286650
2022-01-26 21:40:27,275 P65841 INFO Save best model: monitor(max): 0.940090
2022-01-26 21:40:27,278 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:40:27,319 P65841 INFO Train loss: 0.293569
2022-01-26 21:40:27,319 P65841 INFO ************ Epoch=9 end ************
2022-01-26 21:40:32,750 P65841 INFO [Metrics] AUC: 0.941717 - logloss: 0.281586
2022-01-26 21:40:32,751 P65841 INFO Save best model: monitor(max): 0.941717
2022-01-26 21:40:32,753 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:40:32,798 P65841 INFO Train loss: 0.286057
2022-01-26 21:40:32,798 P65841 INFO ************ Epoch=10 end ************
2022-01-26 21:40:38,147 P65841 INFO [Metrics] AUC: 0.943276 - logloss: 0.277970
2022-01-26 21:40:38,148 P65841 INFO Save best model: monitor(max): 0.943276
2022-01-26 21:40:38,150 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:40:38,194 P65841 INFO Train loss: 0.280638
2022-01-26 21:40:38,194 P65841 INFO ************ Epoch=11 end ************
2022-01-26 21:40:42,472 P65841 INFO [Metrics] AUC: 0.944392 - logloss: 0.274526
2022-01-26 21:40:42,472 P65841 INFO Save best model: monitor(max): 0.944392
2022-01-26 21:40:42,475 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:40:42,518 P65841 INFO Train loss: 0.275469
2022-01-26 21:40:42,518 P65841 INFO ************ Epoch=12 end ************
2022-01-26 21:40:46,581 P65841 INFO [Metrics] AUC: 0.945715 - logloss: 0.271819
2022-01-26 21:40:46,582 P65841 INFO Save best model: monitor(max): 0.945715
2022-01-26 21:40:46,584 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:40:46,631 P65841 INFO Train loss: 0.270992
2022-01-26 21:40:46,631 P65841 INFO ************ Epoch=13 end ************
2022-01-26 21:40:52,166 P65841 INFO [Metrics] AUC: 0.946668 - logloss: 0.269375
2022-01-26 21:40:52,167 P65841 INFO Save best model: monitor(max): 0.946668
2022-01-26 21:40:52,169 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:40:52,210 P65841 INFO Train loss: 0.267168
2022-01-26 21:40:52,210 P65841 INFO ************ Epoch=14 end ************
2022-01-26 21:40:57,643 P65841 INFO [Metrics] AUC: 0.947631 - logloss: 0.267312
2022-01-26 21:40:57,643 P65841 INFO Save best model: monitor(max): 0.947631
2022-01-26 21:40:57,646 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:40:57,696 P65841 INFO Train loss: 0.263765
2022-01-26 21:40:57,696 P65841 INFO ************ Epoch=15 end ************
2022-01-26 21:41:03,333 P65841 INFO [Metrics] AUC: 0.948734 - logloss: 0.264932
2022-01-26 21:41:03,334 P65841 INFO Save best model: monitor(max): 0.948734
2022-01-26 21:41:03,336 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:41:03,380 P65841 INFO Train loss: 0.259212
2022-01-26 21:41:03,380 P65841 INFO ************ Epoch=16 end ************
2022-01-26 21:41:09,027 P65841 INFO [Metrics] AUC: 0.949597 - logloss: 0.262890
2022-01-26 21:41:09,027 P65841 INFO Save best model: monitor(max): 0.949597
2022-01-26 21:41:09,030 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:41:09,076 P65841 INFO Train loss: 0.255359
2022-01-26 21:41:09,076 P65841 INFO ************ Epoch=17 end ************
2022-01-26 21:41:14,556 P65841 INFO [Metrics] AUC: 0.950494 - logloss: 0.261445
2022-01-26 21:41:14,556 P65841 INFO Save best model: monitor(max): 0.950494
2022-01-26 21:41:14,559 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:41:14,601 P65841 INFO Train loss: 0.250753
2022-01-26 21:41:14,602 P65841 INFO ************ Epoch=18 end ************
2022-01-26 21:41:20,032 P65841 INFO [Metrics] AUC: 0.951381 - logloss: 0.259376
2022-01-26 21:41:20,032 P65841 INFO Save best model: monitor(max): 0.951381
2022-01-26 21:41:20,035 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:41:20,076 P65841 INFO Train loss: 0.247413
2022-01-26 21:41:20,076 P65841 INFO ************ Epoch=19 end ************
2022-01-26 21:41:25,481 P65841 INFO [Metrics] AUC: 0.952199 - logloss: 0.257617
2022-01-26 21:41:25,481 P65841 INFO Save best model: monitor(max): 0.952199
2022-01-26 21:41:25,484 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:41:25,527 P65841 INFO Train loss: 0.244376
2022-01-26 21:41:25,527 P65841 INFO ************ Epoch=20 end ************
2022-01-26 21:41:31,015 P65841 INFO [Metrics] AUC: 0.952856 - logloss: 0.256387
2022-01-26 21:41:31,016 P65841 INFO Save best model: monitor(max): 0.952856
2022-01-26 21:41:31,019 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:41:31,061 P65841 INFO Train loss: 0.240888
2022-01-26 21:41:31,061 P65841 INFO ************ Epoch=21 end ************
2022-01-26 21:41:36,645 P65841 INFO [Metrics] AUC: 0.953509 - logloss: 0.255099
2022-01-26 21:41:36,645 P65841 INFO Save best model: monitor(max): 0.953509
2022-01-26 21:41:36,648 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:41:36,691 P65841 INFO Train loss: 0.237881
2022-01-26 21:41:36,691 P65841 INFO ************ Epoch=22 end ************
2022-01-26 21:41:42,239 P65841 INFO [Metrics] AUC: 0.954240 - logloss: 0.253708
2022-01-26 21:41:42,240 P65841 INFO Save best model: monitor(max): 0.954240
2022-01-26 21:41:42,242 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:41:42,285 P65841 INFO Train loss: 0.235184
2022-01-26 21:41:42,285 P65841 INFO ************ Epoch=23 end ************
2022-01-26 21:41:47,777 P65841 INFO [Metrics] AUC: 0.954781 - logloss: 0.252182
2022-01-26 21:41:47,778 P65841 INFO Save best model: monitor(max): 0.954781
2022-01-26 21:41:47,780 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:41:47,821 P65841 INFO Train loss: 0.231764
2022-01-26 21:41:47,821 P65841 INFO ************ Epoch=24 end ************
2022-01-26 21:41:53,326 P65841 INFO [Metrics] AUC: 0.955326 - logloss: 0.251521
2022-01-26 21:41:53,326 P65841 INFO Save best model: monitor(max): 0.955326
2022-01-26 21:41:53,330 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:41:53,376 P65841 INFO Train loss: 0.228126
2022-01-26 21:41:53,376 P65841 INFO ************ Epoch=25 end ************
2022-01-26 21:41:58,888 P65841 INFO [Metrics] AUC: 0.956106 - logloss: 0.249127
2022-01-26 21:41:58,889 P65841 INFO Save best model: monitor(max): 0.956106
2022-01-26 21:41:58,892 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:41:58,934 P65841 INFO Train loss: 0.224872
2022-01-26 21:41:58,934 P65841 INFO ************ Epoch=26 end ************
2022-01-26 21:42:04,470 P65841 INFO [Metrics] AUC: 0.956830 - logloss: 0.247078
2022-01-26 21:42:04,471 P65841 INFO Save best model: monitor(max): 0.956830
2022-01-26 21:42:04,474 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:42:04,517 P65841 INFO Train loss: 0.221521
2022-01-26 21:42:04,517 P65841 INFO ************ Epoch=27 end ************
2022-01-26 21:42:10,023 P65841 INFO [Metrics] AUC: 0.957472 - logloss: 0.245467
2022-01-26 21:42:10,023 P65841 INFO Save best model: monitor(max): 0.957472
2022-01-26 21:42:10,026 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:42:10,071 P65841 INFO Train loss: 0.217927
2022-01-26 21:42:10,071 P65841 INFO ************ Epoch=28 end ************
2022-01-26 21:42:15,628 P65841 INFO [Metrics] AUC: 0.958226 - logloss: 0.243998
2022-01-26 21:42:15,628 P65841 INFO Save best model: monitor(max): 0.958226
2022-01-26 21:42:15,631 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:42:15,675 P65841 INFO Train loss: 0.214801
2022-01-26 21:42:15,676 P65841 INFO ************ Epoch=29 end ************
2022-01-26 21:42:21,171 P65841 INFO [Metrics] AUC: 0.959028 - logloss: 0.241159
2022-01-26 21:42:21,172 P65841 INFO Save best model: monitor(max): 0.959028
2022-01-26 21:42:21,175 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:42:21,219 P65841 INFO Train loss: 0.210193
2022-01-26 21:42:21,220 P65841 INFO ************ Epoch=30 end ************
2022-01-26 21:42:26,745 P65841 INFO [Metrics] AUC: 0.959589 - logloss: 0.239799
2022-01-26 21:42:26,746 P65841 INFO Save best model: monitor(max): 0.959589
2022-01-26 21:42:26,749 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:42:26,790 P65841 INFO Train loss: 0.205817
2022-01-26 21:42:26,790 P65841 INFO ************ Epoch=31 end ************
2022-01-26 21:42:32,303 P65841 INFO [Metrics] AUC: 0.960435 - logloss: 0.237889
2022-01-26 21:42:32,303 P65841 INFO Save best model: monitor(max): 0.960435
2022-01-26 21:42:32,306 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:42:32,349 P65841 INFO Train loss: 0.201570
2022-01-26 21:42:32,349 P65841 INFO ************ Epoch=32 end ************
2022-01-26 21:42:37,443 P65841 INFO [Metrics] AUC: 0.961042 - logloss: 0.235913
2022-01-26 21:42:37,443 P65841 INFO Save best model: monitor(max): 0.961042
2022-01-26 21:42:37,446 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:42:37,487 P65841 INFO Train loss: 0.197176
2022-01-26 21:42:37,488 P65841 INFO ************ Epoch=33 end ************
2022-01-26 21:42:41,582 P65841 INFO [Metrics] AUC: 0.961852 - logloss: 0.233224
2022-01-26 21:42:41,582 P65841 INFO Save best model: monitor(max): 0.961852
2022-01-26 21:42:41,585 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:42:41,627 P65841 INFO Train loss: 0.192013
2022-01-26 21:42:41,627 P65841 INFO ************ Epoch=34 end ************
2022-01-26 21:42:46,228 P65841 INFO [Metrics] AUC: 0.962717 - logloss: 0.230499
2022-01-26 21:42:46,229 P65841 INFO Save best model: monitor(max): 0.962717
2022-01-26 21:42:46,231 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:42:46,272 P65841 INFO Train loss: 0.186243
2022-01-26 21:42:46,273 P65841 INFO ************ Epoch=35 end ************
2022-01-26 21:42:52,004 P65841 INFO [Metrics] AUC: 0.963366 - logloss: 0.228202
2022-01-26 21:42:52,004 P65841 INFO Save best model: monitor(max): 0.963366
2022-01-26 21:42:52,007 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:42:52,049 P65841 INFO Train loss: 0.180078
2022-01-26 21:42:52,050 P65841 INFO ************ Epoch=36 end ************
2022-01-26 21:42:57,894 P65841 INFO [Metrics] AUC: 0.964104 - logloss: 0.225730
2022-01-26 21:42:57,895 P65841 INFO Save best model: monitor(max): 0.964104
2022-01-26 21:42:57,898 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:42:57,939 P65841 INFO Train loss: 0.173908
2022-01-26 21:42:57,939 P65841 INFO ************ Epoch=37 end ************
2022-01-26 21:43:03,688 P65841 INFO [Metrics] AUC: 0.964865 - logloss: 0.223298
2022-01-26 21:43:03,688 P65841 INFO Save best model: monitor(max): 0.964865
2022-01-26 21:43:03,693 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:43:03,738 P65841 INFO Train loss: 0.167541
2022-01-26 21:43:03,739 P65841 INFO ************ Epoch=38 end ************
2022-01-26 21:43:09,548 P65841 INFO [Metrics] AUC: 0.965642 - logloss: 0.221241
2022-01-26 21:43:09,548 P65841 INFO Save best model: monitor(max): 0.965642
2022-01-26 21:43:09,552 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:43:09,598 P65841 INFO Train loss: 0.161663
2022-01-26 21:43:09,599 P65841 INFO ************ Epoch=39 end ************
2022-01-26 21:43:15,349 P65841 INFO [Metrics] AUC: 0.966456 - logloss: 0.218232
2022-01-26 21:43:15,349 P65841 INFO Save best model: monitor(max): 0.966456
2022-01-26 21:43:15,353 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:43:15,399 P65841 INFO Train loss: 0.155831
2022-01-26 21:43:15,399 P65841 INFO ************ Epoch=40 end ************
2022-01-26 21:43:21,103 P65841 INFO [Metrics] AUC: 0.967024 - logloss: 0.217010
2022-01-26 21:43:21,103 P65841 INFO Save best model: monitor(max): 0.967024
2022-01-26 21:43:21,106 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:43:21,148 P65841 INFO Train loss: 0.150570
2022-01-26 21:43:21,148 P65841 INFO ************ Epoch=41 end ************
2022-01-26 21:43:26,946 P65841 INFO [Metrics] AUC: 0.967791 - logloss: 0.214635
2022-01-26 21:43:26,947 P65841 INFO Save best model: monitor(max): 0.967791
2022-01-26 21:43:26,950 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:43:26,994 P65841 INFO Train loss: 0.145004
2022-01-26 21:43:26,994 P65841 INFO ************ Epoch=42 end ************
2022-01-26 21:43:32,699 P65841 INFO [Metrics] AUC: 0.968071 - logloss: 0.214039
2022-01-26 21:43:32,700 P65841 INFO Save best model: monitor(max): 0.968071
2022-01-26 21:43:32,702 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:43:32,759 P65841 INFO Train loss: 0.139420
2022-01-26 21:43:32,759 P65841 INFO ************ Epoch=43 end ************
2022-01-26 21:43:38,492 P65841 INFO [Metrics] AUC: 0.968502 - logloss: 0.213977
2022-01-26 21:43:38,493 P65841 INFO Save best model: monitor(max): 0.968502
2022-01-26 21:43:38,496 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:43:38,538 P65841 INFO Train loss: 0.135939
2022-01-26 21:43:38,538 P65841 INFO ************ Epoch=44 end ************
2022-01-26 21:43:44,202 P65841 INFO [Metrics] AUC: 0.969036 - logloss: 0.214659
2022-01-26 21:43:44,203 P65841 INFO Save best model: monitor(max): 0.969036
2022-01-26 21:43:44,205 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:43:44,251 P65841 INFO Train loss: 0.131205
2022-01-26 21:43:44,251 P65841 INFO ************ Epoch=45 end ************
2022-01-26 21:43:49,903 P65841 INFO [Metrics] AUC: 0.969561 - logloss: 0.211588
2022-01-26 21:43:49,904 P65841 INFO Save best model: monitor(max): 0.969561
2022-01-26 21:43:49,906 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:43:49,949 P65841 INFO Train loss: 0.126558
2022-01-26 21:43:49,949 P65841 INFO ************ Epoch=46 end ************
2022-01-26 21:43:55,602 P65841 INFO [Metrics] AUC: 0.969750 - logloss: 0.211761
2022-01-26 21:43:55,602 P65841 INFO Save best model: monitor(max): 0.969750
2022-01-26 21:43:55,605 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:43:55,647 P65841 INFO Train loss: 0.122449
2022-01-26 21:43:55,647 P65841 INFO ************ Epoch=47 end ************
2022-01-26 21:44:01,251 P65841 INFO [Metrics] AUC: 0.969933 - logloss: 0.213279
2022-01-26 21:44:01,251 P65841 INFO Save best model: monitor(max): 0.969933
2022-01-26 21:44:01,254 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:44:01,299 P65841 INFO Train loss: 0.119003
2022-01-26 21:44:01,299 P65841 INFO ************ Epoch=48 end ************
2022-01-26 21:44:06,926 P65841 INFO [Metrics] AUC: 0.970263 - logloss: 0.213553
2022-01-26 21:44:06,927 P65841 INFO Save best model: monitor(max): 0.970263
2022-01-26 21:44:06,931 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:44:06,973 P65841 INFO Train loss: 0.115279
2022-01-26 21:44:06,974 P65841 INFO ************ Epoch=49 end ************
2022-01-26 21:44:12,654 P65841 INFO [Metrics] AUC: 0.970452 - logloss: 0.213063
2022-01-26 21:44:12,655 P65841 INFO Save best model: monitor(max): 0.970452
2022-01-26 21:44:12,658 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:44:12,702 P65841 INFO Train loss: 0.112103
2022-01-26 21:44:12,702 P65841 INFO ************ Epoch=50 end ************
2022-01-26 21:44:18,409 P65841 INFO [Metrics] AUC: 0.970666 - logloss: 0.213101
2022-01-26 21:44:18,410 P65841 INFO Save best model: monitor(max): 0.970666
2022-01-26 21:44:18,413 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:44:18,454 P65841 INFO Train loss: 0.109035
2022-01-26 21:44:18,454 P65841 INFO ************ Epoch=51 end ************
2022-01-26 21:44:24,160 P65841 INFO [Metrics] AUC: 0.970697 - logloss: 0.213729
2022-01-26 21:44:24,161 P65841 INFO Save best model: monitor(max): 0.970697
2022-01-26 21:44:24,163 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:44:24,205 P65841 INFO Train loss: 0.106604
2022-01-26 21:44:24,205 P65841 INFO ************ Epoch=52 end ************
2022-01-26 21:44:29,786 P65841 INFO [Metrics] AUC: 0.970839 - logloss: 0.216301
2022-01-26 21:44:29,787 P65841 INFO Save best model: monitor(max): 0.970839
2022-01-26 21:44:29,790 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:44:29,832 P65841 INFO Train loss: 0.103116
2022-01-26 21:44:29,832 P65841 INFO ************ Epoch=53 end ************
2022-01-26 21:44:35,471 P65841 INFO [Metrics] AUC: 0.970963 - logloss: 0.218254
2022-01-26 21:44:35,471 P65841 INFO Save best model: monitor(max): 0.970963
2022-01-26 21:44:35,474 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:44:35,520 P65841 INFO Train loss: 0.100637
2022-01-26 21:44:35,520 P65841 INFO ************ Epoch=54 end ************
2022-01-26 21:44:41,058 P65841 INFO [Metrics] AUC: 0.971223 - logloss: 0.217298
2022-01-26 21:44:41,058 P65841 INFO Save best model: monitor(max): 0.971223
2022-01-26 21:44:41,061 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:44:41,103 P65841 INFO Train loss: 0.098331
2022-01-26 21:44:41,103 P65841 INFO ************ Epoch=55 end ************
2022-01-26 21:44:46,740 P65841 INFO [Metrics] AUC: 0.971223 - logloss: 0.219944
2022-01-26 21:44:46,740 P65841 INFO Monitor(max) STOP: 0.971223 !
2022-01-26 21:44:46,740 P65841 INFO Reduce learning rate on plateau: 0.000100
2022-01-26 21:44:46,740 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:44:46,783 P65841 INFO Train loss: 0.095590
2022-01-26 21:44:46,783 P65841 INFO ************ Epoch=56 end ************
2022-01-26 21:44:52,458 P65841 INFO [Metrics] AUC: 0.971269 - logloss: 0.218183
2022-01-26 21:44:52,459 P65841 INFO Save best model: monitor(max): 0.971269
2022-01-26 21:44:52,462 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:44:52,504 P65841 INFO Train loss: 0.092133
2022-01-26 21:44:52,504 P65841 INFO ************ Epoch=57 end ************
2022-01-26 21:44:58,283 P65841 INFO [Metrics] AUC: 0.971317 - logloss: 0.218288
2022-01-26 21:44:58,283 P65841 INFO Save best model: monitor(max): 0.971317
2022-01-26 21:44:58,286 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:44:58,330 P65841 INFO Train loss: 0.091356
2022-01-26 21:44:58,330 P65841 INFO ************ Epoch=58 end ************
2022-01-26 21:45:04,243 P65841 INFO [Metrics] AUC: 0.971360 - logloss: 0.217682
2022-01-26 21:45:04,243 P65841 INFO Save best model: monitor(max): 0.971360
2022-01-26 21:45:04,246 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:45:04,289 P65841 INFO Train loss: 0.091065
2022-01-26 21:45:04,289 P65841 INFO ************ Epoch=59 end ************
2022-01-26 21:45:10,055 P65841 INFO [Metrics] AUC: 0.971415 - logloss: 0.217660
2022-01-26 21:45:10,056 P65841 INFO Save best model: monitor(max): 0.971415
2022-01-26 21:45:10,059 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:45:10,102 P65841 INFO Train loss: 0.090115
2022-01-26 21:45:10,102 P65841 INFO ************ Epoch=60 end ************
2022-01-26 21:45:15,844 P65841 INFO [Metrics] AUC: 0.971419 - logloss: 0.217452
2022-01-26 21:45:15,844 P65841 INFO Save best model: monitor(max): 0.971419
2022-01-26 21:45:15,847 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:45:15,889 P65841 INFO Train loss: 0.089890
2022-01-26 21:45:15,889 P65841 INFO ************ Epoch=61 end ************
2022-01-26 21:45:21,678 P65841 INFO [Metrics] AUC: 0.971434 - logloss: 0.218243
2022-01-26 21:45:21,678 P65841 INFO Save best model: monitor(max): 0.971434
2022-01-26 21:45:21,681 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:45:21,724 P65841 INFO Train loss: 0.089445
2022-01-26 21:45:21,725 P65841 INFO ************ Epoch=62 end ************
2022-01-26 21:45:27,368 P65841 INFO [Metrics] AUC: 0.971383 - logloss: 0.217897
2022-01-26 21:45:27,368 P65841 INFO Monitor(max) STOP: 0.971383 !
2022-01-26 21:45:27,368 P65841 INFO Reduce learning rate on plateau: 0.000010
2022-01-26 21:45:27,369 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:45:27,410 P65841 INFO Train loss: 0.088869
2022-01-26 21:45:27,410 P65841 INFO ************ Epoch=63 end ************
2022-01-26 21:45:32,916 P65841 INFO [Metrics] AUC: 0.971390 - logloss: 0.218013
2022-01-26 21:45:32,917 P65841 INFO Monitor(max) STOP: 0.971390 !
2022-01-26 21:45:32,917 P65841 INFO Reduce learning rate on plateau: 0.000001
2022-01-26 21:45:32,917 P65841 INFO Early stopping at epoch=64
2022-01-26 21:45:32,917 P65841 INFO --- 50/50 batches finished ---
2022-01-26 21:45:32,961 P65841 INFO Train loss: 0.088500
2022-01-26 21:45:32,962 P65841 INFO Training finished.
2022-01-26 21:45:32,962 P65841 INFO Load best model: /home/ma-user/work/FuxiCTRv1.1/benchmarks/Frappe/AFM_frappe_x1/frappe_x1_04e961e9/AFM_frappe_x1_011_2d591f68.model
2022-01-26 21:45:33,002 P65841 INFO ****** Validation evaluation ******
2022-01-26 21:45:33,418 P65841 INFO [Metrics] AUC: 0.971434 - logloss: 0.218243
2022-01-26 21:45:33,462 P65841 INFO ******** Test evaluation ********
2022-01-26 21:45:33,463 P65841 INFO Loading data...
2022-01-26 21:45:33,463 P65841 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-01-26 21:45:33,466 P65841 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-01-26 21:45:33,466 P65841 INFO Loading test data done.
2022-01-26 21:45:33,729 P65841 INFO [Metrics] AUC: 0.969733 - logloss: 0.226424

```
