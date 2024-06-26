## MaskNet_frappe_x1

A hands-on guide to run the MaskNet model on the Frappe_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
  GPU: Tesla P100 16G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 11.4
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  fuxictr: 1.2.1
  ```

### Dataset
Dataset ID: [Frappe_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Frappe#Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.1](https://github.com/reczoo/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [MaskNet](https://github.com/reczoo/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/MaskNet.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [MaskNet_frappe_x1_tuner_config_05](./MaskNet_frappe_x1_tuner_config_05). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd MaskNet_frappe_x1
    nohup python run_expid.py --config ./MaskNet_frappe_x1_tuner_config_05 --expid MaskNet_frappe_x1_028_015da53e --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.983683 | 0.169580  |


### Logs
```python
2022-05-26 11:10:08,457 P41089 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "dnn_hidden_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "emb_layernorm": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.1",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "MaskNet",
    "model_id": "MaskNet_frappe_x1_028_015da53e",
    "model_root": "./Frappe/MaskNet_frappe_x1/",
    "model_type": "ParallelMaskNet",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_layernorm": "True",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_block_dim": "50",
    "parallel_num_blocks": "3",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "2",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Frappe/Frappe_x1/test.csv",
    "train_data": "../data/Frappe/Frappe_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-05-26 11:10:08,458 P41089 INFO Set up feature encoder...
2022-05-26 11:10:08,458 P41089 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-05-26 11:10:08,458 P41089 INFO Loading data...
2022-05-26 11:10:08,460 P41089 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-05-26 11:10:08,472 P41089 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-05-26 11:10:08,475 P41089 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-05-26 11:10:08,475 P41089 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-05-26 11:10:08,476 P41089 INFO Loading train data done.
2022-05-26 11:10:11,867 P41089 INFO Total number of parameters: 571691.
2022-05-26 11:10:11,868 P41089 INFO Start training: 50 batches/epoch
2022-05-26 11:10:11,868 P41089 INFO ************ Epoch=1 start ************
2022-05-26 11:10:15,638 P41089 INFO [Metrics] AUC: 0.935571 - logloss: 0.286634
2022-05-26 11:10:15,639 P41089 INFO Save best model: monitor(max): 0.935571
2022-05-26 11:10:15,646 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:10:15,690 P41089 INFO Train loss: 0.453737
2022-05-26 11:10:15,690 P41089 INFO ************ Epoch=1 end ************
2022-05-26 11:10:18,607 P41089 INFO [Metrics] AUC: 0.948610 - logloss: 0.257620
2022-05-26 11:10:18,608 P41089 INFO Save best model: monitor(max): 0.948610
2022-05-26 11:10:18,616 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:10:18,662 P41089 INFO Train loss: 0.304144
2022-05-26 11:10:18,662 P41089 INFO ************ Epoch=2 end ************
2022-05-26 11:10:21,442 P41089 INFO [Metrics] AUC: 0.961067 - logloss: 0.226449
2022-05-26 11:10:21,443 P41089 INFO Save best model: monitor(max): 0.961067
2022-05-26 11:10:21,451 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:10:21,493 P41089 INFO Train loss: 0.276798
2022-05-26 11:10:21,493 P41089 INFO ************ Epoch=3 end ************
2022-05-26 11:10:24,302 P41089 INFO [Metrics] AUC: 0.967092 - logloss: 0.206629
2022-05-26 11:10:24,303 P41089 INFO Save best model: monitor(max): 0.967092
2022-05-26 11:10:24,310 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:10:24,354 P41089 INFO Train loss: 0.258479
2022-05-26 11:10:24,355 P41089 INFO ************ Epoch=4 end ************
2022-05-26 11:10:27,281 P41089 INFO [Metrics] AUC: 0.972343 - logloss: 0.189962
2022-05-26 11:10:27,282 P41089 INFO Save best model: monitor(max): 0.972343
2022-05-26 11:10:27,290 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:10:27,340 P41089 INFO Train loss: 0.248212
2022-05-26 11:10:27,340 P41089 INFO ************ Epoch=5 end ************
2022-05-26 11:10:30,673 P41089 INFO [Metrics] AUC: 0.975098 - logloss: 0.179620
2022-05-26 11:10:30,674 P41089 INFO Save best model: monitor(max): 0.975098
2022-05-26 11:10:30,682 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:10:30,724 P41089 INFO Train loss: 0.238668
2022-05-26 11:10:30,724 P41089 INFO ************ Epoch=6 end ************
2022-05-26 11:10:33,945 P41089 INFO [Metrics] AUC: 0.975828 - logloss: 0.176516
2022-05-26 11:10:33,946 P41089 INFO Save best model: monitor(max): 0.975828
2022-05-26 11:10:33,951 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:10:33,994 P41089 INFO Train loss: 0.231681
2022-05-26 11:10:33,995 P41089 INFO ************ Epoch=7 end ************
2022-05-26 11:10:37,283 P41089 INFO [Metrics] AUC: 0.977100 - logloss: 0.170911
2022-05-26 11:10:37,284 P41089 INFO Save best model: monitor(max): 0.977100
2022-05-26 11:10:37,293 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:10:37,344 P41089 INFO Train loss: 0.226163
2022-05-26 11:10:37,344 P41089 INFO ************ Epoch=8 end ************
2022-05-26 11:10:40,652 P41089 INFO [Metrics] AUC: 0.977644 - logloss: 0.167801
2022-05-26 11:10:40,653 P41089 INFO Save best model: monitor(max): 0.977644
2022-05-26 11:10:40,662 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:10:40,722 P41089 INFO Train loss: 0.221765
2022-05-26 11:10:40,722 P41089 INFO ************ Epoch=9 end ************
2022-05-26 11:10:44,870 P41089 INFO [Metrics] AUC: 0.978607 - logloss: 0.165084
2022-05-26 11:10:44,870 P41089 INFO Save best model: monitor(max): 0.978607
2022-05-26 11:10:44,878 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:10:44,950 P41089 INFO Train loss: 0.220618
2022-05-26 11:10:44,950 P41089 INFO ************ Epoch=10 end ************
2022-05-26 11:10:49,350 P41089 INFO [Metrics] AUC: 0.978411 - logloss: 0.165176
2022-05-26 11:10:49,351 P41089 INFO Monitor(max) STOP: 0.978411 !
2022-05-26 11:10:49,351 P41089 INFO Reduce learning rate on plateau: 0.000100
2022-05-26 11:10:49,351 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:10:49,398 P41089 INFO Train loss: 0.215685
2022-05-26 11:10:49,398 P41089 INFO ************ Epoch=11 end ************
2022-05-26 11:10:53,732 P41089 INFO [Metrics] AUC: 0.982073 - logloss: 0.153091
2022-05-26 11:10:53,733 P41089 INFO Save best model: monitor(max): 0.982073
2022-05-26 11:10:53,740 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:10:53,776 P41089 INFO Train loss: 0.178363
2022-05-26 11:10:53,776 P41089 INFO ************ Epoch=12 end ************
2022-05-26 11:10:58,030 P41089 INFO [Metrics] AUC: 0.983386 - logloss: 0.151957
2022-05-26 11:10:58,031 P41089 INFO Save best model: monitor(max): 0.983386
2022-05-26 11:10:58,036 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:10:58,073 P41089 INFO Train loss: 0.141984
2022-05-26 11:10:58,073 P41089 INFO ************ Epoch=13 end ************
2022-05-26 11:11:02,145 P41089 INFO [Metrics] AUC: 0.983713 - logloss: 0.154576
2022-05-26 11:11:02,145 P41089 INFO Save best model: monitor(max): 0.983713
2022-05-26 11:11:02,150 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:11:02,186 P41089 INFO Train loss: 0.121058
2022-05-26 11:11:02,186 P41089 INFO ************ Epoch=14 end ************
2022-05-26 11:11:06,136 P41089 INFO [Metrics] AUC: 0.984101 - logloss: 0.156632
2022-05-26 11:11:06,137 P41089 INFO Save best model: monitor(max): 0.984101
2022-05-26 11:11:06,142 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:11:06,196 P41089 INFO Train loss: 0.107267
2022-05-26 11:11:06,196 P41089 INFO ************ Epoch=15 end ************
2022-05-26 11:11:10,097 P41089 INFO [Metrics] AUC: 0.983920 - logloss: 0.160924
2022-05-26 11:11:10,097 P41089 INFO Monitor(max) STOP: 0.983920 !
2022-05-26 11:11:10,098 P41089 INFO Reduce learning rate on plateau: 0.000010
2022-05-26 11:11:10,098 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:11:10,136 P41089 INFO Train loss: 0.098089
2022-05-26 11:11:10,136 P41089 INFO ************ Epoch=16 end ************
2022-05-26 11:11:13,840 P41089 INFO [Metrics] AUC: 0.984202 - logloss: 0.161646
2022-05-26 11:11:13,840 P41089 INFO Save best model: monitor(max): 0.984202
2022-05-26 11:11:13,846 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:11:13,881 P41089 INFO Train loss: 0.088258
2022-05-26 11:11:13,881 P41089 INFO ************ Epoch=17 end ************
2022-05-26 11:11:17,719 P41089 INFO [Metrics] AUC: 0.984271 - logloss: 0.162891
2022-05-26 11:11:17,720 P41089 INFO Save best model: monitor(max): 0.984271
2022-05-26 11:11:17,725 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:11:17,760 P41089 INFO Train loss: 0.086822
2022-05-26 11:11:17,760 P41089 INFO ************ Epoch=18 end ************
2022-05-26 11:11:21,597 P41089 INFO [Metrics] AUC: 0.984278 - logloss: 0.164273
2022-05-26 11:11:21,598 P41089 INFO Save best model: monitor(max): 0.984278
2022-05-26 11:11:21,603 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:11:21,638 P41089 INFO Train loss: 0.085214
2022-05-26 11:11:21,638 P41089 INFO ************ Epoch=19 end ************
2022-05-26 11:11:25,443 P41089 INFO [Metrics] AUC: 0.984296 - logloss: 0.165253
2022-05-26 11:11:25,444 P41089 INFO Save best model: monitor(max): 0.984296
2022-05-26 11:11:25,450 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:11:25,510 P41089 INFO Train loss: 0.083892
2022-05-26 11:11:25,510 P41089 INFO ************ Epoch=20 end ************
2022-05-26 11:11:29,594 P41089 INFO [Metrics] AUC: 0.984217 - logloss: 0.166702
2022-05-26 11:11:29,595 P41089 INFO Monitor(max) STOP: 0.984217 !
2022-05-26 11:11:29,595 P41089 INFO Reduce learning rate on plateau: 0.000001
2022-05-26 11:11:29,595 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:11:29,637 P41089 INFO Train loss: 0.082970
2022-05-26 11:11:29,638 P41089 INFO ************ Epoch=21 end ************
2022-05-26 11:11:33,792 P41089 INFO [Metrics] AUC: 0.984227 - logloss: 0.166614
2022-05-26 11:11:33,793 P41089 INFO Monitor(max) STOP: 0.984227 !
2022-05-26 11:11:33,793 P41089 INFO Reduce learning rate on plateau: 0.000001
2022-05-26 11:11:33,793 P41089 INFO Early stopping at epoch=22
2022-05-26 11:11:33,793 P41089 INFO --- 50/50 batches finished ---
2022-05-26 11:11:33,843 P41089 INFO Train loss: 0.081717
2022-05-26 11:11:33,843 P41089 INFO Training finished.
2022-05-26 11:11:33,843 P41089 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Frappe/MaskNet_frappe_x1/frappe_x1_04e961e9/MaskNet_frappe_x1_028_015da53e.model
2022-05-26 11:11:36,765 P41089 INFO ****** Validation evaluation ******
2022-05-26 11:11:37,170 P41089 INFO [Metrics] AUC: 0.984296 - logloss: 0.165253
2022-05-26 11:11:37,205 P41089 INFO ******** Test evaluation ********
2022-05-26 11:11:37,205 P41089 INFO Loading data...
2022-05-26 11:11:37,206 P41089 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-05-26 11:11:37,208 P41089 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-05-26 11:11:37,208 P41089 INFO Loading test data done.
2022-05-26 11:11:37,493 P41089 INFO [Metrics] AUC: 0.983683 - logloss: 0.169580

```
