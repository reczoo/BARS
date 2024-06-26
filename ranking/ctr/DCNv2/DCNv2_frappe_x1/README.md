## DCNv2_frappe_x1

A hands-on guide to run the DCNv2 model on the Frappe_x1 dataset.

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
  fuxictr: 1.1.0
  ```

### Dataset

Dataset ID: [Frappe_x1](https://github.com/reczoo/Datasets/tree/main/Frappe/Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DCNv2](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DCNv2.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DCNv2_frappe_x1_tuner_config_03](./DCNv2_frappe_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd DCNv2_frappe_x1
   nohup python run_expid.py --config ./DCNv2_frappe_x1_tuner_config_03 --expid DCNv2_frappe_x1_007_c207b717 --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

Total 5 runs:

| Runs | AUC              | logloss          |
|:----:|:----------------:|:----------------:|
| 1    | 0.984478         | 0.149149         |
| 2    | 0.984184         | 0.154896         |
| 3    | 0.983331         | 0.153688         |
| 4    | 0.983632         | 0.150610         |
| 5    | 0.983495         | 0.151837         |
| Avg  | 0.983824         | 0.152036         |
| Std  | &#177;0.00043485 | &#177;0.00206478 |

### Logs

```python
2022-11-01 19:58:23,208 P10947 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "dnn_activations": "relu",
    "embedding_dim": "10",
    "embedding_regularizer": "0.05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "low_rank": "32",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCNv2",
    "model_id": "DCNv2_frappe_x1_007_c207b717",
    "model_root": "./Frappe/DCN_frappe_x1/",
    "model_structure": "parallel",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.4",
    "net_regularizer": "0",
    "num_cross_layers": "4",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_dnn_hidden_units": "[400, 400, 400]",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "stacked_dnn_hidden_units": "[500, 500, 500]",
    "task": "binary_classification",
    "test_data": "../data/Frappe/Frappe_x1/test.csv",
    "train_data": "../data/Frappe/Frappe_x1/train.csv",
    "use_hdf5": "True",
    "use_low_rank_mixture": "False",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-11-01 19:58:23,209 P10947 INFO Set up feature encoder...
2022-11-01 19:58:23,209 P10947 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-11-01 19:58:23,210 P10947 INFO Loading data...
2022-11-01 19:58:23,213 P10947 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-11-01 19:58:23,228 P10947 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-11-01 19:58:23,234 P10947 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-11-01 19:58:23,235 P10947 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-11-01 19:58:23,235 P10947 INFO Loading train data done.
2022-11-01 19:58:26,382 P10947 INFO Total number of parameters: 458391.
2022-11-01 19:58:26,383 P10947 INFO Start training: 50 batches/epoch
2022-11-01 19:58:26,383 P10947 INFO ************ Epoch=1 start ************
2022-11-01 19:58:29,575 P10947 INFO [Metrics] AUC: 0.935947 - logloss: 0.585378
2022-11-01 19:58:29,576 P10947 INFO Save best model: monitor(max): 0.935947
2022-11-01 19:58:29,581 P10947 INFO --- 50/50 batches finished ---
2022-11-01 19:58:29,616 P10947 INFO Train loss: 0.392988
2022-11-01 19:58:29,616 P10947 INFO ************ Epoch=1 end ************
2022-11-01 19:58:33,127 P10947 INFO [Metrics] AUC: 0.944093 - logloss: 0.282885
2022-11-01 19:58:33,127 P10947 INFO Save best model: monitor(max): 0.944093
2022-11-01 19:58:33,132 P10947 INFO --- 50/50 batches finished ---
2022-11-01 19:58:33,168 P10947 INFO Train loss: 0.302932
2022-11-01 19:58:33,169 P10947 INFO ************ Epoch=2 end ************
2022-11-01 19:58:37,355 P10947 INFO [Metrics] AUC: 0.961049 - logloss: 0.246521
2022-11-01 19:58:37,355 P10947 INFO Save best model: monitor(max): 0.961049
2022-11-01 19:58:37,360 P10947 INFO --- 50/50 batches finished ---
2022-11-01 19:58:37,395 P10947 INFO Train loss: 0.272936
2022-11-01 19:58:37,395 P10947 INFO ************ Epoch=3 end ************
2022-11-01 19:58:41,568 P10947 INFO [Metrics] AUC: 0.969913 - logloss: 0.201559
2022-11-01 19:58:41,568 P10947 INFO Save best model: monitor(max): 0.969913
2022-11-01 19:58:41,575 P10947 INFO --- 50/50 batches finished ---
2022-11-01 19:58:41,619 P10947 INFO Train loss: 0.242380
2022-11-01 19:58:41,619 P10947 INFO ************ Epoch=4 end ************
2022-11-01 19:58:45,857 P10947 INFO [Metrics] AUC: 0.974447 - logloss: 0.193042
2022-11-01 19:58:45,858 P10947 INFO Save best model: monitor(max): 0.974447
2022-11-01 19:58:45,862 P10947 INFO --- 50/50 batches finished ---
2022-11-01 19:58:45,904 P10947 INFO Train loss: 0.222645
2022-11-01 19:58:45,904 P10947 INFO ************ Epoch=5 end ************
2022-11-01 19:58:50,043 P10947 INFO [Metrics] AUC: 0.977002 - logloss: 0.171289
2022-11-01 19:58:50,044 P10947 INFO Save best model: monitor(max): 0.977002
2022-11-01 19:58:50,049 P10947 INFO --- 50/50 batches finished ---
2022-11-01 19:58:50,088 P10947 INFO Train loss: 0.213055
2022-11-01 19:58:50,088 P10947 INFO ************ Epoch=6 end ************
2022-11-01 19:58:54,267 P10947 INFO [Metrics] AUC: 0.977740 - logloss: 0.177547
2022-11-01 19:58:54,268 P10947 INFO Save best model: monitor(max): 0.977740
2022-11-01 19:58:54,273 P10947 INFO --- 50/50 batches finished ---
2022-11-01 19:58:54,303 P10947 INFO Train loss: 0.201534
2022-11-01 19:58:54,303 P10947 INFO ************ Epoch=7 end ************
2022-11-01 19:58:58,461 P10947 INFO [Metrics] AUC: 0.979057 - logloss: 0.172863
2022-11-01 19:58:58,462 P10947 INFO Save best model: monitor(max): 0.979057
2022-11-01 19:58:58,467 P10947 INFO --- 50/50 batches finished ---
2022-11-01 19:58:58,499 P10947 INFO Train loss: 0.198045
2022-11-01 19:58:58,499 P10947 INFO ************ Epoch=8 end ************
2022-11-01 19:59:02,688 P10947 INFO [Metrics] AUC: 0.979650 - logloss: 0.160135
2022-11-01 19:59:02,688 P10947 INFO Save best model: monitor(max): 0.979650
2022-11-01 19:59:02,693 P10947 INFO --- 50/50 batches finished ---
2022-11-01 19:59:02,723 P10947 INFO Train loss: 0.193855
2022-11-01 19:59:02,723 P10947 INFO ************ Epoch=9 end ************
2022-11-01 19:59:07,537 P10947 INFO [Metrics] AUC: 0.979855 - logloss: 0.183747
2022-11-01 19:59:07,538 P10947 INFO Save best model: monitor(max): 0.979855
2022-11-01 19:59:07,545 P10947 INFO --- 50/50 batches finished ---
2022-11-01 19:59:07,575 P10947 INFO Train loss: 0.189909
2022-11-01 19:59:07,576 P10947 INFO ************ Epoch=10 end ************
2022-11-01 19:59:13,887 P10947 INFO [Metrics] AUC: 0.977224 - logloss: 0.210676
2022-11-01 19:59:13,888 P10947 INFO Monitor(max) STOP: 0.977224 !
2022-11-01 19:59:13,888 P10947 INFO Reduce learning rate on plateau: 0.000100
2022-11-01 19:59:13,888 P10947 INFO --- 50/50 batches finished ---
2022-11-01 19:59:13,917 P10947 INFO Train loss: 0.186875
2022-11-01 19:59:13,917 P10947 INFO ************ Epoch=11 end ************
2022-11-01 19:59:20,251 P10947 INFO [Metrics] AUC: 0.982924 - logloss: 0.144500
2022-11-01 19:59:20,251 P10947 INFO Save best model: monitor(max): 0.982924
2022-11-01 19:59:20,256 P10947 INFO --- 50/50 batches finished ---
2022-11-01 19:59:20,297 P10947 INFO Train loss: 0.160064
2022-11-01 19:59:20,298 P10947 INFO ************ Epoch=12 end ************
2022-11-01 19:59:26,682 P10947 INFO [Metrics] AUC: 0.983878 - logloss: 0.142811
2022-11-01 19:59:26,683 P10947 INFO Save best model: monitor(max): 0.983878
2022-11-01 19:59:26,690 P10947 INFO --- 50/50 batches finished ---
2022-11-01 19:59:26,720 P10947 INFO Train loss: 0.136297
2022-11-01 19:59:26,720 P10947 INFO ************ Epoch=13 end ************
2022-11-01 19:59:32,747 P10947 INFO [Metrics] AUC: 0.984304 - logloss: 0.143330
2022-11-01 19:59:32,748 P10947 INFO Save best model: monitor(max): 0.984304
2022-11-01 19:59:32,754 P10947 INFO --- 50/50 batches finished ---
2022-11-01 19:59:32,804 P10947 INFO Train loss: 0.121201
2022-11-01 19:59:32,804 P10947 INFO ************ Epoch=14 end ************
2022-11-01 19:59:39,106 P10947 INFO [Metrics] AUC: 0.984540 - logloss: 0.145045
2022-11-01 19:59:39,107 P10947 INFO Save best model: monitor(max): 0.984540
2022-11-01 19:59:39,113 P10947 INFO --- 50/50 batches finished ---
2022-11-01 19:59:39,176 P10947 INFO Train loss: 0.110073
2022-11-01 19:59:39,176 P10947 INFO ************ Epoch=15 end ************
2022-11-01 19:59:45,538 P10947 INFO [Metrics] AUC: 0.984724 - logloss: 0.145804
2022-11-01 19:59:45,539 P10947 INFO Save best model: monitor(max): 0.984724
2022-11-01 19:59:45,546 P10947 INFO --- 50/50 batches finished ---
2022-11-01 19:59:45,589 P10947 INFO Train loss: 0.101770
2022-11-01 19:59:45,589 P10947 INFO ************ Epoch=16 end ************
2022-11-01 19:59:50,633 P10947 INFO [Metrics] AUC: 0.984840 - logloss: 0.148803
2022-11-01 19:59:50,634 P10947 INFO Save best model: monitor(max): 0.984840
2022-11-01 19:59:50,642 P10947 INFO --- 50/50 batches finished ---
2022-11-01 19:59:50,695 P10947 INFO Train loss: 0.095348
2022-11-01 19:59:50,695 P10947 INFO ************ Epoch=17 end ************
2022-11-01 19:59:55,713 P10947 INFO [Metrics] AUC: 0.984678 - logloss: 0.151752
2022-11-01 19:59:55,714 P10947 INFO Monitor(max) STOP: 0.984678 !
2022-11-01 19:59:55,714 P10947 INFO Reduce learning rate on plateau: 0.000010
2022-11-01 19:59:55,714 P10947 INFO --- 50/50 batches finished ---
2022-11-01 19:59:55,776 P10947 INFO Train loss: 0.089402
2022-11-01 19:59:55,776 P10947 INFO ************ Epoch=18 end ************
2022-11-01 20:00:00,540 P10947 INFO [Metrics] AUC: 0.984766 - logloss: 0.151839
2022-11-01 20:00:00,541 P10947 INFO Monitor(max) STOP: 0.984766 !
2022-11-01 20:00:00,541 P10947 INFO Reduce learning rate on plateau: 0.000001
2022-11-01 20:00:00,541 P10947 INFO Early stopping at epoch=19
2022-11-01 20:00:00,541 P10947 INFO --- 50/50 batches finished ---
2022-11-01 20:00:00,600 P10947 INFO Train loss: 0.083298
2022-11-01 20:00:00,600 P10947 INFO Training finished.
2022-11-01 20:00:00,600 P10947 INFO Load best model: /home/FuxiCTR/benchmarks/Frappe/DCN_frappe_x1/frappe_x1_04e961e9/DCNv2_frappe_x1_007_c207b717.model
2022-11-01 20:00:04,205 P10947 INFO ****** Validation evaluation ******
2022-11-01 20:00:04,554 P10947 INFO [Metrics] AUC: 0.984840 - logloss: 0.148803
2022-11-01 20:00:04,622 P10947 INFO ******** Test evaluation ********
2022-11-01 20:00:04,623 P10947 INFO Loading data...
2022-11-01 20:00:04,623 P10947 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-11-01 20:00:04,626 P10947 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-11-01 20:00:04,626 P10947 INFO Loading test data done.
2022-11-01 20:00:04,916 P10947 INFO [Metrics] AUC: 0.984478 - logloss: 0.149149
```
