## AutoInt+_frappe_x1

A hands-on guide to run the AutoInt model on the Frappe_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

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
Dataset ID: [Frappe_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Frappe/README.md#Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [AutoInt](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt+_frappe_x1_tuner_config_03](./AutoInt+_frappe_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AutoInt+_frappe_x1
    nohup python run_expid.py --config ./AutoInt+_frappe_x1_tuner_config_03 --expid AutoInt_frappe_x1_014_a4ced2ad --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.984908 | 0.142825  |
| 2 | 0.984237 | 0.150649  |
| 3 | 0.983083 | 0.149242  |
| 4 | 0.983614 | 0.149170  |
| 5 | 0.982715 | 0.152741  |
| | | | 
| Avg | 0.983711 | 0.148925 |
| Std | &#177;0.00078807 | &#177;0.00331375 |


### Logs
```python
2022-01-24 12:25:05,136 P1980 INFO {
    "attention_dim": "128",
    "attention_layers": "5",
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.1",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AutoInt",
    "model_id": "AutoInt_frappe_x1_014_a4ced2ad",
    "model_root": "./Frappe/AutoInt_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_heads": "1",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Frappe/Frappe_x1/test.csv",
    "train_data": "../data/Frappe/Frappe_x1/train.csv",
    "use_hdf5": "True",
    "use_residual": "False",
    "use_scale": "True",
    "use_wide": "False",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-24 12:25:05,137 P1980 INFO Set up feature encoder...
2022-01-24 12:25:05,137 P1980 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-01-24 12:25:05,138 P1980 INFO Loading data...
2022-01-24 12:25:05,140 P1980 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-01-24 12:25:05,153 P1980 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-01-24 12:25:05,157 P1980 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-01-24 12:25:05,157 P1980 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-01-24 12:25:05,157 P1980 INFO Loading train data done.
2022-01-24 12:25:09,956 P1980 INFO Total number of parameters: 620900.
2022-01-24 12:25:09,957 P1980 INFO Start training: 50 batches/epoch
2022-01-24 12:25:09,957 P1980 INFO ************ Epoch=1 start ************
2022-01-24 12:25:28,112 P1980 INFO [Metrics] AUC: 0.935643 - logloss: 0.579823
2022-01-24 12:25:28,112 P1980 INFO Save best model: monitor(max): 0.935643
2022-01-24 12:25:28,118 P1980 INFO --- 50/50 batches finished ---
2022-01-24 12:25:28,155 P1980 INFO Train loss: 0.393003
2022-01-24 12:25:28,155 P1980 INFO ************ Epoch=1 end ************
2022-01-24 12:25:46,126 P1980 INFO [Metrics] AUC: 0.948975 - logloss: 0.270141
2022-01-24 12:25:46,127 P1980 INFO Save best model: monitor(max): 0.948975
2022-01-24 12:25:46,133 P1980 INFO --- 50/50 batches finished ---
2022-01-24 12:25:46,163 P1980 INFO Train loss: 0.306104
2022-01-24 12:25:46,163 P1980 INFO ************ Epoch=2 end ************
2022-01-24 12:26:04,031 P1980 INFO [Metrics] AUC: 0.966281 - logloss: 0.218559
2022-01-24 12:26:04,031 P1980 INFO Save best model: monitor(max): 0.966281
2022-01-24 12:26:04,038 P1980 INFO --- 50/50 batches finished ---
2022-01-24 12:26:04,076 P1980 INFO Train loss: 0.269736
2022-01-24 12:26:04,076 P1980 INFO ************ Epoch=3 end ************
2022-01-24 12:26:22,249 P1980 INFO [Metrics] AUC: 0.972078 - logloss: 0.205682
2022-01-24 12:26:22,249 P1980 INFO Save best model: monitor(max): 0.972078
2022-01-24 12:26:22,256 P1980 INFO --- 50/50 batches finished ---
2022-01-24 12:26:22,297 P1980 INFO Train loss: 0.249935
2022-01-24 12:26:22,297 P1980 INFO ************ Epoch=4 end ************
2022-01-24 12:26:40,504 P1980 INFO [Metrics] AUC: 0.974612 - logloss: 0.190516
2022-01-24 12:26:40,505 P1980 INFO Save best model: monitor(max): 0.974612
2022-01-24 12:26:40,511 P1980 INFO --- 50/50 batches finished ---
2022-01-24 12:26:40,539 P1980 INFO Train loss: 0.235111
2022-01-24 12:26:40,539 P1980 INFO ************ Epoch=5 end ************
2022-01-24 12:26:58,474 P1980 INFO [Metrics] AUC: 0.975615 - logloss: 0.188210
2022-01-24 12:26:58,475 P1980 INFO Save best model: monitor(max): 0.975615
2022-01-24 12:26:58,481 P1980 INFO --- 50/50 batches finished ---
2022-01-24 12:26:58,513 P1980 INFO Train loss: 0.228919
2022-01-24 12:26:58,513 P1980 INFO ************ Epoch=6 end ************
2022-01-24 12:27:11,685 P1980 INFO [Metrics] AUC: 0.976798 - logloss: 0.203701
2022-01-24 12:27:11,686 P1980 INFO Save best model: monitor(max): 0.976798
2022-01-24 12:27:11,693 P1980 INFO --- 50/50 batches finished ---
2022-01-24 12:27:11,723 P1980 INFO Train loss: 0.224933
2022-01-24 12:27:11,724 P1980 INFO ************ Epoch=7 end ************
2022-01-24 12:27:22,367 P1980 INFO [Metrics] AUC: 0.977863 - logloss: 0.166569
2022-01-24 12:27:22,367 P1980 INFO Save best model: monitor(max): 0.977863
2022-01-24 12:27:22,374 P1980 INFO --- 50/50 batches finished ---
2022-01-24 12:27:22,417 P1980 INFO Train loss: 0.220570
2022-01-24 12:27:22,417 P1980 INFO ************ Epoch=8 end ************
2022-01-24 12:27:33,106 P1980 INFO [Metrics] AUC: 0.978683 - logloss: 0.188115
2022-01-24 12:27:33,106 P1980 INFO Save best model: monitor(max): 0.978683
2022-01-24 12:27:33,113 P1980 INFO --- 50/50 batches finished ---
2022-01-24 12:27:33,146 P1980 INFO Train loss: 0.216184
2022-01-24 12:27:33,146 P1980 INFO ************ Epoch=9 end ************
2022-01-24 12:27:43,622 P1980 INFO [Metrics] AUC: 0.977088 - logloss: 0.171503
2022-01-24 12:27:43,623 P1980 INFO Monitor(max) STOP: 0.977088 !
2022-01-24 12:27:43,623 P1980 INFO Reduce learning rate on plateau: 0.000100
2022-01-24 12:27:43,623 P1980 INFO --- 50/50 batches finished ---
2022-01-24 12:27:43,652 P1980 INFO Train loss: 0.217377
2022-01-24 12:27:43,652 P1980 INFO ************ Epoch=10 end ************
2022-01-24 12:27:54,334 P1980 INFO [Metrics] AUC: 0.982868 - logloss: 0.144701
2022-01-24 12:27:54,334 P1980 INFO Save best model: monitor(max): 0.982868
2022-01-24 12:27:54,341 P1980 INFO --- 50/50 batches finished ---
2022-01-24 12:27:54,370 P1980 INFO Train loss: 0.182663
2022-01-24 12:27:54,370 P1980 INFO ************ Epoch=11 end ************
2022-01-24 12:28:05,028 P1980 INFO [Metrics] AUC: 0.984206 - logloss: 0.140951
2022-01-24 12:28:05,029 P1980 INFO Save best model: monitor(max): 0.984206
2022-01-24 12:28:05,039 P1980 INFO --- 50/50 batches finished ---
2022-01-24 12:28:05,071 P1980 INFO Train loss: 0.146924
2022-01-24 12:28:05,071 P1980 INFO ************ Epoch=12 end ************
2022-01-24 12:28:15,650 P1980 INFO [Metrics] AUC: 0.984813 - logloss: 0.142823
2022-01-24 12:28:15,651 P1980 INFO Save best model: monitor(max): 0.984813
2022-01-24 12:28:15,657 P1980 INFO --- 50/50 batches finished ---
2022-01-24 12:28:15,697 P1980 INFO Train loss: 0.125622
2022-01-24 12:28:15,697 P1980 INFO ************ Epoch=13 end ************
2022-01-24 12:28:26,391 P1980 INFO [Metrics] AUC: 0.985116 - logloss: 0.142445
2022-01-24 12:28:26,391 P1980 INFO Save best model: monitor(max): 0.985116
2022-01-24 12:28:26,398 P1980 INFO --- 50/50 batches finished ---
2022-01-24 12:28:26,427 P1980 INFO Train loss: 0.110943
2022-01-24 12:28:26,427 P1980 INFO ************ Epoch=14 end ************
2022-01-24 12:28:37,120 P1980 INFO [Metrics] AUC: 0.985441 - logloss: 0.142812
2022-01-24 12:28:37,120 P1980 INFO Save best model: monitor(max): 0.985441
2022-01-24 12:28:37,127 P1980 INFO --- 50/50 batches finished ---
2022-01-24 12:28:37,157 P1980 INFO Train loss: 0.099486
2022-01-24 12:28:37,157 P1980 INFO ************ Epoch=15 end ************
2022-01-24 12:28:47,864 P1980 INFO [Metrics] AUC: 0.985216 - logloss: 0.150305
2022-01-24 12:28:47,865 P1980 INFO Monitor(max) STOP: 0.985216 !
2022-01-24 12:28:47,865 P1980 INFO Reduce learning rate on plateau: 0.000010
2022-01-24 12:28:47,865 P1980 INFO --- 50/50 batches finished ---
2022-01-24 12:28:47,893 P1980 INFO Train loss: 0.090415
2022-01-24 12:28:47,893 P1980 INFO ************ Epoch=16 end ************
2022-01-24 12:28:58,655 P1980 INFO [Metrics] AUC: 0.985384 - logloss: 0.147964
2022-01-24 12:28:58,656 P1980 INFO Monitor(max) STOP: 0.985384 !
2022-01-24 12:28:58,656 P1980 INFO Reduce learning rate on plateau: 0.000001
2022-01-24 12:28:58,656 P1980 INFO Early stopping at epoch=17
2022-01-24 12:28:58,656 P1980 INFO --- 50/50 batches finished ---
2022-01-24 12:28:58,687 P1980 INFO Train loss: 0.082160
2022-01-24 12:28:58,687 P1980 INFO Training finished.
2022-01-24 12:28:58,687 P1980 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Frappe/AutoInt_frappe_x1/frappe_x1_04e961e9/AutoInt_frappe_x1_014_a4ced2ad.model
2022-01-24 12:28:58,747 P1980 INFO ****** Validation evaluation ******
2022-01-24 12:28:59,388 P1980 INFO [Metrics] AUC: 0.985441 - logloss: 0.142812
2022-01-24 12:28:59,433 P1980 INFO ******** Test evaluation ********
2022-01-24 12:28:59,433 P1980 INFO Loading data...
2022-01-24 12:28:59,434 P1980 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-01-24 12:28:59,437 P1980 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-01-24 12:28:59,437 P1980 INFO Loading test data done.
2022-01-24 12:28:59,771 P1980 INFO [Metrics] AUC: 0.984908 - logloss: 0.142825

```
