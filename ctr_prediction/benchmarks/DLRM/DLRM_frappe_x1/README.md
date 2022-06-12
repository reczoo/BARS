## DLRM_frappe_x1

A hands-on guide to run the DLRM model on the Frappe_x1 dataset.

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
  fuxictr: 1.2.1
  ```

### Dataset
Dataset ID: [Frappe_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Frappe#Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [DLRM](https://github.com/xue-pai/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/DLRM.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DLRM_frappe_x1_tuner_config_02](./DLRM_frappe_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DLRM_frappe_x1
    nohup python run_expid.py --config ./DLRM_frappe_x1_tuner_config_02 --expid DLRM_frappe_x1_006_216831a3 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.983995 | 0.144441  |


### Logs
```python
2022-05-27 14:35:07,576 P18844 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "bottom_mlp_activations": "ReLU",
    "bottom_mlp_dropout": "0",
    "bottom_mlp_units": "None",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.1",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "0",
    "interaction_op": "cat",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DLRM",
    "model_id": "DLRM_frappe_x1_006_216831a3",
    "model_root": "./Frappe/DLRM_frappe_x1/",
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
    "top_mlp_activations": "ReLU",
    "top_mlp_dropout": "0.4",
    "top_mlp_units": "[400, 400, 400]",
    "train_data": "../data/Frappe/Frappe_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-05-27 14:35:07,576 P18844 INFO Set up feature encoder...
2022-05-27 14:35:07,576 P18844 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-05-27 14:35:07,576 P18844 INFO Loading data...
2022-05-27 14:35:07,578 P18844 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-05-27 14:35:07,589 P18844 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-05-27 14:35:07,593 P18844 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-05-27 14:35:07,593 P18844 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-05-27 14:35:07,593 P18844 INFO Loading train data done.
2022-05-27 14:35:10,767 P18844 INFO Total number of parameters: 417891.
2022-05-27 14:35:10,767 P18844 INFO Start training: 50 batches/epoch
2022-05-27 14:35:10,767 P18844 INFO ************ Epoch=1 start ************
2022-05-27 14:35:14,808 P18844 INFO [Metrics] AUC: 0.934844 - logloss: 0.634514
2022-05-27 14:35:14,809 P18844 INFO Save best model: monitor(max): 0.934844
2022-05-27 14:35:14,815 P18844 INFO --- 50/50 batches finished ---
2022-05-27 14:35:14,899 P18844 INFO Train loss: 0.415569
2022-05-27 14:35:14,899 P18844 INFO ************ Epoch=1 end ************
2022-05-27 14:35:18,769 P18844 INFO [Metrics] AUC: 0.946014 - logloss: 0.279160
2022-05-27 14:35:18,770 P18844 INFO Save best model: monitor(max): 0.946014
2022-05-27 14:35:18,776 P18844 INFO --- 50/50 batches finished ---
2022-05-27 14:35:18,841 P18844 INFO Train loss: 0.308836
2022-05-27 14:35:18,841 P18844 INFO ************ Epoch=2 end ************
2022-05-27 14:35:22,490 P18844 INFO [Metrics] AUC: 0.962075 - logloss: 0.224911
2022-05-27 14:35:22,491 P18844 INFO Save best model: monitor(max): 0.962075
2022-05-27 14:35:22,496 P18844 INFO --- 50/50 batches finished ---
2022-05-27 14:35:22,548 P18844 INFO Train loss: 0.276737
2022-05-27 14:35:22,548 P18844 INFO ************ Epoch=3 end ************
2022-05-27 14:35:25,881 P18844 INFO [Metrics] AUC: 0.969775 - logloss: 0.233633
2022-05-27 14:35:25,882 P18844 INFO Save best model: monitor(max): 0.969775
2022-05-27 14:35:25,886 P18844 INFO --- 50/50 batches finished ---
2022-05-27 14:35:25,936 P18844 INFO Train loss: 0.258146
2022-05-27 14:35:25,936 P18844 INFO ************ Epoch=4 end ************
2022-05-27 14:35:29,183 P18844 INFO [Metrics] AUC: 0.972859 - logloss: 0.188215
2022-05-27 14:35:29,184 P18844 INFO Save best model: monitor(max): 0.972859
2022-05-27 14:35:29,189 P18844 INFO --- 50/50 batches finished ---
2022-05-27 14:35:29,240 P18844 INFO Train loss: 0.247808
2022-05-27 14:35:29,240 P18844 INFO ************ Epoch=5 end ************
2022-05-27 14:35:32,574 P18844 INFO [Metrics] AUC: 0.974283 - logloss: 0.190274
2022-05-27 14:35:32,575 P18844 INFO Save best model: monitor(max): 0.974283
2022-05-27 14:35:32,581 P18844 INFO --- 50/50 batches finished ---
2022-05-27 14:35:32,640 P18844 INFO Train loss: 0.240821
2022-05-27 14:35:32,640 P18844 INFO ************ Epoch=6 end ************
2022-05-27 14:35:36,034 P18844 INFO [Metrics] AUC: 0.975577 - logloss: 0.179150
2022-05-27 14:35:36,035 P18844 INFO Save best model: monitor(max): 0.975577
2022-05-27 14:35:36,039 P18844 INFO --- 50/50 batches finished ---
2022-05-27 14:35:36,076 P18844 INFO Train loss: 0.236149
2022-05-27 14:35:36,076 P18844 INFO ************ Epoch=7 end ************
2022-05-27 14:35:39,469 P18844 INFO [Metrics] AUC: 0.975064 - logloss: 0.210407
2022-05-27 14:35:39,470 P18844 INFO Monitor(max) STOP: 0.975064 !
2022-05-27 14:35:39,470 P18844 INFO Reduce learning rate on plateau: 0.000100
2022-05-27 14:35:39,470 P18844 INFO --- 50/50 batches finished ---
2022-05-27 14:35:39,513 P18844 INFO Train loss: 0.230933
2022-05-27 14:35:39,513 P18844 INFO ************ Epoch=8 end ************
2022-05-27 14:35:42,838 P18844 INFO [Metrics] AUC: 0.982071 - logloss: 0.147709
2022-05-27 14:35:42,839 P18844 INFO Save best model: monitor(max): 0.982071
2022-05-27 14:35:42,843 P18844 INFO --- 50/50 batches finished ---
2022-05-27 14:35:42,893 P18844 INFO Train loss: 0.196917
2022-05-27 14:35:42,893 P18844 INFO ************ Epoch=9 end ************
2022-05-27 14:35:46,409 P18844 INFO [Metrics] AUC: 0.983588 - logloss: 0.141859
2022-05-27 14:35:46,409 P18844 INFO Save best model: monitor(max): 0.983588
2022-05-27 14:35:46,414 P18844 INFO --- 50/50 batches finished ---
2022-05-27 14:35:46,451 P18844 INFO Train loss: 0.161139
2022-05-27 14:35:46,451 P18844 INFO ************ Epoch=10 end ************
2022-05-27 14:35:50,082 P18844 INFO [Metrics] AUC: 0.984340 - logloss: 0.139732
2022-05-27 14:35:50,082 P18844 INFO Save best model: monitor(max): 0.984340
2022-05-27 14:35:50,088 P18844 INFO --- 50/50 batches finished ---
2022-05-27 14:35:50,130 P18844 INFO Train loss: 0.138565
2022-05-27 14:35:50,130 P18844 INFO ************ Epoch=11 end ************
2022-05-27 14:35:53,918 P18844 INFO [Metrics] AUC: 0.984674 - logloss: 0.139663
2022-05-27 14:35:53,919 P18844 INFO Save best model: monitor(max): 0.984674
2022-05-27 14:35:53,923 P18844 INFO --- 50/50 batches finished ---
2022-05-27 14:35:53,974 P18844 INFO Train loss: 0.122550
2022-05-27 14:35:53,974 P18844 INFO ************ Epoch=12 end ************
2022-05-27 14:35:57,709 P18844 INFO [Metrics] AUC: 0.984839 - logloss: 0.140914
2022-05-27 14:35:57,709 P18844 INFO Save best model: monitor(max): 0.984839
2022-05-27 14:35:57,717 P18844 INFO --- 50/50 batches finished ---
2022-05-27 14:35:57,757 P18844 INFO Train loss: 0.112116
2022-05-27 14:35:57,757 P18844 INFO ************ Epoch=13 end ************
2022-05-27 14:36:01,512 P18844 INFO [Metrics] AUC: 0.985190 - logloss: 0.139815
2022-05-27 14:36:01,513 P18844 INFO Save best model: monitor(max): 0.985190
2022-05-27 14:36:01,518 P18844 INFO --- 50/50 batches finished ---
2022-05-27 14:36:01,556 P18844 INFO Train loss: 0.103366
2022-05-27 14:36:01,556 P18844 INFO ************ Epoch=14 end ************
2022-05-27 14:36:03,847 P18844 INFO [Metrics] AUC: 0.984822 - logloss: 0.143240
2022-05-27 14:36:03,848 P18844 INFO Monitor(max) STOP: 0.984822 !
2022-05-27 14:36:03,848 P18844 INFO Reduce learning rate on plateau: 0.000010
2022-05-27 14:36:03,848 P18844 INFO --- 50/50 batches finished ---
2022-05-27 14:36:03,886 P18844 INFO Train loss: 0.097550
2022-05-27 14:36:03,886 P18844 INFO ************ Epoch=15 end ************
2022-05-27 14:36:05,562 P18844 INFO [Metrics] AUC: 0.984737 - logloss: 0.144866
2022-05-27 14:36:05,563 P18844 INFO Monitor(max) STOP: 0.984737 !
2022-05-27 14:36:05,563 P18844 INFO Reduce learning rate on plateau: 0.000001
2022-05-27 14:36:05,563 P18844 INFO Early stopping at epoch=16
2022-05-27 14:36:05,563 P18844 INFO --- 50/50 batches finished ---
2022-05-27 14:36:05,601 P18844 INFO Train loss: 0.088039
2022-05-27 14:36:05,601 P18844 INFO Training finished.
2022-05-27 14:36:05,601 P18844 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Frappe/DLRM_frappe_x1/frappe_x1_04e961e9/DLRM_frappe_x1_006_216831a3.model
2022-05-27 14:36:05,610 P18844 INFO ****** Validation evaluation ******
2022-05-27 14:36:05,922 P18844 INFO [Metrics] AUC: 0.985190 - logloss: 0.139815
2022-05-27 14:36:05,961 P18844 INFO ******** Test evaluation ********
2022-05-27 14:36:05,962 P18844 INFO Loading data...
2022-05-27 14:36:05,962 P18844 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-05-27 14:36:05,965 P18844 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-05-27 14:36:05,965 P18844 INFO Loading test data done.
2022-05-27 14:36:06,205 P18844 INFO [Metrics] AUC: 0.983995 - logloss: 0.144441

```
