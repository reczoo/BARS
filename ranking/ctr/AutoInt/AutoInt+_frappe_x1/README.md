## AutoInt+_frappe_x1

A hands-on guide to run the AutoInt model on the Frappe_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index

[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs) |  [Revision History](#Revision-History)

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
  fuxictr: 1.2.2
  ```

### Dataset

Dataset ID: [Frappe_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Frappe#Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.2](https://github.com/reczoo/FuxiCTR/tree/v1.2.2) for this experiment. See the model code: [AutoInt](https://github.com/reczoo/FuxiCTR/blob/v1.2.2/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.2.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.2.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [fuxictr_version.py](./fuxictr_version.py#L3) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt+_frappe_x1_tuner_config_06](./AutoInt+_frappe_x1_tuner_config_06). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd AutoInt+_frappe_x1
   nohup python run_expid.py --config ./AutoInt+_frappe_x1_tuner_config_06 --expid AutoInt_frappe_x1_006_dba8e61c --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

| AUC      | logloss  |
|:--------:|:--------:|
| 0.984812 | 0.149007 |

### Logs

```python
2022-07-04 10:26:10,939 P5758 INFO {
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
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "layer_norm": "True",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AutoInt",
    "model_id": "AutoInt_frappe_x1_006_dba8e61c",
    "model_root": "./Frappe/AutoInt_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_heads": "4",
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
    "use_hdf5": "True",
    "use_residual": "True",
    "use_scale": "True",
    "use_wide": "False",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-07-04 10:26:10,940 P5758 INFO Set up feature encoder...
2022-07-04 10:26:10,940 P5758 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-07-04 10:26:10,940 P5758 INFO Loading data...
2022-07-04 10:26:10,942 P5758 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-07-04 10:26:10,953 P5758 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-07-04 10:26:10,957 P5758 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-07-04 10:26:10,958 P5758 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-07-04 10:26:10,958 P5758 INFO Loading train data done.
2022-07-04 10:26:15,052 P5758 INFO Total number of parameters: 622180.
2022-07-04 10:26:15,053 P5758 INFO Start training: 50 batches/epoch
2022-07-04 10:26:15,053 P5758 INFO ************ Epoch=1 start ************
2022-07-04 10:26:37,599 P5758 INFO [Metrics] AUC: 0.923242 - logloss: 0.667601
2022-07-04 10:26:37,599 P5758 INFO Save best model: monitor(max): 0.923242
2022-07-04 10:26:37,605 P5758 INFO --- 50/50 batches finished ---
2022-07-04 10:26:37,629 P5758 INFO Train loss: 0.372691
2022-07-04 10:26:37,629 P5758 INFO ************ Epoch=1 end ************
2022-07-04 10:27:00,197 P5758 INFO [Metrics] AUC: 0.967308 - logloss: 0.227230
2022-07-04 10:27:00,198 P5758 INFO Save best model: monitor(max): 0.967308
2022-07-04 10:27:00,205 P5758 INFO --- 50/50 batches finished ---
2022-07-04 10:27:00,234 P5758 INFO Train loss: 0.266678
2022-07-04 10:27:00,234 P5758 INFO ************ Epoch=2 end ************
2022-07-04 10:27:22,807 P5758 INFO [Metrics] AUC: 0.974064 - logloss: 0.198629
2022-07-04 10:27:22,807 P5758 INFO Save best model: monitor(max): 0.974064
2022-07-04 10:27:22,814 P5758 INFO --- 50/50 batches finished ---
2022-07-04 10:27:22,846 P5758 INFO Train loss: 0.238858
2022-07-04 10:27:22,846 P5758 INFO ************ Epoch=3 end ************
2022-07-04 10:27:45,209 P5758 INFO [Metrics] AUC: 0.975797 - logloss: 0.175871
2022-07-04 10:27:45,209 P5758 INFO Save best model: monitor(max): 0.975797
2022-07-04 10:27:45,216 P5758 INFO --- 50/50 batches finished ---
2022-07-04 10:27:45,247 P5758 INFO Train loss: 0.225399
2022-07-04 10:27:45,248 P5758 INFO ************ Epoch=4 end ************
2022-07-04 10:28:07,854 P5758 INFO [Metrics] AUC: 0.978483 - logloss: 0.168441
2022-07-04 10:28:07,854 P5758 INFO Save best model: monitor(max): 0.978483
2022-07-04 10:28:07,861 P5758 INFO --- 50/50 batches finished ---
2022-07-04 10:28:07,891 P5758 INFO Train loss: 0.214387
2022-07-04 10:28:07,892 P5758 INFO ************ Epoch=5 end ************
2022-07-04 10:28:29,976 P5758 INFO [Metrics] AUC: 0.978608 - logloss: 0.166450
2022-07-04 10:28:29,976 P5758 INFO Save best model: monitor(max): 0.978608
2022-07-04 10:28:29,983 P5758 INFO --- 50/50 batches finished ---
2022-07-04 10:28:30,014 P5758 INFO Train loss: 0.209860
2022-07-04 10:28:30,014 P5758 INFO ************ Epoch=6 end ************
2022-07-04 10:28:52,657 P5758 INFO [Metrics] AUC: 0.979044 - logloss: 0.164588
2022-07-04 10:28:52,657 P5758 INFO Save best model: monitor(max): 0.979044
2022-07-04 10:28:52,663 P5758 INFO --- 50/50 batches finished ---
2022-07-04 10:28:52,698 P5758 INFO Train loss: 0.207668
2022-07-04 10:28:52,698 P5758 INFO ************ Epoch=7 end ************
2022-07-04 10:29:15,173 P5758 INFO [Metrics] AUC: 0.980089 - logloss: 0.168088
2022-07-04 10:29:15,174 P5758 INFO Save best model: monitor(max): 0.980089
2022-07-04 10:29:15,180 P5758 INFO --- 50/50 batches finished ---
2022-07-04 10:29:15,214 P5758 INFO Train loss: 0.204472
2022-07-04 10:29:15,214 P5758 INFO ************ Epoch=8 end ************
2022-07-04 10:29:37,720 P5758 INFO [Metrics] AUC: 0.979934 - logloss: 0.164634
2022-07-04 10:29:37,720 P5758 INFO Monitor(max) STOP: 0.979934 !
2022-07-04 10:29:37,720 P5758 INFO Reduce learning rate on plateau: 0.000100
2022-07-04 10:29:37,720 P5758 INFO --- 50/50 batches finished ---
2022-07-04 10:29:37,754 P5758 INFO Train loss: 0.202347
2022-07-04 10:29:37,754 P5758 INFO ************ Epoch=9 end ************
2022-07-04 10:30:00,329 P5758 INFO [Metrics] AUC: 0.984263 - logloss: 0.139078
2022-07-04 10:30:00,329 P5758 INFO Save best model: monitor(max): 0.984263
2022-07-04 10:30:00,335 P5758 INFO --- 50/50 batches finished ---
2022-07-04 10:30:00,362 P5758 INFO Train loss: 0.168747
2022-07-04 10:30:00,362 P5758 INFO ************ Epoch=10 end ************
2022-07-04 10:30:22,885 P5758 INFO [Metrics] AUC: 0.985007 - logloss: 0.139548
2022-07-04 10:30:22,886 P5758 INFO Save best model: monitor(max): 0.985007
2022-07-04 10:30:22,892 P5758 INFO --- 50/50 batches finished ---
2022-07-04 10:30:22,931 P5758 INFO Train loss: 0.133375
2022-07-04 10:30:22,932 P5758 INFO ************ Epoch=11 end ************
2022-07-04 10:30:40,827 P5758 INFO [Metrics] AUC: 0.985372 - logloss: 0.141032
2022-07-04 10:30:40,827 P5758 INFO Save best model: monitor(max): 0.985372
2022-07-04 10:30:40,833 P5758 INFO --- 50/50 batches finished ---
2022-07-04 10:30:40,861 P5758 INFO Train loss: 0.112483
2022-07-04 10:30:40,861 P5758 INFO ************ Epoch=12 end ************
2022-07-04 10:30:58,604 P5758 INFO [Metrics] AUC: 0.985379 - logloss: 0.145408
2022-07-04 10:30:58,604 P5758 INFO Save best model: monitor(max): 0.985379
2022-07-04 10:30:58,610 P5758 INFO --- 50/50 batches finished ---
2022-07-04 10:30:58,639 P5758 INFO Train loss: 0.097755
2022-07-04 10:30:58,639 P5758 INFO ************ Epoch=13 end ************
2022-07-04 10:31:19,614 P5758 INFO [Metrics] AUC: 0.984978 - logloss: 0.151318
2022-07-04 10:31:19,615 P5758 INFO Monitor(max) STOP: 0.984978 !
2022-07-04 10:31:19,615 P5758 INFO Reduce learning rate on plateau: 0.000010
2022-07-04 10:31:19,615 P5758 INFO --- 50/50 batches finished ---
2022-07-04 10:31:19,653 P5758 INFO Train loss: 0.087173
2022-07-04 10:31:19,653 P5758 INFO ************ Epoch=14 end ************
2022-07-04 10:31:40,630 P5758 INFO [Metrics] AUC: 0.985158 - logloss: 0.151753
2022-07-04 10:31:40,630 P5758 INFO Monitor(max) STOP: 0.985158 !
2022-07-04 10:31:40,630 P5758 INFO Reduce learning rate on plateau: 0.000001
2022-07-04 10:31:40,630 P5758 INFO Early stopping at epoch=15
2022-07-04 10:31:40,630 P5758 INFO --- 50/50 batches finished ---
2022-07-04 10:31:40,669 P5758 INFO Train loss: 0.077432
2022-07-04 10:31:40,669 P5758 INFO Training finished.
2022-07-04 10:31:40,669 P5758 INFO Load best model: /home/benchmarks/Frappe/AutoInt_frappe_x1/frappe_x1_04e961e9/AutoInt_frappe_x1_006_dba8e61c.model
2022-07-04 10:31:44,902 P5758 INFO ****** Validation evaluation ******
2022-07-04 10:31:46,337 P5758 INFO [Metrics] AUC: 0.985379 - logloss: 0.145408
2022-07-04 10:31:46,370 P5758 INFO ******** Test evaluation ********
2022-07-04 10:31:46,371 P5758 INFO Loading data...
2022-07-04 10:31:46,371 P5758 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-07-04 10:31:46,374 P5758 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-07-04 10:31:46,374 P5758 INFO Loading test data done.
2022-07-04 10:31:47,184 P5758 INFO [Metrics] AUC: 0.984812 - logloss: 0.149007
```

### Revision History

- [Version 1](https://github.com/openbenchmark/BARS/tree/88d3a0faa4565e975141ae89a52d35d3a8b56eda/ctr_prediction/benchmarks/AutoInt/AutoInt_frappe_x1#autoint_frappe_x1): deprecated due to bug fix [#30](https://github.com/reczoo/FuxiCTR/issues/30) of FuxiCTR.
