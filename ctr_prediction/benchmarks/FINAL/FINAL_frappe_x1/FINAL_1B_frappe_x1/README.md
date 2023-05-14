## FINAL_1B_frappe_x1

A hands-on guide to run the FINAL model on the Frappe_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)


| [Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs) |
|:-----------------------------:|:-----------:|:--------:|:--------:|-------|
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
  fuxictr: 2.0.2
  ```

### Dataset
Please refer to the BARS dataset [Frappe_x1](https://github.com/openbenchmark/BARS/blob/main/datasets/Frappe#Frappe_x1) to get data ready.

### Code

We use the [FINAL](https://github.com/xue-pai/FuxiCTR/blob/v2.0.2/model_zoo/FINAL) model code from [FuxiCTR-v2.0.2](https://github.com/xue-pai/FuxiCTR/tree/v2.0.2) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v2.0.2.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.2
    ```

2. Create a data directory and put the downloaded data files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FINAL_1B_frappe_x1_tuner_config_01](./FINAL_1B_frappe_x1_tuner_config_01). Make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/FINAL
    nohup python run_expid.py --config XXX/benchmarks/FINAL/FINAL_1B_frappe_x1_tuner_config_01 --expid FINAL_frappe_x1_006_75be0578 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.985223 | 0.136554  |


### Logs
```python
2023-01-06 22:45:53,562 P38786 INFO Params: {
    "batch_size": "4096",
    "block1_dropout": "0.2",
    "block1_hidden_activations": "ReLU",
    "block1_hidden_units": "[400, 400]",
    "block2_dropout": "0",
    "block2_hidden_activations": "None",
    "block2_hidden_units": "[64, 64, 64]",
    "block_type": "1B",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_47e6e0df",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "10",
    "embedding_regularizer": "0.1",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "feature_specs": "None",
    "gpu": "1",
    "group_id": "None",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FINAL",
    "model_id": "FINAL_frappe_x1_006_75be0578",
    "model_root": "./checkpoints/FINAL_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "norm_type": "BN",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Frappe/Frappe_x1/test.csv",
    "train_data": "../data/Frappe/Frappe_x1/train.csv",
    "use_field_gate": "False",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1"
}
2023-01-06 22:45:53,563 P38786 INFO Load feature_map from json: ../data/Frappe/frappe_x1_47e6e0df/feature_map.json
2023-01-06 22:45:53,563 P38786 INFO Set column index...
2023-01-06 22:45:53,564 P38786 INFO Feature specs: {
    "city": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 234, 'vocab_size': 235}",
    "cost": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 3, 'vocab_size': 4}",
    "country": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 81, 'vocab_size': 82}",
    "daytime": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 8, 'vocab_size': 9}",
    "homework": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 4, 'vocab_size': 5}",
    "isweekend": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 3, 'vocab_size': 4}",
    "item": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 4083, 'vocab_size': 4084}",
    "user": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 955, 'vocab_size': 956}",
    "weather": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10, 'vocab_size': 11}",
    "weekday": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 8, 'vocab_size': 9}"
}
2023-01-06 22:45:59,746 P38786 INFO Total number of parameters: 256791.
2023-01-06 22:45:59,746 P38786 INFO Loading data...
2023-01-06 22:45:59,746 P38786 INFO Loading data from h5: ../data/Frappe/frappe_x1_47e6e0df/train.h5
2023-01-06 22:45:59,793 P38786 INFO Train samples: total/202027, blocks/1
2023-01-06 22:45:59,793 P38786 INFO Loading data from h5: ../data/Frappe/frappe_x1_47e6e0df/valid.h5
2023-01-06 22:45:59,806 P38786 INFO Validation samples: total/57722, blocks/1
2023-01-06 22:45:59,806 P38786 INFO Loading train and validation data done.
2023-01-06 22:45:59,806 P38786 INFO Start training: 50 batches/epoch
2023-01-06 22:45:59,806 P38786 INFO ************ Epoch=1 start ************
2023-01-06 22:46:07,307 P38786 INFO [Metrics] AUC: 0.930944
2023-01-06 22:46:07,308 P38786 INFO Save best model: monitor(max): 0.930944
2023-01-06 22:46:07,313 P38786 INFO --- 50/50 batches finished ---
2023-01-06 22:46:07,400 P38786 INFO Train loss @epoch 1: 0.382210
2023-01-06 22:46:07,400 P38786 INFO ************ Epoch=1 end ************
2023-01-06 22:46:14,428 P38786 INFO [Metrics] AUC: 0.954366
2023-01-06 22:46:14,428 P38786 INFO Save best model: monitor(max): 0.954366
2023-01-06 22:46:14,432 P38786 INFO --- 50/50 batches finished ---
2023-01-06 22:46:14,491 P38786 INFO Train loss @epoch 2: 0.286990
2023-01-06 22:46:14,491 P38786 INFO ************ Epoch=2 end ************
2023-01-06 22:46:21,731 P38786 INFO [Metrics] AUC: 0.963182
2023-01-06 22:46:21,731 P38786 INFO Save best model: monitor(max): 0.963182
2023-01-06 22:46:21,735 P38786 INFO --- 50/50 batches finished ---
2023-01-06 22:46:21,793 P38786 INFO Train loss @epoch 3: 0.253435
2023-01-06 22:46:21,793 P38786 INFO ************ Epoch=3 end ************
2023-01-06 22:46:29,580 P38786 INFO [Metrics] AUC: 0.971516
2023-01-06 22:46:29,580 P38786 INFO Save best model: monitor(max): 0.971516
2023-01-06 22:46:29,584 P38786 INFO --- 50/50 batches finished ---
2023-01-06 22:46:29,660 P38786 INFO Train loss @epoch 4: 0.236046
2023-01-06 22:46:29,660 P38786 INFO ************ Epoch=4 end ************
2023-01-06 22:46:37,075 P38786 INFO [Metrics] AUC: 0.974068
2023-01-06 22:46:37,076 P38786 INFO Save best model: monitor(max): 0.974068
2023-01-06 22:46:37,081 P38786 INFO --- 50/50 batches finished ---
2023-01-06 22:46:37,143 P38786 INFO Train loss @epoch 5: 0.224477
2023-01-06 22:46:37,143 P38786 INFO ************ Epoch=5 end ************
2023-01-06 22:46:44,256 P38786 INFO [Metrics] AUC: 0.974199
2023-01-06 22:46:44,257 P38786 INFO Save best model: monitor(max): 0.974199
2023-01-06 22:46:44,262 P38786 INFO --- 50/50 batches finished ---
2023-01-06 22:46:44,332 P38786 INFO Train loss @epoch 6: 0.217776
2023-01-06 22:46:44,332 P38786 INFO ************ Epoch=6 end ************
2023-01-06 22:46:51,667 P38786 INFO [Metrics] AUC: 0.975267
2023-01-06 22:46:51,667 P38786 INFO Save best model: monitor(max): 0.975267
2023-01-06 22:46:51,673 P38786 INFO --- 50/50 batches finished ---
2023-01-06 22:46:51,762 P38786 INFO Train loss @epoch 7: 0.212764
2023-01-06 22:46:51,762 P38786 INFO ************ Epoch=7 end ************
2023-01-06 22:46:59,477 P38786 INFO [Metrics] AUC: 0.977986
2023-01-06 22:46:59,478 P38786 INFO Save best model: monitor(max): 0.977986
2023-01-06 22:46:59,483 P38786 INFO --- 50/50 batches finished ---
2023-01-06 22:46:59,565 P38786 INFO Train loss @epoch 8: 0.211160
2023-01-06 22:46:59,566 P38786 INFO ************ Epoch=8 end ************
2023-01-06 22:47:06,587 P38786 INFO [Metrics] AUC: 0.975616
2023-01-06 22:47:06,587 P38786 INFO Monitor(max) STOP: 0.975616 !
2023-01-06 22:47:06,587 P38786 INFO Reduce learning rate on plateau: 0.000100
2023-01-06 22:47:06,588 P38786 INFO --- 50/50 batches finished ---
2023-01-06 22:47:06,658 P38786 INFO Train loss @epoch 9: 0.207098
2023-01-06 22:47:06,658 P38786 INFO ************ Epoch=9 end ************
2023-01-06 22:47:13,859 P38786 INFO [Metrics] AUC: 0.983682
2023-01-06 22:47:13,859 P38786 INFO Save best model: monitor(max): 0.983682
2023-01-06 22:47:13,865 P38786 INFO --- 50/50 batches finished ---
2023-01-06 22:47:13,937 P38786 INFO Train loss @epoch 10: 0.172933
2023-01-06 22:47:13,937 P38786 INFO ************ Epoch=10 end ************
2023-01-06 22:47:21,577 P38786 INFO [Metrics] AUC: 0.985181
2023-01-06 22:47:21,578 P38786 INFO Save best model: monitor(max): 0.985181
2023-01-06 22:47:21,582 P38786 INFO --- 50/50 batches finished ---
2023-01-06 22:47:21,638 P38786 INFO Train loss @epoch 11: 0.138124
2023-01-06 22:47:21,638 P38786 INFO ************ Epoch=11 end ************
2023-01-06 22:47:28,951 P38786 INFO [Metrics] AUC: 0.985528
2023-01-06 22:47:28,951 P38786 INFO Save best model: monitor(max): 0.985528
2023-01-06 22:47:28,957 P38786 INFO --- 50/50 batches finished ---
2023-01-06 22:47:29,032 P38786 INFO Train loss @epoch 12: 0.117746
2023-01-06 22:47:29,032 P38786 INFO ************ Epoch=12 end ************
2023-01-06 22:47:36,992 P38786 INFO [Metrics] AUC: 0.985643
2023-01-06 22:47:36,992 P38786 INFO Save best model: monitor(max): 0.985643
2023-01-06 22:47:36,998 P38786 INFO --- 50/50 batches finished ---
2023-01-06 22:47:37,073 P38786 INFO Train loss @epoch 13: 0.103718
2023-01-06 22:47:37,074 P38786 INFO ************ Epoch=13 end ************
2023-01-06 22:47:44,232 P38786 INFO [Metrics] AUC: 0.985489
2023-01-06 22:47:44,233 P38786 INFO Monitor(max) STOP: 0.985489 !
2023-01-06 22:47:44,233 P38786 INFO Reduce learning rate on plateau: 0.000010
2023-01-06 22:47:44,233 P38786 INFO --- 50/50 batches finished ---
2023-01-06 22:47:44,300 P38786 INFO Train loss @epoch 14: 0.093447
2023-01-06 22:47:44,301 P38786 INFO ************ Epoch=14 end ************
2023-01-06 22:47:51,151 P38786 INFO [Metrics] AUC: 0.985558
2023-01-06 22:47:51,152 P38786 INFO Monitor(max) STOP: 0.985558 !
2023-01-06 22:47:51,152 P38786 INFO Reduce learning rate on plateau: 0.000001
2023-01-06 22:47:51,152 P38786 INFO ********* Epoch==15 early stop *********
2023-01-06 22:47:51,152 P38786 INFO --- 50/50 batches finished ---
2023-01-06 22:47:51,233 P38786 INFO Train loss @epoch 15: 0.083707
2023-01-06 22:47:51,233 P38786 INFO Training finished.
2023-01-06 22:47:51,233 P38786 INFO Load best model: /home/FuxiCTR/benchmark/checkpoints/FINAL_frappe_x1/frappe_x1_47e6e0df/FINAL_frappe_x1_006_75be0578.model
2023-01-06 22:47:51,258 P38786 INFO ****** Validation evaluation ******
2023-01-06 22:47:51,981 P38786 INFO [Metrics] AUC: 0.985643 - logloss: 0.133350
2023-01-06 22:47:52,071 P38786 INFO ******** Test evaluation ********
2023-01-06 22:47:52,071 P38786 INFO Loading data...
2023-01-06 22:47:52,071 P38786 INFO Loading data from h5: ../data/Frappe/frappe_x1_47e6e0df/test.h5
2023-01-06 22:47:52,081 P38786 INFO Test samples: total/28860, blocks/1
2023-01-06 22:47:52,081 P38786 INFO Loading test data done.
2023-01-06 22:47:52,514 P38786 INFO [Metrics] AUC: 0.985223 - logloss: 0.136554

```
