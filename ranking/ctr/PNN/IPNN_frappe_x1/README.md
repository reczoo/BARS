## IPNN_frappe_x1

A hands-on guide to run the PNN model on the Frappe_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [PNN](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/PNN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [PNN_frappe_x1_tuner_config_06](./PNN_frappe_x1_tuner_config_06). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd IPNN_frappe_x1
    nohup python run_expid.py --config ./PNN_frappe_x1_tuner_config_06 --expid PNN_frappe_x1_020_f371a3ab --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.984122 | 0.154003  |


### Logs
```python
2020-12-24 22:26:40,598 P20311 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_7f91d67a",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "PNN",
    "model_id": "PNN_frappe_x1_020_f63bc10d",
    "model_root": "./Frappe/PNN_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "3",
    "pickle_feature_encoder": "True",
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
2020-12-24 22:26:40,599 P20311 INFO Set up feature encoder...
2020-12-24 22:26:40,599 P20311 INFO Load feature_encoder from pickle: ../data/Frappe/frappe_x1_7f91d67a/feature_encoder.pkl
2020-12-24 22:26:40,723 P20311 INFO Total number of parameters: 433491.
2020-12-24 22:26:40,724 P20311 INFO Loading data...
2020-12-24 22:26:40,726 P20311 INFO Loading data from h5: ../data/Frappe/frappe_x1_7f91d67a/train.h5
2020-12-24 22:26:40,738 P20311 INFO Loading data from h5: ../data/Frappe/frappe_x1_7f91d67a/valid.h5
2020-12-24 22:26:40,744 P20311 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2020-12-24 22:26:40,744 P20311 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2020-12-24 22:26:40,745 P20311 INFO Loading train data done.
2020-12-24 22:26:44,658 P20311 INFO Start training: 50 batches/epoch
2020-12-24 22:26:44,659 P20311 INFO ************ Epoch=1 start ************
2020-12-24 22:26:50,437 P20311 INFO [Metrics] AUC: 0.930509 - logloss: 0.320261
2020-12-24 22:26:50,438 P20311 INFO Save best model: monitor(max): 0.930509
2020-12-24 22:26:50,443 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:26:50,496 P20311 INFO Train loss: 0.554655
2020-12-24 22:26:50,496 P20311 INFO ************ Epoch=1 end ************
2020-12-24 22:26:56,379 P20311 INFO [Metrics] AUC: 0.934966 - logloss: 0.288004
2020-12-24 22:26:56,380 P20311 INFO Save best model: monitor(max): 0.934966
2020-12-24 22:26:56,384 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:26:56,441 P20311 INFO Train loss: 0.335423
2020-12-24 22:26:56,441 P20311 INFO ************ Epoch=2 end ************
2020-12-24 22:27:02,363 P20311 INFO [Metrics] AUC: 0.936059 - logloss: 0.285065
2020-12-24 22:27:02,363 P20311 INFO Save best model: monitor(max): 0.936059
2020-12-24 22:27:02,367 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:27:02,424 P20311 INFO Train loss: 0.313346
2020-12-24 22:27:02,424 P20311 INFO ************ Epoch=3 end ************
2020-12-24 22:27:08,506 P20311 INFO [Metrics] AUC: 0.938036 - logloss: 0.281758
2020-12-24 22:27:08,506 P20311 INFO Save best model: monitor(max): 0.938036
2020-12-24 22:27:08,510 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:27:08,576 P20311 INFO Train loss: 0.308690
2020-12-24 22:27:08,576 P20311 INFO ************ Epoch=4 end ************
2020-12-24 22:27:14,670 P20311 INFO [Metrics] AUC: 0.939720 - logloss: 0.279188
2020-12-24 22:27:14,671 P20311 INFO Save best model: monitor(max): 0.939720
2020-12-24 22:27:14,675 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:27:14,730 P20311 INFO Train loss: 0.305999
2020-12-24 22:27:14,730 P20311 INFO ************ Epoch=5 end ************
2020-12-24 22:27:20,782 P20311 INFO [Metrics] AUC: 0.948723 - logloss: 0.253204
2020-12-24 22:27:20,783 P20311 INFO Save best model: monitor(max): 0.948723
2020-12-24 22:27:20,787 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:27:20,861 P20311 INFO Train loss: 0.296110
2020-12-24 22:27:20,861 P20311 INFO ************ Epoch=6 end ************
2020-12-24 22:27:26,561 P20311 INFO [Metrics] AUC: 0.957608 - logloss: 0.235306
2020-12-24 22:27:26,561 P20311 INFO Save best model: monitor(max): 0.957608
2020-12-24 22:27:26,565 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:27:26,624 P20311 INFO Train loss: 0.277713
2020-12-24 22:27:26,624 P20311 INFO ************ Epoch=7 end ************
2020-12-24 22:27:32,204 P20311 INFO [Metrics] AUC: 0.962168 - logloss: 0.221989
2020-12-24 22:27:32,205 P20311 INFO Save best model: monitor(max): 0.962168
2020-12-24 22:27:32,209 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:27:32,264 P20311 INFO Train loss: 0.268853
2020-12-24 22:27:32,264 P20311 INFO ************ Epoch=8 end ************
2020-12-24 22:27:37,659 P20311 INFO [Metrics] AUC: 0.964589 - logloss: 0.213650
2020-12-24 22:27:37,660 P20311 INFO Save best model: monitor(max): 0.964589
2020-12-24 22:27:37,664 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:27:37,724 P20311 INFO Train loss: 0.260554
2020-12-24 22:27:37,724 P20311 INFO ************ Epoch=9 end ************
2020-12-24 22:27:43,271 P20311 INFO [Metrics] AUC: 0.967664 - logloss: 0.206696
2020-12-24 22:27:43,271 P20311 INFO Save best model: monitor(max): 0.967664
2020-12-24 22:27:43,275 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:27:43,344 P20311 INFO Train loss: 0.251519
2020-12-24 22:27:43,344 P20311 INFO ************ Epoch=10 end ************
2020-12-24 22:27:48,876 P20311 INFO [Metrics] AUC: 0.970402 - logloss: 0.192788
2020-12-24 22:27:48,877 P20311 INFO Save best model: monitor(max): 0.970402
2020-12-24 22:27:48,881 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:27:48,972 P20311 INFO Train loss: 0.245647
2020-12-24 22:27:48,972 P20311 INFO ************ Epoch=11 end ************
2020-12-24 22:27:54,554 P20311 INFO [Metrics] AUC: 0.971706 - logloss: 0.189026
2020-12-24 22:27:54,555 P20311 INFO Save best model: monitor(max): 0.971706
2020-12-24 22:27:54,559 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:27:54,616 P20311 INFO Train loss: 0.236814
2020-12-24 22:27:54,616 P20311 INFO ************ Epoch=12 end ************
2020-12-24 22:27:59,796 P20311 INFO [Metrics] AUC: 0.974215 - logloss: 0.182454
2020-12-24 22:27:59,796 P20311 INFO Save best model: monitor(max): 0.974215
2020-12-24 22:27:59,800 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:27:59,858 P20311 INFO Train loss: 0.232498
2020-12-24 22:27:59,858 P20311 INFO ************ Epoch=13 end ************
2020-12-24 22:28:04,342 P20311 INFO [Metrics] AUC: 0.974237 - logloss: 0.180592
2020-12-24 22:28:04,343 P20311 INFO Save best model: monitor(max): 0.974237
2020-12-24 22:28:04,347 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:28:04,403 P20311 INFO Train loss: 0.225898
2020-12-24 22:28:04,403 P20311 INFO ************ Epoch=14 end ************
2020-12-24 22:28:07,963 P20311 INFO [Metrics] AUC: 0.975586 - logloss: 0.175204
2020-12-24 22:28:07,964 P20311 INFO Save best model: monitor(max): 0.975586
2020-12-24 22:28:07,968 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:28:08,021 P20311 INFO Train loss: 0.223441
2020-12-24 22:28:08,021 P20311 INFO ************ Epoch=15 end ************
2020-12-24 22:28:12,241 P20311 INFO [Metrics] AUC: 0.976216 - logloss: 0.173061
2020-12-24 22:28:12,241 P20311 INFO Save best model: monitor(max): 0.976216
2020-12-24 22:28:12,245 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:28:12,304 P20311 INFO Train loss: 0.218607
2020-12-24 22:28:12,304 P20311 INFO ************ Epoch=16 end ************
2020-12-24 22:28:15,763 P20311 INFO [Metrics] AUC: 0.976967 - logloss: 0.171683
2020-12-24 22:28:15,764 P20311 INFO Save best model: monitor(max): 0.976967
2020-12-24 22:28:15,768 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:28:15,823 P20311 INFO Train loss: 0.215725
2020-12-24 22:28:15,823 P20311 INFO ************ Epoch=17 end ************
2020-12-24 22:28:19,992 P20311 INFO [Metrics] AUC: 0.977914 - logloss: 0.167124
2020-12-24 22:28:19,993 P20311 INFO Save best model: monitor(max): 0.977914
2020-12-24 22:28:19,999 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:28:20,054 P20311 INFO Train loss: 0.212653
2020-12-24 22:28:20,054 P20311 INFO ************ Epoch=18 end ************
2020-12-24 22:28:25,419 P20311 INFO [Metrics] AUC: 0.978144 - logloss: 0.165782
2020-12-24 22:28:25,420 P20311 INFO Save best model: monitor(max): 0.978144
2020-12-24 22:28:25,424 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:28:25,486 P20311 INFO Train loss: 0.206651
2020-12-24 22:28:25,486 P20311 INFO ************ Epoch=19 end ************
2020-12-24 22:28:30,942 P20311 INFO [Metrics] AUC: 0.978911 - logloss: 0.164182
2020-12-24 22:28:30,943 P20311 INFO Save best model: monitor(max): 0.978911
2020-12-24 22:28:30,947 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:28:31,002 P20311 INFO Train loss: 0.205947
2020-12-24 22:28:31,002 P20311 INFO ************ Epoch=20 end ************
2020-12-24 22:28:36,407 P20311 INFO [Metrics] AUC: 0.979516 - logloss: 0.168325
2020-12-24 22:28:36,408 P20311 INFO Save best model: monitor(max): 0.979516
2020-12-24 22:28:36,412 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:28:36,468 P20311 INFO Train loss: 0.201577
2020-12-24 22:28:36,468 P20311 INFO ************ Epoch=21 end ************
2020-12-24 22:28:42,019 P20311 INFO [Metrics] AUC: 0.979268 - logloss: 0.161075
2020-12-24 22:28:42,019 P20311 INFO Monitor(max) STOP: 0.979268 !
2020-12-24 22:28:42,020 P20311 INFO Reduce learning rate on plateau: 0.000100
2020-12-24 22:28:42,020 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:28:42,078 P20311 INFO Train loss: 0.198364
2020-12-24 22:28:42,078 P20311 INFO ************ Epoch=22 end ************
2020-12-24 22:28:47,471 P20311 INFO [Metrics] AUC: 0.981928 - logloss: 0.154229
2020-12-24 22:28:47,471 P20311 INFO Save best model: monitor(max): 0.981928
2020-12-24 22:28:47,475 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:28:47,551 P20311 INFO Train loss: 0.165261
2020-12-24 22:28:47,551 P20311 INFO ************ Epoch=23 end ************
2020-12-24 22:28:53,075 P20311 INFO [Metrics] AUC: 0.983195 - logloss: 0.151723
2020-12-24 22:28:53,076 P20311 INFO Save best model: monitor(max): 0.983195
2020-12-24 22:28:53,080 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:28:53,135 P20311 INFO Train loss: 0.138217
2020-12-24 22:28:53,135 P20311 INFO ************ Epoch=24 end ************
2020-12-24 22:28:58,643 P20311 INFO [Metrics] AUC: 0.983940 - logloss: 0.149211
2020-12-24 22:28:58,644 P20311 INFO Save best model: monitor(max): 0.983940
2020-12-24 22:28:58,650 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:28:58,730 P20311 INFO Train loss: 0.122296
2020-12-24 22:28:58,730 P20311 INFO ************ Epoch=25 end ************
2020-12-24 22:29:04,197 P20311 INFO [Metrics] AUC: 0.984187 - logloss: 0.151095
2020-12-24 22:29:04,198 P20311 INFO Save best model: monitor(max): 0.984187
2020-12-24 22:29:04,202 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:29:04,257 P20311 INFO Train loss: 0.111941
2020-12-24 22:29:04,258 P20311 INFO ************ Epoch=26 end ************
2020-12-24 22:29:09,864 P20311 INFO [Metrics] AUC: 0.984351 - logloss: 0.151782
2020-12-24 22:29:09,865 P20311 INFO Save best model: monitor(max): 0.984351
2020-12-24 22:29:09,869 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:29:09,926 P20311 INFO Train loss: 0.104149
2020-12-24 22:29:09,926 P20311 INFO ************ Epoch=27 end ************
2020-12-24 22:29:15,455 P20311 INFO [Metrics] AUC: 0.984579 - logloss: 0.151602
2020-12-24 22:29:15,456 P20311 INFO Save best model: monitor(max): 0.984579
2020-12-24 22:29:15,460 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:29:15,524 P20311 INFO Train loss: 0.098303
2020-12-24 22:29:15,524 P20311 INFO ************ Epoch=28 end ************
2020-12-24 22:29:21,040 P20311 INFO [Metrics] AUC: 0.984819 - logloss: 0.150826
2020-12-24 22:29:21,041 P20311 INFO Save best model: monitor(max): 0.984819
2020-12-24 22:29:21,045 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:29:21,100 P20311 INFO Train loss: 0.093131
2020-12-24 22:29:21,100 P20311 INFO ************ Epoch=29 end ************
2020-12-24 22:29:26,617 P20311 INFO [Metrics] AUC: 0.984874 - logloss: 0.150045
2020-12-24 22:29:26,617 P20311 INFO Save best model: monitor(max): 0.984874
2020-12-24 22:29:26,621 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:29:26,677 P20311 INFO Train loss: 0.089038
2020-12-24 22:29:26,677 P20311 INFO ************ Epoch=30 end ************
2020-12-24 22:29:32,229 P20311 INFO [Metrics] AUC: 0.984811 - logloss: 0.151432
2020-12-24 22:29:32,230 P20311 INFO Monitor(max) STOP: 0.984811 !
2020-12-24 22:29:32,230 P20311 INFO Reduce learning rate on plateau: 0.000010
2020-12-24 22:29:32,230 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:29:32,286 P20311 INFO Train loss: 0.085596
2020-12-24 22:29:32,286 P20311 INFO ************ Epoch=31 end ************
2020-12-24 22:29:37,891 P20311 INFO [Metrics] AUC: 0.984855 - logloss: 0.153012
2020-12-24 22:29:37,891 P20311 INFO Monitor(max) STOP: 0.984855 !
2020-12-24 22:29:37,891 P20311 INFO Reduce learning rate on plateau: 0.000001
2020-12-24 22:29:37,891 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:29:37,948 P20311 INFO Train loss: 0.079974
2020-12-24 22:29:37,949 P20311 INFO ************ Epoch=32 end ************
2020-12-24 22:29:43,154 P20311 INFO [Metrics] AUC: 0.984853 - logloss: 0.153211
2020-12-24 22:29:43,154 P20311 INFO Monitor(max) STOP: 0.984853 !
2020-12-24 22:29:43,154 P20311 INFO Reduce learning rate on plateau: 0.000001
2020-12-24 22:29:43,154 P20311 INFO Early stopping at epoch=33
2020-12-24 22:29:43,154 P20311 INFO --- 50/50 batches finished ---
2020-12-24 22:29:43,210 P20311 INFO Train loss: 0.078996
2020-12-24 22:29:43,210 P20311 INFO Training finished.
2020-12-24 22:29:43,210 P20311 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Frappe/PNN_frappe_x1/frappe_x1_7f91d67a/PNN_frappe_x1_020_f63bc10d_model.ckpt
2020-12-24 22:29:43,259 P20311 INFO ****** Train/validation evaluation ******
2020-12-24 22:29:43,745 P20311 INFO [Metrics] AUC: 0.984874 - logloss: 0.150045
2020-12-24 22:29:43,872 P20311 INFO ******** Test evaluation ********
2020-12-24 22:29:43,872 P20311 INFO Loading data...
2020-12-24 22:29:43,873 P20311 INFO Loading data from h5: ../data/Frappe/frappe_x1_7f91d67a/test.h5
2020-12-24 22:29:43,875 P20311 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2020-12-24 22:29:43,875 P20311 INFO Loading test data done.
2020-12-24 22:29:44,239 P20311 INFO [Metrics] AUC: 0.984122 - logloss: 0.154003

```
