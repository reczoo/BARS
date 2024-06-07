## AutoInt_movielenslatest_x1

A hands-on guide to run the AutoInt model on the Movielenslatest_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index

[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs) | [Revision History](#Revision-History)

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

Dataset ID: [Movielenslatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Movielenslatest#Movielenslatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.2](https://github.com/reczoo/FuxiCTR/tree/v1.2.2) for this experiment. See the model code: [AutoInt](https://github.com/reczoo/FuxiCTR/blob/v1.2.2/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.2.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.2.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [fuxictr_version.py](./fuxictr_version.py#L3) to add the FuxiCTR library to system path
   
   ```python
   sys.path.append('YOUR_PATH_TO_FuxiCTR/')
   ```

2. Create a data directory and put the downloaded csv files in `../data/Movielenslatest/Movielenslatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt_movielenslatest_x1_tuner_config_04](./AutoInt_movielenslatest_x1_tuner_config_04). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.
   
   ```bash
   cd AutoInt_movielenslatest_x1
   nohup python run_expid.py --config ./AutoInt_movielenslatest_x1_tuner_config_04 --expid AutoInt_movielenslatest_x1_004_4795ccb3 --gpu 0 > run.log &
   tail -f run.log
   ```

### Results

| AUC      | logloss  |
|:--------:|:--------:|
| 0.966292 | 0.222845 |

### Logs

```python
2022-06-26 22:29:00,256 P57186 INFO {
    "attention_dim": "128",
    "attention_layers": "2",
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "layer_norm": "True",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AutoInt",
    "model_id": "AutoInt_movielenslatest_x1_004_4795ccb3",
    "model_root": "./Movielens/AutoInt_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
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
    "test_data": "../data/Movielens/MovielensLatest_x1/test.csv",
    "train_data": "../data/Movielens/MovielensLatest_x1/train.csv",
    "use_hdf5": "True",
    "use_residual": "True",
    "use_scale": "True",
    "use_wide": "True",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-06-26 22:29:00,256 P57186 INFO Set up feature encoder...
2022-06-26 22:29:00,257 P57186 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-06-26 22:29:00,257 P57186 INFO Loading data...
2022-06-26 22:29:00,259 P57186 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-06-26 22:29:00,289 P57186 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-06-26 22:29:00,297 P57186 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-06-26 22:29:00,298 P57186 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-06-26 22:29:00,298 P57186 INFO Loading train data done.
2022-06-26 22:29:03,394 P57186 INFO Total number of parameters: 1047798.
2022-06-26 22:29:03,394 P57186 INFO Start training: 343 batches/epoch
2022-06-26 22:29:03,394 P57186 INFO ************ Epoch=1 start ************
2022-06-26 22:29:37,723 P57186 INFO [Metrics] AUC: 0.934201 - logloss: 0.289476
2022-06-26 22:29:37,724 P57186 INFO Save best model: monitor(max): 0.934201
2022-06-26 22:29:37,730 P57186 INFO --- 343/343 batches finished ---
2022-06-26 22:29:37,766 P57186 INFO Train loss: 0.385229
2022-06-26 22:29:37,766 P57186 INFO ************ Epoch=1 end ************
2022-06-26 22:30:12,277 P57186 INFO [Metrics] AUC: 0.942330 - logloss: 0.269865
2022-06-26 22:30:12,278 P57186 INFO Save best model: monitor(max): 0.942330
2022-06-26 22:30:12,287 P57186 INFO --- 343/343 batches finished ---
2022-06-26 22:30:12,326 P57186 INFO Train loss: 0.380693
2022-06-26 22:30:12,326 P57186 INFO ************ Epoch=2 end ************
2022-06-26 22:30:46,859 P57186 INFO [Metrics] AUC: 0.946554 - logloss: 0.259702
2022-06-26 22:30:46,860 P57186 INFO Save best model: monitor(max): 0.946554
2022-06-26 22:30:46,866 P57186 INFO --- 343/343 batches finished ---
2022-06-26 22:30:46,901 P57186 INFO Train loss: 0.391211
2022-06-26 22:30:46,901 P57186 INFO ************ Epoch=3 end ************
2022-06-26 22:31:21,088 P57186 INFO [Metrics] AUC: 0.947817 - logloss: 0.256410
2022-06-26 22:31:21,089 P57186 INFO Save best model: monitor(max): 0.947817
2022-06-26 22:31:21,097 P57186 INFO --- 343/343 batches finished ---
2022-06-26 22:31:21,162 P57186 INFO Train loss: 0.396904
2022-06-26 22:31:21,163 P57186 INFO ************ Epoch=4 end ************
2022-06-26 22:31:55,669 P57186 INFO [Metrics] AUC: 0.949018 - logloss: 0.252803
2022-06-26 22:31:55,670 P57186 INFO Save best model: monitor(max): 0.949018
2022-06-26 22:31:55,676 P57186 INFO --- 343/343 batches finished ---
2022-06-26 22:31:55,742 P57186 INFO Train loss: 0.398487
2022-06-26 22:31:55,742 P57186 INFO ************ Epoch=5 end ************
2022-06-26 22:32:29,650 P57186 INFO [Metrics] AUC: 0.950171 - logloss: 0.249822
2022-06-26 22:32:29,651 P57186 INFO Save best model: monitor(max): 0.950171
2022-06-26 22:32:29,658 P57186 INFO --- 343/343 batches finished ---
2022-06-26 22:32:29,690 P57186 INFO Train loss: 0.400738
2022-06-26 22:32:29,690 P57186 INFO ************ Epoch=6 end ************
2022-06-26 22:33:03,778 P57186 INFO [Metrics] AUC: 0.951501 - logloss: 0.246198
2022-06-26 22:33:03,779 P57186 INFO Save best model: monitor(max): 0.951501
2022-06-26 22:33:03,789 P57186 INFO --- 343/343 batches finished ---
2022-06-26 22:33:03,847 P57186 INFO Train loss: 0.399291
2022-06-26 22:33:03,847 P57186 INFO ************ Epoch=7 end ************
2022-06-26 22:33:38,172 P57186 INFO [Metrics] AUC: 0.952497 - logloss: 0.244017
2022-06-26 22:33:38,173 P57186 INFO Save best model: monitor(max): 0.952497
2022-06-26 22:33:38,182 P57186 INFO --- 343/343 batches finished ---
2022-06-26 22:33:38,221 P57186 INFO Train loss: 0.398734
2022-06-26 22:33:38,221 P57186 INFO ************ Epoch=8 end ************
2022-06-26 22:34:11,999 P57186 INFO [Metrics] AUC: 0.952723 - logloss: 0.243133
2022-06-26 22:34:12,000 P57186 INFO Save best model: monitor(max): 0.952723
2022-06-26 22:34:12,009 P57186 INFO --- 343/343 batches finished ---
2022-06-26 22:34:12,055 P57186 INFO Train loss: 0.398578
2022-06-26 22:34:12,055 P57186 INFO ************ Epoch=9 end ************
2022-06-26 22:34:46,187 P57186 INFO [Metrics] AUC: 0.953847 - logloss: 0.240639
2022-06-26 22:34:46,188 P57186 INFO Save best model: monitor(max): 0.953847
2022-06-26 22:34:46,194 P57186 INFO --- 343/343 batches finished ---
2022-06-26 22:34:46,223 P57186 INFO Train loss: 0.398169
2022-06-26 22:34:46,224 P57186 INFO ************ Epoch=10 end ************
2022-06-26 22:35:20,120 P57186 INFO [Metrics] AUC: 0.953674 - logloss: 0.240539
2022-06-26 22:35:20,121 P57186 INFO Monitor(max) STOP: 0.953674 !
2022-06-26 22:35:20,122 P57186 INFO Reduce learning rate on plateau: 0.000100
2022-06-26 22:35:20,122 P57186 INFO --- 343/343 batches finished ---
2022-06-26 22:35:20,171 P57186 INFO Train loss: 0.395715
2022-06-26 22:35:20,171 P57186 INFO ************ Epoch=11 end ************
2022-06-26 22:35:54,064 P57186 INFO [Metrics] AUC: 0.964528 - logloss: 0.215387
2022-06-26 22:35:54,065 P57186 INFO Save best model: monitor(max): 0.964528
2022-06-26 22:35:54,071 P57186 INFO --- 343/343 batches finished ---
2022-06-26 22:35:54,100 P57186 INFO Train loss: 0.290558
2022-06-26 22:35:54,100 P57186 INFO ************ Epoch=12 end ************
2022-06-26 22:36:28,642 P57186 INFO [Metrics] AUC: 0.966179 - logloss: 0.223125
2022-06-26 22:36:28,644 P57186 INFO Save best model: monitor(max): 0.966179
2022-06-26 22:36:28,655 P57186 INFO --- 343/343 batches finished ---
2022-06-26 22:36:28,699 P57186 INFO Train loss: 0.193461
2022-06-26 22:36:28,699 P57186 INFO ************ Epoch=13 end ************
2022-06-26 22:37:02,756 P57186 INFO [Metrics] AUC: 0.965642 - logloss: 0.244505
2022-06-26 22:37:02,757 P57186 INFO Monitor(max) STOP: 0.965642 !
2022-06-26 22:37:02,757 P57186 INFO Reduce learning rate on plateau: 0.000010
2022-06-26 22:37:02,757 P57186 INFO --- 343/343 batches finished ---
2022-06-26 22:37:02,797 P57186 INFO Train loss: 0.142674
2022-06-26 22:37:02,797 P57186 INFO ************ Epoch=14 end ************
2022-06-26 22:37:37,301 P57186 INFO [Metrics] AUC: 0.965771 - logloss: 0.252204
2022-06-26 22:37:37,302 P57186 INFO Monitor(max) STOP: 0.965771 !
2022-06-26 22:37:37,302 P57186 INFO Reduce learning rate on plateau: 0.000001
2022-06-26 22:37:37,302 P57186 INFO Early stopping at epoch=15
2022-06-26 22:37:37,302 P57186 INFO --- 343/343 batches finished ---
2022-06-26 22:37:37,341 P57186 INFO Train loss: 0.106375
2022-06-26 22:37:37,341 P57186 INFO Training finished.
2022-06-26 22:37:37,341 P57186 INFO Load best model: /home/Movielens/AutoInt_movielenslatest_x1/movielenslatest_x1_cd32d937/AutoInt_movielenslatest_x1_004_4795ccb3.model
2022-06-26 22:37:41,087 P57186 INFO ****** Validation evaluation ******
2022-06-26 22:37:43,461 P57186 INFO [Metrics] AUC: 0.966179 - logloss: 0.223125
2022-06-26 22:37:43,496 P57186 INFO ******** Test evaluation ********
2022-06-26 22:37:43,496 P57186 INFO Loading data...
2022-06-26 22:37:43,496 P57186 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-06-26 22:37:43,500 P57186 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-06-26 22:37:43,500 P57186 INFO Loading test data done.
2022-06-26 22:37:44,731 P57186 INFO [Metrics] AUC: 0.966292 - logloss: 0.222845
```

### Revision History

- [Version 1](https://github.com/openbenchmark/BARS/tree/88d3a0faa4565e975141ae89a52d35d3a8b56eda/ctr_prediction/benchmarks/AutoInt/AutoInt_movielenslatest_x1#autoint_movielenslatest_x1): deprecated due to bug fix [#30](https://github.com/reczoo/FuxiCTR/issues/30) of FuxiCTR.
