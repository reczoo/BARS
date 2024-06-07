## FinalNet_criteo_x4_001

A hands-on guide to run the FinalNet model on the Criteo_x4 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)


| [Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs) |
|:-----------------------------:|:-----------:|:--------:|:--------:|-------|
### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) Gold 6278C CPU @ 2.60GHz
  GPU: Tesla V100 32G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 10.2
  python: 3.7.10
  pytorch: 1.10.2+cu102
  pandas: 1.1.5
  numpy: 1.19.5
  scipy: 1.5.2
  sklearn: 0.22.1
  pyyaml: 6.0.1
  h5py: 2.8.0
  tqdm: 4.64.0
  keras_preprocessing: 1.1.2
  fuxictr: 2.2.0
  ```

### Dataset
Please refer to [Criteo_x4](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x4) to get the dataset details.

### Code

We use the [FinalNet](https://github.com/reczoo/FuxiCTR/tree/v2.2.0/model_zoo/FinalNet) model code from [FuxiCTR-v2.2.0](https://github.com/reczoo/FuxiCTR/tree/v2.2.0) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.2.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.2.0.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.2.0
    ```

2. Create a data directory and put the downloaded data files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FinalNet_criteo_x4_tuner_config_05](./FinalNet_criteo_x4_tuner_config_05). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/FinalNet
    nohup python run_expid.py --config YOUR_PATH/FinalNet/FinalNet_criteo_x4_tuner_config_05 --expid FinalNet_criteo_x4_001_041_449ccb21 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.814966 | 0.437116  |


### Logs
```python
2024-02-21 01:09:14,380 P2661590 INFO Params: {
    "batch_norm": "True",
    "batch_size": "8192",
    "block1_dropout": "0.4",
    "block1_hidden_activations": "ReLU",
    "block1_hidden_units": "[1000, 1000, 1000]",
    "block2_dropout": "0.4",
    "block2_hidden_activations": "ReLU",
    "block2_hidden_units": "[512]",
    "block_type": "2B",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_001_a5e05ce7",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "16",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'fill_na': 0, 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'preprocess': 'convert_to_bucket', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'fill_na': '', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "feature_specs": "None",
    "gpu": "0",
    "group_id": "None",
    "label_col": "{'dtype': 'float', 'name': 'Label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "FinalNet",
    "model_id": "FinalNet_criteo_x4_001_041_449ccb21",
    "model_root": "./checkpoints/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "ordered_features": "None",
    "pickle_feature_encoder": "True",
    "residual_type": "concat",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x4/test.csv",
    "train_data": "../data/Criteo/Criteo_x4/train.csv",
    "use_feature_gating": "True",
    "valid_data": "../data/Criteo/Criteo_x4/valid.csv",
    "verbose": "1"
}
2024-02-21 01:09:14,381 P2661590 INFO Set up feature processor...
2024-02-21 01:09:14,382 P2661590 WARNING Skip rebuilding ../data/Criteo/criteo_x4_001_a5e05ce7/feature_map.json. Please delete it manually if rebuilding is required.
2024-02-21 01:09:14,382 P2661590 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_a5e05ce7/feature_map.json
2024-02-21 01:09:14,382 P2661590 INFO Set column index...
2024-02-21 01:09:14,383 P2661590 INFO Feature specs: {
    "C1": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 1446}",
    "C10": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 39530}",
    "C11": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 5131}",
    "C12": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 156656}",
    "C13": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 3176}",
    "C14": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 28}",
    "C15": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 11043}",
    "C16": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 148913}",
    "C17": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 12}",
    "C18": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 4560}",
    "C19": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 2003}",
    "C2": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 554}",
    "C20": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 5}",
    "C21": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 154564}",
    "C22": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 18}",
    "C23": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 17}",
    "C24": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 53031}",
    "C25": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 82}",
    "C26": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 40955}",
    "C3": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 157339}",
    "C4": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 117822}",
    "C5": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 306}",
    "C6": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 18}",
    "C7": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 11882}",
    "C8": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 630}",
    "C9": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 5}",
    "I1": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 44}",
    "I10": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 6}",
    "I11": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 27}",
    "I12": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 37}",
    "I13": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 72}",
    "I2": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 99}",
    "I3": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 122}",
    "I4": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 41}",
    "I5": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 220}",
    "I6": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 112}",
    "I7": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 80}",
    "I8": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 69}",
    "I9": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'vocab_size': 92}"
}
2024-02-21 01:09:18,212 P2661590 INFO Total number of parameters: 18153050.
2024-02-21 01:09:18,212 P2661590 INFO Loading datasets...
2024-02-21 01:09:54,793 P2661590 INFO Train samples: total/36672493, blocks/1
2024-02-21 01:09:59,444 P2661590 INFO Validation samples: total/4584062, blocks/1
2024-02-21 01:09:59,444 P2661590 INFO Loading train and validation data done.
2024-02-21 01:09:59,444 P2661590 INFO Start training: 4477 batches/epoch
2024-02-21 01:09:59,444 P2661590 INFO ************ Epoch=1 start ************
2024-02-21 01:14:51,100 P2661590 INFO Train loss: 0.460479
2024-02-21 01:14:51,100 P2661590 INFO Evaluation @epoch 1 - batch 4477: 
2024-02-21 01:15:05,073 P2661590 INFO [Metrics] AUC: 0.805437
2024-02-21 01:15:05,074 P2661590 INFO Save best model: monitor(max)=0.805437
2024-02-21 01:15:05,251 P2661590 INFO ************ Epoch=1 end ************
2024-02-21 01:19:58,419 P2661590 INFO Train loss: 0.454513
2024-02-21 01:19:58,419 P2661590 INFO Evaluation @epoch 2 - batch 4477: 
2024-02-21 01:20:12,691 P2661590 INFO [Metrics] AUC: 0.807590
2024-02-21 01:20:12,692 P2661590 INFO Save best model: monitor(max)=0.807590
2024-02-21 01:20:12,879 P2661590 INFO ************ Epoch=2 end ************
2024-02-21 01:25:03,483 P2661590 INFO Train loss: 0.453117
2024-02-21 01:25:03,484 P2661590 INFO Evaluation @epoch 3 - batch 4477: 
2024-02-21 01:25:17,838 P2661590 INFO [Metrics] AUC: 0.808788
2024-02-21 01:25:17,840 P2661590 INFO Save best model: monitor(max)=0.808788
2024-02-21 01:25:18,032 P2661590 INFO ************ Epoch=3 end ************
2024-02-21 01:30:09,428 P2661590 INFO Train loss: 0.452215
2024-02-21 01:30:09,429 P2661590 INFO Evaluation @epoch 4 - batch 4477: 
2024-02-21 01:30:23,782 P2661590 INFO [Metrics] AUC: 0.809404
2024-02-21 01:30:23,783 P2661590 INFO Save best model: monitor(max)=0.809404
2024-02-21 01:30:23,982 P2661590 INFO ************ Epoch=4 end ************
2024-02-21 01:35:14,773 P2661590 INFO Train loss: 0.451528
2024-02-21 01:35:14,773 P2661590 INFO Evaluation @epoch 5 - batch 4477: 
2024-02-21 01:35:29,205 P2661590 INFO [Metrics] AUC: 0.809996
2024-02-21 01:35:29,207 P2661590 INFO Save best model: monitor(max)=0.809996
2024-02-21 01:35:29,392 P2661590 INFO ************ Epoch=5 end ************
2024-02-21 01:40:23,698 P2661590 INFO Train loss: 0.450978
2024-02-21 01:40:23,699 P2661590 INFO Evaluation @epoch 6 - batch 4477: 
2024-02-21 01:40:37,685 P2661590 INFO [Metrics] AUC: 0.810492
2024-02-21 01:40:37,690 P2661590 INFO Save best model: monitor(max)=0.810492
2024-02-21 01:40:37,886 P2661590 INFO ************ Epoch=6 end ************
2024-02-21 01:45:31,541 P2661590 INFO Train loss: 0.450588
2024-02-21 01:45:31,542 P2661590 INFO Evaluation @epoch 7 - batch 4477: 
2024-02-21 01:45:45,835 P2661590 INFO [Metrics] AUC: 0.810681
2024-02-21 01:45:45,836 P2661590 INFO Save best model: monitor(max)=0.810681
2024-02-21 01:45:46,015 P2661590 INFO ************ Epoch=7 end ************
2024-02-21 01:50:42,165 P2661590 INFO Train loss: 0.450212
2024-02-21 01:50:42,165 P2661590 INFO Evaluation @epoch 8 - batch 4477: 
2024-02-21 01:50:56,202 P2661590 INFO [Metrics] AUC: 0.811053
2024-02-21 01:50:56,206 P2661590 INFO Save best model: monitor(max)=0.811053
2024-02-21 01:50:56,395 P2661590 INFO ************ Epoch=8 end ************
2024-02-21 01:55:51,990 P2661590 INFO Train loss: 0.449913
2024-02-21 01:55:51,990 P2661590 INFO Evaluation @epoch 9 - batch 4477: 
2024-02-21 01:56:05,814 P2661590 INFO [Metrics] AUC: 0.811255
2024-02-21 01:56:05,815 P2661590 INFO Save best model: monitor(max)=0.811255
2024-02-21 01:56:06,019 P2661590 INFO ************ Epoch=9 end ************
2024-02-21 02:01:03,880 P2661590 INFO Train loss: 0.449685
2024-02-21 02:01:03,881 P2661590 INFO Evaluation @epoch 10 - batch 4477: 
2024-02-21 02:01:17,759 P2661590 INFO [Metrics] AUC: 0.811483
2024-02-21 02:01:17,761 P2661590 INFO Save best model: monitor(max)=0.811483
2024-02-21 02:01:17,949 P2661590 INFO ************ Epoch=10 end ************
2024-02-21 02:06:12,912 P2661590 INFO Train loss: 0.449436
2024-02-21 02:06:12,913 P2661590 INFO Evaluation @epoch 11 - batch 4477: 
2024-02-21 02:06:26,670 P2661590 INFO [Metrics] AUC: 0.811428
2024-02-21 02:06:26,671 P2661590 INFO Monitor(max)=0.811428 STOP!
2024-02-21 02:06:26,671 P2661590 INFO Reduce learning rate on plateau: 0.000100
2024-02-21 02:06:26,744 P2661590 INFO ************ Epoch=11 end ************
2024-02-21 02:11:23,197 P2661590 INFO Train loss: 0.439517
2024-02-21 02:11:23,197 P2661590 INFO Evaluation @epoch 12 - batch 4477: 
2024-02-21 02:11:36,759 P2661590 INFO [Metrics] AUC: 0.814074
2024-02-21 02:11:36,761 P2661590 INFO Save best model: monitor(max)=0.814074
2024-02-21 02:11:36,947 P2661590 INFO ************ Epoch=12 end ************
2024-02-21 02:16:33,347 P2661590 INFO Train loss: 0.435572
2024-02-21 02:16:33,347 P2661590 INFO Evaluation @epoch 13 - batch 4477: 
2024-02-21 02:16:47,262 P2661590 INFO [Metrics] AUC: 0.814462
2024-02-21 02:16:47,263 P2661590 INFO Save best model: monitor(max)=0.814462
2024-02-21 02:16:47,448 P2661590 INFO ************ Epoch=13 end ************
2024-02-21 02:21:42,706 P2661590 INFO Train loss: 0.433620
2024-02-21 02:21:42,707 P2661590 INFO Evaluation @epoch 14 - batch 4477: 
2024-02-21 02:21:56,657 P2661590 INFO [Metrics] AUC: 0.814439
2024-02-21 02:21:56,662 P2661590 INFO Monitor(max)=0.814439 STOP!
2024-02-21 02:21:56,662 P2661590 INFO Reduce learning rate on plateau: 0.000010
2024-02-21 02:21:56,737 P2661590 INFO ************ Epoch=14 end ************
2024-02-21 02:26:51,667 P2661590 INFO Train loss: 0.430005
2024-02-21 02:26:51,668 P2661590 INFO Evaluation @epoch 15 - batch 4477: 
2024-02-21 02:27:05,795 P2661590 INFO [Metrics] AUC: 0.814147
2024-02-21 02:27:05,796 P2661590 INFO Monitor(max)=0.814147 STOP!
2024-02-21 02:27:05,796 P2661590 INFO Reduce learning rate on plateau: 0.000001
2024-02-21 02:27:05,796 P2661590 INFO ********* Epoch==15 early stop *********
2024-02-21 02:27:05,869 P2661590 INFO Training finished.
2024-02-21 02:27:05,869 P2661590 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/criteo_x4_001_a5e05ce7/FinalNet_criteo_x4_001_041_449ccb21.model
2024-02-21 02:27:05,947 P2661590 INFO ****** Validation evaluation ******
2024-02-21 02:27:21,204 P2661590 INFO [Metrics] AUC: 0.814462 - logloss: 0.437531
2024-02-21 02:27:21,319 P2661590 INFO ******** Test evaluation ********
2024-02-21 02:27:21,320 P2661590 INFO Loading datasets...
2024-02-21 02:27:25,802 P2661590 INFO Test samples: total/4584062, blocks/1
2024-02-21 02:27:25,802 P2661590 INFO Loading test data done.
2024-02-21 02:27:41,261 P2661590 INFO [Metrics] AUC: 0.814966 - logloss: 0.437116

```
