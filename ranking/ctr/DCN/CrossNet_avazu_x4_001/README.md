## CrossNet_avazu_x4_001

A hands-on guide to run the DCN model on the Avazu_x4_001 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

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
  python: 3.6.4
  pytorch: 1.0.0
  pandas: 0.22.0
  numpy: 1.19.2
  scipy: 1.5.4
  sklearn: 0.22.1
  pyyaml: 5.4.1
  h5py: 2.8.0
  tqdm: 4.60.0
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [Avazu_x4](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x4). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [DCN](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DCN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CrossNet_avazu_x4_tuner_config_11](./CrossNet_avazu_x4_tuner_config_11). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CrossNet_avazu_x4_001
    nohup python run_expid.py --config ./CrossNet_avazu_x4_tuner_config_11 --expid DCN_avazu_x4_030_85717c33 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.377924 | 0.783962  |


### Logs
```python
2020-06-15 17:12:38,901 P5968 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "crossing_layers": "8",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DCN",
    "model_id": "DCN_avazu_x4_3bbbc4c9_030_6039e913",
    "model_root": "./Avazu/DCN_avazu/min2/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/avazu_x4_3bbbc4c9/test.h5",
    "train_data": "../data/Avazu/avazu_x4_3bbbc4c9/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/avazu_x4_3bbbc4c9/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-15 17:12:38,901 P5968 INFO Set up feature encoder...
2020-06-15 17:12:38,902 P5968 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-06-15 17:12:38,902 P5968 INFO Loading data...
2020-06-15 17:12:38,904 P5968 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-06-15 17:12:41,917 P5968 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-06-15 17:12:43,585 P5968 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-15 17:12:43,776 P5968 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-15 17:12:43,776 P5968 INFO Loading train data done.
2020-06-15 17:12:49,471 P5968 INFO Start training: 3235 batches/epoch
2020-06-15 17:12:49,471 P5968 INFO ************ Epoch=1 start ************
2020-06-15 17:17:28,916 P5968 INFO [Metrics] logloss: 0.385846 - AUC: 0.769730
2020-06-15 17:17:28,917 P5968 INFO Save best model: monitor(max): 0.383884
2020-06-15 17:17:29,717 P5968 INFO --- 3235/3235 batches finished ---
2020-06-15 17:17:29,781 P5968 INFO Train loss: 0.402222
2020-06-15 17:17:29,781 P5968 INFO ************ Epoch=1 end ************
2020-06-15 17:22:08,788 P5968 INFO [Metrics] logloss: 0.381376 - AUC: 0.777351
2020-06-15 17:22:08,791 P5968 INFO Save best model: monitor(max): 0.395975
2020-06-15 17:22:09,166 P5968 INFO --- 3235/3235 batches finished ---
2020-06-15 17:22:09,231 P5968 INFO Train loss: 0.394735
2020-06-15 17:22:09,231 P5968 INFO ************ Epoch=2 end ************
2020-06-15 17:26:45,171 P5968 INFO [Metrics] logloss: 0.380399 - AUC: 0.779506
2020-06-15 17:26:45,175 P5968 INFO Save best model: monitor(max): 0.399107
2020-06-15 17:26:45,575 P5968 INFO --- 3235/3235 batches finished ---
2020-06-15 17:26:45,646 P5968 INFO Train loss: 0.389082
2020-06-15 17:26:45,646 P5968 INFO ************ Epoch=3 end ************
2020-06-15 17:31:22,747 P5968 INFO [Metrics] logloss: 0.379363 - AUC: 0.781384
2020-06-15 17:31:22,753 P5968 INFO Save best model: monitor(max): 0.402020
2020-06-15 17:31:23,128 P5968 INFO --- 3235/3235 batches finished ---
2020-06-15 17:31:23,192 P5968 INFO Train loss: 0.386271
2020-06-15 17:31:23,193 P5968 INFO ************ Epoch=4 end ************
2020-06-15 17:36:01,635 P5968 INFO [Metrics] logloss: 0.378686 - AUC: 0.782366
2020-06-15 17:36:01,638 P5968 INFO Save best model: monitor(max): 0.403680
2020-06-15 17:36:02,009 P5968 INFO --- 3235/3235 batches finished ---
2020-06-15 17:36:02,077 P5968 INFO Train loss: 0.385017
2020-06-15 17:36:02,077 P5968 INFO ************ Epoch=5 end ************
2020-06-15 17:40:37,664 P5968 INFO [Metrics] logloss: 0.378357 - AUC: 0.783052
2020-06-15 17:40:37,668 P5968 INFO Save best model: monitor(max): 0.404695
2020-06-15 17:40:38,029 P5968 INFO --- 3235/3235 batches finished ---
2020-06-15 17:40:38,091 P5968 INFO Train loss: 0.384586
2020-06-15 17:40:38,091 P5968 INFO ************ Epoch=6 end ************
2020-06-15 17:45:15,901 P5968 INFO [Metrics] logloss: 0.378264 - AUC: 0.783087
2020-06-15 17:45:15,905 P5968 INFO Save best model: monitor(max): 0.404823
2020-06-15 17:45:16,271 P5968 INFO --- 3235/3235 batches finished ---
2020-06-15 17:45:16,358 P5968 INFO Train loss: 0.384407
2020-06-15 17:45:16,358 P5968 INFO ************ Epoch=7 end ************
2020-06-15 17:49:54,145 P5968 INFO [Metrics] logloss: 0.378233 - AUC: 0.783185
2020-06-15 17:49:54,149 P5968 INFO Save best model: monitor(max): 0.404952
2020-06-15 17:49:54,524 P5968 INFO --- 3235/3235 batches finished ---
2020-06-15 17:49:54,599 P5968 INFO Train loss: 0.384397
2020-06-15 17:49:54,599 P5968 INFO ************ Epoch=8 end ************
2020-06-15 17:54:30,623 P5968 INFO [Metrics] logloss: 0.378117 - AUC: 0.783642
2020-06-15 17:54:30,627 P5968 INFO Save best model: monitor(max): 0.405525
2020-06-15 17:54:31,011 P5968 INFO --- 3235/3235 batches finished ---
2020-06-15 17:54:31,087 P5968 INFO Train loss: 0.384321
2020-06-15 17:54:31,087 P5968 INFO ************ Epoch=9 end ************
2020-06-15 17:59:14,202 P5968 INFO [Metrics] logloss: 0.377946 - AUC: 0.783732
2020-06-15 17:59:14,205 P5968 INFO Save best model: monitor(max): 0.405786
2020-06-15 17:59:14,626 P5968 INFO --- 3235/3235 batches finished ---
2020-06-15 17:59:14,703 P5968 INFO Train loss: 0.384376
2020-06-15 17:59:14,703 P5968 INFO ************ Epoch=10 end ************
2020-06-15 18:03:53,400 P5968 INFO [Metrics] logloss: 0.377932 - AUC: 0.783950
2020-06-15 18:03:53,404 P5968 INFO Save best model: monitor(max): 0.406018
2020-06-15 18:03:53,806 P5968 INFO --- 3235/3235 batches finished ---
2020-06-15 18:03:53,880 P5968 INFO Train loss: 0.384268
2020-06-15 18:03:53,881 P5968 INFO ************ Epoch=11 end ************
2020-06-15 18:08:26,656 P5968 INFO [Metrics] logloss: 0.378049 - AUC: 0.783816
2020-06-15 18:08:26,661 P5968 INFO Monitor(max) STOP: 0.405767 !
2020-06-15 18:08:26,661 P5968 INFO Reduce learning rate on plateau: 0.000100
2020-06-15 18:08:26,661 P5968 INFO --- 3235/3235 batches finished ---
2020-06-15 18:08:26,725 P5968 INFO Train loss: 0.384165
2020-06-15 18:08:26,725 P5968 INFO ************ Epoch=12 end ************
2020-06-15 18:12:58,798 P5968 INFO [Metrics] logloss: 0.385897 - AUC: 0.780981
2020-06-15 18:12:58,801 P5968 INFO Monitor(max) STOP: 0.395084 !
2020-06-15 18:12:58,801 P5968 INFO Reduce learning rate on plateau: 0.000010
2020-06-15 18:12:58,802 P5968 INFO Early stopping at epoch=13
2020-06-15 18:12:58,802 P5968 INFO --- 3235/3235 batches finished ---
2020-06-15 18:12:58,864 P5968 INFO Train loss: 0.353398
2020-06-15 18:12:58,865 P5968 INFO Training finished.
2020-06-15 18:12:58,865 P5968 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Avazu/DCN_avazu/min2/avazu_x4_3bbbc4c9/DCN_avazu_x4_3bbbc4c9_030_6039e913_model.ckpt
2020-06-15 18:12:59,170 P5968 INFO ****** Train/validation evaluation ******
2020-06-15 18:16:09,698 P5968 INFO [Metrics] logloss: 0.348514 - AUC: 0.830079
2020-06-15 18:16:33,366 P5968 INFO [Metrics] logloss: 0.377932 - AUC: 0.783950
2020-06-15 18:16:33,452 P5968 INFO ******** Test evaluation ********
2020-06-15 18:16:33,453 P5968 INFO Loading data...
2020-06-15 18:16:33,453 P5968 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-06-15 18:16:33,905 P5968 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-15 18:16:33,905 P5968 INFO Loading test data done.
2020-06-15 18:16:59,305 P5968 INFO [Metrics] logloss: 0.377924 - AUC: 0.783962

```
