## FGCNN_avazu_x4_001

A hands-on guide to run the FGCNN model on the Avazu_x4_001 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

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
Dataset ID: [Avazu_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FGCNN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FGCNN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FGCNN_avazu_x4_tuner_config_02](./FGCNN_avazu_x4_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FGCNN_avazu_x4_001
    nohup python run_expid.py --config ./FGCNN_avazu_x4_tuner_config_02 --expid FGCNN_avazu_x4_010_06d6ae32 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.371139 | 0.794446  |


### Logs
```python
2020-06-18 16:22:03,757 P2168 INFO {
    "batch_size": "10000",
    "channels": "[20, 22, 24, 26]",
    "conv_activation": "Tanh",
    "conv_batch_norm": "True",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "dnn_activations": "ReLU",
    "dnn_batch_norm": "False",
    "dnn_hidden_units": "[2000, 2000, 2000, 2000]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-09",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "kernel_heights": "[9, 9, 9, 9]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FGCNN",
    "model_id": "FGCNN_avazu_x4_3bbbc4c9_010_6722e191",
    "model_root": "./Avazu/FGCNN_avazu/min2/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "pooling_sizes": "[2, 2, 2, 2]",
    "recombined_channels": "[3, 3, 3, 3]",
    "save_best_only": "True",
    "seed": "2019",
    "share_embedding": "False",
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
2020-06-18 16:22:03,759 P2168 INFO Set up feature encoder...
2020-06-18 16:22:03,759 P2168 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-06-18 16:22:03,760 P2168 INFO Loading data...
2020-06-18 16:22:03,764 P2168 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-06-18 16:22:07,399 P2168 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-06-18 16:22:08,737 P2168 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-18 16:22:08,839 P2168 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-18 16:22:08,839 P2168 INFO Loading train data done.
2020-06-18 16:22:17,831 P2168 INFO Start training: 3235 batches/epoch
2020-06-18 16:22:17,832 P2168 INFO ************ Epoch=1 start ************
2020-06-18 16:44:53,138 P2168 INFO [Metrics] logloss: 0.371220 - AUC: 0.794330
2020-06-18 16:44:53,139 P2168 INFO Save best model: monitor(max): 0.423110
2020-06-18 16:44:54,242 P2168 INFO --- 3235/3235 batches finished ---
2020-06-18 16:44:54,289 P2168 INFO Train loss: 0.381013
2020-06-18 16:44:54,289 P2168 INFO ************ Epoch=1 end ************
2020-06-18 17:07:28,669 P2168 INFO [Metrics] logloss: 0.380840 - AUC: 0.788669
2020-06-18 17:07:28,672 P2168 INFO Monitor(max) STOP: 0.407829 !
2020-06-18 17:07:28,673 P2168 INFO Reduce learning rate on plateau: 0.000100
2020-06-18 17:07:28,673 P2168 INFO --- 3235/3235 batches finished ---
2020-06-18 17:07:28,723 P2168 INFO Train loss: 0.329856
2020-06-18 17:07:28,723 P2168 INFO ************ Epoch=2 end ************
2020-06-18 17:30:04,354 P2168 INFO [Metrics] logloss: 0.434746 - AUC: 0.775175
2020-06-18 17:30:04,357 P2168 INFO Monitor(max) STOP: 0.340429 !
2020-06-18 17:30:04,357 P2168 INFO Reduce learning rate on plateau: 0.000010
2020-06-18 17:30:04,357 P2168 INFO Early stopping at epoch=3
2020-06-18 17:30:04,357 P2168 INFO --- 3235/3235 batches finished ---
2020-06-18 17:30:04,406 P2168 INFO Train loss: 0.280297
2020-06-18 17:30:04,406 P2168 INFO Training finished.
2020-06-18 17:30:04,406 P2168 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Avazu/FGCNN_avazu/min2/avazu_x4_3bbbc4c9/FGCNN_avazu_x4_3bbbc4c9_010_6722e191_model.ckpt
2020-06-18 17:30:05,464 P2168 INFO ****** Train/validation evaluation ******
2020-06-18 17:34:30,104 P2168 INFO [Metrics] logloss: 0.334101 - AUC: 0.850783
2020-06-18 17:35:02,570 P2168 INFO [Metrics] logloss: 0.371220 - AUC: 0.794330
2020-06-18 17:35:02,630 P2168 INFO ******** Test evaluation ********
2020-06-18 17:35:02,631 P2168 INFO Loading data...
2020-06-18 17:35:02,631 P2168 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-06-18 17:35:03,226 P2168 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-18 17:35:03,226 P2168 INFO Loading test data done.
2020-06-18 17:35:35,773 P2168 INFO [Metrics] logloss: 0.371139 - AUC: 0.794446

```
