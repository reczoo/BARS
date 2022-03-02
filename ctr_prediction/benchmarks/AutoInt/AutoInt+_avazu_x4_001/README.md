## AutoInt+_avazu_x4_001

A hands-on guide to run the AutoInt model on the Avazu_x4_001 dataset.

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

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [AutoInt](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt+_avazu_x4_tuner_config_03](./AutoInt+_avazu_x4_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AutoInt+_avazu_x4_001
    nohup python run_expid.py --config ./AutoInt+_avazu_x4_tuner_config_03 --expid AutoInt_avazu_x4_001_eec3dc28 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.374625 | 0.790215  |


### Logs
```python
2020-06-14 14:45:17,863 P835 INFO {
    "attention_dim": "128",
    "attention_layers": "7",
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-07",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AutoInt",
    "model_id": "AutoInt_avazu_x4_3bbbc4c9_001_9855ad51",
    "model_root": "./Avazu/AutoInt_avazu/min2/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "1",
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
    "use_residual": "False",
    "use_scale": "False",
    "use_wide": "True",
    "valid_data": "../data/Avazu/avazu_x4_3bbbc4c9/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-14 14:45:17,865 P835 INFO Set up feature encoder...
2020-06-14 14:45:17,865 P835 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-06-14 14:45:17,866 P835 INFO Loading data...
2020-06-14 14:45:17,872 P835 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-06-14 14:45:20,593 P835 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-06-14 14:45:22,132 P835 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-14 14:45:22,341 P835 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-14 14:45:22,341 P835 INFO Loading train data done.
2020-06-14 14:45:32,514 P835 INFO Start training: 3235 batches/epoch
2020-06-14 14:45:32,515 P835 INFO ************ Epoch=1 start ************
2020-06-14 14:57:36,915 P835 INFO [Metrics] logloss: 0.375214 - AUC: 0.787696
2020-06-14 14:57:36,916 P835 INFO Save best model: monitor(max): 0.412482
2020-06-14 14:57:37,236 P835 INFO --- 3235/3235 batches finished ---
2020-06-14 14:57:37,305 P835 INFO Train loss: 0.386751
2020-06-14 14:57:37,305 P835 INFO ************ Epoch=1 end ************
2020-06-14 15:09:37,779 P835 INFO [Metrics] logloss: 0.374783 - AUC: 0.789925
2020-06-14 15:09:37,782 P835 INFO Save best model: monitor(max): 0.415142
2020-06-14 15:09:38,277 P835 INFO --- 3235/3235 batches finished ---
2020-06-14 15:09:38,321 P835 INFO Train loss: 0.355714
2020-06-14 15:09:38,321 P835 INFO ************ Epoch=2 end ************
2020-06-14 15:21:37,598 P835 INFO [Metrics] logloss: 0.379842 - AUC: 0.786737
2020-06-14 15:21:37,600 P835 INFO Monitor(max) STOP: 0.406895 !
2020-06-14 15:21:37,600 P835 INFO Reduce learning rate on plateau: 0.000100
2020-06-14 15:21:37,600 P835 INFO --- 3235/3235 batches finished ---
2020-06-14 15:21:37,651 P835 INFO Train loss: 0.335383
2020-06-14 15:21:37,652 P835 INFO ************ Epoch=3 end ************
2020-06-14 15:33:42,091 P835 INFO [Metrics] logloss: 0.428456 - AUC: 0.773690
2020-06-14 15:33:42,094 P835 INFO Monitor(max) STOP: 0.345233 !
2020-06-14 15:33:42,094 P835 INFO Reduce learning rate on plateau: 0.000010
2020-06-14 15:33:42,097 P835 INFO Early stopping at epoch=4
2020-06-14 15:33:42,097 P835 INFO --- 3235/3235 batches finished ---
2020-06-14 15:33:42,155 P835 INFO Train loss: 0.297909
2020-06-14 15:33:42,155 P835 INFO Training finished.
2020-06-14 15:33:42,155 P835 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Avazu/AutoInt_avazu/min2/avazu_x4_3bbbc4c9/AutoInt_avazu_x4_3bbbc4c9_001_9855ad51_model.ckpt
2020-06-14 15:33:42,758 P835 INFO ****** Train/validation evaluation ******
2020-06-14 15:37:12,220 P835 INFO [Metrics] logloss: 0.320231 - AUC: 0.863627
2020-06-14 15:37:36,970 P835 INFO [Metrics] logloss: 0.374783 - AUC: 0.789925
2020-06-14 15:37:37,039 P835 INFO ******** Test evaluation ********
2020-06-14 15:37:37,039 P835 INFO Loading data...
2020-06-14 15:37:37,039 P835 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-06-14 15:37:37,743 P835 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-14 15:37:37,743 P835 INFO Loading test data done.
2020-06-14 15:38:02,360 P835 INFO [Metrics] logloss: 0.374625 - AUC: 0.790215

```
