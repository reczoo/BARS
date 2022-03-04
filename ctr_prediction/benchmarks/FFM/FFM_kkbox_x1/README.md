## FFM_kkbox_x1

A hands-on guide to run the FFM model on the KKBox_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

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
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [KKBox_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/KKBox/README.md#KKBox_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FFM_kkbox_x1_tuner_config_02](./FFM_kkbox_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FFM_kkbox_x1
    nohup python run_expid.py --config ./FFM_kkbox_x1_tuner_config_02 --expid FFM_kkbox_x1_002_62edeeee --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.500291 | 0.834831  |


### Logs
```python
2020-04-20 09:35:25,626 P14071 INFO {
    "batch_size": "10000",
    "dataset_id": "kkbox_x1_001_c5c9c6e3",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FFM",
    "model_id": "FFM_kkbox_x1_002_be4e6111",
    "model_root": "./KKBox/FFM_kkbox/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-05",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/KKBox/",
    "test_data": "../data/KKBox/kkbox_x1_001_c5c9c6e3/test.h5",
    "train_data": "../data/KKBox/kkbox_x1_001_c5c9c6e3/train.h5",
    "valid_data": "../data/KKBox/kkbox_x1_001_c5c9c6e3/valid.h5",
    "version": "pytorch",
    "gpu": "1"
}
2020-04-20 09:35:25,628 P14071 INFO Set up feature encoder...
2020-04-20 09:35:25,628 P14071 INFO Load feature_map from json: ../data/KKBox/kkbox_x1_001_c5c9c6e3/feature_map.json
2020-04-20 09:35:25,628 P14071 INFO Loading data...
2020-04-20 09:35:25,630 P14071 INFO Loading data from h5: ../data/KKBox/kkbox_x1_001_c5c9c6e3/train.h5
2020-04-20 09:35:25,920 P14071 INFO Loading data from h5: ../data/KKBox/kkbox_x1_001_c5c9c6e3/valid.h5
2020-04-20 09:35:26,113 P14071 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2020-04-20 09:35:26,133 P14071 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-04-20 09:35:26,133 P14071 INFO Loading train data done.
2020-04-20 09:35:33,568 P14071 INFO **** Start training: 591 batches/epoch ****
2020-04-20 09:41:53,137 P14071 INFO [Metrics] logloss: 0.532127 - AUC: 0.808620
2020-04-20 09:41:53,148 P14071 INFO Save best model: monitor(max): 0.276493
2020-04-20 09:41:53,559 P14071 INFO --- 591/591 batches finished ---
2020-04-20 09:41:53,598 P14071 INFO Train loss: 0.584341
2020-04-20 09:41:53,598 P14071 INFO ************ Epoch=1 end ************
2020-04-20 09:48:13,401 P14071 INFO [Metrics] logloss: 0.521322 - AUC: 0.818074
2020-04-20 09:48:13,418 P14071 INFO Save best model: monitor(max): 0.296752
2020-04-20 09:48:14,100 P14071 INFO --- 591/591 batches finished ---
2020-04-20 09:48:14,145 P14071 INFO Train loss: 0.558407
2020-04-20 09:48:14,145 P14071 INFO ************ Epoch=2 end ************
2020-04-20 09:57:39,983 P14071 INFO [Metrics] logloss: 0.516305 - AUC: 0.822039
2020-04-20 09:57:39,996 P14071 INFO Save best model: monitor(max): 0.305735
2020-04-20 09:57:41,061 P14071 INFO --- 591/591 batches finished ---
2020-04-20 09:57:41,104 P14071 INFO Train loss: 0.549400
2020-04-20 09:57:41,105 P14071 INFO ************ Epoch=3 end ************
2020-04-20 10:08:57,166 P14071 INFO [Metrics] logloss: 0.513589 - AUC: 0.824211
2020-04-20 10:08:57,185 P14071 INFO Save best model: monitor(max): 0.310621
2020-04-20 10:08:57,870 P14071 INFO --- 591/591 batches finished ---
2020-04-20 10:08:57,914 P14071 INFO Train loss: 0.542984
2020-04-20 10:08:57,914 P14071 INFO ************ Epoch=4 end ************
2020-04-20 10:20:10,000 P14071 INFO [Metrics] logloss: 0.510789 - AUC: 0.826527
2020-04-20 10:20:10,018 P14071 INFO Save best model: monitor(max): 0.315739
2020-04-20 10:20:11,139 P14071 INFO --- 591/591 batches finished ---
2020-04-20 10:20:11,184 P14071 INFO Train loss: 0.537675
2020-04-20 10:20:11,184 P14071 INFO ************ Epoch=5 end ************
2020-04-20 10:31:34,862 P14071 INFO [Metrics] logloss: 0.509490 - AUC: 0.827553
2020-04-20 10:31:34,879 P14071 INFO Save best model: monitor(max): 0.318063
2020-04-20 10:31:35,934 P14071 INFO --- 591/591 batches finished ---
2020-04-20 10:31:35,990 P14071 INFO Train loss: 0.533076
2020-04-20 10:31:35,990 P14071 INFO ************ Epoch=6 end ************
2020-04-20 10:42:09,770 P14071 INFO [Metrics] logloss: 0.508264 - AUC: 0.828493
2020-04-20 10:42:09,787 P14071 INFO Save best model: monitor(max): 0.320228
2020-04-20 10:42:10,895 P14071 INFO --- 591/591 batches finished ---
2020-04-20 10:42:10,939 P14071 INFO Train loss: 0.528957
2020-04-20 10:42:10,940 P14071 INFO ************ Epoch=7 end ************
2020-04-20 10:53:35,392 P14071 INFO [Metrics] logloss: 0.507192 - AUC: 0.829405
2020-04-20 10:53:35,413 P14071 INFO Save best model: monitor(max): 0.322213
2020-04-20 10:53:36,419 P14071 INFO --- 591/591 batches finished ---
2020-04-20 10:53:36,463 P14071 INFO Train loss: 0.525585
2020-04-20 10:53:36,463 P14071 INFO ************ Epoch=8 end ************
2020-04-20 11:04:55,770 P14071 INFO [Metrics] logloss: 0.506640 - AUC: 0.829839
2020-04-20 11:04:55,785 P14071 INFO Save best model: monitor(max): 0.323199
2020-04-20 11:04:56,479 P14071 INFO --- 591/591 batches finished ---
2020-04-20 11:04:56,544 P14071 INFO Train loss: 0.522550
2020-04-20 11:04:56,544 P14071 INFO ************ Epoch=9 end ************
2020-04-20 11:16:21,195 P14071 INFO [Metrics] logloss: 0.506447 - AUC: 0.830041
2020-04-20 11:16:21,209 P14071 INFO Save best model: monitor(max): 0.323593
2020-04-20 11:16:22,299 P14071 INFO --- 591/591 batches finished ---
2020-04-20 11:16:22,343 P14071 INFO Train loss: 0.519565
2020-04-20 11:16:22,343 P14071 INFO ************ Epoch=10 end ************
2020-04-20 11:27:23,843 P14071 INFO [Metrics] logloss: 0.506462 - AUC: 0.830012
2020-04-20 11:27:23,861 P14071 INFO Monitor(max) STOP: 0.323550 !
2020-04-20 11:27:23,861 P14071 INFO Reduce learning rate on plateau: 0.000100
2020-04-20 11:27:23,862 P14071 INFO --- 591/591 batches finished ---
2020-04-20 11:27:23,946 P14071 INFO Train loss: 0.517544
2020-04-20 11:27:23,946 P14071 INFO ************ Epoch=11 end ************
2020-04-20 11:38:19,531 P14071 INFO [Metrics] logloss: 0.502165 - AUC: 0.833389
2020-04-20 11:38:19,544 P14071 INFO Save best model: monitor(max): 0.331224
2020-04-20 11:38:20,263 P14071 INFO --- 591/591 batches finished ---
2020-04-20 11:38:20,310 P14071 INFO Train loss: 0.488491
2020-04-20 11:38:20,310 P14071 INFO ************ Epoch=12 end ************
2020-04-20 11:49:44,821 P14071 INFO [Metrics] logloss: 0.501226 - AUC: 0.834110
2020-04-20 11:49:44,838 P14071 INFO Save best model: monitor(max): 0.332884
2020-04-20 11:49:46,020 P14071 INFO --- 591/591 batches finished ---
2020-04-20 11:49:46,064 P14071 INFO Train loss: 0.483865
2020-04-20 11:49:46,064 P14071 INFO ************ Epoch=13 end ************
2020-04-20 12:01:09,414 P14071 INFO [Metrics] logloss: 0.500857 - AUC: 0.834409
2020-04-20 12:01:09,433 P14071 INFO Save best model: monitor(max): 0.333551
2020-04-20 12:01:10,683 P14071 INFO --- 591/591 batches finished ---
2020-04-20 12:01:10,729 P14071 INFO Train loss: 0.481654
2020-04-20 12:01:10,729 P14071 INFO ************ Epoch=14 end ************
2020-04-20 12:11:57,191 P14071 INFO [Metrics] logloss: 0.500737 - AUC: 0.834529
2020-04-20 12:11:57,203 P14071 INFO Save best model: monitor(max): 0.333791
2020-04-20 12:11:58,095 P14071 INFO --- 591/591 batches finished ---
2020-04-20 12:11:58,139 P14071 INFO Train loss: 0.480157
2020-04-20 12:11:58,139 P14071 INFO ************ Epoch=15 end ************
2020-04-20 12:23:20,454 P14071 INFO [Metrics] logloss: 0.500691 - AUC: 0.834590
2020-04-20 12:23:20,466 P14071 INFO Save best model: monitor(max): 0.333899
2020-04-20 12:23:21,602 P14071 INFO --- 591/591 batches finished ---
2020-04-20 12:23:21,664 P14071 INFO Train loss: 0.479026
2020-04-20 12:23:21,665 P14071 INFO ************ Epoch=16 end ************
2020-04-20 12:34:39,198 P14071 INFO [Metrics] logloss: 0.500756 - AUC: 0.834579
2020-04-20 12:34:39,211 P14071 INFO Monitor(max) STOP: 0.333823 !
2020-04-20 12:34:39,211 P14071 INFO Reduce learning rate on plateau: 0.000010
2020-04-20 12:34:39,211 P14071 INFO --- 591/591 batches finished ---
2020-04-20 12:34:39,275 P14071 INFO Train loss: 0.478040
2020-04-20 12:34:39,275 P14071 INFO ************ Epoch=17 end ************
2020-04-20 12:45:41,269 P14071 INFO [Metrics] logloss: 0.500742 - AUC: 0.834604
2020-04-20 12:45:41,288 P14071 INFO Monitor(max) STOP: 0.333862 !
2020-04-20 12:45:41,289 P14071 INFO Reduce learning rate on plateau: 0.000001
2020-04-20 12:45:41,289 P14071 INFO Early stopping at epoch=18
2020-04-20 12:45:41,289 P14071 INFO --- 591/591 batches finished ---
2020-04-20 12:45:41,359 P14071 INFO Train loss: 0.473142
2020-04-20 12:45:41,360 P14071 INFO Training finished.
2020-04-20 12:45:41,360 P14071 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/KKBox/FFM_kkbox/kkbox_x1_001_c5c9c6e3/FFM_kkbox_x1_002_be4e6111_kkbox_x1_001_c5c9c6e3_model.ckpt
2020-04-20 12:45:42,255 P14071 INFO ****** Train/validation evaluation ******
2020-04-20 12:47:04,860 P14071 INFO [Metrics] logloss: 0.431065 - AUC: 0.886449
2020-04-20 12:47:25,420 P14071 INFO [Metrics] logloss: 0.500691 - AUC: 0.834590
2020-04-20 12:47:25,514 P14071 INFO ******** Test evaluation ********
2020-04-20 12:47:25,514 P14071 INFO Loading data...
2020-04-20 12:47:25,514 P14071 INFO Loading data from h5: ../data/KKBox/kkbox_x1_001_c5c9c6e3/test.h5
2020-04-20 12:47:25,575 P14071 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-04-20 12:47:25,575 P14071 INFO Loading test data done.
2020-04-20 12:47:38,102 P14071 INFO [Metrics] logloss: 0.500291 - AUC: 0.834831

```
