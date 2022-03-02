## FwFM_criteo_x4_001

A hands-on guide to run the FwFM model on the Criteo_x4_001 dataset.

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
Dataset ID: [Criteo_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [FwFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FwFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FwFM_criteo_x4_tuner_config_02](./FwFM_criteo_x4_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FwFM_criteo_x4_001
    nohup python run_expid.py --config ./FwFM_criteo_x4_tuner_config_02 --expid FwFM_criteo_x4_002_3519edbe --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.440797 | 0.811214  |


### Logs
```python
2022-03-02 14:46:37,272 P56869 INFO {
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_9ea3bdfc",
    "debug": "False",
    "embedding_dim": "16",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'na_value': 0, 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'preprocess': 'convert_to_bucket', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'na_value': '', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'Label'}",
    "learning_rate": "0.001",
    "linear_type": "LW",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "FwFM",
    "model_id": "FwFM_criteo_x4_002_3519edbe",
    "model_root": "./Criteo/FwFM_criteo_x4_001/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "5e-06",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x4/test.csv",
    "train_data": "../data/Criteo/Criteo_x4/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x4/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2022-03-02 14:46:37,273 P56869 INFO Set up feature encoder...
2022-03-02 14:46:37,273 P56869 INFO Load feature_encoder from pickle: ../data/Criteo/criteo_x4_9ea3bdfc/feature_encoder.pkl
2022-03-02 14:46:38,265 P56869 INFO Total number of parameters: 15482778.
2022-03-02 14:46:38,265 P56869 INFO Loading data...
2022-03-02 14:46:38,267 P56869 INFO Loading data from h5: ../data/Criteo/criteo_x4_9ea3bdfc/train.h5
2022-03-02 14:46:43,279 P56869 INFO Loading data from h5: ../data/Criteo/criteo_x4_9ea3bdfc/valid.h5
2022-03-02 14:46:45,140 P56869 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2022-03-02 14:46:45,350 P56869 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2022-03-02 14:46:45,351 P56869 INFO Loading train data done.
2022-03-02 14:46:48,084 P56869 INFO Start training: 3668 batches/epoch
2022-03-02 14:46:48,084 P56869 INFO ************ Epoch=1 start ************
2022-03-02 14:58:43,739 P56869 INFO [Metrics] logloss: 0.450626 - AUC: 0.800116
2022-03-02 14:58:43,743 P56869 INFO Save best model: monitor(max): 0.349490
2022-03-02 14:58:43,798 P56869 INFO --- 3668/3668 batches finished ---
2022-03-02 14:58:43,847 P56869 INFO Train loss: 0.463842
2022-03-02 14:58:43,847 P56869 INFO ************ Epoch=1 end ************
2022-03-02 15:10:46,752 P56869 INFO [Metrics] logloss: 0.446472 - AUC: 0.804799
2022-03-02 15:10:46,754 P56869 INFO Save best model: monitor(max): 0.358327
2022-03-02 15:10:46,864 P56869 INFO --- 3668/3668 batches finished ---
2022-03-02 15:10:46,914 P56869 INFO Train loss: 0.455840
2022-03-02 15:10:46,915 P56869 INFO ************ Epoch=2 end ************
2022-03-02 15:22:46,690 P56869 INFO [Metrics] logloss: 0.445139 - AUC: 0.806271
2022-03-02 15:22:46,691 P56869 INFO Save best model: monitor(max): 0.361132
2022-03-02 15:22:46,786 P56869 INFO --- 3668/3668 batches finished ---
2022-03-02 15:22:46,841 P56869 INFO Train loss: 0.453898
2022-03-02 15:22:46,842 P56869 INFO ************ Epoch=3 end ************
2022-03-02 15:34:44,369 P56869 INFO [Metrics] logloss: 0.444461 - AUC: 0.807067
2022-03-02 15:34:44,371 P56869 INFO Save best model: monitor(max): 0.362607
2022-03-02 15:34:44,464 P56869 INFO --- 3668/3668 batches finished ---
2022-03-02 15:34:44,520 P56869 INFO Train loss: 0.453056
2022-03-02 15:34:44,520 P56869 INFO ************ Epoch=4 end ************
2022-03-02 15:46:44,661 P56869 INFO [Metrics] logloss: 0.444274 - AUC: 0.807399
2022-03-02 15:46:44,663 P56869 INFO Save best model: monitor(max): 0.363125
2022-03-02 15:46:44,759 P56869 INFO --- 3668/3668 batches finished ---
2022-03-02 15:46:44,812 P56869 INFO Train loss: 0.452614
2022-03-02 15:46:44,812 P56869 INFO ************ Epoch=5 end ************
2022-03-02 15:58:44,898 P56869 INFO [Metrics] logloss: 0.444012 - AUC: 0.807562
2022-03-02 15:58:44,899 P56869 INFO Save best model: monitor(max): 0.363549
2022-03-02 15:58:44,996 P56869 INFO --- 3668/3668 batches finished ---
2022-03-02 15:58:45,045 P56869 INFO Train loss: 0.452340
2022-03-02 15:58:45,045 P56869 INFO ************ Epoch=6 end ************
2022-03-02 16:10:42,168 P56869 INFO [Metrics] logloss: 0.443883 - AUC: 0.807693
2022-03-02 16:10:42,170 P56869 INFO Save best model: monitor(max): 0.363810
2022-03-02 16:10:42,260 P56869 INFO --- 3668/3668 batches finished ---
2022-03-02 16:10:42,315 P56869 INFO Train loss: 0.452158
2022-03-02 16:10:42,315 P56869 INFO ************ Epoch=7 end ************
2022-03-02 16:22:44,419 P56869 INFO [Metrics] logloss: 0.443772 - AUC: 0.807870
2022-03-02 16:22:44,420 P56869 INFO Save best model: monitor(max): 0.364098
2022-03-02 16:22:44,515 P56869 INFO --- 3668/3668 batches finished ---
2022-03-02 16:22:44,568 P56869 INFO Train loss: 0.452008
2022-03-02 16:22:44,568 P56869 INFO ************ Epoch=8 end ************
2022-03-02 16:34:42,494 P56869 INFO [Metrics] logloss: 0.443756 - AUC: 0.807828
2022-03-02 16:34:42,496 P56869 INFO Monitor(max) STOP: 0.364072 !
2022-03-02 16:34:42,496 P56869 INFO Reduce learning rate on plateau: 0.000100
2022-03-02 16:34:42,496 P56869 INFO --- 3668/3668 batches finished ---
2022-03-02 16:34:42,550 P56869 INFO Train loss: 0.451890
2022-03-02 16:34:42,551 P56869 INFO ************ Epoch=9 end ************
2022-03-02 16:46:44,027 P56869 INFO [Metrics] logloss: 0.441507 - AUC: 0.810300
2022-03-02 16:46:44,028 P56869 INFO Save best model: monitor(max): 0.368793
2022-03-02 16:46:44,128 P56869 INFO --- 3668/3668 batches finished ---
2022-03-02 16:46:44,181 P56869 INFO Train loss: 0.443317
2022-03-02 16:46:44,182 P56869 INFO ************ Epoch=10 end ************
2022-03-02 16:58:42,705 P56869 INFO [Metrics] logloss: 0.441180 - AUC: 0.810683
2022-03-02 16:58:42,707 P56869 INFO Save best model: monitor(max): 0.369503
2022-03-02 16:58:42,795 P56869 INFO --- 3668/3668 batches finished ---
2022-03-02 16:58:42,849 P56869 INFO Train loss: 0.440620
2022-03-02 16:58:42,850 P56869 INFO ************ Epoch=11 end ************
2022-03-02 17:10:41,220 P56869 INFO [Metrics] logloss: 0.441090 - AUC: 0.810798
2022-03-02 17:10:41,221 P56869 INFO Save best model: monitor(max): 0.369708
2022-03-02 17:10:41,308 P56869 INFO --- 3668/3668 batches finished ---
2022-03-02 17:10:41,363 P56869 INFO Train loss: 0.439427
2022-03-02 17:10:41,363 P56869 INFO ************ Epoch=12 end ************
2022-03-02 17:22:47,589 P56869 INFO [Metrics] logloss: 0.441097 - AUC: 0.810831
2022-03-02 17:22:47,590 P56869 INFO Save best model: monitor(max): 0.369734
2022-03-02 17:22:47,689 P56869 INFO --- 3668/3668 batches finished ---
2022-03-02 17:22:47,744 P56869 INFO Train loss: 0.438564
2022-03-02 17:22:47,744 P56869 INFO ************ Epoch=13 end ************
2022-03-02 17:34:47,911 P56869 INFO [Metrics] logloss: 0.441125 - AUC: 0.810815
2022-03-02 17:34:47,912 P56869 INFO Monitor(max) STOP: 0.369691 !
2022-03-02 17:34:47,912 P56869 INFO Reduce learning rate on plateau: 0.000010
2022-03-02 17:34:47,912 P56869 INFO --- 3668/3668 batches finished ---
2022-03-02 17:34:47,967 P56869 INFO Train loss: 0.437825
2022-03-02 17:34:47,967 P56869 INFO ************ Epoch=14 end ************
2022-03-02 17:46:53,556 P56869 INFO [Metrics] logloss: 0.441087 - AUC: 0.810875
2022-03-02 17:46:53,558 P56869 INFO Save best model: monitor(max): 0.369788
2022-03-02 17:46:53,669 P56869 INFO --- 3668/3668 batches finished ---
2022-03-02 17:46:53,724 P56869 INFO Train loss: 0.435120
2022-03-02 17:46:53,724 P56869 INFO ************ Epoch=15 end ************
2022-03-02 17:59:06,231 P56869 INFO [Metrics] logloss: 0.441106 - AUC: 0.810862
2022-03-02 17:59:06,232 P56869 INFO Monitor(max) STOP: 0.369756 !
2022-03-02 17:59:06,232 P56869 INFO Reduce learning rate on plateau: 0.000001
2022-03-02 17:59:06,232 P56869 INFO --- 3668/3668 batches finished ---
2022-03-02 17:59:06,283 P56869 INFO Train loss: 0.434958
2022-03-02 17:59:06,284 P56869 INFO ************ Epoch=16 end ************
2022-03-02 18:11:12,306 P56869 INFO [Metrics] logloss: 0.441108 - AUC: 0.810861
2022-03-02 18:11:12,307 P56869 INFO Monitor(max) STOP: 0.369752 !
2022-03-02 18:11:12,307 P56869 INFO Reduce learning rate on plateau: 0.000001
2022-03-02 18:11:12,307 P56869 INFO Early stopping at epoch=17
2022-03-02 18:11:12,307 P56869 INFO --- 3668/3668 batches finished ---
2022-03-02 18:11:12,360 P56869 INFO Train loss: 0.434601
2022-03-02 18:11:12,361 P56869 INFO Training finished.
2022-03-02 18:11:12,361 P56869 INFO Load best model: /home/XXX/FuxiCTR_v1.0/benchmarks/Criteo/FwFM_criteo_x4_001/criteo_x4_9ea3bdfc/FwFM_criteo_x4_002_3519edbe_model.ckpt
2022-03-02 18:11:12,461 P56869 INFO ****** Validation evaluation ******
2022-03-02 18:11:40,100 P56869 INFO [Metrics] logloss: 0.441087 - AUC: 0.810875
2022-03-02 18:11:40,171 P56869 INFO ******** Test evaluation ********
2022-03-02 18:11:40,171 P56869 INFO Loading data...
2022-03-02 18:11:40,172 P56869 INFO Loading data from h5: ../data/Criteo/criteo_x4_9ea3bdfc/test.h5
2022-03-02 18:11:41,105 P56869 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2022-03-02 18:11:41,105 P56869 INFO Loading test data done.
2022-03-02 18:12:09,040 P56869 INFO [Metrics] logloss: 0.440797 - AUC: 0.811214

```
