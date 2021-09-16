## AFN_Criteo_x4_002

A notebook to benchmark AFN on Criteo_x4_002 dataset.

Author: [XUEPAI Team](https://github.com/xue-pai)


### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
  RAM: 500G+
  ```
+ Software

  ```python
  python: 3.6.5
  pandas: 1.0.0
  numpy: 1.18.1
  ```

### Dataset
In this setting, we follow the winner's solution of the Criteo challenge to discretize each integer value x to ⌊log2 (x)⌋, if x > 2; and x = 1 otherwise. For all categorical fields, we replace infrequent features with a default <OOV> token by setting the threshold min_category_count=10. Note that we do not follow the exact preprocessing steps in AutoInt, because this preprocessing performs much better.

To make a fair comparison, we fix embedding_dim=16 as with AutoInt.
### Code




### Results
```python
[Metrics] logloss 0.441809 - AUC 0.809716
```


### Logs
```python
2020-06-07 114555,911 P42904 INFO {
    afn_activations relu,
    afn_dropout 0.1,
    afn_hidden_units [2000, 2000, 2000],
    batch_norm True,
    batch_size 2000,
    dataset_id criteo_x4_001_be98441d,
    dnn_activations relu,
    dnn_dropout 0,
    dnn_hidden_units [],
    embedding_dim 40,
    embedding_dropout 0,
    embedding_regularizer 1e-06,
    ensemble_dnn False,
    epochs 100,
    every_x_epochs 1,
    learning_rate 0.001,
    logarithmic_neurons 1500,
    loss binary_crossentropy,
    metrics ['logloss', 'AUC'],
    model AFN,
    model_id AFN_criteo_x4_080_a2c67c31,
    model_root .AvazuAFN_criteo,
    monitor {'AUC' 1, 'logloss' -1},
    monitor_mode max,
    net_regularizer 0,
    optimizer adam,
    patience 2,
    pickle_feature_encoder True,
    save_best_only True,
    seed 2019,
    shuffle True,
    task binary_classification,
    use_hdf5 True,
    verbose 0,
    workers 3,
    data_format h5,
    data_root ..dataCriteo,
    test_data ..dataCriteocriteo_x4_001_be98441dtest.h5,
    train_data ..dataCriteocriteo_x4_001_be98441dtrain.h5,
    valid_data ..dataCriteocriteo_x4_001_be98441dvalid.h5,
    version pytorch,
    gpu 0
}
2020-06-07 114555,912 P42904 INFO Set up feature encoder...
2020-06-07 114555,912 P42904 INFO Load feature_map from json ..dataCriteocriteo_x4_001_be98441dfeature_map.json
2020-06-07 114555,912 P42904 INFO Loading data...
2020-06-07 114555,915 P42904 INFO Loading data from h5 ..dataCriteocriteo_x4_001_be98441dtrain.h5
2020-06-07 114600,188 P42904 INFO Loading data from h5 ..dataCriteocriteo_x4_001_be98441dvalid.h5
2020-06-07 114613,982 P42904 INFO Train samples total36672493, pos9396350, neg27276143, ratio25.62%
2020-06-07 114614,286 P42904 INFO Validation samples total4584062, pos1174544, neg3409518, ratio25.62%
2020-06-07 114614,287 P42904 INFO Loading train data done.
2020-06-07 114643,460 P42904 INFO  Start training 18337 batchesepoch 
2020-06-07 134406,386 P42904 INFO [Metrics] logloss 0.447709 - AUC 0.803430
2020-06-07 134406,500 P42904 INFO Save best model monitor(max) 0.355722
2020-06-07 134421,665 P42904 INFO --- 1833718337 batches finished ---
2020-06-07 134421,723 P42904 INFO Train loss 0.460343
2020-06-07 134421,724 P42904 INFO  Epoch=1 end 
2020-06-07 154144,369 P42904 INFO [Metrics] logloss 0.443359 - AUC 0.808010
2020-06-07 154144,522 P42904 INFO Save best model monitor(max) 0.364651
2020-06-07 154158,322 P42904 INFO --- 1833718337 batches finished ---
2020-06-07 154158,419 P42904 INFO Train loss 0.451480
2020-06-07 154158,419 P42904 INFO  Epoch=2 end 
2020-06-07 173920,725 P42904 INFO [Metrics] logloss 0.442305 - AUC 0.809121
2020-06-07 173920,864 P42904 INFO Save best model monitor(max) 0.366815
2020-06-07 173935,269 P42904 INFO --- 1833718337 batches finished ---
2020-06-07 173935,376 P42904 INFO Train loss 0.447150
2020-06-07 173935,376 P42904 INFO  Epoch=3 end 
2020-06-07 193656,522 P42904 INFO [Metrics] logloss 0.444041 - AUC 0.807416
2020-06-07 193656,647 P42904 INFO Monitor(max) STOP 0.363375 !
2020-06-07 193656,647 P42904 INFO Reduce learning rate on plateau 0.000100
2020-06-07 193656,647 P42904 INFO --- 1833718337 batches finished ---
2020-06-07 193656,765 P42904 INFO Train loss 0.440785
2020-06-07 193656,765 P42904 INFO  Epoch=4 end 
2020-06-07 213412,195 P42904 INFO [Metrics] logloss 0.465326 - AUC 0.796353
2020-06-07 213412,314 P42904 INFO Monitor(max) STOP 0.331027 !
2020-06-07 213412,315 P42904 INFO Reduce learning rate on plateau 0.000010
2020-06-07 213412,315 P42904 INFO Early stopping at epoch=5
2020-06-07 213412,315 P42904 INFO --- 1833718337 batches finished ---
2020-06-07 213412,435 P42904 INFO Train loss 0.409479
2020-06-07 213412,435 P42904 INFO Training finished.
2020-06-07 213412,435 P42904 INFO Load best model homehispacecontainerdataxxxFuxiCTRbenchmarksAvazuAFN_criteocriteo_x4_001_be98441dAFN_criteo_x4_080_a2c67c31_criteo_x4_001_be98441d_model.ckpt
2020-06-07 213444,340 P42904 INFO  Trainvalidation evaluation 
2020-06-07 221109,461 P42904 INFO [Metrics] logloss 0.430320 - AUC 0.822152
2020-06-07 221539,154 P42904 INFO [Metrics] logloss 0.442305 - AUC 0.809121
2020-06-07 221539,381 P42904 INFO  Test evaluation 
2020-06-07 221539,381 P42904 INFO Loading data...
2020-06-07 221539,381 P42904 INFO Loading data from h5 ..dataCriteocriteo_x4_001_be98441dtest.h5
2020-06-07 221540,102 P42904 INFO Test samples total4584062, pos1174544, neg3409518, ratio25.62%
2020-06-07 221540,102 P42904 INFO Loading test data done.
2020-06-07 222008,239 P42904 INFO [Metrics] logloss 0.441809 - AUC 0.809716



```
