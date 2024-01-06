## MF-CCL_amazonbooks_x0 

A notebook to benchmark MF-CCL on amazonbooks_x0 dataset.

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
amazonbooks_x0 follows the data split and preprocessing steps in NGCF and LightGCN. We directly transform the formats of the data from their [repo](https://github.com/kuandeng/LightGCN/tree/master/Data).


### Code
The implementation code is already available at https://github.com/xue-pai/DEEM. 

Please refer to the [configuration file](https://github.com/xue-pai/DEEM/blob/master/benchmarks/Yelp18/SimpleX_yelp18_x0/SimpleX_yelp18_x0_tuner_config.yaml).
### Results
```
2021-01-13 06:08:59,104 P17485 INFO [Metrics] Recall(k=20): 0.055937 - Recall(k=50): 0.096077 - NDCG(k=20): 0.044708 - NDCG(k=50): 0.059571 - HitRate(k=20): 0.294288 - HitRate(k=50): 0.442676
```

### Logs
```
2021-01-13 00:24:46,458 P17485 INFO Set up feature encoder...
2021-01-13 00:24:46,468 P17485 INFO Load feature_map from json: ../data/AmazonBooks/amazonbooks_x0_37e049e0/feature_map.json
2021-01-13 00:24:55,144 P17485 INFO Total number of parameters: 9231616.
2021-01-13 00:24:55,145 P17485 INFO Loading data...
2021-01-13 00:24:55,152 P17485 INFO Loading data from h5: ../data/AmazonBooks/amazonbooks_x0_37e049e0/train.h5
2021-01-13 00:24:55,310 P17485 INFO Loading data from h5: ../data/AmazonBooks/amazonbooks_x0_37e049e0/item_corpus.h5
2021-01-13 00:24:56,368 P17485 INFO Loading data from h5: ../data/AmazonBooks/amazonbooks_x0_37e049e0/valid.h5
2021-01-13 00:24:56,695 P17485 INFO Loading data from h5: ../data/AmazonBooks/amazonbooks_x0_37e049e0/item_corpus.h5
2021-01-13 00:24:56,696 P17485 INFO Train samples: total/2380730, blocks/1
2021-01-13 00:24:56,697 P17485 INFO Validation samples: total/52639, blocks/1
2021-01-13 00:24:56,697 P17485 INFO Loading train data done.
2021-01-13 00:24:56,697 P17485 INFO **** Start training: 4650 batches/epoch ****
2021-01-13 00:24:56,698 P17485 INFO Negative sampling num_negs=1500
2021-01-13 00:32:57,908 P17485 INFO Negative sampling done
2021-01-13 00:35:36,448 P17485 INFO --- Start evaluation ---
2021-01-13 00:35:44,586 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 00:38:15,923 P17485 INFO [Metrics] Recall(k=20): 0.014936 - Recall(k=50): 0.027304 - NDCG(k=20): 0.012565 - NDCG(k=50): 0.017198 - HitRate(k=20): 0.108665 - HitRate(k=50): 0.183970
2021-01-13 00:38:15,951 P17485 INFO Save best model: monitor(max): 0.014936
2021-01-13 00:38:15,988 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 00:38:16,606 P17485 INFO Train loss: 1.141405
2021-01-13 00:38:16,606 P17485 INFO ************ Epoch=1 end ************
2021-01-13 00:38:16,607 P17485 INFO Negative sampling num_negs=1500
2021-01-13 00:39:21,853 P17485 INFO Negative sampling done
2021-01-13 00:47:07,820 P17485 INFO --- Start evaluation ---
2021-01-13 00:47:09,672 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 00:51:02,847 P17485 INFO [Metrics] Recall(k=20): 0.033555 - Recall(k=50): 0.059508 - NDCG(k=20): 0.027114 - NDCG(k=50): 0.036814 - HitRate(k=20): 0.199814 - HitRate(k=50): 0.318908
2021-01-13 00:51:02,877 P17485 INFO Save best model: monitor(max): 0.033555
2021-01-13 00:51:02,916 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 00:51:02,997 P17485 INFO Train loss: 0.982888
2021-01-13 00:51:02,997 P17485 INFO ************ Epoch=2 end ************
2021-01-13 00:51:02,998 P17485 INFO Negative sampling num_negs=1500
2021-01-13 00:52:03,925 P17485 INFO Negative sampling done
2021-01-13 00:59:21,893 P17485 INFO --- Start evaluation ---
2021-01-13 00:59:23,378 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 01:00:45,996 P17485 INFO [Metrics] Recall(k=20): 0.038393 - Recall(k=50): 0.067750 - NDCG(k=20): 0.030834 - NDCG(k=50): 0.041733 - HitRate(k=20): 0.223352 - HitRate(k=50): 0.351488
2021-01-13 01:00:46,018 P17485 INFO Save best model: monitor(max): 0.038393
2021-01-13 01:00:46,062 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 01:00:46,156 P17485 INFO Train loss: 0.972364
2021-01-13 01:00:46,156 P17485 INFO ************ Epoch=3 end ************
2021-01-13 01:00:46,157 P17485 INFO Negative sampling num_negs=1500
2021-01-13 01:01:47,427 P17485 INFO Negative sampling done
2021-01-13 01:09:17,640 P17485 INFO --- Start evaluation ---
2021-01-13 01:09:20,964 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 01:10:45,399 P17485 INFO [Metrics] Recall(k=20): 0.040168 - Recall(k=50): 0.070192 - NDCG(k=20): 0.032471 - NDCG(k=50): 0.043649 - HitRate(k=20): 0.228025 - HitRate(k=50): 0.358688
2021-01-13 01:10:45,420 P17485 INFO Save best model: monitor(max): 0.040168
2021-01-13 01:10:45,463 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 01:10:45,588 P17485 INFO Train loss: 0.968707
2021-01-13 01:10:45,588 P17485 INFO ************ Epoch=4 end ************
2021-01-13 01:10:45,589 P17485 INFO Negative sampling num_negs=1500
2021-01-13 01:11:47,265 P17485 INFO Negative sampling done
2021-01-13 01:19:08,278 P17485 INFO --- Start evaluation ---
2021-01-13 01:19:10,879 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 01:20:39,810 P17485 INFO [Metrics] Recall(k=20): 0.041018 - Recall(k=50): 0.071391 - NDCG(k=20): 0.032711 - NDCG(k=50): 0.043993 - HitRate(k=20): 0.232888 - HitRate(k=50): 0.363913
2021-01-13 01:20:39,843 P17485 INFO Save best model: monitor(max): 0.041018
2021-01-13 01:20:39,897 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 01:20:39,972 P17485 INFO Train loss: 0.964456
2021-01-13 01:20:39,972 P17485 INFO ************ Epoch=5 end ************
2021-01-13 01:20:39,973 P17485 INFO Negative sampling num_negs=1500
2021-01-13 01:21:42,832 P17485 INFO Negative sampling done
2021-01-13 01:28:44,066 P17485 INFO --- Start evaluation ---
2021-01-13 01:28:48,179 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 01:30:08,107 P17485 INFO [Metrics] Recall(k=20): 0.041089 - Recall(k=50): 0.071622 - NDCG(k=20): 0.032779 - NDCG(k=50): 0.044147 - HitRate(k=20): 0.233971 - HitRate(k=50): 0.364141
2021-01-13 01:30:08,124 P17485 INFO Save best model: monitor(max): 0.041089
2021-01-13 01:30:08,167 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 01:30:08,243 P17485 INFO Train loss: 0.961036
2021-01-13 01:30:08,244 P17485 INFO ************ Epoch=6 end ************
2021-01-13 01:30:08,244 P17485 INFO Negative sampling num_negs=1500
2021-01-13 01:31:10,412 P17485 INFO Negative sampling done
2021-01-13 01:37:47,116 P17485 INFO --- Start evaluation ---
2021-01-13 01:37:51,145 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 01:39:11,423 P17485 INFO [Metrics] Recall(k=20): 0.041444 - Recall(k=50): 0.072647 - NDCG(k=20): 0.033262 - NDCG(k=50): 0.044864 - HitRate(k=20): 0.235453 - HitRate(k=50): 0.367883
2021-01-13 01:39:11,441 P17485 INFO Save best model: monitor(max): 0.041444
2021-01-13 01:39:11,483 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 01:39:11,595 P17485 INFO Train loss: 0.958214
2021-01-13 01:39:11,595 P17485 INFO ************ Epoch=7 end ************
2021-01-13 01:39:11,596 P17485 INFO Negative sampling num_negs=1500
2021-01-13 01:40:22,337 P17485 INFO Negative sampling done
2021-01-13 01:47:14,974 P17485 INFO --- Start evaluation ---
2021-01-13 01:47:17,912 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 01:48:47,213 P17485 INFO [Metrics] Recall(k=20): 0.041667 - Recall(k=50): 0.071962 - NDCG(k=20): 0.033340 - NDCG(k=50): 0.044635 - HitRate(k=20): 0.235966 - HitRate(k=50): 0.365983
2021-01-13 01:48:47,231 P17485 INFO Save best model: monitor(max): 0.041667
2021-01-13 01:48:47,274 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 01:48:47,358 P17485 INFO Train loss: 0.955999
2021-01-13 01:48:47,359 P17485 INFO ************ Epoch=8 end ************
2021-01-13 01:48:47,359 P17485 INFO Negative sampling num_negs=1500
2021-01-13 01:50:05,619 P17485 INFO Negative sampling done
2021-01-13 01:58:07,410 P17485 INFO --- Start evaluation ---
2021-01-13 01:58:10,054 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 01:59:53,561 P17485 INFO [Metrics] Recall(k=20): 0.041714 - Recall(k=50): 0.072681 - NDCG(k=20): 0.033677 - NDCG(k=50): 0.045201 - HitRate(k=20): 0.235662 - HitRate(k=50): 0.368111
2021-01-13 01:59:53,584 P17485 INFO Save best model: monitor(max): 0.041714
2021-01-13 01:59:53,631 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 01:59:53,768 P17485 INFO Train loss: 0.954223
2021-01-13 01:59:53,769 P17485 INFO ************ Epoch=9 end ************
2021-01-13 01:59:53,769 P17485 INFO Negative sampling num_negs=1500
2021-01-13 02:01:03,042 P17485 INFO Negative sampling done
2021-01-13 02:08:37,445 P17485 INFO --- Start evaluation ---
2021-01-13 02:08:39,441 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 02:10:02,412 P17485 INFO [Metrics] Recall(k=20): 0.042361 - Recall(k=50): 0.073390 - NDCG(k=20): 0.033914 - NDCG(k=50): 0.045424 - HitRate(k=20): 0.238359 - HitRate(k=50): 0.368130
2021-01-13 02:10:02,439 P17485 INFO Save best model: monitor(max): 0.042361
2021-01-13 02:10:02,492 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 02:10:02,573 P17485 INFO Train loss: 0.952868
2021-01-13 02:10:02,573 P17485 INFO ************ Epoch=10 end ************
2021-01-13 02:10:02,573 P17485 INFO Negative sampling num_negs=1500
2021-01-13 02:11:03,455 P17485 INFO Negative sampling done
2021-01-13 02:18:24,496 P17485 INFO --- Start evaluation ---
2021-01-13 02:18:26,941 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 02:19:48,211 P17485 INFO [Metrics] Recall(k=20): 0.041564 - Recall(k=50): 0.071632 - NDCG(k=20): 0.033578 - NDCG(k=50): 0.044743 - HitRate(k=20): 0.235681 - HitRate(k=50): 0.364882
2021-01-13 02:19:48,230 P17485 INFO Monitor(max) STOP: 0.041564 !
2021-01-13 02:19:48,231 P17485 INFO Reduce learning rate on plateau: 0.000100
2021-01-13 02:19:48,231 P17485 INFO Load best model:  MF_amazonbooks_x0_001_da43988f.model
2021-01-13 02:19:48,263 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 02:19:48,351 P17485 INFO Train loss: 0.951843
2021-01-13 02:19:48,351 P17485 INFO ************ Epoch=11 end ************
2021-01-13 02:19:48,352 P17485 INFO Negative sampling num_negs=1500
2021-01-13 02:20:57,818 P17485 INFO Negative sampling done
2021-01-13 02:28:14,028 P17485 INFO --- Start evaluation ---
2021-01-13 02:28:16,712 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 02:29:39,472 P17485 INFO [Metrics] Recall(k=20): 0.049302 - Recall(k=50): 0.086010 - NDCG(k=20): 0.039330 - NDCG(k=50): 0.052930 - HitRate(k=20): 0.270028 - HitRate(k=50): 0.412565
2021-01-13 02:29:39,497 P17485 INFO Save best model: monitor(max): 0.049302
2021-01-13 02:29:39,539 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 02:29:39,624 P17485 INFO Train loss: 0.874855
2021-01-13 02:29:39,624 P17485 INFO ************ Epoch=12 end ************
2021-01-13 02:29:39,624 P17485 INFO Negative sampling num_negs=1500
2021-01-13 02:30:40,440 P17485 INFO Negative sampling done
2021-01-13 02:38:00,595 P17485 INFO --- Start evaluation ---
2021-01-13 02:38:04,603 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 02:39:23,424 P17485 INFO [Metrics] Recall(k=20): 0.051973 - Recall(k=50): 0.091146 - NDCG(k=20): 0.041481 - NDCG(k=50): 0.056041 - HitRate(k=20): 0.279185 - HitRate(k=50): 0.429149
2021-01-13 02:39:23,440 P17485 INFO Save best model: monitor(max): 0.051973
2021-01-13 02:39:23,484 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 02:39:23,562 P17485 INFO Train loss: 0.838239
2021-01-13 02:39:23,562 P17485 INFO ************ Epoch=13 end ************
2021-01-13 02:39:23,563 P17485 INFO Negative sampling num_negs=1500
2021-01-13 02:40:27,566 P17485 INFO Negative sampling done
2021-01-13 02:47:30,515 P17485 INFO --- Start evaluation ---
2021-01-13 02:47:34,602 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 02:48:59,659 P17485 INFO [Metrics] Recall(k=20): 0.053931 - Recall(k=50): 0.093714 - NDCG(k=20): 0.042722 - NDCG(k=50): 0.057498 - HitRate(k=20): 0.285834 - HitRate(k=50): 0.434716
2021-01-13 02:48:59,695 P17485 INFO Save best model: monitor(max): 0.053931
2021-01-13 02:48:59,770 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 02:48:59,850 P17485 INFO Train loss: 0.819236
2021-01-13 02:48:59,851 P17485 INFO ************ Epoch=14 end ************
2021-01-13 02:48:59,851 P17485 INFO Negative sampling num_negs=1500
2021-01-13 02:50:10,659 P17485 INFO Negative sampling done
2021-01-13 02:57:40,334 P17485 INFO --- Start evaluation ---
2021-01-13 02:57:43,652 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 02:59:07,556 P17485 INFO [Metrics] Recall(k=20): 0.054581 - Recall(k=50): 0.094579 - NDCG(k=20): 0.043374 - NDCG(k=50): 0.058259 - HitRate(k=20): 0.289405 - HitRate(k=50): 0.438743
2021-01-13 02:59:07,587 P17485 INFO Save best model: monitor(max): 0.054581
2021-01-13 02:59:07,634 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 02:59:07,733 P17485 INFO Train loss: 0.807781
2021-01-13 02:59:07,734 P17485 INFO ************ Epoch=15 end ************
2021-01-13 02:59:07,734 P17485 INFO Negative sampling num_negs=1500
2021-01-13 03:00:17,562 P17485 INFO Negative sampling done
2021-01-13 03:08:22,617 P17485 INFO --- Start evaluation ---
2021-01-13 03:08:24,686 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 03:10:00,367 P17485 INFO [Metrics] Recall(k=20): 0.055005 - Recall(k=50): 0.094920 - NDCG(k=20): 0.043690 - NDCG(k=50): 0.058562 - HitRate(k=20): 0.291115 - HitRate(k=50): 0.439693
2021-01-13 03:10:00,455 P17485 INFO Save best model: monitor(max): 0.055005
2021-01-13 03:10:00,612 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 03:10:00,784 P17485 INFO Train loss: 0.800372
2021-01-13 03:10:00,784 P17485 INFO ************ Epoch=16 end ************
2021-01-13 03:10:00,785 P17485 INFO Negative sampling num_negs=1500
2021-01-13 03:11:02,477 P17485 INFO Negative sampling done
2021-01-13 03:18:41,828 P17485 INFO --- Start evaluation ---
2021-01-13 03:18:43,783 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 03:20:05,041 P17485 INFO [Metrics] Recall(k=20): 0.055085 - Recall(k=50): 0.095538 - NDCG(k=20): 0.043822 - NDCG(k=50): 0.058872 - HitRate(k=20): 0.290127 - HitRate(k=50): 0.440073
2021-01-13 03:20:05,057 P17485 INFO Save best model: monitor(max): 0.055085
2021-01-13 03:20:05,104 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 03:20:05,187 P17485 INFO Train loss: 0.795272
2021-01-13 03:20:05,187 P17485 INFO ************ Epoch=17 end ************
2021-01-13 03:20:05,188 P17485 INFO Negative sampling num_negs=1500
2021-01-13 03:21:06,164 P17485 INFO Negative sampling done
2021-01-13 03:27:36,621 P17485 INFO --- Start evaluation ---
2021-01-13 03:27:38,041 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 03:29:04,353 P17485 INFO [Metrics] Recall(k=20): 0.055186 - Recall(k=50): 0.095793 - NDCG(k=20): 0.043845 - NDCG(k=50): 0.058919 - HitRate(k=20): 0.290944 - HitRate(k=50): 0.440529
2021-01-13 03:29:04,374 P17485 INFO Save best model: monitor(max): 0.055186
2021-01-13 03:29:04,418 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 03:29:04,515 P17485 INFO Train loss: 0.791616
2021-01-13 03:29:04,515 P17485 INFO ************ Epoch=18 end ************
2021-01-13 03:29:04,516 P17485 INFO Negative sampling num_negs=1500
2021-01-13 03:30:06,470 P17485 INFO Negative sampling done
2021-01-13 03:37:04,211 P17485 INFO --- Start evaluation ---
2021-01-13 03:37:07,569 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 03:38:27,005 P17485 INFO [Metrics] Recall(k=20): 0.055330 - Recall(k=50): 0.095249 - NDCG(k=20): 0.043972 - NDCG(k=50): 0.058823 - HitRate(k=20): 0.291305 - HitRate(k=50): 0.440225
2021-01-13 03:38:27,026 P17485 INFO Save best model: monitor(max): 0.055330
2021-01-13 03:38:27,071 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 03:38:27,149 P17485 INFO Train loss: 0.788876
2021-01-13 03:38:27,150 P17485 INFO ************ Epoch=19 end ************
2021-01-13 03:38:27,150 P17485 INFO Negative sampling num_negs=1500
2021-01-13 03:39:29,149 P17485 INFO Negative sampling done
2021-01-13 03:46:45,842 P17485 INFO --- Start evaluation ---
2021-01-13 03:46:49,579 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 03:48:15,817 P17485 INFO [Metrics] Recall(k=20): 0.055442 - Recall(k=50): 0.095540 - NDCG(k=20): 0.044073 - NDCG(k=50): 0.058990 - HitRate(k=20): 0.291115 - HitRate(k=50): 0.438515
2021-01-13 03:48:15,853 P17485 INFO Save best model: monitor(max): 0.055442
2021-01-13 03:48:15,897 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 03:48:15,971 P17485 INFO Train loss: 0.786845
2021-01-13 03:48:15,972 P17485 INFO ************ Epoch=20 end ************
2021-01-13 03:48:15,972 P17485 INFO Negative sampling num_negs=1500
2021-01-13 03:49:17,500 P17485 INFO Negative sampling done
2021-01-13 03:56:30,399 P17485 INFO --- Start evaluation ---
2021-01-13 03:56:32,932 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 03:58:01,614 P17485 INFO [Metrics] Recall(k=20): 0.055467 - Recall(k=50): 0.095185 - NDCG(k=20): 0.044263 - NDCG(k=50): 0.059014 - HitRate(k=20): 0.291666 - HitRate(k=50): 0.439598
2021-01-13 03:58:01,631 P17485 INFO Save best model: monitor(max): 0.055467
2021-01-13 03:58:01,673 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 03:58:01,750 P17485 INFO Train loss: 0.785200
2021-01-13 03:58:01,750 P17485 INFO ************ Epoch=21 end ************
2021-01-13 03:58:01,751 P17485 INFO Negative sampling num_negs=1500
2021-01-13 03:59:03,555 P17485 INFO Negative sampling done
2021-01-13 04:07:08,918 P17485 INFO --- Start evaluation ---
2021-01-13 04:07:12,343 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 04:08:40,511 P17485 INFO [Metrics] Recall(k=20): 0.055456 - Recall(k=50): 0.095566 - NDCG(k=20): 0.044047 - NDCG(k=50): 0.059012 - HitRate(k=20): 0.290127 - HitRate(k=50): 0.440016
2021-01-13 04:08:40,531 P17485 INFO Monitor(max) STOP: 0.055456 !
2021-01-13 04:08:40,532 P17485 INFO Reduce learning rate on plateau: 0.000010
2021-01-13 04:08:40,532 P17485 INFO Load best model:  MF_amazonbooks_x0_001_da43988f.model
2021-01-13 04:08:40,559 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 04:08:40,645 P17485 INFO Train loss: 0.783927
2021-01-13 04:08:40,645 P17485 INFO ************ Epoch=22 end ************
2021-01-13 04:08:40,646 P17485 INFO Negative sampling num_negs=1500
2021-01-13 04:09:45,019 P17485 INFO Negative sampling done
2021-01-13 04:17:45,349 P17485 INFO --- Start evaluation ---
2021-01-13 04:17:48,655 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 04:19:15,862 P17485 INFO [Metrics] Recall(k=20): 0.055638 - Recall(k=50): 0.095577 - NDCG(k=20): 0.044439 - NDCG(k=50): 0.059253 - HitRate(k=20): 0.292996 - HitRate(k=50): 0.441954
2021-01-13 04:19:15,883 P17485 INFO Save best model: monitor(max): 0.055638
2021-01-13 04:19:15,928 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 04:19:16,078 P17485 INFO Train loss: 0.764412
2021-01-13 04:19:16,078 P17485 INFO ************ Epoch=23 end ************
2021-01-13 04:19:16,079 P17485 INFO Negative sampling num_negs=1500
2021-01-13 04:20:30,066 P17485 INFO Negative sampling done
2021-01-13 04:28:07,654 P17485 INFO --- Start evaluation ---
2021-01-13 04:28:11,009 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 04:29:27,860 P17485 INFO [Metrics] Recall(k=20): 0.055756 - Recall(k=50): 0.095788 - NDCG(k=20): 0.044549 - NDCG(k=50): 0.059396 - HitRate(k=20): 0.293129 - HitRate(k=50): 0.441897
2021-01-13 04:29:27,892 P17485 INFO Save best model: monitor(max): 0.055756
2021-01-13 04:29:27,939 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 04:29:28,032 P17485 INFO Train loss: 0.762676
2021-01-13 04:29:28,032 P17485 INFO ************ Epoch=24 end ************
2021-01-13 04:29:28,033 P17485 INFO Negative sampling num_negs=1500
2021-01-13 04:30:50,370 P17485 INFO Negative sampling done
2021-01-13 04:39:08,192 P17485 INFO --- Start evaluation ---
2021-01-13 04:39:10,207 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 04:40:40,317 P17485 INFO [Metrics] Recall(k=20): 0.055890 - Recall(k=50): 0.095898 - NDCG(k=20): 0.044613 - NDCG(k=50): 0.059463 - HitRate(k=20): 0.293737 - HitRate(k=50): 0.441821
2021-01-13 04:40:40,353 P17485 INFO Save best model: monitor(max): 0.055890
2021-01-13 04:40:40,406 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 04:40:40,516 P17485 INFO Train loss: 0.761529
2021-01-13 04:40:40,516 P17485 INFO ************ Epoch=25 end ************
2021-01-13 04:40:40,517 P17485 INFO Negative sampling num_negs=1500
2021-01-13 04:41:49,314 P17485 INFO Negative sampling done
2021-01-13 04:50:08,160 P17485 INFO --- Start evaluation ---
2021-01-13 04:50:10,279 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 04:51:38,879 P17485 INFO [Metrics] Recall(k=20): 0.055891 - Recall(k=50): 0.095966 - NDCG(k=20): 0.044663 - NDCG(k=50): 0.059520 - HitRate(k=20): 0.293300 - HitRate(k=50): 0.441517
2021-01-13 04:51:38,898 P17485 INFO Monitor(max) STOP: 0.055891 !
2021-01-13 04:51:38,899 P17485 INFO Reduce learning rate on plateau: 0.000001
2021-01-13 04:51:38,899 P17485 INFO Load best model:  MF_amazonbooks_x0_001_da43988f.model
2021-01-13 04:51:38,924 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 04:51:39,003 P17485 INFO Train loss: 0.760641
2021-01-13 04:51:39,003 P17485 INFO ************ Epoch=26 end ************
2021-01-13 04:51:39,004 P17485 INFO Negative sampling num_negs=1500
2021-01-13 04:52:40,082 P17485 INFO Negative sampling done
2021-01-13 05:00:55,228 P17485 INFO --- Start evaluation ---
2021-01-13 05:00:57,958 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 05:02:30,613 P17485 INFO [Metrics] Recall(k=20): 0.055890 - Recall(k=50): 0.095965 - NDCG(k=20): 0.044684 - NDCG(k=50): 0.059530 - HitRate(k=20): 0.294079 - HitRate(k=50): 0.442296
2021-01-13 05:02:30,636 P17485 INFO Monitor(max) STOP: 0.055890 !
2021-01-13 05:02:30,636 P17485 INFO Reduce learning rate on plateau: 0.000001
2021-01-13 05:02:30,636 P17485 INFO Load best model:  MF_amazonbooks_x0_001_da43988f.model
2021-01-13 05:02:30,664 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 05:02:30,763 P17485 INFO Train loss: 0.758699
2021-01-13 05:02:30,763 P17485 INFO ************ Epoch=27 end ************
2021-01-13 05:02:30,764 P17485 INFO Negative sampling num_negs=1500
2021-01-13 05:03:34,827 P17485 INFO Negative sampling done
2021-01-13 05:11:48,842 P17485 INFO --- Start evaluation ---
2021-01-13 05:11:51,618 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 05:13:20,835 P17485 INFO [Metrics] Recall(k=20): 0.055902 - Recall(k=50): 0.095990 - NDCG(k=20): 0.044664 - NDCG(k=50): 0.059524 - HitRate(k=20): 0.293889 - HitRate(k=50): 0.442239
2021-01-13 05:13:20,853 P17485 INFO Save best model: monitor(max): 0.055902
2021-01-13 05:13:20,897 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 05:13:20,975 P17485 INFO Train loss: 0.758655
2021-01-13 05:13:20,975 P17485 INFO ************ Epoch=28 end ************
2021-01-13 05:13:20,976 P17485 INFO Negative sampling num_negs=1500
2021-01-13 05:14:24,441 P17485 INFO Negative sampling done
2021-01-13 05:22:43,562 P17485 INFO --- Start evaluation ---
2021-01-13 05:22:46,934 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 05:24:17,969 P17485 INFO [Metrics] Recall(k=20): 0.055916 - Recall(k=50): 0.095969 - NDCG(k=20): 0.044692 - NDCG(k=50): 0.059530 - HitRate(k=20): 0.294136 - HitRate(k=50): 0.442277
2021-01-13 05:24:17,989 P17485 INFO Save best model: monitor(max): 0.055916
2021-01-13 05:24:18,034 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 05:24:18,121 P17485 INFO Train loss: 0.758561
2021-01-13 05:24:18,121 P17485 INFO ************ Epoch=29 end ************
2021-01-13 05:24:18,122 P17485 INFO Negative sampling num_negs=1500
2021-01-13 05:25:31,230 P17485 INFO Negative sampling done
2021-01-13 05:32:25,490 P17485 INFO --- Start evaluation ---
2021-01-13 05:32:26,928 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 05:33:54,638 P17485 INFO [Metrics] Recall(k=20): 0.055883 - Recall(k=50): 0.096004 - NDCG(k=20): 0.044678 - NDCG(k=50): 0.059545 - HitRate(k=20): 0.293927 - HitRate(k=50): 0.442543
2021-01-13 05:33:54,659 P17485 INFO Monitor(max) STOP: 0.055883 !
2021-01-13 05:33:54,659 P17485 INFO Reduce learning rate on plateau: 0.000001
2021-01-13 05:33:54,660 P17485 INFO Load best model:  MF_amazonbooks_x0_001_da43988f.model
2021-01-13 05:33:54,686 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 05:33:54,769 P17485 INFO Train loss: 0.758471
2021-01-13 05:33:54,769 P17485 INFO ************ Epoch=30 end ************
2021-01-13 05:33:54,770 P17485 INFO Negative sampling num_negs=1500
2021-01-13 05:34:57,710 P17485 INFO Negative sampling done
2021-01-13 05:41:43,131 P17485 INFO --- Start evaluation ---
2021-01-13 05:41:45,142 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 05:43:03,822 P17485 INFO [Metrics] Recall(k=20): 0.055937 - Recall(k=50): 0.096077 - NDCG(k=20): 0.044708 - NDCG(k=50): 0.059571 - HitRate(k=20): 0.294288 - HitRate(k=50): 0.442676
2021-01-13 05:43:03,843 P17485 INFO Save best model: monitor(max): 0.055937
2021-01-13 05:43:03,891 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 05:43:03,975 P17485 INFO Train loss: 0.758462
2021-01-13 05:43:03,975 P17485 INFO ************ Epoch=31 end ************
2021-01-13 05:43:03,976 P17485 INFO Negative sampling num_negs=1500
2021-01-13 05:44:08,144 P17485 INFO Negative sampling done
2021-01-13 05:50:13,425 P17485 INFO --- Start evaluation ---
2021-01-13 05:50:15,799 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 05:51:36,704 P17485 INFO [Metrics] Recall(k=20): 0.055890 - Recall(k=50): 0.096060 - NDCG(k=20): 0.044705 - NDCG(k=50): 0.059564 - HitRate(k=20): 0.294363 - HitRate(k=50): 0.442600
2021-01-13 05:51:36,725 P17485 INFO Monitor(max) STOP: 0.055890 !
2021-01-13 05:51:36,725 P17485 INFO Reduce learning rate on plateau: 0.000001
2021-01-13 05:51:36,725 P17485 INFO Load best model:  MF_amazonbooks_x0_001_da43988f.model
2021-01-13 05:51:36,754 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 05:51:36,832 P17485 INFO Train loss: 0.758380
2021-01-13 05:51:36,832 P17485 INFO ************ Epoch=32 end ************
2021-01-13 05:51:36,833 P17485 INFO Negative sampling num_negs=1500
2021-01-13 05:52:42,121 P17485 INFO Negative sampling done
2021-01-13 05:58:04,717 P17485 INFO --- Start evaluation ---
2021-01-13 05:58:06,584 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 05:59:34,549 P17485 INFO [Metrics] Recall(k=20): 0.055922 - Recall(k=50): 0.096012 - NDCG(k=20): 0.044715 - NDCG(k=50): 0.059557 - HitRate(k=20): 0.294231 - HitRate(k=50): 0.442543
2021-01-13 05:59:34,571 P17485 INFO Monitor(max) STOP: 0.055922 !
2021-01-13 05:59:34,571 P17485 INFO Reduce learning rate on plateau: 0.000001
2021-01-13 05:59:34,571 P17485 INFO Load best model:  MF_amazonbooks_x0_001_da43988f.model
2021-01-13 05:59:34,599 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 05:59:34,681 P17485 INFO Train loss: 0.758355
2021-01-13 05:59:34,682 P17485 INFO ************ Epoch=33 end ************
2021-01-13 05:59:34,682 P17485 INFO Negative sampling num_negs=1500
2021-01-13 06:00:38,718 P17485 INFO Negative sampling done
2021-01-13 06:06:05,812 P17485 INFO --- Start evaluation ---
2021-01-13 06:06:07,206 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 06:07:36,548 P17485 INFO [Metrics] Recall(k=20): 0.055917 - Recall(k=50): 0.096050 - NDCG(k=20): 0.044689 - NDCG(k=50): 0.059553 - HitRate(k=20): 0.294117 - HitRate(k=50): 0.442581
2021-01-13 06:07:36,570 P17485 INFO Monitor(max) STOP: 0.055917 !
2021-01-13 06:07:36,570 P17485 INFO Reduce learning rate on plateau: 0.000001
2021-01-13 06:07:36,570 P17485 INFO Load best model:  MF_amazonbooks_x0_001_da43988f.model
2021-01-13 06:07:36,598 P17485 INFO Early stopping at epoch=34
2021-01-13 06:07:36,599 P17485 INFO --- 4650/4650 batches finished ---
2021-01-13 06:07:36,681 P17485 INFO Train loss: 0.758363
2021-01-13 06:07:36,681 P17485 INFO Training finished.
2021-01-13 06:07:36,681 P17485 INFO Load best model:  MF_amazonbooks_x0_001_da43988f.model
2021-01-13 06:07:36,732 P17485 INFO ****** Train/validation evaluation ******
2021-01-13 06:07:36,732 P17485 INFO --- Start evaluation ---
2021-01-13 06:07:38,158 P17485 INFO Evaluating metrics for 52639 users...
2021-01-13 06:08:59,104 P17485 INFO [Metrics] Recall(k=20): 0.055937 - Recall(k=50): 0.096077 - NDCG(k=20): 0.044708 - NDCG(k=50): 0.059571 - HitRate(k=20): 0.294288 - HitRate(k=50): 0.442676
```