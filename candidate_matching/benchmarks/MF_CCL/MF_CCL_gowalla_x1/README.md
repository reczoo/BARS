## MF-CCL_gowalla_x0 

A notebook to benchmark MF-CCL on gowalla dataset.

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
Gowalla follows the data split and preprocessing steps in NGCF and LightGCN. We directly transform the formats of the data from their [repo](https://github.com/kuandeng/LightGCN/tree/master/Data).


### Code
The implementation code is already available at https://github.com/xue-pai/DEEM. 

Please refer to the [configuration file](https://github.com/xue-pai/DEEM/blob/master/benchmarks/Yelp18/SimpleX_yelp18_x0/SimpleX_yelp18_x0_tuner_config.yaml).
### Results
```
2021-01-12 10:58:57,677 P43946 INFO [Metrics] Recall(k=20): 0.183730 - Recall(k=50): 0.287096 - NDCG(k=20): 0.149257 - NDCG(k=50): 0.181573 - HitRate(k=20): 0.584031 - HitRate(k=50): 0.724262
```


### Logs
```
2021-01-12 08:25:56,594 P43946 INFO Set up feature encoder...
2021-01-12 08:25:56,594 P43946 INFO Load feature_map from json: ../data/Gowalla/gowalla_x0_4c90e422/feature_map.json
2021-01-12 08:25:59,379 P43946 INFO Total number of parameters: 4533824.
2021-01-12 08:25:59,380 P43946 INFO Loading data...
2021-01-12 08:25:59,384 P43946 INFO Loading data from h5: ../data/Gowalla/gowalla_x0_4c90e422/train.h5
2021-01-12 08:25:59,402 P43946 INFO Loading data from h5: ../data/Gowalla/gowalla_x0_4c90e422/item_corpus.h5
2021-01-12 08:25:59,781 P43946 INFO Loading data from h5: ../data/Gowalla/gowalla_x0_4c90e422/valid.h5
2021-01-12 08:25:59,878 P43946 INFO Loading data from h5: ../data/Gowalla/gowalla_x0_4c90e422/item_corpus.h5
2021-01-12 08:25:59,879 P43946 INFO Train samples: total/810128, blocks/1
2021-01-12 08:25:59,879 P43946 INFO Validation samples: total/29858, blocks/1
2021-01-12 08:25:59,879 P43946 INFO Loading train data done.
2021-01-12 08:25:59,880 P43946 INFO **** Start training: 3165 batches/epoch ****
2021-01-12 08:25:59,881 P43946 INFO Negative sampling num_negs=800
2021-01-12 08:26:12,686 P43946 INFO Negative sampling done
2021-01-12 08:27:03,980 P43946 INFO --- Start evaluation ---
2021-01-12 08:27:04,825 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 08:27:27,187 P43946 INFO [Metrics] Recall(k=20): 0.089541 - Recall(k=50): 0.154621 - NDCG(k=20): 0.067831 - NDCG(k=50): 0.088450 - HitRate(k=20): 0.371827 - HitRate(k=50): 0.518755
2021-01-12 08:27:27,196 P43946 INFO Save best model: monitor(max): 0.089541
2021-01-12 08:27:27,219 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 08:27:27,247 P43946 INFO Train loss: 0.582343
2021-01-12 08:27:27,248 P43946 INFO ************ Epoch=1 end ************
2021-01-12 08:27:27,248 P43946 INFO Negative sampling num_negs=800
2021-01-12 08:27:40,088 P43946 INFO Negative sampling done
2021-01-12 08:28:14,186 P43946 INFO --- Start evaluation ---
2021-01-12 08:28:14,728 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 08:28:36,821 P43946 INFO [Metrics] Recall(k=20): 0.110530 - Recall(k=50): 0.187603 - NDCG(k=20): 0.085094 - NDCG(k=50): 0.109338 - HitRate(k=20): 0.430069 - HitRate(k=50): 0.581017
2021-01-12 08:28:36,835 P43946 INFO Save best model: monitor(max): 0.110530
2021-01-12 08:28:36,859 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 08:28:36,891 P43946 INFO Train loss: 0.190581
2021-01-12 08:28:36,891 P43946 INFO ************ Epoch=2 end ************
2021-01-12 08:28:36,892 P43946 INFO Negative sampling num_negs=800
2021-01-12 08:28:49,863 P43946 INFO Negative sampling done
2021-01-12 08:29:43,115 P43946 INFO --- Start evaluation ---
2021-01-12 08:29:43,994 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 08:30:06,547 P43946 INFO [Metrics] Recall(k=20): 0.125611 - Recall(k=50): 0.209587 - NDCG(k=20): 0.096864 - NDCG(k=50): 0.123209 - HitRate(k=20): 0.466977 - HitRate(k=50): 0.620269
2021-01-12 08:30:06,566 P43946 INFO Save best model: monitor(max): 0.125611
2021-01-12 08:30:06,600 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 08:30:06,629 P43946 INFO Train loss: 0.173005
2021-01-12 08:30:06,629 P43946 INFO ************ Epoch=3 end ************
2021-01-12 08:30:06,630 P43946 INFO Negative sampling num_negs=800
2021-01-12 08:30:19,787 P43946 INFO Negative sampling done
2021-01-12 08:30:57,876 P43946 INFO --- Start evaluation ---
2021-01-12 08:30:58,772 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 08:31:20,965 P43946 INFO [Metrics] Recall(k=20): 0.135432 - Recall(k=50): 0.223517 - NDCG(k=20): 0.105244 - NDCG(k=50): 0.132971 - HitRate(k=20): 0.490354 - HitRate(k=50): 0.640867
2021-01-12 08:31:20,986 P43946 INFO Save best model: monitor(max): 0.135432
2021-01-12 08:31:21,007 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 08:31:21,040 P43946 INFO Train loss: 0.162646
2021-01-12 08:31:21,040 P43946 INFO ************ Epoch=4 end ************
2021-01-12 08:31:21,041 P43946 INFO Negative sampling num_negs=800
2021-01-12 08:31:34,378 P43946 INFO Negative sampling done
2021-01-12 08:34:25,087 P43946 INFO --- Start evaluation ---
2021-01-12 08:34:28,417 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 08:34:51,282 P43946 INFO [Metrics] Recall(k=20): 0.144192 - Recall(k=50): 0.234544 - NDCG(k=20): 0.112669 - NDCG(k=50): 0.140953 - HitRate(k=20): 0.507904 - HitRate(k=50): 0.658316
2021-01-12 08:34:51,293 P43946 INFO Save best model: monitor(max): 0.144192
2021-01-12 08:34:51,315 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 08:34:51,356 P43946 INFO Train loss: 0.155289
2021-01-12 08:34:51,357 P43946 INFO ************ Epoch=5 end ************
2021-01-12 08:34:51,357 P43946 INFO Negative sampling num_negs=800
2021-01-12 08:35:04,384 P43946 INFO Negative sampling done
2021-01-12 08:37:07,856 P43946 INFO --- Start evaluation ---
2021-01-12 08:37:11,236 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 08:37:36,217 P43946 INFO [Metrics] Recall(k=20): 0.149432 - Recall(k=50): 0.242843 - NDCG(k=20): 0.117204 - NDCG(k=50): 0.146417 - HitRate(k=20): 0.515875 - HitRate(k=50): 0.668263
2021-01-12 08:37:36,231 P43946 INFO Save best model: monitor(max): 0.149432
2021-01-12 08:37:36,257 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 08:37:36,307 P43946 INFO Train loss: 0.149962
2021-01-12 08:37:36,308 P43946 INFO ************ Epoch=6 end ************
2021-01-12 08:37:36,309 P43946 INFO Negative sampling num_negs=800
2021-01-12 08:37:49,855 P43946 INFO Negative sampling done
2021-01-12 08:40:48,764 P43946 INFO --- Start evaluation ---
2021-01-12 08:40:50,649 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 08:41:14,733 P43946 INFO [Metrics] Recall(k=20): 0.153185 - Recall(k=50): 0.248474 - NDCG(k=20): 0.119969 - NDCG(k=50): 0.149804 - HitRate(k=20): 0.526124 - HitRate(k=50): 0.673856
2021-01-12 08:41:14,752 P43946 INFO Save best model: monitor(max): 0.153185
2021-01-12 08:41:14,779 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 08:41:14,813 P43946 INFO Train loss: 0.145856
2021-01-12 08:41:14,813 P43946 INFO ************ Epoch=7 end ************
2021-01-12 08:41:14,814 P43946 INFO Negative sampling num_negs=800
2021-01-12 08:41:28,121 P43946 INFO Negative sampling done
2021-01-12 08:45:06,363 P43946 INFO --- Start evaluation ---
2021-01-12 08:45:08,891 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 08:45:41,959 P43946 INFO [Metrics] Recall(k=20): 0.156625 - Recall(k=50): 0.253400 - NDCG(k=20): 0.123102 - NDCG(k=50): 0.153462 - HitRate(k=20): 0.530980 - HitRate(k=50): 0.682263
2021-01-12 08:45:41,979 P43946 INFO Save best model: monitor(max): 0.156625
2021-01-12 08:45:42,010 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 08:45:42,061 P43946 INFO Train loss: 0.142962
2021-01-12 08:45:42,062 P43946 INFO ************ Epoch=8 end ************
2021-01-12 08:45:42,062 P43946 INFO Negative sampling num_negs=800
2021-01-12 08:45:56,168 P43946 INFO Negative sampling done
2021-01-12 08:49:19,204 P43946 INFO --- Start evaluation ---
2021-01-12 08:49:22,640 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 08:49:46,011 P43946 INFO [Metrics] Recall(k=20): 0.160812 - Recall(k=50): 0.257763 - NDCG(k=20): 0.126814 - NDCG(k=50): 0.157181 - HitRate(k=20): 0.539889 - HitRate(k=50): 0.686081
2021-01-12 08:49:46,021 P43946 INFO Save best model: monitor(max): 0.160812
2021-01-12 08:49:46,045 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 08:49:46,087 P43946 INFO Train loss: 0.140373
2021-01-12 08:49:46,087 P43946 INFO ************ Epoch=9 end ************
2021-01-12 08:49:46,087 P43946 INFO Negative sampling num_negs=800
2021-01-12 08:49:59,986 P43946 INFO Negative sampling done
2021-01-12 08:51:16,143 P43946 INFO --- Start evaluation ---
2021-01-12 08:51:18,569 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 08:51:41,959 P43946 INFO [Metrics] Recall(k=20): 0.162652 - Recall(k=50): 0.260933 - NDCG(k=20): 0.128432 - NDCG(k=50): 0.159167 - HitRate(k=20): 0.544812 - HitRate(k=50): 0.692210
2021-01-12 08:51:41,970 P43946 INFO Save best model: monitor(max): 0.162652
2021-01-12 08:51:41,992 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 08:51:42,028 P43946 INFO Train loss: 0.138361
2021-01-12 08:51:42,028 P43946 INFO ************ Epoch=10 end ************
2021-01-12 08:51:42,029 P43946 INFO Negative sampling num_negs=800
2021-01-12 08:51:55,139 P43946 INFO Negative sampling done
2021-01-12 08:55:14,027 P43946 INFO --- Start evaluation ---
2021-01-12 08:55:15,947 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 08:55:45,577 P43946 INFO [Metrics] Recall(k=20): 0.163636 - Recall(k=50): 0.263229 - NDCG(k=20): 0.129421 - NDCG(k=50): 0.160610 - HitRate(k=20): 0.545348 - HitRate(k=50): 0.691942
2021-01-12 08:55:45,589 P43946 INFO Save best model: monitor(max): 0.163636
2021-01-12 08:55:45,612 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 08:55:45,660 P43946 INFO Train loss: 0.136636
2021-01-12 08:55:45,660 P43946 INFO ************ Epoch=11 end ************
2021-01-12 08:55:45,661 P43946 INFO Negative sampling num_negs=800
2021-01-12 08:55:58,963 P43946 INFO Negative sampling done
2021-01-12 08:59:27,368 P43946 INFO --- Start evaluation ---
2021-01-12 08:59:30,741 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 08:59:58,410 P43946 INFO [Metrics] Recall(k=20): 0.166668 - Recall(k=50): 0.266040 - NDCG(k=20): 0.131916 - NDCG(k=50): 0.162979 - HitRate(k=20): 0.552214 - HitRate(k=50): 0.695659
2021-01-12 08:59:58,426 P43946 INFO Save best model: monitor(max): 0.166668
2021-01-12 08:59:58,454 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 08:59:58,504 P43946 INFO Train loss: 0.135245
2021-01-12 08:59:58,504 P43946 INFO ************ Epoch=12 end ************
2021-01-12 08:59:58,505 P43946 INFO Negative sampling num_negs=800
2021-01-12 09:00:13,941 P43946 INFO Negative sampling done
2021-01-12 09:03:29,824 P43946 INFO --- Start evaluation ---
2021-01-12 09:03:32,846 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 09:03:56,306 P43946 INFO [Metrics] Recall(k=20): 0.168954 - Recall(k=50): 0.269029 - NDCG(k=20): 0.133430 - NDCG(k=50): 0.164776 - HitRate(k=20): 0.556501 - HitRate(k=50): 0.702190
2021-01-12 09:03:56,317 P43946 INFO Save best model: monitor(max): 0.168954
2021-01-12 09:03:56,342 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 09:03:56,384 P43946 INFO Train loss: 0.133867
2021-01-12 09:03:56,385 P43946 INFO ************ Epoch=13 end ************
2021-01-12 09:03:56,385 P43946 INFO Negative sampling num_negs=800
2021-01-12 09:04:10,285 P43946 INFO Negative sampling done
2021-01-12 09:07:02,580 P43946 INFO --- Start evaluation ---
2021-01-12 09:07:06,003 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 09:07:31,942 P43946 INFO [Metrics] Recall(k=20): 0.169442 - Recall(k=50): 0.270252 - NDCG(k=20): 0.134538 - NDCG(k=50): 0.166030 - HitRate(k=20): 0.557506 - HitRate(k=50): 0.702626
2021-01-12 09:07:31,954 P43946 INFO Save best model: monitor(max): 0.169442
2021-01-12 09:07:31,976 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 09:07:32,012 P43946 INFO Train loss: 0.132787
2021-01-12 09:07:32,013 P43946 INFO ************ Epoch=14 end ************
2021-01-12 09:07:32,013 P43946 INFO Negative sampling num_negs=800
2021-01-12 09:07:45,067 P43946 INFO Negative sampling done
2021-01-12 09:09:50,565 P43946 INFO --- Start evaluation ---
2021-01-12 09:09:51,802 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 09:10:15,383 P43946 INFO [Metrics] Recall(k=20): 0.170294 - Recall(k=50): 0.271150 - NDCG(k=20): 0.136190 - NDCG(k=50): 0.167694 - HitRate(k=20): 0.559147 - HitRate(k=50): 0.701253
2021-01-12 09:10:15,395 P43946 INFO Save best model: monitor(max): 0.170294
2021-01-12 09:10:15,420 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 09:10:15,466 P43946 INFO Train loss: 0.131839
2021-01-12 09:10:15,467 P43946 INFO ************ Epoch=15 end ************
2021-01-12 09:10:15,467 P43946 INFO Negative sampling num_negs=800
2021-01-12 09:10:28,524 P43946 INFO Negative sampling done
2021-01-12 09:13:56,131 P43946 INFO --- Start evaluation ---
2021-01-12 09:13:58,996 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 09:14:24,865 P43946 INFO [Metrics] Recall(k=20): 0.170963 - Recall(k=50): 0.271302 - NDCG(k=20): 0.136515 - NDCG(k=50): 0.167899 - HitRate(k=20): 0.560553 - HitRate(k=50): 0.703396
2021-01-12 09:14:24,885 P43946 INFO Save best model: monitor(max): 0.170963
2021-01-12 09:14:24,914 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 09:14:24,971 P43946 INFO Train loss: 0.130963
2021-01-12 09:14:24,971 P43946 INFO ************ Epoch=16 end ************
2021-01-12 09:14:24,972 P43946 INFO Negative sampling num_negs=800
2021-01-12 09:14:38,540 P43946 INFO Negative sampling done
2021-01-12 09:18:09,782 P43946 INFO --- Start evaluation ---
2021-01-12 09:18:12,374 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 09:18:39,856 P43946 INFO [Metrics] Recall(k=20): 0.172782 - Recall(k=50): 0.274031 - NDCG(k=20): 0.137360 - NDCG(k=50): 0.168944 - HitRate(k=20): 0.562630 - HitRate(k=50): 0.705807
2021-01-12 09:18:39,870 P43946 INFO Save best model: monitor(max): 0.172782
2021-01-12 09:18:39,902 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 09:18:39,955 P43946 INFO Train loss: 0.130184
2021-01-12 09:18:39,955 P43946 INFO ************ Epoch=17 end ************
2021-01-12 09:18:39,956 P43946 INFO Negative sampling num_negs=800
2021-01-12 09:18:55,322 P43946 INFO Negative sampling done
2021-01-12 09:22:22,399 P43946 INFO --- Start evaluation ---
2021-01-12 09:22:24,553 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 09:22:49,214 P43946 INFO [Metrics] Recall(k=20): 0.173568 - Recall(k=50): 0.273973 - NDCG(k=20): 0.137878 - NDCG(k=50): 0.169313 - HitRate(k=20): 0.564539 - HitRate(k=50): 0.705774
2021-01-12 09:22:49,229 P43946 INFO Save best model: monitor(max): 0.173568
2021-01-12 09:22:49,252 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 09:22:49,292 P43946 INFO Train loss: 0.129333
2021-01-12 09:22:49,292 P43946 INFO ************ Epoch=18 end ************
2021-01-12 09:22:49,293 P43946 INFO Negative sampling num_negs=800
2021-01-12 09:23:06,408 P43946 INFO Negative sampling done
2021-01-12 09:26:21,961 P43946 INFO --- Start evaluation ---
2021-01-12 09:26:25,347 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 09:26:49,357 P43946 INFO [Metrics] Recall(k=20): 0.173364 - Recall(k=50): 0.274960 - NDCG(k=20): 0.138614 - NDCG(k=50): 0.170368 - HitRate(k=20): 0.564907 - HitRate(k=50): 0.707583
2021-01-12 09:26:49,370 P43946 INFO Monitor(max) STOP: 0.173364 !
2021-01-12 09:26:49,370 P43946 INFO Reduce learning rate on plateau: 0.000010
2021-01-12 09:26:49,370 P43946 INFO Load best model: /home/zhujieming/zhujieming/DEEM_20201231_WWW/benchmarks/Gowalla/MF_CCL_gowalla_x0/gowalla_x0_4c90e422/MF_gowalla_x0_001_e5f1ed4e.model
2021-01-12 09:26:49,386 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 09:26:49,424 P43946 INFO Train loss: 0.128749
2021-01-12 09:26:49,424 P43946 INFO ************ Epoch=19 end ************
2021-01-12 09:26:49,425 P43946 INFO Negative sampling num_negs=800
2021-01-12 09:27:03,399 P43946 INFO Negative sampling done
2021-01-12 09:29:51,048 P43946 INFO --- Start evaluation ---
2021-01-12 09:29:54,405 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 09:30:18,432 P43946 INFO [Metrics] Recall(k=20): 0.177624 - Recall(k=50): 0.279106 - NDCG(k=20): 0.142440 - NDCG(k=50): 0.174169 - HitRate(k=20): 0.571673 - HitRate(k=50): 0.713176
2021-01-12 09:30:18,441 P43946 INFO Save best model: monitor(max): 0.177624
2021-01-12 09:30:18,463 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 09:30:18,502 P43946 INFO Train loss: 0.120645
2021-01-12 09:30:18,502 P43946 INFO ************ Epoch=20 end ************
2021-01-12 09:30:18,502 P43946 INFO Negative sampling num_negs=800
2021-01-12 09:30:31,948 P43946 INFO Negative sampling done
2021-01-12 09:33:19,215 P43946 INFO --- Start evaluation ---
2021-01-12 09:33:22,624 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 09:33:46,293 P43946 INFO [Metrics] Recall(k=20): 0.179419 - Recall(k=50): 0.281455 - NDCG(k=20): 0.144660 - NDCG(k=50): 0.176560 - HitRate(k=20): 0.575759 - HitRate(k=50): 0.715822
2021-01-12 09:33:46,303 P43946 INFO Save best model: monitor(max): 0.179419
2021-01-12 09:33:46,324 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 09:33:46,361 P43946 INFO Train loss: 0.119231
2021-01-12 09:33:46,362 P43946 INFO ************ Epoch=21 end ************
2021-01-12 09:33:46,363 P43946 INFO Negative sampling num_negs=800
2021-01-12 09:34:00,519 P43946 INFO Negative sampling done
2021-01-12 09:37:01,437 P43946 INFO --- Start evaluation ---
2021-01-12 09:37:03,086 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 09:37:28,069 P43946 INFO [Metrics] Recall(k=20): 0.180398 - Recall(k=50): 0.283297 - NDCG(k=20): 0.145736 - NDCG(k=50): 0.177855 - HitRate(k=20): 0.576897 - HitRate(k=50): 0.718769
2021-01-12 09:37:28,081 P43946 INFO Save best model: monitor(max): 0.180398
2021-01-12 09:37:28,109 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 09:37:28,163 P43946 INFO Train loss: 0.118393
2021-01-12 09:37:28,163 P43946 INFO ************ Epoch=22 end ************
2021-01-12 09:37:28,164 P43946 INFO Negative sampling num_negs=800
2021-01-12 09:37:41,306 P43946 INFO Negative sampling done
2021-01-12 09:41:12,265 P43946 INFO --- Start evaluation ---
2021-01-12 09:41:14,811 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 09:41:43,765 P43946 INFO [Metrics] Recall(k=20): 0.181093 - Recall(k=50): 0.284094 - NDCG(k=20): 0.146569 - NDCG(k=50): 0.178708 - HitRate(k=20): 0.578739 - HitRate(k=50): 0.718869
2021-01-12 09:41:43,780 P43946 INFO Save best model: monitor(max): 0.181093
2021-01-12 09:41:43,807 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 09:41:43,848 P43946 INFO Train loss: 0.117842
2021-01-12 09:41:43,849 P43946 INFO ************ Epoch=23 end ************
2021-01-12 09:41:43,849 P43946 INFO Negative sampling num_negs=800
2021-01-12 09:41:57,631 P43946 INFO Negative sampling done
2021-01-12 09:45:12,481 P43946 INFO --- Start evaluation ---
2021-01-12 09:45:16,254 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 09:45:39,452 P43946 INFO [Metrics] Recall(k=20): 0.181370 - Recall(k=50): 0.284818 - NDCG(k=20): 0.146927 - NDCG(k=50): 0.179216 - HitRate(k=20): 0.579979 - HitRate(k=50): 0.720979
2021-01-12 09:45:39,462 P43946 INFO Save best model: monitor(max): 0.181370
2021-01-12 09:45:39,486 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 09:45:39,529 P43946 INFO Train loss: 0.117548
2021-01-12 09:45:39,529 P43946 INFO ************ Epoch=24 end ************
2021-01-12 09:45:39,530 P43946 INFO Negative sampling num_negs=800
2021-01-12 09:45:52,772 P43946 INFO Negative sampling done
2021-01-12 09:48:32,825 P43946 INFO --- Start evaluation ---
2021-01-12 09:48:36,209 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 09:48:59,431 P43946 INFO [Metrics] Recall(k=20): 0.181797 - Recall(k=50): 0.285298 - NDCG(k=20): 0.147478 - NDCG(k=50): 0.179816 - HitRate(k=20): 0.580481 - HitRate(k=50): 0.721448
2021-01-12 09:48:59,441 P43946 INFO Save best model: monitor(max): 0.181797
2021-01-12 09:48:59,463 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 09:48:59,512 P43946 INFO Train loss: 0.117225
2021-01-12 09:48:59,513 P43946 INFO ************ Epoch=25 end ************
2021-01-12 09:48:59,513 P43946 INFO Negative sampling num_negs=800
2021-01-12 09:49:12,547 P43946 INFO Negative sampling done
2021-01-12 09:52:00,880 P43946 INFO --- Start evaluation ---
2021-01-12 09:52:02,896 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 09:52:25,740 P43946 INFO [Metrics] Recall(k=20): 0.182327 - Recall(k=50): 0.285319 - NDCG(k=20): 0.147916 - NDCG(k=50): 0.180099 - HitRate(k=20): 0.580950 - HitRate(k=50): 0.722285
2021-01-12 09:52:25,750 P43946 INFO Save best model: monitor(max): 0.182327
2021-01-12 09:52:25,778 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 09:52:25,832 P43946 INFO Train loss: 0.116985
2021-01-12 09:52:25,832 P43946 INFO ************ Epoch=26 end ************
2021-01-12 09:52:25,833 P43946 INFO Negative sampling num_negs=800
2021-01-12 09:52:39,037 P43946 INFO Negative sampling done
2021-01-12 09:55:55,918 P43946 INFO --- Start evaluation ---
2021-01-12 09:55:57,197 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 09:56:22,663 P43946 INFO [Metrics] Recall(k=20): 0.182635 - Recall(k=50): 0.286086 - NDCG(k=20): 0.148128 - NDCG(k=50): 0.180493 - HitRate(k=20): 0.582423 - HitRate(k=50): 0.722821
2021-01-12 09:56:22,673 P43946 INFO Save best model: monitor(max): 0.182635
2021-01-12 09:56:22,696 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 09:56:22,729 P43946 INFO Train loss: 0.116865
2021-01-12 09:56:22,729 P43946 INFO ************ Epoch=27 end ************
2021-01-12 09:56:22,730 P43946 INFO Negative sampling num_negs=800
2021-01-12 09:56:35,739 P43946 INFO Negative sampling done
2021-01-12 10:00:12,307 P43946 INFO --- Start evaluation ---
2021-01-12 10:00:14,344 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 10:00:45,689 P43946 INFO [Metrics] Recall(k=20): 0.183035 - Recall(k=50): 0.286415 - NDCG(k=20): 0.148643 - NDCG(k=50): 0.180997 - HitRate(k=20): 0.583127 - HitRate(k=50): 0.723190
2021-01-12 10:00:45,729 P43946 INFO Save best model: monitor(max): 0.183035
2021-01-12 10:00:45,796 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 10:00:45,868 P43946 INFO Train loss: 0.116661
2021-01-12 10:00:45,868 P43946 INFO ************ Epoch=28 end ************
2021-01-12 10:00:45,869 P43946 INFO Negative sampling num_negs=800
2021-01-12 10:00:59,148 P43946 INFO Negative sampling done
2021-01-12 10:04:16,569 P43946 INFO --- Start evaluation ---
2021-01-12 10:04:19,316 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 10:04:43,113 P43946 INFO [Metrics] Recall(k=20): 0.183053 - Recall(k=50): 0.286945 - NDCG(k=20): 0.148740 - NDCG(k=50): 0.181236 - HitRate(k=20): 0.582825 - HitRate(k=50): 0.724061
2021-01-12 10:04:43,123 P43946 INFO Save best model: monitor(max): 0.183053
2021-01-12 10:04:43,152 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 10:04:43,187 P43946 INFO Train loss: 0.116595
2021-01-12 10:04:43,187 P43946 INFO ************ Epoch=29 end ************
2021-01-12 10:04:43,188 P43946 INFO Negative sampling num_negs=800
2021-01-12 10:04:57,035 P43946 INFO Negative sampling done
2021-01-12 10:07:56,049 P43946 INFO --- Start evaluation ---
2021-01-12 10:07:58,713 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 10:08:23,132 P43946 INFO [Metrics] Recall(k=20): 0.182972 - Recall(k=50): 0.286803 - NDCG(k=20): 0.148829 - NDCG(k=50): 0.181339 - HitRate(k=20): 0.582055 - HitRate(k=50): 0.723994
2021-01-12 10:08:23,142 P43946 INFO Monitor(max) STOP: 0.182972 !
2021-01-12 10:08:23,142 P43946 INFO Reduce learning rate on plateau: 0.000001
2021-01-12 10:08:23,143 P43946 INFO Load best model: /home/zhujieming/zhujieming/DEEM_20201231_WWW/benchmarks/Gowalla/MF_CCL_gowalla_x0/gowalla_x0_4c90e422/MF_gowalla_x0_001_e5f1ed4e.model
2021-01-12 10:08:23,154 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 10:08:23,198 P43946 INFO Train loss: 0.116414
2021-01-12 10:08:23,198 P43946 INFO ************ Epoch=30 end ************
2021-01-12 10:08:23,199 P43946 INFO Negative sampling num_negs=800
2021-01-12 10:08:36,275 P43946 INFO Negative sampling done
2021-01-12 10:11:35,471 P43946 INFO --- Start evaluation ---
2021-01-12 10:11:38,872 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 10:12:03,382 P43946 INFO [Metrics] Recall(k=20): 0.183247 - Recall(k=50): 0.286914 - NDCG(k=20): 0.148869 - NDCG(k=50): 0.181275 - HitRate(k=20): 0.583428 - HitRate(k=50): 0.723994
2021-01-12 10:12:03,392 P43946 INFO Save best model: monitor(max): 0.183247
2021-01-12 10:12:03,413 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 10:12:03,460 P43946 INFO Train loss: 0.115427
2021-01-12 10:12:03,460 P43946 INFO ************ Epoch=31 end ************
2021-01-12 10:12:03,461 P43946 INFO Negative sampling num_negs=800
2021-01-12 10:12:16,864 P43946 INFO Negative sampling done
2021-01-12 10:14:50,563 P43946 INFO --- Start evaluation ---
2021-01-12 10:14:52,186 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 10:15:16,558 P43946 INFO [Metrics] Recall(k=20): 0.183317 - Recall(k=50): 0.286894 - NDCG(k=20): 0.148913 - NDCG(k=50): 0.181292 - HitRate(k=20): 0.583596 - HitRate(k=50): 0.724094
2021-01-12 10:15:16,569 P43946 INFO Save best model: monitor(max): 0.183317
2021-01-12 10:15:16,598 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 10:15:16,640 P43946 INFO Train loss: 0.115418
2021-01-12 10:15:16,640 P43946 INFO ************ Epoch=32 end ************
2021-01-12 10:15:16,641 P43946 INFO Negative sampling num_negs=800
2021-01-12 10:15:30,360 P43946 INFO Negative sampling done
2021-01-12 10:18:49,727 P43946 INFO --- Start evaluation ---
2021-01-12 10:18:51,728 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 10:19:16,814 P43946 INFO [Metrics] Recall(k=20): 0.183350 - Recall(k=50): 0.287041 - NDCG(k=20): 0.149023 - NDCG(k=50): 0.181441 - HitRate(k=20): 0.583696 - HitRate(k=50): 0.724429
2021-01-12 10:19:16,826 P43946 INFO Save best model: monitor(max): 0.183350
2021-01-12 10:19:16,848 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 10:19:16,893 P43946 INFO Train loss: 0.115435
2021-01-12 10:19:16,893 P43946 INFO ************ Epoch=33 end ************
2021-01-12 10:19:16,894 P43946 INFO Negative sampling num_negs=800
2021-01-12 10:19:31,364 P43946 INFO Negative sampling done
2021-01-12 10:22:52,029 P43946 INFO --- Start evaluation ---
2021-01-12 10:22:54,611 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 10:23:22,320 P43946 INFO [Metrics] Recall(k=20): 0.183531 - Recall(k=50): 0.286956 - NDCG(k=20): 0.149102 - NDCG(k=50): 0.181442 - HitRate(k=20): 0.584098 - HitRate(k=50): 0.724128
2021-01-12 10:23:22,332 P43946 INFO Save best model: monitor(max): 0.183531
2021-01-12 10:23:22,363 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 10:23:22,421 P43946 INFO Train loss: 0.115346
2021-01-12 10:23:22,421 P43946 INFO ************ Epoch=34 end ************
2021-01-12 10:23:22,422 P43946 INFO Negative sampling num_negs=800
2021-01-12 10:23:37,120 P43946 INFO Negative sampling done
2021-01-12 10:26:54,619 P43946 INFO --- Start evaluation ---
2021-01-12 10:26:56,863 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 10:27:20,599 P43946 INFO [Metrics] Recall(k=20): 0.183387 - Recall(k=50): 0.287190 - NDCG(k=20): 0.148985 - NDCG(k=50): 0.181448 - HitRate(k=20): 0.583830 - HitRate(k=50): 0.724328
2021-01-12 10:27:20,609 P43946 INFO Monitor(max) STOP: 0.183387 !
2021-01-12 10:27:20,609 P43946 INFO Reduce learning rate on plateau: 0.000001
2021-01-12 10:27:20,609 P43946 INFO Load best model: /home/zhujieming/zhujieming/DEEM_20201231_WWW/benchmarks/Gowalla/MF_CCL_gowalla_x0/gowalla_x0_4c90e422/MF_gowalla_x0_001_e5f1ed4e.model
2021-01-12 10:27:20,623 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 10:27:20,703 P43946 INFO Train loss: 0.115389
2021-01-12 10:27:20,704 P43946 INFO ************ Epoch=35 end ************
2021-01-12 10:27:20,704 P43946 INFO Negative sampling num_negs=800
2021-01-12 10:27:34,762 P43946 INFO Negative sampling done
2021-01-12 10:30:37,606 P43946 INFO --- Start evaluation ---
2021-01-12 10:30:39,694 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 10:31:02,452 P43946 INFO [Metrics] Recall(k=20): 0.183390 - Recall(k=50): 0.287165 - NDCG(k=20): 0.148951 - NDCG(k=50): 0.181412 - HitRate(k=20): 0.583696 - HitRate(k=50): 0.724429
2021-01-12 10:31:02,461 P43946 INFO Monitor(max) STOP: 0.183390 !
2021-01-12 10:31:02,462 P43946 INFO Reduce learning rate on plateau: 0.000001
2021-01-12 10:31:02,462 P43946 INFO Load best model: /home/zhujieming/zhujieming/DEEM_20201231_WWW/benchmarks/Gowalla/MF_CCL_gowalla_x0/gowalla_x0_4c90e422/MF_gowalla_x0_001_e5f1ed4e.model
2021-01-12 10:31:02,473 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 10:31:02,521 P43946 INFO Train loss: 0.115338
2021-01-12 10:31:02,521 P43946 INFO ************ Epoch=36 end ************
2021-01-12 10:31:02,522 P43946 INFO Negative sampling num_negs=800
2021-01-12 10:31:16,800 P43946 INFO Negative sampling done
2021-01-12 10:34:29,138 P43946 INFO --- Start evaluation ---
2021-01-12 10:34:30,939 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 10:34:56,229 P43946 INFO [Metrics] Recall(k=20): 0.183544 - Recall(k=50): 0.287016 - NDCG(k=20): 0.149105 - NDCG(k=50): 0.181479 - HitRate(k=20): 0.584065 - HitRate(k=50): 0.724128
2021-01-12 10:34:56,239 P43946 INFO Save best model: monitor(max): 0.183544
2021-01-12 10:34:56,262 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 10:34:56,309 P43946 INFO Train loss: 0.115279
2021-01-12 10:34:56,309 P43946 INFO ************ Epoch=37 end ************
2021-01-12 10:34:56,310 P43946 INFO Negative sampling num_negs=800
2021-01-12 10:35:09,222 P43946 INFO Negative sampling done
2021-01-12 10:38:36,016 P43946 INFO --- Start evaluation ---
2021-01-12 10:38:40,326 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 10:39:05,428 P43946 INFO [Metrics] Recall(k=20): 0.183605 - Recall(k=50): 0.287042 - NDCG(k=20): 0.149137 - NDCG(k=50): 0.181481 - HitRate(k=20): 0.584132 - HitRate(k=50): 0.724262
2021-01-12 10:39:05,438 P43946 INFO Save best model: monitor(max): 0.183605
2021-01-12 10:39:05,469 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 10:39:05,521 P43946 INFO Train loss: 0.115389
2021-01-12 10:39:05,521 P43946 INFO ************ Epoch=38 end ************
2021-01-12 10:39:05,522 P43946 INFO Negative sampling num_negs=800
2021-01-12 10:39:19,632 P43946 INFO Negative sampling done
2021-01-12 10:42:30,788 P43946 INFO --- Start evaluation ---
2021-01-12 10:42:33,856 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 10:42:58,344 P43946 INFO [Metrics] Recall(k=20): 0.183661 - Recall(k=50): 0.286995 - NDCG(k=20): 0.149191 - NDCG(k=50): 0.181509 - HitRate(k=20): 0.584098 - HitRate(k=50): 0.723893
2021-01-12 10:42:58,354 P43946 INFO Save best model: monitor(max): 0.183661
2021-01-12 10:42:58,377 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 10:42:58,420 P43946 INFO Train loss: 0.115239
2021-01-12 10:42:58,420 P43946 INFO ************ Epoch=39 end ************
2021-01-12 10:42:58,421 P43946 INFO Negative sampling num_negs=800
2021-01-12 10:43:12,567 P43946 INFO Negative sampling done
2021-01-12 10:46:32,846 P43946 INFO --- Start evaluation ---
2021-01-12 10:46:35,262 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 10:47:02,111 P43946 INFO [Metrics] Recall(k=20): 0.183730 - Recall(k=50): 0.287096 - NDCG(k=20): 0.149257 - NDCG(k=50): 0.181573 - HitRate(k=20): 0.584031 - HitRate(k=50): 0.724262
2021-01-12 10:47:02,121 P43946 INFO Save best model: monitor(max): 0.183730
2021-01-12 10:47:02,148 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 10:47:02,207 P43946 INFO Train loss: 0.115187
2021-01-12 10:47:02,207 P43946 INFO ************ Epoch=40 end ************
2021-01-12 10:47:02,207 P43946 INFO Negative sampling num_negs=800
2021-01-12 10:47:15,296 P43946 INFO Negative sampling done
2021-01-12 10:50:37,656 P43946 INFO --- Start evaluation ---
2021-01-12 10:50:40,309 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 10:51:07,340 P43946 INFO [Metrics] Recall(k=20): 0.183655 - Recall(k=50): 0.287102 - NDCG(k=20): 0.149273 - NDCG(k=50): 0.181612 - HitRate(k=20): 0.583830 - HitRate(k=50): 0.724195
2021-01-12 10:51:07,355 P43946 INFO Monitor(max) STOP: 0.183655 !
2021-01-12 10:51:07,356 P43946 INFO Reduce learning rate on plateau: 0.000001
2021-01-12 10:51:07,356 P43946 INFO Load best model: /home/zhujieming/zhujieming/DEEM_20201231_WWW/benchmarks/Gowalla/MF_CCL_gowalla_x0/gowalla_x0_4c90e422/MF_gowalla_x0_001_e5f1ed4e.model
2021-01-12 10:51:07,370 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 10:51:07,425 P43946 INFO Train loss: 0.115239
2021-01-12 10:51:07,425 P43946 INFO ************ Epoch=41 end ************
2021-01-12 10:51:07,425 P43946 INFO Negative sampling num_negs=800
2021-01-12 10:51:21,385 P43946 INFO Negative sampling done
2021-01-12 10:54:40,732 P43946 INFO --- Start evaluation ---
2021-01-12 10:54:43,998 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 10:55:08,138 P43946 INFO [Metrics] Recall(k=20): 0.183727 - Recall(k=50): 0.287048 - NDCG(k=20): 0.149240 - NDCG(k=50): 0.181536 - HitRate(k=20): 0.584199 - HitRate(k=50): 0.723994
2021-01-12 10:55:08,158 P43946 INFO Monitor(max) STOP: 0.183727 !
2021-01-12 10:55:08,158 P43946 INFO Reduce learning rate on plateau: 0.000001
2021-01-12 10:55:08,159 P43946 INFO Load best model: /home/zhujieming/zhujieming/DEEM_20201231_WWW/benchmarks/Gowalla/MF_CCL_gowalla_x0/gowalla_x0_4c90e422/MF_gowalla_x0_001_e5f1ed4e.model
2021-01-12 10:55:08,174 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 10:55:08,215 P43946 INFO Train loss: 0.115191
2021-01-12 10:55:08,216 P43946 INFO ************ Epoch=42 end ************
2021-01-12 10:55:08,216 P43946 INFO Negative sampling num_negs=800
2021-01-12 10:55:22,781 P43946 INFO Negative sampling done
2021-01-12 10:58:04,437 P43946 INFO --- Start evaluation ---
2021-01-12 10:58:07,719 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 10:58:31,863 P43946 INFO [Metrics] Recall(k=20): 0.183671 - Recall(k=50): 0.287160 - NDCG(k=20): 0.149252 - NDCG(k=50): 0.181602 - HitRate(k=20): 0.583864 - HitRate(k=50): 0.723960
2021-01-12 10:58:31,874 P43946 INFO Monitor(max) STOP: 0.183671 !
2021-01-12 10:58:31,874 P43946 INFO Reduce learning rate on plateau: 0.000001
2021-01-12 10:58:31,875 P43946 INFO Load best model: /home/zhujieming/zhujieming/DEEM_20201231_WWW/benchmarks/Gowalla/MF_CCL_gowalla_x0/gowalla_x0_4c90e422/MF_gowalla_x0_001_e5f1ed4e.model
2021-01-12 10:58:31,885 P43946 INFO Early stopping at epoch=43
2021-01-12 10:58:31,885 P43946 INFO --- 3165/3165 batches finished ---
2021-01-12 10:58:31,920 P43946 INFO Train loss: 0.115262
2021-01-12 10:58:31,921 P43946 INFO Training finished.
2021-01-12 10:58:31,921 P43946 INFO Load best model: /home/zhujieming/zhujieming/DEEM_20201231_WWW/benchmarks/Gowalla/MF_CCL_gowalla_x0/gowalla_x0_4c90e422/MF_gowalla_x0_001_e5f1ed4e.model
2021-01-12 10:58:31,941 P43946 INFO ****** Train/validation evaluation ******
2021-01-12 10:58:31,941 P43946 INFO --- Start evaluation ---
2021-01-12 10:58:35,231 P43946 INFO Evaluating metrics for 29858 users...
2021-01-12 10:58:57,677 P43946 INFO [Metrics] Recall(k=20): 0.183730 - Recall(k=50): 0.287096 - NDCG(k=20): 0.149257 - NDCG(k=50): 0.181573 - HitRate(k=20): 0.584031 - HitRate(k=50): 0.724262
```