## MF-CCL_yelp18_x0 

A notebook to benchmark MF-CCL on yelp18 dataset.

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
Yelp18 follows the data split and preprocessing steps in NGCF and LightGCN. We directly transform the formats of the data from their [repo](https://github.com/kuandeng/LightGCN/tree/master/Data).


### Code
The implementation code is already available at https://github.com/xue-pai/DEEM. 

Please refer to the [configuration file](https://github.com/xue-pai/DEEM/blob/master/benchmarks/Yelp18/SimpleX_yelp18_x0/SimpleX_yelp18_x0_tuner_config.yaml).
### Results
```python
2021-01-15 11:07:20,143 P3984 INFO Set up feature encoder...
2021-01-15 11:07:20,144 P3984 INFO Load feature_map from json: ../data/Yelp18/yelp18_x0_0f43e4ba/feature_map.json
2021-01-15 11:07:29,105 P3984 INFO Total number of parameters: 4461952.
2021-01-15 11:07:29,106 P3984 INFO Loading data...
2021-01-15 11:07:29,121 P3984 INFO Loading data from h5: ../data/Yelp18/yelp18_x0_0f43e4ba/train.h5
2021-01-15 11:07:29,172 P3984 INFO Loading data from h5: ../data/Yelp18/yelp18_x0_0f43e4ba/item_corpus.h5
2021-01-15 11:07:30,246 P3984 INFO Loading data from h5: ../data/Yelp18/yelp18_x0_0f43e4ba/valid.h5
2021-01-15 11:07:30,740 P3984 INFO Loading data from h5: ../data/Yelp18/yelp18_x0_0f43e4ba/item_corpus.h5
2021-01-15 11:07:30,752 P3984 INFO Train samples: total/1237259, blocks/1
2021-01-15 11:07:30,752 P3984 INFO Validation samples: total/31668, blocks/1
2021-01-15 11:07:30,752 P3984 INFO Loading train data done.
2021-01-15 11:07:30,753 P3984 INFO **** Start training: 2417 batches/epoch ****
2021-01-15 11:07:30,764 P3984 INFO Negative sampling num_negs=800
2021-01-15 11:08:26,871 P3984 INFO Negative sampling done
2021-01-15 11:10:30,033 P3984 INFO --- Start evaluation ---
2021-01-15 11:10:34,746 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 11:12:30,471 P3984 INFO [Metrics] Recall(k=20): 0.033358 - Recall(k=50): 0.068826 - NDCG(k=20): 0.026848 - NDCG(k=50): 0.040175 - HitRate(k=20): 0.241632 - HitRate(k=50): 0.406088
2021-01-15 11:12:30,531 P3984 INFO Save best model: monitor(max): 0.033358
2021-01-15 11:12:30,686 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 11:12:31,078 P3984 INFO Train loss: 0.470274
2021-01-15 11:12:31,078 P3984 INFO ************ Epoch=1 end ************
2021-01-15 11:12:31,089 P3984 INFO Negative sampling num_negs=800
2021-01-15 11:14:08,317 P3984 INFO Negative sampling done
2021-01-15 11:16:21,234 P3984 INFO --- Start evaluation ---
2021-01-15 11:16:24,207 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 11:17:58,098 P3984 INFO [Metrics] Recall(k=20): 0.042442 - Recall(k=50): 0.085590 - NDCG(k=20): 0.034272 - NDCG(k=50): 0.050342 - HitRate(k=20): 0.288872 - HitRate(k=50): 0.470191
2021-01-15 11:17:58,148 P3984 INFO Save best model: monitor(max): 0.042442
2021-01-15 11:17:58,258 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 11:17:58,608 P3984 INFO Train loss: 0.197617
2021-01-15 11:17:58,608 P3984 INFO ************ Epoch=2 end ************
2021-01-15 11:17:58,610 P3984 INFO Negative sampling num_negs=800
2021-01-15 11:19:11,827 P3984 INFO Negative sampling done
2021-01-15 11:21:05,285 P3984 INFO --- Start evaluation ---
2021-01-15 11:21:08,651 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 11:23:08,521 P3984 INFO [Metrics] Recall(k=20): 0.047097 - Recall(k=50): 0.094089 - NDCG(k=20): 0.038127 - NDCG(k=50): 0.055704 - HitRate(k=20): 0.315460 - HitRate(k=50): 0.499937
2021-01-15 11:23:08,559 P3984 INFO Save best model: monitor(max): 0.047097
2021-01-15 11:23:08,673 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 11:23:08,912 P3984 INFO Train loss: 0.183109
2021-01-15 11:23:08,912 P3984 INFO ************ Epoch=3 end ************
2021-01-15 11:23:08,913 P3984 INFO Negative sampling num_negs=800
2021-01-15 11:24:47,674 P3984 INFO Negative sampling done
2021-01-15 11:27:02,559 P3984 INFO --- Start evaluation ---
2021-01-15 11:27:06,578 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 11:29:07,859 P3984 INFO [Metrics] Recall(k=20): 0.052094 - Recall(k=50): 0.102444 - NDCG(k=20): 0.042204 - NDCG(k=50): 0.060995 - HitRate(k=20): 0.336081 - HitRate(k=50): 0.526273
2021-01-15 11:29:07,874 P3984 INFO Save best model: monitor(max): 0.052094
2021-01-15 11:29:07,955 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 11:29:08,181 P3984 INFO Train loss: 0.174543
2021-01-15 11:29:08,182 P3984 INFO ************ Epoch=4 end ************
2021-01-15 11:29:08,192 P3984 INFO Negative sampling num_negs=800
2021-01-15 11:30:19,569 P3984 INFO Negative sampling done
2021-01-15 11:31:58,576 P3984 INFO --- Start evaluation ---
2021-01-15 11:32:01,138 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 11:33:38,924 P3984 INFO [Metrics] Recall(k=20): 0.055025 - Recall(k=50): 0.107512 - NDCG(k=20): 0.044388 - NDCG(k=50): 0.063913 - HitRate(k=20): 0.353385 - HitRate(k=50): 0.543577
2021-01-15 11:33:38,982 P3984 INFO Save best model: monitor(max): 0.055025
2021-01-15 11:33:39,096 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 11:33:39,533 P3984 INFO Train loss: 0.168441
2021-01-15 11:33:39,533 P3984 INFO ************ Epoch=5 end ************
2021-01-15 11:33:39,543 P3984 INFO Negative sampling num_negs=800
2021-01-15 11:35:19,452 P3984 INFO Negative sampling done
2021-01-15 11:37:27,425 P3984 INFO --- Start evaluation ---
2021-01-15 11:37:31,608 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 11:39:34,910 P3984 INFO [Metrics] Recall(k=20): 0.057059 - Recall(k=50): 0.111085 - NDCG(k=20): 0.046419 - NDCG(k=50): 0.066495 - HitRate(k=20): 0.359290 - HitRate(k=50): 0.551945
2021-01-15 11:39:34,953 P3984 INFO Save best model: monitor(max): 0.057059
2021-01-15 11:39:35,053 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 11:39:35,404 P3984 INFO Train loss: 0.163763
2021-01-15 11:39:35,404 P3984 INFO ************ Epoch=6 end ************
2021-01-15 11:39:35,416 P3984 INFO Negative sampling num_negs=800
2021-01-15 11:41:10,471 P3984 INFO Negative sampling done
2021-01-15 11:42:38,094 P3984 INFO --- Start evaluation ---
2021-01-15 11:42:38,763 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 11:43:53,245 P3984 INFO [Metrics] Recall(k=20): 0.058980 - Recall(k=50): 0.114286 - NDCG(k=20): 0.047511 - NDCG(k=50): 0.068096 - HitRate(k=20): 0.369679 - HitRate(k=50): 0.559555
2021-01-15 11:43:53,286 P3984 INFO Save best model: monitor(max): 0.058980
2021-01-15 11:43:53,368 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 11:43:53,608 P3984 INFO Train loss: 0.160154
2021-01-15 11:43:53,608 P3984 INFO ************ Epoch=7 end ************
2021-01-15 11:43:53,621 P3984 INFO Negative sampling num_negs=800
2021-01-15 11:45:21,047 P3984 INFO Negative sampling done
2021-01-15 11:47:39,264 P3984 INFO --- Start evaluation ---
2021-01-15 11:47:42,254 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 11:49:42,969 P3984 INFO [Metrics] Recall(k=20): 0.060086 - Recall(k=50): 0.116729 - NDCG(k=20): 0.048955 - NDCG(k=50): 0.069949 - HitRate(k=20): 0.372932 - HitRate(k=50): 0.570323
2021-01-15 11:49:43,027 P3984 INFO Save best model: monitor(max): 0.060086
2021-01-15 11:49:43,125 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 11:49:43,472 P3984 INFO Train loss: 0.157452
2021-01-15 11:49:43,472 P3984 INFO ************ Epoch=8 end ************
2021-01-15 11:49:43,473 P3984 INFO Negative sampling num_negs=800
2021-01-15 11:51:24,507 P3984 INFO Negative sampling done
2021-01-15 11:53:24,259 P3984 INFO --- Start evaluation ---
2021-01-15 11:53:26,024 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 11:54:08,746 P3984 INFO [Metrics] Recall(k=20): 0.061219 - Recall(k=50): 0.119202 - NDCG(k=20): 0.049766 - NDCG(k=50): 0.071305 - HitRate(k=20): 0.376689 - HitRate(k=50): 0.574050
2021-01-15 11:54:08,762 P3984 INFO Save best model: monitor(max): 0.061219
2021-01-15 11:54:08,792 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 11:54:09,016 P3984 INFO Train loss: 0.155050
2021-01-15 11:54:09,016 P3984 INFO ************ Epoch=9 end ************
2021-01-15 11:54:09,027 P3984 INFO Negative sampling num_negs=800
2021-01-15 11:55:14,515 P3984 INFO Negative sampling done
2021-01-15 11:57:25,204 P3984 INFO --- Start evaluation ---
2021-01-15 11:57:28,449 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 11:59:32,974 P3984 INFO [Metrics] Recall(k=20): 0.062157 - Recall(k=50): 0.119579 - NDCG(k=20): 0.050720 - NDCG(k=50): 0.072039 - HitRate(k=20): 0.382215 - HitRate(k=50): 0.573797
2021-01-15 11:59:33,015 P3984 INFO Save best model: monitor(max): 0.062157
2021-01-15 11:59:33,107 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 11:59:33,442 P3984 INFO Train loss: 0.153155
2021-01-15 11:59:33,442 P3984 INFO ************ Epoch=10 end ************
2021-01-15 11:59:33,445 P3984 INFO Negative sampling num_negs=800
2021-01-15 12:01:15,076 P3984 INFO Negative sampling done
2021-01-15 12:03:34,160 P3984 INFO --- Start evaluation ---
2021-01-15 12:03:36,946 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 12:04:57,339 P3984 INFO [Metrics] Recall(k=20): 0.063377 - Recall(k=50): 0.121277 - NDCG(k=20): 0.051329 - NDCG(k=50): 0.072825 - HitRate(k=20): 0.386478 - HitRate(k=50): 0.579702
2021-01-15 12:04:57,364 P3984 INFO Save best model: monitor(max): 0.063377
2021-01-15 12:04:57,400 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 12:04:57,604 P3984 INFO Train loss: 0.151631
2021-01-15 12:04:57,605 P3984 INFO ************ Epoch=11 end ************
2021-01-15 12:04:57,610 P3984 INFO Negative sampling num_negs=800
2021-01-15 12:05:33,364 P3984 INFO Negative sampling done
2021-01-15 12:07:45,201 P3984 INFO --- Start evaluation ---
2021-01-15 12:07:48,621 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 12:09:49,231 P3984 INFO [Metrics] Recall(k=20): 0.063654 - Recall(k=50): 0.122828 - NDCG(k=20): 0.052083 - NDCG(k=50): 0.074030 - HitRate(k=20): 0.388720 - HitRate(k=50): 0.587249
2021-01-15 12:09:49,263 P3984 INFO Save best model: monitor(max): 0.063654
2021-01-15 12:09:49,339 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 12:09:49,677 P3984 INFO Train loss: 0.150172
2021-01-15 12:09:49,677 P3984 INFO ************ Epoch=12 end ************
2021-01-15 12:09:49,688 P3984 INFO Negative sampling num_negs=800
2021-01-15 12:11:27,108 P3984 INFO Negative sampling done
2021-01-15 12:13:44,402 P3984 INFO --- Start evaluation ---
2021-01-15 12:13:47,604 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 12:15:30,140 P3984 INFO [Metrics] Recall(k=20): 0.063981 - Recall(k=50): 0.123050 - NDCG(k=20): 0.052190 - NDCG(k=50): 0.074058 - HitRate(k=20): 0.388152 - HitRate(k=50): 0.586270
2021-01-15 12:15:30,170 P3984 INFO Save best model: monitor(max): 0.063981
2021-01-15 12:15:30,229 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 12:15:30,532 P3984 INFO Train loss: 0.149061
2021-01-15 12:15:30,532 P3984 INFO ************ Epoch=13 end ************
2021-01-15 12:15:30,541 P3984 INFO Negative sampling num_negs=800
2021-01-15 12:16:27,165 P3984 INFO Negative sampling done
2021-01-15 12:18:19,040 P3984 INFO --- Start evaluation ---
2021-01-15 12:18:23,270 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 12:20:31,314 P3984 INFO [Metrics] Recall(k=20): 0.064314 - Recall(k=50): 0.124222 - NDCG(k=20): 0.052700 - NDCG(k=50): 0.074895 - HitRate(k=20): 0.389889 - HitRate(k=50): 0.586838
2021-01-15 12:20:31,395 P3984 INFO Save best model: monitor(max): 0.064314
2021-01-15 12:20:31,507 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 12:20:31,889 P3984 INFO Train loss: 0.147952
2021-01-15 12:20:31,889 P3984 INFO ************ Epoch=14 end ************
2021-01-15 12:20:31,901 P3984 INFO Negative sampling num_negs=800
2021-01-15 12:22:18,604 P3984 INFO Negative sampling done
2021-01-15 12:24:25,425 P3984 INFO --- Start evaluation ---
2021-01-15 12:24:28,321 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 12:26:15,768 P3984 INFO [Metrics] Recall(k=20): 0.065062 - Recall(k=50): 0.124532 - NDCG(k=20): 0.053296 - NDCG(k=50): 0.075338 - HitRate(k=20): 0.395004 - HitRate(k=50): 0.590186
2021-01-15 12:26:15,817 P3984 INFO Save best model: monitor(max): 0.065062
2021-01-15 12:26:15,892 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 12:26:16,164 P3984 INFO Train loss: 0.147154
2021-01-15 12:26:16,164 P3984 INFO ************ Epoch=15 end ************
2021-01-15 12:26:16,167 P3984 INFO Negative sampling num_negs=800
2021-01-15 12:27:41,882 P3984 INFO Negative sampling done
2021-01-15 12:29:14,193 P3984 INFO --- Start evaluation ---
2021-01-15 12:29:18,401 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 12:31:30,694 P3984 INFO [Metrics] Recall(k=20): 0.064455 - Recall(k=50): 0.124387 - NDCG(k=20): 0.052790 - NDCG(k=50): 0.074993 - HitRate(k=20): 0.394625 - HitRate(k=50): 0.588417
2021-01-15 12:31:30,774 P3984 INFO Monitor(max) STOP: 0.064455 !
2021-01-15 12:31:30,774 P3984 INFO Reduce learning rate on plateau: 0.000010
2021-01-15 12:31:30,774 P3984 INFO Load best model:  MF_yelp18_x0_001_ab04e533.model
2021-01-15 12:31:30,834 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 12:31:31,137 P3984 INFO Train loss: 0.146278
2021-01-15 12:31:31,138 P3984 INFO ************ Epoch=16 end ************
2021-01-15 12:31:31,146 P3984 INFO Negative sampling num_negs=800
2021-01-15 12:33:15,498 P3984 INFO Negative sampling done
2021-01-15 12:35:38,889 P3984 INFO --- Start evaluation ---
2021-01-15 12:35:41,493 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 12:37:35,348 P3984 INFO [Metrics] Recall(k=20): 0.067182 - Recall(k=50): 0.127913 - NDCG(k=20): 0.055233 - NDCG(k=50): 0.077708 - HitRate(k=20): 0.404888 - HitRate(k=50): 0.599406
2021-01-15 12:37:35,393 P3984 INFO Save best model: monitor(max): 0.067182
2021-01-15 12:37:35,476 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 12:37:35,866 P3984 INFO Train loss: 0.137857
2021-01-15 12:37:35,866 P3984 INFO ************ Epoch=17 end ************
2021-01-15 12:37:35,872 P3984 INFO Negative sampling num_negs=800
2021-01-15 12:39:09,603 P3984 INFO Negative sampling done
2021-01-15 12:40:38,414 P3984 INFO --- Start evaluation ---
2021-01-15 12:40:42,247 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 12:42:44,933 P3984 INFO [Metrics] Recall(k=20): 0.067588 - Recall(k=50): 0.128823 - NDCG(k=20): 0.055709 - NDCG(k=50): 0.078391 - HitRate(k=20): 0.405899 - HitRate(k=50): 0.601048
2021-01-15 12:42:44,995 P3984 INFO Save best model: monitor(max): 0.067588
2021-01-15 12:42:45,089 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 12:42:45,516 P3984 INFO Train loss: 0.136031
2021-01-15 12:42:45,516 P3984 INFO ************ Epoch=18 end ************
2021-01-15 12:42:45,517 P3984 INFO Negative sampling num_negs=800
2021-01-15 12:44:30,425 P3984 INFO Negative sampling done
2021-01-15 12:46:41,071 P3984 INFO --- Start evaluation ---
2021-01-15 12:46:44,110 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 12:48:30,009 P3984 INFO [Metrics] Recall(k=20): 0.068241 - Recall(k=50): 0.129906 - NDCG(k=20): 0.056196 - NDCG(k=50): 0.079054 - HitRate(k=20): 0.407762 - HitRate(k=50): 0.602659
2021-01-15 12:48:30,063 P3984 INFO Save best model: monitor(max): 0.068241
2021-01-15 12:48:30,155 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 12:48:30,525 P3984 INFO Train loss: 0.135218
2021-01-15 12:48:30,526 P3984 INFO ************ Epoch=19 end ************
2021-01-15 12:48:30,529 P3984 INFO Negative sampling num_negs=800
2021-01-15 12:50:03,461 P3984 INFO Negative sampling done
2021-01-15 12:51:30,395 P3984 INFO --- Start evaluation ---
2021-01-15 12:51:33,120 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 12:53:29,678 P3984 INFO [Metrics] Recall(k=20): 0.068653 - Recall(k=50): 0.130654 - NDCG(k=20): 0.056483 - NDCG(k=50): 0.079417 - HitRate(k=20): 0.409309 - HitRate(k=50): 0.605248
2021-01-15 12:53:29,729 P3984 INFO Save best model: monitor(max): 0.068653
2021-01-15 12:53:29,851 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 12:53:30,282 P3984 INFO Train loss: 0.134659
2021-01-15 12:53:30,283 P3984 INFO ************ Epoch=20 end ************
2021-01-15 12:53:30,296 P3984 INFO Negative sampling num_negs=800
2021-01-15 12:55:21,140 P3984 INFO Negative sampling done
2021-01-15 12:57:33,994 P3984 INFO --- Start evaluation ---
2021-01-15 12:57:36,474 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 12:59:30,673 P3984 INFO [Metrics] Recall(k=20): 0.068761 - Recall(k=50): 0.131312 - NDCG(k=20): 0.056468 - NDCG(k=50): 0.079654 - HitRate(k=20): 0.410572 - HitRate(k=50): 0.607995
2021-01-15 12:59:30,724 P3984 INFO Save best model: monitor(max): 0.068761
2021-01-15 12:59:30,816 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 12:59:31,291 P3984 INFO Train loss: 0.134266
2021-01-15 12:59:31,292 P3984 INFO ************ Epoch=21 end ************
2021-01-15 12:59:31,302 P3984 INFO Negative sampling num_negs=800
2021-01-15 13:01:02,318 P3984 INFO Negative sampling done
2021-01-15 13:02:53,995 P3984 INFO --- Start evaluation ---
2021-01-15 13:02:56,251 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 13:04:20,932 P3984 INFO [Metrics] Recall(k=20): 0.069012 - Recall(k=50): 0.132049 - NDCG(k=20): 0.056677 - NDCG(k=50): 0.080017 - HitRate(k=20): 0.410509 - HitRate(k=50): 0.609416
2021-01-15 13:04:20,987 P3984 INFO Save best model: monitor(max): 0.069012
2021-01-15 13:04:21,106 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 13:04:21,454 P3984 INFO Train loss: 0.133955
2021-01-15 13:04:21,454 P3984 INFO ************ Epoch=22 end ************
2021-01-15 13:04:21,463 P3984 INFO Negative sampling num_negs=800
2021-01-15 13:06:15,776 P3984 INFO Negative sampling done
2021-01-15 13:08:26,738 P3984 INFO --- Start evaluation ---
2021-01-15 13:08:29,797 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 13:10:27,846 P3984 INFO [Metrics] Recall(k=20): 0.069245 - Recall(k=50): 0.131831 - NDCG(k=20): 0.056854 - NDCG(k=50): 0.080014 - HitRate(k=20): 0.412025 - HitRate(k=50): 0.608911
2021-01-15 13:10:27,887 P3984 INFO Save best model: monitor(max): 0.069245
2021-01-15 13:10:27,943 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 13:10:28,321 P3984 INFO Train loss: 0.133686
2021-01-15 13:10:28,321 P3984 INFO ************ Epoch=23 end ************
2021-01-15 13:10:28,332 P3984 INFO Negative sampling num_negs=800
2021-01-15 13:11:47,606 P3984 INFO Negative sampling done
2021-01-15 13:13:42,349 P3984 INFO --- Start evaluation ---
2021-01-15 13:13:44,747 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 13:15:00,947 P3984 INFO [Metrics] Recall(k=20): 0.069489 - Recall(k=50): 0.131727 - NDCG(k=20): 0.057006 - NDCG(k=50): 0.080063 - HitRate(k=20): 0.411993 - HitRate(k=50): 0.609290
2021-01-15 13:15:00,979 P3984 INFO Save best model: monitor(max): 0.069489
2021-01-15 13:15:01,034 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 13:15:01,366 P3984 INFO Train loss: 0.133438
2021-01-15 13:15:01,366 P3984 INFO ************ Epoch=24 end ************
2021-01-15 13:15:01,380 P3984 INFO Negative sampling num_negs=800
2021-01-15 13:16:49,492 P3984 INFO Negative sampling done
2021-01-15 13:19:09,278 P3984 INFO --- Start evaluation ---
2021-01-15 13:19:12,457 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 13:20:53,489 P3984 INFO [Metrics] Recall(k=20): 0.069770 - Recall(k=50): 0.132211 - NDCG(k=20): 0.057228 - NDCG(k=50): 0.080342 - HitRate(k=20): 0.412783 - HitRate(k=50): 0.610490
2021-01-15 13:20:53,554 P3984 INFO Save best model: monitor(max): 0.069770
2021-01-15 13:20:53,670 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 13:20:54,081 P3984 INFO Train loss: 0.133314
2021-01-15 13:20:54,082 P3984 INFO ************ Epoch=25 end ************
2021-01-15 13:20:54,099 P3984 INFO Negative sampling num_negs=800
2021-01-15 13:22:38,571 P3984 INFO Negative sampling done
2021-01-15 13:24:31,424 P3984 INFO --- Start evaluation ---
2021-01-15 13:24:34,054 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 13:25:38,815 P3984 INFO [Metrics] Recall(k=20): 0.069662 - Recall(k=50): 0.132362 - NDCG(k=20): 0.057255 - NDCG(k=50): 0.080468 - HitRate(k=20): 0.412719 - HitRate(k=50): 0.611216
2021-01-15 13:25:38,830 P3984 INFO Monitor(max) STOP: 0.069662 !
2021-01-15 13:25:38,831 P3984 INFO Reduce learning rate on plateau: 0.000001
2021-01-15 13:25:38,831 P3984 INFO Load best model:  MF_yelp18_x0_001_ab04e533.model
2021-01-15 13:25:38,849 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 13:25:39,006 P3984 INFO Train loss: 0.133077
2021-01-15 13:25:39,006 P3984 INFO ************ Epoch=26 end ************
2021-01-15 13:25:39,009 P3984 INFO Negative sampling num_negs=800
2021-01-15 13:26:38,775 P3984 INFO Negative sampling done
2021-01-15 13:28:49,394 P3984 INFO --- Start evaluation ---
2021-01-15 13:28:53,168 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 13:30:51,833 P3984 INFO [Metrics] Recall(k=20): 0.069694 - Recall(k=50): 0.132319 - NDCG(k=20): 0.057202 - NDCG(k=50): 0.080392 - HitRate(k=20): 0.412972 - HitRate(k=50): 0.610901
2021-01-15 13:30:51,861 P3984 INFO Monitor(max) STOP: 0.069694 !
2021-01-15 13:30:51,861 P3984 INFO Reduce learning rate on plateau: 0.000001
2021-01-15 13:30:51,864 P3984 INFO Load best model:  MF_yelp18_x0_001_ab04e533.model
2021-01-15 13:30:51,900 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 13:30:52,213 P3984 INFO Train loss: 0.132159
2021-01-15 13:30:52,213 P3984 INFO ************ Epoch=27 end ************
2021-01-15 13:30:52,214 P3984 INFO Negative sampling num_negs=800
2021-01-15 13:32:18,973 P3984 INFO Negative sampling done
2021-01-15 13:34:27,558 P3984 INFO --- Start evaluation ---
2021-01-15 13:34:31,332 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 13:36:06,691 P3984 INFO [Metrics] Recall(k=20): 0.069846 - Recall(k=50): 0.132210 - NDCG(k=20): 0.057244 - NDCG(k=50): 0.080330 - HitRate(k=20): 0.413288 - HitRate(k=50): 0.610553
2021-01-15 13:36:06,723 P3984 INFO Save best model: monitor(max): 0.069846
2021-01-15 13:36:06,822 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 13:36:07,112 P3984 INFO Train loss: 0.132073
2021-01-15 13:36:07,112 P3984 INFO ************ Epoch=28 end ************
2021-01-15 13:36:07,122 P3984 INFO Negative sampling num_negs=800
2021-01-15 13:37:22,881 P3984 INFO Negative sampling done
2021-01-15 13:39:30,670 P3984 INFO --- Start evaluation ---
2021-01-15 13:39:34,375 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 13:41:27,629 P3984 INFO [Metrics] Recall(k=20): 0.069832 - Recall(k=50): 0.132286 - NDCG(k=20): 0.057261 - NDCG(k=50): 0.080380 - HitRate(k=20): 0.413035 - HitRate(k=50): 0.610806
2021-01-15 13:41:27,660 P3984 INFO Monitor(max) STOP: 0.069832 !
2021-01-15 13:41:27,660 P3984 INFO Reduce learning rate on plateau: 0.000001
2021-01-15 13:41:27,660 P3984 INFO Load best model:  MF_yelp18_x0_001_ab04e533.model
2021-01-15 13:41:27,719 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 13:41:28,022 P3984 INFO Train loss: 0.132123
2021-01-15 13:41:28,022 P3984 INFO ************ Epoch=29 end ************
2021-01-15 13:41:28,023 P3984 INFO Negative sampling num_negs=800
2021-01-15 13:42:58,246 P3984 INFO Negative sampling done
2021-01-15 13:45:12,367 P3984 INFO --- Start evaluation ---
2021-01-15 13:45:16,267 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 13:47:03,880 P3984 INFO [Metrics] Recall(k=20): 0.069821 - Recall(k=50): 0.132265 - NDCG(k=20): 0.057251 - NDCG(k=50): 0.080368 - HitRate(k=20): 0.413098 - HitRate(k=50): 0.610680
2021-01-15 13:47:03,908 P3984 INFO Monitor(max) STOP: 0.069821 !
2021-01-15 13:47:03,908 P3984 INFO Reduce learning rate on plateau: 0.000001
2021-01-15 13:47:03,909 P3984 INFO Load best model:  MF_yelp18_x0_001_ab04e533.model
2021-01-15 13:47:03,960 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 13:47:04,238 P3984 INFO Train loss: 0.132100
2021-01-15 13:47:04,238 P3984 INFO ************ Epoch=30 end ************
2021-01-15 13:47:04,241 P3984 INFO Negative sampling num_negs=800
2021-01-15 13:48:25,889 P3984 INFO Negative sampling done
2021-01-15 13:50:22,744 P3984 INFO --- Start evaluation ---
2021-01-15 13:50:26,053 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 13:52:17,876 P3984 INFO [Metrics] Recall(k=20): 0.069839 - Recall(k=50): 0.132338 - NDCG(k=20): 0.057247 - NDCG(k=50): 0.080398 - HitRate(k=20): 0.413004 - HitRate(k=50): 0.610806
2021-01-15 13:52:17,914 P3984 INFO Monitor(max) STOP: 0.069839 !
2021-01-15 13:52:17,933 P3984 INFO Reduce learning rate on plateau: 0.000001
2021-01-15 13:52:17,934 P3984 INFO Load best model:  MF_yelp18_x0_001_ab04e533.model
2021-01-15 13:52:17,988 P3984 INFO Early stopping at epoch=31
2021-01-15 13:52:17,989 P3984 INFO --- 2417/2417 batches finished ---
2021-01-15 13:52:18,395 P3984 INFO Train loss: 0.132120
2021-01-15 13:52:18,395 P3984 INFO Training finished.
2021-01-15 13:52:18,396 P3984 INFO Load best model:  MF_yelp18_x0_001_ab04e533.model
2021-01-15 13:52:18,478 P3984 INFO ****** Train/validation evaluation ******
2021-01-15 13:52:18,478 P3984 INFO --- Start evaluation ---
2021-01-15 13:52:22,755 P3984 INFO Evaluating metrics for 31668 users...
2021-01-15 13:54:18,708 P3984 INFO [Metrics] Recall(k=20): 0.069846 - Recall(k=50): 0.132210 - NDCG(k=20): 0.057244 - NDCG(k=50): 0.080330 - HitRate(k=20): 0.413288 - HitRate(k=50): 0.610553
```


### Logs
```


```