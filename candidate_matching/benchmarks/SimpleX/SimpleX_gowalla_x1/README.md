## SimpleX_gowalla_x0 

A notebook to benchmark SimpleX on gowalla dataset.

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
2021-01-15 19:58:48,489 P49679 INFO [Metrics] Recall(k=20): 0.187213 - Recall(k=50): 0.287394 - NDCG(k=20): 0.155688 - NDCG(k=50): 0.186816 - HitRate(k=20): 0.592705 - HitRate(k=50): 0.727242
```


### Logs
```
2021-01-15 17:07:09,228 P49679 INFO Set up feature encoder...
2021-01-15 17:07:09,229 P49679 INFO Load feature_map from json: ../data/Gowalla/gowalla_x0_52a9ab28/feature_map.json
2021-01-15 17:07:13,664 P49679 INFO Total number of parameters: 7164992.
2021-01-15 17:07:13,665 P49679 INFO Loading data...
2021-01-15 17:07:13,669 P49679 INFO Loading data from h5: ../data/Gowalla/gowalla_x0_52a9ab28/train.h5
2021-01-15 17:07:15,322 P49679 INFO Loading data from h5: ../data/Gowalla/gowalla_x0_52a9ab28/item_corpus.h5
2021-01-15 17:07:15,708 P49679 INFO Loading data from h5: ../data/Gowalla/gowalla_x0_52a9ab28/valid.h5
2021-01-15 17:07:16,142 P49679 INFO Loading data from h5: ../data/Gowalla/gowalla_x0_52a9ab28/item_corpus.h5
2021-01-15 17:07:16,144 P49679 INFO Train samples: total/810128, blocks/1
2021-01-15 17:07:16,144 P49679 INFO Validation samples: total/29858, blocks/1
2021-01-15 17:07:16,144 P49679 INFO Loading train data done.
2021-01-15 17:07:16,144 P49679 INFO **** Start training: 3165 batches/epoch ****
2021-01-15 17:07:16,146 P49679 INFO Negative sampling num_negs=2000
2021-01-15 17:07:49,586 P49679 INFO Negative sampling done
2021-01-15 17:11:05,366 P49679 INFO --- Start evaluation ---
2021-01-15 17:11:07,385 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 17:11:32,987 P49679 INFO [Metrics] Recall(k=20): 0.097691 - Recall(k=50): 0.153711 - NDCG(k=20): 0.080909 - NDCG(k=50): 0.098418 - HitRate(k=20): 0.398922 - HitRate(k=50): 0.524617
2021-01-15 17:11:32,998 P49679 INFO Save best model: monitor(max): 0.080909
2021-01-15 17:11:33,023 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 17:11:33,072 P49679 INFO Train loss: 0.486870
2021-01-15 17:11:33,072 P49679 INFO ************ Epoch=1 end ************
2021-01-15 17:11:33,073 P49679 INFO Negative sampling num_negs=2000
2021-01-15 17:12:08,782 P49679 INFO Negative sampling done
2021-01-15 17:15:56,290 P49679 INFO --- Start evaluation ---
2021-01-15 17:15:58,240 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 17:16:24,838 P49679 INFO [Metrics] Recall(k=20): 0.125913 - Recall(k=50): 0.198700 - NDCG(k=20): 0.104625 - NDCG(k=50): 0.127127 - HitRate(k=20): 0.473273 - HitRate(k=50): 0.606705
2021-01-15 17:16:24,849 P49679 INFO Save best model: monitor(max): 0.104625
2021-01-15 17:16:24,884 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 17:16:24,954 P49679 INFO Train loss: 0.185197
2021-01-15 17:16:24,955 P49679 INFO ************ Epoch=2 end ************
2021-01-15 17:16:24,955 P49679 INFO Negative sampling num_negs=2000
2021-01-15 17:16:58,728 P49679 INFO Negative sampling done
2021-01-15 17:20:53,430 P49679 INFO --- Start evaluation ---
2021-01-15 17:20:55,342 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 17:21:20,422 P49679 INFO [Metrics] Recall(k=20): 0.139886 - Recall(k=50): 0.219085 - NDCG(k=20): 0.115757 - NDCG(k=50): 0.140422 - HitRate(k=20): 0.503383 - HitRate(k=50): 0.638589
2021-01-15 17:21:20,433 P49679 INFO Save best model: monitor(max): 0.115757
2021-01-15 17:21:20,466 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 17:21:20,513 P49679 INFO Train loss: 0.171702
2021-01-15 17:21:20,513 P49679 INFO ************ Epoch=3 end ************
2021-01-15 17:21:20,514 P49679 INFO Negative sampling num_negs=2000
2021-01-15 17:21:53,024 P49679 INFO Negative sampling done
2021-01-15 17:25:44,245 P49679 INFO --- Start evaluation ---
2021-01-15 17:25:45,302 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 17:26:12,611 P49679 INFO [Metrics] Recall(k=20): 0.149454 - Recall(k=50): 0.234229 - NDCG(k=20): 0.123901 - NDCG(k=50): 0.150278 - HitRate(k=20): 0.524014 - HitRate(k=50): 0.660493
2021-01-15 17:26:12,620 P49679 INFO Save best model: monitor(max): 0.123901
2021-01-15 17:26:12,653 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 17:26:12,705 P49679 INFO Train loss: 0.162956
2021-01-15 17:26:12,705 P49679 INFO ************ Epoch=4 end ************
2021-01-15 17:26:12,706 P49679 INFO Negative sampling num_negs=2000
2021-01-15 17:26:46,198 P49679 INFO Negative sampling done
2021-01-15 17:30:28,796 P49679 INFO --- Start evaluation ---
2021-01-15 17:30:30,971 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 17:30:57,096 P49679 INFO [Metrics] Recall(k=20): 0.156501 - Recall(k=50): 0.244043 - NDCG(k=20): 0.128356 - NDCG(k=50): 0.155551 - HitRate(k=20): 0.536908 - HitRate(k=50): 0.674325
2021-01-15 17:30:57,106 P49679 INFO Save best model: monitor(max): 0.128356
2021-01-15 17:30:57,139 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 17:30:57,185 P49679 INFO Train loss: 0.156764
2021-01-15 17:30:57,185 P49679 INFO ************ Epoch=5 end ************
2021-01-15 17:30:57,186 P49679 INFO Negative sampling num_negs=2000
2021-01-15 17:31:31,305 P49679 INFO Negative sampling done
2021-01-15 17:35:10,209 P49679 INFO --- Start evaluation ---
2021-01-15 17:35:12,468 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 17:35:36,030 P49679 INFO [Metrics] Recall(k=20): 0.161708 - Recall(k=50): 0.251414 - NDCG(k=20): 0.133402 - NDCG(k=50): 0.161243 - HitRate(k=20): 0.546219 - HitRate(k=50): 0.685177
2021-01-15 17:35:36,039 P49679 INFO Save best model: monitor(max): 0.133402
2021-01-15 17:35:36,072 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 17:35:36,132 P49679 INFO Train loss: 0.152322
2021-01-15 17:35:36,132 P49679 INFO ************ Epoch=6 end ************
2021-01-15 17:35:36,133 P49679 INFO Negative sampling num_negs=2000
2021-01-15 17:36:09,492 P49679 INFO Negative sampling done
2021-01-15 17:39:45,690 P49679 INFO --- Start evaluation ---
2021-01-15 17:39:47,170 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 17:40:11,100 P49679 INFO [Metrics] Recall(k=20): 0.165597 - Recall(k=50): 0.257199 - NDCG(k=20): 0.136293 - NDCG(k=50): 0.164698 - HitRate(k=20): 0.553386 - HitRate(k=50): 0.690468
2021-01-15 17:40:11,116 P49679 INFO Save best model: monitor(max): 0.136293
2021-01-15 17:40:11,151 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 17:40:11,234 P49679 INFO Train loss: 0.148894
2021-01-15 17:40:11,235 P49679 INFO ************ Epoch=7 end ************
2021-01-15 17:40:11,236 P49679 INFO Negative sampling num_negs=2000
2021-01-15 17:40:46,400 P49679 INFO Negative sampling done
2021-01-15 17:44:19,766 P49679 INFO --- Start evaluation ---
2021-01-15 17:44:21,339 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 17:44:47,049 P49679 INFO [Metrics] Recall(k=20): 0.167824 - Recall(k=50): 0.260890 - NDCG(k=20): 0.137886 - NDCG(k=50): 0.166821 - HitRate(k=20): 0.556836 - HitRate(k=50): 0.695693
2021-01-15 17:44:47,068 P49679 INFO Save best model: monitor(max): 0.137886
2021-01-15 17:44:47,107 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 17:44:47,158 P49679 INFO Train loss: 0.146291
2021-01-15 17:44:47,158 P49679 INFO ************ Epoch=8 end ************
2021-01-15 17:44:47,159 P49679 INFO Negative sampling num_negs=2000
2021-01-15 17:45:22,265 P49679 INFO Negative sampling done
2021-01-15 17:48:54,460 P49679 INFO --- Start evaluation ---
2021-01-15 17:48:56,688 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 17:49:21,212 P49679 INFO [Metrics] Recall(k=20): 0.170697 - Recall(k=50): 0.264183 - NDCG(k=20): 0.140389 - NDCG(k=50): 0.169485 - HitRate(k=20): 0.564706 - HitRate(k=50): 0.700650
2021-01-15 17:49:21,224 P49679 INFO Save best model: monitor(max): 0.140389
2021-01-15 17:49:21,258 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 17:49:21,315 P49679 INFO Train loss: 0.144255
2021-01-15 17:49:21,315 P49679 INFO ************ Epoch=9 end ************
2021-01-15 17:49:21,316 P49679 INFO Negative sampling num_negs=2000
2021-01-15 17:49:56,546 P49679 INFO Negative sampling done
2021-01-15 17:53:28,780 P49679 INFO --- Start evaluation ---
2021-01-15 17:53:30,971 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 17:53:54,999 P49679 INFO [Metrics] Recall(k=20): 0.172090 - Recall(k=50): 0.266368 - NDCG(k=20): 0.141612 - NDCG(k=50): 0.170978 - HitRate(k=20): 0.566046 - HitRate(k=50): 0.702190
2021-01-15 17:53:55,008 P49679 INFO Save best model: monitor(max): 0.141612
2021-01-15 17:53:55,040 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 17:53:55,112 P49679 INFO Train loss: 0.142489
2021-01-15 17:53:55,112 P49679 INFO ************ Epoch=10 end ************
2021-01-15 17:53:55,113 P49679 INFO Negative sampling num_negs=2000
2021-01-15 17:54:30,893 P49679 INFO Negative sampling done
2021-01-15 17:58:04,318 P49679 INFO --- Start evaluation ---
2021-01-15 17:58:06,681 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 17:58:30,452 P49679 INFO [Metrics] Recall(k=20): 0.174130 - Recall(k=50): 0.268793 - NDCG(k=20): 0.143498 - NDCG(k=50): 0.172962 - HitRate(k=20): 0.570567 - HitRate(k=50): 0.705305
2021-01-15 17:58:30,463 P49679 INFO Save best model: monitor(max): 0.143498
2021-01-15 17:58:30,496 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 17:58:30,603 P49679 INFO Train loss: 0.141114
2021-01-15 17:58:30,603 P49679 INFO ************ Epoch=11 end ************
2021-01-15 17:58:30,604 P49679 INFO Negative sampling num_negs=2000
2021-01-15 17:59:05,930 P49679 INFO Negative sampling done
2021-01-15 18:02:38,524 P49679 INFO --- Start evaluation ---
2021-01-15 18:02:40,771 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 18:03:06,643 P49679 INFO [Metrics] Recall(k=20): 0.175738 - Recall(k=50): 0.270159 - NDCG(k=20): 0.145049 - NDCG(k=50): 0.174414 - HitRate(k=20): 0.573548 - HitRate(k=50): 0.708855
2021-01-15 18:03:06,654 P49679 INFO Save best model: monitor(max): 0.145049
2021-01-15 18:03:06,687 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 18:03:06,740 P49679 INFO Train loss: 0.139953
2021-01-15 18:03:06,740 P49679 INFO ************ Epoch=12 end ************
2021-01-15 18:03:06,741 P49679 INFO Negative sampling num_negs=2000
2021-01-15 18:03:42,743 P49679 INFO Negative sampling done
2021-01-15 18:07:04,038 P49679 INFO --- Start evaluation ---
2021-01-15 18:07:06,358 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 18:07:31,044 P49679 INFO [Metrics] Recall(k=20): 0.176974 - Recall(k=50): 0.271712 - NDCG(k=20): 0.146269 - NDCG(k=50): 0.175655 - HitRate(k=20): 0.576428 - HitRate(k=50): 0.710061
2021-01-15 18:07:31,053 P49679 INFO Save best model: monitor(max): 0.146269
2021-01-15 18:07:31,086 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 18:07:31,133 P49679 INFO Train loss: 0.138826
2021-01-15 18:07:31,133 P49679 INFO ************ Epoch=13 end ************
2021-01-15 18:07:31,134 P49679 INFO Negative sampling num_negs=2000
2021-01-15 18:08:06,044 P49679 INFO Negative sampling done
2021-01-15 18:11:11,443 P49679 INFO --- Start evaluation ---
2021-01-15 18:11:14,331 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 18:11:38,325 P49679 INFO [Metrics] Recall(k=20): 0.177052 - Recall(k=50): 0.272756 - NDCG(k=20): 0.146154 - NDCG(k=50): 0.175958 - HitRate(k=20): 0.574687 - HitRate(k=50): 0.710429
2021-01-15 18:11:38,334 P49679 INFO Monitor(max) STOP: 0.146154 !
2021-01-15 18:11:38,334 P49679 INFO Reduce learning rate on plateau: 0.000010
2021-01-15 18:11:38,334 P49679 INFO Load best model:  SimpleX_gowalla_x0_013_4ecb0cbe.model
2021-01-15 18:11:38,359 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 18:11:38,412 P49679 INFO Train loss: 0.137912
2021-01-15 18:11:38,412 P49679 INFO ************ Epoch=14 end ************
2021-01-15 18:11:38,413 P49679 INFO Negative sampling num_negs=2000
2021-01-15 18:12:11,462 P49679 INFO Negative sampling done
2021-01-15 18:14:51,185 P49679 INFO --- Start evaluation ---
2021-01-15 18:14:53,468 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 18:15:18,979 P49679 INFO [Metrics] Recall(k=20): 0.182526 - Recall(k=50): 0.280957 - NDCG(k=20): 0.151674 - NDCG(k=50): 0.182238 - HitRate(k=20): 0.584935 - HitRate(k=50): 0.719740
2021-01-15 18:15:18,991 P49679 INFO Save best model: monitor(max): 0.151674
2021-01-15 18:15:19,023 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 18:15:19,071 P49679 INFO Train loss: 0.129693
2021-01-15 18:15:19,071 P49679 INFO ************ Epoch=15 end ************
2021-01-15 18:15:19,072 P49679 INFO Negative sampling num_negs=2000
2021-01-15 18:15:51,958 P49679 INFO Negative sampling done
2021-01-15 18:18:24,306 P49679 INFO --- Start evaluation ---
2021-01-15 18:18:26,623 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 18:18:50,185 P49679 INFO [Metrics] Recall(k=20): 0.183340 - Recall(k=50): 0.282618 - NDCG(k=20): 0.152803 - NDCG(k=50): 0.183692 - HitRate(k=20): 0.586007 - HitRate(k=50): 0.720343
2021-01-15 18:18:50,195 P49679 INFO Save best model: monitor(max): 0.152803
2021-01-15 18:18:50,228 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 18:18:50,283 P49679 INFO Train loss: 0.128099
2021-01-15 18:18:50,283 P49679 INFO ************ Epoch=16 end ************
2021-01-15 18:18:50,284 P49679 INFO Negative sampling num_negs=2000
2021-01-15 18:19:23,369 P49679 INFO Negative sampling done
2021-01-15 18:22:22,939 P49679 INFO --- Start evaluation ---
2021-01-15 18:22:24,008 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 18:22:47,166 P49679 INFO [Metrics] Recall(k=20): 0.184192 - Recall(k=50): 0.284230 - NDCG(k=20): 0.153423 - NDCG(k=50): 0.184553 - HitRate(k=20): 0.587213 - HitRate(k=50): 0.723759
2021-01-15 18:22:47,177 P49679 INFO Save best model: monitor(max): 0.153423
2021-01-15 18:22:47,209 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 18:22:47,263 P49679 INFO Train loss: 0.127381
2021-01-15 18:22:47,263 P49679 INFO ************ Epoch=17 end ************
2021-01-15 18:22:47,264 P49679 INFO Negative sampling num_negs=2000
2021-01-15 18:23:20,754 P49679 INFO Negative sampling done
2021-01-15 18:26:51,916 P49679 INFO --- Start evaluation ---
2021-01-15 18:26:53,028 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 18:27:17,202 P49679 INFO [Metrics] Recall(k=20): 0.184723 - Recall(k=50): 0.285168 - NDCG(k=20): 0.153850 - NDCG(k=50): 0.185133 - HitRate(k=20): 0.588586 - HitRate(k=50): 0.724730
2021-01-15 18:27:17,212 P49679 INFO Save best model: monitor(max): 0.153850
2021-01-15 18:27:17,245 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 18:27:17,300 P49679 INFO Train loss: 0.126713
2021-01-15 18:27:17,300 P49679 INFO ************ Epoch=18 end ************
2021-01-15 18:27:17,301 P49679 INFO Negative sampling num_negs=2000
2021-01-15 18:27:51,455 P49679 INFO Negative sampling done
2021-01-15 18:31:30,365 P49679 INFO --- Start evaluation ---
2021-01-15 18:31:31,551 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 18:31:59,223 P49679 INFO [Metrics] Recall(k=20): 0.185439 - Recall(k=50): 0.285841 - NDCG(k=20): 0.154231 - NDCG(k=50): 0.185448 - HitRate(k=20): 0.589490 - HitRate(k=50): 0.725769
2021-01-15 18:31:59,255 P49679 INFO Save best model: monitor(max): 0.154231
2021-01-15 18:31:59,294 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 18:31:59,555 P49679 INFO Train loss: 0.126345
2021-01-15 18:31:59,556 P49679 INFO ************ Epoch=19 end ************
2021-01-15 18:31:59,557 P49679 INFO Negative sampling num_negs=2000
2021-01-15 18:32:32,610 P49679 INFO Negative sampling done
2021-01-15 18:36:10,512 P49679 INFO --- Start evaluation ---
2021-01-15 18:36:12,490 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 18:36:38,450 P49679 INFO [Metrics] Recall(k=20): 0.185483 - Recall(k=50): 0.286248 - NDCG(k=20): 0.154255 - NDCG(k=50): 0.185541 - HitRate(k=20): 0.588753 - HitRate(k=50): 0.726003
2021-01-15 18:36:38,465 P49679 INFO Save best model: monitor(max): 0.154255
2021-01-15 18:36:38,500 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 18:36:38,554 P49679 INFO Train loss: 0.126032
2021-01-15 18:36:38,554 P49679 INFO ************ Epoch=20 end ************
2021-01-15 18:36:38,555 P49679 INFO Negative sampling num_negs=2000
2021-01-15 18:37:12,941 P49679 INFO Negative sampling done
2021-01-15 18:40:48,811 P49679 INFO --- Start evaluation ---
2021-01-15 18:40:51,403 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 18:41:16,333 P49679 INFO [Metrics] Recall(k=20): 0.186037 - Recall(k=50): 0.286448 - NDCG(k=20): 0.154821 - NDCG(k=50): 0.186013 - HitRate(k=20): 0.590361 - HitRate(k=50): 0.726037
2021-01-15 18:41:16,342 P49679 INFO Save best model: monitor(max): 0.154821
2021-01-15 18:41:16,375 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 18:41:16,423 P49679 INFO Train loss: 0.125773
2021-01-15 18:41:16,423 P49679 INFO ************ Epoch=21 end ************
2021-01-15 18:41:16,424 P49679 INFO Negative sampling num_negs=2000
2021-01-15 18:41:52,184 P49679 INFO Negative sampling done
2021-01-15 18:45:29,365 P49679 INFO --- Start evaluation ---
2021-01-15 18:45:31,768 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 18:45:55,469 P49679 INFO [Metrics] Recall(k=20): 0.186440 - Recall(k=50): 0.286275 - NDCG(k=20): 0.155032 - NDCG(k=50): 0.186085 - HitRate(k=20): 0.591366 - HitRate(k=50): 0.725936
2021-01-15 18:45:55,482 P49679 INFO Save best model: monitor(max): 0.155032
2021-01-15 18:45:55,519 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 18:45:55,571 P49679 INFO Train loss: 0.125464
2021-01-15 18:45:55,572 P49679 INFO ************ Epoch=22 end ************
2021-01-15 18:45:55,572 P49679 INFO Negative sampling num_negs=2000
2021-01-15 18:46:31,215 P49679 INFO Negative sampling done
2021-01-15 18:50:09,670 P49679 INFO --- Start evaluation ---
2021-01-15 18:50:12,212 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 18:50:37,681 P49679 INFO [Metrics] Recall(k=20): 0.186735 - Recall(k=50): 0.286672 - NDCG(k=20): 0.155318 - NDCG(k=50): 0.186334 - HitRate(k=20): 0.591165 - HitRate(k=50): 0.726070
2021-01-15 18:50:37,691 P49679 INFO Save best model: monitor(max): 0.155318
2021-01-15 18:50:37,725 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 18:50:37,771 P49679 INFO Train loss: 0.125256
2021-01-15 18:50:37,771 P49679 INFO ************ Epoch=23 end ************
2021-01-15 18:50:37,772 P49679 INFO Negative sampling num_negs=2000
2021-01-15 18:51:13,674 P49679 INFO Negative sampling done
2021-01-15 18:54:42,532 P49679 INFO --- Start evaluation ---
2021-01-15 18:54:44,460 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 18:55:08,450 P49679 INFO [Metrics] Recall(k=20): 0.186654 - Recall(k=50): 0.287023 - NDCG(k=20): 0.155116 - NDCG(k=50): 0.186345 - HitRate(k=20): 0.590964 - HitRate(k=50): 0.726840
2021-01-15 18:55:08,459 P49679 INFO Monitor(max) STOP: 0.155116 !
2021-01-15 18:55:08,459 P49679 INFO Reduce learning rate on plateau: 0.000001
2021-01-15 18:55:08,459 P49679 INFO Load best model:  SimpleX_gowalla_x0_013_4ecb0cbe.model
2021-01-15 18:55:08,485 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 18:55:08,536 P49679 INFO Train loss: 0.125131
2021-01-15 18:55:08,536 P49679 INFO ************ Epoch=24 end ************
2021-01-15 18:55:08,537 P49679 INFO Negative sampling num_negs=2000
2021-01-15 18:55:43,622 P49679 INFO Negative sampling done
2021-01-15 18:59:13,687 P49679 INFO --- Start evaluation ---
2021-01-15 18:59:15,593 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 18:59:40,588 P49679 INFO [Metrics] Recall(k=20): 0.186719 - Recall(k=50): 0.286768 - NDCG(k=20): 0.155322 - NDCG(k=50): 0.186404 - HitRate(k=20): 0.591165 - HitRate(k=50): 0.726438
2021-01-15 18:59:40,596 P49679 INFO Save best model: monitor(max): 0.155322
2021-01-15 18:59:40,630 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 18:59:40,684 P49679 INFO Train loss: 0.124131
2021-01-15 18:59:40,684 P49679 INFO ************ Epoch=25 end ************
2021-01-15 18:59:40,685 P49679 INFO Negative sampling num_negs=2000
2021-01-15 19:00:15,525 P49679 INFO Negative sampling done
2021-01-15 19:03:47,389 P49679 INFO --- Start evaluation ---
2021-01-15 19:03:49,208 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 19:04:13,943 P49679 INFO [Metrics] Recall(k=20): 0.186702 - Recall(k=50): 0.286811 - NDCG(k=20): 0.155371 - NDCG(k=50): 0.186454 - HitRate(k=20): 0.591366 - HitRate(k=50): 0.726104
2021-01-15 19:04:13,959 P49679 INFO Save best model: monitor(max): 0.155371
2021-01-15 19:04:13,992 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 19:04:14,052 P49679 INFO Train loss: 0.124006
2021-01-15 19:04:14,052 P49679 INFO ************ Epoch=26 end ************
2021-01-15 19:04:14,053 P49679 INFO Negative sampling num_negs=2000
2021-01-15 19:04:47,918 P49679 INFO Negative sampling done
2021-01-15 19:08:17,213 P49679 INFO --- Start evaluation ---
2021-01-15 19:08:19,470 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 19:08:44,042 P49679 INFO [Metrics] Recall(k=20): 0.186746 - Recall(k=50): 0.286726 - NDCG(k=20): 0.155427 - NDCG(k=50): 0.186476 - HitRate(k=20): 0.591768 - HitRate(k=50): 0.726104
2021-01-15 19:08:44,052 P49679 INFO Save best model: monitor(max): 0.155427
2021-01-15 19:08:44,085 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 19:08:44,132 P49679 INFO Train loss: 0.124088
2021-01-15 19:08:44,132 P49679 INFO ************ Epoch=27 end ************
2021-01-15 19:08:44,133 P49679 INFO Negative sampling num_negs=2000
2021-01-15 19:09:17,635 P49679 INFO Negative sampling done
2021-01-15 19:12:45,344 P49679 INFO --- Start evaluation ---
2021-01-15 19:12:47,528 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 19:13:12,607 P49679 INFO [Metrics] Recall(k=20): 0.186881 - Recall(k=50): 0.286928 - NDCG(k=20): 0.155478 - NDCG(k=50): 0.186549 - HitRate(k=20): 0.591734 - HitRate(k=50): 0.726305
2021-01-15 19:13:12,616 P49679 INFO Save best model: monitor(max): 0.155478
2021-01-15 19:13:12,648 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 19:13:12,703 P49679 INFO Train loss: 0.124040
2021-01-15 19:13:12,703 P49679 INFO ************ Epoch=28 end ************
2021-01-15 19:13:12,704 P49679 INFO Negative sampling num_negs=2000
2021-01-15 19:13:45,898 P49679 INFO Negative sampling done
2021-01-15 19:16:22,677 P49679 INFO --- Start evaluation ---
2021-01-15 19:16:24,672 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 19:16:49,335 P49679 INFO [Metrics] Recall(k=20): 0.186756 - Recall(k=50): 0.286937 - NDCG(k=20): 0.155462 - NDCG(k=50): 0.186587 - HitRate(k=20): 0.591667 - HitRate(k=50): 0.726405
2021-01-15 19:16:49,344 P49679 INFO Monitor(max) STOP: 0.155462 !
2021-01-15 19:16:49,344 P49679 INFO Reduce learning rate on plateau: 0.000001
2021-01-15 19:16:49,344 P49679 INFO Load best model:  SimpleX_gowalla_x0_013_4ecb0cbe.model
2021-01-15 19:16:49,372 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 19:16:49,440 P49679 INFO Train loss: 0.124001
2021-01-15 19:16:49,440 P49679 INFO ************ Epoch=29 end ************
2021-01-15 19:16:49,440 P49679 INFO Negative sampling num_negs=2000
2021-01-15 19:17:23,229 P49679 INFO Negative sampling done
2021-01-15 19:20:01,965 P49679 INFO --- Start evaluation ---
2021-01-15 19:20:03,964 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 19:20:27,009 P49679 INFO [Metrics] Recall(k=20): 0.186747 - Recall(k=50): 0.287018 - NDCG(k=20): 0.155497 - NDCG(k=50): 0.186643 - HitRate(k=20): 0.591701 - HitRate(k=50): 0.726706
2021-01-15 19:20:27,018 P49679 INFO Save best model: monitor(max): 0.155497
2021-01-15 19:20:27,056 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 19:20:27,125 P49679 INFO Train loss: 0.124048
2021-01-15 19:20:27,125 P49679 INFO ************ Epoch=30 end ************
2021-01-15 19:20:27,126 P49679 INFO Negative sampling num_negs=2000
2021-01-15 19:20:59,319 P49679 INFO Negative sampling done
2021-01-15 19:23:38,626 P49679 INFO --- Start evaluation ---
2021-01-15 19:23:40,644 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 19:24:03,899 P49679 INFO [Metrics] Recall(k=20): 0.186736 - Recall(k=50): 0.287008 - NDCG(k=20): 0.155369 - NDCG(k=50): 0.186532 - HitRate(k=20): 0.591734 - HitRate(k=50): 0.726673
2021-01-15 19:24:03,909 P49679 INFO Monitor(max) STOP: 0.155369 !
2021-01-15 19:24:03,909 P49679 INFO Reduce learning rate on plateau: 0.000001
2021-01-15 19:24:03,909 P49679 INFO Load best model:  SimpleX_gowalla_x0_013_4ecb0cbe.model
2021-01-15 19:24:03,931 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 19:24:03,986 P49679 INFO Train loss: 0.123932
2021-01-15 19:24:03,987 P49679 INFO ************ Epoch=31 end ************
2021-01-15 19:24:03,987 P49679 INFO Negative sampling num_negs=2000
2021-01-15 19:24:36,375 P49679 INFO Negative sampling done
2021-01-15 19:27:14,592 P49679 INFO --- Start evaluation ---
2021-01-15 19:27:16,523 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 19:27:40,146 P49679 INFO [Metrics] Recall(k=20): 0.186790 - Recall(k=50): 0.287177 - NDCG(k=20): 0.155463 - NDCG(k=50): 0.186631 - HitRate(k=20): 0.591399 - HitRate(k=50): 0.726907
2021-01-15 19:27:40,155 P49679 INFO Monitor(max) STOP: 0.155463 !
2021-01-15 19:27:40,155 P49679 INFO Reduce learning rate on plateau: 0.000001
2021-01-15 19:27:40,155 P49679 INFO Load best model:  SimpleX_gowalla_x0_013_4ecb0cbe.model
2021-01-15 19:27:40,183 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 19:27:40,239 P49679 INFO Train loss: 0.123971
2021-01-15 19:27:40,239 P49679 INFO ************ Epoch=32 end ************
2021-01-15 19:27:40,239 P49679 INFO Negative sampling num_negs=2000
2021-01-15 19:28:14,641 P49679 INFO Negative sampling done
2021-01-15 19:30:52,711 P49679 INFO --- Start evaluation ---
2021-01-15 19:30:54,757 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 19:31:18,874 P49679 INFO [Metrics] Recall(k=20): 0.186818 - Recall(k=50): 0.286878 - NDCG(k=20): 0.155501 - NDCG(k=50): 0.186602 - HitRate(k=20): 0.591634 - HitRate(k=50): 0.726673
2021-01-15 19:31:18,882 P49679 INFO Save best model: monitor(max): 0.155501
2021-01-15 19:31:18,916 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 19:31:18,975 P49679 INFO Train loss: 0.123942
2021-01-15 19:31:18,975 P49679 INFO ************ Epoch=33 end ************
2021-01-15 19:31:18,976 P49679 INFO Negative sampling num_negs=2000
2021-01-15 19:31:51,691 P49679 INFO Negative sampling done
2021-01-15 19:34:29,958 P49679 INFO --- Start evaluation ---
2021-01-15 19:34:31,904 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 19:34:55,218 P49679 INFO [Metrics] Recall(k=20): 0.186951 - Recall(k=50): 0.287159 - NDCG(k=20): 0.155570 - NDCG(k=50): 0.186697 - HitRate(k=20): 0.592069 - HitRate(k=50): 0.726606
2021-01-15 19:34:55,226 P49679 INFO Save best model: monitor(max): 0.155570
2021-01-15 19:34:55,259 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 19:34:55,317 P49679 INFO Train loss: 0.123879
2021-01-15 19:34:55,317 P49679 INFO ************ Epoch=34 end ************
2021-01-15 19:34:55,318 P49679 INFO Negative sampling num_negs=2000
2021-01-15 19:35:28,740 P49679 INFO Negative sampling done
2021-01-15 19:38:05,235 P49679 INFO --- Start evaluation ---
2021-01-15 19:38:07,234 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 19:38:30,502 P49679 INFO [Metrics] Recall(k=20): 0.186996 - Recall(k=50): 0.287178 - NDCG(k=20): 0.155613 - NDCG(k=50): 0.186728 - HitRate(k=20): 0.592203 - HitRate(k=50): 0.727376
2021-01-15 19:38:30,513 P49679 INFO Save best model: monitor(max): 0.155613
2021-01-15 19:38:30,545 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 19:38:30,603 P49679 INFO Train loss: 0.123893
2021-01-15 19:38:30,603 P49679 INFO ************ Epoch=35 end ************
2021-01-15 19:38:30,603 P49679 INFO Negative sampling num_negs=2000
2021-01-15 19:39:03,596 P49679 INFO Negative sampling done
2021-01-15 19:41:36,668 P49679 INFO --- Start evaluation ---
2021-01-15 19:41:38,622 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 19:42:04,072 P49679 INFO [Metrics] Recall(k=20): 0.186994 - Recall(k=50): 0.287161 - NDCG(k=20): 0.155589 - NDCG(k=50): 0.186731 - HitRate(k=20): 0.592002 - HitRate(k=50): 0.727142
2021-01-15 19:42:04,083 P49679 INFO Monitor(max) STOP: 0.155589 !
2021-01-15 19:42:04,083 P49679 INFO Reduce learning rate on plateau: 0.000001
2021-01-15 19:42:04,083 P49679 INFO Load best model:  SimpleX_gowalla_x0_013_4ecb0cbe.model
2021-01-15 19:42:04,111 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 19:42:04,166 P49679 INFO Train loss: 0.123864
2021-01-15 19:42:04,166 P49679 INFO ************ Epoch=36 end ************
2021-01-15 19:42:04,167 P49679 INFO Negative sampling num_negs=2000
2021-01-15 19:42:36,472 P49679 INFO Negative sampling done
2021-01-15 19:45:06,875 P49679 INFO --- Start evaluation ---
2021-01-15 19:45:08,747 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 19:45:31,922 P49679 INFO [Metrics] Recall(k=20): 0.186914 - Recall(k=50): 0.287283 - NDCG(k=20): 0.155620 - NDCG(k=50): 0.186804 - HitRate(k=20): 0.592170 - HitRate(k=50): 0.727075
2021-01-15 19:45:31,930 P49679 INFO Save best model: monitor(max): 0.155620
2021-01-15 19:45:31,963 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 19:45:32,023 P49679 INFO Train loss: 0.123795
2021-01-15 19:45:32,023 P49679 INFO ************ Epoch=37 end ************
2021-01-15 19:45:32,023 P49679 INFO Negative sampling num_negs=2000
2021-01-15 19:46:04,119 P49679 INFO Negative sampling done
2021-01-15 19:48:30,221 P49679 INFO --- Start evaluation ---
2021-01-15 19:48:32,141 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 19:48:55,413 P49679 INFO [Metrics] Recall(k=20): 0.187213 - Recall(k=50): 0.287394 - NDCG(k=20): 0.155688 - NDCG(k=50): 0.186816 - HitRate(k=20): 0.592705 - HitRate(k=50): 0.727242
2021-01-15 19:48:55,421 P49679 INFO Save best model: monitor(max): 0.155688
2021-01-15 19:48:55,453 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 19:48:55,508 P49679 INFO Train loss: 0.123847
2021-01-15 19:48:55,508 P49679 INFO ************ Epoch=38 end ************
2021-01-15 19:48:55,508 P49679 INFO Negative sampling num_negs=2000
2021-01-15 19:49:28,081 P49679 INFO Negative sampling done
2021-01-15 19:51:55,394 P49679 INFO --- Start evaluation ---
2021-01-15 19:51:57,430 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 19:52:20,743 P49679 INFO [Metrics] Recall(k=20): 0.187205 - Recall(k=50): 0.287341 - NDCG(k=20): 0.155680 - NDCG(k=50): 0.186794 - HitRate(k=20): 0.592572 - HitRate(k=50): 0.727477
2021-01-15 19:52:20,752 P49679 INFO Monitor(max) STOP: 0.155680 !
2021-01-15 19:52:20,752 P49679 INFO Reduce learning rate on plateau: 0.000001
2021-01-15 19:52:20,752 P49679 INFO Load best model:  SimpleX_gowalla_x0_013_4ecb0cbe.model
2021-01-15 19:52:20,779 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 19:52:20,839 P49679 INFO Train loss: 0.123752
2021-01-15 19:52:20,839 P49679 INFO ************ Epoch=39 end ************
2021-01-15 19:52:20,839 P49679 INFO Negative sampling num_negs=2000
2021-01-15 19:52:53,043 P49679 INFO Negative sampling done
2021-01-15 19:55:15,793 P49679 INFO --- Start evaluation ---
2021-01-15 19:55:17,280 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 19:55:40,141 P49679 INFO [Metrics] Recall(k=20): 0.187137 - Recall(k=50): 0.287387 - NDCG(k=20): 0.155678 - NDCG(k=50): 0.186835 - HitRate(k=20): 0.592371 - HitRate(k=50): 0.727611
2021-01-15 19:55:40,149 P49679 INFO Monitor(max) STOP: 0.155678 !
2021-01-15 19:55:40,150 P49679 INFO Reduce learning rate on plateau: 0.000001
2021-01-15 19:55:40,150 P49679 INFO Load best model:  SimpleX_gowalla_x0_013_4ecb0cbe.model
2021-01-15 19:55:40,170 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 19:55:40,223 P49679 INFO Train loss: 0.123747
2021-01-15 19:55:40,223 P49679 INFO ************ Epoch=40 end ************
2021-01-15 19:55:40,223 P49679 INFO Negative sampling num_negs=2000
2021-01-15 19:56:12,965 P49679 INFO Negative sampling done
2021-01-15 19:57:57,093 P49679 INFO --- Start evaluation ---
2021-01-15 19:57:58,564 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 19:58:22,698 P49679 INFO [Metrics] Recall(k=20): 0.187149 - Recall(k=50): 0.287348 - NDCG(k=20): 0.155638 - NDCG(k=50): 0.186768 - HitRate(k=20): 0.592572 - HitRate(k=50): 0.727510
2021-01-15 19:58:22,707 P49679 INFO Monitor(max) STOP: 0.155638 !
2021-01-15 19:58:22,707 P49679 INFO Reduce learning rate on plateau: 0.000001
2021-01-15 19:58:22,707 P49679 INFO Load best model:  SimpleX_gowalla_x0_013_4ecb0cbe.model
2021-01-15 19:58:22,726 P49679 INFO Early stopping at epoch=41
2021-01-15 19:58:22,726 P49679 INFO --- 3165/3165 batches finished ---
2021-01-15 19:58:22,777 P49679 INFO Train loss: 0.123754
2021-01-15 19:58:22,777 P49679 INFO Training finished.
2021-01-15 19:58:22,777 P49679 INFO Load best model:  SimpleX_gowalla_x0_013_4ecb0cbe.model
2021-01-15 19:58:22,811 P49679 INFO ****** Train/validation evaluation ******
2021-01-15 19:58:22,811 P49679 INFO --- Start evaluation ---
2021-01-15 19:58:24,257 P49679 INFO Evaluating metrics for 29858 users...
2021-01-15 19:58:48,489 P49679 INFO [Metrics] Recall(k=20): 0.187213 - Recall(k=50): 0.287394 - NDCG(k=20): 0.155688 - NDCG(k=50): 0.186816 - HitRate(k=20): 0.592705 - HitRate(k=50): 0.727242
```