## SimpleX_amazonbooks_x0 

A notebook to benchmark SimpleX on amazonbooks_x0 dataset.

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
2021-01-14 14:57:03,807 P50829 INFO [Metrics] Recall(k=20): 0.058341 - Recall(k=50): 0.100549 - NDCG(k=20): 0.046792 - NDCG(k=50): 0.062420 - HitRate(k=20): 0.303007 - HitRate(k=50): 0.455727
```


### Logs
```
2021-01-14 10:28:10,672 P50829 INFO Set up feature encoder...
2021-01-14 10:28:10,672 P50829 INFO Load feature_map from json: ../data/AmazonBooks/amazonbooks_x0_fea71f7d/feature_map.json
2021-01-14 10:28:14,431 P50829 INFO Total number of parameters: 9235776.
2021-01-14 10:28:14,431 P50829 INFO Loading data...
2021-01-14 10:28:14,435 P50829 INFO Loading data from h5: ../data/AmazonBooks/amazonbooks_x0_fea71f7d/train.h5
2021-01-14 10:28:16,847 P50829 INFO Loading data from h5: ../data/AmazonBooks/amazonbooks_x0_fea71f7d/item_corpus.h5
2021-01-14 10:28:17,803 P50829 INFO Loading data from h5: ../data/AmazonBooks/amazonbooks_x0_fea71f7d/valid.h5
2021-01-14 10:28:18,725 P50829 INFO Loading data from h5: ../data/AmazonBooks/amazonbooks_x0_fea71f7d/item_corpus.h5
2021-01-14 10:28:18,726 P50829 INFO Train samples: total/2380730, blocks/1
2021-01-14 10:28:18,727 P50829 INFO Validation samples: total/52639, blocks/1
2021-01-14 10:28:18,727 P50829 INFO Loading train data done.
2021-01-14 10:28:18,727 P50829 INFO **** Start training: 2325 batches/epoch ****
2021-01-14 10:28:18,728 P50829 INFO Negative sampling num_negs=1000
2021-01-14 10:28:58,654 P50829 INFO Negative sampling done
2021-01-14 10:32:15,933 P50829 INFO --- Start evaluation ---
2021-01-14 10:32:16,805 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 10:33:45,977 P50829 INFO [Metrics] Recall(k=20): 0.026131 - Recall(k=50): 0.050273 - NDCG(k=20): 0.020140 - NDCG(k=50): 0.029136 - HitRate(k=20): 0.164536 - HitRate(k=50): 0.283364
2021-01-14 10:33:45,995 P50829 INFO Save best model: monitor(max): 0.046271
2021-01-14 10:33:46,031 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 10:33:46,075 P50829 INFO Train loss: 0.728771
2021-01-14 10:33:46,076 P50829 INFO ************ Epoch=1 end ************
2021-01-14 10:33:46,076 P50829 INFO Negative sampling num_negs=1000
2021-01-14 10:34:30,978 P50829 INFO Negative sampling done
2021-01-14 10:37:47,236 P50829 INFO --- Start evaluation ---
2021-01-14 10:37:48,127 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 10:39:13,439 P50829 INFO [Metrics] Recall(k=20): 0.033136 - Recall(k=50): 0.062079 - NDCG(k=20): 0.025841 - NDCG(k=50): 0.036580 - HitRate(k=20): 0.197192 - HitRate(k=50): 0.327267
2021-01-14 10:39:13,456 P50829 INFO Save best model: monitor(max): 0.058976
2021-01-14 10:39:13,496 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 10:39:13,551 P50829 INFO Train loss: 0.658466
2021-01-14 10:39:13,551 P50829 INFO ************ Epoch=2 end ************
2021-01-14 10:39:13,552 P50829 INFO Negative sampling num_negs=1000
2021-01-14 10:39:58,383 P50829 INFO Negative sampling done
2021-01-14 10:45:14,732 P50829 INFO --- Start evaluation ---
2021-01-14 10:45:18,319 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 10:53:09,206 P50829 INFO [Metrics] Recall(k=20): 0.037184 - Recall(k=50): 0.068578 - NDCG(k=20): 0.028705 - NDCG(k=50): 0.040354 - HitRate(k=20): 0.215886 - HitRate(k=50): 0.351944
2021-01-14 10:53:09,228 P50829 INFO Save best model: monitor(max): 0.065889
2021-01-14 10:53:09,265 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 10:53:09,345 P50829 INFO Train loss: 0.652990
2021-01-14 10:53:09,345 P50829 INFO ************ Epoch=3 end ************
2021-01-14 10:53:09,345 P50829 INFO Negative sampling num_negs=1000
2021-01-14 10:53:49,705 P50829 INFO Negative sampling done
2021-01-14 10:58:56,936 P50829 INFO --- Start evaluation ---
2021-01-14 10:58:58,634 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 11:00:26,162 P50829 INFO [Metrics] Recall(k=20): 0.039530 - Recall(k=50): 0.073546 - NDCG(k=20): 0.030087 - NDCG(k=50): 0.042736 - HitRate(k=20): 0.223067 - HitRate(k=50): 0.365186
2021-01-14 11:00:26,183 P50829 INFO Save best model: monitor(max): 0.069617
2021-01-14 11:00:26,226 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 11:00:26,295 P50829 INFO Train loss: 0.652127
2021-01-14 11:00:26,295 P50829 INFO ************ Epoch=4 end ************
2021-01-14 11:00:26,295 P50829 INFO Negative sampling num_negs=1000
2021-01-14 11:01:15,080 P50829 INFO Negative sampling done
2021-01-14 11:05:41,434 P50829 INFO --- Start evaluation ---
2021-01-14 11:05:42,622 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 11:07:08,243 P50829 INFO [Metrics] Recall(k=20): 0.042961 - Recall(k=50): 0.078385 - NDCG(k=20): 0.033623 - NDCG(k=50): 0.046796 - HitRate(k=20): 0.240829 - HitRate(k=50): 0.385399
2021-01-14 11:07:08,265 P50829 INFO Save best model: monitor(max): 0.076584
2021-01-14 11:07:08,308 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 11:07:08,371 P50829 INFO Train loss: 0.652185
2021-01-14 11:07:08,372 P50829 INFO ************ Epoch=5 end ************
2021-01-14 11:07:08,372 P50829 INFO Negative sampling num_negs=1000
2021-01-14 11:07:50,734 P50829 INFO Negative sampling done
2021-01-14 11:12:53,118 P50829 INFO --- Start evaluation ---
2021-01-14 11:12:55,707 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 11:14:20,289 P50829 INFO [Metrics] Recall(k=20): 0.043725 - Recall(k=50): 0.079995 - NDCG(k=20): 0.034132 - NDCG(k=50): 0.047610 - HitRate(k=20): 0.243394 - HitRate(k=50): 0.389483
2021-01-14 11:14:20,306 P50829 INFO Save best model: monitor(max): 0.077857
2021-01-14 11:14:20,347 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 11:14:20,407 P50829 INFO Train loss: 0.652105
2021-01-14 11:14:20,408 P50829 INFO ************ Epoch=6 end ************
2021-01-14 11:14:20,408 P50829 INFO Negative sampling num_negs=1000
2021-01-14 11:15:02,479 P50829 INFO Negative sampling done
2021-01-14 11:19:48,393 P50829 INFO --- Start evaluation ---
2021-01-14 11:19:50,413 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 11:21:12,069 P50829 INFO [Metrics] Recall(k=20): 0.045265 - Recall(k=50): 0.082073 - NDCG(k=20): 0.035355 - NDCG(k=50): 0.049017 - HitRate(k=20): 0.248333 - HitRate(k=50): 0.394479
2021-01-14 11:21:12,089 P50829 INFO Save best model: monitor(max): 0.080620
2021-01-14 11:21:12,131 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 11:21:12,193 P50829 INFO Train loss: 0.651749
2021-01-14 11:21:12,193 P50829 INFO ************ Epoch=7 end ************
2021-01-14 11:21:12,194 P50829 INFO Negative sampling num_negs=1000
2021-01-14 11:21:55,434 P50829 INFO Negative sampling done
2021-01-14 11:26:33,632 P50829 INFO --- Start evaluation ---
2021-01-14 11:26:35,006 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 11:28:11,695 P50829 INFO [Metrics] Recall(k=20): 0.047135 - Recall(k=50): 0.085367 - NDCG(k=20): 0.036795 - NDCG(k=50): 0.050969 - HitRate(k=20): 0.258858 - HitRate(k=50): 0.408993
2021-01-14 11:28:11,714 P50829 INFO Save best model: monitor(max): 0.083930
2021-01-14 11:28:11,755 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 11:28:11,815 P50829 INFO Train loss: 0.651277
2021-01-14 11:28:11,815 P50829 INFO ************ Epoch=8 end ************
2021-01-14 11:28:11,816 P50829 INFO Negative sampling num_negs=1000
2021-01-14 11:28:53,435 P50829 INFO Negative sampling done
2021-01-14 11:33:28,023 P50829 INFO --- Start evaluation ---
2021-01-14 11:33:30,020 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 11:34:52,840 P50829 INFO [Metrics] Recall(k=20): 0.047684 - Recall(k=50): 0.085661 - NDCG(k=20): 0.037500 - NDCG(k=50): 0.051599 - HitRate(k=20): 0.260206 - HitRate(k=50): 0.408898
2021-01-14 11:34:52,860 P50829 INFO Save best model: monitor(max): 0.085184
2021-01-14 11:34:52,901 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 11:34:52,970 P50829 INFO Train loss: 0.650805
2021-01-14 11:34:52,970 P50829 INFO ************ Epoch=9 end ************
2021-01-14 11:34:52,971 P50829 INFO Negative sampling num_negs=1000
2021-01-14 11:35:41,186 P50829 INFO Negative sampling done
2021-01-14 11:40:18,244 P50829 INFO --- Start evaluation ---
2021-01-14 11:40:20,727 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 11:41:41,410 P50829 INFO [Metrics] Recall(k=20): 0.048800 - Recall(k=50): 0.087223 - NDCG(k=20): 0.038200 - NDCG(k=50): 0.052475 - HitRate(k=20): 0.262904 - HitRate(k=50): 0.412109
2021-01-14 11:41:41,430 P50829 INFO Save best model: monitor(max): 0.087001
2021-01-14 11:41:41,475 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 11:41:41,533 P50829 INFO Train loss: 0.650319
2021-01-14 11:41:41,533 P50829 INFO ************ Epoch=10 end ************
2021-01-14 11:41:41,534 P50829 INFO Negative sampling num_negs=1000
2021-01-14 11:42:28,582 P50829 INFO Negative sampling done
2021-01-14 11:47:08,394 P50829 INFO --- Start evaluation ---
2021-01-14 11:47:10,219 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 11:48:30,379 P50829 INFO [Metrics] Recall(k=20): 0.050240 - Recall(k=50): 0.089115 - NDCG(k=20): 0.039550 - NDCG(k=50): 0.053991 - HitRate(k=20): 0.269895 - HitRate(k=50): 0.420297
2021-01-14 11:48:30,487 P50829 INFO Save best model: monitor(max): 0.089789
2021-01-14 11:48:30,664 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 11:48:30,865 P50829 INFO Train loss: 0.649871
2021-01-14 11:48:30,865 P50829 INFO ************ Epoch=11 end ************
2021-01-14 11:48:30,870 P50829 INFO Negative sampling num_negs=1000
2021-01-14 11:49:14,068 P50829 INFO Negative sampling done
2021-01-14 11:54:03,041 P50829 INFO --- Start evaluation ---
2021-01-14 11:54:05,433 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 11:55:38,223 P50829 INFO [Metrics] Recall(k=20): 0.050442 - Recall(k=50): 0.089679 - NDCG(k=20): 0.039984 - NDCG(k=50): 0.054492 - HitRate(k=20): 0.271852 - HitRate(k=50): 0.420981
2021-01-14 11:55:38,247 P50829 INFO Save best model: monitor(max): 0.090426
2021-01-14 11:55:38,315 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 11:55:38,428 P50829 INFO Train loss: 0.649447
2021-01-14 11:55:38,428 P50829 INFO ************ Epoch=12 end ************
2021-01-14 11:55:38,429 P50829 INFO Negative sampling num_negs=1000
2021-01-14 11:56:24,107 P50829 INFO Negative sampling done
2021-01-14 12:01:17,043 P50829 INFO --- Start evaluation ---
2021-01-14 12:01:20,332 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 12:02:52,132 P50829 INFO [Metrics] Recall(k=20): 0.050851 - Recall(k=50): 0.090502 - NDCG(k=20): 0.040049 - NDCG(k=50): 0.054754 - HitRate(k=20): 0.274017 - HitRate(k=50): 0.424400
2021-01-14 12:02:52,156 P50829 INFO Save best model: monitor(max): 0.090900
2021-01-14 12:02:52,200 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 12:02:52,298 P50829 INFO Train loss: 0.649083
2021-01-14 12:02:52,298 P50829 INFO ************ Epoch=13 end ************
2021-01-14 12:02:52,298 P50829 INFO Negative sampling num_negs=1000
2021-01-14 12:07:09,651 P50829 INFO Negative sampling done
2021-01-14 12:12:00,559 P50829 INFO --- Start evaluation ---
2021-01-14 12:12:02,114 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 12:13:34,282 P50829 INFO [Metrics] Recall(k=20): 0.051320 - Recall(k=50): 0.090872 - NDCG(k=20): 0.040416 - NDCG(k=50): 0.055095 - HitRate(k=20): 0.275214 - HitRate(k=50): 0.424761
2021-01-14 12:13:34,313 P50829 INFO Save best model: monitor(max): 0.091736
2021-01-14 12:13:34,359 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 12:13:34,456 P50829 INFO Train loss: 0.648723
2021-01-14 12:13:34,456 P50829 INFO ************ Epoch=14 end ************
2021-01-14 12:13:34,457 P50829 INFO Negative sampling num_negs=1000
2021-01-14 12:14:19,562 P50829 INFO Negative sampling done
2021-01-14 12:19:23,078 P50829 INFO --- Start evaluation ---
2021-01-14 12:19:25,407 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 12:21:34,566 P50829 INFO [Metrics] Recall(k=20): 0.052233 - Recall(k=50): 0.091666 - NDCG(k=20): 0.040948 - NDCG(k=50): 0.055628 - HitRate(k=20): 0.277000 - HitRate(k=50): 0.428200
2021-01-14 12:21:34,986 P50829 INFO Save best model: monitor(max): 0.093181
2021-01-14 12:21:35,355 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 12:21:35,760 P50829 INFO Train loss: 0.648355
2021-01-14 12:21:35,761 P50829 INFO ************ Epoch=15 end ************
2021-01-14 12:21:35,778 P50829 INFO Negative sampling num_negs=1000
2021-01-14 12:22:24,184 P50829 INFO Negative sampling done
2021-01-14 12:27:46,138 P50829 INFO --- Start evaluation ---
2021-01-14 12:27:47,784 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 12:29:26,621 P50829 INFO [Metrics] Recall(k=20): 0.052311 - Recall(k=50): 0.092310 - NDCG(k=20): 0.041506 - NDCG(k=50): 0.056393 - HitRate(k=20): 0.278159 - HitRate(k=50): 0.430213
2021-01-14 12:29:26,642 P50829 INFO Save best model: monitor(max): 0.093816
2021-01-14 12:29:26,686 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 12:29:26,753 P50829 INFO Train loss: 0.648079
2021-01-14 12:29:26,753 P50829 INFO ************ Epoch=16 end ************
2021-01-14 12:29:26,754 P50829 INFO Negative sampling num_negs=1000
2021-01-14 12:30:14,646 P50829 INFO Negative sampling done
2021-01-14 12:35:35,063 P50829 INFO --- Start evaluation ---
2021-01-14 12:35:36,694 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 12:36:59,169 P50829 INFO [Metrics] Recall(k=20): 0.052564 - Recall(k=50): 0.092868 - NDCG(k=20): 0.041364 - NDCG(k=50): 0.056272 - HitRate(k=20): 0.278083 - HitRate(k=50): 0.428827
2021-01-14 12:36:59,189 P50829 INFO Save best model: monitor(max): 0.093928
2021-01-14 12:36:59,229 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 12:36:59,309 P50829 INFO Train loss: 0.647806
2021-01-14 12:36:59,310 P50829 INFO ************ Epoch=17 end ************
2021-01-14 12:36:59,310 P50829 INFO Negative sampling num_negs=1000
2021-01-14 12:37:40,150 P50829 INFO Negative sampling done
2021-01-14 12:42:25,445 P50829 INFO --- Start evaluation ---
2021-01-14 12:42:27,155 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 12:43:49,164 P50829 INFO [Metrics] Recall(k=20): 0.052708 - Recall(k=50): 0.091897 - NDCG(k=20): 0.041514 - NDCG(k=50): 0.056080 - HitRate(k=20): 0.279280 - HitRate(k=50): 0.426889
2021-01-14 12:43:49,180 P50829 INFO Save best model: monitor(max): 0.094221
2021-01-14 12:43:49,221 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 12:43:49,299 P50829 INFO Train loss: 0.647520
2021-01-14 12:43:49,299 P50829 INFO ************ Epoch=18 end ************
2021-01-14 12:43:49,300 P50829 INFO Negative sampling num_negs=1000
2021-01-14 12:44:30,462 P50829 INFO Negative sampling done
2021-01-14 12:49:32,331 P50829 INFO --- Start evaluation ---
2021-01-14 12:49:33,343 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 12:50:56,962 P50829 INFO [Metrics] Recall(k=20): 0.052864 - Recall(k=50): 0.092976 - NDCG(k=20): 0.041787 - NDCG(k=50): 0.056662 - HitRate(k=20): 0.279090 - HitRate(k=50): 0.431790
2021-01-14 12:50:56,985 P50829 INFO Save best model: monitor(max): 0.094651
2021-01-14 12:50:57,028 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 12:50:57,107 P50829 INFO Train loss: 0.647262
2021-01-14 12:50:57,107 P50829 INFO ************ Epoch=19 end ************
2021-01-14 12:50:57,108 P50829 INFO Negative sampling num_negs=1000
2021-01-14 12:51:40,519 P50829 INFO Negative sampling done
2021-01-14 12:56:26,943 P50829 INFO --- Start evaluation ---
2021-01-14 12:56:31,144 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 12:57:49,926 P50829 INFO [Metrics] Recall(k=20): 0.053464 - Recall(k=50): 0.093314 - NDCG(k=20): 0.042152 - NDCG(k=50): 0.056915 - HitRate(k=20): 0.282357 - HitRate(k=50): 0.432987
2021-01-14 12:57:49,954 P50829 INFO Save best model: monitor(max): 0.095616
2021-01-14 12:57:50,012 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 12:57:50,106 P50829 INFO Train loss: 0.647071
2021-01-14 12:57:50,106 P50829 INFO ************ Epoch=20 end ************
2021-01-14 12:57:50,107 P50829 INFO Negative sampling num_negs=1000
2021-01-14 12:58:32,800 P50829 INFO Negative sampling done
2021-01-14 13:03:10,037 P50829 INFO --- Start evaluation ---
2021-01-14 13:03:11,810 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 13:04:37,232 P50829 INFO [Metrics] Recall(k=20): 0.053540 - Recall(k=50): 0.093640 - NDCG(k=20): 0.042283 - NDCG(k=50): 0.057181 - HitRate(k=20): 0.283231 - HitRate(k=50): 0.433671
2021-01-14 13:04:37,255 P50829 INFO Save best model: monitor(max): 0.095823
2021-01-14 13:04:37,304 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 13:04:37,391 P50829 INFO Train loss: 0.646792
2021-01-14 13:04:37,391 P50829 INFO ************ Epoch=21 end ************
2021-01-14 13:04:37,392 P50829 INFO Negative sampling num_negs=1000
2021-01-14 13:05:28,205 P50829 INFO Negative sampling done
2021-01-14 13:10:02,814 P50829 INFO --- Start evaluation ---
2021-01-14 13:10:04,822 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 13:11:28,792 P50829 INFO [Metrics] Recall(k=20): 0.053736 - Recall(k=50): 0.093627 - NDCG(k=20): 0.042428 - NDCG(k=50): 0.057273 - HitRate(k=20): 0.282889 - HitRate(k=50): 0.432398
2021-01-14 13:11:28,812 P50829 INFO Save best model: monitor(max): 0.096164
2021-01-14 13:11:28,854 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 13:11:28,920 P50829 INFO Train loss: 0.646655
2021-01-14 13:11:28,920 P50829 INFO ************ Epoch=22 end ************
2021-01-14 13:11:28,920 P50829 INFO Negative sampling num_negs=1000
2021-01-14 13:12:13,443 P50829 INFO Negative sampling done
2021-01-14 13:16:58,072 P50829 INFO --- Start evaluation ---
2021-01-14 13:17:00,878 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 13:18:23,378 P50829 INFO [Metrics] Recall(k=20): 0.054490 - Recall(k=50): 0.094564 - NDCG(k=20): 0.043335 - NDCG(k=50): 0.058210 - HitRate(k=20): 0.286271 - HitRate(k=50): 0.435970
2021-01-14 13:18:23,399 P50829 INFO Save best model: monitor(max): 0.097826
2021-01-14 13:18:23,439 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 13:18:23,520 P50829 INFO Train loss: 0.646545
2021-01-14 13:18:23,521 P50829 INFO ************ Epoch=23 end ************
2021-01-14 13:18:23,522 P50829 INFO Negative sampling num_negs=1000
2021-01-14 13:19:11,080 P50829 INFO Negative sampling done
2021-01-14 13:23:56,944 P50829 INFO --- Start evaluation ---
2021-01-14 13:23:58,785 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 13:25:20,169 P50829 INFO [Metrics] Recall(k=20): 0.053514 - Recall(k=50): 0.093300 - NDCG(k=20): 0.042354 - NDCG(k=50): 0.057168 - HitRate(k=20): 0.280704 - HitRate(k=50): 0.430726
2021-01-14 13:25:20,191 P50829 INFO Monitor(max) STOP: 0.095867 !
2021-01-14 13:25:20,191 P50829 INFO Reduce learning rate on plateau: 0.000100
2021-01-14 13:25:20,191 P50829 INFO Load best model:  SimpleX_amazonbooks_x0_003_a30a8992.model
2021-01-14 13:25:20,227 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 13:25:20,302 P50829 INFO Train loss: 0.646402
2021-01-14 13:25:20,302 P50829 INFO ************ Epoch=24 end ************
2021-01-14 13:25:20,303 P50829 INFO Negative sampling num_negs=1000
2021-01-14 13:26:02,829 P50829 INFO Negative sampling done
2021-01-14 13:29:35,539 P50829 INFO --- Start evaluation ---
2021-01-14 13:29:37,580 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 13:31:08,679 P50829 INFO [Metrics] Recall(k=20): 0.056726 - Recall(k=50): 0.098587 - NDCG(k=20): 0.045462 - NDCG(k=50): 0.061017 - HitRate(k=20): 0.296111 - HitRate(k=50): 0.449876
2021-01-14 13:31:08,714 P50829 INFO Save best model: monitor(max): 0.102188
2021-01-14 13:31:08,757 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 13:31:08,824 P50829 INFO Train loss: 0.624428
2021-01-14 13:31:08,824 P50829 INFO ************ Epoch=25 end ************
2021-01-14 13:31:08,824 P50829 INFO Negative sampling num_negs=1000
2021-01-14 13:31:51,867 P50829 INFO Negative sampling done
2021-01-14 13:35:42,644 P50829 INFO --- Start evaluation ---
2021-01-14 13:35:44,649 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 13:37:04,754 P50829 INFO [Metrics] Recall(k=20): 0.057088 - Recall(k=50): 0.099543 - NDCG(k=20): 0.045754 - NDCG(k=50): 0.061468 - HitRate(k=20): 0.297517 - HitRate(k=50): 0.451870
2021-01-14 13:37:04,772 P50829 INFO Save best model: monitor(max): 0.102842
2021-01-14 13:37:04,815 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 13:37:04,881 P50829 INFO Train loss: 0.620373
2021-01-14 13:37:04,881 P50829 INFO ************ Epoch=26 end ************
2021-01-14 13:37:04,882 P50829 INFO Negative sampling num_negs=1000
2021-01-14 13:37:49,579 P50829 INFO Negative sampling done
2021-01-14 13:41:44,264 P50829 INFO --- Start evaluation ---
2021-01-14 13:41:45,687 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 13:43:03,388 P50829 INFO [Metrics] Recall(k=20): 0.057287 - Recall(k=50): 0.099795 - NDCG(k=20): 0.045979 - NDCG(k=50): 0.061751 - HitRate(k=20): 0.297327 - HitRate(k=50): 0.453333
2021-01-14 13:43:03,408 P50829 INFO Save best model: monitor(max): 0.103266
2021-01-14 13:43:03,451 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 13:43:03,503 P50829 INFO Train loss: 0.618076
2021-01-14 13:43:03,503 P50829 INFO ************ Epoch=27 end ************
2021-01-14 13:43:03,504 P50829 INFO Negative sampling num_negs=1000
2021-01-14 13:43:47,918 P50829 INFO Negative sampling done
2021-01-14 13:47:42,336 P50829 INFO --- Start evaluation ---
2021-01-14 13:47:45,495 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 13:49:19,126 P50829 INFO [Metrics] Recall(k=20): 0.057413 - Recall(k=50): 0.100085 - NDCG(k=20): 0.046159 - NDCG(k=50): 0.061997 - HitRate(k=20): 0.298923 - HitRate(k=50): 0.453960
2021-01-14 13:49:19,146 P50829 INFO Save best model: monitor(max): 0.103573
2021-01-14 13:49:19,186 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 13:49:19,249 P50829 INFO Train loss: 0.616183
2021-01-14 13:49:19,249 P50829 INFO ************ Epoch=28 end ************
2021-01-14 13:49:19,249 P50829 INFO Negative sampling num_negs=1000
2021-01-14 13:50:01,106 P50829 INFO Negative sampling done
2021-01-14 13:54:04,958 P50829 INFO --- Start evaluation ---
2021-01-14 13:54:06,956 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 13:55:32,208 P50829 INFO [Metrics] Recall(k=20): 0.057753 - Recall(k=50): 0.100321 - NDCG(k=20): 0.046314 - NDCG(k=50): 0.062104 - HitRate(k=20): 0.301032 - HitRate(k=50): 0.455157
2021-01-14 13:55:32,229 P50829 INFO Save best model: monitor(max): 0.104067
2021-01-14 13:55:32,274 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 13:55:32,339 P50829 INFO Train loss: 0.614566
2021-01-14 13:55:32,339 P50829 INFO ************ Epoch=29 end ************
2021-01-14 13:55:32,339 P50829 INFO Negative sampling num_negs=1000
2021-01-14 13:56:18,470 P50829 INFO Negative sampling done
2021-01-14 14:00:29,456 P50829 INFO --- Start evaluation ---
2021-01-14 14:00:31,448 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 14:02:01,365 P50829 INFO [Metrics] Recall(k=20): 0.058138 - Recall(k=50): 0.100787 - NDCG(k=20): 0.046665 - NDCG(k=50): 0.062464 - HitRate(k=20): 0.302228 - HitRate(k=50): 0.456601
2021-01-14 14:02:01,400 P50829 INFO Save best model: monitor(max): 0.104803
2021-01-14 14:02:01,452 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 14:02:01,517 P50829 INFO Train loss: 0.613233
2021-01-14 14:02:01,517 P50829 INFO ************ Epoch=30 end ************
2021-01-14 14:02:01,518 P50829 INFO Negative sampling num_negs=1000
2021-01-14 14:02:45,637 P50829 INFO Negative sampling done
2021-01-14 14:07:05,038 P50829 INFO --- Start evaluation ---
2021-01-14 14:07:07,049 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 14:08:36,564 P50829 INFO [Metrics] Recall(k=20): 0.058175 - Recall(k=50): 0.100251 - NDCG(k=20): 0.046637 - NDCG(k=50): 0.062209 - HitRate(k=20): 0.302019 - HitRate(k=50): 0.454283
2021-01-14 14:08:36,586 P50829 INFO Save best model: monitor(max): 0.104812
2021-01-14 14:08:36,629 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 14:08:36,708 P50829 INFO Train loss: 0.612007
2021-01-14 14:08:36,708 P50829 INFO ************ Epoch=31 end ************
2021-01-14 14:08:36,709 P50829 INFO Negative sampling num_negs=1000
2021-01-14 14:09:19,095 P50829 INFO Negative sampling done
2021-01-14 14:13:38,327 P50829 INFO --- Start evaluation ---
2021-01-14 14:13:40,000 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 14:15:08,851 P50829 INFO [Metrics] Recall(k=20): 0.058028 - Recall(k=50): 0.100345 - NDCG(k=20): 0.046612 - NDCG(k=50): 0.062280 - HitRate(k=20): 0.301355 - HitRate(k=50): 0.455442
2021-01-14 14:15:08,876 P50829 INFO Monitor(max) STOP: 0.104639 !
2021-01-14 14:15:08,876 P50829 INFO Reduce learning rate on plateau: 0.000010
2021-01-14 14:15:08,877 P50829 INFO Load best model:  SimpleX_amazonbooks_x0_003_a30a8992.model
2021-01-14 14:15:08,907 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 14:15:08,990 P50829 INFO Train loss: 0.610898
2021-01-14 14:15:08,990 P50829 INFO ************ Epoch=32 end ************
2021-01-14 14:15:08,991 P50829 INFO Negative sampling num_negs=1000
2021-01-14 14:15:53,938 P50829 INFO Negative sampling done
2021-01-14 14:19:57,741 P50829 INFO --- Start evaluation ---
2021-01-14 14:19:58,916 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 14:21:22,629 P50829 INFO [Metrics] Recall(k=20): 0.058197 - Recall(k=50): 0.100463 - NDCG(k=20): 0.046654 - NDCG(k=50): 0.062317 - HitRate(k=20): 0.302190 - HitRate(k=50): 0.454815
2021-01-14 14:21:22,650 P50829 INFO Save best model: monitor(max): 0.104851
2021-01-14 14:21:22,691 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 14:21:22,764 P50829 INFO Train loss: 0.608174
2021-01-14 14:21:22,764 P50829 INFO ************ Epoch=33 end ************
2021-01-14 14:21:22,765 P50829 INFO Negative sampling num_negs=1000
2021-01-14 14:22:08,768 P50829 INFO Negative sampling done
2021-01-14 14:26:12,983 P50829 INFO --- Start evaluation ---
2021-01-14 14:26:14,181 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 14:27:40,284 P50829 INFO [Metrics] Recall(k=20): 0.058250 - Recall(k=50): 0.100489 - NDCG(k=20): 0.046729 - NDCG(k=50): 0.062380 - HitRate(k=20): 0.302190 - HitRate(k=50): 0.455328
2021-01-14 14:27:40,304 P50829 INFO Save best model: monitor(max): 0.104979
2021-01-14 14:27:40,343 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 14:27:40,403 P50829 INFO Train loss: 0.607895
2021-01-14 14:27:40,403 P50829 INFO ************ Epoch=34 end ************
2021-01-14 14:27:40,404 P50829 INFO Negative sampling num_negs=1000
2021-01-14 14:28:25,930 P50829 INFO Negative sampling done
2021-01-14 14:32:36,698 P50829 INFO --- Start evaluation ---
2021-01-14 14:32:37,938 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 14:34:07,236 P50829 INFO [Metrics] Recall(k=20): 0.058282 - Recall(k=50): 0.100512 - NDCG(k=20): 0.046761 - NDCG(k=50): 0.062400 - HitRate(k=20): 0.302627 - HitRate(k=50): 0.455404
2021-01-14 14:34:07,257 P50829 INFO Save best model: monitor(max): 0.105043
2021-01-14 14:34:07,304 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 14:34:07,365 P50829 INFO Train loss: 0.607677
2021-01-14 14:34:07,365 P50829 INFO ************ Epoch=35 end ************
2021-01-14 14:34:07,365 P50829 INFO Negative sampling num_negs=1000
2021-01-14 14:34:48,031 P50829 INFO Negative sampling done
2021-01-14 14:38:49,986 P50829 INFO --- Start evaluation ---
2021-01-14 14:38:51,649 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 14:40:16,174 P50829 INFO [Metrics] Recall(k=20): 0.058341 - Recall(k=50): 0.100549 - NDCG(k=20): 0.046792 - NDCG(k=50): 0.062420 - HitRate(k=20): 0.303007 - HitRate(k=50): 0.455727
2021-01-14 14:40:16,192 P50829 INFO Save best model: monitor(max): 0.105133
2021-01-14 14:40:16,232 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 14:40:16,327 P50829 INFO Train loss: 0.607535
2021-01-14 14:40:16,327 P50829 INFO ************ Epoch=36 end ************
2021-01-14 14:40:16,328 P50829 INFO Negative sampling num_negs=1000
2021-01-14 14:40:59,835 P50829 INFO Negative sampling done
2021-01-14 14:43:56,151 P50829 INFO --- Start evaluation ---
2021-01-14 14:43:57,831 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 14:45:21,368 P50829 INFO [Metrics] Recall(k=20): 0.058299 - Recall(k=50): 0.100606 - NDCG(k=20): 0.046748 - NDCG(k=50): 0.062420 - HitRate(k=20): 0.302665 - HitRate(k=50): 0.455898
2021-01-14 14:45:21,388 P50829 INFO Monitor(max) STOP: 0.105047 !
2021-01-14 14:45:21,388 P50829 INFO Reduce learning rate on plateau: 0.000001
2021-01-14 14:45:21,389 P50829 INFO Load best model:  SimpleX_amazonbooks_x0_003_a30a8992.model
2021-01-14 14:45:21,420 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 14:45:21,502 P50829 INFO Train loss: 0.607394
2021-01-14 14:45:21,503 P50829 INFO ************ Epoch=37 end ************
2021-01-14 14:45:21,503 P50829 INFO Negative sampling num_negs=1000
2021-01-14 14:46:07,904 P50829 INFO Negative sampling done
2021-01-14 14:49:16,960 P50829 INFO --- Start evaluation ---
2021-01-14 14:49:18,535 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 14:50:39,759 P50829 INFO [Metrics] Recall(k=20): 0.058337 - Recall(k=50): 0.100534 - NDCG(k=20): 0.046787 - NDCG(k=50): 0.062408 - HitRate(k=20): 0.302988 - HitRate(k=50): 0.455613
2021-01-14 14:50:39,780 P50829 INFO Monitor(max) STOP: 0.105124 !
2021-01-14 14:50:39,780 P50829 INFO Reduce learning rate on plateau: 0.000001
2021-01-14 14:50:39,780 P50829 INFO Load best model:  SimpleX_amazonbooks_x0_003_a30a8992.model
2021-01-14 14:50:39,808 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 14:50:39,883 P50829 INFO Train loss: 0.607024
2021-01-14 14:50:39,884 P50829 INFO ************ Epoch=38 end ************
2021-01-14 14:50:39,884 P50829 INFO Negative sampling num_negs=1000
2021-01-14 14:51:20,256 P50829 INFO Negative sampling done
2021-01-14 14:54:16,777 P50829 INFO --- Start evaluation ---
2021-01-14 14:54:18,477 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 14:55:45,231 P50829 INFO [Metrics] Recall(k=20): 0.058323 - Recall(k=50): 0.100536 - NDCG(k=20): 0.046783 - NDCG(k=50): 0.062412 - HitRate(k=20): 0.302893 - HitRate(k=50): 0.455613
2021-01-14 14:55:45,247 P50829 INFO Monitor(max) STOP: 0.105106 !
2021-01-14 14:55:45,247 P50829 INFO Reduce learning rate on plateau: 0.000001
2021-01-14 14:55:45,248 P50829 INFO Load best model:  SimpleX_amazonbooks_x0_003_a30a8992.model
2021-01-14 14:55:45,271 P50829 INFO Early stopping at epoch=39
2021-01-14 14:55:45,271 P50829 INFO --- 2325/2325 batches finished ---
2021-01-14 14:55:45,323 P50829 INFO Train loss: 0.607100
2021-01-14 14:55:45,323 P50829 INFO Training finished.
2021-01-14 14:55:45,323 P50829 INFO Load best model:  SimpleX_amazonbooks_x0_003_a30a8992.model
2021-01-14 14:55:45,364 P50829 INFO ****** Train/validation evaluation ******
2021-01-14 14:55:45,364 P50829 INFO --- Start evaluation ---
2021-01-14 14:55:46,295 P50829 INFO Evaluating metrics for 52639 users...
2021-01-14 14:57:03,807 P50829 INFO [Metrics] Recall(k=20): 0.058341 - Recall(k=50): 0.100549 - NDCG(k=20): 0.046792 - NDCG(k=50): 0.062420 - HitRate(k=20): 0.303007 - HitRate(k=50): 0.455727

```