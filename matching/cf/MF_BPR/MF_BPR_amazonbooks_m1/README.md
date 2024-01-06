## MF-BPR_amazonbooks_x0 

A notebook to benchmark MF-BPR on amazonbooks_x0 dataset.

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
2021-01-02 20:54:27,848 P40150 INFO [Metrics] Recall(k=20): 0.033768 - Recall(k=50): 0.066024 - NDCG(k=20): 0.026070 - NDCG(k=50): 0.038028 - HitRate(k=20): 0.210338 - HitRate(k=50): 0.353027
```


### Logs
```
2021-01-02 19:54:08,835 P40150 INFO Set up feature encoder...
2021-01-02 19:54:08,835 P40150 INFO Load feature_map from json: ../data/AmazonBooks/amazonbooks_x0_37e049e0/feature_map.json
2021-01-02 19:54:14,068 P40150 INFO Total number of parameters: 9231616.
2021-01-02 19:54:14,068 P40150 INFO Loading data...
2021-01-02 19:54:14,073 P40150 INFO Loading data from h5: ../data/AmazonBooks/amazonbooks_x0_37e049e0/train.h5
2021-01-02 19:54:14,109 P40150 INFO Loading data from h5: ../data/AmazonBooks/amazonbooks_x0_37e049e0/item_corpus.h5
2021-01-02 19:54:15,714 P40150 INFO Loading data from h5: ../data/AmazonBooks/amazonbooks_x0_37e049e0/valid.h5
2021-01-02 19:54:16,352 P40150 INFO Loading data from h5: ../data/AmazonBooks/amazonbooks_x0_37e049e0/item_corpus.h5
2021-01-02 19:54:16,354 P40150 INFO Train samples: total/2380730, blocks/1
2021-01-02 19:54:16,354 P40150 INFO Validation samples: total/52639, blocks/1
2021-01-02 19:54:16,354 P40150 INFO Loading train data done.
2021-01-02 19:54:16,354 P40150 INFO **** Start training: 2325 batches/epoch ****
2021-01-02 19:54:16,370 P40150 INFO Negative sampling num_negs=20
2021-01-02 19:54:17,889 P40150 INFO Negative sampling done
2021-01-02 19:54:50,820 P40150 INFO --- Start evaluation ---
2021-01-02 19:54:52,271 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 19:56:36,296 P40150 INFO [Metrics] Recall(k=20): 0.012292 - Recall(k=50): 0.026583 - NDCG(k=20): 0.010170 - NDCG(k=50): 0.015457 - HitRate(k=20): 0.095006 - HitRate(k=50): 0.180570
2021-01-02 19:56:36,315 P40150 INFO Save best model: monitor(max): 0.012292
2021-01-02 19:56:36,359 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 19:56:36,495 P40150 INFO Train loss: 0.508371
2021-01-02 19:56:36,495 P40150 INFO ************ Epoch=1 end ************
2021-01-02 19:56:36,500 P40150 INFO Negative sampling num_negs=20
2021-01-02 19:56:38,159 P40150 INFO Negative sampling done
2021-01-02 19:57:14,289 P40150 INFO --- Start evaluation ---
2021-01-02 19:57:15,639 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 19:58:55,869 P40150 INFO [Metrics] Recall(k=20): 0.017782 - Recall(k=50): 0.036804 - NDCG(k=20): 0.014309 - NDCG(k=50): 0.021361 - HitRate(k=20): 0.131366 - HitRate(k=50): 0.238074
2021-01-02 19:58:55,904 P40150 INFO Save best model: monitor(max): 0.017782
2021-01-02 19:58:56,008 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 19:58:56,103 P40150 INFO Train loss: 0.228207
2021-01-02 19:58:56,103 P40150 INFO ************ Epoch=2 end ************
2021-01-02 19:58:56,108 P40150 INFO Negative sampling num_negs=20
2021-01-02 19:58:57,323 P40150 INFO Negative sampling done
2021-01-02 19:59:29,467 P40150 INFO --- Start evaluation ---
2021-01-02 19:59:30,764 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:01:12,027 P40150 INFO [Metrics] Recall(k=20): 0.020566 - Recall(k=50): 0.042360 - NDCG(k=20): 0.016586 - NDCG(k=50): 0.024618 - HitRate(k=20): 0.149224 - HitRate(k=50): 0.266532
2021-01-02 20:01:12,043 P40150 INFO Save best model: monitor(max): 0.020566
2021-01-02 20:01:12,106 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:01:12,207 P40150 INFO Train loss: 0.145104
2021-01-02 20:01:12,208 P40150 INFO ************ Epoch=3 end ************
2021-01-02 20:01:12,208 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:01:13,360 P40150 INFO Negative sampling done
2021-01-02 20:01:50,702 P40150 INFO --- Start evaluation ---
2021-01-02 20:01:52,245 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:03:31,554 P40150 INFO [Metrics] Recall(k=20): 0.022799 - Recall(k=50): 0.046459 - NDCG(k=20): 0.018114 - NDCG(k=50): 0.026906 - HitRate(k=20): 0.162237 - HitRate(k=50): 0.286195
2021-01-02 20:03:31,602 P40150 INFO Save best model: monitor(max): 0.022799
2021-01-02 20:03:31,724 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:03:31,842 P40150 INFO Train loss: 0.106270
2021-01-02 20:03:31,843 P40150 INFO ************ Epoch=4 end ************
2021-01-02 20:03:31,847 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:03:33,324 P40150 INFO Negative sampling done
2021-01-02 20:04:15,732 P40150 INFO --- Start evaluation ---
2021-01-02 20:04:17,112 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:05:57,289 P40150 INFO [Metrics] Recall(k=20): 0.024468 - Recall(k=50): 0.049649 - NDCG(k=20): 0.019386 - NDCG(k=50): 0.028768 - HitRate(k=20): 0.169988 - HitRate(k=50): 0.298942
2021-01-02 20:05:57,307 P40150 INFO Save best model: monitor(max): 0.024468
2021-01-02 20:05:57,366 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:05:57,476 P40150 INFO Train loss: 0.083997
2021-01-02 20:05:57,476 P40150 INFO ************ Epoch=5 end ************
2021-01-02 20:05:57,481 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:05:58,614 P40150 INFO Negative sampling done
2021-01-02 20:06:38,605 P40150 INFO --- Start evaluation ---
2021-01-02 20:06:40,024 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:08:20,950 P40150 INFO [Metrics] Recall(k=20): 0.025894 - Recall(k=50): 0.052426 - NDCG(k=20): 0.020450 - NDCG(k=50): 0.030321 - HitRate(k=20): 0.176371 - HitRate(k=50): 0.308232
2021-01-02 20:08:20,967 P40150 INFO Save best model: monitor(max): 0.025894
2021-01-02 20:08:21,032 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:08:21,159 P40150 INFO Train loss: 0.069344
2021-01-02 20:08:21,159 P40150 INFO ************ Epoch=6 end ************
2021-01-02 20:08:21,164 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:08:22,455 P40150 INFO Negative sampling done
2021-01-02 20:08:56,314 P40150 INFO --- Start evaluation ---
2021-01-02 20:08:57,841 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:10:42,337 P40150 INFO [Metrics] Recall(k=20): 0.027032 - Recall(k=50): 0.054545 - NDCG(k=20): 0.021227 - NDCG(k=50): 0.031443 - HitRate(k=20): 0.181690 - HitRate(k=50): 0.316932
2021-01-02 20:10:42,361 P40150 INFO Save best model: monitor(max): 0.027032
2021-01-02 20:10:42,486 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:10:42,588 P40150 INFO Train loss: 0.058730
2021-01-02 20:10:42,588 P40150 INFO ************ Epoch=7 end ************
2021-01-02 20:10:42,593 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:10:44,178 P40150 INFO Negative sampling done
2021-01-02 20:11:21,348 P40150 INFO --- Start evaluation ---
2021-01-02 20:11:22,727 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:13:03,164 P40150 INFO [Metrics] Recall(k=20): 0.028429 - Recall(k=50): 0.056999 - NDCG(k=20): 0.022196 - NDCG(k=50): 0.032822 - HitRate(k=20): 0.189517 - HitRate(k=50): 0.326393
2021-01-02 20:13:03,199 P40150 INFO Save best model: monitor(max): 0.028429
2021-01-02 20:13:03,275 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:13:03,371 P40150 INFO Train loss: 0.050775
2021-01-02 20:13:03,371 P40150 INFO ************ Epoch=8 end ************
2021-01-02 20:13:03,376 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:13:04,761 P40150 INFO Negative sampling done
2021-01-02 20:13:45,168 P40150 INFO --- Start evaluation ---
2021-01-02 20:13:46,645 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:15:30,264 P40150 INFO [Metrics] Recall(k=20): 0.028975 - Recall(k=50): 0.058843 - NDCG(k=20): 0.022650 - NDCG(k=50): 0.033744 - HitRate(k=20): 0.191721 - HitRate(k=50): 0.333536
2021-01-02 20:15:30,280 P40150 INFO Save best model: monitor(max): 0.028975
2021-01-02 20:15:30,338 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:15:30,438 P40150 INFO Train loss: 0.044634
2021-01-02 20:15:30,438 P40150 INFO ************ Epoch=9 end ************
2021-01-02 20:15:30,451 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:15:32,070 P40150 INFO Negative sampling done
2021-01-02 20:16:10,809 P40150 INFO --- Start evaluation ---
2021-01-02 20:16:12,351 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:17:54,587 P40150 INFO [Metrics] Recall(k=20): 0.029812 - Recall(k=50): 0.059908 - NDCG(k=20): 0.023106 - NDCG(k=50): 0.034307 - HitRate(k=20): 0.193488 - HitRate(k=50): 0.335341
2021-01-02 20:17:54,604 P40150 INFO Save best model: monitor(max): 0.029812
2021-01-02 20:17:54,659 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:17:54,762 P40150 INFO Train loss: 0.039860
2021-01-02 20:17:54,762 P40150 INFO ************ Epoch=10 end ************
2021-01-02 20:17:54,767 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:17:55,900 P40150 INFO Negative sampling done
2021-01-02 20:18:30,315 P40150 INFO --- Start evaluation ---
2021-01-02 20:18:31,848 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:20:19,835 P40150 INFO [Metrics] Recall(k=20): 0.030308 - Recall(k=50): 0.060635 - NDCG(k=20): 0.023546 - NDCG(k=50): 0.034810 - HitRate(k=20): 0.195805 - HitRate(k=50): 0.337050
2021-01-02 20:20:19,853 P40150 INFO Save best model: monitor(max): 0.030308
2021-01-02 20:20:19,913 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:20:20,061 P40150 INFO Train loss: 0.036181
2021-01-02 20:20:20,062 P40150 INFO ************ Epoch=11 end ************
2021-01-02 20:20:20,062 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:20:21,388 P40150 INFO Negative sampling done
2021-01-02 20:21:00,962 P40150 INFO --- Start evaluation ---
2021-01-02 20:21:02,425 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:22:45,597 P40150 INFO [Metrics] Recall(k=20): 0.031067 - Recall(k=50): 0.062278 - NDCG(k=20): 0.024117 - NDCG(k=50): 0.035679 - HitRate(k=20): 0.200327 - HitRate(k=50): 0.341952
2021-01-02 20:22:45,624 P40150 INFO Save best model: monitor(max): 0.031067
2021-01-02 20:22:45,685 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:22:45,833 P40150 INFO Train loss: 0.033253
2021-01-02 20:22:45,833 P40150 INFO ************ Epoch=12 end ************
2021-01-02 20:22:45,834 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:22:47,044 P40150 INFO Negative sampling done
2021-01-02 20:23:26,295 P40150 INFO --- Start evaluation ---
2021-01-02 20:23:27,992 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:25:11,354 P40150 INFO [Metrics] Recall(k=20): 0.031720 - Recall(k=50): 0.062851 - NDCG(k=20): 0.024556 - NDCG(k=50): 0.036104 - HitRate(k=20): 0.203366 - HitRate(k=50): 0.345770
2021-01-02 20:25:11,389 P40150 INFO Save best model: monitor(max): 0.031720
2021-01-02 20:25:11,491 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:25:11,647 P40150 INFO Train loss: 0.031011
2021-01-02 20:25:11,647 P40150 INFO ************ Epoch=13 end ************
2021-01-02 20:25:11,652 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:25:12,792 P40150 INFO Negative sampling done
2021-01-02 20:25:46,950 P40150 INFO --- Start evaluation ---
2021-01-02 20:25:48,331 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:27:30,822 P40150 INFO [Metrics] Recall(k=20): 0.031981 - Recall(k=50): 0.063080 - NDCG(k=20): 0.024817 - NDCG(k=50): 0.036351 - HitRate(k=20): 0.203252 - HitRate(k=50): 0.345067
2021-01-02 20:27:30,860 P40150 INFO Save best model: monitor(max): 0.031981
2021-01-02 20:27:30,985 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:27:31,130 P40150 INFO Train loss: 0.029145
2021-01-02 20:27:31,130 P40150 INFO ************ Epoch=14 end ************
2021-01-02 20:27:31,135 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:27:32,907 P40150 INFO Negative sampling done
2021-01-02 20:28:12,403 P40150 INFO --- Start evaluation ---
2021-01-02 20:28:13,943 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:30:01,190 P40150 INFO [Metrics] Recall(k=20): 0.032208 - Recall(k=50): 0.064058 - NDCG(k=20): 0.024876 - NDCG(k=50): 0.036726 - HitRate(k=20): 0.203822 - HitRate(k=50): 0.348449
2021-01-02 20:30:01,209 P40150 INFO Save best model: monitor(max): 0.032208
2021-01-02 20:30:01,271 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:30:01,398 P40150 INFO Train loss: 0.027791
2021-01-02 20:30:01,398 P40150 INFO ************ Epoch=15 end ************
2021-01-02 20:30:01,399 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:30:02,551 P40150 INFO Negative sampling done
2021-01-02 20:30:41,099 P40150 INFO --- Start evaluation ---
2021-01-02 20:30:42,532 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:32:28,707 P40150 INFO [Metrics] Recall(k=20): 0.032215 - Recall(k=50): 0.064205 - NDCG(k=20): 0.025050 - NDCG(k=50): 0.036942 - HitRate(k=20): 0.204050 - HitRate(k=50): 0.349095
2021-01-02 20:32:28,724 P40150 INFO Save best model: monitor(max): 0.032215
2021-01-02 20:32:28,787 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:32:28,883 P40150 INFO Train loss: 0.026618
2021-01-02 20:32:28,883 P40150 INFO ************ Epoch=16 end ************
2021-01-02 20:32:28,885 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:32:30,469 P40150 INFO Negative sampling done
2021-01-02 20:33:05,424 P40150 INFO --- Start evaluation ---
2021-01-02 20:33:06,768 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:34:47,487 P40150 INFO [Metrics] Recall(k=20): 0.032434 - Recall(k=50): 0.064653 - NDCG(k=20): 0.025054 - NDCG(k=50): 0.037014 - HitRate(k=20): 0.204677 - HitRate(k=50): 0.351014
2021-01-02 20:34:47,503 P40150 INFO Save best model: monitor(max): 0.032434
2021-01-02 20:34:47,561 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:34:47,632 P40150 INFO Train loss: 0.025699
2021-01-02 20:34:47,632 P40150 INFO ************ Epoch=17 end ************
2021-01-02 20:34:47,633 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:34:48,770 P40150 INFO Negative sampling done
2021-01-02 20:35:13,619 P40150 INFO --- Start evaluation ---
2021-01-02 20:35:14,583 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:36:38,420 P40150 INFO [Metrics] Recall(k=20): 0.032843 - Recall(k=50): 0.065042 - NDCG(k=20): 0.025406 - NDCG(k=50): 0.037355 - HitRate(k=20): 0.206501 - HitRate(k=50): 0.350482
2021-01-02 20:36:38,440 P40150 INFO Save best model: monitor(max): 0.032843
2021-01-02 20:36:38,500 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:36:38,622 P40150 INFO Train loss: 0.024939
2021-01-02 20:36:38,623 P40150 INFO ************ Epoch=18 end ************
2021-01-02 20:36:38,630 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:36:39,996 P40150 INFO Negative sampling done
2021-01-02 20:37:04,260 P40150 INFO --- Start evaluation ---
2021-01-02 20:37:05,251 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:38:37,514 P40150 INFO [Metrics] Recall(k=20): 0.032808 - Recall(k=50): 0.065488 - NDCG(k=20): 0.025332 - NDCG(k=50): 0.037456 - HitRate(k=20): 0.206786 - HitRate(k=50): 0.352970
2021-01-02 20:38:37,528 P40150 INFO Monitor(max) STOP: 0.032808 !
2021-01-02 20:38:37,528 P40150 INFO Reduce learning rate on plateau: 0.000100
2021-01-02 20:38:37,529 P40150 INFO Load best model:  MF_amazonbooks_x0_001_a85b5723_model.ckpt
2021-01-02 20:38:37,555 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:38:37,624 P40150 INFO Train loss: 0.024252
2021-01-02 20:38:37,624 P40150 INFO ************ Epoch=19 end ************
2021-01-02 20:38:37,625 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:38:38,772 P40150 INFO Negative sampling done
2021-01-02 20:38:59,896 P40150 INFO --- Start evaluation ---
2021-01-02 20:39:00,784 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:40:28,518 P40150 INFO [Metrics] Recall(k=20): 0.033067 - Recall(k=50): 0.065490 - NDCG(k=20): 0.025538 - NDCG(k=50): 0.037581 - HitRate(k=20): 0.207432 - HitRate(k=50): 0.351602
2021-01-02 20:40:28,532 P40150 INFO Save best model: monitor(max): 0.033067
2021-01-02 20:40:28,589 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:40:28,649 P40150 INFO Train loss: 0.023043
2021-01-02 20:40:28,649 P40150 INFO ************ Epoch=20 end ************
2021-01-02 20:40:28,650 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:40:29,790 P40150 INFO Negative sampling done
2021-01-02 20:40:48,798 P40150 INFO --- Start evaluation ---
2021-01-02 20:40:49,641 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:42:14,733 P40150 INFO [Metrics] Recall(k=20): 0.033137 - Recall(k=50): 0.065584 - NDCG(k=20): 0.025626 - NDCG(k=50): 0.037675 - HitRate(k=20): 0.207983 - HitRate(k=50): 0.352324
2021-01-02 20:42:14,746 P40150 INFO Save best model: monitor(max): 0.033137
2021-01-02 20:42:14,798 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:42:14,865 P40150 INFO Train loss: 0.022860
2021-01-02 20:42:14,865 P40150 INFO ************ Epoch=21 end ************
2021-01-02 20:42:14,866 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:42:15,833 P40150 INFO Negative sampling done
2021-01-02 20:42:39,143 P40150 INFO --- Start evaluation ---
2021-01-02 20:42:40,046 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:44:09,952 P40150 INFO [Metrics] Recall(k=20): 0.033378 - Recall(k=50): 0.065894 - NDCG(k=20): 0.025785 - NDCG(k=50): 0.037828 - HitRate(k=20): 0.209123 - HitRate(k=50): 0.352704
2021-01-02 20:44:09,965 P40150 INFO Save best model: monitor(max): 0.033378
2021-01-02 20:44:10,016 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:44:10,084 P40150 INFO Train loss: 0.022716
2021-01-02 20:44:10,085 P40150 INFO ************ Epoch=22 end ************
2021-01-02 20:44:10,085 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:44:11,227 P40150 INFO Negative sampling done
2021-01-02 20:44:31,968 P40150 INFO --- Start evaluation ---
2021-01-02 20:44:32,832 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:45:55,184 P40150 INFO [Metrics] Recall(k=20): 0.033571 - Recall(k=50): 0.065912 - NDCG(k=20): 0.025904 - NDCG(k=50): 0.037897 - HitRate(k=20): 0.209844 - HitRate(k=50): 0.352609
2021-01-02 20:45:55,193 P40150 INFO Save best model: monitor(max): 0.033571
2021-01-02 20:45:55,232 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:45:55,300 P40150 INFO Train loss: 0.022551
2021-01-02 20:45:55,300 P40150 INFO ************ Epoch=23 end ************
2021-01-02 20:45:55,300 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:45:56,165 P40150 INFO Negative sampling done
2021-01-02 20:46:14,626 P40150 INFO --- Start evaluation ---
2021-01-02 20:46:15,240 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:47:26,163 P40150 INFO [Metrics] Recall(k=20): 0.033603 - Recall(k=50): 0.065933 - NDCG(k=20): 0.025945 - NDCG(k=50): 0.037927 - HitRate(k=20): 0.209426 - HitRate(k=50): 0.352723
2021-01-02 20:47:26,172 P40150 INFO Save best model: monitor(max): 0.033603
2021-01-02 20:47:26,211 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:47:26,261 P40150 INFO Train loss: 0.022464
2021-01-02 20:47:26,261 P40150 INFO ************ Epoch=24 end ************
2021-01-02 20:47:26,262 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:47:27,127 P40150 INFO Negative sampling done
2021-01-02 20:47:43,107 P40150 INFO --- Start evaluation ---
2021-01-02 20:47:43,702 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:48:52,708 P40150 INFO [Metrics] Recall(k=20): 0.033768 - Recall(k=50): 0.066024 - NDCG(k=20): 0.026070 - NDCG(k=50): 0.038028 - HitRate(k=20): 0.210338 - HitRate(k=50): 0.353027
2021-01-02 20:48:52,716 P40150 INFO Save best model: monitor(max): 0.033768
2021-01-02 20:48:52,758 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:48:52,810 P40150 INFO Train loss: 0.022323
2021-01-02 20:48:52,810 P40150 INFO ************ Epoch=25 end ************
2021-01-02 20:48:52,810 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:48:53,672 P40150 INFO Negative sampling done
2021-01-02 20:49:10,471 P40150 INFO --- Start evaluation ---
2021-01-02 20:49:11,075 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:50:19,847 P40150 INFO [Metrics] Recall(k=20): 0.033739 - Recall(k=50): 0.066187 - NDCG(k=20): 0.026083 - NDCG(k=50): 0.038121 - HitRate(k=20): 0.209692 - HitRate(k=50): 0.353407
2021-01-02 20:50:19,860 P40150 INFO Monitor(max) STOP: 0.033739 !
2021-01-02 20:50:19,861 P40150 INFO Reduce learning rate on plateau: 0.000010
2021-01-02 20:50:19,861 P40150 INFO Load best model:  MF_amazonbooks_x0_001_a85b5723_model.ckpt
2021-01-02 20:50:19,890 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:50:19,942 P40150 INFO Train loss: 0.022243
2021-01-02 20:50:19,942 P40150 INFO ************ Epoch=26 end ************
2021-01-02 20:50:19,942 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:50:20,808 P40150 INFO Negative sampling done
2021-01-02 20:50:37,852 P40150 INFO --- Start evaluation ---
2021-01-02 20:50:38,411 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:51:47,291 P40150 INFO [Metrics] Recall(k=20): 0.033757 - Recall(k=50): 0.066092 - NDCG(k=20): 0.026065 - NDCG(k=50): 0.038057 - HitRate(k=20): 0.210091 - HitRate(k=50): 0.353160
2021-01-02 20:51:47,299 P40150 INFO Monitor(max) STOP: 0.033757 !
2021-01-02 20:51:47,300 P40150 INFO Reduce learning rate on plateau: 0.000001
2021-01-02 20:51:47,300 P40150 INFO Load best model:  MF_amazonbooks_x0_001_a85b5723_model.ckpt
2021-01-02 20:51:47,324 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:51:47,366 P40150 INFO Train loss: 0.022153
2021-01-02 20:51:47,366 P40150 INFO ************ Epoch=27 end ************
2021-01-02 20:51:47,367 P40150 INFO Negative sampling num_negs=20
2021-01-02 20:51:48,232 P40150 INFO Negative sampling done
2021-01-02 20:52:06,087 P40150 INFO --- Start evaluation ---
2021-01-02 20:52:06,684 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:53:15,441 P40150 INFO [Metrics] Recall(k=20): 0.033764 - Recall(k=50): 0.066028 - NDCG(k=20): 0.026070 - NDCG(k=50): 0.038033 - HitRate(k=20): 0.210357 - HitRate(k=50): 0.353103
2021-01-02 20:53:15,450 P40150 INFO Monitor(max) STOP: 0.033764 !
2021-01-02 20:53:15,450 P40150 INFO Reduce learning rate on plateau: 0.000001
2021-01-02 20:53:15,450 P40150 INFO Load best model:  MF_amazonbooks_x0_001_a85b5723_model.ckpt
2021-01-02 20:53:15,474 P40150 INFO Early stopping at epoch=28
2021-01-02 20:53:15,474 P40150 INFO --- 2325/2325 batches finished ---
2021-01-02 20:53:15,554 P40150 INFO Train loss: 0.022166
2021-01-02 20:53:15,554 P40150 INFO Training finished.
2021-01-02 20:53:15,554 P40150 INFO Load best model:  MF_amazonbooks_x0_001_a85b5723_model.ckpt
2021-01-02 20:53:15,593 P40150 INFO ****** Train/validation evaluation ******
2021-01-02 20:53:15,593 P40150 INFO --- Start evaluation ---
2021-01-02 20:53:16,177 P40150 INFO Evaluating metrics for 52639 users...
2021-01-02 20:54:27,848 P40150 INFO [Metrics] Recall(k=20): 0.033768 - Recall(k=50): 0.066024 - NDCG(k=20): 0.026070 - NDCG(k=50): 0.038028 - HitRate(k=20): 0.210338 - HitRate(k=50): 0.353027
```