## MF-BPR_gowalla_x0 

A notebook to benchmark MF-BPR on gowalla dataset.

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
2021-01-02 20:38:17,007 P40291 INFO [Metrics] Recall(k=20): 0.162737 - Recall(k=50): 0.253303 - NDCG(k=20): 0.137805 - NDCG(k=50): 0.166195 - HitRate(k=20): 0.554391 - HitRate(k=50): 0.693616
```


### Logs
```
2021-01-02 19:54:49,322 P40291 INFO Set up feature encoder...
2021-01-02 19:54:49,322 P40291 INFO Load feature_map from json: ../data/Gowalla/gowalla_x0_4c90e422/feature_map.json
2021-01-02 19:54:53,986 P40291 INFO Total number of parameters: 4533824.
2021-01-02 19:54:53,986 P40291 INFO Loading data...
2021-01-02 19:54:53,993 P40291 INFO Loading data from h5: ../data/Gowalla/gowalla_x0_4c90e422/train.h5
2021-01-02 19:54:54,011 P40291 INFO Loading data from h5: ../data/Gowalla/gowalla_x0_4c90e422/item_corpus.h5
2021-01-02 19:54:54,537 P40291 INFO Loading data from h5: ../data/Gowalla/gowalla_x0_4c90e422/valid.h5
2021-01-02 19:54:54,715 P40291 INFO Loading data from h5: ../data/Gowalla/gowalla_x0_4c90e422/item_corpus.h5
2021-01-02 19:54:54,717 P40291 INFO Train samples: total/810128, blocks/1
2021-01-02 19:54:54,717 P40291 INFO Validation samples: total/29858, blocks/1
2021-01-02 19:54:54,717 P40291 INFO Loading train data done.
2021-01-02 19:54:54,717 P40291 INFO **** Start training: 3165 batches/epoch ****
2021-01-02 19:54:54,721 P40291 INFO Negative sampling num_negs=300
2021-01-02 19:55:01,984 P40291 INFO Negative sampling done
2021-01-02 19:55:40,104 P40291 INFO --- Start evaluation ---
2021-01-02 19:55:41,627 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 19:56:17,353 P40291 INFO [Metrics] Recall(k=20): 0.048168 - Recall(k=50): 0.071945 - NDCG(k=20): 0.043027 - NDCG(k=50): 0.050141 - HitRate(k=20): 0.236218 - HitRate(k=50): 0.316398
2021-01-02 19:56:17,381 P40291 INFO Save best model: monitor(max): 0.048168
2021-01-02 19:56:17,744 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 19:56:17,852 P40291 INFO Train loss: 0.594389
2021-01-02 19:56:17,852 P40291 INFO ************ Epoch=1 end ************
2021-01-02 19:56:17,857 P40291 INFO Negative sampling num_negs=300
2021-01-02 19:56:25,935 P40291 INFO Negative sampling done
2021-01-02 19:57:00,198 P40291 INFO --- Start evaluation ---
2021-01-02 19:57:01,667 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 19:57:34,685 P40291 INFO [Metrics] Recall(k=20): 0.083709 - Recall(k=50): 0.128095 - NDCG(k=20): 0.074185 - NDCG(k=50): 0.087823 - HitRate(k=20): 0.359502 - HitRate(k=50): 0.471197
2021-01-02 19:57:34,704 P40291 INFO Save best model: monitor(max): 0.083709
2021-01-02 19:57:34,756 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 19:57:34,850 P40291 INFO Train loss: 0.385577
2021-01-02 19:57:34,850 P40291 INFO ************ Epoch=2 end ************
2021-01-02 19:57:34,851 P40291 INFO Negative sampling num_negs=300
2021-01-02 19:57:42,165 P40291 INFO Negative sampling done
2021-01-02 19:58:22,379 P40291 INFO --- Start evaluation ---
2021-01-02 19:58:24,074 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 19:58:56,468 P40291 INFO [Metrics] Recall(k=20): 0.112807 - Recall(k=50): 0.175168 - NDCG(k=20): 0.097661 - NDCG(k=50): 0.116985 - HitRate(k=20): 0.441691 - HitRate(k=50): 0.568926
2021-01-02 19:58:56,490 P40291 INFO Save best model: monitor(max): 0.112807
2021-01-02 19:58:56,547 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 19:58:56,704 P40291 INFO Train loss: 0.221537
2021-01-02 19:58:56,705 P40291 INFO ************ Epoch=3 end ************
2021-01-02 19:58:56,709 P40291 INFO Negative sampling num_negs=300
2021-01-02 19:59:03,928 P40291 INFO Negative sampling done
2021-01-02 19:59:39,453 P40291 INFO --- Start evaluation ---
2021-01-02 19:59:40,922 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:00:13,817 P40291 INFO [Metrics] Recall(k=20): 0.128896 - Recall(k=50): 0.199213 - NDCG(k=20): 0.111274 - NDCG(k=50): 0.132991 - HitRate(k=20): 0.481948 - HitRate(k=50): 0.612499
2021-01-02 20:00:13,838 P40291 INFO Save best model: monitor(max): 0.128896
2021-01-02 20:00:13,867 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:00:14,023 P40291 INFO Train loss: 0.138797
2021-01-02 20:00:14,024 P40291 INFO ************ Epoch=4 end ************
2021-01-02 20:00:14,024 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:00:21,439 P40291 INFO Negative sampling done
2021-01-02 20:01:00,465 P40291 INFO --- Start evaluation ---
2021-01-02 20:01:01,813 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:01:32,437 P40291 INFO [Metrics] Recall(k=20): 0.136651 - Recall(k=50): 0.213901 - NDCG(k=20): 0.118046 - NDCG(k=50): 0.141973 - HitRate(k=20): 0.501976 - HitRate(k=50): 0.636513
2021-01-02 20:01:32,446 P40291 INFO Save best model: monitor(max): 0.136651
2021-01-02 20:01:32,474 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:01:32,558 P40291 INFO Train loss: 0.102663
2021-01-02 20:01:32,558 P40291 INFO ************ Epoch=5 end ************
2021-01-02 20:01:32,561 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:01:39,868 P40291 INFO Negative sampling done
2021-01-02 20:02:15,803 P40291 INFO --- Start evaluation ---
2021-01-02 20:02:17,440 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:02:49,511 P40291 INFO [Metrics] Recall(k=20): 0.142525 - Recall(k=50): 0.222449 - NDCG(k=20): 0.122899 - NDCG(k=50): 0.147665 - HitRate(k=20): 0.513330 - HitRate(k=50): 0.649139
2021-01-02 20:02:49,529 P40291 INFO Save best model: monitor(max): 0.142525
2021-01-02 20:02:49,574 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:02:49,685 P40291 INFO Train loss: 0.085104
2021-01-02 20:02:49,685 P40291 INFO ************ Epoch=6 end ************
2021-01-02 20:02:49,687 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:02:57,162 P40291 INFO Negative sampling done
2021-01-02 20:03:32,958 P40291 INFO --- Start evaluation ---
2021-01-02 20:03:34,260 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:04:06,119 P40291 INFO [Metrics] Recall(k=20): 0.146062 - Recall(k=50): 0.229535 - NDCG(k=20): 0.125669 - NDCG(k=50): 0.151474 - HitRate(k=20): 0.520731 - HitRate(k=50): 0.659589
2021-01-02 20:04:06,133 P40291 INFO Save best model: monitor(max): 0.146062
2021-01-02 20:04:06,166 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:04:06,313 P40291 INFO Train loss: 0.075479
2021-01-02 20:04:06,314 P40291 INFO ************ Epoch=7 end ************
2021-01-02 20:04:06,319 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:04:13,409 P40291 INFO Negative sampling done
2021-01-02 20:04:50,847 P40291 INFO --- Start evaluation ---
2021-01-02 20:04:52,428 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:05:24,425 P40291 INFO [Metrics] Recall(k=20): 0.148959 - Recall(k=50): 0.233372 - NDCG(k=20): 0.127587 - NDCG(k=50): 0.153781 - HitRate(k=20): 0.526291 - HitRate(k=50): 0.666522
2021-01-02 20:05:24,438 P40291 INFO Save best model: monitor(max): 0.148959
2021-01-02 20:05:24,485 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:05:24,633 P40291 INFO Train loss: 0.069511
2021-01-02 20:05:24,633 P40291 INFO ************ Epoch=8 end ************
2021-01-02 20:05:24,638 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:05:32,086 P40291 INFO Negative sampling done
2021-01-02 20:06:08,125 P40291 INFO --- Start evaluation ---
2021-01-02 20:06:09,666 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:06:42,416 P40291 INFO [Metrics] Recall(k=20): 0.151720 - Recall(k=50): 0.235739 - NDCG(k=20): 0.129698 - NDCG(k=50): 0.155844 - HitRate(k=20): 0.532688 - HitRate(k=50): 0.670005
2021-01-02 20:06:42,425 P40291 INFO Save best model: monitor(max): 0.151720
2021-01-02 20:06:42,453 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:06:42,558 P40291 INFO Train loss: 0.065529
2021-01-02 20:06:42,558 P40291 INFO ************ Epoch=9 end ************
2021-01-02 20:06:42,559 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:06:49,695 P40291 INFO Negative sampling done
2021-01-02 20:07:26,998 P40291 INFO --- Start evaluation ---
2021-01-02 20:07:28,465 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:07:59,156 P40291 INFO [Metrics] Recall(k=20): 0.152564 - Recall(k=50): 0.237972 - NDCG(k=20): 0.130642 - NDCG(k=50): 0.157269 - HitRate(k=20): 0.534262 - HitRate(k=50): 0.672148
2021-01-02 20:07:59,167 P40291 INFO Save best model: monitor(max): 0.152564
2021-01-02 20:07:59,195 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:07:59,333 P40291 INFO Train loss: 0.062668
2021-01-02 20:07:59,333 P40291 INFO ************ Epoch=10 end ************
2021-01-02 20:07:59,338 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:08:07,589 P40291 INFO Negative sampling done
2021-01-02 20:08:46,267 P40291 INFO --- Start evaluation ---
2021-01-02 20:08:48,045 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:09:21,631 P40291 INFO [Metrics] Recall(k=20): 0.153635 - Recall(k=50): 0.240063 - NDCG(k=20): 0.131424 - NDCG(k=50): 0.158382 - HitRate(k=20): 0.536841 - HitRate(k=50): 0.675296
2021-01-02 20:09:21,647 P40291 INFO Save best model: monitor(max): 0.153635
2021-01-02 20:09:21,700 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:09:21,860 P40291 INFO Train loss: 0.060492
2021-01-02 20:09:21,860 P40291 INFO ************ Epoch=11 end ************
2021-01-02 20:09:21,865 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:09:28,872 P40291 INFO Negative sampling done
2021-01-02 20:10:08,460 P40291 INFO --- Start evaluation ---
2021-01-02 20:10:09,798 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:10:41,315 P40291 INFO [Metrics] Recall(k=20): 0.154951 - Recall(k=50): 0.241780 - NDCG(k=20): 0.132411 - NDCG(k=50): 0.159582 - HitRate(k=20): 0.538784 - HitRate(k=50): 0.676569
2021-01-02 20:10:41,342 P40291 INFO Save best model: monitor(max): 0.154951
2021-01-02 20:10:41,397 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:10:41,489 P40291 INFO Train loss: 0.058850
2021-01-02 20:10:41,489 P40291 INFO ************ Epoch=12 end ************
2021-01-02 20:10:41,494 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:10:48,660 P40291 INFO Negative sampling done
2021-01-02 20:11:24,689 P40291 INFO --- Start evaluation ---
2021-01-02 20:11:26,261 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:11:58,925 P40291 INFO [Metrics] Recall(k=20): 0.155069 - Recall(k=50): 0.243412 - NDCG(k=20): 0.132536 - NDCG(k=50): 0.160170 - HitRate(k=20): 0.539118 - HitRate(k=50): 0.678411
2021-01-02 20:11:58,942 P40291 INFO Save best model: monitor(max): 0.155069
2021-01-02 20:11:59,002 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:11:59,184 P40291 INFO Train loss: 0.057467
2021-01-02 20:11:59,184 P40291 INFO ************ Epoch=13 end ************
2021-01-02 20:11:59,187 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:12:06,582 P40291 INFO Negative sampling done
2021-01-02 20:12:47,403 P40291 INFO --- Start evaluation ---
2021-01-02 20:12:48,949 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:13:22,683 P40291 INFO [Metrics] Recall(k=20): 0.156404 - Recall(k=50): 0.244258 - NDCG(k=20): 0.133468 - NDCG(k=50): 0.160945 - HitRate(k=20): 0.541697 - HitRate(k=50): 0.681024
2021-01-02 20:13:22,691 P40291 INFO Save best model: monitor(max): 0.156404
2021-01-02 20:13:22,721 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:13:22,900 P40291 INFO Train loss: 0.056387
2021-01-02 20:13:22,900 P40291 INFO ************ Epoch=14 end ************
2021-01-02 20:13:22,901 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:13:30,424 P40291 INFO Negative sampling done
2021-01-02 20:14:06,690 P40291 INFO --- Start evaluation ---
2021-01-02 20:14:08,141 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:14:40,186 P40291 INFO [Metrics] Recall(k=20): 0.157291 - Recall(k=50): 0.246373 - NDCG(k=20): 0.134112 - NDCG(k=50): 0.161904 - HitRate(k=20): 0.544209 - HitRate(k=50): 0.684507
2021-01-02 20:14:40,204 P40291 INFO Save best model: monitor(max): 0.157291
2021-01-02 20:14:40,268 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:14:40,422 P40291 INFO Train loss: 0.055471
2021-01-02 20:14:40,422 P40291 INFO ************ Epoch=15 end ************
2021-01-02 20:14:40,423 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:14:47,682 P40291 INFO Negative sampling done
2021-01-02 20:15:26,179 P40291 INFO --- Start evaluation ---
2021-01-02 20:15:27,742 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:16:03,440 P40291 INFO [Metrics] Recall(k=20): 0.157869 - Recall(k=50): 0.247042 - NDCG(k=20): 0.134337 - NDCG(k=50): 0.162189 - HitRate(k=20): 0.545047 - HitRate(k=50): 0.683736
2021-01-02 20:16:03,450 P40291 INFO Save best model: monitor(max): 0.157869
2021-01-02 20:16:03,479 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:16:03,595 P40291 INFO Train loss: 0.054663
2021-01-02 20:16:03,595 P40291 INFO ************ Epoch=16 end ************
2021-01-02 20:16:03,596 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:16:10,853 P40291 INFO Negative sampling done
2021-01-02 20:16:49,705 P40291 INFO --- Start evaluation ---
2021-01-02 20:16:51,090 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:17:27,667 P40291 INFO [Metrics] Recall(k=20): 0.158127 - Recall(k=50): 0.247233 - NDCG(k=20): 0.134806 - NDCG(k=50): 0.162645 - HitRate(k=20): 0.546889 - HitRate(k=50): 0.684976
2021-01-02 20:17:27,676 P40291 INFO Save best model: monitor(max): 0.158127
2021-01-02 20:17:27,704 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:17:27,881 P40291 INFO Train loss: 0.053946
2021-01-02 20:17:27,882 P40291 INFO ************ Epoch=17 end ************
2021-01-02 20:17:27,887 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:17:35,122 P40291 INFO Negative sampling done
2021-01-02 20:18:13,263 P40291 INFO --- Start evaluation ---
2021-01-02 20:18:14,959 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:18:45,596 P40291 INFO [Metrics] Recall(k=20): 0.158890 - Recall(k=50): 0.248182 - NDCG(k=20): 0.135173 - NDCG(k=50): 0.163082 - HitRate(k=20): 0.547625 - HitRate(k=50): 0.686315
2021-01-02 20:18:45,606 P40291 INFO Save best model: monitor(max): 0.158890
2021-01-02 20:18:45,633 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:18:45,783 P40291 INFO Train loss: 0.053359
2021-01-02 20:18:45,783 P40291 INFO ************ Epoch=18 end ************
2021-01-02 20:18:45,786 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:18:53,330 P40291 INFO Negative sampling done
2021-01-02 20:19:31,547 P40291 INFO --- Start evaluation ---
2021-01-02 20:19:33,055 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:20:07,193 P40291 INFO [Metrics] Recall(k=20): 0.159239 - Recall(k=50): 0.249325 - NDCG(k=20): 0.135276 - NDCG(k=50): 0.163460 - HitRate(k=20): 0.548865 - HitRate(k=50): 0.686851
2021-01-02 20:20:07,202 P40291 INFO Save best model: monitor(max): 0.159239
2021-01-02 20:20:07,230 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:20:07,369 P40291 INFO Train loss: 0.052804
2021-01-02 20:20:07,370 P40291 INFO ************ Epoch=19 end ************
2021-01-02 20:20:07,383 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:20:15,685 P40291 INFO Negative sampling done
2021-01-02 20:20:53,017 P40291 INFO --- Start evaluation ---
2021-01-02 20:20:54,514 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:21:30,311 P40291 INFO [Metrics] Recall(k=20): 0.159746 - Recall(k=50): 0.250372 - NDCG(k=20): 0.135678 - NDCG(k=50): 0.163958 - HitRate(k=20): 0.549903 - HitRate(k=50): 0.690368
2021-01-02 20:21:30,328 P40291 INFO Save best model: monitor(max): 0.159746
2021-01-02 20:21:30,409 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:21:30,519 P40291 INFO Train loss: 0.052371
2021-01-02 20:21:30,520 P40291 INFO ************ Epoch=20 end ************
2021-01-02 20:21:30,524 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:21:38,316 P40291 INFO Negative sampling done
2021-01-02 20:22:19,166 P40291 INFO --- Start evaluation ---
2021-01-02 20:22:20,998 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:22:53,217 P40291 INFO [Metrics] Recall(k=20): 0.160454 - Recall(k=50): 0.250352 - NDCG(k=20): 0.135975 - NDCG(k=50): 0.164141 - HitRate(k=20): 0.551176 - HitRate(k=50): 0.691674
2021-01-02 20:22:53,236 P40291 INFO Save best model: monitor(max): 0.160454
2021-01-02 20:22:53,297 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:22:53,403 P40291 INFO Train loss: 0.051943
2021-01-02 20:22:53,404 P40291 INFO ************ Epoch=21 end ************
2021-01-02 20:22:53,409 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:23:00,608 P40291 INFO Negative sampling done
2021-01-02 20:23:38,332 P40291 INFO --- Start evaluation ---
2021-01-02 20:23:39,737 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:24:11,529 P40291 INFO [Metrics] Recall(k=20): 0.160598 - Recall(k=50): 0.250367 - NDCG(k=20): 0.136329 - NDCG(k=50): 0.164413 - HitRate(k=20): 0.552314 - HitRate(k=50): 0.690636
2021-01-02 20:24:11,546 P40291 INFO Save best model: monitor(max): 0.160598
2021-01-02 20:24:11,612 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:24:11,709 P40291 INFO Train loss: 0.051592
2021-01-02 20:24:11,709 P40291 INFO ************ Epoch=22 end ************
2021-01-02 20:24:11,713 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:24:19,925 P40291 INFO Negative sampling done
2021-01-02 20:25:00,103 P40291 INFO --- Start evaluation ---
2021-01-02 20:25:01,647 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:25:32,740 P40291 INFO [Metrics] Recall(k=20): 0.160800 - Recall(k=50): 0.251126 - NDCG(k=20): 0.136405 - NDCG(k=50): 0.164672 - HitRate(k=20): 0.552515 - HitRate(k=50): 0.691339
2021-01-02 20:25:32,749 P40291 INFO Save best model: monitor(max): 0.160800
2021-01-02 20:25:32,778 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:25:32,894 P40291 INFO Train loss: 0.051217
2021-01-02 20:25:32,894 P40291 INFO ************ Epoch=23 end ************
2021-01-02 20:25:32,899 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:25:40,286 P40291 INFO Negative sampling done
2021-01-02 20:26:17,608 P40291 INFO --- Start evaluation ---
2021-01-02 20:26:19,148 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:26:52,479 P40291 INFO [Metrics] Recall(k=20): 0.161268 - Recall(k=50): 0.251534 - NDCG(k=20): 0.136648 - NDCG(k=50): 0.164973 - HitRate(k=20): 0.552582 - HitRate(k=50): 0.690736
2021-01-02 20:26:52,496 P40291 INFO Save best model: monitor(max): 0.161268
2021-01-02 20:26:52,530 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:26:52,670 P40291 INFO Train loss: 0.050910
2021-01-02 20:26:52,670 P40291 INFO ************ Epoch=24 end ************
2021-01-02 20:26:52,675 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:27:00,330 P40291 INFO Negative sampling done
2021-01-02 20:27:38,894 P40291 INFO --- Start evaluation ---
2021-01-02 20:27:40,270 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:28:13,172 P40291 INFO [Metrics] Recall(k=20): 0.161175 - Recall(k=50): 0.251798 - NDCG(k=20): 0.136892 - NDCG(k=50): 0.165270 - HitRate(k=20): 0.552214 - HitRate(k=50): 0.692176
2021-01-02 20:28:13,181 P40291 INFO Monitor(max) STOP: 0.161175 !
2021-01-02 20:28:13,181 P40291 INFO Reduce learning rate on plateau: 0.000100
2021-01-02 20:28:13,181 P40291 INFO Load best model:  MF_gowalla_x0_001_b72fc48a_model.ckpt
2021-01-02 20:28:13,197 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:28:13,331 P40291 INFO Train loss: 0.050683
2021-01-02 20:28:13,332 P40291 INFO ************ Epoch=25 end ************
2021-01-02 20:28:13,336 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:28:20,336 P40291 INFO Negative sampling done
2021-01-02 20:28:59,934 P40291 INFO --- Start evaluation ---
2021-01-02 20:29:01,373 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:29:34,720 P40291 INFO [Metrics] Recall(k=20): 0.161782 - Recall(k=50): 0.252264 - NDCG(k=20): 0.137056 - NDCG(k=50): 0.165399 - HitRate(k=20): 0.553520 - HitRate(k=50): 0.691908
2021-01-02 20:29:34,729 P40291 INFO Save best model: monitor(max): 0.161782
2021-01-02 20:29:34,757 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:29:34,966 P40291 INFO Train loss: 0.047807
2021-01-02 20:29:34,966 P40291 INFO ************ Epoch=26 end ************
2021-01-02 20:29:34,971 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:29:42,375 P40291 INFO Negative sampling done
2021-01-02 20:30:21,018 P40291 INFO --- Start evaluation ---
2021-01-02 20:30:22,548 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:30:55,387 P40291 INFO [Metrics] Recall(k=20): 0.162138 - Recall(k=50): 0.252401 - NDCG(k=20): 0.137357 - NDCG(k=50): 0.165638 - HitRate(k=20): 0.553687 - HitRate(k=50): 0.692009
2021-01-02 20:30:55,395 P40291 INFO Save best model: monitor(max): 0.162138
2021-01-02 20:30:55,424 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:30:55,548 P40291 INFO Train loss: 0.047740
2021-01-02 20:30:55,549 P40291 INFO ************ Epoch=27 end ************
2021-01-02 20:30:55,553 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:31:03,576 P40291 INFO Negative sampling done
2021-01-02 20:31:44,662 P40291 INFO --- Start evaluation ---
2021-01-02 20:31:46,335 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:32:20,336 P40291 INFO [Metrics] Recall(k=20): 0.162395 - Recall(k=50): 0.252765 - NDCG(k=20): 0.137640 - NDCG(k=50): 0.165953 - HitRate(k=20): 0.553353 - HitRate(k=50): 0.692411
2021-01-02 20:32:20,352 P40291 INFO Save best model: monitor(max): 0.162395
2021-01-02 20:32:20,387 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:32:20,480 P40291 INFO Train loss: 0.047653
2021-01-02 20:32:20,481 P40291 INFO ************ Epoch=28 end ************
2021-01-02 20:32:20,485 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:32:27,784 P40291 INFO Negative sampling done
2021-01-02 20:33:04,040 P40291 INFO --- Start evaluation ---
2021-01-02 20:33:05,568 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:33:37,465 P40291 INFO [Metrics] Recall(k=20): 0.162571 - Recall(k=50): 0.253166 - NDCG(k=20): 0.137777 - NDCG(k=50): 0.166165 - HitRate(k=20): 0.554022 - HitRate(k=50): 0.693449
2021-01-02 20:33:37,488 P40291 INFO Save best model: monitor(max): 0.162571
2021-01-02 20:33:37,547 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:33:37,660 P40291 INFO Train loss: 0.047578
2021-01-02 20:33:37,661 P40291 INFO ************ Epoch=29 end ************
2021-01-02 20:33:37,665 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:33:44,860 P40291 INFO Negative sampling done
2021-01-02 20:34:19,871 P40291 INFO --- Start evaluation ---
2021-01-02 20:34:21,336 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:34:53,120 P40291 INFO [Metrics] Recall(k=20): 0.162737 - Recall(k=50): 0.253303 - NDCG(k=20): 0.137805 - NDCG(k=50): 0.166195 - HitRate(k=20): 0.554391 - HitRate(k=50): 0.693616
2021-01-02 20:34:53,131 P40291 INFO Save best model: monitor(max): 0.162737
2021-01-02 20:34:53,162 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:34:53,267 P40291 INFO Train loss: 0.047505
2021-01-02 20:34:53,267 P40291 INFO ************ Epoch=30 end ************
2021-01-02 20:34:53,268 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:34:59,815 P40291 INFO Negative sampling done
2021-01-02 20:35:23,088 P40291 INFO --- Start evaluation ---
2021-01-02 20:35:24,150 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:35:53,153 P40291 INFO [Metrics] Recall(k=20): 0.162648 - Recall(k=50): 0.253427 - NDCG(k=20): 0.137898 - NDCG(k=50): 0.166352 - HitRate(k=20): 0.554324 - HitRate(k=50): 0.694018
2021-01-02 20:35:53,166 P40291 INFO Monitor(max) STOP: 0.162648 !
2021-01-02 20:35:53,166 P40291 INFO Reduce learning rate on plateau: 0.000010
2021-01-02 20:35:53,166 P40291 INFO Load best model:  MF_gowalla_x0_001_b72fc48a_model.ckpt
2021-01-02 20:35:53,192 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:35:53,290 P40291 INFO Train loss: 0.047485
2021-01-02 20:35:53,290 P40291 INFO ************ Epoch=31 end ************
2021-01-02 20:35:53,290 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:36:00,488 P40291 INFO Negative sampling done
2021-01-02 20:36:23,571 P40291 INFO --- Start evaluation ---
2021-01-02 20:36:24,575 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:36:51,484 P40291 INFO [Metrics] Recall(k=20): 0.162733 - Recall(k=50): 0.253303 - NDCG(k=20): 0.137800 - NDCG(k=50): 0.166191 - HitRate(k=20): 0.554458 - HitRate(k=50): 0.693482
2021-01-02 20:36:51,489 P40291 INFO Monitor(max) STOP: 0.162733 !
2021-01-02 20:36:51,489 P40291 INFO Reduce learning rate on plateau: 0.000001
2021-01-02 20:36:51,489 P40291 INFO Load best model:  MF_gowalla_x0_001_b72fc48a_model.ckpt
2021-01-02 20:36:51,502 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:36:51,576 P40291 INFO Train loss: 0.047135
2021-01-02 20:36:51,576 P40291 INFO ************ Epoch=32 end ************
2021-01-02 20:36:51,577 P40291 INFO Negative sampling num_negs=300
2021-01-02 20:36:58,134 P40291 INFO Negative sampling done
2021-01-02 20:37:20,800 P40291 INFO --- Start evaluation ---
2021-01-02 20:37:21,843 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:37:48,688 P40291 INFO [Metrics] Recall(k=20): 0.162703 - Recall(k=50): 0.253272 - NDCG(k=20): 0.137808 - NDCG(k=50): 0.166201 - HitRate(k=20): 0.554357 - HitRate(k=50): 0.693683
2021-01-02 20:37:48,696 P40291 INFO Monitor(max) STOP: 0.162703 !
2021-01-02 20:37:48,696 P40291 INFO Reduce learning rate on plateau: 0.000001
2021-01-02 20:37:48,696 P40291 INFO Load best model:  MF_gowalla_x0_001_b72fc48a_model.ckpt
2021-01-02 20:37:48,708 P40291 INFO Early stopping at epoch=33
2021-01-02 20:37:48,709 P40291 INFO --- 3165/3165 batches finished ---
2021-01-02 20:37:48,791 P40291 INFO Train loss: 0.047162
2021-01-02 20:37:48,791 P40291 INFO Training finished.
2021-01-02 20:37:48,791 P40291 INFO Load best model:  MF_gowalla_x0_001_b72fc48a_model.ckpt
2021-01-02 20:37:48,811 P40291 INFO ****** Train/validation evaluation ******
2021-01-02 20:37:48,812 P40291 INFO --- Start evaluation ---
2021-01-02 20:37:49,790 P40291 INFO Evaluating metrics for 29858 users...
2021-01-02 20:38:17,007 P40291 INFO [Metrics] Recall(k=20): 0.162737 - Recall(k=50): 0.253303 - NDCG(k=20): 0.137805 - NDCG(k=50): 0.166195 - HitRate(k=20): 0.554391 - HitRate(k=50): 0.693616
```