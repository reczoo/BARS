## YoutubeNet_yelp18_x0 

A notebook to benchmark YoutubeNet on yelp18 dataset.

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
```
2021-03-07 23:45:09,460 P22674 INFO [Metrics] Recall(k=20): 0.068646 - Recall(k=50): 0.131169 - NDCG(k=20): 0.056720 - NDCG(k=50): 0.079797 - HitRate(k=20): 0.410004 - HitRate(k=50): 0.606006
```


### Logs
```
2021-03-07 23:16:20,694 P22674 INFO Set up feature encoder...
2021-03-07 23:16:20,694 P22674 INFO Load feature_map from json: ../data/Yelp18/yelp18_x0_5c65fb11/feature_map.json
2021-03-07 23:16:24,438 P22674 INFO Total number of parameters: 2435200.
2021-03-07 23:16:24,439 P22674 INFO Loading data...
2021-03-07 23:16:24,442 P22674 INFO Loading data from h5: ../data/Yelp18/yelp18_x0_5c65fb11/train.h5
2021-03-07 23:16:25,282 P22674 INFO Loading data from h5: ../data/Yelp18/yelp18_x0_5c65fb11/item_corpus.h5
2021-03-07 23:16:25,693 P22674 INFO Loading data from h5: ../data/Yelp18/yelp18_x0_5c65fb11/valid.h5
2021-03-07 23:16:26,123 P22674 INFO Loading data from h5: ../data/Yelp18/yelp18_x0_5c65fb11/item_corpus.h5
2021-03-07 23:16:26,126 P22674 INFO Train samples: total/1237259, blocks/1
2021-03-07 23:16:26,126 P22674 INFO Validation samples: total/31668, blocks/1
2021-03-07 23:16:26,126 P22674 INFO Loading train data done.
2021-03-07 23:16:26,126 P22674 INFO **** Start training: 4834 batches/epoch ****
2021-03-07 23:16:26,133 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:16:31,751 P22674 INFO Negative sampling done
2021-03-07 23:16:54,880 P22674 INFO --- Start evaluation ---
2021-03-07 23:16:55,521 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:17:16,252 P22674 INFO [Metrics] Recall(k=20): 0.038674 - Recall(k=50): 0.074902 - NDCG(k=20): 0.031855 - NDCG(k=50): 0.045297 - HitRate(k=20): 0.262347 - HitRate(k=50): 0.417835
2021-03-07 23:17:16,261 P22674 INFO Save best model: monitor(max): 0.038674
2021-03-07 23:17:16,270 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:17:16,310 P22674 INFO Train loss: 3.620921
2021-03-07 23:17:16,310 P22674 INFO ************ Epoch=1 end ************
2021-03-07 23:17:16,310 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:17:21,897 P22674 INFO Negative sampling done
2021-03-07 23:17:44,807 P22674 INFO --- Start evaluation ---
2021-03-07 23:17:45,474 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:18:06,658 P22674 INFO [Metrics] Recall(k=20): 0.042764 - Recall(k=50): 0.083465 - NDCG(k=20): 0.035084 - NDCG(k=50): 0.050141 - HitRate(k=20): 0.286188 - HitRate(k=50): 0.457212
2021-03-07 23:18:06,666 P22674 INFO Save best model: monitor(max): 0.042764
2021-03-07 23:18:06,678 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:18:06,726 P22674 INFO Train loss: 3.245390
2021-03-07 23:18:06,726 P22674 INFO ************ Epoch=2 end ************
2021-03-07 23:18:06,726 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:18:12,287 P22674 INFO Negative sampling done
2021-03-07 23:18:35,103 P22674 INFO --- Start evaluation ---
2021-03-07 23:18:35,763 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:18:57,094 P22674 INFO [Metrics] Recall(k=20): 0.047230 - Recall(k=50): 0.092133 - NDCG(k=20): 0.039116 - NDCG(k=50): 0.055753 - HitRate(k=20): 0.310597 - HitRate(k=50): 0.491095
2021-03-07 23:18:57,104 P22674 INFO Save best model: monitor(max): 0.047230
2021-03-07 23:18:57,116 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:18:57,155 P22674 INFO Train loss: 3.115476
2021-03-07 23:18:57,156 P22674 INFO ************ Epoch=3 end ************
2021-03-07 23:18:57,156 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:19:02,709 P22674 INFO Negative sampling done
2021-03-07 23:19:25,720 P22674 INFO --- Start evaluation ---
2021-03-07 23:19:26,397 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:19:48,554 P22674 INFO [Metrics] Recall(k=20): 0.051040 - Recall(k=50): 0.099248 - NDCG(k=20): 0.041888 - NDCG(k=50): 0.059697 - HitRate(k=20): 0.331091 - HitRate(k=50): 0.513326
2021-03-07 23:19:48,563 P22674 INFO Save best model: monitor(max): 0.051040
2021-03-07 23:19:48,575 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:19:48,620 P22674 INFO Train loss: 3.004776
2021-03-07 23:19:48,620 P22674 INFO ************ Epoch=4 end ************
2021-03-07 23:19:48,621 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:19:54,212 P22674 INFO Negative sampling done
2021-03-07 23:20:17,033 P22674 INFO --- Start evaluation ---
2021-03-07 23:20:17,678 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:20:39,283 P22674 INFO [Metrics] Recall(k=20): 0.053977 - Recall(k=50): 0.104150 - NDCG(k=20): 0.044183 - NDCG(k=50): 0.062842 - HitRate(k=20): 0.343280 - HitRate(k=50): 0.531104
2021-03-07 23:20:39,293 P22674 INFO Save best model: monitor(max): 0.053977
2021-03-07 23:20:39,305 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:20:39,347 P22674 INFO Train loss: 2.915432
2021-03-07 23:20:39,347 P22674 INFO ************ Epoch=5 end ************
2021-03-07 23:20:39,347 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:20:44,932 P22674 INFO Negative sampling done
2021-03-07 23:21:07,837 P22674 INFO --- Start evaluation ---
2021-03-07 23:21:08,521 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:21:30,614 P22674 INFO [Metrics] Recall(k=20): 0.055935 - Recall(k=50): 0.107974 - NDCG(k=20): 0.045745 - NDCG(k=50): 0.065061 - HitRate(k=20): 0.352722 - HitRate(k=50): 0.542914
2021-03-07 23:21:30,622 P22674 INFO Save best model: monitor(max): 0.055935
2021-03-07 23:21:30,635 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:21:30,681 P22674 INFO Train loss: 2.848292
2021-03-07 23:21:30,681 P22674 INFO ************ Epoch=6 end ************
2021-03-07 23:21:30,682 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:21:36,344 P22674 INFO Negative sampling done
2021-03-07 23:21:59,373 P22674 INFO --- Start evaluation ---
2021-03-07 23:22:00,044 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:22:22,104 P22674 INFO [Metrics] Recall(k=20): 0.057424 - Recall(k=50): 0.111396 - NDCG(k=20): 0.047083 - NDCG(k=50): 0.067079 - HitRate(k=20): 0.361848 - HitRate(k=50): 0.554471
2021-03-07 23:22:22,113 P22674 INFO Save best model: monitor(max): 0.057424
2021-03-07 23:22:22,125 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:22:22,172 P22674 INFO Train loss: 2.798640
2021-03-07 23:22:22,172 P22674 INFO ************ Epoch=7 end ************
2021-03-07 23:22:22,173 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:22:27,836 P22674 INFO Negative sampling done
2021-03-07 23:22:50,627 P22674 INFO --- Start evaluation ---
2021-03-07 23:22:51,303 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:23:13,127 P22674 INFO [Metrics] Recall(k=20): 0.059307 - Recall(k=50): 0.114265 - NDCG(k=20): 0.048638 - NDCG(k=50): 0.069031 - HitRate(k=20): 0.368574 - HitRate(k=50): 0.562050
2021-03-07 23:23:13,136 P22674 INFO Save best model: monitor(max): 0.059307
2021-03-07 23:23:13,148 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:23:13,194 P22674 INFO Train loss: 2.760862
2021-03-07 23:23:13,195 P22674 INFO ************ Epoch=8 end ************
2021-03-07 23:23:13,195 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:23:18,781 P22674 INFO Negative sampling done
2021-03-07 23:23:41,749 P22674 INFO --- Start evaluation ---
2021-03-07 23:23:42,442 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:24:05,092 P22674 INFO [Metrics] Recall(k=20): 0.060977 - Recall(k=50): 0.116941 - NDCG(k=20): 0.049883 - NDCG(k=50): 0.070541 - HitRate(k=20): 0.376753 - HitRate(k=50): 0.570260
2021-03-07 23:24:05,101 P22674 INFO Save best model: monitor(max): 0.060977
2021-03-07 23:24:05,113 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:24:05,159 P22674 INFO Train loss: 2.729547
2021-03-07 23:24:05,159 P22674 INFO ************ Epoch=9 end ************
2021-03-07 23:24:05,159 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:24:10,695 P22674 INFO Negative sampling done
2021-03-07 23:24:33,570 P22674 INFO --- Start evaluation ---
2021-03-07 23:24:34,228 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:24:56,607 P22674 INFO [Metrics] Recall(k=20): 0.062148 - Recall(k=50): 0.119010 - NDCG(k=20): 0.050676 - NDCG(k=50): 0.071714 - HitRate(k=20): 0.382247 - HitRate(k=50): 0.576860
2021-03-07 23:24:56,614 P22674 INFO Save best model: monitor(max): 0.062148
2021-03-07 23:24:56,626 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:24:56,675 P22674 INFO Train loss: 2.701127
2021-03-07 23:24:56,676 P22674 INFO ************ Epoch=10 end ************
2021-03-07 23:24:56,676 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:25:02,225 P22674 INFO Negative sampling done
2021-03-07 23:25:25,139 P22674 INFO --- Start evaluation ---
2021-03-07 23:25:25,829 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:25:47,669 P22674 INFO [Metrics] Recall(k=20): 0.063213 - Recall(k=50): 0.121540 - NDCG(k=20): 0.051660 - NDCG(k=50): 0.073217 - HitRate(k=20): 0.388405 - HitRate(k=50): 0.580965
2021-03-07 23:25:47,676 P22674 INFO Save best model: monitor(max): 0.063213
2021-03-07 23:25:47,688 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:25:47,735 P22674 INFO Train loss: 2.675190
2021-03-07 23:25:47,736 P22674 INFO ************ Epoch=11 end ************
2021-03-07 23:25:47,736 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:25:53,339 P22674 INFO Negative sampling done
2021-03-07 23:26:16,000 P22674 INFO --- Start evaluation ---
2021-03-07 23:26:16,688 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:26:39,323 P22674 INFO [Metrics] Recall(k=20): 0.064040 - Recall(k=50): 0.122340 - NDCG(k=20): 0.052449 - NDCG(k=50): 0.074037 - HitRate(k=20): 0.389099 - HitRate(k=50): 0.584723
2021-03-07 23:26:39,332 P22674 INFO Save best model: monitor(max): 0.064040
2021-03-07 23:26:39,343 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:26:39,399 P22674 INFO Train loss: 2.651395
2021-03-07 23:26:39,400 P22674 INFO ************ Epoch=12 end ************
2021-03-07 23:26:39,400 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:26:44,971 P22674 INFO Negative sampling done
2021-03-07 23:27:07,546 P22674 INFO --- Start evaluation ---
2021-03-07 23:27:08,218 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:27:30,802 P22674 INFO [Metrics] Recall(k=20): 0.064613 - Recall(k=50): 0.124641 - NDCG(k=20): 0.052911 - NDCG(k=50): 0.075095 - HitRate(k=20): 0.392573 - HitRate(k=50): 0.589144
2021-03-07 23:27:30,811 P22674 INFO Save best model: monitor(max): 0.064613
2021-03-07 23:27:30,822 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:27:30,868 P22674 INFO Train loss: 2.629526
2021-03-07 23:27:30,868 P22674 INFO ************ Epoch=13 end ************
2021-03-07 23:27:30,869 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:27:36,431 P22674 INFO Negative sampling done
2021-03-07 23:27:58,929 P22674 INFO --- Start evaluation ---
2021-03-07 23:27:59,613 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:28:21,860 P22674 INFO [Metrics] Recall(k=20): 0.065385 - Recall(k=50): 0.126399 - NDCG(k=20): 0.053777 - NDCG(k=50): 0.076227 - HitRate(k=20): 0.395983 - HitRate(k=50): 0.594733
2021-03-07 23:28:21,869 P22674 INFO Save best model: monitor(max): 0.065385
2021-03-07 23:28:21,880 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:28:21,926 P22674 INFO Train loss: 2.609507
2021-03-07 23:28:21,926 P22674 INFO ************ Epoch=14 end ************
2021-03-07 23:28:21,927 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:28:27,475 P22674 INFO Negative sampling done
2021-03-07 23:28:50,005 P22674 INFO --- Start evaluation ---
2021-03-07 23:28:50,683 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:29:13,474 P22674 INFO [Metrics] Recall(k=20): 0.065950 - Recall(k=50): 0.126499 - NDCG(k=20): 0.054278 - NDCG(k=50): 0.076618 - HitRate(k=20): 0.400467 - HitRate(k=50): 0.595775
2021-03-07 23:29:13,482 P22674 INFO Save best model: monitor(max): 0.065950
2021-03-07 23:29:13,494 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:29:13,545 P22674 INFO Train loss: 2.590145
2021-03-07 23:29:13,545 P22674 INFO ************ Epoch=15 end ************
2021-03-07 23:29:13,546 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:29:19,096 P22674 INFO Negative sampling done
2021-03-07 23:29:41,804 P22674 INFO --- Start evaluation ---
2021-03-07 23:29:42,466 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:30:05,317 P22674 INFO [Metrics] Recall(k=20): 0.066267 - Recall(k=50): 0.127813 - NDCG(k=20): 0.054696 - NDCG(k=50): 0.077400 - HitRate(k=20): 0.399804 - HitRate(k=50): 0.599817
2021-03-07 23:30:05,325 P22674 INFO Save best model: monitor(max): 0.066267
2021-03-07 23:30:05,337 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:30:05,384 P22674 INFO Train loss: 2.572098
2021-03-07 23:30:05,384 P22674 INFO ************ Epoch=16 end ************
2021-03-07 23:30:05,384 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:30:10,962 P22674 INFO Negative sampling done
2021-03-07 23:30:33,526 P22674 INFO --- Start evaluation ---
2021-03-07 23:30:34,208 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:30:56,777 P22674 INFO [Metrics] Recall(k=20): 0.066651 - Recall(k=50): 0.128365 - NDCG(k=20): 0.054840 - NDCG(k=50): 0.077722 - HitRate(k=20): 0.400720 - HitRate(k=50): 0.599943
2021-03-07 23:30:56,786 P22674 INFO Save best model: monitor(max): 0.066651
2021-03-07 23:30:56,798 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:30:56,848 P22674 INFO Train loss: 2.555371
2021-03-07 23:30:56,848 P22674 INFO ************ Epoch=17 end ************
2021-03-07 23:30:56,848 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:31:02,429 P22674 INFO Negative sampling done
2021-03-07 23:31:25,177 P22674 INFO --- Start evaluation ---
2021-03-07 23:31:25,853 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:31:49,238 P22674 INFO [Metrics] Recall(k=20): 0.067373 - Recall(k=50): 0.128987 - NDCG(k=20): 0.055475 - NDCG(k=50): 0.078272 - HitRate(k=20): 0.405930 - HitRate(k=50): 0.600733
2021-03-07 23:31:49,247 P22674 INFO Save best model: monitor(max): 0.067373
2021-03-07 23:31:49,259 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:31:49,316 P22674 INFO Train loss: 2.539137
2021-03-07 23:31:49,316 P22674 INFO ************ Epoch=18 end ************
2021-03-07 23:31:49,317 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:31:54,910 P22674 INFO Negative sampling done
2021-03-07 23:32:17,454 P22674 INFO --- Start evaluation ---
2021-03-07 23:32:18,122 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:32:40,765 P22674 INFO [Metrics] Recall(k=20): 0.067020 - Recall(k=50): 0.129351 - NDCG(k=20): 0.055352 - NDCG(k=50): 0.078344 - HitRate(k=20): 0.403183 - HitRate(k=50): 0.601743
2021-03-07 23:32:40,773 P22674 INFO Monitor(max) STOP: 0.067020 !
2021-03-07 23:32:40,773 P22674 INFO Reduce learning rate on plateau: 0.000100
2021-03-07 23:32:40,773 P22674 INFO Load best model: YoutubeDNN_yelp18_x0_001_9677fc1e.model
2021-03-07 23:32:40,781 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:32:40,826 P22674 INFO Train loss: 2.524245
2021-03-07 23:32:40,826 P22674 INFO ************ Epoch=19 end ************
2021-03-07 23:32:40,827 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:32:46,407 P22674 INFO Negative sampling done
2021-03-07 23:33:09,230 P22674 INFO --- Start evaluation ---
2021-03-07 23:33:09,902 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:33:32,058 P22674 INFO [Metrics] Recall(k=20): 0.068052 - Recall(k=50): 0.130539 - NDCG(k=20): 0.056314 - NDCG(k=50): 0.079400 - HitRate(k=20): 0.408456 - HitRate(k=50): 0.604206
2021-03-07 23:33:32,067 P22674 INFO Save best model: monitor(max): 0.068052
2021-03-07 23:33:32,078 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:33:32,126 P22674 INFO Train loss: 2.483828
2021-03-07 23:33:32,126 P22674 INFO ************ Epoch=20 end ************
2021-03-07 23:33:32,127 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:33:37,718 P22674 INFO Negative sampling done
2021-03-07 23:34:00,768 P22674 INFO --- Start evaluation ---
2021-03-07 23:34:01,443 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:34:23,532 P22674 INFO [Metrics] Recall(k=20): 0.068589 - Recall(k=50): 0.131115 - NDCG(k=20): 0.056686 - NDCG(k=50): 0.079769 - HitRate(k=20): 0.409846 - HitRate(k=50): 0.605974
2021-03-07 23:34:23,541 P22674 INFO Save best model: monitor(max): 0.068589
2021-03-07 23:34:23,553 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:34:23,610 P22674 INFO Train loss: 2.477729
2021-03-07 23:34:23,610 P22674 INFO ************ Epoch=21 end ************
2021-03-07 23:34:23,610 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:34:29,199 P22674 INFO Negative sampling done
2021-03-07 23:34:53,152 P22674 INFO --- Start evaluation ---
2021-03-07 23:34:53,848 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:35:16,591 P22674 INFO [Metrics] Recall(k=20): 0.068353 - Recall(k=50): 0.131109 - NDCG(k=20): 0.056643 - NDCG(k=50): 0.079811 - HitRate(k=20): 0.408235 - HitRate(k=50): 0.605343
2021-03-07 23:35:16,600 P22674 INFO Monitor(max) STOP: 0.068353 !
2021-03-07 23:35:16,600 P22674 INFO Reduce learning rate on plateau: 0.000010
2021-03-07 23:35:16,600 P22674 INFO Load best model: YoutubeDNN_yelp18_x0_001_9677fc1e.model
2021-03-07 23:35:16,607 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:35:16,653 P22674 INFO Train loss: 2.473840
2021-03-07 23:35:16,654 P22674 INFO ************ Epoch=22 end ************
2021-03-07 23:35:16,654 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:35:22,282 P22674 INFO Negative sampling done
2021-03-07 23:35:45,265 P22674 INFO --- Start evaluation ---
2021-03-07 23:35:45,927 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:36:08,196 P22674 INFO [Metrics] Recall(k=20): 0.068573 - Recall(k=50): 0.131165 - NDCG(k=20): 0.056684 - NDCG(k=50): 0.079796 - HitRate(k=20): 0.409688 - HitRate(k=50): 0.605848
2021-03-07 23:36:08,205 P22674 INFO Monitor(max) STOP: 0.068573 !
2021-03-07 23:36:08,205 P22674 INFO Reduce learning rate on plateau: 0.000001
2021-03-07 23:36:08,205 P22674 INFO Load best model: YoutubeDNN_yelp18_x0_001_9677fc1e.model
2021-03-07 23:36:08,213 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:36:08,252 P22674 INFO Train loss: 2.468032
2021-03-07 23:36:08,252 P22674 INFO ************ Epoch=23 end ************
2021-03-07 23:36:08,253 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:36:13,858 P22674 INFO Negative sampling done
2021-03-07 23:36:36,926 P22674 INFO --- Start evaluation ---
2021-03-07 23:36:37,560 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:36:59,673 P22674 INFO [Metrics] Recall(k=20): 0.068597 - Recall(k=50): 0.131098 - NDCG(k=20): 0.056700 - NDCG(k=50): 0.079773 - HitRate(k=20): 0.409877 - HitRate(k=50): 0.605880
2021-03-07 23:36:59,682 P22674 INFO Save best model: monitor(max): 0.068597
2021-03-07 23:36:59,694 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:36:59,739 P22674 INFO Train loss: 2.467424
2021-03-07 23:36:59,739 P22674 INFO ************ Epoch=24 end ************
2021-03-07 23:36:59,739 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:37:05,323 P22674 INFO Negative sampling done
2021-03-07 23:37:28,422 P22674 INFO --- Start evaluation ---
2021-03-07 23:37:29,119 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:37:51,212 P22674 INFO [Metrics] Recall(k=20): 0.068623 - Recall(k=50): 0.131164 - NDCG(k=20): 0.056715 - NDCG(k=50): 0.079798 - HitRate(k=20): 0.409909 - HitRate(k=50): 0.606069
2021-03-07 23:37:51,220 P22674 INFO Save best model: monitor(max): 0.068623
2021-03-07 23:37:51,233 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:37:51,278 P22674 INFO Train loss: 2.467703
2021-03-07 23:37:51,279 P22674 INFO ************ Epoch=25 end ************
2021-03-07 23:37:51,279 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:37:56,880 P22674 INFO Negative sampling done
2021-03-07 23:38:19,699 P22674 INFO --- Start evaluation ---
2021-03-07 23:38:20,388 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:38:43,147 P22674 INFO [Metrics] Recall(k=20): 0.068611 - Recall(k=50): 0.131179 - NDCG(k=20): 0.056709 - NDCG(k=50): 0.079802 - HitRate(k=20): 0.409814 - HitRate(k=50): 0.606006
2021-03-07 23:38:43,154 P22674 INFO Monitor(max) STOP: 0.068611 !
2021-03-07 23:38:43,154 P22674 INFO Reduce learning rate on plateau: 0.000001
2021-03-07 23:38:43,155 P22674 INFO Load best model: YoutubeDNN_yelp18_x0_001_9677fc1e.model
2021-03-07 23:38:43,162 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:38:43,208 P22674 INFO Train loss: 2.468073
2021-03-07 23:38:43,208 P22674 INFO ************ Epoch=26 end ************
2021-03-07 23:38:43,209 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:38:48,819 P22674 INFO Negative sampling done
2021-03-07 23:39:11,556 P22674 INFO --- Start evaluation ---
2021-03-07 23:39:12,238 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:39:34,498 P22674 INFO [Metrics] Recall(k=20): 0.068627 - Recall(k=50): 0.131171 - NDCG(k=20): 0.056716 - NDCG(k=50): 0.079799 - HitRate(k=20): 0.409877 - HitRate(k=50): 0.606038
2021-03-07 23:39:34,506 P22674 INFO Save best model: monitor(max): 0.068627
2021-03-07 23:39:34,518 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:39:34,566 P22674 INFO Train loss: 2.467642
2021-03-07 23:39:34,566 P22674 INFO ************ Epoch=27 end ************
2021-03-07 23:39:34,567 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:39:40,154 P22674 INFO Negative sampling done
2021-03-07 23:40:03,108 P22674 INFO --- Start evaluation ---
2021-03-07 23:40:03,780 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:40:25,931 P22674 INFO [Metrics] Recall(k=20): 0.068627 - Recall(k=50): 0.131145 - NDCG(k=20): 0.056706 - NDCG(k=50): 0.079780 - HitRate(k=20): 0.409877 - HitRate(k=50): 0.605911
2021-03-07 23:40:25,940 P22674 INFO Monitor(max) STOP: 0.068627 !
2021-03-07 23:40:25,940 P22674 INFO Reduce learning rate on plateau: 0.000001
2021-03-07 23:40:25,940 P22674 INFO Load best model: YoutubeDNN_yelp18_x0_001_9677fc1e.model
2021-03-07 23:40:25,947 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:40:25,996 P22674 INFO Train loss: 2.467259
2021-03-07 23:40:25,996 P22674 INFO ************ Epoch=28 end ************
2021-03-07 23:40:25,997 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:40:31,573 P22674 INFO Negative sampling done
2021-03-07 23:40:54,529 P22674 INFO --- Start evaluation ---
2021-03-07 23:40:55,211 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:41:18,247 P22674 INFO [Metrics] Recall(k=20): 0.068612 - Recall(k=50): 0.131179 - NDCG(k=20): 0.056709 - NDCG(k=50): 0.079802 - HitRate(k=20): 0.409941 - HitRate(k=50): 0.606006
2021-03-07 23:41:18,255 P22674 INFO Monitor(max) STOP: 0.068612 !
2021-03-07 23:41:18,255 P22674 INFO Reduce learning rate on plateau: 0.000001
2021-03-07 23:41:18,255 P22674 INFO Load best model: YoutubeDNN_yelp18_x0_001_9677fc1e.model
2021-03-07 23:41:18,262 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:41:18,310 P22674 INFO Train loss: 2.467526
2021-03-07 23:41:18,310 P22674 INFO ************ Epoch=29 end ************
2021-03-07 23:41:18,310 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:41:23,882 P22674 INFO Negative sampling done
2021-03-07 23:41:46,822 P22674 INFO --- Start evaluation ---
2021-03-07 23:41:47,511 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:42:10,049 P22674 INFO [Metrics] Recall(k=20): 0.068646 - Recall(k=50): 0.131169 - NDCG(k=20): 0.056720 - NDCG(k=50): 0.079797 - HitRate(k=20): 0.410004 - HitRate(k=50): 0.606006
2021-03-07 23:42:10,058 P22674 INFO Save best model: monitor(max): 0.068646
2021-03-07 23:42:10,069 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:42:10,115 P22674 INFO Train loss: 2.467494
2021-03-07 23:42:10,115 P22674 INFO ************ Epoch=30 end ************
2021-03-07 23:42:10,116 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:42:15,725 P22674 INFO Negative sampling done
2021-03-07 23:42:38,725 P22674 INFO --- Start evaluation ---
2021-03-07 23:42:39,404 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:43:02,115 P22674 INFO [Metrics] Recall(k=20): 0.068623 - Recall(k=50): 0.131139 - NDCG(k=20): 0.056709 - NDCG(k=50): 0.079786 - HitRate(k=20): 0.409972 - HitRate(k=50): 0.605974
2021-03-07 23:43:02,123 P22674 INFO Monitor(max) STOP: 0.068623 !
2021-03-07 23:43:02,124 P22674 INFO Reduce learning rate on plateau: 0.000001
2021-03-07 23:43:02,124 P22674 INFO Load best model: YoutubeDNN_yelp18_x0_001_9677fc1e.model
2021-03-07 23:43:02,132 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:43:02,185 P22674 INFO Train loss: 2.467122
2021-03-07 23:43:02,185 P22674 INFO ************ Epoch=31 end ************
2021-03-07 23:43:02,186 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:43:07,743 P22674 INFO Negative sampling done
2021-03-07 23:43:30,638 P22674 INFO --- Start evaluation ---
2021-03-07 23:43:31,329 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:43:54,168 P22674 INFO [Metrics] Recall(k=20): 0.068619 - Recall(k=50): 0.131160 - NDCG(k=20): 0.056712 - NDCG(k=50): 0.079798 - HitRate(k=20): 0.409972 - HitRate(k=50): 0.606006
2021-03-07 23:43:54,176 P22674 INFO Monitor(max) STOP: 0.068619 !
2021-03-07 23:43:54,177 P22674 INFO Reduce learning rate on plateau: 0.000001
2021-03-07 23:43:54,177 P22674 INFO Load best model: YoutubeDNN_yelp18_x0_001_9677fc1e.model
2021-03-07 23:43:54,183 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:43:54,229 P22674 INFO Train loss: 2.467861
2021-03-07 23:43:54,229 P22674 INFO ************ Epoch=32 end ************
2021-03-07 23:43:54,229 P22674 INFO Negative sampling num_negs=200
2021-03-07 23:43:59,848 P22674 INFO Negative sampling done
2021-03-07 23:44:22,916 P22674 INFO --- Start evaluation ---
2021-03-07 23:44:23,595 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:44:45,957 P22674 INFO [Metrics] Recall(k=20): 0.068635 - Recall(k=50): 0.131154 - NDCG(k=20): 0.056716 - NDCG(k=50): 0.079793 - HitRate(k=20): 0.409972 - HitRate(k=50): 0.605911
2021-03-07 23:44:45,965 P22674 INFO Monitor(max) STOP: 0.068635 !
2021-03-07 23:44:45,965 P22674 INFO Reduce learning rate on plateau: 0.000001
2021-03-07 23:44:45,965 P22674 INFO Load best model: YoutubeDNN_yelp18_x0_001_9677fc1e.model
2021-03-07 23:44:45,972 P22674 INFO Early stopping at epoch=33
2021-03-07 23:44:45,973 P22674 INFO --- 4834/4834 batches finished ---
2021-03-07 23:44:46,020 P22674 INFO Train loss: 2.467841
2021-03-07 23:44:46,020 P22674 INFO Training finished.
2021-03-07 23:44:46,020 P22674 INFO Load best model: YoutubeDNN_yelp18_x0_001_9677fc1e.model
2021-03-07 23:44:46,029 P22674 INFO ****** Train/validation evaluation ******
2021-03-07 23:44:46,029 P22674 INFO --- Start evaluation ---
2021-03-07 23:44:46,690 P22674 INFO Evaluating metrics for 31668 users...
2021-03-07 23:45:09,460 P22674 INFO [Metrics] Recall(k=20): 0.068646 - Recall(k=50): 0.131169 - NDCG(k=20): 0.056720 - NDCG(k=50): 0.079797 - HitRate(k=20): 0.410004 - HitRate(k=50): 0.606006


```