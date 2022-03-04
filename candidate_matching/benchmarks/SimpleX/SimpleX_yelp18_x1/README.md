## SimpleX_yelp18_x0 

A notebook to benchmark SimpleX on yelp18 dataset.

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
2021-01-14 21:02:01,306 P19772 INFO [Metrics] Recall(k=20): 0.070111 - Recall(k=50): 0.132237 - NDCG(k=20): 0.057514 - NDCG(k=50): 0.080518 - HitRate(k=20): 0.415151 - HitRate(k=50): 0.608753
```


### Logs
```
2021-01-14 18:24:29,535 P19772 INFO Set up feature encoder...
2021-01-14 18:24:29,536 P19772 INFO Load feature_map from json: ../data/Yelp18/yelp18_x0_9217a019/feature_map.json
2021-01-14 18:24:34,959 P19772 INFO Total number of parameters: 4466112.
2021-01-14 18:24:34,960 P19772 INFO Loading data...
2021-01-14 18:24:34,963 P19772 INFO Loading data from h5: ../data/Yelp18/yelp18_x0_9217a019/train.h5
2021-01-14 18:24:36,011 P19772 INFO Loading data from h5: ../data/Yelp18/yelp18_x0_9217a019/item_corpus.h5
2021-01-14 18:24:36,520 P19772 INFO Loading data from h5: ../data/Yelp18/yelp18_x0_9217a019/valid.h5
2021-01-14 18:24:36,966 P19772 INFO Loading data from h5: ../data/Yelp18/yelp18_x0_9217a019/item_corpus.h5
2021-01-14 18:24:36,968 P19772 INFO Train samples: total/1237259, blocks/1
2021-01-14 18:24:36,968 P19772 INFO Validation samples: total/31668, blocks/1
2021-01-14 18:24:36,968 P19772 INFO Loading train data done.
2021-01-14 18:24:36,968 P19772 INFO **** Start training: 2417 batches/epoch ****
2021-01-14 18:24:36,970 P19772 INFO Negative sampling num_negs=1000
2021-01-14 18:25:04,587 P19772 INFO Negative sampling done
2021-01-14 18:28:29,595 P19772 INFO --- Start evaluation ---
2021-01-14 18:28:30,944 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 18:28:58,776 P19772 INFO [Metrics] Recall(k=20): 0.033943 - Recall(k=50): 0.070097 - NDCG(k=20): 0.027329 - NDCG(k=50): 0.040822 - HitRate(k=20): 0.245705 - HitRate(k=50): 0.413098
2021-01-14 18:28:58,791 P19772 INFO Save best model: monitor(max): 0.033943
2021-01-14 18:28:58,809 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 18:28:58,871 P19772 INFO Train loss: 0.471202
2021-01-14 18:28:58,871 P19772 INFO ************ Epoch=1 end ************
2021-01-14 18:28:58,871 P19772 INFO Negative sampling num_negs=1000
2021-01-14 18:29:29,913 P19772 INFO Negative sampling done
2021-01-14 18:32:44,931 P19772 INFO --- Start evaluation ---
2021-01-14 18:32:46,513 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 18:33:10,684 P19772 INFO [Metrics] Recall(k=20): 0.043586 - Recall(k=50): 0.088035 - NDCG(k=20): 0.034997 - NDCG(k=50): 0.051615 - HitRate(k=20): 0.298472 - HitRate(k=50): 0.480043
2021-01-14 18:33:10,697 P19772 INFO Save best model: monitor(max): 0.043586
2021-01-14 18:33:10,719 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 18:33:10,771 P19772 INFO Train loss: 0.198615
2021-01-14 18:33:10,771 P19772 INFO ************ Epoch=2 end ************
2021-01-14 18:33:10,772 P19772 INFO Negative sampling num_negs=1000
2021-01-14 18:33:38,605 P19772 INFO Negative sampling done
2021-01-14 18:36:52,166 P19772 INFO --- Start evaluation ---
2021-01-14 18:36:53,762 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 18:37:18,230 P19772 INFO [Metrics] Recall(k=20): 0.047923 - Recall(k=50): 0.095725 - NDCG(k=20): 0.038628 - NDCG(k=50): 0.056432 - HitRate(k=20): 0.321144 - HitRate(k=50): 0.508336
2021-01-14 18:37:18,243 P19772 INFO Save best model: monitor(max): 0.047923
2021-01-14 18:37:18,264 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 18:37:18,336 P19772 INFO Train loss: 0.183599
2021-01-14 18:37:18,336 P19772 INFO ************ Epoch=3 end ************
2021-01-14 18:37:18,337 P19772 INFO Negative sampling num_negs=1000
2021-01-14 18:37:47,702 P19772 INFO Negative sampling done
2021-01-14 18:41:03,504 P19772 INFO --- Start evaluation ---
2021-01-14 18:41:05,089 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 18:41:29,779 P19772 INFO [Metrics] Recall(k=20): 0.052673 - Recall(k=50): 0.103790 - NDCG(k=20): 0.042144 - NDCG(k=50): 0.061149 - HitRate(k=20): 0.341575 - HitRate(k=50): 0.530504
2021-01-14 18:41:29,793 P19772 INFO Save best model: monitor(max): 0.052673
2021-01-14 18:41:29,814 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 18:41:29,870 P19772 INFO Train loss: 0.174642
2021-01-14 18:41:29,870 P19772 INFO ************ Epoch=4 end ************
2021-01-14 18:41:29,870 P19772 INFO Negative sampling num_negs=1000
2021-01-14 18:41:57,153 P19772 INFO Negative sampling done
2021-01-14 18:45:18,325 P19772 INFO --- Start evaluation ---
2021-01-14 18:45:19,950 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 18:45:44,681 P19772 INFO [Metrics] Recall(k=20): 0.054972 - Recall(k=50): 0.108370 - NDCG(k=20): 0.044435 - NDCG(k=50): 0.064304 - HitRate(k=20): 0.351554 - HitRate(k=50): 0.544114
2021-01-14 18:45:44,693 P19772 INFO Save best model: monitor(max): 0.054972
2021-01-14 18:45:44,715 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 18:45:44,767 P19772 INFO Train loss: 0.168399
2021-01-14 18:45:44,767 P19772 INFO ************ Epoch=5 end ************
2021-01-14 18:45:44,768 P19772 INFO Negative sampling num_negs=1000
2021-01-14 18:46:12,334 P19772 INFO Negative sampling done
2021-01-14 18:49:27,356 P19772 INFO --- Start evaluation ---
2021-01-14 18:49:28,970 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 18:49:53,818 P19772 INFO [Metrics] Recall(k=20): 0.057973 - Recall(k=50): 0.112817 - NDCG(k=20): 0.046887 - NDCG(k=50): 0.067305 - HitRate(k=20): 0.365479 - HitRate(k=50): 0.556713
2021-01-14 18:49:53,831 P19772 INFO Save best model: monitor(max): 0.057973
2021-01-14 18:49:53,852 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 18:49:53,906 P19772 INFO Train loss: 0.163667
2021-01-14 18:49:53,907 P19772 INFO ************ Epoch=6 end ************
2021-01-14 18:49:53,907 P19772 INFO Negative sampling num_negs=1000
2021-01-14 18:50:22,246 P19772 INFO Negative sampling done
2021-01-14 18:53:22,002 P19772 INFO --- Start evaluation ---
2021-01-14 18:53:23,347 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 18:53:50,409 P19772 INFO [Metrics] Recall(k=20): 0.059234 - Recall(k=50): 0.115363 - NDCG(k=20): 0.048456 - NDCG(k=50): 0.069328 - HitRate(k=20): 0.369079 - HitRate(k=50): 0.564324
2021-01-14 18:53:50,421 P19772 INFO Save best model: monitor(max): 0.059234
2021-01-14 18:53:50,442 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 18:53:50,495 P19772 INFO Train loss: 0.160047
2021-01-14 18:53:50,496 P19772 INFO ************ Epoch=7 end ************
2021-01-14 18:53:50,496 P19772 INFO Negative sampling num_negs=1000
2021-01-14 18:54:19,172 P19772 INFO Negative sampling done
2021-01-14 18:57:04,959 P19772 INFO --- Start evaluation ---
2021-01-14 18:57:06,319 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 18:57:30,989 P19772 INFO [Metrics] Recall(k=20): 0.060172 - Recall(k=50): 0.116692 - NDCG(k=20): 0.048881 - NDCG(k=50): 0.069852 - HitRate(k=20): 0.374542 - HitRate(k=50): 0.568681
2021-01-14 18:57:31,001 P19772 INFO Save best model: monitor(max): 0.060172
2021-01-14 18:57:31,022 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 18:57:31,076 P19772 INFO Train loss: 0.157248
2021-01-14 18:57:31,076 P19772 INFO ************ Epoch=8 end ************
2021-01-14 18:57:31,077 P19772 INFO Negative sampling num_negs=1000
2021-01-14 18:57:58,514 P19772 INFO Negative sampling done
2021-01-14 19:00:45,160 P19772 INFO --- Start evaluation ---
2021-01-14 19:00:47,274 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 19:01:12,033 P19772 INFO [Metrics] Recall(k=20): 0.062063 - Recall(k=50): 0.120754 - NDCG(k=20): 0.050696 - NDCG(k=50): 0.072433 - HitRate(k=20): 0.382152 - HitRate(k=50): 0.576228
2021-01-14 19:01:12,046 P19772 INFO Save best model: monitor(max): 0.062063
2021-01-14 19:01:12,066 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 19:01:12,127 P19772 INFO Train loss: 0.154829
2021-01-14 19:01:12,127 P19772 INFO ************ Epoch=9 end ************
2021-01-14 19:01:12,128 P19772 INFO Negative sampling num_negs=1000
2021-01-14 19:01:40,762 P19772 INFO Negative sampling done
2021-01-14 19:04:24,322 P19772 INFO --- Start evaluation ---
2021-01-14 19:04:25,688 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 19:04:50,629 P19772 INFO [Metrics] Recall(k=20): 0.063353 - Recall(k=50): 0.121034 - NDCG(k=20): 0.051592 - NDCG(k=50): 0.072959 - HitRate(k=20): 0.388310 - HitRate(k=50): 0.580239
2021-01-14 19:04:50,644 P19772 INFO Save best model: monitor(max): 0.063353
2021-01-14 19:04:50,666 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 19:04:50,728 P19772 INFO Train loss: 0.152958
2021-01-14 19:04:50,729 P19772 INFO ************ Epoch=10 end ************
2021-01-14 19:04:50,729 P19772 INFO Negative sampling num_negs=1000
2021-01-14 19:05:18,511 P19772 INFO Negative sampling done
2021-01-14 19:08:01,696 P19772 INFO --- Start evaluation ---
2021-01-14 19:08:03,067 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 19:08:29,472 P19772 INFO [Metrics] Recall(k=20): 0.063348 - Recall(k=50): 0.121333 - NDCG(k=20): 0.051705 - NDCG(k=50): 0.073261 - HitRate(k=20): 0.386384 - HitRate(k=50): 0.581028
2021-01-14 19:08:29,486 P19772 INFO Monitor(max) STOP: 0.063348 !
2021-01-14 19:08:29,486 P19772 INFO Reduce learning rate on plateau: 0.000010
2021-01-14 19:08:29,486 P19772 INFO Load best model:  SimpleX_yelp18_x0_034_297a4b82.model
2021-01-14 19:08:29,500 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 19:08:29,561 P19772 INFO Train loss: 0.151518
2021-01-14 19:08:29,561 P19772 INFO ************ Epoch=11 end ************
2021-01-14 19:08:29,562 P19772 INFO Negative sampling num_negs=1000
2021-01-14 19:08:57,032 P19772 INFO Negative sampling done
2021-01-14 19:11:41,067 P19772 INFO --- Start evaluation ---
2021-01-14 19:11:42,471 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 19:12:07,831 P19772 INFO [Metrics] Recall(k=20): 0.066283 - Recall(k=50): 0.126734 - NDCG(k=20): 0.054381 - NDCG(k=50): 0.076743 - HitRate(k=20): 0.400404 - HitRate(k=50): 0.596091
2021-01-14 19:12:07,844 P19772 INFO Save best model: monitor(max): 0.066283
2021-01-14 19:12:07,867 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 19:12:07,931 P19772 INFO Train loss: 0.141026
2021-01-14 19:12:07,932 P19772 INFO ************ Epoch=12 end ************
2021-01-14 19:12:07,932 P19772 INFO Negative sampling num_negs=1000
2021-01-14 19:12:35,390 P19772 INFO Negative sampling done
2021-01-14 19:15:18,746 P19772 INFO --- Start evaluation ---
2021-01-14 19:15:20,134 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 19:15:45,262 P19772 INFO [Metrics] Recall(k=20): 0.067191 - Recall(k=50): 0.128240 - NDCG(k=20): 0.055176 - NDCG(k=50): 0.077777 - HitRate(k=20): 0.404320 - HitRate(k=50): 0.599817
2021-01-14 19:15:45,277 P19772 INFO Save best model: monitor(max): 0.067191
2021-01-14 19:15:45,299 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 19:15:45,361 P19772 INFO Train loss: 0.138436
2021-01-14 19:15:45,361 P19772 INFO ************ Epoch=13 end ************
2021-01-14 19:15:45,362 P19772 INFO Negative sampling num_negs=1000
2021-01-14 19:16:13,354 P19772 INFO Negative sampling done
2021-01-14 19:18:56,904 P19772 INFO --- Start evaluation ---
2021-01-14 19:18:58,334 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 19:19:23,889 P19772 INFO [Metrics] Recall(k=20): 0.067869 - Recall(k=50): 0.129253 - NDCG(k=20): 0.055685 - NDCG(k=50): 0.078418 - HitRate(k=20): 0.406088 - HitRate(k=50): 0.602564
2021-01-14 19:19:23,904 P19772 INFO Save best model: monitor(max): 0.067869
2021-01-14 19:19:23,927 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 19:19:24,003 P19772 INFO Train loss: 0.137362
2021-01-14 19:19:24,003 P19772 INFO ************ Epoch=14 end ************
2021-01-14 19:19:24,004 P19772 INFO Negative sampling num_negs=1000
2021-01-14 19:19:51,904 P19772 INFO Negative sampling done
2021-01-14 19:22:36,291 P19772 INFO --- Start evaluation ---
2021-01-14 19:22:37,722 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 19:23:02,122 P19772 INFO [Metrics] Recall(k=20): 0.068474 - Recall(k=50): 0.130325 - NDCG(k=20): 0.056029 - NDCG(k=50): 0.078900 - HitRate(k=20): 0.408804 - HitRate(k=50): 0.605311
2021-01-14 19:23:02,136 P19772 INFO Save best model: monitor(max): 0.068474
2021-01-14 19:23:02,158 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 19:23:02,218 P19772 INFO Train loss: 0.136661
2021-01-14 19:23:02,218 P19772 INFO ************ Epoch=15 end ************
2021-01-14 19:23:02,219 P19772 INFO Negative sampling num_negs=1000
2021-01-14 19:23:29,538 P19772 INFO Negative sampling done
2021-01-14 19:26:13,951 P19772 INFO --- Start evaluation ---
2021-01-14 19:26:15,312 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 19:26:39,714 P19772 INFO [Metrics] Recall(k=20): 0.068872 - Recall(k=50): 0.130879 - NDCG(k=20): 0.056374 - NDCG(k=50): 0.079286 - HitRate(k=20): 0.410288 - HitRate(k=50): 0.606101
2021-01-14 19:26:39,727 P19772 INFO Save best model: monitor(max): 0.068872
2021-01-14 19:26:39,747 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 19:26:39,801 P19772 INFO Train loss: 0.136058
2021-01-14 19:26:39,801 P19772 INFO ************ Epoch=16 end ************
2021-01-14 19:26:39,802 P19772 INFO Negative sampling num_negs=1000
2021-01-14 19:27:07,143 P19772 INFO Negative sampling done
2021-01-14 19:29:51,279 P19772 INFO --- Start evaluation ---
2021-01-14 19:29:52,695 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 19:30:17,175 P19772 INFO [Metrics] Recall(k=20): 0.069106 - Recall(k=50): 0.131119 - NDCG(k=20): 0.056706 - NDCG(k=50): 0.079644 - HitRate(k=20): 0.410951 - HitRate(k=50): 0.605690
2021-01-14 19:30:17,190 P19772 INFO Save best model: monitor(max): 0.069106
2021-01-14 19:30:17,213 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 19:30:17,276 P19772 INFO Train loss: 0.135735
2021-01-14 19:30:17,276 P19772 INFO ************ Epoch=17 end ************
2021-01-14 19:30:17,277 P19772 INFO Negative sampling num_negs=1000
2021-01-14 19:30:45,208 P19772 INFO Negative sampling done
2021-01-14 19:33:29,620 P19772 INFO --- Start evaluation ---
2021-01-14 19:33:30,984 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 19:33:55,179 P19772 INFO [Metrics] Recall(k=20): 0.069248 - Recall(k=50): 0.131418 - NDCG(k=20): 0.056816 - NDCG(k=50): 0.079793 - HitRate(k=20): 0.411488 - HitRate(k=50): 0.605753
2021-01-14 19:33:55,195 P19772 INFO Save best model: monitor(max): 0.069248
2021-01-14 19:33:55,217 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 19:33:55,279 P19772 INFO Train loss: 0.135349
2021-01-14 19:33:55,279 P19772 INFO ************ Epoch=18 end ************
2021-01-14 19:33:55,280 P19772 INFO Negative sampling num_negs=1000
2021-01-14 19:34:22,870 P19772 INFO Negative sampling done
2021-01-14 19:37:07,955 P19772 INFO --- Start evaluation ---
2021-01-14 19:37:09,383 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 19:37:33,713 P19772 INFO [Metrics] Recall(k=20): 0.069455 - Recall(k=50): 0.131704 - NDCG(k=20): 0.056874 - NDCG(k=50): 0.079903 - HitRate(k=20): 0.412877 - HitRate(k=50): 0.606890
2021-01-14 19:37:33,726 P19772 INFO Save best model: monitor(max): 0.069455
2021-01-14 19:37:33,749 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 19:37:33,811 P19772 INFO Train loss: 0.135132
2021-01-14 19:37:33,811 P19772 INFO ************ Epoch=19 end ************
2021-01-14 19:37:33,811 P19772 INFO Negative sampling num_negs=1000
2021-01-14 19:38:01,220 P19772 INFO Negative sampling done
2021-01-14 19:40:46,029 P19772 INFO --- Start evaluation ---
2021-01-14 19:40:47,496 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 19:41:13,529 P19772 INFO [Metrics] Recall(k=20): 0.069500 - Recall(k=50): 0.131914 - NDCG(k=20): 0.057073 - NDCG(k=50): 0.080184 - HitRate(k=20): 0.413256 - HitRate(k=50): 0.607080
2021-01-14 19:41:13,543 P19772 INFO Save best model: monitor(max): 0.069500
2021-01-14 19:41:13,566 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 19:41:13,628 P19772 INFO Train loss: 0.134839
2021-01-14 19:41:13,628 P19772 INFO ************ Epoch=20 end ************
2021-01-14 19:41:13,629 P19772 INFO Negative sampling num_negs=1000
2021-01-14 19:41:41,085 P19772 INFO Negative sampling done
2021-01-14 19:44:26,624 P19772 INFO --- Start evaluation ---
2021-01-14 19:44:28,040 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 19:44:52,249 P19772 INFO [Metrics] Recall(k=20): 0.069334 - Recall(k=50): 0.132311 - NDCG(k=20): 0.057031 - NDCG(k=50): 0.080352 - HitRate(k=20): 0.412972 - HitRate(k=50): 0.609006
2021-01-14 19:44:52,261 P19772 INFO Monitor(max) STOP: 0.069334 !
2021-01-14 19:44:52,262 P19772 INFO Reduce learning rate on plateau: 0.000001
2021-01-14 19:44:52,262 P19772 INFO Load best model:  SimpleX_yelp18_x0_034_297a4b82.model
2021-01-14 19:44:52,276 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 19:44:52,333 P19772 INFO Train loss: 0.134603
2021-01-14 19:44:52,333 P19772 INFO ************ Epoch=21 end ************
2021-01-14 19:44:52,334 P19772 INFO Negative sampling num_negs=1000
2021-01-14 19:45:19,644 P19772 INFO Negative sampling done
2021-01-14 19:48:07,346 P19772 INFO --- Start evaluation ---
2021-01-14 19:48:08,729 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 19:48:32,967 P19772 INFO [Metrics] Recall(k=20): 0.069459 - Recall(k=50): 0.131891 - NDCG(k=20): 0.057065 - NDCG(k=50): 0.080184 - HitRate(k=20): 0.413035 - HitRate(k=50): 0.607427
2021-01-14 19:48:32,979 P19772 INFO Monitor(max) STOP: 0.069459 !
2021-01-14 19:48:32,979 P19772 INFO Reduce learning rate on plateau: 0.000001
2021-01-14 19:48:32,979 P19772 INFO Load best model:  SimpleX_yelp18_x0_034_297a4b82.model
2021-01-14 19:48:32,993 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 19:48:33,055 P19772 INFO Train loss: 0.133417
2021-01-14 19:48:33,055 P19772 INFO ************ Epoch=22 end ************
2021-01-14 19:48:33,056 P19772 INFO Negative sampling num_negs=1000
2021-01-14 19:49:00,495 P19772 INFO Negative sampling done
2021-01-14 19:51:50,313 P19772 INFO --- Start evaluation ---
2021-01-14 19:51:51,699 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 19:52:15,596 P19772 INFO [Metrics] Recall(k=20): 0.069526 - Recall(k=50): 0.131962 - NDCG(k=20): 0.057051 - NDCG(k=50): 0.080176 - HitRate(k=20): 0.413446 - HitRate(k=50): 0.607774
2021-01-14 19:52:15,609 P19772 INFO Save best model: monitor(max): 0.069526
2021-01-14 19:52:15,630 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 19:52:15,689 P19772 INFO Train loss: 0.133377
2021-01-14 19:52:15,689 P19772 INFO ************ Epoch=23 end ************
2021-01-14 19:52:15,690 P19772 INFO Negative sampling num_negs=1000
2021-01-14 19:52:43,155 P19772 INFO Negative sampling done
2021-01-14 19:55:18,079 P19772 INFO --- Start evaluation ---
2021-01-14 19:55:19,475 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 19:55:43,697 P19772 INFO [Metrics] Recall(k=20): 0.069556 - Recall(k=50): 0.132039 - NDCG(k=20): 0.057089 - NDCG(k=50): 0.080236 - HitRate(k=20): 0.413540 - HitRate(k=50): 0.608216
2021-01-14 19:55:43,711 P19772 INFO Save best model: monitor(max): 0.069556
2021-01-14 19:55:43,734 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 19:55:43,784 P19772 INFO Train loss: 0.133268
2021-01-14 19:55:43,785 P19772 INFO ************ Epoch=24 end ************
2021-01-14 19:55:43,785 P19772 INFO Negative sampling num_negs=1000
2021-01-14 19:56:11,147 P19772 INFO Negative sampling done
2021-01-14 19:58:52,930 P19772 INFO --- Start evaluation ---
2021-01-14 19:58:54,311 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 19:59:18,242 P19772 INFO [Metrics] Recall(k=20): 0.069535 - Recall(k=50): 0.132026 - NDCG(k=20): 0.057134 - NDCG(k=50): 0.080291 - HitRate(k=20): 0.413477 - HitRate(k=50): 0.608469
2021-01-14 19:59:18,255 P19772 INFO Monitor(max) STOP: 0.069535 !
2021-01-14 19:59:18,255 P19772 INFO Reduce learning rate on plateau: 0.000001
2021-01-14 19:59:18,255 P19772 INFO Load best model:  SimpleX_yelp18_x0_034_297a4b82.model
2021-01-14 19:59:18,269 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 19:59:18,330 P19772 INFO Train loss: 0.133231
2021-01-14 19:59:18,330 P19772 INFO ************ Epoch=25 end ************
2021-01-14 19:59:18,330 P19772 INFO Negative sampling num_negs=1000
2021-01-14 19:59:45,694 P19772 INFO Negative sampling done
2021-01-14 20:02:28,080 P19772 INFO --- Start evaluation ---
2021-01-14 20:02:29,751 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 20:02:53,979 P19772 INFO [Metrics] Recall(k=20): 0.069540 - Recall(k=50): 0.132047 - NDCG(k=20): 0.057110 - NDCG(k=50): 0.080271 - HitRate(k=20): 0.413383 - HitRate(k=50): 0.608311
2021-01-14 20:02:53,991 P19772 INFO Monitor(max) STOP: 0.069540 !
2021-01-14 20:02:53,991 P19772 INFO Reduce learning rate on plateau: 0.000001
2021-01-14 20:02:53,991 P19772 INFO Load best model:  SimpleX_yelp18_x0_034_297a4b82.model
2021-01-14 20:02:54,001 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 20:02:54,059 P19772 INFO Train loss: 0.133233
2021-01-14 20:02:54,060 P19772 INFO ************ Epoch=26 end ************
2021-01-14 20:02:54,060 P19772 INFO Negative sampling num_negs=1000
2021-01-14 20:03:21,352 P19772 INFO Negative sampling done
2021-01-14 20:06:03,217 P19772 INFO --- Start evaluation ---
2021-01-14 20:06:04,588 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 20:06:28,630 P19772 INFO [Metrics] Recall(k=20): 0.069648 - Recall(k=50): 0.132086 - NDCG(k=20): 0.057164 - NDCG(k=50): 0.080289 - HitRate(k=20): 0.414014 - HitRate(k=50): 0.608406
2021-01-14 20:06:28,644 P19772 INFO Save best model: monitor(max): 0.069648
2021-01-14 20:06:28,666 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 20:06:28,723 P19772 INFO Train loss: 0.133235
2021-01-14 20:06:28,723 P19772 INFO ************ Epoch=27 end ************
2021-01-14 20:06:28,723 P19772 INFO Negative sampling num_negs=1000
2021-01-14 20:06:56,333 P19772 INFO Negative sampling done
2021-01-14 20:09:39,060 P19772 INFO --- Start evaluation ---
2021-01-14 20:09:40,422 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 20:10:05,419 P19772 INFO [Metrics] Recall(k=20): 0.069747 - Recall(k=50): 0.132082 - NDCG(k=20): 0.057201 - NDCG(k=50): 0.080283 - HitRate(k=20): 0.414204 - HitRate(k=50): 0.608469
2021-01-14 20:10:05,432 P19772 INFO Save best model: monitor(max): 0.069747
2021-01-14 20:10:05,455 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 20:10:05,516 P19772 INFO Train loss: 0.133126
2021-01-14 20:10:05,516 P19772 INFO ************ Epoch=28 end ************
2021-01-14 20:10:05,516 P19772 INFO Negative sampling num_negs=1000
2021-01-14 20:10:34,028 P19772 INFO Negative sampling done
2021-01-14 20:13:15,685 P19772 INFO --- Start evaluation ---
2021-01-14 20:13:17,108 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 20:13:41,191 P19772 INFO [Metrics] Recall(k=20): 0.069754 - Recall(k=50): 0.132108 - NDCG(k=20): 0.057231 - NDCG(k=50): 0.080325 - HitRate(k=20): 0.414204 - HitRate(k=50): 0.608595
2021-01-14 20:13:41,203 P19772 INFO Save best model: monitor(max): 0.069754
2021-01-14 20:13:41,225 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 20:13:41,279 P19772 INFO Train loss: 0.133185
2021-01-14 20:13:41,279 P19772 INFO ************ Epoch=29 end ************
2021-01-14 20:13:41,280 P19772 INFO Negative sampling num_negs=1000
2021-01-14 20:14:08,632 P19772 INFO Negative sampling done
2021-01-14 20:16:50,654 P19772 INFO --- Start evaluation ---
2021-01-14 20:16:52,052 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 20:17:16,241 P19772 INFO [Metrics] Recall(k=20): 0.069817 - Recall(k=50): 0.132109 - NDCG(k=20): 0.057271 - NDCG(k=50): 0.080339 - HitRate(k=20): 0.414330 - HitRate(k=50): 0.608816
2021-01-14 20:17:16,252 P19772 INFO Save best model: monitor(max): 0.069817
2021-01-14 20:17:16,274 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 20:17:16,334 P19772 INFO Train loss: 0.133081
2021-01-14 20:17:16,334 P19772 INFO ************ Epoch=30 end ************
2021-01-14 20:17:16,335 P19772 INFO Negative sampling num_negs=1000
2021-01-14 20:17:43,907 P19772 INFO Negative sampling done
2021-01-14 20:20:02,839 P19772 INFO --- Start evaluation ---
2021-01-14 20:20:04,010 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 20:20:28,155 P19772 INFO [Metrics] Recall(k=20): 0.069787 - Recall(k=50): 0.132200 - NDCG(k=20): 0.057280 - NDCG(k=50): 0.080394 - HitRate(k=20): 0.414519 - HitRate(k=50): 0.608532
2021-01-14 20:20:28,167 P19772 INFO Monitor(max) STOP: 0.069787 !
2021-01-14 20:20:28,168 P19772 INFO Reduce learning rate on plateau: 0.000001
2021-01-14 20:20:28,168 P19772 INFO Load best model:  SimpleX_yelp18_x0_034_297a4b82.model
2021-01-14 20:20:28,181 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 20:20:28,240 P19772 INFO Train loss: 0.133116
2021-01-14 20:20:28,240 P19772 INFO ************ Epoch=31 end ************
2021-01-14 20:20:28,241 P19772 INFO Negative sampling num_negs=1000
2021-01-14 20:20:55,501 P19772 INFO Negative sampling done
2021-01-14 20:23:11,724 P19772 INFO --- Start evaluation ---
2021-01-14 20:23:12,903 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 20:23:36,655 P19772 INFO [Metrics] Recall(k=20): 0.069759 - Recall(k=50): 0.132227 - NDCG(k=20): 0.057264 - NDCG(k=50): 0.080391 - HitRate(k=20): 0.414267 - HitRate(k=50): 0.608880
2021-01-14 20:23:36,668 P19772 INFO Monitor(max) STOP: 0.069759 !
2021-01-14 20:23:36,668 P19772 INFO Reduce learning rate on plateau: 0.000001
2021-01-14 20:23:36,669 P19772 INFO Load best model:  SimpleX_yelp18_x0_034_297a4b82.model
2021-01-14 20:23:36,681 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 20:23:36,742 P19772 INFO Train loss: 0.133028
2021-01-14 20:23:36,742 P19772 INFO ************ Epoch=32 end ************
2021-01-14 20:23:36,743 P19772 INFO Negative sampling num_negs=1000
2021-01-14 20:24:04,152 P19772 INFO Negative sampling done
2021-01-14 20:26:20,265 P19772 INFO --- Start evaluation ---
2021-01-14 20:26:21,441 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 20:26:45,512 P19772 INFO [Metrics] Recall(k=20): 0.069826 - Recall(k=50): 0.132150 - NDCG(k=20): 0.057297 - NDCG(k=50): 0.080382 - HitRate(k=20): 0.414140 - HitRate(k=50): 0.608501
2021-01-14 20:26:45,525 P19772 INFO Save best model: monitor(max): 0.069826
2021-01-14 20:26:45,547 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 20:26:45,607 P19772 INFO Train loss: 0.133019
2021-01-14 20:26:45,607 P19772 INFO ************ Epoch=33 end ************
2021-01-14 20:26:45,608 P19772 INFO Negative sampling num_negs=1000
2021-01-14 20:27:12,917 P19772 INFO Negative sampling done
2021-01-14 20:29:29,096 P19772 INFO --- Start evaluation ---
2021-01-14 20:29:30,267 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 20:29:53,975 P19772 INFO [Metrics] Recall(k=20): 0.069887 - Recall(k=50): 0.132212 - NDCG(k=20): 0.057318 - NDCG(k=50): 0.080388 - HitRate(k=20): 0.414204 - HitRate(k=50): 0.608406
2021-01-14 20:29:53,987 P19772 INFO Save best model: monitor(max): 0.069887
2021-01-14 20:29:54,009 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 20:29:54,068 P19772 INFO Train loss: 0.132962
2021-01-14 20:29:54,068 P19772 INFO ************ Epoch=34 end ************
2021-01-14 20:29:54,069 P19772 INFO Negative sampling num_negs=1000
2021-01-14 20:30:21,403 P19772 INFO Negative sampling done
2021-01-14 20:32:37,868 P19772 INFO --- Start evaluation ---
2021-01-14 20:32:39,038 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 20:33:02,833 P19772 INFO [Metrics] Recall(k=20): 0.069876 - Recall(k=50): 0.132189 - NDCG(k=20): 0.057339 - NDCG(k=50): 0.080414 - HitRate(k=20): 0.414425 - HitRate(k=50): 0.608216
2021-01-14 20:33:02,847 P19772 INFO Monitor(max) STOP: 0.069876 !
2021-01-14 20:33:02,847 P19772 INFO Reduce learning rate on plateau: 0.000001
2021-01-14 20:33:02,847 P19772 INFO Load best model:  SimpleX_yelp18_x0_034_297a4b82.model
2021-01-14 20:33:02,860 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 20:33:02,920 P19772 INFO Train loss: 0.132971
2021-01-14 20:33:02,920 P19772 INFO ************ Epoch=35 end ************
2021-01-14 20:33:02,921 P19772 INFO Negative sampling num_negs=1000
2021-01-14 20:33:30,181 P19772 INFO Negative sampling done
2021-01-14 20:35:46,722 P19772 INFO --- Start evaluation ---
2021-01-14 20:35:47,896 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 20:36:11,891 P19772 INFO [Metrics] Recall(k=20): 0.069864 - Recall(k=50): 0.132236 - NDCG(k=20): 0.057293 - NDCG(k=50): 0.080384 - HitRate(k=20): 0.414235 - HitRate(k=50): 0.608438
2021-01-14 20:36:11,903 P19772 INFO Monitor(max) STOP: 0.069864 !
2021-01-14 20:36:11,903 P19772 INFO Reduce learning rate on plateau: 0.000001
2021-01-14 20:36:11,903 P19772 INFO Load best model:  SimpleX_yelp18_x0_034_297a4b82.model
2021-01-14 20:36:11,914 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 20:36:11,978 P19772 INFO Train loss: 0.132968
2021-01-14 20:36:11,978 P19772 INFO ************ Epoch=36 end ************
2021-01-14 20:36:11,979 P19772 INFO Negative sampling num_negs=1000
2021-01-14 20:36:39,206 P19772 INFO Negative sampling done
2021-01-14 20:38:56,299 P19772 INFO --- Start evaluation ---
2021-01-14 20:38:57,482 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 20:39:21,284 P19772 INFO [Metrics] Recall(k=20): 0.069895 - Recall(k=50): 0.132188 - NDCG(k=20): 0.057333 - NDCG(k=50): 0.080408 - HitRate(k=20): 0.414362 - HitRate(k=50): 0.608406
2021-01-14 20:39:21,296 P19772 INFO Save best model: monitor(max): 0.069895
2021-01-14 20:39:21,319 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 20:39:21,378 P19772 INFO Train loss: 0.132960
2021-01-14 20:39:21,378 P19772 INFO ************ Epoch=37 end ************
2021-01-14 20:39:21,378 P19772 INFO Negative sampling num_negs=1000
2021-01-14 20:39:48,724 P19772 INFO Negative sampling done
2021-01-14 20:42:05,398 P19772 INFO --- Start evaluation ---
2021-01-14 20:42:06,531 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 20:42:30,432 P19772 INFO [Metrics] Recall(k=20): 0.069871 - Recall(k=50): 0.132213 - NDCG(k=20): 0.057365 - NDCG(k=50): 0.080457 - HitRate(k=20): 0.414677 - HitRate(k=50): 0.608659
2021-01-14 20:42:30,445 P19772 INFO Monitor(max) STOP: 0.069871 !
2021-01-14 20:42:30,445 P19772 INFO Reduce learning rate on plateau: 0.000001
2021-01-14 20:42:30,445 P19772 INFO Load best model:  SimpleX_yelp18_x0_034_297a4b82.model
2021-01-14 20:42:30,459 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 20:42:30,519 P19772 INFO Train loss: 0.132959
2021-01-14 20:42:30,520 P19772 INFO ************ Epoch=38 end ************
2021-01-14 20:42:30,520 P19772 INFO Negative sampling num_negs=1000
2021-01-14 20:42:57,791 P19772 INFO Negative sampling done
2021-01-14 20:45:19,496 P19772 INFO --- Start evaluation ---
2021-01-14 20:45:20,701 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 20:45:44,666 P19772 INFO [Metrics] Recall(k=20): 0.069948 - Recall(k=50): 0.132154 - NDCG(k=20): 0.057386 - NDCG(k=50): 0.080411 - HitRate(k=20): 0.414646 - HitRate(k=50): 0.608311
2021-01-14 20:45:44,678 P19772 INFO Save best model: monitor(max): 0.069948
2021-01-14 20:45:44,699 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 20:45:44,760 P19772 INFO Train loss: 0.132949
2021-01-14 20:45:44,760 P19772 INFO ************ Epoch=39 end ************
2021-01-14 20:45:44,760 P19772 INFO Negative sampling num_negs=1000
2021-01-14 20:46:12,025 P19772 INFO Negative sampling done
2021-01-14 20:48:33,966 P19772 INFO --- Start evaluation ---
2021-01-14 20:48:35,157 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 20:48:58,670 P19772 INFO [Metrics] Recall(k=20): 0.070040 - Recall(k=50): 0.132201 - NDCG(k=20): 0.057476 - NDCG(k=50): 0.080477 - HitRate(k=20): 0.414930 - HitRate(k=50): 0.608438
2021-01-14 20:48:58,683 P19772 INFO Save best model: monitor(max): 0.070040
2021-01-14 20:48:58,707 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 20:48:58,771 P19772 INFO Train loss: 0.132895
2021-01-14 20:48:58,772 P19772 INFO ************ Epoch=40 end ************
2021-01-14 20:48:58,772 P19772 INFO Negative sampling num_negs=1000
2021-01-14 20:49:26,225 P19772 INFO Negative sampling done
2021-01-14 20:51:42,272 P19772 INFO --- Start evaluation ---
2021-01-14 20:51:43,456 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 20:52:07,654 P19772 INFO [Metrics] Recall(k=20): 0.070111 - Recall(k=50): 0.132237 - NDCG(k=20): 0.057514 - NDCG(k=50): 0.080518 - HitRate(k=20): 0.415151 - HitRate(k=50): 0.608753
2021-01-14 20:52:07,667 P19772 INFO Save best model: monitor(max): 0.070111
2021-01-14 20:52:07,691 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 20:52:07,751 P19772 INFO Train loss: 0.132868
2021-01-14 20:52:07,752 P19772 INFO ************ Epoch=41 end ************
2021-01-14 20:52:07,752 P19772 INFO Negative sampling num_negs=1000
2021-01-14 20:52:35,037 P19772 INFO Negative sampling done
2021-01-14 20:54:51,572 P19772 INFO --- Start evaluation ---
2021-01-14 20:54:52,741 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 20:55:17,044 P19772 INFO [Metrics] Recall(k=20): 0.070002 - Recall(k=50): 0.132244 - NDCG(k=20): 0.057477 - NDCG(k=50): 0.080529 - HitRate(k=20): 0.414835 - HitRate(k=50): 0.608564
2021-01-14 20:55:17,057 P19772 INFO Monitor(max) STOP: 0.070002 !
2021-01-14 20:55:17,057 P19772 INFO Reduce learning rate on plateau: 0.000001
2021-01-14 20:55:17,057 P19772 INFO Load best model:  SimpleX_yelp18_x0_034_297a4b82.model
2021-01-14 20:55:17,069 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 20:55:17,132 P19772 INFO Train loss: 0.132845
2021-01-14 20:55:17,132 P19772 INFO ************ Epoch=42 end ************
2021-01-14 20:55:17,133 P19772 INFO Negative sampling num_negs=1000
2021-01-14 20:55:44,384 P19772 INFO Negative sampling done
2021-01-14 20:58:01,567 P19772 INFO --- Start evaluation ---
2021-01-14 20:58:02,716 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 20:58:26,522 P19772 INFO [Metrics] Recall(k=20): 0.070027 - Recall(k=50): 0.132278 - NDCG(k=20): 0.057438 - NDCG(k=50): 0.080492 - HitRate(k=20): 0.415056 - HitRate(k=50): 0.608848
2021-01-14 20:58:26,535 P19772 INFO Monitor(max) STOP: 0.070027 !
2021-01-14 20:58:26,535 P19772 INFO Reduce learning rate on plateau: 0.000001
2021-01-14 20:58:26,535 P19772 INFO Load best model:  SimpleX_yelp18_x0_034_297a4b82.model
2021-01-14 20:58:26,548 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 20:58:26,611 P19772 INFO Train loss: 0.132837
2021-01-14 20:58:26,611 P19772 INFO ************ Epoch=43 end ************
2021-01-14 20:58:26,612 P19772 INFO Negative sampling num_negs=1000
2021-01-14 20:58:54,359 P19772 INFO Negative sampling done
2021-01-14 21:01:10,620 P19772 INFO --- Start evaluation ---
2021-01-14 21:01:11,787 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 21:01:36,113 P19772 INFO [Metrics] Recall(k=20): 0.070043 - Recall(k=50): 0.132299 - NDCG(k=20): 0.057496 - NDCG(k=50): 0.080531 - HitRate(k=20): 0.415119 - HitRate(k=50): 0.608659
2021-01-14 21:01:36,127 P19772 INFO Monitor(max) STOP: 0.070043 !
2021-01-14 21:01:36,127 P19772 INFO Reduce learning rate on plateau: 0.000001
2021-01-14 21:01:36,127 P19772 INFO Load best model:  SimpleX_yelp18_x0_034_297a4b82.model
2021-01-14 21:01:36,139 P19772 INFO Early stopping at epoch=44
2021-01-14 21:01:36,139 P19772 INFO --- 2417/2417 batches finished ---
2021-01-14 21:01:36,196 P19772 INFO Train loss: 0.132810
2021-01-14 21:01:36,196 P19772 INFO Training finished.
2021-01-14 21:01:36,196 P19772 INFO Load best model:  SimpleX_yelp18_x0_034_297a4b82.model
2021-01-14 21:01:36,215 P19772 INFO ****** Train/validation evaluation ******
2021-01-14 21:01:36,215 P19772 INFO --- Start evaluation ---
2021-01-14 21:01:37,360 P19772 INFO Evaluating metrics for 31668 users...
2021-01-14 21:02:01,306 P19772 INFO [Metrics] Recall(k=20): 0.070111 - Recall(k=50): 0.132237 - NDCG(k=20): 0.057514 - NDCG(k=50): 0.080518 - HitRate(k=20): 0.415151 - HitRate(k=50): 0.608753

```