## MF-BPR_yelp18_x0 

A notebook to benchmark MF-BPR on yelp18 dataset.

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
2021-01-02 20:33:35,654 P40223 INFO [Metrics] Recall(k=20): 0.057632 - Recall(k=50): 0.112270 - NDCG(k=20): 0.046756 - NDCG(k=50): 0.067085 - HitRate(k=20): 0.362448 - HitRate(k=50): 0.557692
```


### Logs
```
2021-01-02 19:54:29,477 P40223 INFO Set up feature encoder...
2021-01-02 19:54:29,478 P40223 INFO Load feature_map from json: ../data/Yelp18/yelp18_x0_0f43e4ba/feature_map.json
2021-01-02 19:54:34,258 P40223 INFO Total number of parameters: 4461952.
2021-01-02 19:54:34,259 P40223 INFO Loading data...
2021-01-02 19:54:34,278 P40223 INFO Loading data from h5: ../data/Yelp18/yelp18_x0_0f43e4ba/train.h5
2021-01-02 19:54:34,328 P40223 INFO Loading data from h5: ../data/Yelp18/yelp18_x0_0f43e4ba/item_corpus.h5
2021-01-02 19:54:35,417 P40223 INFO Loading data from h5: ../data/Yelp18/yelp18_x0_0f43e4ba/valid.h5
2021-01-02 19:54:35,752 P40223 INFO Loading data from h5: ../data/Yelp18/yelp18_x0_0f43e4ba/item_corpus.h5
2021-01-02 19:54:35,754 P40223 INFO Train samples: total/1237259, blocks/1
2021-01-02 19:54:35,754 P40223 INFO Validation samples: total/31668, blocks/1
2021-01-02 19:54:35,754 P40223 INFO Loading train data done.
2021-01-02 19:54:35,754 P40223 INFO **** Start training: 1209 batches/epoch ****
2021-01-02 19:54:35,764 P40223 INFO Negative sampling num_negs=50
2021-01-02 19:54:37,640 P40223 INFO Negative sampling done
2021-01-02 19:54:54,598 P40223 INFO --- Start evaluation ---
2021-01-02 19:54:56,061 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 19:55:31,814 P40223 INFO [Metrics] Recall(k=20): 0.033210 - Recall(k=50): 0.063688 - NDCG(k=20): 0.027195 - NDCG(k=50): 0.038495 - HitRate(k=20): 0.231211 - HitRate(k=50): 0.371637
2021-01-02 19:55:31,827 P40223 INFO Save best model: monitor(max): 0.033210
2021-01-02 19:55:32,050 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 19:55:32,126 P40223 INFO Train loss: 0.473593
2021-01-02 19:55:32,126 P40223 INFO ************ Epoch=1 end ************
2021-01-02 19:55:32,127 P40223 INFO Negative sampling num_negs=50
2021-01-02 19:55:34,486 P40223 INFO Negative sampling done
2021-01-02 19:55:57,295 P40223 INFO --- Start evaluation ---
2021-01-02 19:55:58,214 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 19:56:34,318 P40223 INFO [Metrics] Recall(k=20): 0.035294 - Recall(k=50): 0.070725 - NDCG(k=20): 0.029400 - NDCG(k=50): 0.042499 - HitRate(k=20): 0.245042 - HitRate(k=50): 0.397562
2021-01-02 19:56:34,331 P40223 INFO Save best model: monitor(max): 0.035294
2021-01-02 19:56:34,357 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 19:56:34,437 P40223 INFO Train loss: 0.187060
2021-01-02 19:56:34,437 P40223 INFO ************ Epoch=2 end ************
2021-01-02 19:56:34,442 P40223 INFO Negative sampling num_negs=50
2021-01-02 19:56:36,618 P40223 INFO Negative sampling done
2021-01-02 19:56:54,772 P40223 INFO --- Start evaluation ---
2021-01-02 19:56:55,563 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 19:57:31,504 P40223 INFO [Metrics] Recall(k=20): 0.038170 - Recall(k=50): 0.075349 - NDCG(k=20): 0.031560 - NDCG(k=50): 0.045279 - HitRate(k=20): 0.262694 - HitRate(k=50): 0.422319
2021-01-02 19:57:31,526 P40223 INFO Save best model: monitor(max): 0.038170
2021-01-02 19:57:31,572 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 19:57:31,658 P40223 INFO Train loss: 0.142809
2021-01-02 19:57:31,658 P40223 INFO ************ Epoch=3 end ************
2021-01-02 19:57:31,663 P40223 INFO Negative sampling num_negs=50
2021-01-02 19:57:33,602 P40223 INFO Negative sampling done
2021-01-02 19:57:57,170 P40223 INFO --- Start evaluation ---
2021-01-02 19:57:58,151 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 19:58:34,372 P40223 INFO [Metrics] Recall(k=20): 0.040359 - Recall(k=50): 0.080547 - NDCG(k=20): 0.033248 - NDCG(k=50): 0.048179 - HitRate(k=20): 0.277567 - HitRate(k=50): 0.445781
2021-01-02 19:58:34,387 P40223 INFO Save best model: monitor(max): 0.040359
2021-01-02 19:58:34,414 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 19:58:34,483 P40223 INFO Train loss: 0.124676
2021-01-02 19:58:34,483 P40223 INFO ************ Epoch=4 end ************
2021-01-02 19:58:34,484 P40223 INFO Negative sampling num_negs=50
2021-01-02 19:58:36,364 P40223 INFO Negative sampling done
2021-01-02 19:58:56,700 P40223 INFO --- Start evaluation ---
2021-01-02 19:58:57,653 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 19:59:34,153 P40223 INFO [Metrics] Recall(k=20): 0.042613 - Recall(k=50): 0.083720 - NDCG(k=20): 0.034928 - NDCG(k=50): 0.050164 - HitRate(k=20): 0.289125 - HitRate(k=50): 0.459770
2021-01-02 19:59:34,168 P40223 INFO Save best model: monitor(max): 0.042613
2021-01-02 19:59:34,196 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 19:59:34,274 P40223 INFO Train loss: 0.112880
2021-01-02 19:59:34,274 P40223 INFO ************ Epoch=5 end ************
2021-01-02 19:59:34,279 P40223 INFO Negative sampling num_negs=50
2021-01-02 19:59:36,126 P40223 INFO Negative sampling done
2021-01-02 19:59:56,595 P40223 INFO --- Start evaluation ---
2021-01-02 19:59:57,623 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:00:35,116 P40223 INFO [Metrics] Recall(k=20): 0.044184 - Recall(k=50): 0.087158 - NDCG(k=20): 0.036119 - NDCG(k=50): 0.052078 - HitRate(k=20): 0.298377 - HitRate(k=50): 0.476222
2021-01-02 20:00:35,131 P40223 INFO Save best model: monitor(max): 0.044184
2021-01-02 20:00:35,159 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:00:35,248 P40223 INFO Train loss: 0.103614
2021-01-02 20:00:35,248 P40223 INFO ************ Epoch=6 end ************
2021-01-02 20:00:35,253 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:00:37,712 P40223 INFO Negative sampling done
2021-01-02 20:00:57,886 P40223 INFO --- Start evaluation ---
2021-01-02 20:00:58,774 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:01:34,496 P40223 INFO [Metrics] Recall(k=20): 0.045703 - Recall(k=50): 0.090455 - NDCG(k=20): 0.037273 - NDCG(k=50): 0.053879 - HitRate(k=20): 0.305608 - HitRate(k=50): 0.486643
2021-01-02 20:01:34,526 P40223 INFO Save best model: monitor(max): 0.045703
2021-01-02 20:01:34,591 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:01:34,666 P40223 INFO Train loss: 0.096025
2021-01-02 20:01:34,666 P40223 INFO ************ Epoch=7 end ************
2021-01-02 20:01:34,671 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:01:36,983 P40223 INFO Negative sampling done
2021-01-02 20:01:54,801 P40223 INFO --- Start evaluation ---
2021-01-02 20:01:55,581 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:02:30,328 P40223 INFO [Metrics] Recall(k=20): 0.046745 - Recall(k=50): 0.093699 - NDCG(k=20): 0.038285 - NDCG(k=50): 0.055727 - HitRate(k=20): 0.311924 - HitRate(k=50): 0.497569
2021-01-02 20:02:30,346 P40223 INFO Save best model: monitor(max): 0.046745
2021-01-02 20:02:30,399 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:02:30,477 P40223 INFO Train loss: 0.089706
2021-01-02 20:02:30,477 P40223 INFO ************ Epoch=8 end ************
2021-01-02 20:02:30,481 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:02:32,416 P40223 INFO Negative sampling done
2021-01-02 20:02:52,744 P40223 INFO --- Start evaluation ---
2021-01-02 20:02:53,548 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:03:28,847 P40223 INFO [Metrics] Recall(k=20): 0.048452 - Recall(k=50): 0.096373 - NDCG(k=20): 0.039636 - NDCG(k=50): 0.057446 - HitRate(k=20): 0.320008 - HitRate(k=50): 0.506789
2021-01-02 20:03:28,858 P40223 INFO Save best model: monitor(max): 0.048452
2021-01-02 20:03:28,885 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:03:28,960 P40223 INFO Train loss: 0.084487
2021-01-02 20:03:28,960 P40223 INFO ************ Epoch=9 end ************
2021-01-02 20:03:28,965 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:03:30,790 P40223 INFO Negative sampling done
2021-01-02 20:03:52,966 P40223 INFO --- Start evaluation ---
2021-01-02 20:03:53,748 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:04:30,765 P40223 INFO [Metrics] Recall(k=20): 0.050080 - Recall(k=50): 0.099074 - NDCG(k=20): 0.040874 - NDCG(k=50): 0.059029 - HitRate(k=20): 0.327839 - HitRate(k=50): 0.517210
2021-01-02 20:04:30,778 P40223 INFO Save best model: monitor(max): 0.050080
2021-01-02 20:04:30,805 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:04:30,869 P40223 INFO Train loss: 0.080039
2021-01-02 20:04:30,869 P40223 INFO ************ Epoch=10 end ************
2021-01-02 20:04:30,874 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:04:32,934 P40223 INFO Negative sampling done
2021-01-02 20:04:52,904 P40223 INFO --- Start evaluation ---
2021-01-02 20:04:53,748 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:05:29,325 P40223 INFO [Metrics] Recall(k=20): 0.050858 - Recall(k=50): 0.101220 - NDCG(k=20): 0.041599 - NDCG(k=50): 0.060307 - HitRate(k=20): 0.331060 - HitRate(k=50): 0.524094
2021-01-02 20:05:29,347 P40223 INFO Save best model: monitor(max): 0.050858
2021-01-02 20:05:29,392 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:05:29,467 P40223 INFO Train loss: 0.076242
2021-01-02 20:05:29,467 P40223 INFO ************ Epoch=11 end ************
2021-01-02 20:05:29,472 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:05:31,653 P40223 INFO Negative sampling done
2021-01-02 20:05:49,226 P40223 INFO --- Start evaluation ---
2021-01-02 20:05:50,071 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:06:29,097 P40223 INFO [Metrics] Recall(k=20): 0.051583 - Recall(k=50): 0.103179 - NDCG(k=20): 0.041939 - NDCG(k=50): 0.061138 - HitRate(k=20): 0.336775 - HitRate(k=50): 0.529715
2021-01-02 20:06:29,110 P40223 INFO Save best model: monitor(max): 0.051583
2021-01-02 20:06:29,139 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:06:29,223 P40223 INFO Train loss: 0.072877
2021-01-02 20:06:29,223 P40223 INFO ************ Epoch=12 end ************
2021-01-02 20:06:29,228 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:06:31,795 P40223 INFO Negative sampling done
2021-01-02 20:06:51,317 P40223 INFO --- Start evaluation ---
2021-01-02 20:06:52,227 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:07:30,694 P40223 INFO [Metrics] Recall(k=20): 0.052692 - Recall(k=50): 0.104790 - NDCG(k=20): 0.042865 - NDCG(k=50): 0.062281 - HitRate(k=20): 0.340944 - HitRate(k=50): 0.535967
2021-01-02 20:07:30,716 P40223 INFO Save best model: monitor(max): 0.052692
2021-01-02 20:07:30,781 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:07:30,855 P40223 INFO Train loss: 0.070040
2021-01-02 20:07:30,855 P40223 INFO ************ Epoch=13 end ************
2021-01-02 20:07:30,860 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:07:33,113 P40223 INFO Negative sampling done
2021-01-02 20:07:54,893 P40223 INFO --- Start evaluation ---
2021-01-02 20:07:55,784 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:08:30,986 P40223 INFO [Metrics] Recall(k=20): 0.053285 - Recall(k=50): 0.105982 - NDCG(k=20): 0.043240 - NDCG(k=50): 0.062832 - HitRate(k=20): 0.346059 - HitRate(k=50): 0.540040
2021-01-02 20:08:31,007 P40223 INFO Save best model: monitor(max): 0.053285
2021-01-02 20:08:31,056 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:08:31,147 P40223 INFO Train loss: 0.067529
2021-01-02 20:08:31,147 P40223 INFO ************ Epoch=14 end ************
2021-01-02 20:08:31,152 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:08:33,543 P40223 INFO Negative sampling done
2021-01-02 20:08:53,263 P40223 INFO --- Start evaluation ---
2021-01-02 20:08:53,997 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:09:30,977 P40223 INFO [Metrics] Recall(k=20): 0.054122 - Recall(k=50): 0.107413 - NDCG(k=20): 0.044296 - NDCG(k=50): 0.064100 - HitRate(k=20): 0.349154 - HitRate(k=50): 0.544556
2021-01-02 20:09:30,990 P40223 INFO Save best model: monitor(max): 0.054122
2021-01-02 20:09:31,016 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:09:31,099 P40223 INFO Train loss: 0.065322
2021-01-02 20:09:31,099 P40223 INFO ************ Epoch=15 end ************
2021-01-02 20:09:31,104 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:09:33,110 P40223 INFO Negative sampling done
2021-01-02 20:09:53,009 P40223 INFO --- Start evaluation ---
2021-01-02 20:09:53,929 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:10:32,315 P40223 INFO [Metrics] Recall(k=20): 0.054640 - Recall(k=50): 0.108051 - NDCG(k=20): 0.044408 - NDCG(k=50): 0.064218 - HitRate(k=20): 0.351838 - HitRate(k=50): 0.547366
2021-01-02 20:10:32,336 P40223 INFO Save best model: monitor(max): 0.054640
2021-01-02 20:10:32,386 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:10:32,491 P40223 INFO Train loss: 0.063420
2021-01-02 20:10:32,491 P40223 INFO ************ Epoch=16 end ************
2021-01-02 20:10:32,504 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:10:34,946 P40223 INFO Negative sampling done
2021-01-02 20:10:54,584 P40223 INFO --- Start evaluation ---
2021-01-02 20:10:55,451 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:11:31,750 P40223 INFO [Metrics] Recall(k=20): 0.055028 - Recall(k=50): 0.108838 - NDCG(k=20): 0.044861 - NDCG(k=50): 0.064912 - HitRate(k=20): 0.352880 - HitRate(k=50): 0.549703
2021-01-02 20:11:31,763 P40223 INFO Save best model: monitor(max): 0.055028
2021-01-02 20:11:31,789 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:11:31,861 P40223 INFO Train loss: 0.061839
2021-01-02 20:11:31,861 P40223 INFO ************ Epoch=17 end ************
2021-01-02 20:11:31,874 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:11:33,870 P40223 INFO Negative sampling done
2021-01-02 20:11:56,327 P40223 INFO --- Start evaluation ---
2021-01-02 20:11:57,070 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:12:33,556 P40223 INFO [Metrics] Recall(k=20): 0.055537 - Recall(k=50): 0.108966 - NDCG(k=20): 0.045025 - NDCG(k=50): 0.064892 - HitRate(k=20): 0.354869 - HitRate(k=50): 0.548977
2021-01-02 20:12:33,585 P40223 INFO Save best model: monitor(max): 0.055537
2021-01-02 20:12:33,620 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:12:33,701 P40223 INFO Train loss: 0.060419
2021-01-02 20:12:33,701 P40223 INFO ************ Epoch=18 end ************
2021-01-02 20:12:33,704 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:12:36,355 P40223 INFO Negative sampling done
2021-01-02 20:12:56,508 P40223 INFO --- Start evaluation ---
2021-01-02 20:12:57,294 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:13:37,149 P40223 INFO [Metrics] Recall(k=20): 0.055911 - Recall(k=50): 0.109726 - NDCG(k=20): 0.045354 - NDCG(k=50): 0.065395 - HitRate(k=20): 0.356164 - HitRate(k=50): 0.551598
2021-01-02 20:13:37,169 P40223 INFO Save best model: monitor(max): 0.055911
2021-01-02 20:13:37,220 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:13:37,294 P40223 INFO Train loss: 0.059263
2021-01-02 20:13:37,294 P40223 INFO ************ Epoch=19 end ************
2021-01-02 20:13:37,299 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:13:39,423 P40223 INFO Negative sampling done
2021-01-02 20:13:57,707 P40223 INFO --- Start evaluation ---
2021-01-02 20:13:58,559 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:14:36,571 P40223 INFO [Metrics] Recall(k=20): 0.056153 - Recall(k=50): 0.109754 - NDCG(k=20): 0.045701 - NDCG(k=50): 0.065647 - HitRate(k=20): 0.356732 - HitRate(k=50): 0.552071
2021-01-02 20:14:36,584 P40223 INFO Save best model: monitor(max): 0.056153
2021-01-02 20:14:36,613 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:14:36,726 P40223 INFO Train loss: 0.058180
2021-01-02 20:14:36,727 P40223 INFO ************ Epoch=20 end ************
2021-01-02 20:14:36,732 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:14:39,561 P40223 INFO Negative sampling done
2021-01-02 20:15:01,871 P40223 INFO --- Start evaluation ---
2021-01-02 20:15:02,673 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:15:38,834 P40223 INFO [Metrics] Recall(k=20): 0.056185 - Recall(k=50): 0.110415 - NDCG(k=20): 0.045613 - NDCG(k=50): 0.065836 - HitRate(k=20): 0.357901 - HitRate(k=50): 0.555261
2021-01-02 20:15:38,844 P40223 INFO Save best model: monitor(max): 0.056185
2021-01-02 20:15:38,872 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:15:38,941 P40223 INFO Train loss: 0.057335
2021-01-02 20:15:38,942 P40223 INFO ************ Epoch=21 end ************
2021-01-02 20:15:38,946 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:15:40,812 P40223 INFO Negative sampling done
2021-01-02 20:16:00,859 P40223 INFO --- Start evaluation ---
2021-01-02 20:16:01,707 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:16:38,940 P40223 INFO [Metrics] Recall(k=20): 0.056504 - Recall(k=50): 0.110946 - NDCG(k=20): 0.045856 - NDCG(k=50): 0.066101 - HitRate(k=20): 0.357301 - HitRate(k=50): 0.555734
2021-01-02 20:16:38,953 P40223 INFO Save best model: monitor(max): 0.056504
2021-01-02 20:16:38,985 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:16:39,054 P40223 INFO Train loss: 0.056552
2021-01-02 20:16:39,054 P40223 INFO ************ Epoch=22 end ************
2021-01-02 20:16:39,055 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:16:41,281 P40223 INFO Negative sampling done
2021-01-02 20:17:04,285 P40223 INFO --- Start evaluation ---
2021-01-02 20:17:05,312 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:17:42,669 P40223 INFO [Metrics] Recall(k=20): 0.056758 - Recall(k=50): 0.111592 - NDCG(k=20): 0.046063 - NDCG(k=50): 0.066470 - HitRate(k=20): 0.359416 - HitRate(k=50): 0.556240
2021-01-02 20:17:42,680 P40223 INFO Save best model: monitor(max): 0.056758
2021-01-02 20:17:42,708 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:17:42,809 P40223 INFO Train loss: 0.055911
2021-01-02 20:17:42,809 P40223 INFO ************ Epoch=23 end ************
2021-01-02 20:17:42,823 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:17:44,951 P40223 INFO Negative sampling done
2021-01-02 20:18:02,914 P40223 INFO --- Start evaluation ---
2021-01-02 20:18:03,674 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:18:38,982 P40223 INFO [Metrics] Recall(k=20): 0.056577 - Recall(k=50): 0.111352 - NDCG(k=20): 0.045969 - NDCG(k=50): 0.066423 - HitRate(k=20): 0.358406 - HitRate(k=50): 0.555703
2021-01-02 20:18:39,001 P40223 INFO Monitor(max) STOP: 0.056577 !
2021-01-02 20:18:39,001 P40223 INFO Reduce learning rate on plateau: 0.000100
2021-01-02 20:18:39,001 P40223 INFO Load best model:  MF_yelp18_x0_001_1426380b_model.ckpt
2021-01-02 20:18:39,024 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:18:39,133 P40223 INFO Train loss: 0.055332
2021-01-02 20:18:39,133 P40223 INFO ************ Epoch=24 end ************
2021-01-02 20:18:39,138 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:18:41,326 P40223 INFO Negative sampling done
2021-01-02 20:19:00,018 P40223 INFO --- Start evaluation ---
2021-01-02 20:19:01,336 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:19:38,163 P40223 INFO [Metrics] Recall(k=20): 0.056931 - Recall(k=50): 0.111801 - NDCG(k=20): 0.046253 - NDCG(k=50): 0.066698 - HitRate(k=20): 0.359637 - HitRate(k=50): 0.557219
2021-01-02 20:19:38,176 P40223 INFO Save best model: monitor(max): 0.056931
2021-01-02 20:19:38,205 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:19:38,263 P40223 INFO Train loss: 0.053563
2021-01-02 20:19:38,263 P40223 INFO ************ Epoch=25 end ************
2021-01-02 20:19:38,268 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:19:40,227 P40223 INFO Negative sampling done
2021-01-02 20:20:00,896 P40223 INFO --- Start evaluation ---
2021-01-02 20:20:01,742 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:20:37,176 P40223 INFO [Metrics] Recall(k=20): 0.057150 - Recall(k=50): 0.112024 - NDCG(k=20): 0.046465 - NDCG(k=50): 0.066896 - HitRate(k=20): 0.359859 - HitRate(k=50): 0.558134
2021-01-02 20:20:37,196 P40223 INFO Save best model: monitor(max): 0.057150
2021-01-02 20:20:37,252 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:20:37,332 P40223 INFO Train loss: 0.053428
2021-01-02 20:20:37,333 P40223 INFO ************ Epoch=26 end ************
2021-01-02 20:20:37,337 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:20:39,625 P40223 INFO Negative sampling done
2021-01-02 20:20:59,969 P40223 INFO --- Start evaluation ---
2021-01-02 20:21:00,695 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:21:39,069 P40223 INFO [Metrics] Recall(k=20): 0.057232 - Recall(k=50): 0.112086 - NDCG(k=20): 0.046549 - NDCG(k=50): 0.066981 - HitRate(k=20): 0.360332 - HitRate(k=50): 0.557882
2021-01-02 20:21:39,100 P40223 INFO Save best model: monitor(max): 0.057232
2021-01-02 20:21:39,155 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:21:39,250 P40223 INFO Train loss: 0.053362
2021-01-02 20:21:39,251 P40223 INFO ************ Epoch=27 end ************
2021-01-02 20:21:39,256 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:21:41,189 P40223 INFO Negative sampling done
2021-01-02 20:22:03,693 P40223 INFO --- Start evaluation ---
2021-01-02 20:22:04,607 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:22:43,247 P40223 INFO [Metrics] Recall(k=20): 0.057310 - Recall(k=50): 0.112090 - NDCG(k=20): 0.046586 - NDCG(k=50): 0.066993 - HitRate(k=20): 0.360774 - HitRate(k=50): 0.558229
2021-01-02 20:22:43,259 P40223 INFO Save best model: monitor(max): 0.057310
2021-01-02 20:22:43,285 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:22:43,383 P40223 INFO Train loss: 0.053302
2021-01-02 20:22:43,383 P40223 INFO ************ Epoch=28 end ************
2021-01-02 20:22:43,388 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:22:45,540 P40223 INFO Negative sampling done
2021-01-02 20:23:05,690 P40223 INFO --- Start evaluation ---
2021-01-02 20:23:06,582 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:23:42,722 P40223 INFO [Metrics] Recall(k=20): 0.057372 - Recall(k=50): 0.112213 - NDCG(k=20): 0.046609 - NDCG(k=50): 0.067028 - HitRate(k=20): 0.360806 - HitRate(k=50): 0.558198
2021-01-02 20:23:42,733 P40223 INFO Save best model: monitor(max): 0.057372
2021-01-02 20:23:42,759 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:23:42,839 P40223 INFO Train loss: 0.053175
2021-01-02 20:23:42,839 P40223 INFO ************ Epoch=29 end ************
2021-01-02 20:23:42,840 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:23:44,828 P40223 INFO Negative sampling done
2021-01-02 20:24:06,916 P40223 INFO --- Start evaluation ---
2021-01-02 20:24:07,847 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:24:48,473 P40223 INFO [Metrics] Recall(k=20): 0.057390 - Recall(k=50): 0.112291 - NDCG(k=20): 0.046661 - NDCG(k=50): 0.067104 - HitRate(k=20): 0.361122 - HitRate(k=50): 0.558450
2021-01-02 20:24:48,485 P40223 INFO Save best model: monitor(max): 0.057390
2021-01-02 20:24:48,542 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:24:48,620 P40223 INFO Train loss: 0.053138
2021-01-02 20:24:48,620 P40223 INFO ************ Epoch=30 end ************
2021-01-02 20:24:48,625 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:24:50,604 P40223 INFO Negative sampling done
2021-01-02 20:25:10,273 P40223 INFO --- Start evaluation ---
2021-01-02 20:25:11,347 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:25:47,626 P40223 INFO [Metrics] Recall(k=20): 0.057443 - Recall(k=50): 0.112214 - NDCG(k=20): 0.046704 - NDCG(k=50): 0.067091 - HitRate(k=20): 0.361469 - HitRate(k=50): 0.558798
2021-01-02 20:25:47,638 P40223 INFO Save best model: monitor(max): 0.057443
2021-01-02 20:25:47,667 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:25:47,732 P40223 INFO Train loss: 0.053082
2021-01-02 20:25:47,732 P40223 INFO ************ Epoch=31 end ************
2021-01-02 20:25:47,737 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:25:50,342 P40223 INFO Negative sampling done
2021-01-02 20:26:11,428 P40223 INFO --- Start evaluation ---
2021-01-02 20:26:12,264 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:26:49,682 P40223 INFO [Metrics] Recall(k=20): 0.057626 - Recall(k=50): 0.112296 - NDCG(k=20): 0.046753 - NDCG(k=50): 0.067092 - HitRate(k=20): 0.362543 - HitRate(k=50): 0.557787
2021-01-02 20:26:49,713 P40223 INFO Save best model: monitor(max): 0.057626
2021-01-02 20:26:49,744 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:26:49,815 P40223 INFO Train loss: 0.053033
2021-01-02 20:26:49,815 P40223 INFO ************ Epoch=32 end ************
2021-01-02 20:26:49,829 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:26:51,748 P40223 INFO Negative sampling done
2021-01-02 20:27:09,639 P40223 INFO --- Start evaluation ---
2021-01-02 20:27:10,336 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:27:47,181 P40223 INFO [Metrics] Recall(k=20): 0.057554 - Recall(k=50): 0.112169 - NDCG(k=20): 0.046761 - NDCG(k=50): 0.067108 - HitRate(k=20): 0.362385 - HitRate(k=50): 0.558040
2021-01-02 20:27:47,204 P40223 INFO Monitor(max) STOP: 0.057554 !
2021-01-02 20:27:47,204 P40223 INFO Reduce learning rate on plateau: 0.000010
2021-01-02 20:27:47,204 P40223 INFO Load best model:  MF_yelp18_x0_001_1426380b_model.ckpt
2021-01-02 20:27:47,231 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:27:47,309 P40223 INFO Train loss: 0.052948
2021-01-02 20:27:47,309 P40223 INFO ************ Epoch=33 end ************
2021-01-02 20:27:47,314 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:27:49,416 P40223 INFO Negative sampling done
2021-01-02 20:28:09,414 P40223 INFO --- Start evaluation ---
2021-01-02 20:28:10,194 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:28:47,357 P40223 INFO [Metrics] Recall(k=20): 0.057592 - Recall(k=50): 0.112294 - NDCG(k=20): 0.046735 - NDCG(k=50): 0.067089 - HitRate(k=20): 0.362448 - HitRate(k=50): 0.557850
2021-01-02 20:28:47,377 P40223 INFO Monitor(max) STOP: 0.057592 !
2021-01-02 20:28:47,377 P40223 INFO Reduce learning rate on plateau: 0.000001
2021-01-02 20:28:47,377 P40223 INFO Load best model:  MF_yelp18_x0_001_1426380b_model.ckpt
2021-01-02 20:28:47,408 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:28:47,483 P40223 INFO Train loss: 0.052752
2021-01-02 20:28:47,484 P40223 INFO ************ Epoch=34 end ************
2021-01-02 20:28:47,488 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:28:49,653 P40223 INFO Negative sampling done
2021-01-02 20:29:12,158 P40223 INFO --- Start evaluation ---
2021-01-02 20:29:13,220 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:29:54,526 P40223 INFO [Metrics] Recall(k=20): 0.057632 - Recall(k=50): 0.112270 - NDCG(k=20): 0.046756 - NDCG(k=50): 0.067085 - HitRate(k=20): 0.362448 - HitRate(k=50): 0.557692
2021-01-02 20:29:54,539 P40223 INFO Save best model: monitor(max): 0.057632
2021-01-02 20:29:54,574 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:29:54,659 P40223 INFO Train loss: 0.052741
2021-01-02 20:29:54,660 P40223 INFO ************ Epoch=35 end ************
2021-01-02 20:29:54,660 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:29:56,524 P40223 INFO Negative sampling done
2021-01-02 20:30:17,579 P40223 INFO --- Start evaluation ---
2021-01-02 20:30:18,366 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:30:55,621 P40223 INFO [Metrics] Recall(k=20): 0.057629 - Recall(k=50): 0.112261 - NDCG(k=20): 0.046754 - NDCG(k=50): 0.067082 - HitRate(k=20): 0.362353 - HitRate(k=50): 0.557661
2021-01-02 20:30:55,641 P40223 INFO Monitor(max) STOP: 0.057629 !
2021-01-02 20:30:55,641 P40223 INFO Reduce learning rate on plateau: 0.000001
2021-01-02 20:30:55,642 P40223 INFO Load best model:  MF_yelp18_x0_001_1426380b_model.ckpt
2021-01-02 20:30:55,666 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:30:55,724 P40223 INFO Train loss: 0.052713
2021-01-02 20:30:55,724 P40223 INFO ************ Epoch=36 end ************
2021-01-02 20:30:55,725 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:30:57,659 P40223 INFO Negative sampling done
2021-01-02 20:31:16,804 P40223 INFO --- Start evaluation ---
2021-01-02 20:31:17,773 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:31:54,606 P40223 INFO [Metrics] Recall(k=20): 0.057631 - Recall(k=50): 0.112278 - NDCG(k=20): 0.046743 - NDCG(k=50): 0.067076 - HitRate(k=20): 0.362511 - HitRate(k=50): 0.557661
2021-01-02 20:31:54,623 P40223 INFO Monitor(max) STOP: 0.057631 !
2021-01-02 20:31:54,623 P40223 INFO Reduce learning rate on plateau: 0.000001
2021-01-02 20:31:54,623 P40223 INFO Load best model:  MF_yelp18_x0_001_1426380b_model.ckpt
2021-01-02 20:31:54,639 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:31:54,742 P40223 INFO Train loss: 0.052726
2021-01-02 20:31:54,742 P40223 INFO ************ Epoch=37 end ************
2021-01-02 20:31:54,747 P40223 INFO Negative sampling num_negs=50
2021-01-02 20:31:57,289 P40223 INFO Negative sampling done
2021-01-02 20:32:17,733 P40223 INFO --- Start evaluation ---
2021-01-02 20:32:18,608 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:32:55,857 P40223 INFO [Metrics] Recall(k=20): 0.057622 - Recall(k=50): 0.112274 - NDCG(k=20): 0.046747 - NDCG(k=50): 0.067082 - HitRate(k=20): 0.362416 - HitRate(k=50): 0.557629
2021-01-02 20:32:55,870 P40223 INFO Monitor(max) STOP: 0.057622 !
2021-01-02 20:32:55,871 P40223 INFO Reduce learning rate on plateau: 0.000001
2021-01-02 20:32:55,871 P40223 INFO Load best model:  MF_yelp18_x0_001_1426380b_model.ckpt
2021-01-02 20:32:55,883 P40223 INFO Early stopping at epoch=38
2021-01-02 20:32:55,883 P40223 INFO --- 1209/1209 batches finished ---
2021-01-02 20:32:55,952 P40223 INFO Train loss: 0.052741
2021-01-02 20:32:55,952 P40223 INFO Training finished.
2021-01-02 20:32:55,952 P40223 INFO Load best model:  MF_yelp18_x0_001_1426380b_model.ckpt
2021-01-02 20:32:55,973 P40223 INFO ****** Train/validation evaluation ******
2021-01-02 20:32:55,973 P40223 INFO --- Start evaluation ---
2021-01-02 20:32:56,820 P40223 INFO Evaluating metrics for 31668 users...
2021-01-02 20:33:35,654 P40223 INFO [Metrics] Recall(k=20): 0.057632 - Recall(k=50): 0.112270 - NDCG(k=20): 0.046756 - NDCG(k=50): 0.067085 - HitRate(k=20): 0.362448 - HitRate(k=50): 0.557692
```