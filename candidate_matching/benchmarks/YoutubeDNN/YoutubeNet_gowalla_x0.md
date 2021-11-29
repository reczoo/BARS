## YoutubeNet_gowalla_x0 

A notebook to benchmark YoutubeNet on gowalla dataset.

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
2021-01-03 18:00:56,225 P3152 INFO [Metrics] Recall(k=20): 0.175436 - Recall(k=50): 0.271103 - NDCG(k=20): 0.147322 - NDCG(k=50): 0.177231 - HitRate(k=20): 0.562797 - HitRate(k=50): 0.697836
```


### Logs
```
2021-01-03 17:21:14,528 P3152 INFO Set up feature encoder...
2021-01-03 17:21:14,528 P3152 INFO Reading file: ../data/Gowalla/gowalla_x0/train.csv
2021-01-03 17:21:20,531 P3152 INFO Reading file: ../data/Gowalla/gowalla_x0/item_corpus.csv
2021-01-03 17:21:20,589 P3152 INFO Preprocess feature columns...
2021-01-03 17:21:20,607 P3152 INFO Preprocess feature columns...
2021-01-03 17:21:22,154 P3152 INFO Fit feature encoder...
2021-01-03 17:21:22,284 P3152 INFO Processing column: {'active': True, 'dtype': 'int', 'name': 'query_index', 'type': 'index'}
2021-01-03 17:21:22,284 P3152 INFO Processing column: {'active': True, 'dtype': 'int', 'name': 'corpus_index', 'type': 'index'}
2021-01-03 17:21:22,284 P3152 INFO Processing column: {'active': True, 'dtype': 'str', 'embedding_callback': 'layers.MaskedAveragePooling()', 'max_len': 500, 'name': 'user_history', 'padding': 'pre', 'source': 'user', 'splitter': '^', 'type': 'sequence'}
2021-01-03 17:22:23,291 P3152 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'item_id', 'share_embedding': 'user_history', 'source': 'item', 'type': 'categorical'}
2021-01-03 17:22:23,291 P3152 INFO Pickle feature_encode: ../data/Gowalla/gowalla_x0_2bb544c2/feature_encoder.pkl
2021-01-03 17:22:23,303 P3152 INFO Save feature_map to json: ../data/Gowalla/gowalla_x0_2bb544c2/feature_map.json
2021-01-03 17:22:23,305 P3152 INFO Set feature encoder done.
2021-01-03 17:22:23,393 P3152 INFO Transform feature columns...
2021-01-03 17:22:23,428 P3152 INFO Saving data to h5: ../data/Gowalla/gowalla_x0_2bb544c2/item_corpus.h5
2021-01-03 17:22:23,492 P3152 INFO Transform feature columns...
2021-01-03 17:23:16,597 P3152 INFO Saving data to h5: ../data/Gowalla/gowalla_x0_2bb544c2/train.h5
2021-01-03 17:23:17,776 P3152 INFO Reading file: ../data/Gowalla/gowalla_x0/test.csv
2021-01-03 17:23:18,427 P3152 INFO Preprocess feature columns...
2021-01-03 17:23:18,597 P3152 INFO Transform feature columns...
2021-01-03 17:23:26,717 P3152 INFO Saving data to h5: ../data/Gowalla/gowalla_x0_2bb544c2/valid.h5
2021-01-03 17:23:27,047 P3152 INFO Transform csv data to h5 done.
2021-01-03 17:23:29,676 P3152 INFO Total number of parameters: 2622912.
2021-01-03 17:23:29,676 P3152 INFO Loading data...
2021-01-03 17:23:29,680 P3152 INFO Loading data from h5: ../data/Gowalla/gowalla_x0_2bb544c2/train.h5
2021-01-03 17:23:30,429 P3152 INFO Loading data from h5: ../data/Gowalla/gowalla_x0_2bb544c2/item_corpus.h5
2021-01-03 17:23:30,722 P3152 INFO Loading data from h5: ../data/Gowalla/gowalla_x0_2bb544c2/valid.h5
2021-01-03 17:23:31,040 P3152 INFO Loading data from h5: ../data/Gowalla/gowalla_x0_2bb544c2/item_corpus.h5
2021-01-03 17:23:31,042 P3152 INFO Train samples: total/810128, blocks/1
2021-01-03 17:23:31,043 P3152 INFO Validation samples: total/29858, blocks/1
2021-01-03 17:23:31,043 P3152 INFO Loading train data done.
2021-01-03 17:23:31,043 P3152 INFO **** Start training: 1583 batches/epoch ****
2021-01-03 17:23:31,044 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:23:43,863 P3152 INFO Negative sampling done
2021-01-03 17:24:02,724 P3152 INFO --- Start evaluation ---
2021-01-03 17:24:03,355 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:24:24,690 P3152 INFO [Metrics] Recall(k=20): 0.104559 - Recall(k=50): 0.160314 - NDCG(k=20): 0.093041 - NDCG(k=50): 0.109980 - HitRate(k=20): 0.418179 - HitRate(k=50): 0.539386
2021-01-03 17:24:24,699 P3152 INFO Save best model: monitor(max): 0.104559
2021-01-03 17:24:24,708 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:24:24,759 P3152 INFO Train loss: 5.387370
2021-01-03 17:24:24,759 P3152 INFO ************ Epoch=1 end ************
2021-01-03 17:24:24,759 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:24:37,541 P3152 INFO Negative sampling done
2021-01-03 17:24:56,057 P3152 INFO --- Start evaluation ---
2021-01-03 17:24:56,678 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:25:17,219 P3152 INFO [Metrics] Recall(k=20): 0.118808 - Recall(k=50): 0.181244 - NDCG(k=20): 0.104635 - NDCG(k=50): 0.123766 - HitRate(k=20): 0.454485 - HitRate(k=50): 0.579007
2021-01-03 17:25:17,227 P3152 INFO Save best model: monitor(max): 0.118808
2021-01-03 17:25:17,239 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:25:17,287 P3152 INFO Train loss: 4.534333
2021-01-03 17:25:17,287 P3152 INFO ************ Epoch=2 end ************
2021-01-03 17:25:17,287 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:25:30,458 P3152 INFO Negative sampling done
2021-01-03 17:25:48,719 P3152 INFO --- Start evaluation ---
2021-01-03 17:25:49,355 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:26:10,058 P3152 INFO [Metrics] Recall(k=20): 0.126506 - Recall(k=50): 0.193915 - NDCG(k=20): 0.111310 - NDCG(k=50): 0.131990 - HitRate(k=20): 0.474111 - HitRate(k=50): 0.601212
2021-01-03 17:26:10,069 P3152 INFO Save best model: monitor(max): 0.126506
2021-01-03 17:26:10,081 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:26:10,138 P3152 INFO Train loss: 4.286157
2021-01-03 17:26:10,139 P3152 INFO ************ Epoch=3 end ************
2021-01-03 17:26:10,139 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:26:22,976 P3152 INFO Negative sampling done
2021-01-03 17:26:41,793 P3152 INFO --- Start evaluation ---
2021-01-03 17:26:42,448 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:27:03,334 P3152 INFO [Metrics] Recall(k=20): 0.132699 - Recall(k=50): 0.203873 - NDCG(k=20): 0.115854 - NDCG(k=50): 0.137657 - HitRate(k=20): 0.487240 - HitRate(k=50): 0.617021
2021-01-03 17:27:03,344 P3152 INFO Save best model: monitor(max): 0.132699
2021-01-03 17:27:03,356 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:27:03,403 P3152 INFO Train loss: 4.143870
2021-01-03 17:27:03,403 P3152 INFO ************ Epoch=4 end ************
2021-01-03 17:27:03,404 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:27:16,224 P3152 INFO Negative sampling done
2021-01-03 17:27:34,912 P3152 INFO --- Start evaluation ---
2021-01-03 17:27:35,548 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:27:56,496 P3152 INFO [Metrics] Recall(k=20): 0.137739 - Recall(k=50): 0.212680 - NDCG(k=20): 0.119754 - NDCG(k=50): 0.142841 - HitRate(k=20): 0.499632 - HitRate(k=50): 0.629982
2021-01-03 17:27:56,507 P3152 INFO Save best model: monitor(max): 0.137739
2021-01-03 17:27:56,519 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:27:56,561 P3152 INFO Train loss: 4.042655
2021-01-03 17:27:56,561 P3152 INFO ************ Epoch=5 end ************
2021-01-03 17:27:56,562 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:28:09,429 P3152 INFO Negative sampling done
2021-01-03 17:28:27,915 P3152 INFO --- Start evaluation ---
2021-01-03 17:28:28,560 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:28:50,244 P3152 INFO [Metrics] Recall(k=20): 0.142140 - Recall(k=50): 0.218830 - NDCG(k=20): 0.123737 - NDCG(k=50): 0.147380 - HitRate(k=20): 0.507938 - HitRate(k=50): 0.637216
2021-01-03 17:28:50,252 P3152 INFO Save best model: monitor(max): 0.142140
2021-01-03 17:28:50,266 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:28:50,315 P3152 INFO Train loss: 3.963017
2021-01-03 17:28:50,316 P3152 INFO ************ Epoch=6 end ************
2021-01-03 17:28:50,316 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:29:03,132 P3152 INFO Negative sampling done
2021-01-03 17:29:22,309 P3152 INFO --- Start evaluation ---
2021-01-03 17:29:22,937 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:29:45,500 P3152 INFO [Metrics] Recall(k=20): 0.146666 - Recall(k=50): 0.226464 - NDCG(k=20): 0.126078 - NDCG(k=50): 0.150742 - HitRate(k=20): 0.516445 - HitRate(k=50): 0.647599
2021-01-03 17:29:45,508 P3152 INFO Save best model: monitor(max): 0.146666
2021-01-03 17:29:45,520 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:29:45,570 P3152 INFO Train loss: 3.896832
2021-01-03 17:29:45,570 P3152 INFO ************ Epoch=7 end ************
2021-01-03 17:29:45,570 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:29:58,357 P3152 INFO Negative sampling done
2021-01-03 17:30:18,005 P3152 INFO --- Start evaluation ---
2021-01-03 17:30:18,697 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:30:41,313 P3152 INFO [Metrics] Recall(k=20): 0.149407 - Recall(k=50): 0.231431 - NDCG(k=20): 0.128372 - NDCG(k=50): 0.153753 - HitRate(k=20): 0.520698 - HitRate(k=50): 0.653225
2021-01-03 17:30:41,323 P3152 INFO Save best model: monitor(max): 0.149407
2021-01-03 17:30:41,334 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:30:41,388 P3152 INFO Train loss: 3.839689
2021-01-03 17:30:41,388 P3152 INFO ************ Epoch=8 end ************
2021-01-03 17:30:41,389 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:30:54,361 P3152 INFO Negative sampling done
2021-01-03 17:31:14,436 P3152 INFO --- Start evaluation ---
2021-01-03 17:31:15,124 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:31:37,808 P3152 INFO [Metrics] Recall(k=20): 0.152897 - Recall(k=50): 0.236610 - NDCG(k=20): 0.131096 - NDCG(k=50): 0.157075 - HitRate(k=20): 0.527162 - HitRate(k=50): 0.660995
2021-01-03 17:31:37,820 P3152 INFO Save best model: monitor(max): 0.152897
2021-01-03 17:31:37,833 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:31:37,881 P3152 INFO Train loss: 3.789163
2021-01-03 17:31:37,881 P3152 INFO ************ Epoch=9 end ************
2021-01-03 17:31:37,882 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:31:50,830 P3152 INFO Negative sampling done
2021-01-03 17:32:12,278 P3152 INFO --- Start evaluation ---
2021-01-03 17:32:12,958 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:32:36,879 P3152 INFO [Metrics] Recall(k=20): 0.154207 - Recall(k=50): 0.240294 - NDCG(k=20): 0.132615 - NDCG(k=50): 0.159325 - HitRate(k=20): 0.529272 - HitRate(k=50): 0.664713
2021-01-03 17:32:36,891 P3152 INFO Save best model: monitor(max): 0.154207
2021-01-03 17:32:36,903 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:32:36,955 P3152 INFO Train loss: 3.743421
2021-01-03 17:32:36,955 P3152 INFO ************ Epoch=10 end ************
2021-01-03 17:32:36,955 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:32:50,310 P3152 INFO Negative sampling done
2021-01-03 17:33:09,163 P3152 INFO --- Start evaluation ---
2021-01-03 17:33:09,853 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:33:31,215 P3152 INFO [Metrics] Recall(k=20): 0.157770 - Recall(k=50): 0.245084 - NDCG(k=20): 0.134845 - NDCG(k=50): 0.161858 - HitRate(k=20): 0.534798 - HitRate(k=50): 0.668832
2021-01-03 17:33:31,224 P3152 INFO Save best model: monitor(max): 0.157770
2021-01-03 17:33:31,235 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:33:31,282 P3152 INFO Train loss: 3.703778
2021-01-03 17:33:31,282 P3152 INFO ************ Epoch=11 end ************
2021-01-03 17:33:31,283 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:33:44,962 P3152 INFO Negative sampling done
2021-01-03 17:34:02,937 P3152 INFO --- Start evaluation ---
2021-01-03 17:34:03,631 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:34:25,468 P3152 INFO [Metrics] Recall(k=20): 0.158938 - Recall(k=50): 0.246756 - NDCG(k=20): 0.136477 - NDCG(k=50): 0.163675 - HitRate(k=20): 0.538181 - HitRate(k=50): 0.669804
2021-01-03 17:34:25,479 P3152 INFO Save best model: monitor(max): 0.158938
2021-01-03 17:34:25,491 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:34:25,541 P3152 INFO Train loss: 3.666709
2021-01-03 17:34:25,541 P3152 INFO ************ Epoch=12 end ************
2021-01-03 17:34:25,542 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:34:38,848 P3152 INFO Negative sampling done
2021-01-03 17:34:57,334 P3152 INFO --- Start evaluation ---
2021-01-03 17:34:57,993 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:35:21,471 P3152 INFO [Metrics] Recall(k=20): 0.161185 - Recall(k=50): 0.250224 - NDCG(k=20): 0.138054 - NDCG(k=50): 0.165627 - HitRate(k=20): 0.541865 - HitRate(k=50): 0.674493
2021-01-03 17:35:21,481 P3152 INFO Save best model: monitor(max): 0.161185
2021-01-03 17:35:21,494 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:35:21,543 P3152 INFO Train loss: 3.632344
2021-01-03 17:35:21,543 P3152 INFO ************ Epoch=13 end ************
2021-01-03 17:35:21,544 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:35:36,439 P3152 INFO Negative sampling done
2021-01-03 17:35:54,849 P3152 INFO --- Start evaluation ---
2021-01-03 17:35:55,493 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:36:18,563 P3152 INFO [Metrics] Recall(k=20): 0.163387 - Recall(k=50): 0.252967 - NDCG(k=20): 0.139048 - NDCG(k=50): 0.166872 - HitRate(k=20): 0.544544 - HitRate(k=50): 0.677741
2021-01-03 17:36:18,572 P3152 INFO Save best model: monitor(max): 0.163387
2021-01-03 17:36:18,585 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:36:18,662 P3152 INFO Train loss: 3.598568
2021-01-03 17:36:18,662 P3152 INFO ************ Epoch=14 end ************
2021-01-03 17:36:18,662 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:36:32,024 P3152 INFO Negative sampling done
2021-01-03 17:36:50,592 P3152 INFO --- Start evaluation ---
2021-01-03 17:36:51,269 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:37:14,068 P3152 INFO [Metrics] Recall(k=20): 0.165342 - Recall(k=50): 0.255197 - NDCG(k=20): 0.140868 - NDCG(k=50): 0.168838 - HitRate(k=20): 0.547726 - HitRate(k=50): 0.680387
2021-01-03 17:37:14,082 P3152 INFO Save best model: monitor(max): 0.165342
2021-01-03 17:37:14,094 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:37:14,156 P3152 INFO Train loss: 3.567834
2021-01-03 17:37:14,156 P3152 INFO ************ Epoch=15 end ************
2021-01-03 17:37:14,157 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:37:27,520 P3152 INFO Negative sampling done
2021-01-03 17:37:46,105 P3152 INFO --- Start evaluation ---
2021-01-03 17:37:46,759 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:38:10,652 P3152 INFO [Metrics] Recall(k=20): 0.166560 - Recall(k=50): 0.257243 - NDCG(k=20): 0.140904 - NDCG(k=50): 0.169161 - HitRate(k=20): 0.549099 - HitRate(k=50): 0.682095
2021-01-03 17:38:10,662 P3152 INFO Save best model: monitor(max): 0.166560
2021-01-03 17:38:10,673 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:38:10,723 P3152 INFO Train loss: 3.538930
2021-01-03 17:38:10,723 P3152 INFO ************ Epoch=16 end ************
2021-01-03 17:38:10,724 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:38:24,872 P3152 INFO Negative sampling done
2021-01-03 17:38:43,882 P3152 INFO --- Start evaluation ---
2021-01-03 17:38:44,523 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:39:06,355 P3152 INFO [Metrics] Recall(k=20): 0.167544 - Recall(k=50): 0.260373 - NDCG(k=20): 0.142784 - NDCG(k=50): 0.171627 - HitRate(k=20): 0.550841 - HitRate(k=50): 0.686148
2021-01-03 17:39:06,364 P3152 INFO Save best model: monitor(max): 0.167544
2021-01-03 17:39:06,377 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:39:06,429 P3152 INFO Train loss: 3.511080
2021-01-03 17:39:06,429 P3152 INFO ************ Epoch=17 end ************
2021-01-03 17:39:06,430 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:39:19,265 P3152 INFO Negative sampling done
2021-01-03 17:39:37,775 P3152 INFO --- Start evaluation ---
2021-01-03 17:39:38,414 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:40:00,997 P3152 INFO [Metrics] Recall(k=20): 0.168873 - Recall(k=50): 0.261784 - NDCG(k=20): 0.143236 - NDCG(k=50): 0.172203 - HitRate(k=20): 0.552750 - HitRate(k=50): 0.687387
2021-01-03 17:40:01,006 P3152 INFO Save best model: monitor(max): 0.168873
2021-01-03 17:40:01,018 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:40:01,069 P3152 INFO Train loss: 3.486084
2021-01-03 17:40:01,069 P3152 INFO ************ Epoch=18 end ************
2021-01-03 17:40:01,070 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:40:13,994 P3152 INFO Negative sampling done
2021-01-03 17:40:33,446 P3152 INFO --- Start evaluation ---
2021-01-03 17:40:34,117 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:40:56,295 P3152 INFO [Metrics] Recall(k=20): 0.169795 - Recall(k=50): 0.263548 - NDCG(k=20): 0.143675 - NDCG(k=50): 0.172903 - HitRate(k=20): 0.554089 - HitRate(k=50): 0.689798
2021-01-03 17:40:56,306 P3152 INFO Save best model: monitor(max): 0.169795
2021-01-03 17:40:56,317 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:40:56,362 P3152 INFO Train loss: 3.462335
2021-01-03 17:40:56,363 P3152 INFO ************ Epoch=19 end ************
2021-01-03 17:40:56,363 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:41:10,741 P3152 INFO Negative sampling done
2021-01-03 17:41:32,449 P3152 INFO --- Start evaluation ---
2021-01-03 17:41:33,090 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:41:55,382 P3152 INFO [Metrics] Recall(k=20): 0.170466 - Recall(k=50): 0.264192 - NDCG(k=20): 0.143755 - NDCG(k=50): 0.172988 - HitRate(k=20): 0.554391 - HitRate(k=50): 0.690468
2021-01-03 17:41:55,391 P3152 INFO Save best model: monitor(max): 0.170466
2021-01-03 17:41:55,404 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:41:55,452 P3152 INFO Train loss: 3.439102
2021-01-03 17:41:55,452 P3152 INFO ************ Epoch=20 end ************
2021-01-03 17:41:55,452 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:42:08,272 P3152 INFO Negative sampling done
2021-01-03 17:42:27,047 P3152 INFO --- Start evaluation ---
2021-01-03 17:42:27,694 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:42:50,679 P3152 INFO [Metrics] Recall(k=20): 0.171923 - Recall(k=50): 0.265505 - NDCG(k=20): 0.144655 - NDCG(k=50): 0.173855 - HitRate(k=20): 0.556601 - HitRate(k=50): 0.690870
2021-01-03 17:42:50,688 P3152 INFO Save best model: monitor(max): 0.171923
2021-01-03 17:42:50,701 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:42:50,758 P3152 INFO Train loss: 3.417743
2021-01-03 17:42:50,758 P3152 INFO ************ Epoch=21 end ************
2021-01-03 17:42:50,759 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:43:03,683 P3152 INFO Negative sampling done
2021-01-03 17:43:23,093 P3152 INFO --- Start evaluation ---
2021-01-03 17:43:23,765 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:43:45,998 P3152 INFO [Metrics] Recall(k=20): 0.172209 - Recall(k=50): 0.266546 - NDCG(k=20): 0.145303 - NDCG(k=50): 0.174825 - HitRate(k=20): 0.558276 - HitRate(k=50): 0.692444
2021-01-03 17:43:46,008 P3152 INFO Save best model: monitor(max): 0.172209
2021-01-03 17:43:46,020 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:43:46,074 P3152 INFO Train loss: 3.397077
2021-01-03 17:43:46,074 P3152 INFO ************ Epoch=22 end ************
2021-01-03 17:43:46,075 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:43:58,942 P3152 INFO Negative sampling done
2021-01-03 17:44:22,444 P3152 INFO --- Start evaluation ---
2021-01-03 17:44:23,114 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:44:46,045 P3152 INFO [Metrics] Recall(k=20): 0.172655 - Recall(k=50): 0.267562 - NDCG(k=20): 0.145416 - NDCG(k=50): 0.175093 - HitRate(k=20): 0.558343 - HitRate(k=50): 0.693583
2021-01-03 17:44:46,053 P3152 INFO Save best model: monitor(max): 0.172655
2021-01-03 17:44:46,065 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:44:46,120 P3152 INFO Train loss: 3.377572
2021-01-03 17:44:46,121 P3152 INFO ************ Epoch=23 end ************
2021-01-03 17:44:46,121 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:44:59,121 P3152 INFO Negative sampling done
2021-01-03 17:45:17,611 P3152 INFO --- Start evaluation ---
2021-01-03 17:45:18,253 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:45:39,860 P3152 INFO [Metrics] Recall(k=20): 0.173089 - Recall(k=50): 0.267577 - NDCG(k=20): 0.145179 - NDCG(k=50): 0.174809 - HitRate(k=20): 0.558845 - HitRate(k=50): 0.693148
2021-01-03 17:45:39,871 P3152 INFO Save best model: monitor(max): 0.173089
2021-01-03 17:45:39,884 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:45:39,935 P3152 INFO Train loss: 3.359567
2021-01-03 17:45:39,935 P3152 INFO ************ Epoch=24 end ************
2021-01-03 17:45:39,935 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:45:52,824 P3152 INFO Negative sampling done
2021-01-03 17:46:12,373 P3152 INFO --- Start evaluation ---
2021-01-03 17:46:13,152 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:46:35,973 P3152 INFO [Metrics] Recall(k=20): 0.173378 - Recall(k=50): 0.268598 - NDCG(k=20): 0.145548 - NDCG(k=50): 0.175321 - HitRate(k=20): 0.560285 - HitRate(k=50): 0.695392
2021-01-03 17:46:35,982 P3152 INFO Save best model: monitor(max): 0.173378
2021-01-03 17:46:35,995 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:46:36,054 P3152 INFO Train loss: 3.342532
2021-01-03 17:46:36,054 P3152 INFO ************ Epoch=25 end ************
2021-01-03 17:46:36,054 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:46:48,849 P3152 INFO Negative sampling done
2021-01-03 17:47:09,674 P3152 INFO --- Start evaluation ---
2021-01-03 17:47:10,301 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:47:34,736 P3152 INFO [Metrics] Recall(k=20): 0.173443 - Recall(k=50): 0.268400 - NDCG(k=20): 0.145254 - NDCG(k=50): 0.175001 - HitRate(k=20): 0.559850 - HitRate(k=50): 0.693650
2021-01-03 17:47:34,746 P3152 INFO Save best model: monitor(max): 0.173443
2021-01-03 17:47:34,758 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:47:34,807 P3152 INFO Train loss: 3.326945
2021-01-03 17:47:34,807 P3152 INFO ************ Epoch=26 end ************
2021-01-03 17:47:34,808 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:47:47,589 P3152 INFO Negative sampling done
2021-01-03 17:48:06,050 P3152 INFO --- Start evaluation ---
2021-01-03 17:48:06,710 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:48:29,064 P3152 INFO [Metrics] Recall(k=20): 0.174330 - Recall(k=50): 0.269631 - NDCG(k=20): 0.146185 - NDCG(k=50): 0.176003 - HitRate(k=20): 0.562462 - HitRate(k=50): 0.695291
2021-01-03 17:48:29,074 P3152 INFO Save best model: monitor(max): 0.174330
2021-01-03 17:48:29,086 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:48:29,161 P3152 INFO Train loss: 3.311802
2021-01-03 17:48:29,161 P3152 INFO ************ Epoch=27 end ************
2021-01-03 17:48:29,162 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:48:42,086 P3152 INFO Negative sampling done
2021-01-03 17:49:01,531 P3152 INFO --- Start evaluation ---
2021-01-03 17:49:02,243 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:49:24,693 P3152 INFO [Metrics] Recall(k=20): 0.174033 - Recall(k=50): 0.269485 - NDCG(k=20): 0.145197 - NDCG(k=50): 0.175126 - HitRate(k=20): 0.561123 - HitRate(k=50): 0.695291
2021-01-03 17:49:24,703 P3152 INFO Monitor(max) STOP: 0.174033 !
2021-01-03 17:49:24,703 P3152 INFO Reduce learning rate on plateau: 0.000100
2021-01-03 17:49:24,703 P3152 INFO Load best model:  YouTubeNet_gowalla_x0_001_10a47ae1_model.ckpt
2021-01-03 17:49:24,712 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:49:24,774 P3152 INFO Train loss: 3.296436
2021-01-03 17:49:24,774 P3152 INFO ************ Epoch=28 end ************
2021-01-03 17:49:24,774 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:49:37,722 P3152 INFO Negative sampling done
2021-01-03 17:49:56,591 P3152 INFO --- Start evaluation ---
2021-01-03 17:49:57,241 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:50:22,663 P3152 INFO [Metrics] Recall(k=20): 0.174704 - Recall(k=50): 0.270422 - NDCG(k=20): 0.146444 - NDCG(k=50): 0.176405 - HitRate(k=20): 0.562295 - HitRate(k=50): 0.696530
2021-01-03 17:50:22,673 P3152 INFO Save best model: monitor(max): 0.174704
2021-01-03 17:50:22,686 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:50:22,748 P3152 INFO Train loss: 3.258965
2021-01-03 17:50:22,748 P3152 INFO ************ Epoch=29 end ************
2021-01-03 17:50:22,748 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:50:35,553 P3152 INFO Negative sampling done
2021-01-03 17:50:54,586 P3152 INFO --- Start evaluation ---
2021-01-03 17:50:55,211 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:51:16,936 P3152 INFO [Metrics] Recall(k=20): 0.174903 - Recall(k=50): 0.270654 - NDCG(k=20): 0.146824 - NDCG(k=50): 0.176780 - HitRate(k=20): 0.562965 - HitRate(k=50): 0.697368
2021-01-03 17:51:16,945 P3152 INFO Save best model: monitor(max): 0.174903
2021-01-03 17:51:16,957 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:51:17,009 P3152 INFO Train loss: 3.255395
2021-01-03 17:51:17,009 P3152 INFO ************ Epoch=30 end ************
2021-01-03 17:51:17,010 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:51:29,860 P3152 INFO Negative sampling done
2021-01-03 17:51:49,688 P3152 INFO --- Start evaluation ---
2021-01-03 17:51:50,481 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:52:13,746 P3152 INFO [Metrics] Recall(k=20): 0.175124 - Recall(k=50): 0.270942 - NDCG(k=20): 0.147070 - NDCG(k=50): 0.177034 - HitRate(k=20): 0.562496 - HitRate(k=50): 0.697535
2021-01-03 17:52:13,758 P3152 INFO Save best model: monitor(max): 0.175124
2021-01-03 17:52:13,772 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:52:13,834 P3152 INFO Train loss: 3.250947
2021-01-03 17:52:13,834 P3152 INFO ************ Epoch=31 end ************
2021-01-03 17:52:13,834 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:52:26,783 P3152 INFO Negative sampling done
2021-01-03 17:52:46,364 P3152 INFO --- Start evaluation ---
2021-01-03 17:52:47,009 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:53:11,064 P3152 INFO [Metrics] Recall(k=20): 0.175413 - Recall(k=50): 0.271160 - NDCG(k=20): 0.147177 - NDCG(k=50): 0.177107 - HitRate(k=20): 0.562496 - HitRate(k=50): 0.697535
2021-01-03 17:53:11,075 P3152 INFO Save best model: monitor(max): 0.175413
2021-01-03 17:53:11,088 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:53:11,153 P3152 INFO Train loss: 3.249035
2021-01-03 17:53:11,153 P3152 INFO ************ Epoch=32 end ************
2021-01-03 17:53:11,154 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:53:25,364 P3152 INFO Negative sampling done
2021-01-03 17:53:44,743 P3152 INFO --- Start evaluation ---
2021-01-03 17:53:45,386 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:54:08,127 P3152 INFO [Metrics] Recall(k=20): 0.175425 - Recall(k=50): 0.271080 - NDCG(k=20): 0.147334 - NDCG(k=50): 0.177241 - HitRate(k=20): 0.562697 - HitRate(k=50): 0.697769
2021-01-03 17:54:08,137 P3152 INFO Save best model: monitor(max): 0.175425
2021-01-03 17:54:08,150 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:54:08,200 P3152 INFO Train loss: 3.246985
2021-01-03 17:54:08,201 P3152 INFO ************ Epoch=33 end ************
2021-01-03 17:54:08,201 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:54:21,009 P3152 INFO Negative sampling done
2021-01-03 17:54:39,887 P3152 INFO --- Start evaluation ---
2021-01-03 17:54:41,179 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:55:04,472 P3152 INFO [Metrics] Recall(k=20): 0.175418 - Recall(k=50): 0.271478 - NDCG(k=20): 0.147354 - NDCG(k=50): 0.177365 - HitRate(k=20): 0.563233 - HitRate(k=50): 0.697836
2021-01-03 17:55:04,482 P3152 INFO Monitor(max) STOP: 0.175418 !
2021-01-03 17:55:04,482 P3152 INFO Reduce learning rate on plateau: 0.000010
2021-01-03 17:55:04,482 P3152 INFO Load best model:  YouTubeNet_gowalla_x0_001_10a47ae1_model.ckpt
2021-01-03 17:55:04,490 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:55:04,555 P3152 INFO Train loss: 3.247173
2021-01-03 17:55:04,555 P3152 INFO ************ Epoch=34 end ************
2021-01-03 17:55:04,556 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:55:18,407 P3152 INFO Negative sampling done
2021-01-03 17:55:35,946 P3152 INFO --- Start evaluation ---
2021-01-03 17:55:36,673 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:55:58,626 P3152 INFO [Metrics] Recall(k=20): 0.175392 - Recall(k=50): 0.271098 - NDCG(k=20): 0.147323 - NDCG(k=50): 0.177245 - HitRate(k=20): 0.562663 - HitRate(k=50): 0.697903
2021-01-03 17:55:58,636 P3152 INFO Monitor(max) STOP: 0.175392 !
2021-01-03 17:55:58,637 P3152 INFO Reduce learning rate on plateau: 0.000001
2021-01-03 17:55:58,637 P3152 INFO Load best model:  YouTubeNet_gowalla_x0_001_10a47ae1_model.ckpt
2021-01-03 17:55:58,645 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:55:58,906 P3152 INFO Train loss: 3.241156
2021-01-03 17:55:58,911 P3152 INFO ************ Epoch=35 end ************
2021-01-03 17:55:58,926 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:56:14,787 P3152 INFO Negative sampling done
2021-01-03 17:56:32,784 P3152 INFO --- Start evaluation ---
2021-01-03 17:56:33,450 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:56:55,464 P3152 INFO [Metrics] Recall(k=20): 0.175428 - Recall(k=50): 0.271084 - NDCG(k=20): 0.147319 - NDCG(k=50): 0.177226 - HitRate(k=20): 0.562764 - HitRate(k=50): 0.697836
2021-01-03 17:56:55,472 P3152 INFO Save best model: monitor(max): 0.175428
2021-01-03 17:56:55,484 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:56:55,539 P3152 INFO Train loss: 3.240731
2021-01-03 17:56:55,539 P3152 INFO ************ Epoch=36 end ************
2021-01-03 17:56:55,540 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:57:08,326 P3152 INFO Negative sampling done
2021-01-03 17:57:25,581 P3152 INFO --- Start evaluation ---
2021-01-03 17:57:26,359 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:57:48,896 P3152 INFO [Metrics] Recall(k=20): 0.175436 - Recall(k=50): 0.271103 - NDCG(k=20): 0.147322 - NDCG(k=50): 0.177231 - HitRate(k=20): 0.562797 - HitRate(k=50): 0.697836
2021-01-03 17:57:48,906 P3152 INFO Save best model: monitor(max): 0.175436
2021-01-03 17:57:48,919 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:57:48,976 P3152 INFO Train loss: 3.240630
2021-01-03 17:57:48,977 P3152 INFO ************ Epoch=37 end ************
2021-01-03 17:57:48,977 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:58:01,906 P3152 INFO Negative sampling done
2021-01-03 17:58:20,065 P3152 INFO --- Start evaluation ---
2021-01-03 17:58:20,731 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:58:43,295 P3152 INFO [Metrics] Recall(k=20): 0.175410 - Recall(k=50): 0.271062 - NDCG(k=20): 0.147303 - NDCG(k=50): 0.177210 - HitRate(k=20): 0.562764 - HitRate(k=50): 0.697836
2021-01-03 17:58:43,304 P3152 INFO Monitor(max) STOP: 0.175410 !
2021-01-03 17:58:43,304 P3152 INFO Reduce learning rate on plateau: 0.000001
2021-01-03 17:58:43,304 P3152 INFO Load best model:  YouTubeNet_gowalla_x0_001_10a47ae1_model.ckpt
2021-01-03 17:58:43,311 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:58:43,372 P3152 INFO Train loss: 3.241339
2021-01-03 17:58:43,372 P3152 INFO ************ Epoch=38 end ************
2021-01-03 17:58:43,372 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:58:56,458 P3152 INFO Negative sampling done
2021-01-03 17:59:18,064 P3152 INFO --- Start evaluation ---
2021-01-03 17:59:18,711 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 17:59:40,995 P3152 INFO [Metrics] Recall(k=20): 0.175415 - Recall(k=50): 0.271090 - NDCG(k=20): 0.147304 - NDCG(k=50): 0.177215 - HitRate(k=20): 0.562730 - HitRate(k=50): 0.697803
2021-01-03 17:59:41,005 P3152 INFO Monitor(max) STOP: 0.175415 !
2021-01-03 17:59:41,005 P3152 INFO Reduce learning rate on plateau: 0.000001
2021-01-03 17:59:41,005 P3152 INFO Load best model:  YouTubeNet_gowalla_x0_001_10a47ae1_model.ckpt
2021-01-03 17:59:41,012 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 17:59:41,056 P3152 INFO Train loss: 3.240834
2021-01-03 17:59:41,056 P3152 INFO ************ Epoch=39 end ************
2021-01-03 17:59:41,056 P3152 INFO Negative sampling num_negs=800
2021-01-03 17:59:53,887 P3152 INFO Negative sampling done
2021-01-03 18:00:10,867 P3152 INFO --- Start evaluation ---
2021-01-03 18:00:11,505 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 18:00:33,637 P3152 INFO [Metrics] Recall(k=20): 0.175422 - Recall(k=50): 0.271120 - NDCG(k=20): 0.147310 - NDCG(k=50): 0.177227 - HitRate(k=20): 0.562764 - HitRate(k=50): 0.697870
2021-01-03 18:00:33,645 P3152 INFO Monitor(max) STOP: 0.175422 !
2021-01-03 18:00:33,646 P3152 INFO Reduce learning rate on plateau: 0.000001
2021-01-03 18:00:33,646 P3152 INFO Load best model:  YouTubeNet_gowalla_x0_001_10a47ae1_model.ckpt
2021-01-03 18:00:33,652 P3152 INFO Early stopping at epoch=40
2021-01-03 18:00:33,652 P3152 INFO --- 1583/1583 batches finished ---
2021-01-03 18:00:33,702 P3152 INFO Train loss: 3.240180
2021-01-03 18:00:33,702 P3152 INFO Training finished.
2021-01-03 18:00:33,702 P3152 INFO Load best model:  YouTubeNet_gowalla_x0_001_10a47ae1_model.ckpt
2021-01-03 18:00:33,711 P3152 INFO ****** Train/validation evaluation ******
2021-01-03 18:00:33,712 P3152 INFO --- Start evaluation ---
2021-01-03 18:00:34,340 P3152 INFO Evaluating metrics for 29858 users...
2021-01-03 18:00:56,225 P3152 INFO [Metrics] Recall(k=20): 0.175436 - Recall(k=50): 0.271103 - NDCG(k=20): 0.147322 - NDCG(k=50): 0.177231 - HitRate(k=20): 0.562797 - HitRate(k=50): 0.697836

```