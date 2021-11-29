## YoutubeNet_amazonbooks_x0 

A notebook to benchmark YoutubeNet on amazonbooks_x0 dataset.

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
2021-01-03 19:35:34,704 P2834 INFO [Metrics] Recall(k=20): 0.050157 - Recall(k=50): 0.092365 - NDCG(k=20): 0.038842 - NDCG(k=50): 0.054534 - HitRate(k=20): 0.275708 - HitRate(k=50): 0.435419
```


### Logs
```
2021-01-03 17:20:33,125 P2834 INFO Set up feature encoder...
2021-01-03 17:20:33,125 P2834 INFO Reading file: ../data/AmazonBooks/amazonbooks_x0/train.csv
2021-01-03 17:20:58,250 P2834 INFO Reading file: ../data/AmazonBooks/amazonbooks_x0/item_corpus.csv
2021-01-03 17:20:58,282 P2834 INFO Preprocess feature columns...
2021-01-03 17:20:58,380 P2834 INFO Preprocess feature columns...
2021-01-03 17:21:00,995 P2834 INFO Fit feature encoder...
2021-01-03 17:21:01,399 P2834 INFO Processing column: {'active': True, 'dtype': 'int', 'name': 'query_index', 'type': 'index'}
2021-01-03 17:21:01,399 P2834 INFO Processing column: {'active': True, 'dtype': 'int', 'name': 'corpus_index', 'type': 'index'}
2021-01-03 17:21:01,399 P2834 INFO Processing column: {'active': True, 'dtype': 'str', 'embedding_callback': 'layers.MaskedAveragePooling()', 'max_len': 500, 'name': 'user_history', 'padding': 'pre', 'source': 'user', 'splitter': '^', 'type': 'sequence'}
2021-01-03 17:25:39,579 P2834 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'item_id', 'share_embedding': 'user_history', 'source': 'item', 'type': 'categorical'}
2021-01-03 17:25:39,580 P2834 INFO Pickle feature_encode: ../data/AmazonBooks/amazonbooks_x0_53bbcdfe/feature_encoder.pkl
2021-01-03 17:25:39,603 P2834 INFO Save feature_map to json: ../data/AmazonBooks/amazonbooks_x0_53bbcdfe/feature_map.json
2021-01-03 17:25:39,603 P2834 INFO Set feature encoder done.
2021-01-03 17:25:39,971 P2834 INFO Transform feature columns...
2021-01-03 17:25:40,018 P2834 INFO Saving data to h5: ../data/AmazonBooks/amazonbooks_x0_53bbcdfe/item_corpus.h5
2021-01-03 17:25:40,074 P2834 INFO Transform feature columns...
2021-01-03 17:29:08,525 P2834 INFO Saving data to h5: ../data/AmazonBooks/amazonbooks_x0_53bbcdfe/train.h5
2021-01-03 17:29:12,069 P2834 INFO Reading file: ../data/AmazonBooks/amazonbooks_x0/test.csv
2021-01-03 17:29:15,734 P2834 INFO Preprocess feature columns...
2021-01-03 17:29:16,247 P2834 INFO Transform feature columns...
2021-01-03 17:30:05,223 P2834 INFO Saving data to h5: ../data/AmazonBooks/amazonbooks_x0_53bbcdfe/valid.h5
2021-01-03 17:30:06,010 P2834 INFO Transform csv data to h5 done.
2021-01-03 17:30:08,631 P2834 INFO Total number of parameters: 5862464.
2021-01-03 17:30:08,631 P2834 INFO Loading data...
2021-01-03 17:30:08,634 P2834 INFO Loading data from h5: ../data/AmazonBooks/amazonbooks_x0_53bbcdfe/train.h5
2021-01-03 17:30:10,724 P2834 INFO Loading data from h5: ../data/AmazonBooks/amazonbooks_x0_53bbcdfe/item_corpus.h5
2021-01-03 17:30:11,628 P2834 INFO Loading data from h5: ../data/AmazonBooks/amazonbooks_x0_53bbcdfe/valid.h5
2021-01-03 17:30:12,552 P2834 INFO Loading data from h5: ../data/AmazonBooks/amazonbooks_x0_53bbcdfe/item_corpus.h5
2021-01-03 17:30:12,557 P2834 INFO Train samples: total/2380730, blocks/1
2021-01-03 17:30:12,557 P2834 INFO Validation samples: total/52639, blocks/1
2021-01-03 17:30:12,557 P2834 INFO Loading train data done.
2021-01-03 17:30:12,557 P2834 INFO **** Start training: 2325 batches/epoch ****
2021-01-03 17:30:12,559 P2834 INFO Negative sampling num_negs=1000
2021-01-03 17:30:54,365 P2834 INFO Negative sampling done
2021-01-03 17:31:49,887 P2834 INFO --- Start evaluation ---
2021-01-03 17:31:51,462 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 17:33:09,005 P2834 INFO [Metrics] Recall(k=20): 0.017721 - Recall(k=50): 0.035320 - NDCG(k=20): 0.014086 - NDCG(k=50): 0.020575 - HitRate(k=20): 0.125496 - HitRate(k=50): 0.221224
2021-01-03 17:33:09,026 P2834 INFO Save best model: monitor(max): 0.017721
2021-01-03 17:33:09,053 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 17:33:09,182 P2834 INFO Train loss: 5.726095
2021-01-03 17:33:09,182 P2834 INFO ************ Epoch=1 end ************
2021-01-03 17:33:09,183 P2834 INFO Negative sampling num_negs=1000
2021-01-03 17:33:51,547 P2834 INFO Negative sampling done
2021-01-03 17:34:48,445 P2834 INFO --- Start evaluation ---
2021-01-03 17:34:50,047 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 17:36:04,005 P2834 INFO [Metrics] Recall(k=20): 0.023980 - Recall(k=50): 0.047744 - NDCG(k=20): 0.018867 - NDCG(k=50): 0.027673 - HitRate(k=20): 0.161249 - HitRate(k=50): 0.280476
2021-01-03 17:36:04,021 P2834 INFO Save best model: monitor(max): 0.023980
2021-01-03 17:36:04,056 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 17:36:04,199 P2834 INFO Train loss: 5.029850
2021-01-03 17:36:04,199 P2834 INFO ************ Epoch=2 end ************
2021-01-03 17:36:04,200 P2834 INFO Negative sampling num_negs=1000
2021-01-03 17:36:46,427 P2834 INFO Negative sampling done
2021-01-03 17:37:43,181 P2834 INFO --- Start evaluation ---
2021-01-03 17:37:44,774 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 17:39:04,660 P2834 INFO [Metrics] Recall(k=20): 0.028017 - Recall(k=50): 0.055336 - NDCG(k=20): 0.022157 - NDCG(k=50): 0.032294 - HitRate(k=20): 0.180836 - HitRate(k=50): 0.308991
2021-01-03 17:39:04,676 P2834 INFO Save best model: monitor(max): 0.028017
2021-01-03 17:39:04,711 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 17:39:04,878 P2834 INFO Train loss: 4.708134
2021-01-03 17:39:04,879 P2834 INFO ************ Epoch=3 end ************
2021-01-03 17:39:04,879 P2834 INFO Negative sampling num_negs=1000
2021-01-03 17:39:46,114 P2834 INFO Negative sampling done
2021-01-03 17:40:44,172 P2834 INFO --- Start evaluation ---
2021-01-03 17:40:46,286 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 17:42:02,693 P2834 INFO [Metrics] Recall(k=20): 0.030845 - Recall(k=50): 0.060337 - NDCG(k=20): 0.024205 - NDCG(k=50): 0.035150 - HitRate(k=20): 0.193887 - HitRate(k=50): 0.328369
2021-01-03 17:42:02,709 P2834 INFO Save best model: monitor(max): 0.030845
2021-01-03 17:42:02,743 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 17:42:02,900 P2834 INFO Train loss: 4.517374
2021-01-03 17:42:02,900 P2834 INFO ************ Epoch=4 end ************
2021-01-03 17:42:02,901 P2834 INFO Negative sampling num_negs=1000
2021-01-03 17:42:45,428 P2834 INFO Negative sampling done
2021-01-03 17:43:42,921 P2834 INFO --- Start evaluation ---
2021-01-03 17:43:44,432 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 17:44:58,596 P2834 INFO [Metrics] Recall(k=20): 0.033072 - Recall(k=50): 0.064244 - NDCG(k=20): 0.025974 - NDCG(k=50): 0.037552 - HitRate(k=20): 0.204772 - HitRate(k=50): 0.341591
2021-01-03 17:44:58,611 P2834 INFO Save best model: monitor(max): 0.033072
2021-01-03 17:44:58,647 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 17:44:58,792 P2834 INFO Train loss: 4.388779
2021-01-03 17:44:58,792 P2834 INFO ************ Epoch=5 end ************
2021-01-03 17:44:58,792 P2834 INFO Negative sampling num_negs=1000
2021-01-03 17:45:42,280 P2834 INFO Negative sampling done
2021-01-03 17:46:39,499 P2834 INFO --- Start evaluation ---
2021-01-03 17:46:41,067 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 17:47:59,857 P2834 INFO [Metrics] Recall(k=20): 0.034958 - Recall(k=50): 0.067727 - NDCG(k=20): 0.027254 - NDCG(k=50): 0.039416 - HitRate(k=20): 0.213226 - HitRate(k=50): 0.353844
2021-01-03 17:47:59,877 P2834 INFO Save best model: monitor(max): 0.034958
2021-01-03 17:47:59,912 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 17:48:00,080 P2834 INFO Train loss: 4.292842
2021-01-03 17:48:00,080 P2834 INFO ************ Epoch=6 end ************
2021-01-03 17:48:00,081 P2834 INFO Negative sampling num_negs=1000
2021-01-03 17:48:41,346 P2834 INFO Negative sampling done
2021-01-03 17:49:39,129 P2834 INFO --- Start evaluation ---
2021-01-03 17:49:40,797 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 17:50:58,223 P2834 INFO [Metrics] Recall(k=20): 0.036347 - Recall(k=50): 0.070324 - NDCG(k=20): 0.028282 - NDCG(k=50): 0.040905 - HitRate(k=20): 0.219229 - HitRate(k=50): 0.362279
2021-01-03 17:50:58,238 P2834 INFO Save best model: monitor(max): 0.036347
2021-01-03 17:50:58,272 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 17:50:58,410 P2834 INFO Train loss: 4.216810
2021-01-03 17:50:58,410 P2834 INFO ************ Epoch=7 end ************
2021-01-03 17:50:58,411 P2834 INFO Negative sampling num_negs=1000
2021-01-03 17:51:40,498 P2834 INFO Negative sampling done
2021-01-03 17:52:39,509 P2834 INFO --- Start evaluation ---
2021-01-03 17:52:41,157 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 17:54:01,294 P2834 INFO [Metrics] Recall(k=20): 0.037620 - Recall(k=50): 0.073127 - NDCG(k=20): 0.029125 - NDCG(k=50): 0.042307 - HitRate(k=20): 0.224263 - HitRate(k=50): 0.369764
2021-01-03 17:54:01,311 P2834 INFO Save best model: monitor(max): 0.037620
2021-01-03 17:54:01,346 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 17:54:01,496 P2834 INFO Train loss: 4.153558
2021-01-03 17:54:01,496 P2834 INFO ************ Epoch=8 end ************
2021-01-03 17:54:01,497 P2834 INFO Negative sampling num_negs=1000
2021-01-03 17:54:42,079 P2834 INFO Negative sampling done
2021-01-03 17:55:38,435 P2834 INFO --- Start evaluation ---
2021-01-03 17:55:39,974 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 17:56:58,974 P2834 INFO [Metrics] Recall(k=20): 0.038829 - Recall(k=50): 0.075336 - NDCG(k=20): 0.030178 - NDCG(k=50): 0.043726 - HitRate(k=20): 0.229488 - HitRate(k=50): 0.377078
2021-01-03 17:56:58,990 P2834 INFO Save best model: monitor(max): 0.038829
2021-01-03 17:56:59,024 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 17:56:59,159 P2834 INFO Train loss: 4.100913
2021-01-03 17:56:59,159 P2834 INFO ************ Epoch=9 end ************
2021-01-03 17:56:59,160 P2834 INFO Negative sampling num_negs=1000
2021-01-03 17:57:40,111 P2834 INFO Negative sampling done
2021-01-03 17:58:36,335 P2834 INFO --- Start evaluation ---
2021-01-03 17:58:38,110 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 17:59:54,699 P2834 INFO [Metrics] Recall(k=20): 0.040105 - Recall(k=50): 0.077258 - NDCG(k=20): 0.031150 - NDCG(k=50): 0.044935 - HitRate(k=20): 0.233971 - HitRate(k=50): 0.384069
2021-01-03 17:59:54,715 P2834 INFO Save best model: monitor(max): 0.040105
2021-01-03 17:59:54,752 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 17:59:54,902 P2834 INFO Train loss: 4.055512
2021-01-03 17:59:54,902 P2834 INFO ************ Epoch=10 end ************
2021-01-03 17:59:54,903 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:00:35,990 P2834 INFO Negative sampling done
2021-01-03 18:01:32,107 P2834 INFO --- Start evaluation ---
2021-01-03 18:01:33,784 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:02:48,575 P2834 INFO [Metrics] Recall(k=20): 0.041084 - Recall(k=50): 0.079077 - NDCG(k=20): 0.032095 - NDCG(k=50): 0.046183 - HitRate(k=20): 0.238397 - HitRate(k=50): 0.389920
2021-01-03 18:02:48,591 P2834 INFO Save best model: monitor(max): 0.041084
2021-01-03 18:02:48,626 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:02:48,797 P2834 INFO Train loss: 4.015890
2021-01-03 18:02:48,798 P2834 INFO ************ Epoch=11 end ************
2021-01-03 18:02:48,798 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:03:28,692 P2834 INFO Negative sampling done
2021-01-03 18:04:22,590 P2834 INFO --- Start evaluation ---
2021-01-03 18:04:24,325 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:05:39,022 P2834 INFO [Metrics] Recall(k=20): 0.041927 - Recall(k=50): 0.080238 - NDCG(k=20): 0.032659 - NDCG(k=50): 0.046867 - HitRate(k=20): 0.242121 - HitRate(k=50): 0.394175
2021-01-03 18:05:39,037 P2834 INFO Save best model: monitor(max): 0.041927
2021-01-03 18:05:39,073 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:05:39,236 P2834 INFO Train loss: 3.980365
2021-01-03 18:05:39,236 P2834 INFO ************ Epoch=12 end ************
2021-01-03 18:05:39,237 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:06:18,752 P2834 INFO Negative sampling done
2021-01-03 18:07:13,726 P2834 INFO --- Start evaluation ---
2021-01-03 18:07:15,335 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:08:33,316 P2834 INFO [Metrics] Recall(k=20): 0.042284 - Recall(k=50): 0.081518 - NDCG(k=20): 0.032928 - NDCG(k=50): 0.047476 - HitRate(k=20): 0.243394 - HitRate(k=50): 0.398545
2021-01-03 18:08:33,333 P2834 INFO Save best model: monitor(max): 0.042284
2021-01-03 18:08:33,369 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:08:33,528 P2834 INFO Train loss: 3.949085
2021-01-03 18:08:33,528 P2834 INFO ************ Epoch=13 end ************
2021-01-03 18:08:33,529 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:09:13,060 P2834 INFO Negative sampling done
2021-01-03 18:10:07,939 P2834 INFO --- Start evaluation ---
2021-01-03 18:10:09,529 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:11:25,139 P2834 INFO [Metrics] Recall(k=20): 0.043360 - Recall(k=50): 0.082719 - NDCG(k=20): 0.033785 - NDCG(k=50): 0.048375 - HitRate(k=20): 0.248181 - HitRate(k=50): 0.402154
2021-01-03 18:11:25,154 P2834 INFO Save best model: monitor(max): 0.043360
2021-01-03 18:11:25,187 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:11:25,337 P2834 INFO Train loss: 3.919732
2021-01-03 18:11:25,337 P2834 INFO ************ Epoch=14 end ************
2021-01-03 18:11:25,337 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:12:04,865 P2834 INFO Negative sampling done
2021-01-03 18:12:59,376 P2834 INFO --- Start evaluation ---
2021-01-03 18:13:00,935 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:14:14,684 P2834 INFO [Metrics] Recall(k=20): 0.043900 - Recall(k=50): 0.083913 - NDCG(k=20): 0.034283 - NDCG(k=50): 0.049153 - HitRate(k=20): 0.250195 - HitRate(k=50): 0.406391
2021-01-03 18:14:14,699 P2834 INFO Save best model: monitor(max): 0.043900
2021-01-03 18:14:14,731 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:14:14,928 P2834 INFO Train loss: 3.893706
2021-01-03 18:14:14,928 P2834 INFO ************ Epoch=15 end ************
2021-01-03 18:14:14,928 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:14:54,457 P2834 INFO Negative sampling done
2021-01-03 18:15:48,665 P2834 INFO --- Start evaluation ---
2021-01-03 18:15:50,233 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:17:05,174 P2834 INFO [Metrics] Recall(k=20): 0.044503 - Recall(k=50): 0.084946 - NDCG(k=20): 0.034641 - NDCG(k=50): 0.049696 - HitRate(k=20): 0.252094 - HitRate(k=50): 0.410494
2021-01-03 18:17:05,191 P2834 INFO Save best model: monitor(max): 0.044503
2021-01-03 18:17:05,224 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:17:05,373 P2834 INFO Train loss: 3.868665
2021-01-03 18:17:05,374 P2834 INFO ************ Epoch=16 end ************
2021-01-03 18:17:05,374 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:17:44,958 P2834 INFO Negative sampling done
2021-01-03 18:18:39,794 P2834 INFO --- Start evaluation ---
2021-01-03 18:18:41,361 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:19:55,892 P2834 INFO [Metrics] Recall(k=20): 0.045492 - Recall(k=50): 0.086136 - NDCG(k=20): 0.035346 - NDCG(k=50): 0.050451 - HitRate(k=20): 0.256540 - HitRate(k=50): 0.413116
2021-01-03 18:19:55,907 P2834 INFO Save best model: monitor(max): 0.045492
2021-01-03 18:19:55,939 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:19:56,080 P2834 INFO Train loss: 3.846081
2021-01-03 18:19:56,081 P2834 INFO ************ Epoch=17 end ************
2021-01-03 18:19:56,081 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:20:35,602 P2834 INFO Negative sampling done
2021-01-03 18:21:30,205 P2834 INFO --- Start evaluation ---
2021-01-03 18:21:31,742 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:22:51,233 P2834 INFO [Metrics] Recall(k=20): 0.046035 - Recall(k=50): 0.087106 - NDCG(k=20): 0.035675 - NDCG(k=50): 0.050905 - HitRate(k=20): 0.259446 - HitRate(k=50): 0.418055
2021-01-03 18:22:51,249 P2834 INFO Save best model: monitor(max): 0.046035
2021-01-03 18:22:51,275 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:22:51,410 P2834 INFO Train loss: 3.824226
2021-01-03 18:22:51,411 P2834 INFO ************ Epoch=18 end ************
2021-01-03 18:22:51,411 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:23:30,935 P2834 INFO Negative sampling done
2021-01-03 18:24:25,283 P2834 INFO --- Start evaluation ---
2021-01-03 18:24:26,826 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:25:41,875 P2834 INFO [Metrics] Recall(k=20): 0.046152 - Recall(k=50): 0.087349 - NDCG(k=20): 0.035980 - NDCG(k=50): 0.051280 - HitRate(k=20): 0.260244 - HitRate(k=50): 0.418340
2021-01-03 18:25:41,890 P2834 INFO Save best model: monitor(max): 0.046152
2021-01-03 18:25:41,916 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:25:42,053 P2834 INFO Train loss: 3.803511
2021-01-03 18:25:42,053 P2834 INFO ************ Epoch=19 end ************
2021-01-03 18:25:42,054 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:26:22,145 P2834 INFO Negative sampling done
2021-01-03 18:27:16,336 P2834 INFO --- Start evaluation ---
2021-01-03 18:27:17,913 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:28:32,553 P2834 INFO [Metrics] Recall(k=20): 0.046589 - Recall(k=50): 0.088290 - NDCG(k=20): 0.036276 - NDCG(k=50): 0.051771 - HitRate(k=20): 0.261441 - HitRate(k=50): 0.420867
2021-01-03 18:28:32,569 P2834 INFO Save best model: monitor(max): 0.046589
2021-01-03 18:28:32,595 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:28:32,753 P2834 INFO Train loss: 3.783868
2021-01-03 18:28:32,753 P2834 INFO ************ Epoch=20 end ************
2021-01-03 18:28:32,753 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:29:12,344 P2834 INFO Negative sampling done
2021-01-03 18:30:07,063 P2834 INFO --- Start evaluation ---
2021-01-03 18:30:08,597 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:31:23,621 P2834 INFO [Metrics] Recall(k=20): 0.047139 - Recall(k=50): 0.088754 - NDCG(k=20): 0.036647 - NDCG(k=50): 0.052105 - HitRate(k=20): 0.264272 - HitRate(k=50): 0.423374
2021-01-03 18:31:23,637 P2834 INFO Save best model: monitor(max): 0.047139
2021-01-03 18:31:23,663 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:31:23,814 P2834 INFO Train loss: 3.765492
2021-01-03 18:31:23,814 P2834 INFO ************ Epoch=21 end ************
2021-01-03 18:31:23,815 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:32:03,326 P2834 INFO Negative sampling done
2021-01-03 18:32:57,559 P2834 INFO --- Start evaluation ---
2021-01-03 18:32:59,224 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:34:13,612 P2834 INFO [Metrics] Recall(k=20): 0.047448 - Recall(k=50): 0.089240 - NDCG(k=20): 0.036717 - NDCG(k=50): 0.052264 - HitRate(k=20): 0.264500 - HitRate(k=50): 0.424666
2021-01-03 18:34:13,628 P2834 INFO Save best model: monitor(max): 0.047448
2021-01-03 18:34:13,655 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:34:13,829 P2834 INFO Train loss: 3.747638
2021-01-03 18:34:13,829 P2834 INFO ************ Epoch=22 end ************
2021-01-03 18:34:13,830 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:34:53,494 P2834 INFO Negative sampling done
2021-01-03 18:35:47,948 P2834 INFO --- Start evaluation ---
2021-01-03 18:35:49,481 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:37:06,481 P2834 INFO [Metrics] Recall(k=20): 0.047578 - Recall(k=50): 0.090053 - NDCG(k=20): 0.037047 - NDCG(k=50): 0.052838 - HitRate(k=20): 0.266228 - HitRate(k=50): 0.426281
2021-01-03 18:37:06,496 P2834 INFO Save best model: monitor(max): 0.047578
2021-01-03 18:37:06,523 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:37:06,664 P2834 INFO Train loss: 3.731755
2021-01-03 18:37:06,665 P2834 INFO ************ Epoch=23 end ************
2021-01-03 18:37:06,665 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:37:46,224 P2834 INFO Negative sampling done
2021-01-03 18:38:40,787 P2834 INFO --- Start evaluation ---
2021-01-03 18:38:42,364 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:39:58,343 P2834 INFO [Metrics] Recall(k=20): 0.047959 - Recall(k=50): 0.090132 - NDCG(k=20): 0.037370 - NDCG(k=50): 0.053043 - HitRate(k=20): 0.266798 - HitRate(k=50): 0.427060
2021-01-03 18:39:58,358 P2834 INFO Save best model: monitor(max): 0.047959
2021-01-03 18:39:58,384 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:39:58,539 P2834 INFO Train loss: 3.715694
2021-01-03 18:39:58,539 P2834 INFO ************ Epoch=24 end ************
2021-01-03 18:39:58,540 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:40:38,118 P2834 INFO Negative sampling done
2021-01-03 18:41:32,345 P2834 INFO --- Start evaluation ---
2021-01-03 18:41:33,880 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:42:48,816 P2834 INFO [Metrics] Recall(k=20): 0.048477 - Recall(k=50): 0.090376 - NDCG(k=20): 0.037601 - NDCG(k=50): 0.053178 - HitRate(k=20): 0.268869 - HitRate(k=50): 0.428580
2021-01-03 18:42:48,832 P2834 INFO Save best model: monitor(max): 0.048477
2021-01-03 18:42:48,859 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:42:49,007 P2834 INFO Train loss: 3.700593
2021-01-03 18:42:49,008 P2834 INFO ************ Epoch=25 end ************
2021-01-03 18:42:49,008 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:43:28,524 P2834 INFO Negative sampling done
2021-01-03 18:44:22,531 P2834 INFO --- Start evaluation ---
2021-01-03 18:44:24,060 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:45:40,063 P2834 INFO [Metrics] Recall(k=20): 0.048623 - Recall(k=50): 0.090848 - NDCG(k=20): 0.037834 - NDCG(k=50): 0.053558 - HitRate(k=20): 0.267976 - HitRate(k=50): 0.429757
2021-01-03 18:45:40,079 P2834 INFO Save best model: monitor(max): 0.048623
2021-01-03 18:45:40,104 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:45:40,264 P2834 INFO Train loss: 3.686433
2021-01-03 18:45:40,264 P2834 INFO ************ Epoch=26 end ************
2021-01-03 18:45:40,265 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:46:19,741 P2834 INFO Negative sampling done
2021-01-03 18:47:14,430 P2834 INFO --- Start evaluation ---
2021-01-03 18:47:15,969 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:48:30,877 P2834 INFO [Metrics] Recall(k=20): 0.048969 - Recall(k=50): 0.091549 - NDCG(k=20): 0.037979 - NDCG(k=50): 0.053804 - HitRate(k=20): 0.271567 - HitRate(k=50): 0.432569
2021-01-03 18:48:30,892 P2834 INFO Save best model: monitor(max): 0.048969
2021-01-03 18:48:30,920 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:48:31,069 P2834 INFO Train loss: 3.672492
2021-01-03 18:48:31,070 P2834 INFO ************ Epoch=27 end ************
2021-01-03 18:48:31,070 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:49:10,565 P2834 INFO Negative sampling done
2021-01-03 18:50:04,619 P2834 INFO --- Start evaluation ---
2021-01-03 18:50:06,166 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:51:20,776 P2834 INFO [Metrics] Recall(k=20): 0.048994 - Recall(k=50): 0.091071 - NDCG(k=20): 0.038158 - NDCG(k=50): 0.053791 - HitRate(k=20): 0.270446 - HitRate(k=50): 0.431372
2021-01-03 18:51:20,792 P2834 INFO Save best model: monitor(max): 0.048994
2021-01-03 18:51:20,818 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:51:20,957 P2834 INFO Train loss: 3.659637
2021-01-03 18:51:20,957 P2834 INFO ************ Epoch=28 end ************
2021-01-03 18:51:20,958 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:52:00,497 P2834 INFO Negative sampling done
2021-01-03 18:52:54,969 P2834 INFO --- Start evaluation ---
2021-01-03 18:52:56,525 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:54:11,911 P2834 INFO [Metrics] Recall(k=20): 0.049142 - Recall(k=50): 0.091493 - NDCG(k=20): 0.038198 - NDCG(k=50): 0.053966 - HitRate(k=20): 0.271244 - HitRate(k=50): 0.432721
2021-01-03 18:54:11,925 P2834 INFO Save best model: monitor(max): 0.049142
2021-01-03 18:54:11,951 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:54:12,087 P2834 INFO Train loss: 3.647861
2021-01-03 18:54:12,087 P2834 INFO ************ Epoch=29 end ************
2021-01-03 18:54:12,088 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:54:51,599 P2834 INFO Negative sampling done
2021-01-03 18:55:46,165 P2834 INFO --- Start evaluation ---
2021-01-03 18:55:47,706 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 18:57:07,141 P2834 INFO [Metrics] Recall(k=20): 0.049379 - Recall(k=50): 0.091541 - NDCG(k=20): 0.038389 - NDCG(k=50): 0.054095 - HitRate(k=20): 0.272270 - HitRate(k=50): 0.432968
2021-01-03 18:57:07,156 P2834 INFO Save best model: monitor(max): 0.049379
2021-01-03 18:57:07,182 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 18:57:07,338 P2834 INFO Train loss: 3.636214
2021-01-03 18:57:07,338 P2834 INFO ************ Epoch=30 end ************
2021-01-03 18:57:07,339 P2834 INFO Negative sampling num_negs=1000
2021-01-03 18:57:46,828 P2834 INFO Negative sampling done
2021-01-03 18:58:40,992 P2834 INFO --- Start evaluation ---
2021-01-03 18:58:42,531 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 19:00:00,328 P2834 INFO [Metrics] Recall(k=20): 0.049539 - Recall(k=50): 0.091572 - NDCG(k=20): 0.038539 - NDCG(k=50): 0.054154 - HitRate(k=20): 0.272726 - HitRate(k=50): 0.433576
2021-01-03 19:00:00,344 P2834 INFO Save best model: monitor(max): 0.049539
2021-01-03 19:00:00,370 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 19:00:00,508 P2834 INFO Train loss: 3.625325
2021-01-03 19:00:00,508 P2834 INFO ************ Epoch=31 end ************
2021-01-03 19:00:00,508 P2834 INFO Negative sampling num_negs=1000
2021-01-03 19:00:40,102 P2834 INFO Negative sampling done
2021-01-03 19:01:34,816 P2834 INFO --- Start evaluation ---
2021-01-03 19:01:36,853 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 19:02:52,015 P2834 INFO [Metrics] Recall(k=20): 0.049648 - Recall(k=50): 0.091602 - NDCG(k=20): 0.038604 - NDCG(k=50): 0.054183 - HitRate(k=20): 0.273599 - HitRate(k=50): 0.433424
2021-01-03 19:02:52,031 P2834 INFO Save best model: monitor(max): 0.049648
2021-01-03 19:02:52,057 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 19:02:52,213 P2834 INFO Train loss: 3.615126
2021-01-03 19:02:52,213 P2834 INFO ************ Epoch=32 end ************
2021-01-03 19:02:52,214 P2834 INFO Negative sampling num_negs=1000
2021-01-03 19:03:31,765 P2834 INFO Negative sampling done
2021-01-03 19:04:26,257 P2834 INFO --- Start evaluation ---
2021-01-03 19:04:27,824 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 19:05:42,504 P2834 INFO [Metrics] Recall(k=20): 0.049684 - Recall(k=50): 0.092068 - NDCG(k=20): 0.038598 - NDCG(k=50): 0.054349 - HitRate(k=20): 0.274739 - HitRate(k=50): 0.434678
2021-01-03 19:05:42,519 P2834 INFO Save best model: monitor(max): 0.049684
2021-01-03 19:05:42,545 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 19:05:42,715 P2834 INFO Train loss: 3.605616
2021-01-03 19:05:42,715 P2834 INFO ************ Epoch=33 end ************
2021-01-03 19:05:42,715 P2834 INFO Negative sampling num_negs=1000
2021-01-03 19:06:22,295 P2834 INFO Negative sampling done
2021-01-03 19:07:17,165 P2834 INFO --- Start evaluation ---
2021-01-03 19:07:18,758 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 19:08:34,595 P2834 INFO [Metrics] Recall(k=20): 0.050072 - Recall(k=50): 0.092129 - NDCG(k=20): 0.038759 - NDCG(k=50): 0.054411 - HitRate(k=20): 0.275005 - HitRate(k=50): 0.435058
2021-01-03 19:08:34,610 P2834 INFO Save best model: monitor(max): 0.050072
2021-01-03 19:08:34,636 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 19:08:34,790 P2834 INFO Train loss: 3.596699
2021-01-03 19:08:34,791 P2834 INFO ************ Epoch=34 end ************
2021-01-03 19:08:34,791 P2834 INFO Negative sampling num_negs=1000
2021-01-03 19:09:14,479 P2834 INFO Negative sampling done
2021-01-03 19:10:08,579 P2834 INFO --- Start evaluation ---
2021-01-03 19:10:10,138 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 19:11:26,133 P2834 INFO [Metrics] Recall(k=20): 0.050029 - Recall(k=50): 0.092232 - NDCG(k=20): 0.038864 - NDCG(k=50): 0.054559 - HitRate(k=20): 0.275043 - HitRate(k=50): 0.434735
2021-01-03 19:11:26,156 P2834 INFO Monitor(max) STOP: 0.050029 !
2021-01-03 19:11:26,156 P2834 INFO Reduce learning rate on plateau: 0.000100
2021-01-03 19:11:26,156 P2834 INFO Load best model: YouTubeNet_amazonbooks_x0_001_cf071603_model.ckpt
2021-01-03 19:11:26,172 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 19:11:26,315 P2834 INFO Train loss: 3.587060
2021-01-03 19:11:26,315 P2834 INFO ************ Epoch=35 end ************
2021-01-03 19:11:26,316 P2834 INFO Negative sampling num_negs=1000
2021-01-03 19:12:05,781 P2834 INFO Negative sampling done
2021-01-03 19:13:00,076 P2834 INFO --- Start evaluation ---
2021-01-03 19:13:01,616 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 19:14:16,625 P2834 INFO [Metrics] Recall(k=20): 0.050099 - Recall(k=50): 0.092283 - NDCG(k=20): 0.038779 - NDCG(k=50): 0.054464 - HitRate(k=20): 0.275613 - HitRate(k=50): 0.435647
2021-01-03 19:14:16,640 P2834 INFO Save best model: monitor(max): 0.050099
2021-01-03 19:14:16,666 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 19:14:16,840 P2834 INFO Train loss: 3.538490
2021-01-03 19:14:16,840 P2834 INFO ************ Epoch=36 end ************
2021-01-03 19:14:16,841 P2834 INFO Negative sampling num_negs=1000
2021-01-03 19:14:56,549 P2834 INFO Negative sampling done
2021-01-03 19:15:50,764 P2834 INFO --- Start evaluation ---
2021-01-03 19:15:52,309 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 19:17:08,803 P2834 INFO [Metrics] Recall(k=20): 0.050151 - Recall(k=50): 0.092372 - NDCG(k=20): 0.038840 - NDCG(k=50): 0.054535 - HitRate(k=20): 0.275689 - HitRate(k=50): 0.435457
2021-01-03 19:17:08,819 P2834 INFO Save best model: monitor(max): 0.050151
2021-01-03 19:17:08,846 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 19:17:09,004 P2834 INFO Train loss: 3.536734
2021-01-03 19:17:09,005 P2834 INFO ************ Epoch=37 end ************
2021-01-03 19:17:09,005 P2834 INFO Negative sampling num_negs=1000
2021-01-03 19:17:48,493 P2834 INFO Negative sampling done
2021-01-03 19:18:42,573 P2834 INFO --- Start evaluation ---
2021-01-03 19:18:44,115 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 19:19:59,351 P2834 INFO [Metrics] Recall(k=20): 0.050081 - Recall(k=50): 0.092300 - NDCG(k=20): 0.038868 - NDCG(k=50): 0.054559 - HitRate(k=20): 0.275803 - HitRate(k=50): 0.435248
2021-01-03 19:19:59,367 P2834 INFO Monitor(max) STOP: 0.050081 !
2021-01-03 19:19:59,367 P2834 INFO Reduce learning rate on plateau: 0.000010
2021-01-03 19:19:59,367 P2834 INFO Load best model: YouTubeNet_amazonbooks_x0_001_cf071603_model.ckpt
2021-01-03 19:19:59,383 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 19:19:59,539 P2834 INFO Train loss: 3.534499
2021-01-03 19:19:59,539 P2834 INFO ************ Epoch=38 end ************
2021-01-03 19:19:59,540 P2834 INFO Negative sampling num_negs=1000
2021-01-03 19:20:39,015 P2834 INFO Negative sampling done
2021-01-03 19:21:33,254 P2834 INFO --- Start evaluation ---
2021-01-03 19:21:34,784 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 19:22:49,640 P2834 INFO [Metrics] Recall(k=20): 0.050144 - Recall(k=50): 0.092344 - NDCG(k=20): 0.038848 - NDCG(k=50): 0.054535 - HitRate(k=20): 0.275651 - HitRate(k=50): 0.435267
2021-01-03 19:22:49,657 P2834 INFO Monitor(max) STOP: 0.050144 !
2021-01-03 19:22:49,657 P2834 INFO Reduce learning rate on plateau: 0.000001
2021-01-03 19:22:49,657 P2834 INFO Load best model: YouTubeNet_amazonbooks_x0_001_cf071603_model.ckpt
2021-01-03 19:22:49,672 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 19:22:49,846 P2834 INFO Train loss: 3.529141
2021-01-03 19:22:49,846 P2834 INFO ************ Epoch=39 end ************
2021-01-03 19:22:49,846 P2834 INFO Negative sampling num_negs=1000
2021-01-03 19:23:29,690 P2834 INFO Negative sampling done
2021-01-03 19:24:23,503 P2834 INFO --- Start evaluation ---
2021-01-03 19:24:25,093 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 19:25:41,006 P2834 INFO [Metrics] Recall(k=20): 0.050157 - Recall(k=50): 0.092365 - NDCG(k=20): 0.038842 - NDCG(k=50): 0.054534 - HitRate(k=20): 0.275708 - HitRate(k=50): 0.435419
2021-01-03 19:25:41,022 P2834 INFO Save best model: monitor(max): 0.050157
2021-01-03 19:25:41,051 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 19:25:41,208 P2834 INFO Train loss: 3.528417
2021-01-03 19:25:41,208 P2834 INFO ************ Epoch=40 end ************
2021-01-03 19:25:41,209 P2834 INFO Negative sampling num_negs=1000
2021-01-03 19:26:21,386 P2834 INFO Negative sampling done
2021-01-03 19:27:16,613 P2834 INFO --- Start evaluation ---
2021-01-03 19:27:18,196 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 19:28:34,836 P2834 INFO [Metrics] Recall(k=20): 0.050158 - Recall(k=50): 0.092363 - NDCG(k=20): 0.038846 - NDCG(k=50): 0.054536 - HitRate(k=20): 0.275708 - HitRate(k=50): 0.435438
2021-01-03 19:28:34,851 P2834 INFO Monitor(max) STOP: 0.050158 !
2021-01-03 19:28:34,851 P2834 INFO Reduce learning rate on plateau: 0.000001
2021-01-03 19:28:34,851 P2834 INFO Load best model: YouTubeNet_amazonbooks_x0_001_cf071603_model.ckpt
2021-01-03 19:28:34,870 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 19:28:35,044 P2834 INFO Train loss: 3.528497
2021-01-03 19:28:35,044 P2834 INFO ************ Epoch=41 end ************
2021-01-03 19:28:35,044 P2834 INFO Negative sampling num_negs=1000
2021-01-03 19:29:14,534 P2834 INFO Negative sampling done
2021-01-03 19:30:08,523 P2834 INFO --- Start evaluation ---
2021-01-03 19:30:10,062 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 19:31:27,646 P2834 INFO [Metrics] Recall(k=20): 0.050142 - Recall(k=50): 0.092359 - NDCG(k=20): 0.038838 - NDCG(k=50): 0.054533 - HitRate(k=20): 0.275689 - HitRate(k=50): 0.435438
2021-01-03 19:31:27,660 P2834 INFO Monitor(max) STOP: 0.050142 !
2021-01-03 19:31:27,660 P2834 INFO Reduce learning rate on plateau: 0.000001
2021-01-03 19:31:27,661 P2834 INFO Load best model: YouTubeNet_amazonbooks_x0_001_cf071603_model.ckpt
2021-01-03 19:31:27,675 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 19:31:27,812 P2834 INFO Train loss: 3.529272
2021-01-03 19:31:27,812 P2834 INFO ************ Epoch=42 end ************
2021-01-03 19:31:27,812 P2834 INFO Negative sampling num_negs=1000
2021-01-03 19:32:07,456 P2834 INFO Negative sampling done
2021-01-03 19:33:01,226 P2834 INFO --- Start evaluation ---
2021-01-03 19:33:02,745 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 19:34:17,444 P2834 INFO [Metrics] Recall(k=20): 0.050139 - Recall(k=50): 0.092368 - NDCG(k=20): 0.038837 - NDCG(k=50): 0.054537 - HitRate(k=20): 0.275651 - HitRate(k=50): 0.435476
2021-01-03 19:34:17,460 P2834 INFO Monitor(max) STOP: 0.050139 !
2021-01-03 19:34:17,460 P2834 INFO Reduce learning rate on plateau: 0.000001
2021-01-03 19:34:17,460 P2834 INFO Load best model: YouTubeNet_amazonbooks_x0_001_cf071603_model.ckpt
2021-01-03 19:34:17,475 P2834 INFO Early stopping at epoch=43
2021-01-03 19:34:17,475 P2834 INFO --- 2325/2325 batches finished ---
2021-01-03 19:34:17,612 P2834 INFO Train loss: 3.528681
2021-01-03 19:34:17,612 P2834 INFO Training finished.
2021-01-03 19:34:17,612 P2834 INFO Load best model: YouTubeNet_amazonbooks_x0_001_cf071603_model.ckpt
2021-01-03 19:34:17,636 P2834 INFO ****** Train/validation evaluation ******
2021-01-03 19:34:17,636 P2834 INFO --- Start evaluation ---
2021-01-03 19:34:19,180 P2834 INFO Evaluating metrics for 52639 users...
2021-01-03 19:35:34,704 P2834 INFO [Metrics] Recall(k=20): 0.050157 - Recall(k=50): 0.092365 - NDCG(k=20): 0.038842 - NDCG(k=50): 0.054534 - HitRate(k=20): 0.275708 - HitRate(k=50): 0.435419

```