
## ENMF_amazonbooks_x0 

A notebook to benchmark ENMF on amazonbooks_x0 dataset.

Author: Jinpeng Wang, Tsinghua University

Edited by [XUEPAI Team](https://github.com/xue-pai)


### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs) 

### Environments
+ Hardware

    ```python
    CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
    RAM: 500G+
    GPU: Tesla P100, 16G memory
    ```
+ Software

    ```python
    python: 3.6.5
    tensorflow: 1.14.0
    pandas: 1.0.0
    numpy: 1.18.1
    ```

### Dataset
amazonbooks_x0 follows the data split and preprocessing steps in NGCF and LightGCN. We directly transform the formats of the data from their [repo](https://github.com/kuandeng/LightGCN/tree/master/Data).


### Code

The details of the code are described [here](./README.md).

1. Downalod the dataset from LightGCN repo and run the preprocessing script for format transformation.

    ```python
    cd data/AmazonBooks/amazonbooks_x0
    python ENMF_data_process.py
    cd benchmarks/ENMF
    ```

2. Run the following script to reproduce the result.

    ```python
    # Hyper-parameters:
    dataset = "amazonbooks_x0"
    train_data = "../../data/AmazonBooks/amazonbooks_x0/train_enmf.txt"
    test_data = "../../data/AmazonBooks/amazonbooks_x0/test_enmf.txt"
    max_epochs = 500
    batch_size = 256
    embed_size = 64
    lr = 0.05
    dropout = 0.7 # dropout keep_prob
    negative_weight = 0.01 # weight of non-observed data, tuned among [0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5]
    verbose = 1 # Evaluation interval
    topK = "20 50" # Metrics at TopK
    ```
    ```python
    # Run ENMF
    python -u ENMF.py --gpu 0 --dataset amazonbooks_x0 --train_data ../../data/AmazonBooks/amazonbooks_x0/train_enmf.txt --test_data ../../data/AmazonBooks/amazonbooks_x0/test_enmf.txt --verbose 10 --batch_size 256 --epochs 500 --embed_size 64 --lr 0.05 --dropout 0.7 --negative_weight 0.01 --topK 20 50
    ```

### Results
```python
2020-11-06 08:22:54.479162 evaluate at epoch 61
2020-11-06 08:25:30.234817 hitrate@20=0.2187, recall@20=0.0359, ndcg@20=0.0281
2020-11-06 08:25:30.234901 hitrate@50=0.3649, recall@50=0.0691, ndcg@50=0.0404
```

### Logs
```python
tensorflow version: 1.14.0
2020-11-06 05:45:25.768480 loading data ../../data/AmazonBooks/amazonbooks_x0/train_enmf.txt
2020-11-06 05:45:26.082815 loading data ../../data/AmazonBooks/amazonbooks_x0/test_enmf.txt
2020-11-06 05:45:26.167166 finish loading
2020-11-06 05:45:27.227914 #users=52643, #items=91599
2020-11-06 05:45:30.234593 maximum number of items per user is 10682
2020-11-06 05:48:42.581243 begin to construct the graph

2020-11-06 05:49:29.925491 run epoch 1:  time=136.70s
2020-11-06 05:51:46.629545 loss=0.0707, loss_no_reg=0.0707, loss_reg=0.0000

2020-11-06 05:51:46.629707 evaluate at epoch 1
2020-11-06 05:54:23.639922 hitrate@20=0.0243, recall@20=0.0024, ndcg@20=0.0021
2020-11-06 05:54:23.640013 hitrate@50=0.0458, recall@50=0.0048, ndcg@50=0.0030

2020-11-06 05:54:23.641335 run epoch 2:  time=137.18s
2020-11-06 05:56:40.825232 loss=-1379.5341, loss_no_reg=-1379.5341, loss_reg=0.0000

2020-11-06 05:56:40.825283 run epoch 3:  time=134.73s
2020-11-06 05:58:55.555720 loss=-3452.8939, loss_no_reg=-3452.8939, loss_reg=0.0000

2020-11-06 05:58:55.555776 run epoch 4:  time=134.52s
2020-11-06 06:01:10.073358 loss=-4419.9082, loss_no_reg=-4419.9082, loss_reg=0.0000

2020-11-06 06:01:10.073410 run epoch 5:  time=135.10s
2020-11-06 06:03:25.173985 loss=-4977.2329, loss_no_reg=-4977.2329, loss_reg=0.0000

2020-11-06 06:03:25.174050 run epoch 6:  time=135.68s
2020-11-06 06:05:40.856970 loss=-5305.3428, loss_no_reg=-5305.3428, loss_reg=0.0000

2020-11-06 06:05:40.857045 run epoch 7:  time=134.92s
2020-11-06 06:07:55.781267 loss=-5522.2682, loss_no_reg=-5522.2682, loss_reg=0.0000

2020-11-06 06:07:55.781331 run epoch 8:  time=135.10s
2020-11-06 06:10:10.878470 loss=-5668.6487, loss_no_reg=-5668.6487, loss_reg=0.0000

2020-11-06 06:10:10.878604 run epoch 9:  time=134.36s
2020-11-06 06:12:25.243320 loss=-5768.0168, loss_no_reg=-5768.0168, loss_reg=0.0000

2020-11-06 06:12:25.243371 run epoch 10:  time=134.49s
2020-11-06 06:14:39.737612 loss=-5862.4424, loss_no_reg=-5862.4424, loss_reg=0.0000

2020-11-06 06:14:39.737716 run epoch 11:  time=135.59s
2020-11-06 06:16:55.331601 loss=-5935.9186, loss_no_reg=-5935.9186, loss_reg=0.0000

2020-11-06 06:16:55.331679 evaluate at epoch 11
2020-11-06 06:19:34.945003 hitrate@20=0.2079, recall@20=0.0340, ndcg@20=0.0267
2020-11-06 06:19:34.945093 hitrate@50=0.3464, recall@50=0.0649, ndcg@50=0.0382

2020-11-06 06:19:34.946863 run epoch 12:  time=138.43s
2020-11-06 06:21:53.375184 loss=-5987.4873, loss_no_reg=-5987.4873, loss_reg=0.0000

2020-11-06 06:21:53.375261 run epoch 13:  time=132.61s
2020-11-06 06:24:05.981657 loss=-6031.0488, loss_no_reg=-6031.0488, loss_reg=0.0000

2020-11-06 06:24:05.981707 run epoch 14:  time=135.65s
2020-11-06 06:26:21.632337 loss=-6072.2747, loss_no_reg=-6072.2747, loss_reg=0.0000

2020-11-06 06:26:21.632391 run epoch 15:  time=135.24s
2020-11-06 06:28:36.869293 loss=-6112.8540, loss_no_reg=-6112.8540, loss_reg=0.0000

2020-11-06 06:28:36.869343 run epoch 16:  time=134.13s
2020-11-06 06:30:50.998609 loss=-6138.4941, loss_no_reg=-6138.4941, loss_reg=0.0000

2020-11-06 06:30:50.998662 run epoch 17:  time=134.77s
2020-11-06 06:33:05.771387 loss=-6158.8857, loss_no_reg=-6158.8857, loss_reg=0.0000

2020-11-06 06:33:05.771443 run epoch 18:  time=130.48s
2020-11-06 06:35:16.246864 loss=-6182.6639, loss_no_reg=-6182.6639, loss_reg=0.0000

2020-11-06 06:35:16.246975 run epoch 19:  time=135.70s
2020-11-06 06:37:31.947512 loss=-6207.2177, loss_no_reg=-6207.2177, loss_reg=0.0000

2020-11-06 06:37:31.947603 run epoch 20:  time=135.08s
2020-11-06 06:39:47.022957 loss=-6221.0033, loss_no_reg=-6221.0033, loss_reg=0.0000

2020-11-06 06:39:47.023029 run epoch 21:  time=136.10s
2020-11-06 06:42:03.125720 loss=-6239.9632, loss_no_reg=-6239.9632, loss_reg=0.0000

2020-11-06 06:42:03.125854 evaluate at epoch 21
2020-11-06 06:44:38.634923 hitrate@20=0.2153, recall@20=0.0356, ndcg@20=0.0277
2020-11-06 06:44:38.635035 hitrate@50=0.3593, recall@50=0.0682, ndcg@50=0.0399

2020-11-06 06:44:38.637228 run epoch 22:  time=136.84s
2020-11-06 06:46:55.474935 loss=-6259.0759, loss_no_reg=-6259.0759, loss_reg=0.0000

2020-11-06 06:46:55.475041 run epoch 23:  time=133.35s
2020-11-06 06:49:08.821817 loss=-6262.4290, loss_no_reg=-6262.4290, loss_reg=0.0000

2020-11-06 06:49:08.821865 run epoch 24:  time=133.34s
2020-11-06 06:51:22.158832 loss=-6277.4436, loss_no_reg=-6277.4436, loss_reg=0.0000

2020-11-06 06:51:22.158931 run epoch 25:  time=134.71s
2020-11-06 06:53:36.869962 loss=-6298.7278, loss_no_reg=-6298.7278, loss_reg=0.0000

2020-11-06 06:53:36.870010 run epoch 26:  time=133.68s
2020-11-06 06:55:50.552305 loss=-6307.0603, loss_no_reg=-6307.0603, loss_reg=0.0000

2020-11-06 06:55:50.552378 run epoch 27:  time=136.39s
2020-11-06 06:58:06.943242 loss=-6318.5313, loss_no_reg=-6318.5313, loss_reg=0.0000

2020-11-06 06:58:06.943288 run epoch 28:  time=133.18s
2020-11-06 07:00:20.121866 loss=-6315.8156, loss_no_reg=-6315.8156, loss_reg=0.0000

2020-11-06 07:00:20.121931 run epoch 29:  time=133.52s
2020-11-06 07:02:33.646062 loss=-6334.5134, loss_no_reg=-6334.5134, loss_reg=0.0000

2020-11-06 07:02:33.646109 run epoch 30:  time=134.90s
2020-11-06 07:04:48.542346 loss=-6341.2879, loss_no_reg=-6341.2879, loss_reg=0.0000

2020-11-06 07:04:48.542389 run epoch 31:  time=134.62s
2020-11-06 07:07:03.164697 loss=-6352.4307, loss_no_reg=-6352.4307, loss_reg=0.0000

2020-11-06 07:07:03.164751 evaluate at epoch 31
2020-11-06 07:09:38.575530 hitrate@20=0.2173, recall@20=0.0358, ndcg@20=0.0279
2020-11-06 07:09:38.575611 hitrate@50=0.3628, recall@50=0.0690, ndcg@50=0.0402

2020-11-06 07:09:38.576983 run epoch 32:  time=133.17s
2020-11-06 07:11:51.744413 loss=-6354.4063, loss_no_reg=-6354.4063, loss_reg=0.0000

2020-11-06 07:11:51.744473 run epoch 33:  time=134.41s
2020-11-06 07:14:06.159179 loss=-6359.2676, loss_no_reg=-6359.2676, loss_reg=0.0000

2020-11-06 07:14:06.159226 run epoch 34:  time=139.03s
2020-11-06 07:16:25.187014 loss=-6373.3869, loss_no_reg=-6373.3869, loss_reg=0.0000

2020-11-06 07:16:25.187056 run epoch 35:  time=130.23s
2020-11-06 07:18:35.417860 loss=-6379.0455, loss_no_reg=-6379.0455, loss_reg=0.0000

2020-11-06 07:18:35.418497 run epoch 36:  time=141.29s
2020-11-06 07:20:56.703964 loss=-6375.7214, loss_no_reg=-6375.7214, loss_reg=0.0000

2020-11-06 07:20:56.704007 run epoch 37:  time=140.37s
2020-11-06 07:23:17.071560 loss=-6389.7689, loss_no_reg=-6389.7689, loss_reg=0.0000

2020-11-06 07:23:17.071621 run epoch 38:  time=138.37s
2020-11-06 07:25:35.436946 loss=-6397.6665, loss_no_reg=-6397.6665, loss_reg=0.0000

2020-11-06 07:25:35.437011 run epoch 39:  time=138.34s
2020-11-06 07:27:53.777809 loss=-6400.7035, loss_no_reg=-6400.7035, loss_reg=0.0000

2020-11-06 07:27:53.777852 run epoch 40:  time=135.09s
2020-11-06 07:30:08.871187 loss=-6401.2059, loss_no_reg=-6401.2059, loss_reg=0.0000

2020-11-06 07:30:08.871246 run epoch 41:  time=141.26s
2020-11-06 07:32:30.134811 loss=-6404.4845, loss_no_reg=-6404.4845, loss_reg=0.0000

2020-11-06 07:32:30.134874 evaluate at epoch 41
2020-11-06 07:35:15.899779 hitrate@20=0.2171, recall@20=0.0358, ndcg@20=0.0279
2020-11-06 07:35:15.899862 hitrate@50=0.3628, recall@50=0.0690, ndcg@50=0.0403

2020-11-06 07:35:15.901656 run epoch 42:  time=138.49s
2020-11-06 07:37:34.396634 loss=-6408.9375, loss_no_reg=-6408.9375, loss_reg=0.0000

2020-11-06 07:37:34.396687 run epoch 43:  time=137.42s
2020-11-06 07:39:51.818472 loss=-6422.6081, loss_no_reg=-6422.6081, loss_reg=0.0000

2020-11-06 07:39:51.818540 run epoch 44:  time=136.98s
2020-11-06 07:42:08.795291 loss=-6421.7659, loss_no_reg=-6421.7659, loss_reg=0.0000

2020-11-06 07:42:08.795341 run epoch 45:  time=137.17s
2020-11-06 07:44:25.967955 loss=-6420.7278, loss_no_reg=-6420.7278, loss_reg=0.0000

2020-11-06 07:44:25.968007 run epoch 46:  time=133.28s
2020-11-06 07:46:39.248431 loss=-6428.2215, loss_no_reg=-6428.2215, loss_reg=0.0000

2020-11-06 07:46:39.248480 run epoch 47:  time=138.59s
2020-11-06 07:48:57.840374 loss=-6429.2670, loss_no_reg=-6429.2670, loss_reg=0.0000

2020-11-06 07:48:57.840418 run epoch 48:  time=133.25s
2020-11-06 07:51:11.088303 loss=-6431.7048, loss_no_reg=-6431.7048, loss_reg=0.0000

2020-11-06 07:51:11.088348 run epoch 49:  time=134.45s
2020-11-06 07:53:25.538435 loss=-6441.3291, loss_no_reg=-6441.3291, loss_reg=0.0000

2020-11-06 07:53:25.538483 run epoch 50:  time=136.63s
2020-11-06 07:55:42.172608 loss=-6434.3541, loss_no_reg=-6434.3541, loss_reg=0.0000

2020-11-06 07:55:42.172690 run epoch 51:  time=132.39s
2020-11-06 07:57:54.562010 loss=-6436.9094, loss_no_reg=-6436.9094, loss_reg=0.0000

2020-11-06 07:57:54.562058 evaluate at epoch 51
2020-11-06 08:00:40.724914 hitrate@20=0.2174, recall@20=0.0358, ndcg@20=0.0280
2020-11-06 08:00:40.724993 hitrate@50=0.3636, recall@50=0.0689, ndcg@50=0.0404

2020-11-06 08:00:40.727493 run epoch 52:  time=131.82s
2020-11-06 08:02:52.543732 loss=-6439.4969, loss_no_reg=-6439.4969, loss_reg=0.0000

2020-11-06 08:02:52.543781 run epoch 53:  time=131.26s
2020-11-06 08:05:03.802840 loss=-6448.9734, loss_no_reg=-6448.9734, loss_reg=0.0000

2020-11-06 08:05:03.802885 run epoch 54:  time=136.48s
2020-11-06 08:07:20.278563 loss=-6444.8945, loss_no_reg=-6444.8945, loss_reg=0.0000

2020-11-06 08:07:20.278647 run epoch 55:  time=131.15s
2020-11-06 08:09:31.432545 loss=-6450.8291, loss_no_reg=-6450.8291, loss_reg=0.0000

2020-11-06 08:09:31.433162 run epoch 56:  time=134.11s
2020-11-06 08:11:45.540193 loss=-6450.6365, loss_no_reg=-6450.6365, loss_reg=0.0000

2020-11-06 08:11:45.540248 run epoch 57:  time=134.59s
2020-11-06 08:14:00.133203 loss=-6454.3728, loss_no_reg=-6454.3728, loss_reg=0.0000

2020-11-06 08:14:00.133250 run epoch 58:  time=132.08s
2020-11-06 08:16:12.213878 loss=-6446.0437, loss_no_reg=-6446.0437, loss_reg=0.0000

2020-11-06 08:16:12.213923 run epoch 59:  time=137.55s
2020-11-06 08:18:29.762202 loss=-6458.4703, loss_no_reg=-6458.4703, loss_reg=0.0000

2020-11-06 08:18:29.762271 run epoch 60:  time=131.34s
2020-11-06 08:20:41.104872 loss=-6455.2013, loss_no_reg=-6455.2013, loss_reg=0.0000

2020-11-06 08:20:41.104949 run epoch 61:  time=133.37s
2020-11-06 08:22:54.479100 loss=-6471.2458, loss_no_reg=-6471.2458, loss_reg=0.0000

2020-11-06 08:22:54.479162 evaluate at epoch 61
2020-11-06 08:25:30.234817 hitrate@20=0.2187, recall@20=0.0359, ndcg@20=0.0281
2020-11-06 08:25:30.234901 hitrate@50=0.3649, recall@50=0.0691, ndcg@50=0.0404

2020-11-06 08:25:30.236411 run epoch 62:  time=131.53s
2020-11-06 08:27:41.766342 loss=-6471.4592, loss_no_reg=-6471.4592, loss_reg=0.0000

2020-11-06 08:27:41.767009 run epoch 63:  time=139.49s
2020-11-06 08:30:01.259951 loss=-6468.1724, loss_no_reg=-6468.1724, loss_reg=0.0000

2020-11-06 08:30:01.260014 run epoch 64:  time=131.60s
2020-11-06 08:32:12.863298 loss=-6472.2885, loss_no_reg=-6472.2885, loss_reg=0.0000

2020-11-06 08:32:12.863999 run epoch 65:  time=134.60s
2020-11-06 08:34:27.460253 loss=-6465.5716, loss_no_reg=-6465.5716, loss_reg=0.0000

2020-11-06 08:34:27.460300 run epoch 66:  time=134.30s
2020-11-06 08:36:41.759190 loss=-6475.9582, loss_no_reg=-6475.9582, loss_reg=0.0000

2020-11-06 08:36:41.759830 run epoch 67:  time=133.35s
2020-11-06 08:38:55.112298 loss=-6472.4335, loss_no_reg=-6472.4335, loss_reg=0.0000

2020-11-06 08:38:55.112391 run epoch 68:  time=138.27s
2020-11-06 08:41:13.383425 loss=-6481.4930, loss_no_reg=-6481.4930, loss_reg=0.0000

2020-11-06 08:41:13.383511 run epoch 69:  time=131.29s
2020-11-06 08:43:24.676579 loss=-6481.2254, loss_no_reg=-6481.2254, loss_reg=0.0000

2020-11-06 08:43:24.676639 run epoch 70:  time=133.99s
2020-11-06 08:45:38.667961 loss=-6479.0282, loss_no_reg=-6479.0282, loss_reg=0.0000

2020-11-06 08:45:38.668006 run epoch 71:  time=136.83s
2020-11-06 08:47:55.496758 loss=-6480.5271, loss_no_reg=-6480.5271, loss_reg=0.0000

2020-11-06 08:47:55.496803 evaluate at epoch 71
2020-11-06 08:50:31.667662 hitrate@20=0.2169, recall@20=0.0356, ndcg@20=0.0278
2020-11-06 08:50:31.667835 hitrate@50=0.3645, recall@50=0.0692, ndcg@50=0.0403
2020-11-06 08:50:31.667870 the monitor counts down its patience to 4!

2020-11-06 08:50:31.669544 run epoch 72:  time=138.32s
2020-11-06 08:52:49.987448 loss=-6489.9388, loss_no_reg=-6489.9388, loss_reg=0.0000

2020-11-06 08:52:49.987512 run epoch 73:  time=131.20s
2020-11-06 08:55:01.190291 loss=-6486.3060, loss_no_reg=-6486.3060, loss_reg=0.0000

2020-11-06 08:55:01.190926 run epoch 74:  time=130.61s
2020-11-06 08:57:11.797046 loss=-6482.0321, loss_no_reg=-6482.0321, loss_reg=0.0000

2020-11-06 08:57:11.797090 run epoch 75:  time=134.72s
2020-11-06 08:59:26.513163 loss=-6488.7335, loss_no_reg=-6488.7335, loss_reg=0.0000

2020-11-06 08:59:26.513227 run epoch 76:  time=130.75s
2020-11-06 09:01:37.264683 loss=-6494.2336, loss_no_reg=-6494.2336, loss_reg=0.0000

2020-11-06 09:01:37.265314 run epoch 77:  time=136.02s
2020-11-06 09:03:53.289292 loss=-6488.2916, loss_no_reg=-6488.2916, loss_reg=0.0000

2020-11-06 09:03:53.289361 run epoch 78:  time=132.01s
2020-11-06 09:06:05.298062 loss=-6487.9697, loss_no_reg=-6487.9697, loss_reg=0.0000

2020-11-06 09:06:05.298609 run epoch 79:  time=130.44s
2020-11-06 09:08:15.734213 loss=-6497.6587, loss_no_reg=-6497.6587, loss_reg=0.0000

2020-11-06 09:08:15.734259 run epoch 80:  time=129.60s
2020-11-06 09:10:25.338702 loss=-6499.3579, loss_no_reg=-6499.3579, loss_reg=0.0000

2020-11-06 09:10:25.338751 run epoch 81:  time=131.28s
2020-11-06 09:12:36.622645 loss=-6497.8310, loss_no_reg=-6497.8310, loss_reg=0.0000

2020-11-06 09:12:36.623267 evaluate at epoch 81
2020-11-06 09:15:06.478336 hitrate@20=0.2175, recall@20=0.0357, ndcg@20=0.0279
2020-11-06 09:15:06.478415 hitrate@50=0.3652, recall@50=0.0693, ndcg@50=0.0404
2020-11-06 09:15:06.478443 the monitor counts down its patience to 3!

2020-11-06 09:15:06.479886 run epoch 82:  time=130.84s
2020-11-06 09:17:17.320211 loss=-6501.2795, loss_no_reg=-6501.2795, loss_reg=0.0000

2020-11-06 09:17:17.320270 run epoch 83:  time=131.98s
2020-11-06 09:19:29.300965 loss=-6495.3173, loss_no_reg=-6495.3173, loss_reg=0.0000

2020-11-06 09:19:29.301021 run epoch 84:  time=130.86s
2020-11-06 09:21:40.158665 loss=-6500.0196, loss_no_reg=-6500.0196, loss_reg=0.0000

2020-11-06 09:21:40.158715 run epoch 85:  time=134.04s
2020-11-06 09:23:54.196505 loss=-6496.7117, loss_no_reg=-6496.7117, loss_reg=0.0000

2020-11-06 09:23:54.196549 run epoch 86:  time=127.63s
2020-11-06 09:26:01.825740 loss=-6501.6037, loss_no_reg=-6501.6037, loss_reg=0.0000

2020-11-06 09:26:01.825786 run epoch 87:  time=130.31s
2020-11-06 09:28:12.140171 loss=-6505.8674, loss_no_reg=-6505.8674, loss_reg=0.0000

2020-11-06 09:28:12.140223 run epoch 88:  time=131.51s
2020-11-06 09:30:23.647419 loss=-6510.8120, loss_no_reg=-6510.8120, loss_reg=0.0000

2020-11-06 09:30:23.648620 run epoch 89:  time=130.66s
2020-11-06 09:32:34.306667 loss=-6505.2653, loss_no_reg=-6505.2653, loss_reg=0.0000

2020-11-06 09:32:34.306713 run epoch 90:  time=130.65s
2020-11-06 09:34:44.961786 loss=-6513.2303, loss_no_reg=-6513.2303, loss_reg=0.0000

2020-11-06 09:34:44.961832 run epoch 91:  time=130.16s
2020-11-06 09:36:55.123792 loss=-6509.3516, loss_no_reg=-6509.3516, loss_reg=0.0000

2020-11-06 09:36:55.123839 evaluate at epoch 91
2020-11-06 09:39:29.779269 hitrate@20=0.2176, recall@20=0.0357, ndcg@20=0.0279
2020-11-06 09:39:29.779361 hitrate@50=0.3657, recall@50=0.0691, ndcg@50=0.0403
2020-11-06 09:39:29.779393 the monitor counts down its patience to 2!

2020-11-06 09:39:29.780896 run epoch 92:  time=131.00s
2020-11-06 09:41:40.776543 loss=-6511.0829, loss_no_reg=-6511.0829, loss_reg=0.0000

2020-11-06 09:41:40.776606 run epoch 93:  time=131.77s
2020-11-06 09:43:52.546939 loss=-6510.4870, loss_no_reg=-6510.4870, loss_reg=0.0000

2020-11-06 09:43:52.547633 run epoch 94:  time=130.77s
2020-11-06 09:46:03.316620 loss=-6508.7580, loss_no_reg=-6508.7580, loss_reg=0.0000

2020-11-06 09:46:03.316686 run epoch 95:  time=127.72s
2020-11-06 09:48:11.036812 loss=-6509.4072, loss_no_reg=-6509.4072, loss_reg=0.0000

2020-11-06 09:48:11.036857 run epoch 96:  time=130.88s
2020-11-06 09:50:21.914533 loss=-6509.1167, loss_no_reg=-6509.1167, loss_reg=0.0000

2020-11-06 09:50:21.914578 run epoch 97:  time=131.67s
2020-11-06 09:52:33.581672 loss=-6514.0196, loss_no_reg=-6514.0196, loss_reg=0.0000

2020-11-06 09:52:33.581734 run epoch 98:  time=130.28s
2020-11-06 09:54:43.862454 loss=-6514.8112, loss_no_reg=-6514.8112, loss_reg=0.0000

2020-11-06 09:54:43.862500 run epoch 99:  time=131.14s
2020-11-06 09:56:55.002398 loss=-6513.0889, loss_no_reg=-6513.0889, loss_reg=0.0000

2020-11-06 09:56:55.002446 run epoch 100:  time=131.28s
2020-11-06 09:59:06.283930 loss=-6511.1466, loss_no_reg=-6511.1466, loss_reg=0.0000

2020-11-06 09:59:06.283996 run epoch 101:  time=127.59s
2020-11-06 10:01:13.872532 loss=-6511.7291, loss_no_reg=-6511.7291, loss_reg=0.0000

2020-11-06 10:01:13.872576 evaluate at epoch 101
2020-11-06 10:03:48.590264 hitrate@20=0.2182, recall@20=0.0357, ndcg@20=0.0280
2020-11-06 10:03:48.590364 hitrate@50=0.3647, recall@50=0.0690, ndcg@50=0.0403
2020-11-06 10:03:48.590399 the monitor counts down its patience to 1!

2020-11-06 10:03:48.592035 run epoch 102:  time=130.86s
2020-11-06 10:05:59.447575 loss=-6510.8229, loss_no_reg=-6510.8229, loss_reg=0.0000

2020-11-06 10:05:59.448276 run epoch 103:  time=131.01s
2020-11-06 10:08:10.458889 loss=-6521.3739, loss_no_reg=-6521.3739, loss_reg=0.0000

2020-11-06 10:08:10.458935 run epoch 104:  time=131.08s
2020-11-06 10:10:21.543175 loss=-6523.9877, loss_no_reg=-6523.9877, loss_reg=0.0000

2020-11-06 10:10:21.543232 run epoch 105:  time=131.26s
2020-11-06 10:12:32.802528 loss=-6515.0665, loss_no_reg=-6515.0665, loss_reg=0.0000

2020-11-06 10:12:32.803137 run epoch 106:  time=131.68s
2020-11-06 10:14:44.483522 loss=-6519.0530, loss_no_reg=-6519.0530, loss_reg=0.0000

2020-11-06 10:14:44.483582 run epoch 107:  time=129.60s
2020-11-06 10:16:54.080723 loss=-6514.4166, loss_no_reg=-6514.4166, loss_reg=0.0000

2020-11-06 10:16:54.081355 run epoch 108:  time=130.89s
2020-11-06 10:19:04.970278 loss=-6520.3047, loss_no_reg=-6520.3047, loss_reg=0.0000

2020-11-06 10:19:04.970343 run epoch 109:  time=129.45s
2020-11-06 10:21:14.417001 loss=-6518.1794, loss_no_reg=-6518.1794, loss_reg=0.0000

2020-11-06 10:21:14.417048 run epoch 110:  time=130.33s
2020-11-06 10:23:24.751493 loss=-6520.9980, loss_no_reg=-6520.9980, loss_reg=0.0000

2020-11-06 10:23:24.751539 run epoch 111:  time=130.58s
2020-11-06 10:25:35.336386 loss=-6521.6481, loss_no_reg=-6521.6481, loss_reg=0.0000

2020-11-06 10:25:35.336430 evaluate at epoch 111
2020-11-06 10:28:10.298409 hitrate@20=0.2180, recall@20=0.0357, ndcg@20=0.0280
2020-11-06 10:28:10.298488 hitrate@50=0.3659, recall@50=0.0691, ndcg@50=0.0404
2020-11-06 10:28:10.298520 the monitor counts down its patience to 0!
2020-11-06 10:28:10.299932 early stop at epoch 111
```