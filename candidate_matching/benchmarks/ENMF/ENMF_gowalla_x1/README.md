
## ENMF_gowalla_x0 

A notebook to benchmark ENMF on gowalla_x0 dataset.

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
gowalla_x0 follows the data split and preprocessing steps in NGCF and LightGCN. We directly transform the formats of the data from their [repo](https://github.com/kuandeng/LightGCN/tree/master/Data).

### Code

The details of the code are described [here](./README.md).

1. Downalod the dataset from LightGCN repo and run the preprocessing script for format transformation.

    ```python
    cd data/Gowalla/gowalla_x0
    python ENMF_data_process.py
    cd benchmarks/ENMF
    ```

2. Run the following script to reproduce the result.

    ```python
    # Hyper-parameters:
    dataset = "gowalla_x0"
    train_data = "../../data/Gowalla/gowalla_x0/train_enmf.txt"
    test_data = "../../data/Gowalla/gowalla_x0/test_enmf.txt"
    max_epochs = 500
    batch_size = 256
    embed_size = 64
    lr = 0.05
    dropout = 0.7 # dropout keep_prob
    negative_weight = 0.02 # loss weight of non-observed data, tuned among [0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5]
    verbose = 1 # Evaluation interval
    topK = "20 50" # Metrics at TopK
    ```
    ```python
    # Run ENMF
    python -u ENMF.py --gpu 0 --dataset gowalla_x0 --train_data ../../data/Gowalla/gowalla_x0/train_enmf.txt --test_data ../../data/Gowalla/gowalla_x0/test_enmf.txt --verbose 10 --batch_size 256 --epochs 500 --embed_size 64 --lr 0.05 --dropout 0.7 --negative_weight 0.02 --topK 20 50
    ```

### Results
```python
2020-11-05 14:41:54.163967 evaluate at epoch 51
2020-11-05 14:42:35.926767 hitrate@20=0.5336, recall@20=0.1523, ndcg@20=0.1315
2020-11-05 14:42:35.926861 hitrate@50=0.6701, recall@50=0.2379, ndcg@50=0.1583
```

### Logs
```python
tensorflow version: 1.14.0
2020-11-05 14:24:40.545104 loading data ../../data/Gowalla/gowalla_x0/train_enmf.txt
2020-11-05 14:24:40.667923 loading data ../../data/Gowalla/gowalla_x0/test_enmf.txt
2020-11-05 14:24:40.708847 finish loading
2020-11-05 14:24:41.137991 #users=29858, #items=40981
2020-11-05 14:24:42.083721 maximum number of items per user is 811
2020-11-05 14:24:50.588335 begin to construct the graph

2020-11-05 14:24:55.457867 run epoch 1:  time=17.97s
2020-11-05 14:25:13.430076 loss=0.0705, loss_no_reg=0.0705, loss_reg=0.0000

2020-11-05 14:25:13.430126 evaluate at epoch 1
2020-11-05 14:26:00.009660 hitrate@20=0.0540, recall@20=0.0081, ndcg@20=0.0064
2020-11-05 14:26:00.009767 hitrate@50=0.0923, recall@50=0.0151, ndcg@50=0.0086

2020-11-05 14:26:00.009847 run epoch 2:  time=16.25s
2020-11-05 14:26:16.257048 loss=-678.5478, loss_no_reg=-678.5478, loss_reg=0.0000

2020-11-05 14:26:16.257116 run epoch 3:  time=14.50s
2020-11-05 14:26:30.756757 loss=-1970.0730, loss_no_reg=-1970.0730, loss_reg=0.0000

2020-11-05 14:26:30.756799 run epoch 4:  time=14.85s
2020-11-05 14:26:45.609898 loss=-2527.4052, loss_no_reg=-2527.4052, loss_reg=0.0000

2020-11-05 14:26:45.609942 run epoch 5:  time=16.36s
2020-11-05 14:27:01.970865 loss=-2859.7927, loss_no_reg=-2859.7927, loss_reg=0.0000

2020-11-05 14:27:01.970936 run epoch 6:  time=16.61s
2020-11-05 14:27:18.576329 loss=-3057.7846, loss_no_reg=-3057.7846, loss_reg=0.0000

2020-11-05 14:27:18.576374 run epoch 7:  time=15.06s
2020-11-05 14:27:33.641032 loss=-3192.6114, loss_no_reg=-3192.6114, loss_reg=0.0000

2020-11-05 14:27:33.641078 run epoch 8:  time=15.12s
2020-11-05 14:27:48.765894 loss=-3286.1758, loss_no_reg=-3286.1758, loss_reg=0.0000

2020-11-05 14:27:48.765952 run epoch 9:  time=16.12s
2020-11-05 14:28:04.884585 loss=-3355.3137, loss_no_reg=-3355.3137, loss_reg=0.0000

2020-11-05 14:28:04.884633 run epoch 10:  time=15.79s
2020-11-05 14:28:20.678887 loss=-3411.6708, loss_no_reg=-3411.6708, loss_reg=0.0000

2020-11-05 14:28:20.678931 run epoch 11:  time=15.40s
2020-11-05 14:28:36.074148 loss=-3448.2753, loss_no_reg=-3448.2753, loss_reg=0.0000

2020-11-05 14:28:36.074199 evaluate at epoch 11
2020-11-05 14:29:19.948857 hitrate@20=0.5250, recall@20=0.1492, ndcg@20=0.1266
2020-11-05 14:29:19.948929 hitrate@50=0.6595, recall@50=0.2309, ndcg@50=0.1523

2020-11-05 14:29:19.948988 run epoch 12:  time=15.82s
2020-11-05 14:29:35.767270 loss=-3484.0659, loss_no_reg=-3484.0659, loss_reg=0.0000

2020-11-05 14:29:35.767305 run epoch 13:  time=15.34s
2020-11-05 14:29:51.110989 loss=-3511.4755, loss_no_reg=-3511.4755, loss_reg=0.0000

2020-11-05 14:29:51.111035 run epoch 14:  time=15.93s
2020-11-05 14:30:07.044272 loss=-3527.9598, loss_no_reg=-3527.9598, loss_reg=0.0000

2020-11-05 14:30:07.044446 run epoch 15:  time=15.49s
2020-11-05 14:30:22.536574 loss=-3548.5888, loss_no_reg=-3548.5888, loss_reg=0.0000

2020-11-05 14:30:22.536619 run epoch 16:  time=15.82s
2020-11-05 14:30:38.353090 loss=-3570.7002, loss_no_reg=-3570.7002, loss_reg=0.0000

2020-11-05 14:30:38.353133 run epoch 17:  time=14.92s
2020-11-05 14:30:53.275545 loss=-3577.2883, loss_no_reg=-3577.2883, loss_reg=0.0000

2020-11-05 14:30:53.275669 run epoch 18:  time=16.19s
2020-11-05 14:31:09.466921 loss=-3591.6693, loss_no_reg=-3591.6693, loss_reg=0.0000

2020-11-05 14:31:09.466972 run epoch 19:  time=14.00s
2020-11-05 14:31:23.464993 loss=-3598.6239, loss_no_reg=-3598.6239, loss_reg=0.0000

2020-11-05 14:31:23.465038 run epoch 20:  time=14.09s
2020-11-05 14:31:37.555500 loss=-3616.6610, loss_no_reg=-3616.6610, loss_reg=0.0000

2020-11-05 14:31:37.555546 run epoch 21:  time=14.45s
2020-11-05 14:31:52.003870 loss=-3628.4575, loss_no_reg=-3628.4575, loss_reg=0.0000

2020-11-05 14:31:52.003915 evaluate at epoch 21
2020-11-05 14:32:37.149083 hitrate@20=0.5307, recall@20=0.1520, ndcg@20=0.1301
2020-11-05 14:32:37.149166 hitrate@50=0.6677, recall@50=0.2368, ndcg@50=0.1568

2020-11-05 14:32:37.149231 run epoch 22:  time=15.72s
2020-11-05 14:32:52.864660 loss=-3625.4096, loss_no_reg=-3625.4096, loss_reg=0.0000

2020-11-05 14:32:52.864714 run epoch 23:  time=16.37s
2020-11-05 14:33:09.233670 loss=-3638.5632, loss_no_reg=-3638.5632, loss_reg=0.0000

2020-11-05 14:33:09.233718 run epoch 24:  time=16.66s
2020-11-05 14:33:25.888913 loss=-3651.4722, loss_no_reg=-3651.4722, loss_reg=0.0000

2020-11-05 14:33:25.888958 run epoch 25:  time=13.74s
2020-11-05 14:33:39.626729 loss=-3653.3814, loss_no_reg=-3653.3814, loss_reg=0.0000

2020-11-05 14:33:39.626777 run epoch 26:  time=14.07s
2020-11-05 14:33:53.702450 loss=-3652.8853, loss_no_reg=-3652.8853, loss_reg=0.0000

2020-11-05 14:33:53.702537 run epoch 27:  time=14.54s
2020-11-05 14:34:08.242295 loss=-3664.5434, loss_no_reg=-3664.5434, loss_reg=0.0000

2020-11-05 14:34:08.242946 run epoch 28:  time=17.00s
2020-11-05 14:34:25.243797 loss=-3668.4953, loss_no_reg=-3668.4953, loss_reg=0.0000

2020-11-05 14:34:25.243857 run epoch 29:  time=15.60s
2020-11-05 14:34:40.847417 loss=-3674.4141, loss_no_reg=-3674.4141, loss_reg=0.0000

2020-11-05 14:34:40.847478 run epoch 30:  time=15.72s
2020-11-05 14:34:56.566528 loss=-3680.4181, loss_no_reg=-3680.4181, loss_reg=0.0000

2020-11-05 14:34:56.566593 run epoch 31:  time=15.77s
2020-11-05 14:35:12.340207 loss=-3680.2908, loss_no_reg=-3680.2908, loss_reg=0.0000

2020-11-05 14:35:12.340254 evaluate at epoch 31
2020-11-05 14:35:58.905301 hitrate@20=0.5340, recall@20=0.1522, ndcg@20=0.1307
2020-11-05 14:35:58.905408 hitrate@50=0.6705, recall@50=0.2379, ndcg@50=0.1575

2020-11-05 14:35:58.905492 run epoch 32:  time=14.38s
2020-11-05 14:36:13.280900 loss=-3677.2866, loss_no_reg=-3677.2866, loss_reg=0.0000

2020-11-05 14:36:13.280952 run epoch 33:  time=15.87s
2020-11-05 14:36:29.151417 loss=-3685.9787, loss_no_reg=-3685.9787, loss_reg=0.0000

2020-11-05 14:36:29.151465 run epoch 34:  time=16.41s
2020-11-05 14:36:45.561384 loss=-3690.5077, loss_no_reg=-3690.5077, loss_reg=0.0000

2020-11-05 14:36:45.561451 run epoch 35:  time=15.37s
2020-11-05 14:37:00.930460 loss=-3696.0664, loss_no_reg=-3696.0664, loss_reg=0.0000

2020-11-05 14:37:00.930513 run epoch 36:  time=15.75s
2020-11-05 14:37:16.678925 loss=-3694.9426, loss_no_reg=-3694.9426, loss_reg=0.0000

2020-11-05 14:37:16.678975 run epoch 37:  time=16.65s
2020-11-05 14:37:33.333791 loss=-3693.8995, loss_no_reg=-3693.8995, loss_reg=0.0000

2020-11-05 14:37:33.333841 run epoch 38:  time=15.54s
2020-11-05 14:37:48.875474 loss=-3696.3235, loss_no_reg=-3696.3235, loss_reg=0.0000

2020-11-05 14:37:48.875528 run epoch 39:  time=14.44s
2020-11-05 14:38:03.314864 loss=-3704.1815, loss_no_reg=-3704.1815, loss_reg=0.0000

2020-11-05 14:38:03.314929 run epoch 40:  time=14.38s
2020-11-05 14:38:17.697130 loss=-3709.1519, loss_no_reg=-3709.1519, loss_reg=0.0000

2020-11-05 14:38:17.697192 run epoch 41:  time=14.89s
2020-11-05 14:38:32.589376 loss=-3711.0266, loss_no_reg=-3711.0266, loss_reg=0.0000

2020-11-05 14:38:32.589428 evaluate at epoch 41
2020-11-05 14:39:18.180012 hitrate@20=0.5333, recall@20=0.1519, ndcg@20=0.1308
2020-11-05 14:39:18.180094 hitrate@50=0.6715, recall@50=0.2378, ndcg@50=0.1578
2020-11-05 14:39:18.180140 the monitor counts down its patience to 4!

2020-11-05 14:39:18.180187 run epoch 42:  time=17.00s
2020-11-05 14:39:35.180107 loss=-3706.6879, loss_no_reg=-3706.6879, loss_reg=0.0000

2020-11-05 14:39:35.180185 run epoch 43:  time=15.77s
2020-11-05 14:39:50.953448 loss=-3707.8078, loss_no_reg=-3707.8078, loss_reg=0.0000

2020-11-05 14:39:50.953499 run epoch 44:  time=13.92s
2020-11-05 14:40:04.875291 loss=-3714.0457, loss_no_reg=-3714.0457, loss_reg=0.0000

2020-11-05 14:40:04.875349 run epoch 45:  time=13.44s
2020-11-05 14:40:18.316037 loss=-3723.0038, loss_no_reg=-3723.0038, loss_reg=0.0000

2020-11-05 14:40:18.316085 run epoch 46:  time=15.05s
2020-11-05 14:40:33.364032 loss=-3709.3070, loss_no_reg=-3709.3070, loss_reg=0.0000

2020-11-05 14:40:33.364086 run epoch 47:  time=16.38s
2020-11-05 14:40:49.748375 loss=-3718.1117, loss_no_reg=-3718.1117, loss_reg=0.0000

2020-11-05 14:40:49.748422 run epoch 48:  time=15.73s
2020-11-05 14:41:05.477056 loss=-3715.8040, loss_no_reg=-3715.8040, loss_reg=0.0000

2020-11-05 14:41:05.477119 run epoch 49:  time=15.51s
2020-11-05 14:41:20.986787 loss=-3726.9318, loss_no_reg=-3726.9318, loss_reg=0.0000

2020-11-05 14:41:20.986841 run epoch 50:  time=16.46s
2020-11-05 14:41:37.443625 loss=-3723.8979, loss_no_reg=-3723.8979, loss_reg=0.0000

2020-11-05 14:41:37.443674 run epoch 51:  time=16.72s
2020-11-05 14:41:54.163918 loss=-3730.4407, loss_no_reg=-3730.4407, loss_reg=0.0000

2020-11-05 14:41:54.163967 evaluate at epoch 51
2020-11-05 14:42:35.926767 hitrate@20=0.5336, recall@20=0.1523, ndcg@20=0.1315
2020-11-05 14:42:35.926861 hitrate@50=0.6701, recall@50=0.2379, ndcg@50=0.1583

2020-11-05 14:42:35.926945 run epoch 52:  time=16.65s
2020-11-05 14:42:52.578979 loss=-3719.3408, loss_no_reg=-3719.3408, loss_reg=0.0000

2020-11-05 14:42:52.579058 run epoch 53:  time=15.16s
2020-11-05 14:43:07.739631 loss=-3724.0628, loss_no_reg=-3724.0628, loss_reg=0.0000

2020-11-05 14:43:07.739683 run epoch 54:  time=15.33s
2020-11-05 14:43:23.068120 loss=-3724.3541, loss_no_reg=-3724.3541, loss_reg=0.0000

2020-11-05 14:43:23.068184 run epoch 55:  time=16.19s
2020-11-05 14:43:39.256042 loss=-3730.7261, loss_no_reg=-3730.7261, loss_reg=0.0000

2020-11-05 14:43:39.256092 run epoch 56:  time=16.56s
2020-11-05 14:43:55.817813 loss=-3725.1721, loss_no_reg=-3725.1721, loss_reg=0.0000

2020-11-05 14:43:55.817862 run epoch 57:  time=16.06s
2020-11-05 14:44:11.873796 loss=-3732.6612, loss_no_reg=-3732.6612, loss_reg=0.0000

2020-11-05 14:44:11.873848 run epoch 58:  time=14.50s
2020-11-05 14:44:26.371149 loss=-3730.0294, loss_no_reg=-3730.0294, loss_reg=0.0000

2020-11-05 14:44:26.371208 run epoch 59:  time=13.28s
2020-11-05 14:44:39.650435 loss=-3731.3148, loss_no_reg=-3731.3148, loss_reg=0.0000

2020-11-05 14:44:39.650504 run epoch 60:  time=15.00s
2020-11-05 14:44:54.653415 loss=-3736.1424, loss_no_reg=-3736.1424, loss_reg=0.0000

2020-11-05 14:44:54.653482 run epoch 61:  time=16.36s
2020-11-05 14:45:11.014866 loss=-3740.1074, loss_no_reg=-3740.1074, loss_reg=0.0000

2020-11-05 14:45:11.014925 evaluate at epoch 61
2020-11-05 14:45:56.805999 hitrate@20=0.5337, recall@20=0.1521, ndcg@20=0.1311
2020-11-05 14:45:56.806084 hitrate@50=0.6713, recall@50=0.2374, ndcg@50=0.1579
2020-11-05 14:45:56.806118 the monitor counts down its patience to 4!

2020-11-05 14:45:56.806161 run epoch 62:  time=16.62s
2020-11-05 14:46:13.426087 loss=-3735.8558, loss_no_reg=-3735.8558, loss_reg=0.0000

2020-11-05 14:46:13.426177 run epoch 63:  time=14.93s
2020-11-05 14:46:28.355400 loss=-3737.6473, loss_no_reg=-3737.6473, loss_reg=0.0000

2020-11-05 14:46:28.355473 run epoch 64:  time=13.60s
2020-11-05 14:46:41.953543 loss=-3739.6287, loss_no_reg=-3739.6287, loss_reg=0.0000

2020-11-05 14:46:41.953592 run epoch 65:  time=14.58s
2020-11-05 14:46:56.531500 loss=-3740.9554, loss_no_reg=-3740.9554, loss_reg=0.0000

2020-11-05 14:46:56.531549 run epoch 66:  time=15.82s
2020-11-05 14:47:12.347099 loss=-3734.0559, loss_no_reg=-3734.0559, loss_reg=0.0000

2020-11-05 14:47:12.347181 run epoch 67:  time=14.23s
2020-11-05 14:47:26.580507 loss=-3741.5519, loss_no_reg=-3741.5519, loss_reg=0.0000

2020-11-05 14:47:26.580568 run epoch 68:  time=15.67s
2020-11-05 14:47:42.245805 loss=-3742.6511, loss_no_reg=-3742.6511, loss_reg=0.0000

2020-11-05 14:47:42.245854 run epoch 69:  time=16.11s
2020-11-05 14:47:58.355282 loss=-3743.4645, loss_no_reg=-3743.4645, loss_reg=0.0000

2020-11-05 14:47:58.355348 run epoch 70:  time=17.17s
2020-11-05 14:48:15.527714 loss=-3745.4726, loss_no_reg=-3745.4726, loss_reg=0.0000

2020-11-05 14:48:15.527773 run epoch 71:  time=15.22s
2020-11-05 14:48:30.747955 loss=-3740.1211, loss_no_reg=-3740.1211, loss_reg=0.0000

2020-11-05 14:48:30.748001 evaluate at epoch 71
2020-11-05 14:49:16.214479 hitrate@20=0.5331, recall@20=0.1521, ndcg@20=0.1313
2020-11-05 14:49:16.214562 hitrate@50=0.6710, recall@50=0.2373, ndcg@50=0.1581
2020-11-05 14:49:16.214596 the monitor counts down its patience to 3!

2020-11-05 14:49:16.214632 run epoch 72:  time=15.50s
2020-11-05 14:49:31.715396 loss=-3749.4860, loss_no_reg=-3749.4860, loss_reg=0.0000

2020-11-05 14:49:31.715448 run epoch 73:  time=15.89s
2020-11-05 14:49:47.602338 loss=-3745.1134, loss_no_reg=-3745.1134, loss_reg=0.0000

2020-11-05 14:49:47.602388 run epoch 74:  time=15.82s
2020-11-05 14:50:03.417855 loss=-3741.1661, loss_no_reg=-3741.1661, loss_reg=0.0000

2020-11-05 14:50:03.417896 run epoch 75:  time=16.60s
2020-11-05 14:50:20.022490 loss=-3748.1676, loss_no_reg=-3748.1676, loss_reg=0.0000

2020-11-05 14:50:20.022540 run epoch 76:  time=15.68s
2020-11-05 14:50:35.704285 loss=-3741.1869, loss_no_reg=-3741.1869, loss_reg=0.0000

2020-11-05 14:50:35.704356 run epoch 77:  time=15.87s
2020-11-05 14:50:51.575402 loss=-3752.8621, loss_no_reg=-3752.8621, loss_reg=0.0000

2020-11-05 14:50:51.575451 run epoch 78:  time=16.35s
2020-11-05 14:51:07.922660 loss=-3754.3046, loss_no_reg=-3754.3046, loss_reg=0.0000

2020-11-05 14:51:07.922712 run epoch 79:  time=13.48s
2020-11-05 14:51:21.400981 loss=-3750.3807, loss_no_reg=-3750.3807, loss_reg=0.0000

2020-11-05 14:51:21.401019 run epoch 80:  time=13.28s
2020-11-05 14:51:34.678700 loss=-3747.8591, loss_no_reg=-3747.8591, loss_reg=0.0000

2020-11-05 14:51:34.678751 run epoch 81:  time=15.38s
2020-11-05 14:51:50.059322 loss=-3751.6217, loss_no_reg=-3751.6217, loss_reg=0.0000

2020-11-05 14:51:50.059372 evaluate at epoch 81
2020-11-05 14:52:37.744352 hitrate@20=0.5330, recall@20=0.1517, ndcg@20=0.1311
2020-11-05 14:52:37.744442 hitrate@50=0.6702, recall@50=0.2370, ndcg@50=0.1579
2020-11-05 14:52:37.744476 the monitor counts down its patience to 2!

2020-11-05 14:52:37.744512 run epoch 82:  time=15.83s
2020-11-05 14:52:53.572596 loss=-3751.1920, loss_no_reg=-3751.1920, loss_reg=0.0000

2020-11-05 14:52:53.572663 run epoch 83:  time=15.74s
2020-11-05 14:53:09.309967 loss=-3757.1159, loss_no_reg=-3757.1159, loss_reg=0.0000

2020-11-05 14:53:09.310019 run epoch 84:  time=14.37s
2020-11-05 14:53:23.683123 loss=-3752.0206, loss_no_reg=-3752.0206, loss_reg=0.0000

2020-11-05 14:53:23.683202 run epoch 85:  time=13.42s
2020-11-05 14:53:37.102905 loss=-3754.5468, loss_no_reg=-3754.5468, loss_reg=0.0000

2020-11-05 14:53:37.102999 run epoch 86:  time=15.59s
2020-11-05 14:53:52.693022 loss=-3750.8158, loss_no_reg=-3750.8158, loss_reg=0.0000

2020-11-05 14:53:52.693070 run epoch 87:  time=15.90s
2020-11-05 14:54:08.589706 loss=-3756.5955, loss_no_reg=-3756.5955, loss_reg=0.0000

2020-11-05 14:54:08.589750 run epoch 88:  time=15.77s
2020-11-05 14:54:24.364286 loss=-3750.9872, loss_no_reg=-3750.9872, loss_reg=0.0000

2020-11-05 14:54:24.364336 run epoch 89:  time=15.68s
2020-11-05 14:54:40.045180 loss=-3754.7238, loss_no_reg=-3754.7238, loss_reg=0.0000

2020-11-05 14:54:40.045228 run epoch 90:  time=16.44s
2020-11-05 14:54:56.483431 loss=-3756.4831, loss_no_reg=-3756.4831, loss_reg=0.0000

2020-11-05 14:54:56.483494 run epoch 91:  time=16.20s
2020-11-05 14:55:12.685749 loss=-3757.3408, loss_no_reg=-3757.3408, loss_reg=0.0000

2020-11-05 14:55:12.685822 evaluate at epoch 91
2020-11-05 14:55:56.206961 hitrate@20=0.5336, recall@20=0.1516, ndcg@20=0.1309
2020-11-05 14:55:56.207045 hitrate@50=0.6704, recall@50=0.2365, ndcg@50=0.1576
2020-11-05 14:55:56.207080 the monitor counts down its patience to 1!

2020-11-05 14:55:56.207116 run epoch 92:  time=15.88s
2020-11-05 14:56:12.089922 loss=-3752.3973, loss_no_reg=-3752.3973, loss_reg=0.0000

2020-11-05 14:56:12.089979 run epoch 93:  time=15.45s
2020-11-05 14:56:27.543177 loss=-3751.5166, loss_no_reg=-3751.5166, loss_reg=0.0000

2020-11-05 14:56:27.543232 run epoch 94:  time=15.88s
2020-11-05 14:56:43.420126 loss=-3755.8486, loss_no_reg=-3755.8486, loss_reg=0.0000

2020-11-05 14:56:43.420192 run epoch 95:  time=15.68s
2020-11-05 14:56:59.098850 loss=-3759.2409, loss_no_reg=-3759.2409, loss_reg=0.0000

2020-11-05 14:56:59.098928 run epoch 96:  time=15.83s
2020-11-05 14:57:14.927888 loss=-3759.5891, loss_no_reg=-3759.5891, loss_reg=0.0000

2020-11-05 14:57:14.927938 run epoch 97:  time=15.53s
2020-11-05 14:57:30.461194 loss=-3755.7791, loss_no_reg=-3755.7791, loss_reg=0.0000

2020-11-05 14:57:30.461263 run epoch 98:  time=13.35s
2020-11-05 14:57:43.812561 loss=-3764.7513, loss_no_reg=-3764.7513, loss_reg=0.0000

2020-11-05 14:57:43.812626 run epoch 99:  time=13.42s
2020-11-05 14:57:57.227810 loss=-3764.6951, loss_no_reg=-3764.6951, loss_reg=0.0000

2020-11-05 14:57:57.227856 run epoch 100:  time=15.79s
2020-11-05 14:58:13.018082 loss=-3755.9523, loss_no_reg=-3755.9523, loss_reg=0.0000

2020-11-05 14:58:13.018154 run epoch 101:  time=15.71s
2020-11-05 14:58:28.724636 loss=-3756.5302, loss_no_reg=-3756.5302, loss_reg=0.0000

2020-11-05 14:58:28.724692 evaluate at epoch 101
2020-11-05 14:59:16.514599 hitrate@20=0.5308, recall@20=0.1511, ndcg@20=0.1309
2020-11-05 14:59:16.514683 hitrate@50=0.6707, recall@50=0.2364, ndcg@50=0.1577
2020-11-05 14:59:16.514733 the monitor counts down its patience to 0!
2020-11-05 14:59:16.514769 early stop at epoch 101
```
