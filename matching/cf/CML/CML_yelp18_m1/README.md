## CML_yelp18_x0 

A notebook to benchmark CML on yelp18_x0 dataset.

Author: Jinpeng Wang, Tsinghua University

Edited by [XUEPAI Team](https://github.com/xue-pai)


### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

```python
CPU: Intel(R) Xeon(R) CPU E5-2680 v2 @ 2.80GHz
RAM: 64G
GPU: GeForce GTX 1080 Ti, 11G memory
```

+ Software

```bash
python: 3.7.8
tensorflow: 1.15.0
pandas: 1.0.5
numpy: 1.19.1
```

### Dataset
We follow the data split and preprocessing steps in NGCF and LightGCN. We directly transform the formats of the data from their [repo](https://github.com/kuandeng/LightGCN/tree/master/Data).

### Code

The benchmark is implemented based on the original CML code released by the authors at https://github.com/changun/CollMetric/tree/d9026cfce7c6e8dd2640b842ad524b61031b29ac, we use the version with commit hash: d9026cf. We also made the following modifications for our benchmarking. These changes can be viewed via a diff comparison here: https://github.com/xue-pai/Open-CF-Benchmarks/compare/5415b7...1d6bd6?diff=split

1. Tensorflow APIs updates, for example:
    ```python
    tf.nn.dropout(x, keep_prob) => tf.nn.dropout(x, rate=1 - keep_prob)
    ```
2. Add the class `Evaluator`, which evaluates the metircs (hitrate@k,recall@k,ndcg@k).
3. Add the class `Monitor`, which records the metircs (hitrate@20,recall@20,ndcg@20,hitrate@50,recall@50,ndcg@50) for validation and determines whether to early stop.
4. Add the method `dataset_to_uimatrix`, which reads and preprocess the input data as a training dataset.
5. Add calculation of hitrate@K, ndcg@K and the monitor into the method `optimize`.
6. Fix a bug of optimization ([issue#13](https://github.com/changun/CollMetric/issues/13)) by replacing the optimizer with Adam and setting the learning rate as 0.001.


Run the following script to reproduce the benchmarking result:

```python
# Hyper-parameters:
dataset = "yelp18_x0"
train_data = "../../data/Yelp18/yelp18_x0/train.txt"
test_data = "../../data/Yelp18/yelp18_x0/test.txt"
batch_size = 50000
max_step=5000
num_negative=20
embed_dim=100
margin=1.9
clip_norm=1
lr = 0.001
dropout = 0.2 # dropout rate
verbose = 1 # Evaluation interval
topK = "20 50" # Metrics at TopK
```

```bash
python -u main.py --gpu 1 --dataset yelp18_x0 --train_data ../../data/Yelp18/yelp18_x0/train.txt --test_data ../../data/Yelp18/yelp18_x0/test.txt --verbose 30 --batch_size 50000 --max_step 5000 --embed_dim 100 --lr 0.001 --dropout 0.2 --margin 1.9 --clip_norm 1 --num_negative 90 --topK 20 50
```


### Results
```bash
2020-11-16 00:07:10.282834 validation 70:
2020-11-16 00:07:10.282890 hitrate@20=0.3810, recall@20=0.0622, ndcg@20=0.0536
2020-11-16 00:07:10.282909 hitrate@50=0.5510, recall@50=0.1181, ndcg@50=0.0738
```


### Logs
```bash
2020-11-15 22:33:08.696847 reading dataset yelp18_x0
2020-11-15 22:33:36.197629 #users=31668, #items=38048
WARNING:tensorflow:From /home/wjp/anaconda3/envs/py37torch/lib/python3.7/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-11-15 22:33:36.650016: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
2020-11-15 22:33:36.688435: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2799850000 Hz
2020-11-15 22:33:36.691698: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d0098f02a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-11-15 22:33:36.691737: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-11-15 22:33:36.694104: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-11-15 22:33:37.291281: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:42:00.0
2020-11-15 22:33:37.291584: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-11-15 22:33:37.293322: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-11-15 22:33:37.294627: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2020-11-15 22:33:37.294916: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2020-11-15 22:33:37.296899: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2020-11-15 22:33:37.298409: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2020-11-15 22:33:37.303502: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-11-15 22:33:37.305151: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-11-15 22:33:37.305221: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-11-15 22:33:37.423936: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-11-15 22:33:37.424014: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
2020-11-15 22:33:37.424028: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
2020-11-15 22:33:37.426360: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10481 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:42:00.0, compute capability: 6.1)
2020-11-15 22:33:37.430649: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d00b5dbe30 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-11-15 22:33:37.430678: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce GTX 1080 Ti, Compute Capability 6.1
2020-11-15 22:33:40.774690 validation 1:
2020-11-15 22:33:40.774753 hitrate@20=0.0050, recall@20=0.0009, ndcg@20=0.0005
2020-11-15 22:33:40.774780 hitrate@50=0.0080, recall@50=0.0013, ndcg@50=0.0007

Optimizing: 100%|███████████████████████████████| 30/30 [01:23<00:00,  2.80s/it]
2020-11-15 22:35:04.655949 Step 0, training loss 1211507.75

2020-11-15 22:35:07.387307 validation 2:
2020-11-15 22:35:07.387402 hitrate@20=0.0150, recall@20=0.0017, ndcg@20=0.0015
2020-11-15 22:35:07.387427 hitrate@50=0.0460, recall@50=0.0056, ndcg@50=0.0029

Optimizing: 100%|███████████████████████████████| 30/30 [01:16<00:00,  2.55s/it]
2020-11-15 22:36:23.804206 Step 1, training loss 1183846.375

2020-11-15 22:36:26.633831 validation 3:
2020-11-15 22:36:26.633878 hitrate@20=0.0220, recall@20=0.0025, ndcg@20=0.0019
2020-11-15 22:36:26.633913 hitrate@50=0.0470, recall@50=0.0058, ndcg@50=0.0031

Optimizing: 100%|███████████████████████████████| 30/30 [01:16<00:00,  2.54s/it]
2020-11-15 22:37:42.895213 Step 2, training loss 1167440.75

2020-11-15 22:37:45.349086 validation 4:
2020-11-15 22:37:45.349154 hitrate@20=0.0190, recall@20=0.0024, ndcg@20=0.0018
2020-11-15 22:37:45.349194 hitrate@50=0.0460, recall@50=0.0057, ndcg@50=0.0030

2020-11-15 22:37:45.349230 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [01:16<00:00,  2.55s/it]
2020-11-15 22:39:01.717962 Step 3, training loss 1152962.375

2020-11-15 22:39:04.154333 validation 5:
2020-11-15 22:39:04.154388 hitrate@20=0.0210, recall@20=0.0025, ndcg@20=0.0018
2020-11-15 22:39:04.154428 hitrate@50=0.0470, recall@50=0.0057, ndcg@50=0.0030

2020-11-15 22:39:04.154462 the monitor loses its patience to 3!
Optimizing: 100%|███████████████████████████████| 30/30 [01:17<00:00,  2.60s/it]
2020-11-15 22:40:22.130510 Step 4, training loss 1108319.25

2020-11-15 22:40:24.613607 validation 6:
2020-11-15 22:40:24.613684 hitrate@20=0.0220, recall@20=0.0023, ndcg@20=0.0019
2020-11-15 22:40:24.613710 hitrate@50=0.0450, recall@50=0.0051, ndcg@50=0.0029

2020-11-15 22:40:24.613742 the monitor loses its patience to 2!
Optimizing: 100%|███████████████████████████████| 30/30 [01:16<00:00,  2.55s/it]
2020-11-15 22:41:40.999360 Step 5, training loss 1100773.875

2020-11-15 22:41:43.507603 validation 7:
2020-11-15 22:41:43.507654 hitrate@20=0.0210, recall@20=0.0024, ndcg@20=0.0018
2020-11-15 22:41:43.507689 hitrate@50=0.0540, recall@50=0.0061, ndcg@50=0.0032

Optimizing: 100%|███████████████████████████████| 30/30 [01:17<00:00,  2.57s/it]
2020-11-15 22:43:00.523092 Step 6, training loss 1093863.75

2020-11-15 22:43:03.020366 validation 8:
2020-11-15 22:43:03.020417 hitrate@20=0.0250, recall@20=0.0027, ndcg@20=0.0024
2020-11-15 22:43:03.020454 hitrate@50=0.0630, recall@50=0.0064, ndcg@50=0.0039

Optimizing: 100%|███████████████████████████████| 30/30 [01:16<00:00,  2.55s/it]
2020-11-15 22:44:19.649585 Step 7, training loss 1086499.25

2020-11-15 22:44:22.593368 validation 9:
2020-11-15 22:44:22.593445 hitrate@20=0.0250, recall@20=0.0027, ndcg@20=0.0029
2020-11-15 22:44:22.593472 hitrate@50=0.0610, recall@50=0.0062, ndcg@50=0.0044

Optimizing: 100%|███████████████████████████████| 30/30 [01:17<00:00,  2.58s/it]
2020-11-15 22:45:40.138668 Step 8, training loss 1054282.125

2020-11-15 22:45:42.693233 validation 10:
2020-11-15 22:45:42.693291 hitrate@20=0.0350, recall@20=0.0031, ndcg@20=0.0030
2020-11-15 22:45:42.693330 hitrate@50=0.0660, recall@50=0.0064, ndcg@50=0.0044

Optimizing: 100%|███████████████████████████████| 30/30 [01:14<00:00,  2.50s/it]
2020-11-15 22:46:57.692268 Step 9, training loss 1048491.0625

2020-11-15 22:47:00.279614 validation 11:
2020-11-15 22:47:00.279669 hitrate@20=0.0560, recall@20=0.0051, ndcg@20=0.0054
2020-11-15 22:47:00.279698 hitrate@50=0.1110, recall@50=0.0110, ndcg@50=0.0075

Optimizing: 100%|███████████████████████████████| 30/30 [01:15<00:00,  2.50s/it]
2020-11-15 22:48:15.353062 Step 10, training loss 1042616.6875

2020-11-15 22:48:18.522683 validation 12:
2020-11-15 22:48:18.522750 hitrate@20=0.0810, recall@20=0.0071, ndcg@20=0.0074
2020-11-15 22:48:18.522774 hitrate@50=0.1370, recall@50=0.0153, ndcg@50=0.0102

Optimizing: 100%|███████████████████████████████| 30/30 [01:15<00:00,  2.53s/it]
2020-11-15 22:49:34.305306 Step 11, training loss 1036733.0

2020-11-15 22:49:36.998464 validation 13:
2020-11-15 22:49:36.998533 hitrate@20=0.1090, recall@20=0.0114, ndcg@20=0.0117
2020-11-15 22:49:36.998573 hitrate@50=0.1890, recall@50=0.0222, ndcg@50=0.0155

Optimizing: 100%|███████████████████████████████| 30/30 [01:15<00:00,  2.51s/it]
2020-11-15 22:50:52.241130 Step 12, training loss 1011554.125

2020-11-15 22:50:55.441597 validation 14:
2020-11-15 22:50:55.441667 hitrate@20=0.1210, recall@20=0.0130, ndcg@20=0.0137
2020-11-15 22:50:55.441691 hitrate@50=0.2150, recall@50=0.0274, ndcg@50=0.0190

Optimizing: 100%|███████████████████████████████| 30/30 [01:13<00:00,  2.46s/it]
2020-11-15 22:52:09.243094 Step 13, training loss 1008860.9375

2020-11-15 22:52:12.421144 validation 15:
2020-11-15 22:52:12.421213 hitrate@20=0.1510, recall@20=0.0172, ndcg@20=0.0168
2020-11-15 22:52:12.421251 hitrate@50=0.2430, recall@50=0.0337, ndcg@50=0.0229

Optimizing: 100%|███████████████████████████████| 30/30 [01:14<00:00,  2.48s/it]
2020-11-15 22:53:26.812584 Step 14, training loss 1007028.25

2020-11-15 22:53:30.139847 validation 16:
2020-11-15 22:53:30.139914 hitrate@20=0.1720, recall@20=0.0207, ndcg@20=0.0195
2020-11-15 22:53:30.139939 hitrate@50=0.2830, recall@50=0.0408, ndcg@50=0.0273

Optimizing: 100%|███████████████████████████████| 30/30 [01:15<00:00,  2.51s/it]
2020-11-15 22:54:45.538428 Step 15, training loss 1004049.625

2020-11-15 22:54:48.663090 validation 17:
2020-11-15 22:54:48.663144 hitrate@20=0.2030, recall@20=0.0255, ndcg@20=0.0235
2020-11-15 22:54:48.663165 hitrate@50=0.3280, recall@50=0.0503, ndcg@50=0.0327

Optimizing: 100%|███████████████████████████████| 30/30 [01:15<00:00,  2.51s/it]
2020-11-15 22:56:04.107660 Step 16, training loss 981820.25

2020-11-15 22:56:06.878689 validation 18:
2020-11-15 22:56:06.878744 hitrate@20=0.2130, recall@20=0.0271, ndcg@20=0.0241
2020-11-15 22:56:06.878779 hitrate@50=0.3390, recall@50=0.0545, ndcg@50=0.0343

Optimizing: 100%|███████████████████████████████| 30/30 [01:14<00:00,  2.48s/it]
2020-11-15 22:57:21.314882 Step 17, training loss 979473.0625

2020-11-15 22:57:24.106965 validation 19:
2020-11-15 22:57:24.107034 hitrate@20=0.2410, recall@20=0.0306, ndcg@20=0.0265
2020-11-15 22:57:24.107058 hitrate@50=0.3630, recall@50=0.0591, ndcg@50=0.0370

Optimizing: 100%|███████████████████████████████| 30/30 [01:15<00:00,  2.52s/it]
2020-11-15 22:58:39.567140 Step 18, training loss 978270.1875

2020-11-15 22:58:42.880425 validation 20:
2020-11-15 22:58:42.880488 hitrate@20=0.2590, recall@20=0.0342, ndcg@20=0.0292
2020-11-15 22:58:42.880522 hitrate@50=0.3760, recall@50=0.0640, ndcg@50=0.0402

Optimizing: 100%|███████████████████████████████| 30/30 [01:16<00:00,  2.55s/it]
2020-11-15 22:59:59.293911 Step 19, training loss 976213.75

2020-11-15 23:00:02.557375 validation 21:
2020-11-15 23:00:02.557428 hitrate@20=0.2550, recall@20=0.0345, ndcg@20=0.0297
2020-11-15 23:00:02.557462 hitrate@50=0.3810, recall@50=0.0647, ndcg@50=0.0410

Optimizing: 100%|███████████████████████████████| 30/30 [01:16<00:00,  2.56s/it]
2020-11-15 23:01:19.314002 Step 20, training loss 954834.3125

2020-11-15 23:01:22.204753 validation 22:
2020-11-15 23:01:22.204811 hitrate@20=0.2670, recall@20=0.0364, ndcg@20=0.0311
2020-11-15 23:01:22.204847 hitrate@50=0.4040, recall@50=0.0691, ndcg@50=0.0435

Optimizing: 100%|███████████████████████████████| 30/30 [01:16<00:00,  2.54s/it]
2020-11-15 23:02:38.389655 Step 21, training loss 954128.4375

2020-11-15 23:02:41.796271 validation 23:
2020-11-15 23:02:41.796328 hitrate@20=0.2710, recall@20=0.0381, ndcg@20=0.0321
2020-11-15 23:02:41.796348 hitrate@50=0.4120, recall@50=0.0730, ndcg@50=0.0454

Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.60s/it]
2020-11-15 23:03:59.810344 Step 22, training loss 955815.6875

2020-11-15 23:04:02.871272 validation 24:
2020-11-15 23:04:02.871330 hitrate@20=0.2780, recall@20=0.0393, ndcg@20=0.0340
2020-11-15 23:04:02.871352 hitrate@50=0.4420, recall@50=0.0780, ndcg@50=0.0486

Optimizing: 100%|███████████████████████████████| 30/30 [01:20<00:00,  2.68s/it]
2020-11-15 23:05:23.325974 Step 23, training loss 951871.375

2020-11-15 23:05:26.370015 validation 25:
2020-11-15 23:05:26.370075 hitrate@20=0.2880, recall@20=0.0411, ndcg@20=0.0367
2020-11-15 23:05:26.370096 hitrate@50=0.4250, recall@50=0.0775, ndcg@50=0.0502

Optimizing: 100%|███████████████████████████████| 30/30 [01:16<00:00,  2.55s/it]
2020-11-15 23:06:42.884406 Step 24, training loss 929730.0625

2020-11-15 23:06:45.937660 validation 26:
2020-11-15 23:06:45.937724 hitrate@20=0.3010, recall@20=0.0435, ndcg@20=0.0381
2020-11-15 23:06:45.937754 hitrate@50=0.4410, recall@50=0.0805, ndcg@50=0.0516

Optimizing: 100%|███████████████████████████████| 30/30 [01:16<00:00,  2.54s/it]
2020-11-15 23:08:02.067025 Step 25, training loss 926311.125

2020-11-15 23:08:05.056044 validation 27:
2020-11-15 23:08:05.056102 hitrate@20=0.3020, recall@20=0.0447, ndcg@20=0.0390
2020-11-15 23:08:05.056138 hitrate@50=0.4540, recall@50=0.0833, ndcg@50=0.0532

Optimizing: 100%|███████████████████████████████| 30/30 [01:14<00:00,  2.49s/it]
2020-11-15 23:09:19.705393 Step 26, training loss 926367.75

2020-11-15 23:09:22.735435 validation 28:
2020-11-15 23:09:22.735493 hitrate@20=0.3030, recall@20=0.0456, ndcg@20=0.0397
2020-11-15 23:09:22.735529 hitrate@50=0.4530, recall@50=0.0855, ndcg@50=0.0545

Optimizing: 100%|███████████████████████████████| 30/30 [01:14<00:00,  2.48s/it]
2020-11-15 23:10:37.073586 Step 27, training loss 923191.9375

2020-11-15 23:10:40.147599 validation 29:
2020-11-15 23:10:40.147671 hitrate@20=0.3210, recall@20=0.0474, ndcg@20=0.0403
2020-11-15 23:10:40.147709 hitrate@50=0.4630, recall@50=0.0883, ndcg@50=0.0556

Optimizing: 100%|███████████████████████████████| 30/30 [01:14<00:00,  2.50s/it]
2020-11-15 23:11:55.077044 Step 28, training loss 898886.0625

2020-11-15 23:11:58.228955 validation 30:
2020-11-15 23:11:58.229030 hitrate@20=0.3020, recall@20=0.0462, ndcg@20=0.0396
2020-11-15 23:11:58.229055 hitrate@50=0.4680, recall@50=0.0895, ndcg@50=0.0558

2020-11-15 23:11:58.229111 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [01:17<00:00,  2.59s/it]
2020-11-15 23:13:15.845686 Step 29, training loss 900853.3125

2020-11-15 23:13:19.096339 validation 31:
2020-11-15 23:13:19.096403 hitrate@20=0.3140, recall@20=0.0484, ndcg@20=0.0415
2020-11-15 23:13:19.096424 hitrate@50=0.4720, recall@50=0.0894, ndcg@50=0.0567

Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.61s/it]
2020-11-15 23:14:37.492665 Step 30, training loss 903659.875

2020-11-15 23:14:40.547306 validation 32:
2020-11-15 23:14:40.547368 hitrate@20=0.3040, recall@20=0.0468, ndcg@20=0.0405
2020-11-15 23:14:40.547387 hitrate@50=0.4800, recall@50=0.0918, ndcg@50=0.0572

Optimizing: 100%|███████████████████████████████| 30/30 [01:19<00:00,  2.66s/it]
2020-11-15 23:16:00.406073 Step 31, training loss 903503.0625

2020-11-15 23:16:03.674556 validation 33:
2020-11-15 23:16:03.674626 hitrate@20=0.3210, recall@20=0.0494, ndcg@20=0.0425
2020-11-15 23:16:03.674649 hitrate@50=0.4850, recall@50=0.0941, ndcg@50=0.0591

Optimizing: 100%|███████████████████████████████| 30/30 [01:19<00:00,  2.64s/it]
2020-11-15 23:17:22.745664 Step 32, training loss 883259.375

2020-11-15 23:17:26.243280 validation 34:
2020-11-15 23:17:26.243408 hitrate@20=0.3080, recall@20=0.0476, ndcg@20=0.0422
2020-11-15 23:17:26.243433 hitrate@50=0.4890, recall@50=0.0942, ndcg@50=0.0596

2020-11-15 23:17:26.243469 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.63s/it]
2020-11-15 23:18:45.218653 Step 33, training loss 885346.9375

2020-11-15 23:18:48.544864 validation 35:
2020-11-15 23:18:48.544923 hitrate@20=0.3230, recall@20=0.0499, ndcg@20=0.0434
2020-11-15 23:18:48.544943 hitrate@50=0.4880, recall@50=0.0963, ndcg@50=0.0607

Optimizing: 100%|███████████████████████████████| 30/30 [01:19<00:00,  2.65s/it]
2020-11-15 23:20:08.064251 Step 34, training loss 891593.3125

2020-11-15 23:20:11.439001 validation 36:
2020-11-15 23:20:11.439077 hitrate@20=0.3290, recall@20=0.0500, ndcg@20=0.0431
2020-11-15 23:20:11.439113 hitrate@50=0.4960, recall@50=0.0974, ndcg@50=0.0606

Optimizing: 100%|███████████████████████████████| 30/30 [01:19<00:00,  2.64s/it]
2020-11-15 23:21:30.600484 Step 35, training loss 892180.25

2020-11-15 23:21:33.923291 validation 37:
2020-11-15 23:21:33.923350 hitrate@20=0.3460, recall@20=0.0528, ndcg@20=0.0447
2020-11-15 23:21:33.923379 hitrate@50=0.4960, recall@50=0.0985, ndcg@50=0.0616

Optimizing: 100%|███████████████████████████████| 30/30 [01:21<00:00,  2.70s/it]
2020-11-15 23:22:55.032616 Step 36, training loss 872374.0625

2020-11-15 23:22:58.497561 validation 38:
2020-11-15 23:22:58.497625 hitrate@20=0.3460, recall@20=0.0540, ndcg@20=0.0449
2020-11-15 23:22:58.497646 hitrate@50=0.5090, recall@50=0.1005, ndcg@50=0.0620

Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.63s/it]
2020-11-15 23:24:17.399402 Step 37, training loss 875345.125

2020-11-15 23:24:20.785447 validation 39:
2020-11-15 23:24:20.785511 hitrate@20=0.3490, recall@20=0.0541, ndcg@20=0.0450
2020-11-15 23:24:20.785531 hitrate@50=0.5130, recall@50=0.1026, ndcg@50=0.0632

Optimizing: 100%|███████████████████████████████| 30/30 [01:19<00:00,  2.65s/it]
2020-11-15 23:25:40.339350 Step 38, training loss 881997.875

2020-11-15 23:25:43.817361 validation 40:
2020-11-15 23:25:43.817425 hitrate@20=0.3500, recall@20=0.0551, ndcg@20=0.0452
2020-11-15 23:25:43.817444 hitrate@50=0.5240, recall@50=0.1046, ndcg@50=0.0638

Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.63s/it]
2020-11-15 23:27:02.765072 Step 39, training loss 882908.3125

2020-11-15 23:27:06.274391 validation 41:
2020-11-15 23:27:06.274455 hitrate@20=0.3470, recall@20=0.0537, ndcg@20=0.0461
2020-11-15 23:27:06.274475 hitrate@50=0.5290, recall@50=0.1044, ndcg@50=0.0651

Optimizing: 100%|███████████████████████████████| 30/30 [01:20<00:00,  2.69s/it]
2020-11-15 23:28:26.974646 Step 40, training loss 863820.3125

2020-11-15 23:28:30.508052 validation 42:
2020-11-15 23:28:30.508106 hitrate@20=0.3410, recall@20=0.0528, ndcg@20=0.0457
2020-11-15 23:28:30.508124 hitrate@50=0.5230, recall@50=0.1035, ndcg@50=0.0647

2020-11-15 23:28:30.508159 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [01:19<00:00,  2.65s/it]
2020-11-15 23:29:50.020238 Step 41, training loss 867555.25

2020-11-15 23:29:53.596634 validation 43:
2020-11-15 23:29:53.596693 hitrate@20=0.3530, recall@20=0.0552, ndcg@20=0.0468
2020-11-15 23:29:53.596712 hitrate@50=0.5090, recall@50=0.1017, ndcg@50=0.0642

2020-11-15 23:29:53.596749 the monitor loses its patience to 3!
Optimizing: 100%|███████████████████████████████| 30/30 [01:19<00:00,  2.65s/it]
2020-11-15 23:31:13.235554 Step 42, training loss 874743.0

2020-11-15 23:31:16.800774 validation 44:
2020-11-15 23:31:16.800833 hitrate@20=0.3400, recall@20=0.0544, ndcg@20=0.0462
2020-11-15 23:31:16.800855 hitrate@50=0.5220, recall@50=0.1041, ndcg@50=0.0649

Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.63s/it]
2020-11-15 23:32:35.711760 Step 43, training loss 876385.375

2020-11-15 23:32:39.265668 validation 45:
2020-11-15 23:32:39.265731 hitrate@20=0.3460, recall@20=0.0551, ndcg@20=0.0469
2020-11-15 23:32:39.265752 hitrate@50=0.5250, recall@50=0.1053, ndcg@50=0.0657

Optimizing: 100%|███████████████████████████████| 30/30 [01:20<00:00,  2.69s/it]
2020-11-15 23:34:00.031311 Step 44, training loss 859493.4375

2020-11-15 23:34:03.679530 validation 46:
2020-11-15 23:34:03.679606 hitrate@20=0.3530, recall@20=0.0556, ndcg@20=0.0474
2020-11-15 23:34:03.679628 hitrate@50=0.5240, recall@50=0.1030, ndcg@50=0.0651

2020-11-15 23:34:03.679663 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.62s/it]
2020-11-15 23:35:22.201807 Step 45, training loss 860684.375

2020-11-15 23:35:25.845399 validation 47:
2020-11-15 23:35:25.845457 hitrate@20=0.3460, recall@20=0.0550, ndcg@20=0.0471
2020-11-15 23:35:25.845484 hitrate@50=0.5250, recall@50=0.1055, ndcg@50=0.0656

Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.61s/it]
2020-11-15 23:36:44.249460 Step 46, training loss 867140.625

2020-11-15 23:36:47.877535 validation 48:
2020-11-15 23:36:47.877609 hitrate@20=0.3420, recall@20=0.0541, ndcg@20=0.0462
2020-11-15 23:36:47.877630 hitrate@50=0.5310, recall@50=0.1055, ndcg@50=0.0653

2020-11-15 23:36:47.877666 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.63s/it]
2020-11-15 23:38:06.749849 Step 47, training loss 868447.875

2020-11-15 23:38:10.456200 validation 49:
2020-11-15 23:38:10.456263 hitrate@20=0.3420, recall@20=0.0543, ndcg@20=0.0467
2020-11-15 23:38:10.456281 hitrate@50=0.5320, recall@50=0.1081, ndcg@50=0.0667

Optimizing: 100%|███████████████████████████████| 30/30 [01:19<00:00,  2.64s/it]
2020-11-15 23:39:29.685202 Step 48, training loss 854196.1875

2020-11-15 23:39:33.463242 validation 50:
2020-11-15 23:39:33.463308 hitrate@20=0.3570, recall@20=0.0567, ndcg@20=0.0484
2020-11-15 23:39:33.463330 hitrate@50=0.5280, recall@50=0.1084, ndcg@50=0.0674

Optimizing: 100%|███████████████████████████████| 30/30 [01:17<00:00,  2.60s/it]
2020-11-15 23:40:51.455093 Step 49, training loss 855792.75

2020-11-15 23:40:55.121129 validation 51:
2020-11-15 23:40:55.121190 hitrate@20=0.3550, recall@20=0.0560, ndcg@20=0.0487
2020-11-15 23:40:55.121211 hitrate@50=0.5340, recall@50=0.1090, ndcg@50=0.0683

Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.61s/it]
2020-11-15 23:42:13.467004 Step 50, training loss 863474.9375

2020-11-15 23:42:17.176400 validation 52:
2020-11-15 23:42:17.176463 hitrate@20=0.3600, recall@20=0.0579, ndcg@20=0.0493
2020-11-15 23:42:17.176482 hitrate@50=0.5330, recall@50=0.1101, ndcg@50=0.0685

Optimizing: 100%|███████████████████████████████| 30/30 [01:20<00:00,  2.67s/it]
2020-11-15 23:43:37.334924 Step 51, training loss 863365.0

2020-11-15 23:43:41.068386 validation 53:
2020-11-15 23:43:41.068438 hitrate@20=0.3540, recall@20=0.0569, ndcg@20=0.0490
2020-11-15 23:43:41.068456 hitrate@50=0.5380, recall@50=0.1111, ndcg@50=0.0690

Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.61s/it]
2020-11-15 23:44:59.264142 Step 52, training loss 850196.75

2020-11-15 23:45:03.148453 validation 54:
2020-11-15 23:45:03.148519 hitrate@20=0.3570, recall@20=0.0568, ndcg@20=0.0483
2020-11-15 23:45:03.148547 hitrate@50=0.5450, recall@50=0.1120, ndcg@50=0.0688

2020-11-15 23:45:03.148601 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.61s/it]
2020-11-15 23:46:21.390975 Step 53, training loss 851873.3125

2020-11-15 23:46:25.250639 validation 55:
2020-11-15 23:46:25.250725 hitrate@20=0.3640, recall@20=0.0573, ndcg@20=0.0492
2020-11-15 23:46:25.250749 hitrate@50=0.5400, recall@50=0.1116, ndcg@50=0.0692

Optimizing: 100%|███████████████████████████████| 30/30 [01:19<00:00,  2.64s/it]
2020-11-15 23:47:44.396595 Step 54, training loss 860414.375

2020-11-15 23:47:48.220769 validation 56:
2020-11-15 23:47:48.220854 hitrate@20=0.3710, recall@20=0.0593, ndcg@20=0.0507
2020-11-15 23:47:48.220876 hitrate@50=0.5380, recall@50=0.1104, ndcg@50=0.0696

Optimizing: 100%|███████████████████████████████| 30/30 [01:19<00:00,  2.65s/it]
2020-11-15 23:49:07.760605 Step 55, training loss 858589.0

2020-11-15 23:49:11.645747 validation 57:
2020-11-15 23:49:11.645828 hitrate@20=0.3680, recall@20=0.0592, ndcg@20=0.0512
2020-11-15 23:49:11.645849 hitrate@50=0.5420, recall@50=0.1107, ndcg@50=0.0701

Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.63s/it]
2020-11-15 23:50:30.535357 Step 56, training loss 847423.9375

2020-11-15 23:50:34.366156 validation 58:
2020-11-15 23:50:34.366214 hitrate@20=0.3650, recall@20=0.0586, ndcg@20=0.0510
2020-11-15 23:50:34.366233 hitrate@50=0.5420, recall@50=0.1107, ndcg@50=0.0703

2020-11-15 23:50:34.366269 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.62s/it]
2020-11-15 23:51:53.068145 Step 57, training loss 848262.75

2020-11-15 23:51:56.902149 validation 59:
2020-11-15 23:51:56.902211 hitrate@20=0.3710, recall@20=0.0590, ndcg@20=0.0510
2020-11-15 23:51:56.902231 hitrate@50=0.5420, recall@50=0.1108, ndcg@50=0.0701

2020-11-15 23:51:56.902266 the monitor loses its patience to 3!
Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.62s/it]
2020-11-15 23:53:15.485941 Step 58, training loss 856620.6875

2020-11-15 23:53:19.265438 validation 60:
2020-11-15 23:53:19.265521 hitrate@20=0.3650, recall@20=0.0578, ndcg@20=0.0504
2020-11-15 23:53:19.265547 hitrate@50=0.5490, recall@50=0.1118, ndcg@50=0.0703

2020-11-15 23:53:19.265602 the monitor loses its patience to 2!
Optimizing: 100%|███████████████████████████████| 30/30 [01:20<00:00,  2.67s/it]
2020-11-15 23:54:39.476900 Step 59, training loss 856239.3125

2020-11-15 23:54:43.461526 validation 61:
2020-11-15 23:54:43.461616 hitrate@20=0.3630, recall@20=0.0569, ndcg@20=0.0502
2020-11-15 23:54:43.461644 hitrate@50=0.5430, recall@50=0.1132, ndcg@50=0.0709

2020-11-15 23:54:43.461689 the monitor loses its patience to 1!
Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.62s/it]
2020-11-15 23:56:02.153759 Step 60, training loss 844652.25

2020-11-15 23:56:06.228722 validation 62:
2020-11-15 23:56:06.228795 hitrate@20=0.3770, recall@20=0.0600, ndcg@20=0.0508
2020-11-15 23:56:06.228816 hitrate@50=0.5370, recall@50=0.1134, ndcg@50=0.0704

Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.61s/it]
2020-11-15 23:57:24.539714 Step 61, training loss 845778.75

2020-11-15 23:57:28.515648 validation 63:
2020-11-15 23:57:28.515720 hitrate@20=0.3760, recall@20=0.0608, ndcg@20=0.0519
2020-11-15 23:57:28.515744 hitrate@50=0.5360, recall@50=0.1133, ndcg@50=0.0712

Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.61s/it]
2020-11-15 23:58:46.894605 Step 62, training loss 853185.9375

2020-11-15 23:58:50.857084 validation 64:
2020-11-15 23:58:50.857142 hitrate@20=0.3690, recall@20=0.0600, ndcg@20=0.0518
2020-11-15 23:58:50.857172 hitrate@50=0.5520, recall@50=0.1159, ndcg@50=0.0725

Optimizing: 100%|███████████████████████████████| 30/30 [01:19<00:00,  2.66s/it]
2020-11-16 00:00:10.612473 Step 63, training loss 853135.9375

2020-11-16 00:00:14.644094 validation 65:
2020-11-16 00:00:14.644157 hitrate@20=0.3740, recall@20=0.0603, ndcg@20=0.0520
2020-11-16 00:00:14.644188 hitrate@50=0.5550, recall@50=0.1181, ndcg@50=0.0735

Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.61s/it]
2020-11-16 00:01:32.967986 Step 64, training loss 842230.3125

2020-11-16 00:01:36.957993 validation 66:
2020-11-16 00:01:36.958046 hitrate@20=0.3770, recall@20=0.0607, ndcg@20=0.0523
2020-11-16 00:01:36.958064 hitrate@50=0.5500, recall@50=0.1157, ndcg@50=0.0725

2020-11-16 00:01:36.958100 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.62s/it]
2020-11-16 00:02:55.456319 Step 65, training loss 842829.25

2020-11-16 00:02:59.321926 validation 67:
2020-11-16 00:02:59.321985 hitrate@20=0.3690, recall@20=0.0603, ndcg@20=0.0524
2020-11-16 00:02:59.322005 hitrate@50=0.5550, recall@50=0.1162, ndcg@50=0.0732

2020-11-16 00:02:59.322038 the monitor loses its patience to 3!
Optimizing: 100%|███████████████████████████████| 30/30 [01:18<00:00,  2.62s/it]
2020-11-16 00:04:17.977031 Step 66, training loss 850847.125

2020-11-16 00:04:21.916353 validation 68:
2020-11-16 00:04:21.916398 hitrate@20=0.3680, recall@20=0.0600, ndcg@20=0.0518
2020-11-16 00:04:21.916416 hitrate@50=0.5590, recall@50=0.1176, ndcg@50=0.0734

2020-11-16 00:04:21.916448 the monitor loses its patience to 2!
Optimizing: 100%|███████████████████████████████| 30/30 [01:19<00:00,  2.66s/it]
2020-11-16 00:05:41.765746 Step 67, training loss 851051.125

2020-11-16 00:05:45.886095 validation 69:
2020-11-16 00:05:45.886165 hitrate@20=0.3780, recall@20=0.0612, ndcg@20=0.0527
2020-11-16 00:05:45.886187 hitrate@50=0.5700, recall@50=0.1210, ndcg@50=0.0745

Optimizing: 100%|███████████████████████████████| 30/30 [01:20<00:00,  2.67s/it]
2020-11-16 00:07:06.119773 Step 68, training loss 842527.875

2020-11-16 00:07:10.282834 validation 70:
2020-11-16 00:07:10.282890 hitrate@20=0.3810, recall@20=0.0622, ndcg@20=0.0536
2020-11-16 00:07:10.282909 hitrate@50=0.5510, recall@50=0.1181, ndcg@50=0.0738

2020-11-16 00:07:10.282957 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [01:17<00:00,  2.59s/it]
2020-11-16 00:08:27.933520 Step 69, training loss 841645.3125

2020-11-16 00:08:32.041134 validation 71:
2020-11-16 00:08:32.041194 hitrate@20=0.3820, recall@20=0.0611, ndcg@20=0.0528
2020-11-16 00:08:32.041214 hitrate@50=0.5570, recall@50=0.1187, ndcg@50=0.0738

2020-11-16 00:08:32.041247 the monitor loses its patience to 3!
Optimizing: 100%|███████████████████████████████| 30/30 [01:17<00:00,  2.59s/it]
2020-11-16 00:09:49.724574 Step 70, training loss 849187.875

2020-11-16 00:09:53.837700 validation 72:
2020-11-16 00:09:53.837756 hitrate@20=0.3800, recall@20=0.0608, ndcg@20=0.0529
2020-11-16 00:09:53.837775 hitrate@50=0.5600, recall@50=0.1180, ndcg@50=0.0740

2020-11-16 00:09:53.837810 the monitor loses its patience to 2!
Optimizing: 100%|███████████████████████████████| 30/30 [01:19<00:00,  2.65s/it]
2020-11-16 00:11:13.231853 Step 71, training loss 847833.0

2020-11-16 00:11:17.255423 validation 73:
2020-11-16 00:11:17.255471 hitrate@20=0.3730, recall@20=0.0593, ndcg@20=0.0526
2020-11-16 00:11:17.255490 hitrate@50=0.5630, recall@50=0.1180, ndcg@50=0.0746

2020-11-16 00:11:17.255526 the monitor loses its patience to 1!
Optimizing: 100%|███████████████████████████████| 30/30 [01:17<00:00,  2.59s/it]
2020-11-16 00:12:35.012518 Step 72, training loss 839123.6875

2020-11-16 00:12:39.026752 validation 74:
2020-11-16 00:12:39.026814 hitrate@20=0.3840, recall@20=0.0624, ndcg@20=0.0527
2020-11-16 00:12:39.026836 hitrate@50=0.5640, recall@50=0.1187, ndcg@50=0.0736

2020-11-16 00:12:39.026870 the monitor loses its patience to 0!
2020-11-16 00:12:39.026884 early stop at step 73
2020-11-16 00:12:39.489242 close sampler, close and save to log file
2020-11-16 00:12:39.622375 log file and sampler have closed
```

