## CML_amazonbooks_x0 

A notebook to benchmark CML on amazonbooks_x0 dataset.

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

```python
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
dataset = "amazonbooks_x0"
train_data = "../../data/AmazonBooks/amazonbooks_x0/train.txt"
test_data = "../../data/AmazonBooks/amazonbooks_x0/test.txt"
batch_size = 100000
max_step=5000
num_negative=75
embed_dim=100
margin=1.8
clip_norm=1
lr = 0.001
dropout = 0.3 # dropout rate
verbose = 30 # Evaluation interval
topK = "20 50" # Metrics at TopK
```

```bash
python -u main.py --gpu 1 --dataset amazonbooks_x0 --train_data ../../data/AmazonBooks/amazonbooks_x0/train.txt --test_data ../../data/AmazonBooks/amazonbooks_x0/test.txt --verbose 30 --batch_size 100000 --max_step 5000 --embed_dim 100 --lr 0.001 --dropout 0.3 --margin 1.8 --clip_norm 1 --num_negative 75 --topK 20 50
```

### Results
```bash
2020-11-14 00:49:42.400649 validation 73:
2020-11-14 00:49:42.400738 hitrate@20=0.2840, recall@20=0.0522, ndcg@20=0.0428
2020-11-14 00:49:42.400765 hitrate@50=0.4410, recall@50=0.0953, ndcg@50=0.0591
```

### Logs
```bash
2020-11-13 21:54:17.845959 reading dataset amazonbooks_x0
2020-11-13 21:55:13.317242 #users=52643, #items=91599
WARNING:tensorflow:From /home/wjp/anaconda3/envs/py37torch/lib/python3.7/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-11-13 21:55:13.852340: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
2020-11-13 21:55:13.907419: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2799775000 Hz
2020-11-13 21:55:13.910562: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d3348daa00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-11-13 21:55:13.913647: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-11-13 21:55:13.932724: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-11-13 21:55:14.591535: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:42:00.0
2020-11-13 21:55:14.663319: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-11-13 21:55:14.918653: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-11-13 21:55:15.075901: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2020-11-13 21:55:15.130870: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2020-11-13 21:55:15.445260: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2020-11-13 21:55:15.648400: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2020-11-13 21:55:16.135453: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-11-13 21:55:16.137414: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-11-13 21:55:16.144566: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-11-13 21:55:16.330264: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-11-13 21:55:16.330392: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
2020-11-13 21:55:16.330426: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
2020-11-13 21:55:16.333235: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10481 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:42:00.0, compute capability: 6.1)
2020-11-13 21:55:16.351763: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d3348e9b70 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-11-13 21:55:16.351905: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce GTX 1080 Ti, Compute Capability 6.1
2020-11-13 21:55:27.080885 validation 1:
2020-11-13 21:55:27.080979 hitrate@20=0.0020, recall@20=0.0002, ndcg@20=0.0001
2020-11-13 21:55:27.081006 hitrate@50=0.0060, recall@50=0.0006, ndcg@50=0.0003

Optimizing: 100%|███████████████████████████████| 30/30 [02:15<00:00,  4.51s/it]
2020-11-13 21:57:42.516561 Step 0, training loss 2503220.25

2020-11-13 21:57:47.351114 validation 2:
2020-11-13 21:57:47.351174 hitrate@20=0.0060, recall@20=0.0007, ndcg@20=0.0006
2020-11-13 21:57:47.351216 hitrate@50=0.0170, recall@50=0.0018, ndcg@50=0.0011

Optimizing: 100%|███████████████████████████████| 30/30 [02:03<00:00,  4.12s/it]
2020-11-13 21:59:50.839197 Step 1, training loss 2444477.0

2020-11-13 21:59:55.456458 validation 3:
2020-11-13 21:59:55.456516 hitrate@20=0.0110, recall@20=0.0011, ndcg@20=0.0008
2020-11-13 21:59:55.456551 hitrate@50=0.0250, recall@50=0.0027, ndcg@50=0.0014

Optimizing: 100%|███████████████████████████████| 30/30 [02:03<00:00,  4.13s/it]
2020-11-13 22:01:59.423577 Step 2, training loss 2408391.25

2020-11-13 22:02:04.315092 validation 4:
2020-11-13 22:02:04.315154 hitrate@20=0.0160, recall@20=0.0013, ndcg@20=0.0010
2020-11-13 22:02:04.315174 hitrate@50=0.0310, recall@50=0.0030, ndcg@50=0.0017

Optimizing: 100%|███████████████████████████████| 30/30 [02:06<00:00,  4.23s/it]
2020-11-13 22:04:11.162612 Step 3, training loss 2363802.75

2020-11-13 22:04:15.870624 validation 5:
2020-11-13 22:04:15.870681 hitrate@20=0.0160, recall@20=0.0014, ndcg@20=0.0010
2020-11-13 22:04:15.870700 hitrate@50=0.0270, recall@50=0.0027, ndcg@50=0.0015

2020-11-13 22:04:15.870736 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [02:05<00:00,  4.18s/it]
2020-11-13 22:06:21.217179 Step 4, training loss 2281225.25

2020-11-13 22:06:26.240691 validation 6:
2020-11-13 22:06:26.240774 hitrate@20=0.0130, recall@20=0.0010, ndcg@20=0.0009
2020-11-13 22:06:26.240800 hitrate@50=0.0330, recall@50=0.0034, ndcg@50=0.0018

Optimizing: 100%|███████████████████████████████| 30/30 [02:05<00:00,  4.17s/it]
2020-11-13 22:08:31.295451 Step 5, training loss 2265676.5

2020-11-13 22:08:35.532485 validation 7:
2020-11-13 22:08:35.532563 hitrate@20=0.0170, recall@20=0.0015, ndcg@20=0.0012
2020-11-13 22:08:35.532602 hitrate@50=0.0340, recall@50=0.0035, ndcg@50=0.0019

Optimizing: 100%|███████████████████████████████| 30/30 [02:06<00:00,  4.23s/it]
2020-11-13 22:10:42.511306 Step 6, training loss 2250280.75

2020-11-13 22:10:47.613129 validation 8:
2020-11-13 22:10:47.613189 hitrate@20=0.0150, recall@20=0.0016, ndcg@20=0.0013
2020-11-13 22:10:47.613227 hitrate@50=0.0280, recall@50=0.0027, ndcg@50=0.0017

2020-11-13 22:10:47.613262 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [02:06<00:00,  4.21s/it]
2020-11-13 22:12:53.892586 Step 7, training loss 2216637.75

2020-11-13 22:12:58.678994 validation 9:
2020-11-13 22:12:58.679053 hitrate@20=0.0210, recall@20=0.0020, ndcg@20=0.0019
2020-11-13 22:12:58.679074 hitrate@50=0.0350, recall@50=0.0036, ndcg@50=0.0025

Optimizing: 100%|███████████████████████████████| 30/30 [02:04<00:00,  4.14s/it]
2020-11-13 22:15:02.940324 Step 8, training loss 2170043.0

2020-11-13 22:15:07.625091 validation 10:
2020-11-13 22:15:07.625165 hitrate@20=0.0260, recall@20=0.0022, ndcg@20=0.0022
2020-11-13 22:15:07.625189 hitrate@50=0.0420, recall@50=0.0038, ndcg@50=0.0027

Optimizing: 100%|███████████████████████████████| 30/30 [02:05<00:00,  4.18s/it]
2020-11-13 22:17:12.979655 Step 9, training loss 2163723.0

2020-11-13 22:17:17.366223 validation 11:
2020-11-13 22:17:17.366280 hitrate@20=0.0230, recall@20=0.0023, ndcg@20=0.0025
2020-11-13 22:17:17.366316 hitrate@50=0.0360, recall@50=0.0034, ndcg@50=0.0028

2020-11-13 22:17:17.366350 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [02:12<00:00,  4.41s/it]
2020-11-13 22:19:29.541040 Step 10, training loss 2155276.75

2020-11-13 22:19:34.581481 validation 12:
2020-11-13 22:19:34.581547 hitrate@20=0.0300, recall@20=0.0031, ndcg@20=0.0032
2020-11-13 22:19:34.581569 hitrate@50=0.0480, recall@50=0.0047, ndcg@50=0.0036

Optimizing: 100%|███████████████████████████████| 30/30 [02:15<00:00,  4.53s/it]
2020-11-13 22:21:50.476263 Step 11, training loss 2123112.75

2020-11-13 22:21:55.000440 validation 13:
2020-11-13 22:21:55.000504 hitrate@20=0.0410, recall@20=0.0037, ndcg@20=0.0039
2020-11-13 22:21:55.000525 hitrate@50=0.0650, recall@50=0.0068, ndcg@50=0.0050

Optimizing: 100%|███████████████████████████████| 30/30 [02:11<00:00,  4.38s/it]
2020-11-13 22:24:06.427074 Step 12, training loss 2094263.25

2020-11-13 22:24:11.819523 validation 14:
2020-11-13 22:24:11.819599 hitrate@20=0.0540, recall@20=0.0052, ndcg@20=0.0053
2020-11-13 22:24:11.819631 hitrate@50=0.0840, recall@50=0.0095, ndcg@50=0.0067

Optimizing: 100%|███████████████████████████████| 30/30 [02:11<00:00,  4.39s/it]
2020-11-13 22:26:23.530834 Step 13, training loss 2090831.75

2020-11-13 22:26:28.193041 validation 15:
2020-11-13 22:26:28.193143 hitrate@20=0.0570, recall@20=0.0057, ndcg@20=0.0053
2020-11-13 22:26:28.193171 hitrate@50=0.0980, recall@50=0.0116, ndcg@50=0.0075

Optimizing: 100%|███████████████████████████████| 30/30 [02:12<00:00,  4.41s/it]
2020-11-13 22:28:40.469487 Step 14, training loss 2085638.75

2020-11-13 22:28:46.010813 validation 16:
2020-11-13 22:28:46.010883 hitrate@20=0.0770, recall@20=0.0076, ndcg@20=0.0069
2020-11-13 22:28:46.010907 hitrate@50=0.1290, recall@50=0.0153, ndcg@50=0.0099

Optimizing: 100%|███████████████████████████████| 30/30 [02:15<00:00,  4.52s/it]
2020-11-13 22:31:01.490585 Step 15, training loss 2049190.0

2020-11-13 22:31:06.190883 validation 17:
2020-11-13 22:31:06.190942 hitrate@20=0.0850, recall@20=0.0092, ndcg@20=0.0085
2020-11-13 22:31:06.190961 hitrate@50=0.1380, recall@50=0.0182, ndcg@50=0.0118

Optimizing: 100%|███████████████████████████████| 30/30 [02:12<00:00,  4.43s/it]
2020-11-13 22:33:19.007288 Step 16, training loss 2030420.125

2020-11-13 22:33:24.774343 validation 18:
2020-11-13 22:33:24.774407 hitrate@20=0.0920, recall@20=0.0110, ndcg@20=0.0102
2020-11-13 22:33:24.774431 hitrate@50=0.1610, recall@50=0.0217, ndcg@50=0.0145

Optimizing: 100%|███████████████████████████████| 30/30 [02:11<00:00,  4.37s/it]
2020-11-13 22:35:35.925926 Step 17, training loss 2027048.0

2020-11-13 22:35:41.503208 validation 19:
2020-11-13 22:35:41.503276 hitrate@20=0.1040, recall@20=0.0137, ndcg@20=0.0127
2020-11-13 22:35:41.503297 hitrate@50=0.1770, recall@50=0.0259, ndcg@50=0.0175

Optimizing: 100%|███████████████████████████████| 30/30 [02:11<00:00,  4.38s/it]
2020-11-13 22:37:52.896431 Step 18, training loss 2022139.125

2020-11-13 22:37:58.033805 validation 20:
2020-11-13 22:37:58.033953 hitrate@20=0.1230, recall@20=0.0168, ndcg@20=0.0150
2020-11-13 22:37:58.033981 hitrate@50=0.2060, recall@50=0.0303, ndcg@50=0.0200

Optimizing: 100%|███████████████████████████████| 30/30 [02:15<00:00,  4.52s/it]
2020-11-13 22:40:13.606710 Step 19, training loss 1974066.75

2020-11-13 22:40:18.709190 validation 21:
2020-11-13 22:40:18.709316 hitrate@20=0.1330, recall@20=0.0191, ndcg@20=0.0166
2020-11-13 22:40:18.709345 hitrate@50=0.2310, recall@50=0.0349, ndcg@50=0.0226

Optimizing: 100%|███████████████████████████████| 30/30 [02:11<00:00,  4.39s/it]
2020-11-13 22:42:30.397991 Step 20, training loss 1956912.0

2020-11-13 22:42:35.737008 validation 22:
2020-11-13 22:42:35.737067 hitrate@20=0.1470, recall@20=0.0204, ndcg@20=0.0183
2020-11-13 22:42:35.737088 hitrate@50=0.2460, recall@50=0.0384, ndcg@50=0.0252

Optimizing: 100%|███████████████████████████████| 30/30 [02:11<00:00,  4.40s/it]
2020-11-13 22:44:47.716795 Step 21, training loss 1952649.375

2020-11-13 22:44:52.496296 validation 23:
2020-11-13 22:44:52.496357 hitrate@20=0.1680, recall@20=0.0247, ndcg@20=0.0206
2020-11-13 22:44:52.496389 hitrate@50=0.2740, recall@50=0.0438, ndcg@50=0.0281

Optimizing: 100%|███████████████████████████████| 30/30 [02:13<00:00,  4.47s/it]
2020-11-13 22:47:06.460488 Step 22, training loss 1945504.75

2020-11-13 22:47:12.731838 validation 24:
2020-11-13 22:47:12.731899 hitrate@20=0.1680, recall@20=0.0273, ndcg@20=0.0232
2020-11-13 22:47:12.731919 hitrate@50=0.2720, recall@50=0.0464, ndcg@50=0.0306

Optimizing: 100%|███████████████████████████████| 30/30 [02:12<00:00,  4.41s/it]
2020-11-13 22:49:24.990232 Step 23, training loss 1888504.875

2020-11-13 22:49:30.026747 validation 25:
2020-11-13 22:49:30.026841 hitrate@20=0.1710, recall@20=0.0282, ndcg@20=0.0234
2020-11-13 22:49:30.026869 hitrate@50=0.2980, recall@50=0.0510, ndcg@50=0.0325

Optimizing: 100%|███████████████████████████████| 30/30 [02:12<00:00,  4.43s/it]
2020-11-13 22:51:42.940912 Step 24, training loss 1883347.875

2020-11-13 22:51:48.512555 validation 26:
2020-11-13 22:51:48.512615 hitrate@20=0.1900, recall@20=0.0311, ndcg@20=0.0262
2020-11-13 22:51:48.512642 hitrate@50=0.3050, recall@50=0.0528, ndcg@50=0.0346

Optimizing: 100%|███████████████████████████████| 30/30 [02:11<00:00,  4.39s/it]
2020-11-13 22:54:00.309355 Step 25, training loss 1879459.875

2020-11-13 22:54:05.507778 validation 27:
2020-11-13 22:54:05.507845 hitrate@20=0.1980, recall@20=0.0335, ndcg@20=0.0277
2020-11-13 22:54:05.507866 hitrate@50=0.3210, recall@50=0.0581, ndcg@50=0.0371

Optimizing: 100%|███████████████████████████████| 30/30 [02:15<00:00,  4.51s/it]
2020-11-13 22:56:20.734534 Step 26, training loss 1864522.5

2020-11-13 22:56:26.458772 validation 28:
2020-11-13 22:56:26.458843 hitrate@20=0.2090, recall@20=0.0359, ndcg@20=0.0291
2020-11-13 22:56:26.458865 hitrate@50=0.3340, recall@50=0.0611, ndcg@50=0.0387

Optimizing: 100%|███████████████████████████████| 30/30 [02:11<00:00,  4.40s/it]
2020-11-13 22:58:38.398023 Step 27, training loss 1809161.75

2020-11-13 22:58:43.419833 validation 29:
2020-11-13 22:58:43.419895 hitrate@20=0.2080, recall@20=0.0370, ndcg@20=0.0300
2020-11-13 22:58:43.419917 hitrate@50=0.3440, recall@50=0.0638, ndcg@50=0.0404

Optimizing: 100%|███████████████████████████████| 30/30 [02:13<00:00,  4.45s/it]
2020-11-13 23:00:57.065474 Step 28, training loss 1813805.875

2020-11-13 23:01:02.698068 validation 30:
2020-11-13 23:01:02.698144 hitrate@20=0.2010, recall@20=0.0358, ndcg@20=0.0304
2020-11-13 23:01:02.698174 hitrate@50=0.3440, recall@50=0.0654, ndcg@50=0.0418

Optimizing: 100%|███████████████████████████████| 30/30 [02:12<00:00,  4.42s/it]
2020-11-13 23:03:15.239856 Step 29, training loss 1820098.75

2020-11-13 23:03:20.596316 validation 31:
2020-11-13 23:03:20.596390 hitrate@20=0.2170, recall@20=0.0373, ndcg@20=0.0318
2020-11-13 23:03:20.596423 hitrate@50=0.3670, recall@50=0.0693, ndcg@50=0.0440

Optimizing: 100%|███████████████████████████████| 30/30 [03:39<00:00,  7.31s/it]
2020-11-13 23:06:59.945672 Step 30, training loss 1801850.25

2020-11-13 23:07:13.346446 validation 32:
2020-11-13 23:07:13.346569 hitrate@20=0.2270, recall@20=0.0391, ndcg@20=0.0331
2020-11-13 23:07:13.346611 hitrate@50=0.3650, recall@50=0.0692, ndcg@50=0.0447

Optimizing: 100%|███████████████████████████████| 30/30 [03:44<00:00,  7.48s/it]
2020-11-13 23:10:57.906634 Step 31, training loss 1752350.5

2020-11-13 23:11:14.402331 validation 33:
2020-11-13 23:11:14.402608 hitrate@20=0.2270, recall@20=0.0401, ndcg@20=0.0344
2020-11-13 23:11:14.402680 hitrate@50=0.3830, recall@50=0.0735, ndcg@50=0.0469

Optimizing: 100%|███████████████████████████████| 30/30 [03:49<00:00,  7.64s/it]
2020-11-13 23:15:03.618375 Step 32, training loss 1755974.125

2020-11-13 23:15:18.648311 validation 34:
2020-11-13 23:15:18.648520 hitrate@20=0.2310, recall@20=0.0398, ndcg@20=0.0346
2020-11-13 23:15:18.648566 hitrate@50=0.3790, recall@50=0.0748, ndcg@50=0.0480

Optimizing: 100%|███████████████████████████████| 30/30 [03:42<00:00,  7.42s/it]
2020-11-13 23:19:01.160950 Step 33, training loss 1763549.875

2020-11-13 23:19:16.029712 validation 35:
2020-11-13 23:19:16.029964 hitrate@20=0.2380, recall@20=0.0408, ndcg@20=0.0349
2020-11-13 23:19:16.030032 hitrate@50=0.3840, recall@50=0.0759, ndcg@50=0.0481

Optimizing: 100%|███████████████████████████████| 30/30 [03:57<00:00,  7.92s/it]
2020-11-13 23:23:13.623537 Step 34, training loss 1731235.375

2020-11-13 23:23:27.788793 validation 36:
2020-11-13 23:23:27.789075 hitrate@20=0.2350, recall@20=0.0403, ndcg@20=0.0350
2020-11-13 23:23:27.789136 hitrate@50=0.3820, recall@50=0.0769, ndcg@50=0.0488

Optimizing: 100%|███████████████████████████████| 30/30 [03:26<00:00,  6.88s/it]
2020-11-13 23:26:54.308864 Step 35, training loss 1717289.5

2020-11-13 23:27:04.654122 validation 37:
2020-11-13 23:27:04.654225 hitrate@20=0.2340, recall@20=0.0406, ndcg@20=0.0354
2020-11-13 23:27:04.654250 hitrate@50=0.3900, recall@50=0.0770, ndcg@50=0.0490

Optimizing: 100%|███████████████████████████████| 30/30 [03:16<00:00,  6.57s/it]
2020-11-13 23:30:21.632919 Step 36, training loss 1726557.625

2020-11-13 23:30:35.794482 validation 38:
2020-11-13 23:30:35.795160 hitrate@20=0.2370, recall@20=0.0409, ndcg@20=0.0349
2020-11-13 23:30:35.795355 hitrate@50=0.3790, recall@50=0.0766, ndcg@50=0.0486

2020-11-13 23:30:35.795732 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [02:23<00:00,  4.77s/it]
2020-11-13 23:32:58.912854 Step 37, training loss 1721274.875

2020-11-13 23:33:05.345337 validation 39:
2020-11-13 23:33:05.345432 hitrate@20=0.2350, recall@20=0.0409, ndcg@20=0.0354
2020-11-13 23:33:05.345459 hitrate@50=0.3830, recall@50=0.0767, ndcg@50=0.0492

Optimizing: 100%|███████████████████████████████| 30/30 [02:02<00:00,  4.10s/it]
2020-11-13 23:35:08.340139 Step 38, training loss 1700866.5

2020-11-13 23:35:14.068774 validation 40:
2020-11-13 23:35:14.068901 hitrate@20=0.2300, recall@20=0.0403, ndcg@20=0.0355
2020-11-13 23:35:14.068930 hitrate@50=0.3930, recall@50=0.0796, ndcg@50=0.0504

Optimizing: 100%|███████████████████████████████| 30/30 [02:07<00:00,  4.24s/it]
2020-11-13 23:37:21.396795 Step 39, training loss 1693472.375

2020-11-13 23:37:26.605148 validation 41:
2020-11-13 23:37:26.605261 hitrate@20=0.2360, recall@20=0.0419, ndcg@20=0.0361
2020-11-13 23:37:26.605283 hitrate@50=0.3960, recall@50=0.0816, ndcg@50=0.0510

Optimizing: 100%|███████████████████████████████| 30/30 [02:05<00:00,  4.17s/it]
2020-11-13 23:39:31.741254 Step 40, training loss 1699395.5

2020-11-13 23:39:37.804143 validation 42:
2020-11-13 23:39:37.804231 hitrate@20=0.2520, recall@20=0.0442, ndcg@20=0.0365
2020-11-13 23:39:37.804257 hitrate@50=0.3940, recall@50=0.0804, ndcg@50=0.0503

Optimizing: 100%|███████████████████████████████| 30/30 [02:04<00:00,  4.14s/it]
2020-11-13 23:41:42.110866 Step 41, training loss 1691174.75

2020-11-13 23:41:47.499237 validation 43:
2020-11-13 23:41:47.499288 hitrate@20=0.2530, recall@20=0.0450, ndcg@20=0.0377
2020-11-13 23:41:47.499314 hitrate@50=0.4100, recall@50=0.0852, ndcg@50=0.0528

Optimizing: 100%|███████████████████████████████| 30/30 [02:05<00:00,  4.18s/it]
2020-11-13 23:43:52.878595 Step 42, training loss 1678006.125

2020-11-13 23:43:59.048078 validation 44:
2020-11-13 23:43:59.048147 hitrate@20=0.2630, recall@20=0.0464, ndcg@20=0.0381
2020-11-13 23:43:59.048184 hitrate@50=0.4020, recall@50=0.0835, ndcg@50=0.0521

2020-11-13 23:43:59.048233 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [02:01<00:00,  4.06s/it]
2020-11-13 23:46:00.970352 Step 43, training loss 1672432.75

2020-11-13 23:46:06.371344 validation 45:
2020-11-13 23:46:06.371407 hitrate@20=0.2600, recall@20=0.0467, ndcg@20=0.0379
2020-11-13 23:46:06.371430 hitrate@50=0.4100, recall@50=0.0847, ndcg@50=0.0521

Optimizing: 100%|███████████████████████████████| 30/30 [02:02<00:00,  4.08s/it]
2020-11-13 23:48:08.704883 Step 44, training loss 1678216.375

2020-11-13 23:48:15.540357 validation 46:
2020-11-13 23:48:15.540416 hitrate@20=0.2550, recall@20=0.0454, ndcg@20=0.0376
2020-11-13 23:48:15.540437 hitrate@50=0.4120, recall@50=0.0848, ndcg@50=0.0525

2020-11-13 23:48:15.540470 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [02:07<00:00,  4.25s/it]
2020-11-13 23:50:22.933978 Step 45, training loss 1665344.625

2020-11-13 23:50:28.623054 validation 47:
2020-11-13 23:50:28.623134 hitrate@20=0.2540, recall@20=0.0455, ndcg@20=0.0380
2020-11-13 23:50:28.623189 hitrate@50=0.4240, recall@50=0.0859, ndcg@50=0.0532

Optimizing: 100%|███████████████████████████████| 30/30 [02:05<00:00,  4.19s/it]
2020-11-13 23:52:34.385536 Step 46, training loss 1661887.125

2020-11-13 23:52:41.080060 validation 48:
2020-11-13 23:52:41.080118 hitrate@20=0.2530, recall@20=0.0454, ndcg@20=0.0378
2020-11-13 23:52:41.080158 hitrate@50=0.4190, recall@50=0.0856, ndcg@50=0.0528

2020-11-13 23:52:41.080208 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [02:05<00:00,  4.17s/it]
2020-11-13 23:54:46.138834 Step 47, training loss 1654917.125

2020-11-13 23:54:51.637496 validation 49:
2020-11-13 23:54:51.637576 hitrate@20=0.2500, recall@20=0.0455, ndcg@20=0.0389
2020-11-13 23:54:51.637596 hitrate@50=0.4190, recall@50=0.0861, ndcg@50=0.0541

Optimizing: 100%|███████████████████████████████| 30/30 [02:02<00:00,  4.07s/it]
2020-11-13 23:56:53.814010 Step 48, training loss 1664587.75

2020-11-13 23:57:00.791690 validation 50:
2020-11-13 23:57:00.791769 hitrate@20=0.2580, recall@20=0.0468, ndcg@20=0.0391
2020-11-13 23:57:00.791795 hitrate@50=0.4260, recall@50=0.0869, ndcg@50=0.0542

Optimizing: 100%|███████████████████████████████| 30/30 [02:04<00:00,  4.15s/it]
2020-11-13 23:59:05.211070 Step 49, training loss 1649552.375

2020-11-13 23:59:12.160986 validation 51:
2020-11-13 23:59:12.161050 hitrate@20=0.2630, recall@20=0.0474, ndcg@20=0.0391
2020-11-13 23:59:12.161099 hitrate@50=0.4180, recall@50=0.0868, ndcg@50=0.0539

Optimizing: 100%|███████████████████████████████| 30/30 [02:02<00:00,  4.10s/it]
2020-11-14 00:01:15.121962 Step 50, training loss 1648895.5

2020-11-14 00:01:22.659285 validation 52:
2020-11-14 00:01:22.659349 hitrate@20=0.2650, recall@20=0.0481, ndcg@20=0.0391
2020-11-14 00:01:22.659379 hitrate@50=0.4290, recall@50=0.0875, ndcg@50=0.0539

Optimizing: 100%|███████████████████████████████| 30/30 [02:05<00:00,  4.20s/it]
2020-11-14 00:03:28.607148 Step 51, training loss 1642485.25

2020-11-14 00:03:35.776200 validation 53:
2020-11-14 00:03:35.776261 hitrate@20=0.2580, recall@20=0.0467, ndcg@20=0.0384
2020-11-14 00:03:35.776283 hitrate@50=0.4250, recall@50=0.0883, ndcg@50=0.0541

2020-11-14 00:03:35.776331 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [02:13<00:00,  4.46s/it]
2020-11-14 00:05:49.492653 Step 52, training loss 1651523.25

2020-11-14 00:05:56.439951 validation 54:
2020-11-14 00:05:56.440024 hitrate@20=0.2660, recall@20=0.0481, ndcg@20=0.0394
2020-11-14 00:05:56.440052 hitrate@50=0.4310, recall@50=0.0885, ndcg@50=0.0545

Optimizing: 100%|███████████████████████████████| 30/30 [02:10<00:00,  4.35s/it]
2020-11-14 00:08:06.951918 Step 53, training loss 1636836.0

2020-11-14 00:08:12.908768 validation 55:
2020-11-14 00:08:12.908844 hitrate@20=0.2660, recall@20=0.0486, ndcg@20=0.0393
2020-11-14 00:08:12.908870 hitrate@50=0.4340, recall@50=0.0875, ndcg@50=0.0540

2020-11-14 00:08:12.908945 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [02:12<00:00,  4.41s/it]
2020-11-14 00:10:25.096171 Step 54, training loss 1635921.75

2020-11-14 00:10:31.913705 validation 56:
2020-11-14 00:10:31.913775 hitrate@20=0.2670, recall@20=0.0478, ndcg@20=0.0393
2020-11-14 00:10:31.913799 hitrate@50=0.4370, recall@50=0.0890, ndcg@50=0.0546

Optimizing: 100%|███████████████████████████████| 30/30 [02:11<00:00,  4.39s/it]
2020-11-14 00:12:43.756683 Step 55, training loss 1634108.125

2020-11-14 00:12:49.760752 validation 57:
2020-11-14 00:12:49.760839 hitrate@20=0.2600, recall@20=0.0485, ndcg@20=0.0395
2020-11-14 00:12:49.760867 hitrate@50=0.4410, recall@50=0.0905, ndcg@50=0.0554

Optimizing: 100%|███████████████████████████████| 30/30 [02:14<00:00,  4.47s/it]
2020-11-14 00:15:03.852805 Step 56, training loss 1640507.25

2020-11-14 00:15:10.208372 validation 58:
2020-11-14 00:15:10.208436 hitrate@20=0.2670, recall@20=0.0480, ndcg@20=0.0400
2020-11-14 00:15:10.208459 hitrate@50=0.4410, recall@50=0.0920, ndcg@50=0.0563

Optimizing: 100%|███████████████████████████████| 30/30 [02:12<00:00,  4.41s/it]
2020-11-14 00:17:22.454358 Step 57, training loss 1627962.875

2020-11-14 00:17:28.260355 validation 59:
2020-11-14 00:17:28.260424 hitrate@20=0.2690, recall@20=0.0480, ndcg@20=0.0397
2020-11-14 00:17:28.260446 hitrate@50=0.4510, recall@50=0.0932, ndcg@50=0.0566

Optimizing: 100%|███████████████████████████████| 30/30 [02:12<00:00,  4.40s/it]
2020-11-14 00:19:40.392997 Step 58, training loss 1626636.125

2020-11-14 00:19:46.621742 validation 60:
2020-11-14 00:19:46.621815 hitrate@20=0.2730, recall@20=0.0482, ndcg@20=0.0397
2020-11-14 00:19:46.621842 hitrate@50=0.4360, recall@50=0.0900, ndcg@50=0.0554

2020-11-14 00:19:46.621878 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [02:11<00:00,  4.40s/it]
2020-11-14 00:21:58.603466 Step 59, training loss 1629947.625

2020-11-14 00:22:04.316201 validation 61:
2020-11-14 00:22:04.316263 hitrate@20=0.2780, recall@20=0.0490, ndcg@20=0.0403
2020-11-14 00:22:04.316286 hitrate@50=0.4490, recall@50=0.0913, ndcg@50=0.0561

2020-11-14 00:22:04.316321 the monitor loses its patience to 3!
Optimizing: 100%|███████████████████████████████| 30/30 [02:16<00:00,  4.56s/it]
2020-11-14 00:24:21.150049 Step 60, training loss 1631598.0

2020-11-14 00:24:28.107070 validation 62:
2020-11-14 00:24:28.107137 hitrate@20=0.2800, recall@20=0.0505, ndcg@20=0.0405
2020-11-14 00:24:28.107160 hitrate@50=0.4410, recall@50=0.0918, ndcg@50=0.0560

Optimizing: 100%|███████████████████████████████| 30/30 [02:11<00:00,  4.38s/it]
2020-11-14 00:26:39.446688 Step 61, training loss 1620899.375

2020-11-14 00:26:45.611559 validation 63:
2020-11-14 00:26:45.611634 hitrate@20=0.2780, recall@20=0.0502, ndcg@20=0.0398
2020-11-14 00:26:45.611660 hitrate@50=0.4480, recall@50=0.0930, ndcg@50=0.0561

Optimizing: 100%|███████████████████████████████| 30/30 [02:10<00:00,  4.37s/it]
2020-11-14 00:28:56.581217 Step 62, training loss 1620132.0

2020-11-14 00:29:02.934641 validation 64:
2020-11-14 00:29:02.934690 hitrate@20=0.2800, recall@20=0.0520, ndcg@20=0.0409
2020-11-14 00:29:02.934712 hitrate@50=0.4470, recall@50=0.0944, ndcg@50=0.0569

Optimizing: 100%|███████████████████████████████| 30/30 [02:10<00:00,  4.35s/it]
2020-11-14 00:31:13.335783 Step 63, training loss 1627104.875

2020-11-14 00:31:19.215508 validation 65:
2020-11-14 00:31:19.215574 hitrate@20=0.2800, recall@20=0.0518, ndcg@20=0.0415
2020-11-14 00:31:19.215599 hitrate@50=0.4510, recall@50=0.0945, ndcg@50=0.0575

Optimizing: 100%|███████████████████████████████| 30/30 [02:14<00:00,  4.47s/it]
2020-11-14 00:33:33.292444 Step 64, training loss 1622400.875

2020-11-14 00:33:40.226144 validation 66:
2020-11-14 00:33:40.226213 hitrate@20=0.2830, recall@20=0.0523, ndcg@20=0.0417
2020-11-14 00:33:40.226236 hitrate@50=0.4450, recall@50=0.0952, ndcg@50=0.0578

Optimizing: 100%|███████████████████████████████| 30/30 [02:09<00:00,  4.32s/it]
2020-11-14 00:35:49.785555 Step 65, training loss 1616841.5

2020-11-14 00:35:55.965492 validation 67:
2020-11-14 00:35:55.965564 hitrate@20=0.2830, recall@20=0.0514, ndcg@20=0.0418
2020-11-14 00:35:55.965590 hitrate@50=0.4490, recall@50=0.0953, ndcg@50=0.0584

Optimizing: 100%|███████████████████████████████| 30/30 [02:12<00:00,  4.43s/it]
2020-11-14 00:38:08.955448 Step 66, training loss 1614712.25

2020-11-14 00:38:15.448515 validation 68:
2020-11-14 00:38:15.448579 hitrate@20=0.2810, recall@20=0.0498, ndcg@20=0.0410
2020-11-14 00:38:15.448603 hitrate@50=0.4470, recall@50=0.0953, ndcg@50=0.0580

2020-11-14 00:38:15.448638 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [02:09<00:00,  4.32s/it]
2020-11-14 00:40:25.064376 Step 67, training loss 1622360.25

2020-11-14 00:40:31.248998 validation 69:
2020-11-14 00:40:31.249079 hitrate@20=0.2850, recall@20=0.0505, ndcg@20=0.0417
2020-11-14 00:40:31.249104 hitrate@50=0.4470, recall@50=0.0966, ndcg@50=0.0588

Optimizing: 100%|███████████████████████████████| 30/30 [02:10<00:00,  4.36s/it]
2020-11-14 00:42:42.023705 Step 68, training loss 1616427.75

2020-11-14 00:42:48.705620 validation 70:
2020-11-14 00:42:48.705690 hitrate@20=0.2800, recall@20=0.0506, ndcg@20=0.0418
2020-11-14 00:42:48.705712 hitrate@50=0.4440, recall@50=0.0947, ndcg@50=0.0583

2020-11-14 00:42:48.705748 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [02:10<00:00,  4.36s/it]
2020-11-14 00:44:59.586339 Step 69, training loss 1612598.875

2020-11-14 00:45:05.530177 validation 71:
2020-11-14 00:45:05.530245 hitrate@20=0.2850, recall@20=0.0517, ndcg@20=0.0429
2020-11-14 00:45:05.530267 hitrate@50=0.4480, recall@50=0.0962, ndcg@50=0.0595

Optimizing: 100%|███████████████████████████████| 30/30 [02:13<00:00,  4.46s/it]
2020-11-14 00:47:19.354911 Step 70, training loss 1606704.875

2020-11-14 00:47:26.368856 validation 72:
2020-11-14 00:47:26.368920 hitrate@20=0.2850, recall@20=0.0527, ndcg@20=0.0427
2020-11-14 00:47:26.368943 hitrate@50=0.4420, recall@50=0.0947, ndcg@50=0.0586

2020-11-14 00:47:26.368977 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [02:09<00:00,  4.33s/it]
2020-11-14 00:49:36.364724 Step 71, training loss 1620388.125

2020-11-14 00:49:42.400649 validation 73:
2020-11-14 00:49:42.400738 hitrate@20=0.2840, recall@20=0.0522, ndcg@20=0.0428
2020-11-14 00:49:42.400765 hitrate@50=0.4410, recall@50=0.0953, ndcg@50=0.0591

2020-11-14 00:49:42.400802 the monitor loses its patience to 3!
Optimizing: 100%|███████████████████████████████| 30/30 [02:12<00:00,  4.42s/it]
2020-11-14 00:51:54.857164 Step 72, training loss 1609395.25

2020-11-14 00:52:02.366536 validation 74:
2020-11-14 00:52:02.366603 hitrate@20=0.2790, recall@20=0.0519, ndcg@20=0.0424
2020-11-14 00:52:02.366623 hitrate@50=0.4350, recall@50=0.0936, ndcg@50=0.0582

2020-11-14 00:52:02.366669 the monitor loses its patience to 2!
Optimizing: 100%|███████████████████████████████| 30/30 [02:10<00:00,  4.35s/it]
2020-11-14 00:54:12.761608 Step 73, training loss 1607137.625

2020-11-14 00:54:18.850184 validation 75:
2020-11-14 00:54:18.850233 hitrate@20=0.2750, recall@20=0.0507, ndcg@20=0.0412
2020-11-14 00:54:18.850262 hitrate@50=0.4270, recall@50=0.0930, ndcg@50=0.0573

2020-11-14 00:54:18.850307 the monitor loses its patience to 1!
Optimizing: 100%|███████████████████████████████| 30/30 [02:10<00:00,  4.36s/it]
2020-11-14 00:56:29.504789 Step 74, training loss 1604335.75

2020-11-14 00:56:36.157757 validation 76:
2020-11-14 00:56:36.157839 hitrate@20=0.2770, recall@20=0.0510, ndcg@20=0.0417
2020-11-14 00:56:36.157862 hitrate@50=0.4320, recall@50=0.0931, ndcg@50=0.0575

2020-11-14 00:56:36.157906 the monitor loses its patience to 0!
2020-11-14 00:56:36.157921 early stop at step 75
2020-11-14 00:56:36.810447 close sampler, close and save to log file
2020-11-14 00:56:37.154456 log file and sampler have closed
```


