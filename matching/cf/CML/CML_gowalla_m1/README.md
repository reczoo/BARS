## CML_gowalla_x0 

A notebook to benchmark CML on gowalla_x0 dataset.

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
dataset = "gowalla_x0"
train_data = "../../data/Gowalla/gowalla_x0/train.txt"
test_data = "../../data/Gowalla/gowalla_x0/test.txt"
batch_size = 50000
max_step=5000
num_negative=100
embed_dim=100
margin=1.9
clip_norm=1
lr = 0.001
dropout = 0.1 # dropout rate
verbose = 1 # Evaluation interval
topK = "20 50" # Metrics at TopK
```

```bash
python -u main.py --gpu 0 --dataset gowalla_x0 --train_data ../../data/Gowalla/gowalla_x0/train.txt --test_data ../../data/Gowalla/gowalla_x0/test.txt --verbose 30 --batch_size 50000 --max_step 5000 --embed_dim 100 --lr 0.001 --dropout 0.1 --margin 1.9 --clip_norm 1 --num_negative 100 --topK 20 50
```

### Results

```bash
2020-11-15 22:13:19.949682 validation 40:
2020-11-15 22:13:19.949741 hitrate@20=0.5410, recall@20=0.1670, ndcg@20=0.1292
2020-11-15 22:13:19.949762 hitrate@50=0.6750, recall@50=0.2602, ndcg@50=0.1587
```

### Logs

```bash
2020-11-15 21:15:47.441598 reading dataset gowalla_x0
2020-11-15 21:16:04.822505 #users=29858, #items=40981
WARNING:tensorflow:From /home/wjp/anaconda3/envs/py37torch/lib/python3.7/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-11-15 21:16:05.206747: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
2020-11-15 21:16:05.244440: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2799850000 Hz
2020-11-15 21:16:05.247275: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5643512f6360 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-11-15 21:16:05.247332: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-11-15 21:16:05.248950: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-11-15 21:16:05.906109: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:04:00.0
2020-11-15 21:16:05.906966: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-11-15 21:16:05.910078: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-11-15 21:16:05.912463: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2020-11-15 21:16:05.912871: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2020-11-15 21:16:05.915400: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2020-11-15 21:16:05.917137: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2020-11-15 21:16:05.925432: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-11-15 21:16:05.927505: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-11-15 21:16:05.927658: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-11-15 21:16:06.087627: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-11-15 21:16:06.087791: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
2020-11-15 21:16:06.087813: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
2020-11-15 21:16:06.092010: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10481 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:04:00.0, compute capability: 6.1)
2020-11-15 21:16:06.096617: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564353949830 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-11-15 21:16:06.096738: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce GTX 1080 Ti, Compute Capability 6.1
2020-11-15 21:16:09.093587 validation 1:
2020-11-15 21:16:09.093647 hitrate@20=0.0010, recall@20=0.0003, ndcg@20=0.0001
2020-11-15 21:16:09.093671 hitrate@50=0.0090, recall@50=0.0014, ndcg@50=0.0005

Optimizing: 100%|███████████████████████████████| 30/30 [01:30<00:00,  3.03s/it]
2020-11-15 21:17:39.945882 Step 0, training loss 1227726.0

2020-11-15 21:17:42.092232 validation 2:
2020-11-15 21:17:42.092366 hitrate@20=0.0250, recall@20=0.0042, ndcg@20=0.0022
2020-11-15 21:17:42.092400 hitrate@50=0.0420, recall@50=0.0070, ndcg@50=0.0030

Optimizing: 100%|███████████████████████████████| 30/30 [01:25<00:00,  2.84s/it]
2020-11-15 21:19:07.270789 Step 1, training loss 1199303.25

2020-11-15 21:19:09.547174 validation 3:
2020-11-15 21:19:09.547250 hitrate@20=0.0550, recall@20=0.0077, ndcg@20=0.0056
2020-11-15 21:19:09.547275 hitrate@50=0.0920, recall@50=0.0143, ndcg@50=0.0075

Optimizing: 100%|███████████████████████████████| 30/30 [01:27<00:00,  2.90s/it]
2020-11-15 21:20:36.639082 Step 2, training loss 1168580.125

2020-11-15 21:20:38.978787 validation 4:
2020-11-15 21:20:38.978841 hitrate@20=0.0580, recall@20=0.0077, ndcg@20=0.0054
2020-11-15 21:20:38.978862 hitrate@50=0.0990, recall@50=0.0147, ndcg@50=0.0076

Optimizing: 100%|███████████████████████████████| 30/30 [01:26<00:00,  2.90s/it]
2020-11-15 21:22:05.852811 Step 3, training loss 1130678.375

2020-11-15 21:22:08.061828 validation 5:
2020-11-15 21:22:08.061882 hitrate@20=0.0610, recall@20=0.0078, ndcg@20=0.0061
2020-11-15 21:22:08.061920 hitrate@50=0.1020, recall@50=0.0152, ndcg@50=0.0082

Optimizing: 100%|███████████████████████████████| 30/30 [01:27<00:00,  2.91s/it]
2020-11-15 21:23:35.236312 Step 4, training loss 1121809.5

2020-11-15 21:23:37.116598 validation 6:
2020-11-15 21:23:37.116653 hitrate@20=0.0590, recall@20=0.0078, ndcg@20=0.0063
2020-11-15 21:23:37.116672 hitrate@50=0.1010, recall@50=0.0152, ndcg@50=0.0084

Optimizing: 100%|███████████████████████████████| 30/30 [01:30<00:00,  3.03s/it]
2020-11-15 21:25:08.040926 Step 5, training loss 1093150.75

2020-11-15 21:25:10.107233 validation 7:
2020-11-15 21:25:10.107287 hitrate@20=0.0600, recall@20=0.0075, ndcg@20=0.0065
2020-11-15 21:25:10.107307 hitrate@50=0.1010, recall@50=0.0142, ndcg@50=0.0084

2020-11-15 21:25:10.107340 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [01:29<00:00,  2.99s/it]
2020-11-15 21:26:39.883981 Step 6, training loss 1077487.875

2020-11-15 21:26:41.873263 validation 8:
2020-11-15 21:26:41.873315 hitrate@20=0.0690, recall@20=0.0084, ndcg@20=0.0076
2020-11-15 21:26:41.873334 hitrate@50=0.1040, recall@50=0.0145, ndcg@50=0.0094

Optimizing: 100%|███████████████████████████████| 30/30 [01:30<00:00,  3.01s/it]
2020-11-15 21:28:12.084131 Step 7, training loss 1073506.25

2020-11-15 21:28:14.256512 validation 9:
2020-11-15 21:28:14.256564 hitrate@20=0.0750, recall@20=0.0099, ndcg@20=0.0089
2020-11-15 21:28:14.256582 hitrate@50=0.1130, recall@50=0.0160, ndcg@50=0.0106

Optimizing: 100%|███████████████████████████████| 30/30 [01:31<00:00,  3.03s/it]
2020-11-15 21:29:45.300731 Step 8, training loss 1042043.4375

2020-11-15 21:29:47.403093 validation 10:
2020-11-15 21:29:47.403148 hitrate@20=0.0900, recall@20=0.0114, ndcg@20=0.0110
2020-11-15 21:29:47.403176 hitrate@50=0.1310, recall@50=0.0188, ndcg@50=0.0131

Optimizing: 100%|███████████████████████████████| 30/30 [01:30<00:00,  3.01s/it]
2020-11-15 21:31:17.583341 Step 9, training loss 1039255.8125

2020-11-15 21:31:19.853722 validation 11:
2020-11-15 21:31:19.853774 hitrate@20=0.1140, recall@20=0.0157, ndcg@20=0.0152
2020-11-15 21:31:19.853795 hitrate@50=0.1600, recall@50=0.0254, ndcg@50=0.0181

Optimizing: 100%|███████████████████████████████| 30/30 [01:30<00:00,  3.03s/it]
2020-11-15 21:32:50.721034 Step 10, training loss 1027587.0625

2020-11-15 21:32:52.938659 validation 12:
2020-11-15 21:32:52.938716 hitrate@20=0.1390, recall@20=0.0215, ndcg@20=0.0217
2020-11-15 21:32:52.938737 hitrate@50=0.1950, recall@50=0.0349, ndcg@50=0.0259

Optimizing: 100%|███████████████████████████████| 30/30 [01:29<00:00,  2.98s/it]
2020-11-15 21:34:22.289642 Step 11, training loss 1006920.1875

2020-11-15 21:34:24.997022 validation 13:
2020-11-15 21:34:24.997076 hitrate@20=0.1840, recall@20=0.0310, ndcg@20=0.0294
2020-11-15 21:34:24.997097 hitrate@50=0.2320, recall@50=0.0455, ndcg@50=0.0339

Optimizing: 100%|███████████████████████████████| 30/30 [01:22<00:00,  2.75s/it]
2020-11-15 21:35:47.450577 Step 12, training loss 1002420.0625

2020-11-15 21:35:50.423277 validation 14:
2020-11-15 21:35:50.423351 hitrate@20=0.2220, recall@20=0.0386, ndcg@20=0.0386
2020-11-15 21:35:50.423375 hitrate@50=0.2860, recall@50=0.0613, ndcg@50=0.0455

Optimizing: 100%|███████████████████████████████| 30/30 [01:23<00:00,  2.78s/it]
2020-11-15 21:37:13.696735 Step 13, training loss 979466.625

2020-11-15 21:37:16.694252 validation 15:
2020-11-15 21:37:16.694309 hitrate@20=0.2610, recall@20=0.0519, ndcg@20=0.0474
2020-11-15 21:37:16.694346 hitrate@50=0.3360, recall@50=0.0808, ndcg@50=0.0564

Optimizing: 100%|███████████████████████████████| 30/30 [01:22<00:00,  2.75s/it]
2020-11-15 21:38:39.311234 Step 14, training loss 964952.6875

2020-11-15 21:38:41.913126 validation 16:
2020-11-15 21:38:41.913181 hitrate@20=0.3020, recall@20=0.0677, ndcg@20=0.0592
2020-11-15 21:38:41.913202 hitrate@50=0.3690, recall@50=0.0977, ndcg@50=0.0690

Optimizing: 100%|███████████████████████████████| 30/30 [01:24<00:00,  2.80s/it]
2020-11-15 21:40:05.918400 Step 15, training loss 960261.75

2020-11-15 21:40:08.380865 validation 17:
2020-11-15 21:40:08.380920 hitrate@20=0.3410, recall@20=0.0794, ndcg@20=0.0679
2020-11-15 21:40:08.380956 hitrate@50=0.4250, recall@50=0.1205, ndcg@50=0.0810

Optimizing: 100%|███████████████████████████████| 30/30 [01:24<00:00,  2.80s/it]
2020-11-15 21:41:32.468951 Step 16, training loss 931460.625

2020-11-15 21:41:34.874486 validation 18:
2020-11-15 21:41:34.874537 hitrate@20=0.3650, recall@20=0.0902, ndcg@20=0.0765
2020-11-15 21:41:34.874563 hitrate@50=0.4670, recall@50=0.1386, ndcg@50=0.0919

Optimizing: 100%|███████████████████████████████| 30/30 [01:25<00:00,  2.84s/it]
2020-11-15 21:42:59.953364 Step 17, training loss 926971.9375

2020-11-15 21:43:02.400519 validation 19:
2020-11-15 21:43:02.400579 hitrate@20=0.3990, recall@20=0.1024, ndcg@20=0.0864
2020-11-15 21:43:02.400602 hitrate@50=0.5060, recall@50=0.1565, ndcg@50=0.1034

Optimizing: 100%|███████████████████████████████| 30/30 [01:23<00:00,  2.79s/it]
2020-11-15 21:44:26.201114 Step 18, training loss 913304.3125

2020-11-15 21:44:28.661938 validation 20:
2020-11-15 21:44:28.661996 hitrate@20=0.4280, recall@20=0.1150, ndcg@20=0.0941
2020-11-15 21:44:28.662033 hitrate@50=0.5470, recall@50=0.1771, ndcg@50=0.1136

Optimizing: 100%|███████████████████████████████| 30/30 [01:23<00:00,  2.79s/it]
2020-11-15 21:45:52.295390 Step 19, training loss 888676.375

2020-11-15 21:45:55.422667 validation 21:
2020-11-15 21:45:55.422783 hitrate@20=0.4400, recall@20=0.1197, ndcg@20=0.0979
2020-11-15 21:45:55.422820 hitrate@50=0.5760, recall@50=0.1905, ndcg@50=0.1203

Optimizing: 100%|███████████████████████████████| 30/30 [01:23<00:00,  2.78s/it]
2020-11-15 21:47:18.904890 Step 20, training loss 887842.6875

2020-11-15 21:47:21.761960 validation 22:
2020-11-15 21:47:21.762030 hitrate@20=0.4490, recall@20=0.1258, ndcg@20=0.1025
2020-11-15 21:47:21.762054 hitrate@50=0.5900, recall@50=0.1996, ndcg@50=0.1257

Optimizing: 100%|███████████████████████████████| 30/30 [01:25<00:00,  2.84s/it]
2020-11-15 21:48:46.908006 Step 21, training loss 867397.25

2020-11-15 21:48:50.033075 validation 23:
2020-11-15 21:48:50.033133 hitrate@20=0.4690, recall@20=0.1318, ndcg@20=0.1071
2020-11-15 21:48:50.033173 hitrate@50=0.6020, recall@50=0.2082, ndcg@50=0.1313

Optimizing: 100%|███████████████████████████████| 30/30 [01:22<00:00,  2.76s/it]
2020-11-15 21:50:12.689117 Step 22, training loss 859652.5625

2020-11-15 21:50:15.682127 validation 24:
2020-11-15 21:50:15.682185 hitrate@20=0.4790, recall@20=0.1387, ndcg@20=0.1100
2020-11-15 21:50:15.682223 hitrate@50=0.6130, recall@50=0.2162, ndcg@50=0.1347

Optimizing: 100%|███████████████████████████████| 30/30 [01:23<00:00,  2.79s/it]
2020-11-15 21:51:39.313830 Step 23, training loss 862426.0

2020-11-15 21:51:42.641438 validation 25:
2020-11-15 21:51:42.641542 hitrate@20=0.4820, recall@20=0.1417, ndcg@20=0.1121
2020-11-15 21:51:42.641568 hitrate@50=0.6190, recall@50=0.2183, ndcg@50=0.1370

Optimizing: 100%|███████████████████████████████| 30/30 [01:23<00:00,  2.78s/it]
2020-11-15 21:53:05.978899 Step 24, training loss 837701.8125

2020-11-15 21:53:08.897765 validation 26:
2020-11-15 21:53:08.897817 hitrate@20=0.4900, recall@20=0.1462, ndcg@20=0.1139
2020-11-15 21:53:08.897854 hitrate@50=0.6310, recall@50=0.2234, ndcg@50=0.1386

Optimizing: 100%|███████████████████████████████| 30/30 [01:22<00:00,  2.76s/it]
2020-11-15 21:54:31.602205 Step 25, training loss 843166.4375

2020-11-15 21:54:34.647463 validation 27:
2020-11-15 21:54:34.647549 hitrate@20=0.4960, recall@20=0.1490, ndcg@20=0.1165
2020-11-15 21:54:34.647575 hitrate@50=0.6320, recall@50=0.2289, ndcg@50=0.1422

Optimizing: 100%|███████████████████████████████| 30/30 [01:23<00:00,  2.77s/it]
2020-11-15 21:55:57.688860 Step 26, training loss 838034.5625

2020-11-15 21:56:00.647457 validation 28:
2020-11-15 21:56:00.647512 hitrate@20=0.5000, recall@20=0.1509, ndcg@20=0.1176
2020-11-15 21:56:00.647557 hitrate@50=0.6360, recall@50=0.2310, ndcg@50=0.1430

Optimizing: 100%|███████████████████████████████| 30/30 [01:22<00:00,  2.74s/it]
2020-11-15 21:57:22.889208 Step 27, training loss 822968.0

2020-11-15 21:57:25.874507 validation 29:
2020-11-15 21:57:25.874577 hitrate@20=0.4970, recall@20=0.1516, ndcg@20=0.1186
2020-11-15 21:57:25.874600 hitrate@50=0.6390, recall@50=0.2343, ndcg@50=0.1448

Optimizing: 100%|███████████████████████████████| 30/30 [01:22<00:00,  2.75s/it]
2020-11-15 21:58:48.449751 Step 28, training loss 831381.25

2020-11-15 21:58:51.472739 validation 30:
2020-11-15 21:58:51.472793 hitrate@20=0.5050, recall@20=0.1533, ndcg@20=0.1195
2020-11-15 21:58:51.472813 hitrate@50=0.6430, recall@50=0.2382, ndcg@50=0.1461

Optimizing: 100%|███████████████████████████████| 30/30 [01:23<00:00,  2.78s/it]
2020-11-15 22:00:14.899230 Step 29, training loss 817437.375

2020-11-15 22:00:18.072424 validation 31:
2020-11-15 22:00:18.072496 hitrate@20=0.5160, recall@20=0.1566, ndcg@20=0.1211
2020-11-15 22:00:18.072531 hitrate@50=0.6460, recall@50=0.2371, ndcg@50=0.1465

Optimizing: 100%|███████████████████████████████| 30/30 [01:22<00:00,  2.76s/it]
2020-11-15 22:01:40.938294 Step 30, training loss 815070.125

2020-11-15 22:01:44.080338 validation 32:
2020-11-15 22:01:44.080392 hitrate@20=0.5060, recall@20=0.1537, ndcg@20=0.1208
2020-11-15 22:01:44.080433 hitrate@50=0.6640, recall@50=0.2418, ndcg@50=0.1486

Optimizing: 100%|███████████████████████████████| 30/30 [01:24<00:00,  2.81s/it]
2020-11-15 22:03:08.453730 Step 31, training loss 823731.3125

2020-11-15 22:03:11.604545 validation 33:
2020-11-15 22:03:11.604602 hitrate@20=0.5150, recall@20=0.1540, ndcg@20=0.1225
2020-11-15 22:03:11.604624 hitrate@50=0.6650, recall@50=0.2455, ndcg@50=0.1515

Optimizing: 100%|███████████████████████████████| 30/30 [01:23<00:00,  2.79s/it]
2020-11-15 22:04:35.361387 Step 32, training loss 804585.5625

2020-11-15 22:04:38.295770 validation 34:
2020-11-15 22:04:38.295827 hitrate@20=0.5180, recall@20=0.1542, ndcg@20=0.1218
2020-11-15 22:04:38.295864 hitrate@50=0.6580, recall@50=0.2426, ndcg@50=0.1500

2020-11-15 22:04:38.295912 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [01:23<00:00,  2.80s/it]
2020-11-15 22:06:02.284818 Step 33, training loss 811763.5625

2020-11-15 22:06:05.435509 validation 35:
2020-11-15 22:06:05.435569 hitrate@20=0.5210, recall@20=0.1567, ndcg@20=0.1241
2020-11-15 22:06:05.435593 hitrate@50=0.6640, recall@50=0.2487, ndcg@50=0.1533

Optimizing: 100%|███████████████████████████████| 30/30 [01:25<00:00,  2.84s/it]
2020-11-15 22:07:30.731299 Step 34, training loss 810745.375

2020-11-15 22:07:33.944597 validation 36:
2020-11-15 22:07:33.944651 hitrate@20=0.5240, recall@20=0.1603, ndcg@20=0.1253
2020-11-15 22:07:33.944672 hitrate@50=0.6730, recall@50=0.2523, ndcg@50=0.1548

Optimizing: 100%|███████████████████████████████| 30/30 [01:23<00:00,  2.78s/it]
2020-11-15 22:08:57.216359 Step 35, training loss 798058.6875

2020-11-15 22:09:00.400408 validation 37:
2020-11-15 22:09:00.400463 hitrate@20=0.5380, recall@20=0.1636, ndcg@20=0.1276
2020-11-15 22:09:00.400501 hitrate@50=0.6740, recall@50=0.2545, ndcg@50=0.1567

Optimizing: 100%|███████████████████████████████| 30/30 [01:23<00:00,  2.78s/it]
2020-11-15 22:10:23.861894 Step 36, training loss 808465.4375

2020-11-15 22:10:26.683641 validation 38:
2020-11-15 22:10:26.683699 hitrate@20=0.5370, recall@20=0.1663, ndcg@20=0.1290
2020-11-15 22:10:26.683734 hitrate@50=0.6680, recall@50=0.2534, ndcg@50=0.1570

Optimizing: 100%|███████████████████████████████| 30/30 [01:24<00:00,  2.82s/it]
2020-11-15 22:11:51.289447 Step 37, training loss 798589.125

2020-11-15 22:11:54.621169 validation 39:
2020-11-15 22:11:54.621224 hitrate@20=0.5240, recall@20=0.1631, ndcg@20=0.1271
2020-11-15 22:11:54.621256 hitrate@50=0.6570, recall@50=0.2535, ndcg@50=0.1561

2020-11-15 22:11:54.621298 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [01:22<00:00,  2.73s/it]
2020-11-15 22:13:16.668409 Step 38, training loss 798385.8125

2020-11-15 22:13:19.949682 validation 40:
2020-11-15 22:13:19.949741 hitrate@20=0.5410, recall@20=0.1670, ndcg@20=0.1292
2020-11-15 22:13:19.949762 hitrate@50=0.6750, recall@50=0.2602, ndcg@50=0.1587

Optimizing: 100%|███████████████████████████████| 30/30 [01:22<00:00,  2.75s/it]
2020-11-15 22:14:42.476775 Step 39, training loss 806514.9375

2020-11-15 22:14:45.869174 validation 41:
2020-11-15 22:14:45.869237 hitrate@20=0.5470, recall@20=0.1679, ndcg@20=0.1284
2020-11-15 22:14:45.869277 hitrate@50=0.6790, recall@50=0.2577, ndcg@50=0.1570

2020-11-15 22:14:45.869312 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [01:21<00:00,  2.72s/it]
2020-11-15 22:16:07.606060 Step 40, training loss 789015.0

2020-11-15 22:16:11.034896 validation 42:
2020-11-15 22:16:11.034967 hitrate@20=0.5350, recall@20=0.1654, ndcg@20=0.1272
2020-11-15 22:16:11.034996 hitrate@50=0.6690, recall@50=0.2575, ndcg@50=0.1563

2020-11-15 22:16:11.035030 the monitor loses its patience to 3!
Optimizing: 100%|███████████████████████████████| 30/30 [01:21<00:00,  2.72s/it]
2020-11-15 22:17:32.725192 Step 41, training loss 797929.125

2020-11-15 22:17:36.126152 validation 43:
2020-11-15 22:17:36.126207 hitrate@20=0.5340, recall@20=0.1647, ndcg@20=0.1273
2020-11-15 22:17:36.126228 hitrate@50=0.6890, recall@50=0.2631, ndcg@50=0.1587

2020-11-15 22:17:36.126260 the monitor loses its patience to 2!
Optimizing: 100%|███████████████████████████████| 30/30 [01:24<00:00,  2.80s/it]
2020-11-15 22:19:00.211632 Step 42, training loss 797060.9375

2020-11-15 22:19:03.397422 validation 44:
2020-11-15 22:19:03.397482 hitrate@20=0.5340, recall@20=0.1635, ndcg@20=0.1277
2020-11-15 22:19:03.397521 hitrate@50=0.6830, recall@50=0.2640, ndcg@50=0.1596

2020-11-15 22:19:03.397557 the monitor loses its patience to 1!
Optimizing: 100%|███████████████████████████████| 30/30 [01:23<00:00,  2.79s/it]
2020-11-15 22:20:27.071727 Step 43, training loss 786370.375

2020-11-15 22:20:30.574027 validation 45:
2020-11-15 22:20:30.574098 hitrate@20=0.5390, recall@20=0.1637, ndcg@20=0.1260
2020-11-15 22:20:30.574124 hitrate@50=0.6900, recall@50=0.2668, ndcg@50=0.1587

Optimizing: 100%|███████████████████████████████| 30/30 [01:22<00:00,  2.76s/it]
2020-11-15 22:21:53.385345 Step 44, training loss 797628.6875

2020-11-15 22:21:56.902899 validation 46:
2020-11-15 22:21:56.902958 hitrate@20=0.5330, recall@20=0.1587, ndcg@20=0.1263
2020-11-15 22:21:56.902996 hitrate@50=0.6820, recall@50=0.2643, ndcg@50=0.1592

2020-11-15 22:21:56.903029 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [01:23<00:00,  2.79s/it]
2020-11-15 22:23:20.474320 Step 45, training loss 786160.625

2020-11-15 22:23:23.634873 validation 47:
2020-11-15 22:23:23.634957 hitrate@20=0.5430, recall@20=0.1649, ndcg@20=0.1281
2020-11-15 22:23:23.634998 hitrate@50=0.6760, recall@50=0.2626, ndcg@50=0.1588

2020-11-15 22:23:23.635041 the monitor loses its patience to 3!
Optimizing: 100%|███████████████████████████████| 30/30 [01:22<00:00,  2.76s/it]
2020-11-15 22:24:46.458577 Step 46, training loss 786441.625

2020-11-15 22:24:49.542940 validation 48:
2020-11-15 22:24:49.543023 hitrate@20=0.5390, recall@20=0.1630, ndcg@20=0.1264
2020-11-15 22:24:49.543049 hitrate@50=0.6760, recall@50=0.2627, ndcg@50=0.1579

2020-11-15 22:24:49.543103 the monitor loses its patience to 2!
Optimizing: 100%|███████████████████████████████| 30/30 [01:24<00:00,  2.80s/it]
2020-11-15 22:26:13.654944 Step 47, training loss 793838.8125

2020-11-15 22:26:17.258534 validation 49:
2020-11-15 22:26:17.258594 hitrate@20=0.5410, recall@20=0.1654, ndcg@20=0.1270
2020-11-15 22:26:17.258636 hitrate@50=0.6870, recall@50=0.2642, ndcg@50=0.1584

2020-11-15 22:26:17.258670 the monitor loses its patience to 1!
Optimizing: 100%|███████████████████████████████| 30/30 [01:23<00:00,  2.78s/it]
2020-11-15 22:27:40.659925 Step 48, training loss 776240.0625

2020-11-15 22:27:44.231278 validation 50:
2020-11-15 22:27:44.231348 hitrate@20=0.5370, recall@20=0.1654, ndcg@20=0.1261
2020-11-15 22:27:44.231388 hitrate@50=0.6970, recall@50=0.2674, ndcg@50=0.1584

Optimizing: 100%|███████████████████████████████| 30/30 [01:23<00:00,  2.78s/it]
2020-11-15 22:29:07.554374 Step 49, training loss 784841.5625

2020-11-15 22:29:11.149549 validation 51:
2020-11-15 22:29:11.149604 hitrate@20=0.5240, recall@20=0.1606, ndcg@20=0.1247
2020-11-15 22:29:11.149640 hitrate@50=0.6910, recall@50=0.2639, ndcg@50=0.1575

2020-11-15 22:29:11.149685 the monitor loses its patience to 4!
Optimizing: 100%|███████████████████████████████| 30/30 [01:23<00:00,  2.78s/it]
2020-11-15 22:30:34.457681 Step 50, training loss 785478.125

2020-11-15 22:30:37.803911 validation 52:
2020-11-15 22:30:37.803972 hitrate@20=0.5330, recall@20=0.1635, ndcg@20=0.1255
2020-11-15 22:30:37.804010 hitrate@50=0.6870, recall@50=0.2639, ndcg@50=0.1572

2020-11-15 22:30:37.804043 the monitor loses its patience to 3!
Optimizing: 100%|███████████████████████████████| 30/30 [01:22<00:00,  2.74s/it]
2020-11-15 22:32:00.148962 Step 51, training loss 774387.6875

2020-11-15 22:32:03.998952 validation 53:
2020-11-15 22:32:03.999017 hitrate@20=0.5380, recall@20=0.1641, ndcg@20=0.1251
2020-11-15 22:32:03.999059 hitrate@50=0.6860, recall@50=0.2645, ndcg@50=0.1572

2020-11-15 22:32:03.999101 the monitor loses its patience to 2!
Optimizing: 100%|███████████████████████████████| 30/30 [01:22<00:00,  2.74s/it]
2020-11-15 22:33:26.312584 Step 52, training loss 785608.8125

2020-11-15 22:33:29.732795 validation 54:
2020-11-15 22:33:29.732866 hitrate@20=0.5350, recall@20=0.1632, ndcg@20=0.1244
2020-11-15 22:33:29.732905 hitrate@50=0.6900, recall@50=0.2664, ndcg@50=0.1569

2020-11-15 22:33:29.732937 the monitor loses its patience to 1!
Optimizing: 100%|███████████████████████████████| 30/30 [01:28<00:00,  2.94s/it]
2020-11-15 22:34:58.052958 Step 53, training loss 775699.3125

2020-11-15 22:35:01.441011 validation 55:
2020-11-15 22:35:01.441077 hitrate@20=0.5420, recall@20=0.1653, ndcg@20=0.1247
2020-11-15 22:35:01.441110 hitrate@50=0.6870, recall@50=0.2659, ndcg@50=0.1568

2020-11-15 22:35:01.441158 the monitor loses its patience to 0!
2020-11-15 22:35:01.441174 early stop at step 54
2020-11-15 22:35:01.796057 close sampler, close and save to log file
2020-11-15 22:35:01.906120 log file and sampler have closed
```


