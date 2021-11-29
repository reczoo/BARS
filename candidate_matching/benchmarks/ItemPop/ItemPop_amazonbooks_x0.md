
## ItemPop_amazonbooks_x0

A notebook to benchmark ItemPop on AmazonBooks dataset.

Author: Kelong Mao, Tsinghua University

Edited by [XUEPAI Team](https://github.com/xue-pai)


### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

```python
CPU: Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz
RAM: 128G
```
+ Software

```python
python: 3.6.9
pandas: 0.25.0
numpy: 1.19.1
```

### Dataset
We follow the data split and preprocessing steps in NGCF and LightGCN. We directly transform the formats of the data from their [repo](https://github.com/kuandeng/LightGCN/tree/master/Data).

### Code

1. This benchmark is implemented based on a public repository for recommendation, daisyRec: https://github.com/AmazingDD/daisyRec/tree/dff66b71a4d360eae7bf4edec5df1d4941937cb2. We use the version with commit hash: dff66b7.

2. We add ``RecallPrecision_ATk``, ``MRRatK_r``, ``NDCGatK_r``, ``HRK_r`` in ``daisy/utils/metrics.py`` for our benchmarking.
Three functions are copied from the code of [LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch/blob/b06c6b9db8391de4fbcf45ad436536743a6c896d/code/utils.py). You can view these changes via a diff comparison through this link: https://github.com/xue-pai/Open-CF-Benchmarks/compare/943043...28c87c?diff=split

3. Run the following script to reproduce the result.

    ```bash
    python ItemPop.py --dataset=AmazonBooks --topk=[20,50]
    ```

### Results

```bash
HR@20 = 0.04190490663526015, Recall@20 = 0.005102069690895848, NDCG@20 = 0.004355760427855336
HR@50 = 0.07636342913587751, Recall@50 = 0.010077022078757243, NDCG@50 = 0.006137977798402748
```

### Logs

```bash
Namespace(dataset='AmazonBooks', topk='[20,50]')
model fitting...
Generate recommend list...
0 ok, hit = 0
2000 ok, hit = 159
4000 ok, hit = 329
6000 ok, hit = 546
8000 ok, hit = 768
10000 ok, hit = 980
12000 ok, hit = 1181
14000 ok, hit = 1371
16000 ok, hit = 1525
18000 ok, hit = 1716
20000 ok, hit = 1928
22000 ok, hit = 2190
24000 ok, hit = 2416
26000 ok, hit = 2652
28000 ok, hit = 2848
30000 ok, hit = 3051
32000 ok, hit = 3238
34000 ok, hit = 3490
36000 ok, hit = 3737
38000 ok, hit = 3989
40000 ok, hit = 4229
42000 ok, hit = 4421
44000 ok, hit = 4613
46000 ok, hit = 4790
48000 ok, hit = 4951
50000 ok, hit = 5042
52000 ok, hit = 5138
HR@20 = 0.04190490663526015, Recall@20 = 0.005102069690895848, NDCG@20 = 0.004355760427855336
HR@50 = 0.07636342913587751, Recall@50 = 0.010077022078757243, NDCG@50 = 0.006137977798402748
Finished

```