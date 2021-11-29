
## ItemPop_yelp18_x0

A notebook to benchmark ItemPop on Yelp2018 dataset.

Author: Kelong Mao, Tsinghua University

Edited by [XUEPAI Team](https://github.com/xue-pai)


### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

```bash
CPU: Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz
RAM: 128G
```
+ Software

```bash
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
    python ItemPop.py --dataset=Yelp18 --topk=[20,50]
    ```

### Results
```bash
HR@20 = 0.08308071239105722, Recall@20 = 0.012446934769941874, NDCG@20 = 0.01011326127952093
HR@50 = 0.1492989768851838, Recall@50 = 0.024158638777874785, NDCG@50 = 0.014455232010635744
```

### Logs
```bash
Namespace(dataset='Yelp18', topk='[20,50]')
model fitting...
Generate recommend list...
0 ok, hit = 0
2000 ok, hit = 1047
4000 ok, hit = 1759
6000 ok, hit = 2394
8000 ok, hit = 2976
10000 ok, hit = 3448
12000 ok, hit = 3886
14000 ok, hit = 4271
16000 ok, hit = 4612
18000 ok, hit = 4926
20000 ok, hit = 5228
22000 ok, hit = 5477
24000 ok, hit = 5722
26000 ok, hit = 5960
28000 ok, hit = 6213
30000 ok, hit = 6403
HR@20 = 0.08308071239105722, Recall@20 = 0.012446934769941874, NDCG@20 = 0.01011326127952093
HR@50 = 0.1492989768851838, Recall@50 = 0.024158638777874785, NDCG@50 = 0.014455232010635744
Finished

```