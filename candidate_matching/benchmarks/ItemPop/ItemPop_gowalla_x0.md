
## ItemPop_gowalla_x0

A notebook to benchmark ItemPop on Gowalla dataset.

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
    python ItemPop.py --dataset=Gowalla --topk=[20,50]
    ```

### Results
```bash
HR@20 = 0.2038314689530444, Recall@20 = 0.04163097892995584, NDCG@20 = 0.03168992927421666
HR@50 = 0.27768102351128676, Recall@50 = 0.06237059308999754, NDCG@50 = 0.03789331440607942
```


### Logs
```bash
Namespace(dataset='Gowalla', topk='[20,50]')
model fitting...
Generate recommend list...
0 ok, hit = 1
2000 ok, hit = 1530
4000 ok, hit = 2562
6000 ok, hit = 3474
8000 ok, hit = 4359
10000 ok, hit = 5255
12000 ok, hit = 6377
14000 ok, hit = 7195
16000 ok, hit = 8015
18000 ok, hit = 8844
20000 ok, hit = 9621
22000 ok, hit = 10300
24000 ok, hit = 10949
26000 ok, hit = 11588
28000 ok, hit = 12146
HR@20 = 0.2038314689530444, Recall@20 = 0.04163097892995584, NDCG@20 = 0.03168992927421666
HR@50 = 0.27768102351128676, Recall@50 = 0.06237059308999754, NDCG@50 = 0.03789331440607942
Finished
```