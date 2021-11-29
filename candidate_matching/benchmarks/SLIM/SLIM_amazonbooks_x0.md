## SLIM_amazonbooks_x0

A notebook to benchmark SLIM on Amazonbooks dataset.

Author: Kelong Mao, Tsinghua University

Edited by [XUEPAI Team](https://github.com/xue-pai)


### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

    ```bash
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
We follow the data split and preprocessing steps in NGCF and LightGCN. We directly transform the formats of the data from their [repo](https://github.com/kuandeng/LightGCN/tree/master/Data).

### Code
1. This benchmark is implemented based on a public repository for recommendation, daisyRec: https://github.com/AmazingDD/daisyRec/tree/dff66b71a4d360eae7bf4edec5df1d4941937cb2. We use the version with commit hash: dff66b7.

2. We add ``RecallPrecision_ATk``, ``MRRatK_r``, ``NDCGatK_r``, ``HRK_r`` in ``daisy/utils/metrics.py`` for our benchmarking.
Three functions are copied from the code of [LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch/blob/b06c6b9db8391de4fbcf45ad436536743a6c896d/code/utils.py). You can view these changes via a diff comparison through this link: https://github.com/xue-pai/Open-CF-Benchmarks/compare/943043...28c87c?diff=split

3. Run the following script to reproduce the result.

    `Hyperparameters: alpha tuned from [1e-2, 2e-3, 1e-3, 5e-4, 1e-4], l1 tuned from [1e-3, 1e-4, 1e-5, 1e-6]`

    ```bash
    python -u SLIM.py --dataset=AmazonBooks --topk=[20,50] --alpha=0.001 --l1=0.001
    ```

### Results
```bash 
HR@20 = 0.3873069543908972, Recall@20 = 0.07548407053126333, NDCG@20 = 0.06015578801234101, HR@50 = 0.5472332503846665, Recall@50 = 0.12569538837382385, NDCG@50 = 0.07911547470326016
```

### Logs
```bash
Namespace(alpha=0.001, dataset='AmazonBooks', l1=0.001, topk='[20,50]')
user num: 52643, item num: 91599
model fitting...
SLIMElasticNetRecommender: Processed 1000 ( 1.09% ) in 3.37 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 2000 ( 2.18% ) in 6.70 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 3000 ( 3.28% ) in 9.99 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 4000 ( 4.37% ) in 13.28 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 5000 ( 5.46% ) in 16.57 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 6000 ( 6.55% ) in 19.86 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 7000 ( 7.64% ) in 23.14 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 8000 ( 8.73% ) in 26.40 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 9000 ( 9.83% ) in 29.67 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 10000 ( 10.92% ) in 32.94 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 11000 ( 12.01% ) in 36.20 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 12000 ( 13.10% ) in 39.46 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 13000 ( 14.19% ) in 42.71 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 14000 ( 15.28% ) in 45.94 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 15000 ( 16.38% ) in 49.17 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 16000 ( 17.47% ) in 52.39 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 17000 ( 18.56% ) in 55.63 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 18000 ( 19.65% ) in 58.86 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 19000 ( 20.74% ) in 62.11 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 20000 ( 21.83% ) in 65.36 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 21000 ( 22.93% ) in 68.62 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 22000 ( 24.02% ) in 71.87 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 23000 ( 25.11% ) in 75.09 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 24000 ( 26.20% ) in 78.37 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 25000 ( 27.29% ) in 81.98 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 26000 ( 28.38% ) in 85.16 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 27000 ( 29.48% ) in 88.38 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 27792 ( 30.34% ) in 93.39 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 28000 ( 30.57% ) in 95.30 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 28541 ( 31.16% ) in 100.30 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 29000 ( 31.66% ) in 104.75 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 29532 ( 32.24% ) in 109.75 minutes. Items per second: 4
SLIMElasticNetRecommender: Processed 30000 ( 32.75% ) in 114.06 minutes. Items per second: 4
SLIMElasticNetRecommender: Processed 31000 ( 33.84% ) in 118.13 minutes. Items per second: 4
SLIMElasticNetRecommender: Processed 32000 ( 34.93% ) in 121.34 minutes. Items per second: 4
SLIMElasticNetRecommender: Processed 33000 ( 36.03% ) in 124.50 minutes. Items per second: 4
SLIMElasticNetRecommender: Processed 34000 ( 37.12% ) in 127.64 minutes. Items per second: 4
SLIMElasticNetRecommender: Processed 35000 ( 38.21% ) in 130.77 minutes. Items per second: 4
SLIMElasticNetRecommender: Processed 36000 ( 39.30% ) in 133.90 minutes. Items per second: 4
SLIMElasticNetRecommender: Processed 37000 ( 40.39% ) in 137.06 minutes. Items per second: 4
SLIMElasticNetRecommender: Processed 38000 ( 41.49% ) in 140.21 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 39000 ( 42.58% ) in 143.34 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 40000 ( 43.67% ) in 146.60 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 41000 ( 44.76% ) in 150.06 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 42000 ( 45.85% ) in 153.62 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 43000 ( 46.94% ) in 156.72 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 44000 ( 48.04% ) in 159.82 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 45000 ( 49.13% ) in 162.89 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 46000 ( 50.22% ) in 165.99 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 47000 ( 51.31% ) in 169.06 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 48000 ( 52.40% ) in 172.16 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 49000 ( 53.49% ) in 175.28 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 50000 ( 54.59% ) in 178.42 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 51000 ( 55.68% ) in 181.53 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 52000 ( 56.77% ) in 184.64 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 53000 ( 57.86% ) in 187.74 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 54000 ( 58.95% ) in 190.82 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 55000 ( 60.04% ) in 194.15 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 56000 ( 61.14% ) in 198.41 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 57000 ( 62.23% ) in 202.82 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 58000 ( 63.32% ) in 207.31 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 59000 ( 64.41% ) in 211.78 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 60000 ( 65.50% ) in 215.68 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 61000 ( 66.59% ) in 219.43 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 62000 ( 67.69% ) in 222.94 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 63000 ( 68.78% ) in 226.07 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 64000 ( 69.87% ) in 229.17 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 65000 ( 70.96% ) in 232.29 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 66000 ( 72.05% ) in 235.36 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 67000 ( 73.14% ) in 238.46 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 68000 ( 74.24% ) in 241.57 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 69000 ( 75.33% ) in 244.70 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 70000 ( 76.42% ) in 247.79 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 71000 ( 77.51% ) in 250.90 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 72000 ( 78.60% ) in 254.01 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 73000 ( 79.70% ) in 257.07 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 74000 ( 80.79% ) in 260.14 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 75000 ( 81.88% ) in 263.22 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 76000 ( 82.97% ) in 266.96 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 77000 ( 84.06% ) in 270.07 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 78000 ( 85.15% ) in 273.30 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 79000 ( 86.25% ) in 276.51 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 80000 ( 87.34% ) in 280.48 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 81000 ( 88.43% ) in 284.21 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 82000 ( 89.52% ) in 287.31 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 83000 ( 90.61% ) in 290.38 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 84000 ( 91.70% ) in 293.42 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 85000 ( 92.80% ) in 296.57 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 86000 ( 93.89% ) in 299.59 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 87000 ( 94.98% ) in 302.61 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 88000 ( 96.07% ) in 305.61 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 89000 ( 97.16% ) in 308.59 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 90000 ( 98.25% ) in 311.55 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 91000 ( 99.35% ) in 314.45 minutes. Items per second: 5
SLIMElasticNetRecommender: Processed 91599 ( 100.00% ) in 316.03 minutes. Items per second: 5
Generate recommend list...
0 ok, hit = 5
2000 ok, hit = 3395
4000 ok, hit = 5349
6000 ok, hit = 7210
8000 ok, hit = 9116
10000 ok, hit = 10857
12000 ok, hit = 12613
14000 ok, hit = 14407
16000 ok, hit = 16235
18000 ok, hit = 18306
20000 ok, hit = 20301
22000 ok, hit = 22293
24000 ok, hit = 24499
26000 ok, hit = 27001
28000 ok, hit = 29178
30000 ok, hit = 31559
32000 ok, hit = 33848
34000 ok, hit = 36357
36000 ok, hit = 38812
38000 ok, hit = 41251
40000 ok, hit = 43650
42000 ok, hit = 46079
44000 ok, hit = 48408
46000 ok, hit = 50887
48000 ok, hit = 53291
50000 ok, hit = 55664
52000 ok, hit = 58003
HR@20 = 0.3873069543908972, Recall@20 = 0.07548407053126333, NDCG@20 = 0.06015578801234101, HR@50 = 0.5472332503846665, Recall@50 = 0.12569538837382385, NDCG@50 = 0.07911547470326016, 
Finished
```