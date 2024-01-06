## SLIM_gowalla_x0

A notebook to benchmark SLIM on Gowalla dataset.

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

    `Hyperparameters: alpha tuned from [0.01, 0.015, 0.005], l1 tuned from [1e-3, 1e-4, 1e-5, 1e-6]`

    ```bash
    python -u SLIM.py --dataset=Gowalla --topk=[20,50] --alpha=0.005 --l1=1e-6
    ```

### Results
```bash
HR@20 = 0.5564337865898586, Recall@20 = 0.1698757481743364, NDCG@20 = 0.13821795172035953, HR@50 = 0.6960278652287494, Recall@50 = 0.2657772365068036, NDCG@50 = 0.16866578711178626
```

### Logs
```bash
Namespace(alpha=0.005, dataset='Gowalla', l1=1e-06, topk='[20,50]')
user num: 29858, item num: 40981
model fitting...
SLIMElasticNetRecommender: Processed 1000 ( 2.44% ) in 1.72 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 2000 ( 4.88% ) in 3.38 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 3000 ( 7.32% ) in 5.00 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 4000 ( 9.76% ) in 6.61 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 5000 ( 12.20% ) in 8.21 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 6000 ( 14.64% ) in 9.80 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 7000 ( 17.08% ) in 11.40 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 8000 ( 19.52% ) in 12.93 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 9000 ( 21.96% ) in 14.51 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 10000 ( 24.40% ) in 16.07 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 11000 ( 26.84% ) in 17.65 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 12000 ( 29.28% ) in 19.20 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 13000 ( 31.72% ) in 20.75 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 14000 ( 34.16% ) in 22.27 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 15000 ( 36.60% ) in 23.80 minutes. Items per second: 11
SLIMElasticNetRecommender: Processed 16000 ( 39.04% ) in 25.33 minutes. Items per second: 11
SLIMElasticNetRecommender: Processed 17000 ( 41.48% ) in 26.85 minutes. Items per second: 11
SLIMElasticNetRecommender: Processed 18000 ( 43.92% ) in 28.36 minutes. Items per second: 11
SLIMElasticNetRecommender: Processed 19000 ( 46.36% ) in 29.86 minutes. Items per second: 11
SLIMElasticNetRecommender: Processed 20000 ( 48.80% ) in 31.38 minutes. Items per second: 11
SLIMElasticNetRecommender: Processed 21000 ( 51.24% ) in 32.86 minutes. Items per second: 11
SLIMElasticNetRecommender: Processed 22000 ( 53.68% ) in 34.37 minutes. Items per second: 11
SLIMElasticNetRecommender: Processed 23000 ( 56.12% ) in 36.17 minutes. Items per second: 11
SLIMElasticNetRecommender: Processed 24000 ( 58.56% ) in 38.50 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 25000 ( 61.00% ) in 41.11 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 26000 ( 63.44% ) in 43.57 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 27000 ( 65.88% ) in 45.62 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 28000 ( 68.32% ) in 47.66 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 29000 ( 70.76% ) in 50.04 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 30000 ( 73.20% ) in 52.47 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 31000 ( 75.64% ) in 54.80 minutes. Items per second: 9
SLIMElasticNetRecommender: Processed 32000 ( 78.08% ) in 57.02 minutes. Items per second: 9
SLIMElasticNetRecommender: Processed 33000 ( 80.53% ) in 59.22 minutes. Items per second: 9
SLIMElasticNetRecommender: Processed 34000 ( 82.97% ) in 61.44 minutes. Items per second: 9
SLIMElasticNetRecommender: Processed 35000 ( 85.41% ) in 63.63 minutes. Items per second: 9
SLIMElasticNetRecommender: Processed 36000 ( 87.85% ) in 65.63 minutes. Items per second: 9
SLIMElasticNetRecommender: Processed 37000 ( 90.29% ) in 67.55 minutes. Items per second: 9
SLIMElasticNetRecommender: Processed 38000 ( 92.73% ) in 69.18 minutes. Items per second: 9
SLIMElasticNetRecommender: Processed 39000 ( 95.17% ) in 71.06 minutes. Items per second: 9
SLIMElasticNetRecommender: Processed 40000 ( 97.61% ) in 72.88 minutes. Items per second: 9
SLIMElasticNetRecommender: Processed 40981 ( 100.00% ) in 74.31 minutes. Items per second: 9
Generate recommend list...
0 ok, hit = 7
2000 ok, hit = 5807
4000 ok, hit = 10084
6000 ok, hit = 13942
8000 ok, hit = 17545
10000 ok, hit = 21078
12000 ok, hit = 24280
14000 ok, hit = 27408
16000 ok, hit = 30634
18000 ok, hit = 33883
20000 ok, hit = 36912
22000 ok, hit = 39926
24000 ok, hit = 42782
26000 ok, hit = 45476
28000 ok, hit = 48035
HR@20 = 0.5564337865898586, Recall@20 = 0.1698757481743364, NDCG@20 = 0.13821795172035953, HR@50 = 0.6960278652287494, Recall@50 = 0.2657772365068036, NDCG@50 = 0.16866578711178626, 
Finished
```