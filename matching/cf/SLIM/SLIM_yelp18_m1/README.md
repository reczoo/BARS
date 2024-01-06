## SLIM_yelp18_x0

A notebook to benchmark SLIM on Yelp2018 dataset.

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

    `Hyperparameters: alpha tuned from [0.1, 0.02, 0.01, 0.015, 0.005], l1 tuned from [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]`

    ```bash
    python -u SLIM.py --dataset=Yelp18 --topk=[20,50] --alpha=0.01 --l1=0.0001
    ```

### Results
```bash
HR@20 = 0.3912151067323481, Recall@20 = 0.06456412147121535, NDCG@20 = 0.05407667471159522, HR@50 = 0.5798913729948213, Recall@50 = 0.1212562152408041, NDCG@50 = 0.07512081886959462
```


### Logs
```bash
Namespace(alpha=0.01, dataset='Yelp18', l1=0.0001, topk='[20,50]')
user num: 31668, item num: 38048
model fitting...
SLIMElasticNetRecommender: Processed 1000 ( 2.63% ) in 1.39 minutes. Items per second: 12
SLIMElasticNetRecommender: Processed 2000 ( 5.26% ) in 2.89 minutes. Items per second: 12
SLIMElasticNetRecommender: Processed 3000 ( 7.88% ) in 4.96 minutes. Items per second: 10
SLIMElasticNetRecommender: Processed 4000 ( 10.51% ) in 7.22 minutes. Items per second: 9
SLIMElasticNetRecommender: Processed 5000 ( 13.14% ) in 9.47 minutes. Items per second: 9
SLIMElasticNetRecommender: Processed 6000 ( 15.77% ) in 11.70 minutes. Items per second: 9
SLIMElasticNetRecommender: Processed 7000 ( 18.40% ) in 13.93 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 8000 ( 21.03% ) in 16.10 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 9000 ( 23.65% ) in 18.29 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 10000 ( 26.28% ) in 20.43 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 11000 ( 28.91% ) in 22.62 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 12000 ( 31.54% ) in 24.69 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 13000 ( 34.17% ) in 26.93 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 14000 ( 36.80% ) in 29.13 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 15000 ( 39.42% ) in 31.34 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 16000 ( 42.05% ) in 33.41 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 17000 ( 44.68% ) in 35.62 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 18000 ( 47.31% ) in 37.75 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 19000 ( 49.94% ) in 39.86 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 20000 ( 52.57% ) in 42.12 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 21000 ( 55.19% ) in 44.32 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 22000 ( 57.82% ) in 46.56 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 23000 ( 60.45% ) in 48.89 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 24000 ( 63.08% ) in 51.27 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 25000 ( 65.71% ) in 53.67 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 26000 ( 68.33% ) in 55.96 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 27000 ( 70.96% ) in 58.26 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 28000 ( 73.59% ) in 60.57 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 29000 ( 76.22% ) in 62.80 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 30000 ( 78.85% ) in 65.06 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 31000 ( 81.48% ) in 67.29 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 32000 ( 84.10% ) in 69.55 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 33000 ( 86.73% ) in 71.83 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 34000 ( 89.36% ) in 74.12 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 35000 ( 91.99% ) in 76.29 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 36000 ( 94.62% ) in 78.48 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 37000 ( 97.25% ) in 80.71 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 38000 ( 99.87% ) in 82.79 minutes. Items per second: 8
SLIMElasticNetRecommender: Processed 38048 ( 100.00% ) in 82.86 minutes. Items per second: 8
Generate recommend list...
0 ok, hit = 0
2000 ok, hit = 3214
4000 ok, hit = 6011
6000 ok, hit = 8707
8000 ok, hit = 11107
10000 ok, hit = 13527
12000 ok, hit = 15747
14000 ok, hit = 17874
16000 ok, hit = 20023
18000 ok, hit = 22018
20000 ok, hit = 23997
22000 ok, hit = 25928
24000 ok, hit = 27945
26000 ok, hit = 29975
28000 ok, hit = 32047
30000 ok, hit = 34119
HR@20 = 0.3912151067323481, Recall@20 = 0.06456412147121535, NDCG@20 = 0.05407667471159522, HR@50 = 0.5798913729948213, Recall@50 = 0.1212562152408041, NDCG@50 = 0.07512081886959462, 
Finished
```