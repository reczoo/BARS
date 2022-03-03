# BARS-CTR Datasets


## Reusable Data Splits

| Dataset   | Dataset ID   | Benchmark Protocol     |  Used by                           |
|:-----------|:--------------------|:------------------------|:---------------------------------------------|
| [Criteo](./Criteo)    | [Criteo_x1](./Criteo/README.md#Criteo_x1)          | default     |  [Cheng et al., AAAI'20](https://ojs.aaai.org/index.php/AAAI/article/view/5768)     |
|           | [Criteo_x2](./Criteo/README.md#Criteo_x2)          |  default        |  [Liu et al., SIGIR'20](https://dl.acm.org/doi/abs/10.1145/3397271.3401082)    |
|           | [Criteo_x3](./Criteo/README.md#Criteo_x3)          | default      |  [Sun et al., WWW'21](https://arxiv.org/abs/2102.12994)    |
|           | [Criteo_x4](./Criteo/README.md#Criteo_x4)          | [Criteo_x4_001](./Criteo/README.md#Criteo_x4_001)          |  [Song et al., CIKM'20](https://arxiv.org/abs/1810.11921)    |
|           |                    | [Criteo_x4_002](./Criteo/README.md#Criteo_x4_002)           | [Zhu et al., CIKM'21](https://arxiv.org/abs/2009.05794)   |
| [Avazu](./Avazu)     | [Avazu_x1](./Avazu/README.md#Avazu_x1)           | default        |  [Cheng et al., AAAI'20](https://ojs.aaai.org/index.php/AAAI/article/view/5768)   |
|           | [Avazu_x2](./Avazu/README.md#Avazu_x2)          | default        |  [Liu et al., SIGIR'20](https://dl.acm.org/doi/abs/10.1145/3397271.3401082)    |
|           | [Avazu_x3](./Avazu/README.md#Avazu_x3)           | default         |  [Sun et al., WWW'21](https://arxiv.org/abs/2102.12994)   |
|           | [Avazu_x4](./Avazu/README.md#Avazu_x4)           | [Avazu_x4_001](./Avazu/README.md#Avazu_x4_001)           |  [Song et al., CIKM'20](https://arxiv.org/abs/1810.11921)   |
|           |                    | [Avazu_x4_002](./Avazu/README.md#Avazu_x4_002)           | [Zhu et al., CIKM'21](https://arxiv.org/abs/2009.05794)    |
| [KKBox](./KKBox)     | [KKBox_x1](./KKBox/README.md#KKBox_x1)           | default        |  TBA  |
| [Taobao](./Taobao)    | [Taobao_x1](./Taobao/README.md#Taobao_x1)          | default        |  TBA  |
|     | [Taobao_x2](./Taobao/README.md#Taobao_x2)          | default       |  [Feng et al., IJCAI'19](https://arxiv.org/abs/1905.06482)  |
| [MicroVideo1.7M](./MicroVideo1.7M)    | [MicroVideo1.7M_x1](./MicroVideo1.7M/README.md#MicroVideo17M_x1)          |  default        | [Chen et al., MM'18](https://dl.acm.org/doi/abs/10.1145/3240508.3240617) |
| [Frappe](./Frappe)    | [Frappe_x1](./Frappe/README.md#Frappe_x1)          | default         |  [Cheng et al., AAAI'20](https://ojs.aaai.org/index.php/AAAI/article/view/5768) |
| [MovieLens](./MovieLens) | [MovielensLatest_x1](./MovieLens/README.md#MovielensLatest_x1) | default | [Cheng et al., AAAI'20](https://ojs.aaai.org/index.php/AAAI/article/view/5768)  |


## Tracking Records

We track dataset splits from the published papers in order to make the research results reproducible and reusable. We directly reuse the data splits or preprocessing steps if a paper has open the details. If not, we request the data splits by sending emails to the authors.


| Dataset Splits    |  Paper   |   
|:-----------|:--------------------|
|  | [**WWW'21**] [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/pdf/2008.13535.pdf), Ruoxi Wang, Rakesh Shivanna, Derek Z. Cheng, Sagar Jain, Dong Lin, Lichan Hong, Ed H. Chi.    |    
|   |  [**WWW'21**] [FM^2: Field-matrixed Factorization Machines for Recommender Systems](https://arxiv.org/abs/2102.12994v2), Yang Sun, Junwei Pan, Alex Zhang, Aaron Flores.    |   
|  [Criteo_x1](./Criteo/README.md#Criteo_x1), [Avazu_x1](./Avazu/README.md#Avazu_x1), Frappe_x1, MovielensLatest_x1     |  [**AAAI'20**] [Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions](https://ojs.aaai.org/index.php/AAAI/article/view/5768), Weiyu Cheng, Yanyan Shen, Linpeng Huang.    |
 |  [Criteo_x4](./Criteo/README.md#Criteo_x4), [Avazu_x4](./Avazu/README.md#Avazu_x4) |  [**CIKM'19**] [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921), Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, Jian Tang.      |


