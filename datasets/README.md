# Datasets

+ [Datasets for CTR Prediction](#Datasets-for-CTR-Prediction)
+ [Datasets for Candidate Item Matching](#Datasets-for-Candidate-Item-Matching)
+ Datasets for Reranking

## Datasets for CTR Prediction

| Dataset   | Dataset ID   |  Used by                           |  Domain  |  Target Topics   |
|:-----------|:--------------------|:------------------------|:-------------------- |:---------------------------------------------|
| [Criteo](./Criteo)    | [Criteo_x1](./Criteo/README.md#Criteo_x1)              |  [Cheng et al., AAAI'20](https://ojs.aaai.org/index.php/AAAI/article/view/5768)     | Ads | Feature interactions |
|           | [Criteo_x2](./Criteo/README.md#Criteo_x2)              |  [Liu et al., SIGIR'20](https://dl.acm.org/doi/abs/10.1145/3397271.3401082)    | Ads | Feature interactions |
|           | [Criteo_x3](./Criteo/README.md#Criteo_x3)              |  [Sun et al., WWW'21](https://arxiv.org/abs/2102.12994)    | Ads |Feature interactions |
|           | [Criteo_x4](./Criteo/README.md#Criteo_x4)              |  [Zhu et al., CIKM'21](https://arxiv.org/abs/2009.05794)    | Ads |Feature interactions |
| [Avazu](./Avazu)     | [Avazu_x1](./Avazu/README.md#Avazu_x1)              |  [Cheng et al., AAAI'20](https://ojs.aaai.org/index.php/AAAI/article/view/5768)   | Ads |Feature interactions |
|           | [Avazu_x2](./Avazu/README.md#Avazu_x2)             |  [Liu et al., SIGIR'20](https://dl.acm.org/doi/abs/10.1145/3397271.3401082)    | Ads |Feature interactions |
|           | [Avazu_x3](./Avazu/README.md#Avazu_x3)               |  [Sun et al., WWW'21](https://arxiv.org/abs/2102.12994)   | Ads |Feature interactions |
|           | [Avazu_x4](./Avazu/README.md#Avazu_x4)                |  [Zhu et al., CIKM'21](https://arxiv.org/abs/2009.05794)  | Ads |Feature interactions |
| [KKBox](./KKBox)     | [KKBox_x1](./KKBox/README.md#KKBox_x1)             |  TBA  | Music | Feature interactions |
| [Frappe](./Frappe)    | [Frappe_x1](./Frappe/README.md#Frappe_x1)              |  [Cheng et al., AAAI'20](https://ojs.aaai.org/index.php/AAAI/article/view/5768) | Apps | Feature interactions |
| [MovieLens](./MovieLens) | [MovielensLatest_x1](./MovieLens/README.md#MovielensLatest_x1) | [Cheng et al., AAAI'20](https://ojs.aaai.org/index.php/AAAI/article/view/5768)  | Movies | Feature interactions |
| [Taobao](./Taobao)    | [TaobaoAds_x1](./Taobao/README.md#TaobaoAds_x1)             |  TBA  | Ads | Feature interactions |
| [Amazon](./Amazon)            | [AmazonElectronics_x1](./Amazon/README.md#AmazonElectronics_x1)       | DIN | Electronics | Sequence |
| [MicroVideo1.7M](./MicroVideo1.7M)    | [MicroVideo1.7M_x1](./MicroVideo1.7M/README.md#MicroVideo17M_x1)               | [Chen et al., MM'18](https://dl.acm.org/doi/abs/10.1145/3240508.3240617) | Video | Sequence, Multimodal |
| KuaiShou            | KuaiVideo_x1      |  TBA | Video | Sequence, Multimodal |

### Datasets for Candidate Item Matching

| Dataset           | Dataset ID           |     Used by           |  Domain | Target Topics                         |
|-------------------|----------------------|:-----------------|:-------------|:----------------------|
| Amazon            | [AmazonBooks_m1](./Amazon#AmazonBooks_m1)       |   [LightGCN, SIGIR'20](https://github.com/kuandeng/LightGCN/tree/master/Data/amazon-book)  | Books | CF, GNN |
|                   | AmazonBooks_m2       |   [ComiRec, KDD'20](https://github.com/THUDM/ComiRec)  | Books |  Multi-interest, Sequential |
|                   | AmazonCDs_m1         |   [BGCF, KDD'20](https://dl.acm.org/doi/abs/10.1145/3394486.3403254)    | CDs | CF, GNN | 
|                   | AmazonMovies_m1      |   [BGCF, KDD'20](https://dl.acm.org/doi/abs/10.1145/3394486.3403254)       | Movies     | CF, GNN |
|                   | AmazonBeauty_m1      |   [BGCF, KDD'20](https://dl.acm.org/doi/abs/10.1145/3394486.3403254)         | Beauty     | CF, GNN | 
|                   | AmazonElectronics_m1 |   [NBPO, SIGIR'20](https://github.com/Wenhui-Yu/NBPO/tree/master/dataset/amazon)  | Electronics | CF | 
| Yelp              | [Yelp18_m1](./Yelp#Yelp18_m1)            |   [LightGCN, SIGIR'20](https://github.com/kuandeng/LightGCN/tree/master/Data/yelp2018)  |  Restaurants | CF, GNN |
| Gowalla           | [Gowalla_m1](./Gowalla#Gowalla_m1)           |   [LightGCN, SIGIR'20](https://github.com/kuandeng/LightGCN/tree/master/Data/gowalla)  | Locations | CF, GNN |
| MovieLens         | MovieLens1M_m1       |   [LCFN, ICML'20](https://github.com/Wenhui-Yu/LCFN/tree/master/dataset/Movielens)               | Movies |    CF, GNN |
|                   | MovieLens1M_m2       |   [NCF, WWW'17](https://github.com/hexiangnan/neural_collaborative_filtering/tree/master/Data)                | Movies |  CF |
| CiteULike-A       | Citeulikea_m1        |   [DHCF](https://github.com/chenchongthu/ENMF#4-dhcf-kdd-2020dual-channel-hypergraph-collaborative-filtering) |  | CF, GNN | 
| Taobao            | Taobao_m1            |   [ComiRec, KDD'20](https://github.com/THUDM/ComiRec) |  | Multi-interest, Sequential |
| KuaiShou          | Kuaishou_m1          |   [NGAT4Rec, Arxiv'21](https://github.com/ShortVideoRecommendation/NGAT4Rec/tree/master/Data/kuaishou) | Video |  CF, GNN | 



## Tracking Records

We track dataset splits from the published papers in order to make the research results reproducible and reusable. We directly reuse the data splits or preprocessing steps if a paper has open the details. If not, we request the data splits by sending emails to the authors.


| Dataset Splits    |  Paper   |   
|:-----------|:--------------------|
|  | [**WWW'21**] [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/pdf/2008.13535.pdf), Ruoxi Wang, Rakesh Shivanna, Derek Z. Cheng, Sagar Jain, Dong Lin, Lichan Hong, Ed H. Chi.    |    
|   |  [**WWW'21**] [FM^2: Field-matrixed Factorization Machines for Recommender Systems](https://arxiv.org/abs/2102.12994v2), Yang Sun, Junwei Pan, Alex Zhang, Aaron Flores.    |   
|  [Criteo_x1](./Criteo/README.md#Criteo_x1), [Avazu_x1](./Avazu/README.md#Avazu_x1), Frappe_x1, MovielensLatest_x1     |  [**AAAI'20**] [Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions](https://ojs.aaai.org/index.php/AAAI/article/view/5768), Weiyu Cheng, Yanyan Shen, Linpeng Huang.    |
 |  [Criteo_x4](./Criteo/README.md#Criteo_x4), [Avazu_x4](./Avazu/README.md#Avazu_x4) |  [**CIKM'19**] [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921), Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, Jian Tang.      |


