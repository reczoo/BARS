# Benchmarks

To allow extensibility of the BARS benchmark, the code implementation is not binded to any single framework or library. 



## Model List

The following models have been benchmarked with open-source code and detailed reproducing steps.

| Publication |    Model   |  Paper Title                                                                                      |
| ----:|:----------:|:--------------------------------------------------------------------------------------------|
|  WWW'01    |   ItemCF  |        Item-Based Collaborative Filtering Recommendation Algorithms                                                                                        |
| UAI'09 |   MF-BPR   |      BPR: Bayesian Personalized Ranking from Implicit Feedback                         |
| ICDM'11 |    SLIM    |    SLIM: Sparse Linear Methods for Top-N Recommender Systems                        |
| RecSys'16 | YoutubeNet |    Deep Neural Networks for YouTube Recommendations                               |
| WWW'17 |    NeuMF   |       Neural Collaborative Filtering                                                    |
| WWW'17 |     CML    |     Collaborative Metric Learning                                                     |
| SIGIR'19 |    NGCF    |   Neural Graph Collaborative Filtering                                            |
| WWW'19 |    EASE^R    |    Embarrassingly Shallow Autoencoders for Sparse Data                                         |
| AAAI'20 |  LR-GCCF  |    Revisiting Graph based Collaborative Filtering: A Linear Residual Graph Convolutional Network Approach |
| SIGIR'20 |  LightGCN  |   LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation |
| TOIS'20 |    ENMF    |      Efficient Neural Matrix Factorization without Sampling for Recommendation        |
| CIKM'21 |    SimpleX    |    SimpleX: A Simple and Strong Baseline for Collaborative Filtering  |


## AmazonBooks

:pushpin: Note that we fix **embedding_dim=64** following the setting in NGCF/LightGCN for fair comparisons.

|  Publication |     Models      | Recall@20  | Recall@50 |  NDCG@20   | NDCG@50 | HitRate@20 | HitRate@50 |                Steps-to-Reproduce                | Contributed-by                                               |
| -----------: | :-------------: | :--------: | :-------: | :--------: | :-----: | :--------: | :--------: | :----------------------------------------------: | ------------------------------------------------------------ |
|              |     ItemPop     |   0.0051   |  0.0101   |   0.0044   | 0.0061  |   0.0419   |   0.0764   |   [link](./ItemPop/ItemPop_amazonbooks_x1.md)    | Kelong Mao                                                   |
|     WWW'2001 |     ItemKNN     |   0.0736   |  0.1175   |   0.0606   | 0.0771  |   0.3765   |   0.5234   |   [link](./ItemKNN/ItemKNN_amazonbooks_x1.md)    | Jinpeng Wang                                                 |
|     UAI'2009 |     MF-BPR      | ~~0.0250~~ |     /     | ~~0.0196~~ |    /    |     /      |     /      |                        /                         | [Reported by NGCF paper](https://arxiv.org/abs/1905.08108)   |
|              |  MF-BPR (ours)  |   0.0338   |  0.0660   |   0.0261   | 0.0380  |   0.2103   |   0.3530   |                       link                       | XUEPAI                                                       |
|    ICDM'2011 |      SLIM       |   0.0755   |  0.1257   |   0.0602   | 0.0791  |   0.3873   |   0.5472   |      [link](./SLIM/SLIM_amazonbooks_x1.md)       | Kelong Mao                                                   |
|    NIPS'2005 |      GRMF       |   0.0354   |     /     |   0.0270   |    /    |     /      |     /      |                        /                         | [Reported by LightGCN paper](https://arxiv.org/abs/2002.02126) |
|    MLSP'2016 |    Item2Vec     |   0.0326   |  0.0623   |   0.0252   | 0.0361  |   0.1897   |   0.3192   |  [link](./Item2Vec/Item2Vec_amazonbooks_x1.md)   | Yi Li                                                        |
|  RecSys'2016 |   YoutubeDNN    |   0.0502   |  0.0924   |   0.0388   | 0.0545  |   0.2757   |   0.4354   |                       link                       | XUEPAI                                                       |
|     WWW'2017 |      NeuMF      |   0.0258   |     /     |   0.0200   |    /    |     /      |     /      |                        /                         | [Reported by NGCF paper](https://arxiv.org/abs/1905.08108)   |
|     WWW'2017 |       CML       |   0.0522   |  0.0953   |   0.0428   | 0.0591  |   0.2840   |   0.4410   |       [link](./CML/CML_amazonbooks_x1.md)        | Jinpeng Wang                                                 |
|   SIGIR'2018 |       CMN       |   0.0267   |     /     |   0.0218   |    /    |     /      |     /      |                        /                         | [Reported by NGCF paper](https://arxiv.org/abs/1905.08108)   |
|    KDDW'2018 |      GC-MC      |   0.0288   |     /     |   0.0224   |    /    |     /      |     /      |                        /                         | [Reported by NGCF paper](https://arxiv.org/abs/1905.08108)   |
|     KDD'2018 |     PinSage     |   0.0282   |     /     |   0.0219   |    /    |     /      |     /      |                        /                         | [Reported by NGCF paper](https://arxiv.org/abs/1905.08108)   |
|  RecSys'2018 |     HOP-Rec     |   0.0309   |     /     |   0.0232   |    /    |     /      |     /      |                        /                         | [Reported by NGCF paper](https://arxiv.org/abs/1905.08108)   |
|    ICLR'2018 |       GAT       |   0.0326   |     /     |   0.0235   |    /    |     /      |     /      |                        /                         | [Reported by NAT4Rec paper](https://arxiv.org/abs/2010.12256) |
|     WWW'2018 |     MultVAE     |   0.0407   |     /     |   0.0315   |    /    |     /      |     /      |                        /                         | [Reported by LightGCN paper](https://arxiv.org/abs/2002.02126) |
|    ICML'2019 |    DisenGCN     |   0.0329   |     /     |   0.0254   |    /    |     /      |     /      |                        /                         | [Reported by DGCF paper](https://arxiv.org/pdf/2007.01764)   |
|   SIGIR'2019 |      NGCF       |   0.0344   |     /     |   0.0263   |    /    |     /      |     /      |                        /                         | [Reported by NGCF paper](https://arxiv.org/abs/1905.08108)   |
| NeurIPS'2019 |    MacridVAE    |   0.0383   |     /     |   0.0295   |    /    |     /      |     /      |                        /                         | [Reported by DGCF paper](https://arxiv.org/pdf/2007.01764)   |
|    ICDM'2019 |   Multi-GCCF    |   0.0363   |     /     |     /      |    /    |     /      |     /      |                        /                         | [Reported by Multi-GCCF  paper](https://arxiv.org/abs/2001.00267) |
|     WWW'2019 |     EASE^R      |   0.0710   |  0.1177   |   0.0567   | 0.0744  |   0.3710   |   0.5293   |     [link](./EASE_r/EASE_amazonbooks_x1.md)      | XUEPAI                                                       |
|    TOIS'2020 |      ENMF       |   0.0359   |  0.0691   |   0.0281   | 0.0404  |   0.2187   |   0.3649   |      [link](./ENMF/ENMF_amazonbooks_x1.md)       | Jinpeng Wang                                                 |
|    AAAI'2020 |     LR-GCCF     |   0.0335   |           |   0.0265   |         |   0.0349   |            |   [link](./LR-GCCF/LR-GCCF_amazonbooks_x1.md)    | Yi Li                                                        |
|   SIGIR'2020 |     NIA-GCN     |   0.0369   |     /     |   0.0287   |    /    |     /      |     /      |                        /                         | [Reported by NAT4Rec paper](https://arxiv.org/abs/2010.12256) |
|   SIGIR'2020 |    LightGCN     |   0.0411   |     /     |   0.0315   |    /    |     /      |     /      |                        /                         | [Reported by LightGCN paper](https://arxiv.org/abs/2002.02126) |
|              | LightGCN (ours) |   0.0411   |  0.0799   |   0.0318   | 0.0461  |   0.2423   |   0.4019   | [link](./LightGCN/LightGCN_TF_amazonbooks_x1.md) | Yi Li                                                        |
|   SIGIR'2020 |      DGCF       |   0.0422   |     /     |   0.0324   |    /    |     /      |     /      |                        /                         | [Reported by DGCF paper](https://arxiv.org/pdf/2007.01764)   |
|   Arxiv'2020 |    NGAT4Rec     |   0.0457   |     /     |   0.0358   |    /    |     /      |     /      |                        /                         | [Reported by NAT4Rec paper](https://arxiv.org/abs/2010.12256) |
|   Arxiv'2020 |     SGL-ED      |   0.0478   |     /     |   0.0379   |    /    |     /      |     /      |                        /                         | [Reported by SGL-ED paper](https://arxiv.org/pdf/2010.10783.pdf) |

## Yelp18

:pushpin: Note that we fix **embedding_dim=64** following the setting in NGCF/LightGCN for fair comparisons.

| Publication |  Models |   Recall@20   |   Recall@50   |   NDCG@20   |   NDCG@50   |   HitRate@20   |   HitRate@50   | Steps-to-Reproduce | Contributed-by |
| -------------: | :--------------------------:|:-------------:|:-------------:|:-----------:|:-----------:|:--------------:|:--------------:|:------------------:|----------------|
| |                 ItemPop     |     0.0124          |   0.0242            |    0.0101         |      0.0145       |        0.0831        |     0.1493           |     [link](./ItemPop/ItemPop_yelp18_x1.md)               |      Kelong Mao          |
| WWW'2001 |                     ItemKNN |   0.0639            |   0.1219            |    0.0531         |     0.0746        |      0.3876          |    0.5753            |     [link](./ItemKNN/ItemKNN_yelp18_x1.md)               |      Jinpeng Wang          |
| UAI'2009 |                    MF-BPR |      ~~0.0433~~         |      /         |   ~~0.0354~~          |      /        |         /         |      /           |       /     |     [Reported by NGCF paper](https://arxiv.org/abs/1905.08108)            |
| |                       MF-BPR (ours)               |    0.0576           |      0.1123         |     0.0468        |   0.0671          |      0.3624          |       0.5577         |      link      |       XUEPAI         |
| ICDM'2011 |                        SLIM |    0.0646           |     0.1213          |    0.0541         |     0.0751        |    0.3912            |    0.5799            |     [link](./SLIM/SLIM_yelp18_x1.md)                 |     Kelong Mao              |
| MLSP'2016 | Item2Vec | 0.0503 | 0.0971 | 0.0411 | 0.0585 | 0.3267 | 0.5114 | [link](./Item2Vec/Item2Vec_yelp18_x1.md) | Yi Li |
| WWW'2017 |                         CML |     0.0622          |   0.1181            |   0.0536          |   0.0738          |      0.3810          |     0.5510           |     [link](./CML/CML_yelp18_x1.md)                |     Jinpeng Wang               |
|    WWW'2019 |                     EASE^R |     0.0657          |    0.1225           |     0.0552        |   0.0762          |      0.3966          |    0.5839            |    [link](./EASE_r/EASE_yelp18_x1.md)                |    XUEPAI            |
|     TOIS'2020 |                 ENMF |    0.0624          |  0.1189         |  0.0515       |     0.0723      |      0.3848         |    0.5792       |   [link](./ENMF/ENMF_yelp18_x1.md)               |    Jinpeng Wang            |
| AAAI'2020 | LR-GCCF | 0.0558 |  | 0.0343 |  | 0.0561 |  | [link](./LR-GCCF/LR-GCCF_yelp18_x1.md) | Yi Li |
| SIGIR'2020 | LightGCN | 0.0649 | / | 0.0530 | / | / | / | / | [Reported by LightGCN paper](https://arxiv.org/abs/2002.02126) |
|  | LightGCN (ours) | 0.0653 | 0.1254 | 0.0532 | 0.0756 | 0.3974 | 0.5922 | [link](./LightGCN/LightGCN_TF_yelp18_x1.md) | Yi Li |



## Gowalla

:pushpin: Note that we fix **embedding_dim=64** following the setting in NGCF/LightGCN for fair comparisons.

| Publication |  Models |   Recall@20   |   Recall@50   |   NDCG@20   |   NDCG@50   |   HitRate@20   |   HitRate@50   | Steps-to-Reproduce | Contributed-by |
| -----------: | :----------------------------:|:-------------:|:-------------:|:-----------:|:-----------:|:--------------:|:--------------:|:------------------:|----------------|
|  |               ItemPop     |    0.0416           |    0.0624           |     0.0317        |     0.0379        |       0.2038         |      0.2777          |     [link](./ItemPop/ItemPop_gowalla_x1.md)               |     Kelong Mao           |
|    WWW'2001 |             ItemKNN  |   0.1570           |   0.2549            |    0.1214         |     0.1527       |      0.5094         |    0.6650            |     [link](./ItemKNN/ItemKNN_gowalla_x1.md)               |      Jinpeng Wang          |
|    UAI'2009 |           MF-BPR |   ~~0.1291~~         |       -        |     ~~0.1109~~        |    -           |     -             |       -           |      -        |          [Reported by NGCF paper](https://arxiv.org/abs/1905.08108)   |
|          |         MF-BPR (ours)                 |   0.1627            |    0.2533           |    0.1378         |   0.1662          |        0.5544        |    0.6936            |      link      |       XUEPAI         |
|     ICDM'2011 |                   SLIM |    0.1699           |    0.2658           |    0.1382         |     0.1687        |       0.5564         |    0.6960            |   [link](./SLIM/SLIM_gowalla_x1.md)                  |     Kelong Mao            |
| MLSP'2016 | Item2Vec | 0.1326 | 0.2158 | 0.1057 | 0.1320 | 0.4743 | 0.6188 | [link](./Item2Vec/Item2Vec_gowalla_x1.md) | Yi Li |
|          WWW'2017 |               CML |   0.1670            |   0.2602            |   0.1292          |     0.1587        |      0.5410          |        0.6750        |   [link](./benchmarks/CML/CML_gowalla_x1.md)                  |     Jinpeng Wang              |
|        WWW'2019 |            EASE^R |    0.1765           |    0.2701           |    0.1467         |     0.1760        |      0.5727          |      0.7081          |  [link](./EASE_r/EASE_gowalla_x1.md)                  |     XUEPAI           |
|        TOIS'2020 |              ENMF |    0.1523        |  0.2379        |  0.1315     |     0.1583     |     0.5336      |   0.6701    |   [link](./ENMF/ENMF_gowalla_x1.md)               |    Jinpeng Wang            |
| AAAI'2020 | LR-GCCF | 0.1519 |  | 0.1285 |  | 0.1555 |  | [link](./LR-GCCF/LR-GCCF_gowalla_x1.md) | Yi Li |
| SIGIR'2020 | LightGCN | 0.1830 | / | 0.1550 | / | / | / | / | [Reported by LightGCN paper](https://arxiv.org/abs/2002.02126) |
|  | LightGCN (ours) | 01820 | 0.2821 | 0.1547 | 0.1859 | 0.5924 | 0.7295 | [link](./LightGCN/LightGCN_TF_gowalla_x1.md) | Yi Li |
