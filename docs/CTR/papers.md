# CTR Prediction

A curated list of CTR prediction models

### Model List


``````{tab-set}
`````{tab-item} 2023

````{admonition} Feature Interaction
```{table}
:align: left
:width: 94%
|   |   |   |    |    | |
| :---------:|:------:|:------:|:------:|:------:|:------:|
|  SIGIR'23  | [FinalNet](https://dl.acm.org/doi/10.1145/3539618.3591988) {cite}`FinalNet`<br>Huawei   |  AAAI'23 | [FinalMLP](https://arxiv.org/abs/2304.00902) {cite}`FinalMLP`<br>Huawei    | SIGIR'23  | [EulerNet](https://arxiv.org/abs/2304.10711) {cite}`EulerNet`<br>Huawei | 
| CIKM'23 |  [GDCN](https://arxiv.org/abs/2311.04635) {cite}`GDCN`<br>Microsoft  |  CIKM'23  | [MemoNet](https://arxiv.org/abs/2211.01334) {cite}`MemoNet`<br>Sina Weibo | 

```
````
````{admonition} Behaviour Sequence Modeling
:class: tip
```{table}
:align: left
:width: 94%
|   |   |   |    |    | |
| :---------:|:------:|:------:|:------:|:------:|:------:|
| KDD'23  | [TWIN](https://arxiv.org/abs/2302.02352) {cite}`TWIN`<br>Kuaishou | CIKM'23 | [DCIN](https://arxiv.org/pdf/2308.06037.pdf) {cite}`DCIN`<br>Meituan  |  
```
````
````{admonition} Multi-Domain Learning
:class: important
```{table}
:align: left
:width: 94%
|   |   |   |    |    | |
| :---------:|:------:|:------:|:------:|:------:|:------:|
| KDD'23  | [SATrans](https://dl.acm.org/doi/10.1145/3580305.3599936) {cite}`SATrans`<br>Tencent |  
```
````
````{admonition} Pretraining
:class: warning
```{table}
:align: left
:width: 94%
|   |   |   |    |    | |
| :---------:|:------:|:------:|:------:|:------:|:------:|
| KDD'23  | [MAP](https://arxiv.org/abs/2308.01737) {cite}`MAP`<br>Huawei |  KDD'23  | [BERT4CTR](https://arxiv.org/abs/2308.11527) {cite}`BERT4CTR`<br>Microsoft | 
```
````

`````
`````{tab-item} 2022

````{admonition} Feature Interaction
```{table}
:align: left
:width: 94%
|   |   |   |    |    | |
| :---------:|:------:|:------:|:------:|:------:|:------:|
|  SIGIR'22  |  [FRNet](https://arxiv.org/abs/2204.08758) {cite}`FRNet`<br>Microsoft |  NeurIPS'22  | [APG](https://arxiv.org/abs/2203.16218) {cite}`APG`<br>Alibaba   |  ICASSP'22 | [FINT](https://arxiv.org/abs/2107.01999) {cite}`FINT`<br>iQIYI    | 

```
````
````{admonition} Behaviour Sequence Modeling
:class: tip
```{table}
:align: left
:width: 94%
|   |   |   |    |    | |
| :---------:|:------:|:------:|:------:|:------:|:------:|
| CIKM'22 | [SDIM](https://arxiv.org/abs/2205.10249) {cite}`SDIM`<br>Meituan  |  SDM'22 | [DINMP](https://arxiv.org/abs/2104.06312) {cite}`DINMP`<br>Alibaba | 
```
````

`````
`````{tab-item} 2021

````{admonition} Feature Interaction
```{table}
:align: left
:width: 94%
|   |   |   |    |    | |
| :---------:|:------:|:------:|:------:|:------:|:------:|
| WWW'21  |  [DCN-V2](https://arxiv.org/abs/2008.13535) {cite}`DCNv2`<br>Google  | WWW'21 |  [FM2](https://arxiv.org/abs/2102.12994) {cite}`FM2`<br>Yahoo  | CIKM'21  |  [EDCN](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf) {cite}`EDCN`<br>Huawei |
| CIKM'21  | [DESTINE](https://arxiv.org/abs/2101.03654) {cite}`DESTINE`<br>Alibaba | SIGIR'21   |  [SAM](https://arxiv.org/abs/2105.05563) {cite}`SAM`<br>BOSS Zhipin  | SIGIR'21 |  [PCF-GNN](https://arxiv.org/abs/2105.07752) {cite}`PCF-GNN`<br>Alibaba |
|   SIGIR'21     | [xLightFM](https://dl.acm.org/doi/10.1145/3404835.3462941) {cite}`xLightFM` |   KDD'21    | [AOANet](https://dl.acm.org/doi/10.1145/3447548.3467133) {cite}`AOANet`<br>Didi Chuxing  |    CIKM'21    | [DCAP](https://arxiv.org/abs/2105.08649) {cite}`DCAP` | |
```
````
````{admonition} Behaviour Sequence Modeling
:class: tip
```{table}
:align: left
:width: 94%
|   |   |   |    |    | |
| :---------:|:------:|:------:|:------:|:------:|:------:|
| TKDD'21  |  [CIN](https://dl.acm.org/doi/fullHtml/10.1145/3428079) {cite}`CIN` |  CIKM'21 | [HyperCTR](https://arxiv.org/pdf/2109.02398) {cite}`HyperCTR` | 
```
````
````{admonition} Multi-Domain/Multi-Task Learning
:class: important
```{table}
:align: left
:width: 94%
|   |   |   |    |    | |
| :---------:|:------:|:------:|:------:|:------:|:------:|
|  CIKM'21   | [STAR](https://arxiv.org/abs/2101.11427) {cite}`STAR`<br>Alibaba |    KDD'21 | [DASL](https://arxiv.org/abs/2106.02768) {cite}`DASL`<br>Alibaba  |  CIKM'21  | [MetaCTR](https://dl.acm.org/doi/abs/10.1145/3459637.3481912) {cite}`MetaCTR`<br>Baidu |  

```
````
````{admonition} Embedding Learning
:class: warning
```{table}
:align: left
:width: 94%
|   |   |   |    |    | |
| :---------:|:------:|:------:|:------:|:------:|:------:|
|  KDD'21  | [AutoDis](https://arxiv.org/abs/2012.08986) {cite}`AutoDis`<br>Huawei |    KDD'21 | [DG-ENN](https://arxiv.org/abs/2106.00314) {cite}`DG-ENN`<br>Huawei  |   KDD'21 | [GME](https://arxiv.org/abs/2105.08909) {cite}`GME`<br>Alibaba  |
```
````
`````

`````{tab-item} 2020

````{admonition} Feature Interaction
```{table}
:align: left
:width: 94%
|   |   |   |    |    | |
| :---------:|:------:|:------:|:------:|:------:|:------:|
|   AAAI'20   |  [AFN](https://ojs.aaai.org/index.php/AAAI/article/view/5768) {cite}`AFN` |  CIKM'20   | [DeepIM](https://dl.acm.org/doi/10.1145/3340531.3412077) {cite}`DeepIM`<br>Alibaba  |  SIGIR'20   | [AutoGroup](https://dl.acm.org/doi/abs/10.1145/3397271.3401082) {cite}`AutoGroup`<br>Huawei | 
|  NeurIPS'20  | [FWL](https://arxiv.org/abs/2012.00202) {cite}`FWL` | NeuralNet'20   | [ONN](https://arxiv.org/pdf/1904.12579) {cite}`ONN`  | IJCAI'20 | [DIFM](https://www.ijcai.org/Proceedings/2020/0434.pdf) {cite}`DIFM` |
|   KDD'20   |   [AutoFIS](https://arxiv.org/abs/2003.11235) {cite}`AutoFIS`<br>Huawei | KDD'20   | [AutoCTR](https://arxiv.org/abs/2007.06434) {cite}`AutoCTR`<br>Facebook  |  ICLR'20 |[GLIDER](https://arxiv.org/abs/2006.10966) {cite}`GLIDER`<br>Facebook  | 

```
````
````{admonition} Behaviour Sequence Modeling
:class: tip
```{table}
:align: left
:width: 94%
|   |   |   |    |    | |
| :---------:|:------:|:------:|:------:|:------:|:------:|
| CIKM'20     | [DMIN](https://www.researchgate.net/profile/Luwei-Yang-2/publication/345125472_Deep_Multi-Interest_Network_for_Click-through_Rate_Prediction/links/5f9e1d6b458515b7cfaeffce/Deep-Multi-Interest-Network-for-Click-through-Rate-Prediction.pdf) {cite}`DMIN`<br>Alibaba |  WWW'20 | [MARN](https://arxiv.org/abs/2003.07162) {cite}`MARN`<br>Alibaba | 
```
````
`````
`````{tab-item} 2019

````{admonition} Feature Interaction
```{table}
:align: left
:width: 94%
|   |   |   |    |    | |
| :---------:|:------:|:------:|:------:|:------:|:------:|
|    CIKM'19    | [AutoInt](https://arxiv.org/abs/1810.11921) {cite}`AutoInt` |  CIKM'19     | [FiGNN](https://arxiv.org/abs/1910.05552) {cite}`FiGNN` |   WWW'19    | [FGCNN](https://arxiv.org/abs/1904.04447) {cite}`FGCNN`<br>Huawei | 
| RecSys'19     | [FiBiNET](https://arxiv.org/abs/1905.09433) {cite}`FiBiNET`<br>Sina Weibo  |    AAAI'19         | [HFM](https://ojs.aaai.org//index.php/AAAI/article/view/4448) {cite}`HFM`  |  Arxiv'19    | [DLRM](https://arxiv.org/abs/1906.00091) {cite}`DLRM`<br>Facebook  |
|  IJCAI'19 | [IFM](https://www.ijcai.org/proceedings/2019/203) {cite}`IFM` | 
```
````
````{admonition} Behaviour Sequence Modeling
:class: tip
```{table}
:align: left
:width: 94%
|   |   |   |    |    | |
| :---------:|:------:|:------:|:------:|:------:|:------:|
|    IJCAI'19         |  [DSIN](https://arxiv.org/abs/1905.06482) {cite}`DSIN`<br>Alibaba   | AAAI'19 | [DIEN](https://arxiv.org/abs/1809.03672) {cite}`DIEN`<br>Alibaba  | KDD'19     | [DSTN](https://arxiv.org/abs/1906.03776) {cite}`DSTN`<br>Alibaba | 
| KDD'19     | [MIMN](https://arxiv.org/abs/1905.09248) {cite}`MIMN`<br>Alibaba  | DLP-KDD'19 |  [BST](https://arxiv.org/abs/1905.06874) {cite}`BST`<br>Alibaba  |      SIGIR'19     | [GIN](https://arxiv.org/abs/2103.16164) {cite}`GIN`<br>Alibaba |
```
````
````{admonition} Multi-Task Learning
:class: important
```{table}
:align: left
:width: 94%
|   |   |   |    |    | |
| :---------:|:------:|:------:|:------:|:------:|:------:|
|  IJCAI'19   | [DeepMCP](https://arxiv.org/abs/1906.04365) {cite}`DeepMCP`<br>Alibaba |   SIGIR'19  |  [MetaEmbedding](https://arxiv.org/abs/1904.11547) {cite}`MetaEmbedding`   |
```
````
`````
`````{tab-item} 2018
````{admonition} Feature Interaction
```{table}
:align: left
:width: 94%
|   |   |   |    |    | |
| :---------:|:------:|:------:|:------:|:------:|:------:|
|    WWW'18     | [FwFM](https://dl.acm.org/doi/10.1145/3178876.3186040) {cite}`FwFM`<br>Yahoo | KDD'18     | [xDeepFM](https://arxiv.org/pdf/1803.05170.pdf) {cite}`xDeepFM`<br>Microsoft | 
```
````

````{admonition} Behaviour Sequence Modeling
:class: tip
```{table}
:align: left
:width: 94%
|   |   |   |    |    | |
| :---------:|:------:|:------:|:------:|:------:|:------:|
|  KDD'18 | [DIN](https://www.kdd.org/kdd2018/accepted-papers/view/deep-interest-network-for-click-through-rate-prediction) {cite}`DIN`<br>Alibaba | | |
```
````

`````
`````{tab-item} 2017
````{admonition} Feature Interaction
```{table}
:align: left
:width: 94%
|   |   |   |    |    | |
| :---------:|:------:|:------:|:------:|:------:|:------:|
| SIGIR'17 |  [NFM](https://arxiv.org/abs/1708.05027) {cite}`NFM` | WWW'17 | [FFM](https://arxiv.org/pdf/1701.04099.pdf) {cite}`FFM2`<br>Criteo |   ADKDD'17  |  [DCN](https://arxiv.org/abs/1708.05123) {cite}`DCN`<br>Google | 
|  IJCAI'17         |  [DeepFM](https://arxiv.org/abs/1703.04247) {cite}`DeepFM`<br>Huawei  |  IJCAI'17 | [AFM](https://www.ijcai.org/proceedings/2017/0435.pdf) {cite}`AFM` |


```
````
`````
`````{tab-item} 2016&Before

````{admonition} Feature Interaction
```{table}
:align: left
:width: 94%
|     |  |   |    |   
| :---------:|:------:|:------:|:------:|
|  RecSys'16  | [FFM](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) {cite}`FFM`    |  RecSys'16 |  [YoutubeDNN](https://research.google.com/pubs/archive/45530.pdf) {cite}`YoutubeDNN`<br>Google | 
| ICDM'16| [PNN](https://arxiv.org/pdf/1611.00144.pdf) {cite}`PNN`   | DLRS'16   |[Wide&Deep](https://arxiv.org/pdf/1606.07792.pdf) {cite}`WideDeep`<br>Google | 
|  KDD'16      |    [DeepCrossing](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf) {cite}`DeepCrossing`<br>Microsoft      |       NIPS'16    |     [HOFM](https://arxiv.org/abs/1607.07195) {cite}`HOFM`           |
|MM'16  |  [DeepCTR](https://arxiv.org/abs/1609.06018) {cite}`DeepCTR` | CIKM'15 | [CCPM](https://arxiv.org/abs/1609.06018) {cite}`CCPM` | 
| ADKDD'14 |  [LR+GBDT](https://arxiv.org/abs/1609.06018) {cite}`LR_GBDT`<br>Facebook |KDD'13 |  [FTRL](https://research.google.com/pubs/archive/41159.pdf) {cite}`FTRL`<br>Google | 
|ICDM'10  | [FM](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) {cite}`FM` |  WWW'07 |[LR](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/predictingclicks.pdf) {cite}`LR`<br>Microsoft |

```
````
`````
``````


### Paper List

```{bibliography}
:style: unsrt
:filter: docname in docnames
```
