# Benchmarking Results

Indexed by datasets
+ [Avazu](#Avazu)
+ [Criteo](#Criteo)
+ [KKBox](#KKBox)
+ [Taobao](#Taobao)


## Avazu
[A Kaggle dataset for Avazu CTR prediction challenge](https://www.kaggle.com/c/avazu-ctr-prediction/data) 

[Avazu](http://avazuinc.com/home/) is one of the leading mobile advertising platforms globally. This Kaggle competition targets at predicting whether a mobile ad will be clicked and has provided 11 days worth of Avazu data to build and test prediction models. It consists of 10 days of labeled click-through data for training and 1 day of unlabeled ads data for testing. Note that we only use the first 10 days of labeled data as our benchmarking set. 

Data fields:
+ id: ad identifier
+ click: 0/1 for non-click/click
+ hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
+ C1 -- anonymized categorical variable
+ banner_pos
+ site_id
+ site_domain
+ site_category
+ app_id
+ app_domain
+ app_category
+ device_id
+ device_ip
+ device_model
+ device_type
+ device_conn_type
+ C14-C21 -- anonymized categorical variables

### avazu_x4
Following most of previous work, we randomly split the data into 8:1:1 as the training set, validation set, and test set, respectively. To make it exactly reproducible and easy to compare with existing work, we reuse the code provided by AutoInt and control the random seed (i.e., seed=2018) for splitting. 

Especially, we remove the sample\_id field that is useless for CTR prediction. In addition, we transform and expand the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we replace infrequent features (min_category_count=2 after tuning) with a default "<UNK>" feature.

Note that we fix **embedding_dim=16** for fair comparisons.

|  Publication |   Model   | Logloss |   AUC  | Steps-to-Reproduce |
|-------------:|:---------:|:-------:|:------:|:------------------:|
|       -      |     LR    |  0.3815 | 0.7775 |        link        |
|    ICDM'2010 |     FM    |  0.3754 | 0.7887 |        link        |
|    CIKM'2015 |    CCPM   |  0.3745 | 0.7892 |        link        |
|    NIPS'2016 |    HOFM   |  0.3754 | 0.7891 |        link        |
|  RecSys'2016 |    FFM    |  0.3715 | 0.7942 |        link        |
|  RecSys'2016 |    DNN    |  0.3722 | 0.7928 |        link        |
|    ECIR'2016 |    FNN    |  0.3774 | 0.7883 |        link        |
| RecSysW'2016 |  WideDeep |  0.3720 | 0.7929 |        link        |
|    ICDM'2016 |    PNN    |  0.3712 | 0.7944 |        link        |
|     KDD'2016 | DeepCross |  0.3721 | 0.7930 |        link        |
|   SIGIR'2017 |    NFM    |  0.3743 | 0.7894 |        link        |
|   IJCAI'2017 |    AFM    |  0.3792 | 0.7825 |        link        |
|   IJCAI'2017 |   DeepFM  |  0.3719 | 0.7930 |        link        |
|   ADKDD'2017 |  CrossNet |  0.3779 | 0.7840 |        link        |
|   ADKDD'2017 |    DCN    |  0.3719 | 0.7931 |        link        |
|     WWW'2018 |    FwFM   |  0.3744 | 0.7907 |        link        |
|     KDD'2018 |    CIN    |  0.3742 | 0.7894 |        link        |
|     KDD'2018 |  xDeepFM  |  0.3718 | 0.7933 |        link        |
|    AAAI'2019 |    HFM    |  0.3757 | 0.7879 |        link        |
|    AAAI'2019 |    HFM+   |  0.3714 | 0.7944 |        link        |
|     WWW'2019 |   FGCNN   |  0.3711 | 0.7944 |        link        |
|    CIKM'2019 |  AutoInt  |  0.3745 | 0.7891 |        link        |
|    CIKM'2019 |  AutoInt+ |  0.3746 | 0.7902 |        link        |
|    CIKM'2019 |   FiGNN   |  0.3736 | 0.7915 |        link        |
|   Arxiv'2019 |    ONN    |  0.3683 | 0.7992 |        link        |
|  RecSys'2019 |  FiBiNET  |  0.3705 | 0.7953 |        link        |
|    AAAI'2020 | LorentzFM |  0.3756 | 0.7885 |        link        |
|    AAAI'2020 |    AFN    |  0.3740 | 0.7907 |        link        |
|    AAAI'2020 |    AFN+   |  0.3726 | 0.7929 |        link        |
|    WSDM'2020 |  InterHAt |  0.3749 | 0.7882 |        link        |
|    KDDW'2020 |    FLEN   |  0.3719 | 0.7929 |        link        |
|     KDD'2020 |  AutoFIS  |         |        |        TODO        |
|   SIGIR'2020 | AutoGroup |         |        |        TODO        |


## Criteo
[A Kaggle dataset for Criteo display advertising challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/data) 

[Criteo](https://www.criteo.com/) is a personalized retargeting company that works with Internet retailers to serve personalized online display advertisements to consumers. The goal of this Kaggle challenge is to predict click-through rates on display ads. It offers a week’s worth of data from Criteo's traffic. In the labeled training set over a period of 7 days, each row corresponds to a display ad served by Criteo. The samples are chronologically ordered. Positive (clicked) and negatives (non-clicked) samples have both been subsampled at different rates in order to reduce the dataset size. There are 13 count features and 26 categorical features. The semantic of these features is undisclosed. Some features may have missing values. [The dataset is currently available for downloading at AWS](https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz). Note that we only use the labeled part of data as our benchmarking set. 

Data fields:
+ Label - Target variable that indicates if an ad was clicked (1) or not (0).
+ I1-I13 - A total of 13 columns of integer features (mostly count features).
+ C1-C26 - A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes. 

### Criteo_x4
Following most of previous work, we randomly split the data into 8:1:1 as the training set, validation set, and test set, respectively. To make it exactly reproducible and easy to compare with existing work, we reuse the code provided by AutoInt and control the random seed (i.e., seed=2018) for splitting. 


## Taobao





