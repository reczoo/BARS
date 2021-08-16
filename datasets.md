# Datasets

Indexed
+ [Avazu](#Avazu)
+ [Criteo](#Criteo)
+ [KKBox](#KKBox)
+ [Taobao](#Taobao)


## Avazu
[This is a Kaggle dataset for Avazu CTR prediction challenge.](https://www.kaggle.com/c/avazu-ctr-prediction/data) 

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

+ avazu_x4_001
  
  Note that we fix **embedding_dim=16** for fair comparisons.
  
+ avazu_x4_002


## Criteo
[This is a Kaggle dataset for Criteo display advertising challenge.](https://www.kaggle.com/c/criteo-display-ad-challenge/data) 

[Criteo](https://www.criteo.com/) is a personalized retargeting company that works with Internet retailers to serve personalized online display advertisements to consumers. The goal of this Kaggle challenge is to predict click-through rates on display ads. It offers a week’s worth of data from Criteo's traffic. In the labeled training set over a period of 7 days, each row corresponds to a display ad served by Criteo. The samples are chronologically ordered. Positive (clicked) and negatives (non-clicked) samples have both been subsampled at different rates in order to reduce the dataset size. There are 13 count features and 26 categorical features. The semantic of these features is undisclosed. Some features may have missing values. [The dataset is currently available for downloading at AWS](https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz). Note that we only use the labeled part of data as our benchmarking set. 

Data fields:
+ Label - Target variable that indicates if an ad was clicked (1) or not (0).
+ I1-I13 - A total of 13 columns of integer features (mostly count features).
+ C1-C26 - A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes. 

  
### criteo_x4
Following most of previous work, we randomly split the data into 8:1:1 as the training set, validation set, and test set, respectively. To make it exactly reproducible and easy to compare with existing work, we reuse the code provided by AutoInt and control the random seed (i.e., seed=2018) for splitting. 

+ criteo_x4_001
+ criteo_x4_002

  
## Taobao





