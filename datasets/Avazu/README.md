# Avazu

+ [Avazu_x0](#Avazu_x0)
  - [Avazu_x0_001](#Avazu_x0_001)
+ [Avazu_x4](#Avazu_x4)
  - [Avazu_x4_001](#Avazu_x4_001)
  - [Avazu_x4_002](#Avazu_x4_002)


It is a [Kaggle challenge dataset for Avazu CTR prediction.](https://www.kaggle.com/c/avazu-ctr-prediction/data). [Avazu](http://avazuinc.com/home/) is one of the leading mobile advertising platforms globally. The Kaggle competition targets at predicting whether a mobile ad will be clicked and has provided 11 days worth of Avazu data to build and test prediction models. It consists of 10 days of labeled click-through data for training and 1 day of unlabeled ads data for testing. Note that only the first 10 days of labeled data are used as the benchmarking set. 

Data fields consist of:
+ id: ad identifier (``Remark: This column is more like unique sample id, where each row has a distinct value, and thus should be dropped.``)
+ click: 0/1 for non-click/click
+ hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC. (``Remark: It is a common practice to bucketize the timestamp into hour, day, is_weekend, and so on.``)
+ C1: anonymized categorical variable
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
+ C14-C21: anonymized categorical variables



## Avazu_x0

This dataset split follows the setting in the AFN work. That is, we randomly split the data into 8:1:1 as the training set, validation set, and test set, respectively. The data preprocessing script is provided on Github and we directly download the preprocessed data.

+ Reproducing steps:
  - Step1: Download the preprocessed data via [the script](./Avazu_x0/download_avazu_x0.py).


### Avazu_x0_001
  
In this setting, we follow the AFN work to fix **embedding_dim=16**, **batch_size=4096**, and **MLP_hidden_units=[400, 400, 400]** to make fair comparisons.
  

  
## Avazu_x4

This dataset split follows the setting in the [AutoInt work](https://arxiv.org/abs/1810.11921). Specifically, we randomly split the data into 8:1:1 as the training set, validation set, and test set, respectively. To make it exactly reproducible, we reuse the code provided by AutoInt and control the random seed (i.e., seed=2018) for splitting. 


+ Reproducing steps:
  + Step1: Download [the raw data](https://www.kaggle.com/c/avazu-ctr-prediction/data).
  + Step2: Split the data using [the script](./Avazu_x4/split_avazu_x4.py).


### Avazu_x4_001

In this setting, we preprocess the data split by removing the ``id`` field that is useless for CTR prediction. In addition, we transform the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we filter infrequent features by setting the threshold min_category_count=2 (performs well) and replace them with a default ``<OOV>`` token. Note that we do not follow the exact preprocessing steps in AutoInt, because the authors neither remove the useless ``id`` field nor specially preprocess the timestamp field. 

To make a fair comparison, we fix **embedding_dim=16** as with AutoInt.

  
### Avazu_x4_002

In this setting, we preprocess the data split by removing the ``id`` field that is useless for CTR prediction. In addition, we transform the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we filter infrequent features by setting the threshold min_category_count=1 and replace them with a default ``<OOV>`` token. Note that we found that min_category_count=1 performs the best, which is surprising.

We fix **embedding_dim=40** following the existing [FGCNN work](https://arxiv.org/abs/1904.04447).

