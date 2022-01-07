# Avazu

+ [Avazu_x1](#Avazu_x1)
+ [Avazu_x2](#Avazu_x2)
+ [Avazu_x3](#Avazu_x3)
+ [Avazu_x4](#Avazu_x4)


It is a [Kaggle challenge dataset](https://www.kaggle.com/c/avazu-ctr-prediction/data) for Avazu CTR prediction. [Avazu](http://avazuinc.com/home/) is one of the leading mobile advertising platforms globally. The Kaggle competition targets at predicting whether a mobile ad will be clicked and has provided 11 days worth of Avazu data to build and test prediction models. It consists of 10 days of labeled click-through data for training and 1 day of ads data for testing (yet without labels). Note that only the first 10 days of labeled data are used for benchmarking. 

Data fields consist of:
+ id: ad identifier (``Note: This column is more like unique sample id, where each row has a distinct value, and thus should be dropped.``)
+ click: 0/1 for non-click/click
+ hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC. (``Note: It is a common practice to bucketize the timestamp into hour, day, is_weekend, and so on.``)
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


Dataset statistics are summarized as follows:

| Dataset  | Total | #Train | #Validation | #Test | 
| :--------: | :-----: |:-----: | :----------: | :----: | 
| Avazu_x1 |  40,428,967     | 28,300,276   |  4,042,897     |  8,085,794    |          
| Avazu_x2 |  40,428,967     | 32,343,173     |      |  8,085,794    |                
| Avazu_x4 |  40,428,967     |  32,343,172   |  4,042,897     | 4,042,898     |                


## Avazu_x1

+ Dataset description

This dataset contains about 10 days of labeled click-through data on mobile advertisements. It has 22 feature fields including user features and advertisement attributes. We reuse the preprocessed data released by the [AFN](https://ojs.aaai.org/index.php/AAAI/article/view/5768) work, which are randomly split into 7:1:2\* as the training set, validation set, and test set, respectively. For consistency of evaluation, we obtain the preprocessed data accessible from the [BARS benchmark](https://github.com/openbenchmark/BARS/click_prediction/datasets). 

\* Note that the released data have a ratio of 7:1:2, which is different from 8:1:1 as reported in the AFN paper. 

+ How to get the dataset?

  + Solution#1: Run the following scripts:
      ```bash
      $ cd datasets/Avazu/Avazu_x1
      $ python download_avazu_x1.py
      $ python convert_avazu_x1.py (please modify the path accordingly)
      ```
  + Solution#2: For ease of reuse, the preprocessed data are available for [downloading here](https://zenodo.org/record/5700987/files/Avazu_x1.zip).
  + Check the md5sum for consistency.
      ```bash
      $ md5sum train.csv valid.csv test.csv
      f1114a07aea9e996842c71648e0f6395  train.csv
      d9568f246357d156c4b8030fadb8b623  valid.csv
      9e2fe9c48705c9315ae7a0953eb57acf  test.csv
      ```

+ Default setting
  
  In this benchmark setting, we follow the AFN work to fix **embedding_dim=10**, **batch_size=4096**, and **MLP_hidden_units=[400, 400, 400]** to make fair comparisons.
  

## Avazu_x2

+ Dataset description

This dataset contains about 10 days of labeled click-through data on mobile advertisements. It has 22 feature fields including user features and advertisement attributes. Following the same setting in the [AutoGroup](https://dl.acm.org/doi/abs/10.1145/3397271.3401082) work, we randomly split 80% of the data for training and validation, and the remaining 20% for testing, respectively. For consistency of evaluation, we directly reuse the preprocessed data accessible from the [BARS benchmark](https://github.com/openbenchmark/BARS/click_prediction/datasets). 

+ How to get the dataset?
  + For ease of reuse, the preprocessed data are available for [downloading here](https://zenodo.org/record/5700987/files/Avazu_x2.zip).
  + Check the md5sum for consistency.
      ```bash
      $ md5sum train.csv test.csv
      c41d786896e2ebe68e08a022199f0ce8  train.csv
      e641ea94c72cdc99b49656d3404f536e  test.csv
      ```

+ Default setting

  For all categorical fields, we filter infrequent features by setting the threshold min_category_count=20 and replace them with a default ``<OOV>`` token.


## Avazu_x3
TBA


+ Default setting



## Avazu_x4

+ Dataset description

This dataset contains about 10 days of labeled click-through data on mobile advertisements. It has 22 feature fields including user features and advertisement attributes. Following the same setting in the [AutoInt](https://arxiv.org/abs/1810.11921) work, we split the data randomly into 8:1:1 as the training set, validation set, and test set, respectively. For better reproduciblity, we directly reuse the code provided by AutoInt and control the random seed (i.e., seed=2018) for splitting. The preprocessed data are accessible from the [BARS benchmark](https://github.com/openbenchmark/BARS/click_prediction/datasets).

+ How to get the dataset?
  + Solution#1: Download the raw dataset, and run the following scripts:
      ```bash
      $ cd datasets/Avazu/Avazu_x4
      $ python split_avazu_x4.py
      ```
  + Solution#2: For ease of reuse, the preprocessed data are available for [downloading here](https://zenodo.org/record/5700987/files/Avazu_x4.zip).
  + Check the md5sum for consistency.
      ```bash
      $ md5sum train.csv valid.csv test.csv
      de3a27264cdabf66adf09df82328ccaa  train.csv
      33232931d84d6452d3f956e936cab2c9  valid.csv
      3ebb774a9ca74d05919b84a3d402986d  test.csv
      ```

#### Avazu_x4_001

In this benchmark setting, we preprocess the data split by removing the ``id`` field that is useless for CTR prediction. In addition, we transform the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we filter infrequent features by setting the threshold min_category_count=2 (performs well) and replace them with a default ``<OOV>`` token. Note that we do not follow the exact preprocessing steps in AutoInt, because the authors neither remove the useless ``id`` field nor specially preprocess the timestamp field. To make a fair comparison, we fix **embedding_dim=16** as with AutoInt.

  
#### Avazu_x4_002

In this setting, we preprocess the data split by removing the ``id`` field that is useless for CTR prediction. In addition, we transform the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we filter infrequent features by setting the threshold min_category_count=1 and replace them with a default ``<OOV>`` token. Note that we found that min_category_count=1 performs the best, which is surprising. We fix **embedding_dim=40** following the existing [FGCNN work](https://arxiv.org/abs/1904.04447).

