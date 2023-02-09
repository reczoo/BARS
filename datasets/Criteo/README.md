# Criteo

+ [Criteo_x1](#Criteo_x1)
+ [Criteo_x2](#Criteo_x2)
+ [Criteo_x3](#Criteo_x1)
+ [Criteo_x4](#Criteo_x4)


The dataset is from a [Kaggle challenge for Criteo display advertising](https://www.kaggle.com/c/criteo-display-ad-challenge/data). Criteo is a personalized retargeting company that works with Internet retailers to serve personalized online display advertisements to consumers. The goal of this Kaggle challenge is to predict click-through rates on display ads. It offers a week's worth of data from Criteo's traffic. In the labeled training set over a period of 7 days, each row corresponds to a display ad served by Criteo. The samples are chronologically ordered. Positive and negatives samples have both been subsampled at different rates in order to reduce the dataset size. There are 13 count features and 26 categorical features. The semantic of these features is undisclosed. Some feature have missing values. Note that only the labeled part (i.e., `train.txt`) of the data is used for benchmarking. 

Data fields consist of:
+ Label: Target variable that indicates if an ad was clicked (1) or not (0).
+ I1-I13: A total of 13 columns of integer features (mostly count features).
+ C1-C26: A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes. 


Dataset statistics are summarized as follows:

| Dataset Split  | Total | #Train | #Validation | #Test | 
| :--------: | :-----: |:-----: | :----------: | :----: | 
| Criteo_x1 |  45,840,617     | 33,003,326   |  8,250,124     | 4,587,167     |             
| Criteo_x2 |   99,616,043    |  86,883,012    |      |  12,733,031    |                
| Criteo_x4 |  45,840,617     |   36,672,493  |   4,584,062    |  4,584,062    |                


## Criteo_x1

+ Dataset description

  The Criteo dataset is a widely-used benchmark dataset for CTR prediction, which contains about one week of click-through data for display advertising. It has 13 numerical feature fields and 26 categorical feature fields. We reuse the preprocessed data released by the [AFN](https://ojs.aaai.org/index.php/AAAI/article/view/5768) work, which are randomly split into 7:2:1\* as the training set, validation set, and test set, respectively. For consistency of evaluation, we obtain the preprocessed data accessible from the [BARS benchmark](https://github.com/openbenchmark/BARS/click_prediction/datasets). 

  \* Note that the released data splits from the authors have a ratio of 7:2:1, which is different from 8:1:1 as reported in the AFN paper. 

+ How to get the dataset?
  + Solution#1: Run the following scripts:
      ```bash
      $ cd datasets/Criteo/Criteo_x1
      $ python download_criteo_x1.py
      $ python convert_criteo_x1.py (please modify the path accordingly)
      ```
  + Solution#2: For ease of reuse, the preprocessed data are available for [downloading here](https://zenodo.org/record/5700987/files/Criteo_x1.zip).
  + Check the md5sum for consistency.
      ```bash
      $ md5sum train.csv valid.csv test.csv
      30b89c1c7213013b92df52ec44f52dc5  train.csv
      f73c71fb3c4f66b6ebdfa032646bea72  valid.csv
      2c48b26e84c04a69b948082edae46f8c  test.csv
      ```


+ Default setting
  
  In this benchmark setting, we follow the AFN work to fix **embedding_dim=10**, **batch_size=4096**, and **MLP_hidden_units=[400, 400, 400]** to make fair comparisons.


## Criteo_x2

+ Dataset description

  This dataset employs the [Criteo 1TB Click Logs](https://ailab.criteo.com/criteo-1tb-click-logs-dataset/) for display advertising, which contains one month of click-through data with billions of data samples. Following the same setting in the [AutoGroup](https://dl.acm.org/doi/abs/10.1145/3397271.3401082) work, we select "data 6-12" as the training set while using "day-13" for testing. To reduce label imbalance, we perform negative sub-sampling to keep the positive ratio roughly at 50%. It has 13 numerical feature fields and 26 categorical feature fields. For consistency of evaluation, we directly reuse the preprocessed data accessible from the [BARS benchmark](https://github.com/openbenchmark/BARS/click_prediction/datasets). 

+ How to get the dataset?
  1. For ease of reuse, the preprocessed data are available for [downloading here](https://zenodo.org/record/5700987/files/Criteo_x2.zip).
  3. Check the md5sum for consistency.
      ```bash
      $ md5sum train.csv test.csv
      d4d08405e95836ee049455cae0f8b0d6  train.csv
      32c14fbc7bfe02e72b501793e8db660b  test.csv
      ```

+ Default setting

  In this setting, 13 numerical fields are converted into categorical values through bucketizing, while categorical features appearing less than 20 times are set as a default ``<OOV>`` feature.


## Criteo_x3
TBA



## Criteo_x4

+ Dataset description

  The Criteo dataset is a widely-used benchmark dataset for CTR prediction, which contains about one week of click-through data for display advertising. It has 13 numerical feature fields and 26 categorical feature fields. Following the setting in the [AutoInt work](https://arxiv.org/abs/1810.11921), we randomly split the data into 8:1:1 as the training set, validation set, and test set, respectively. For better reproduciblity, we directly reuse the code provided by AutoInt and control the random seed (i.e., seed=2018) for splitting. The preprocessed data are accessible from the [BARS benchmark](https://github.com/openbenchmark/BARS/click_prediction/datasets).

+ How to get the dataset?
  + Solution#1: Download the raw dataset, and run the following scripts:
      ```bash
      $ cd datasets/Criteo/Criteo_x4
      $ python split_criteo_x4.py
      ```
  + Solution#2: For ease of reuse, the preprocessed data are available for [downloading here](https://zenodo.org/record/5700987/files/Criteo_x4.zip).
  + Check the md5sum for consistency.
      ```bash
      $ md5sum train.csv valid.csv test.csv
      4a53bb7cbc0e4ee25f9d6a73ed824b1a  train.csv
      fba5428b22895016e790e2dec623cb56  valid.csv
      cfc37da0d75c4d2d8778e76997df2976  test.csv
      ```

#### Criteo_x4_001

In this setting, we follow the winner's solution of the Criteo challenge to discretize each integer value x to ⌊log2
(x)⌋, if x > 2; and x = 1 otherwise. For all categorical fields, we replace infrequent features with a default ``<OOV>`` token by setting the threshold min_category_count=10. Note that we do not follow the exact preprocessing steps in AutoInt, because this preprocessing performs much better. To make a fair comparison, we fix **embedding_dim=16** as with AutoInt.

  
#### Criteo_x4_002

In this setting, we follow the winner's solution of the Criteo challenge to discretize each integer value x to ⌊log2
(x)⌋, if x > 2; and x = 1 otherwise. For all categorical fields, we replace infrequent features with a default ``<OOV>`` token by setting the threshold min_category_count=2. We fix **embedding_dim=40** in this setting.

