# Criteo

+ [Criteo_x0](#Criteo_x0)
  - [Criteo_x0_001](#Criteo_x0_001)
+ [Criteo_x4](#Avazu_x4)
  - [Criteo_x4_001](#Criteo_x4_001)
  - [Criteo_x4_002](#Criteo_x4_002)


It is a [Kaggle challenge dataset for Criteo display advertising](https://www.kaggle.com/c/criteo-display-ad-challenge/data). [Criteo](https://www.criteo.com/) is a personalized retargeting company that works with Internet retailers to serve personalized online display advertisements to consumers. The goal of this Kaggle challenge is to predict click-through rates on display ads. It offers a week's worth of data from Criteo's traffic. In the labeled training set over a period of 7 days, each row corresponds to a display ad served by Criteo. The samples are chronologically ordered. Positive and negatives samples have both been subsampled at different rates in order to reduce the dataset size. There are 13 count features and 26 categorical features. The semantic of these features is undisclosed. Some features may have missing values. [The dataset is openly available for downloading at AWS](https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz). Note that only the labeled part of the data is used for benchmarking. 

Data fields consist of:
+ Label: Target variable that indicates if an ad was clicked (1) or not (0).
+ I1-I13: A total of 13 columns of integer features (mostly count features).
+ C1-C26: A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes. 


## Criteo_x0

This dataset split follows the setting in the AFN work. That is, we randomly split the data into 8:1:1 as the training set, validation set, and test set, respectively. The data preprocessing script is provided on Github and we directly download the preprocessed data.

+ Reproducing steps:
  - Step1: Download the preprocessed data via [the script](./Avazu_x0/download_criteo_x0.py).


### Criteo_x0_001
  
In this setting, we follow the AFN work to fix **embedding_dim=16**, **batch_size=4096**, and **MLP_hidden_units=[400, 400, 400]** to make fair comparisons.


## Criteo_x4

This dataset split follows the setting in the [AutoInt work](https://arxiv.org/abs/1810.11921). Specifically, we randomly split the data into 8:1:1 as the training set, validation set, and test set, respectively. To make it exactly reproducible, we reuse the code provided by AutoInt and control the random seed (i.e., seed=2018) for splitting. 

+ Reproducing steps:
  + Step1: Download [the raw data](https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz).
  + Step2: Split the data using [the script](./Criteo_x4/split_criteo_x4.py).

### Criteo_x4_001

In this setting, we follow the winner's solution of the Criteo challenge to discretize each integer value x to ⌊log2
(x)⌋, if x > 2; and x = 1 otherwise. For all categorical fields, we replace infrequent features with a default ``<OOV>`` token by setting the threshold min_category_count=10. Note that we do not follow the exact preprocessing steps in AutoInt, because this preprocessing performs much better. 

To make a fair comparison, we fix **embedding_dim=16** as with AutoInt.

  
### Criteo_x4_002

In this setting, we follow the winner's solution of the Criteo challenge to discretize each integer value x to ⌊log2
(x)⌋, if x > 2; and x = 1 otherwise. For all categorical fields, we replace infrequent features with a default ``<OOV>`` token by setting the threshold min_category_count=2. 

We fix **embedding_dim=40** in this setting.

