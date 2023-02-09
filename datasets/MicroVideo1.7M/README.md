# MicroVideo1.7M

MicroVideo-1.7M is an open dataset provided by the paper of "[Temporal Hierarchical Attention at Category- and Item-Level for Micro-Video Click-Through Prediction](https://github.com/whn09/THACIL)" published in MM'2018. It contains 12,737,617 (ps: 12,737,619 rows including headers) interactions that 10,986 users have made on 1,704,880 micro-videos. The labels include click or non-click, while the features include user_id, item_id, category, and the extracted image embedding vectors of cover images of micro-videos. 


## MicroVideo1.7M_x1

+ Dataset description

    MicroVideo-1.7M is a dataset provided by the [THACIL work](https://github.com/whn09/THACIL), which contains 12,737,617 interactions that 10,986 users have made on 1,704,880 micro-videos. The features include user id, item id, category, and the extracted image embedding vectors of cover images of micro-videos. For fairness of comparison, we follow the exact dataset splitting for training and testing, respectively. For reproducibility, we open the preprocessed data in the [BARS benchmark](https://github.com/openbenchmark/BARS/click_prediction/datasets). 

    The dataset statistics are summarized as follows:

    | Dataset Split  | Total | #Train | #Validation | #Test | 
    | :--------: | :-----: |:-----: | :----------: | :----: | 
    | MicroVideo1.7M_x1 |  12,737,617    | 8,970,309  |      | 3,767,308    | 


+ How to get the dataset?
    + Solution#1: Download [the raw dataset](https://github.com/whn09/THACIL) and run the following scripts:
      ```bash
      $ cd datasets/MicroVideo1.7M/MicroVideo1.7M_x1
      $ python convert_microvideo1.7m_x1.py
      ```
    + Solution#2: For ease of reuse, the preprocessed data are available for [downloading here](https://zenodo.org/record/5700987/files/MicroVideo1.7M_x1.zip).
    + Check the md5sum for consistency.
      ```bash
      $ md5sum train.csv test.csv
      936e6612714c887e76226a60829b4e0a  train.csv
      9417a18304fb62411ac27c26c5e0de56  test.csv
      ```

+ Default setting

  In this setting, we set the length of user click sequence to be 128 via truncating or padding. For fair comparisons, we fix **embedding_dim=64** as with THACIL, and set **hidden_units=[1024, 512, 256]** after some tuning.

  
