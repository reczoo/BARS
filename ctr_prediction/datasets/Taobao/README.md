# Taobao

+ [Taobao_x1](#Taobao_x1)
+ [Taobao_x2](#Taobao_x2)


The dataset is [provided by Alibaba](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56&userId=1), which contains 8 days of ad click logs that are randomly sampled from 1140000 users at the website of Taobao. The dataset has a total of 26 million records. By default, the first 7 days of samples are used as training samples (i.e., 20170506-20170512), and the last day's samples are used as test samples (i.e., 20170513). Meanwhile, the dataset also covers the shopping behavior of all users in the recent 22 days, including totally seven hundred million records.


Data fields consist of:

+ user: User ID (int);
+ time_stamp: time stamp (Bigint, 1494032110 stands for 2017-05-06 08:55:10);
+ adgroup_id: adgroup ID (int);
+ pid: scenario;
+ noclk: 1 for not click, 0 for click;
+ clk: 1 for click, 0 for not click;

ad_feature:
+ adgroup_id：Ad ID (int) ;
+ cate_id：category ID;
+ campaign_id：campaign ID;
+ brand：brand ID;
+ customer_id: Advertiser ID;
+ price: the price of item

user_profile:
+ userid: user ID;
+ cms_segid: Micro group ID;
+ cms_group_id: cms_group_id;
+ final_gender_code: gender 1 for male , 2 for female
+ age_level: age_level
+ pvalue_level: Consumption grade, 1: low,  2: mid,  3: high
+ shopping_level: Shopping depth, 1: shallow user, 2: moderate user, 3: depth user
+ occupation: Is the college student 1: yes, 0: no?
+ new_user_class_level: City level


raw_behavior_log:
+ nick: User ID(int);
+ time_stamp: time stamp (Bigint, 1494032110 stands for 2017-05-06 08:55:10)；
+ btag: Types of behavior, include: ipv/cart/fav/buy;
+ cate: category ID(int);
+ brand: brand ID(int);



## Taobao_x1

+ Dataset description

    Taobao is a dataset provided by Alibaba, which contains 8 days of ad click-through data (26 million records) that are randomly sampled from 1140000 users. Following the original data split, we use the first 7 days (i.e., 20170506-20170512) of samples for training, and the last day's samples (i.e., 20170513) for testing. To enable reproducibility, we open the preprocessed data in the [BARS benchmark](https://github.com/openbenchmark/BARS/click_prediction/datasets).

    The dataset statistics are summarized as follows:

    | Dataset Split  | Total | #Train | #Validation | #Test | 
    | :--------: | :-----: |:-----: | :----------: | :----: | 
    | Taobao_x1 |  26,557,961     | 23,249,296   |      | 3,308,665    |             


+ How to get the dataset?
    + Solution#1: Download [the raw dataset](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56&userId=1) and run the following scripts:
      ```bash
      $ cd datasets/Taobao/Taobao_x1
      $ python split_taobao_x1.py
      ```
    + Solution#2: For ease of reuse, the preprocessed data are available for [downloading here](https://zenodo.org/record/5700987/files/Taobao_x1.zip).
    + Check the md5sum for consistency.
      ```bash
      $ md5sum train.csv test.csv
      e4487021d20750121725e880556bfdc1  train.csv
      1de0b75cbb473b0c3ea2dd146dc4af28  test.csv
      ```

+ Default setting

  In this setting, we replace infrequent categorical features with a default ``<OOV>`` token by setting the threshold min_category_count=10. We further set the length of user click sequence to be 128 via truncating or padding. To make a fair comparison, after some tuning, we fix **embedding_dim=10**, and **hidden_units=[512, 256, 128]** for all compared models.


## Taobao_x2
TBA




