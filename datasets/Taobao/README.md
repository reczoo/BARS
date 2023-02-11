# Taobao

+ [TaobaoAd_x1](#taobaoad_x1)


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



## TaobaoAd_x1

+ Dataset description

    Taobao is a dataset provided by Alibaba, which contains 8 days of ad click-through data (26 million records) that are randomly sampled from 1140000 users. Following the original data split, we use the first 7 days (i.e., 20170506-20170512) of samples for training, and the last day's samples (i.e., 20170513) for testing. To enable reproducibility, we open the preprocessed data in the [BARS benchmark](https://github.com/openbenchmark/BARS/datasets).

    The dataset statistics are summarized as follows:

    | Dataset Split  | Total | #Train | #Validation | #Test | 
    | :--------: | :-----: |:-----: | :----------: | :----: | 
    | TaobaoAd_x1 |  26,557,961     | XX   |      | XX    |         


+ How to get the dataset?
    + Solution#1: Download [the preprocessed dataset](https://aistudio.baidu.com/aistudio/datasetdetail/81892) for the DMR work and run the following scripts:
      ```bash
      $ cd Taobao/TaobaoAd_x1
      $ python convert_taobaoad_x1.py
      ```

+ Benchmark setting

  In this setting, we replace infrequent categorical features with a default <OOV> token by setting the threshold min_category_count=10. We further set the length of user click sequence to be 50 via truncating or padding. To make a fair comparison, after some tuning, we fix **embedding_dim=32**, and suggest **hidden_units=[512, 256, 128]** for fair comparisons.


