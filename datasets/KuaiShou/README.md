# KuaiShou

+ [KuaiVideo_x1](#kuaivideo_x1)
+ KuaiRand


## KuaiVideo_x1

+ Dataset description

  The raw dataset is released by the Kuaishou Competition in the China MM 2018 conference, which aims to predict users' click probabilities for new micro-videos. In this dataset, there are multiple types of interactions between users and micro-videos, such as "click", "not click", "like", and "follow". Particularly, "not click" means the user did not click the micro-video after previewing its thumbnail. Note that the timestamp associated with each behaviour has been processed such that the absolute time is unknown, but the sequential order can be obtained according to the timestamp. For each micro-video, we can access its 2,048-d visual embedding of its thumbnail. In total, 10,000 users and their 3,239,534 interacted micro-videos are randomly selected. We reuse the preprocessed dataset obtained from the [ALPINE](https://github.com/liyongqi67/ALPINE) work. 

  The dataset statistics are summarized as follows.

  | Dataset Split  | Total | #Train | #Validation | #Test | 
  | :--------: | :-----: |:-----: | :----------: | :----: | 
  | KuaiVideo_x1 | 13,661,383    | 10,931,092  |      | 2,730,291   |


+ How to get the dataset?
    + Solution#1: Download [the preprocessed dataset](https://github.com/liyongqi67/ALPINE) from the ALPINE work and run the following scripts:
      ```bash
      $ cd KuaiShou/KuaiVideo_x1
      $ python convert_kuaivideo_x1.py
      ```

    + Check the md5sum for consistency.
      ```bash
      $ md5sum train.csv test.csv
      16f13734411532cc313caf2180bfcd56  train.csv
      ba26c01caaf6c65c272af11aa451fc7a  test.csv
      ```


+ Benchmark setting
  
  In this setting, we filter infrequent categorical features with the threshold min_category_count=10. We further set the maximal length of user behavior sequence to 100. To make a fair comparison, we suggest to set **embedding_dim=64**, and **hidden_units=[1024, 512, 256]** by default.


