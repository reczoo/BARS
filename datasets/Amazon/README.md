# Amazon

+ [AmazonElectronics_x1](#amazonelectronics_x1)
+ [AmazonBooks_m1](#amazonbooks_m1)


## AmazonElectronics_x1

+ Dataset description

  The [Amazon dataset](http://jmcauley.ucsd.edu/data/amazon/) contains product reviews and metadata from Amazon, which is a widely-used benchmark dataset. We reuse the preprocessed dataset from the [DIN work](https://github.com/zhougr1993/DeepInterestNetwork) that is a subset named Electronics. It contains 192,403 users, 63,001 goods, 801 categories and 1,689,188 samples. User behaviors in this dataset are rich, with more than 5 reviews for each user and goods. Features include goods_id, cate_id, user reviewed goods_id_list and cate_id_list. Following DIN, the task is to predict the probability of reviewing the (k+1)-th goods by making use of the first k reviewed goods. The last item of each behavior sequence is reserved for testing.

  The dataset statistics are summarized as follows.

  | Dataset Split  | Total | #Train | #Validation | #Test | 
  | :--------: | :-----: |:-----: | :----------: | :----: | 
  | AmazonElectronics_x1 | 2,993,570   | 2,608,764 |      | 384,806  |


+ How to get the dataset?
    + Solution#1: Download [the pickled dataset](https://github.com/zhougr1993/DeepInterestNetwork/tree/master/din) from the DIN work and run the following scripts:
      ```bash
      $ cd Amazon/AmazonElectronics_x1
      $ python convert_amazonelectronics_x1.py
      ```

    + Check the md5sum for consistency.
      ```bash
      $ md5sum train.csv test.csv
      e9bf80b92985e463db18fdc753d347b5  train.csv
      57a20e82fe736dd495f2eaf0669bf6d0  test.csv
      ```


+ Benchmark setting
  
  In this setting, we set the maximal length of user behavior sequence to 100. To make a fair comparison, we suggest to set **embedding_dim=64**, and **hidden_units=[1024, 512, 256]** by default.



## AmazonBooks_m1






