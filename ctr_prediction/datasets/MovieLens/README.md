# MovieLens

MovieLens datasets https://grouplens.org/datasets/movielens

The MovieLens datasets are collected by GroupLens Research from the MovieLens web site (https://movielens.org) where movie rating data are made available. The datasets have been widely used in various research on recommender systems.


## MovielensLatest_x1

+ Dataset description

    The MovieLens dataset consists of users' tagging records on movies. The task is formulated as personalized tag recommendation with each tagging record (user_id, item_id, tag_id) as an data instance. The target value denotes whether the user has assigned a particular tag to the movie. We reuse the preprocessed data released by the [AFN](https://ojs.aaai.org/index.php/AAAI/article/view/5768) work, which are randomly split into 7:2:1\* as the training set, validation set, and test set, respectively. For consistency of evaluation, we obtain the preprocessed data accessible from the [BARS benchmark](https://github.com/openbenchmark/BARS/click_prediction/datasets). 

    \* Note that the released data splits from the authors have a ratio of 7:2:1, which is different from 8:1:1 as reported in the AFN paper. 

    The dataset statistics are summarized as follows:

    | Dataset Split  | Total | #Train | #Validation | #Test | 
    | :--------: | :-----: |:-----: | :----------: | :----: | 
    | Frappe_x1 |  2,006,859   | 1,404,801  |  401,373    | 200,686   |   


+ How to get the dataset?
  + Solution#1: Run the following scripts:
      ```bash
      $ cd datasets/MovieLens/MovielensLatest_x1
      $ python convert_movielenslatest_x1.py (please modify the path accordingly)
      ```
  + Solution#2: For ease of reuse, the preprocessed data are available for [downloading here](https://zenodo.org/record/6324454/files/Movielenslatest_x1.zip?download=1).
  + Check the md5sum for consistency.
      ```bash
      $ md5sum train.csv valid.csv test.csv
      efc8bceeaa0e895d566470fc99f3f271 train.csv
      e1930223a5026e910ed5a48687de8af1 valid.csv
      54e8c6baff2e059fe067fb9b69e692d0 test.csv
      ```

+ Default setting
  
  In this benchmark setting, we follow the AFN work to fix **embedding_dim=10**, **batch_size=4096**, and **MLP_hidden_units=[400, 400, 400]** to make fair comparisons.






