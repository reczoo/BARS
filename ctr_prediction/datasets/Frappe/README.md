# Frappe

Frappe dataset https://www.baltrunas.info/context-aware

The frappe dataset contains a context-aware app usage log. It consists of 96203 entries by 957 users for 4082 apps used in various contexts.

Data fields consist of:
+ user: anonymized user id
+ item: anonymized app id
+ daytime
+ weekday
+ isweekend
+ homework
+ cost
+ weather
+ country
+ city
+ cnt: how many times the app has been used by the user


Any scientific publications that use this dataset should cite the following paper:

+ Linas Baltrunas, Karen Church, Alexandros Karatzoglou, Nuria Oliver. [Frappe: Understanding the Usage and Perception of Mobile App Recommendations In-The-Wild](https://arxiv.org/abs/1505.03014), Arxiv 1505.03014, 2015.


## Frappe_x1

+ Dataset description

    The Frappe dataset contains a context-aware app usage log, which comprises 96203 entries by 957 users for 4082 apps used in various contexts. It has 10 feature fields including user_id, item_id, daytime, weekday, isweekend, homework, cost, weather, country, city. The target value indicates whether the user has used the app under the context. We reuse the preprocessed data released by the [AFN](https://ojs.aaai.org/index.php/AAAI/article/view/5768) work, which are randomly split into 7:2:1\* as the training set, validation set, and test set, respectively. For consistency of evaluation, we obtain the preprocessed data accessible from the [BARS benchmark](https://github.com/openbenchmark/BARS/click_prediction/datasets). 

    \* Note that the released data splits from the authors have a ratio of 7:2:1, which is different from 8:1:1 as reported in the AFN paper. 

    The dataset statistics are summarized as follows:

    | Dataset Split  | Total | #Train | #Validation | #Test | 
    | :--------: | :-----: |:-----: | :----------: | :----: | 
    | Frappe_x1 |  288,609   | 202,027  |  57,722    | 28,860    |   


+ How to get the dataset?
  + Solution#1: Run the following scripts:
      ```bash
      $ cd datasets/Frappe/Frappe_x1
      $ python convert_frappe_x1.py (please modify the path accordingly)
      ```
  + Solution#2: For ease of reuse, the preprocessed data are available for [downloading here](https://zenodo.org/record/5700987/files/Frappe_x1.zip).
  + Check the md5sum for consistency.
      ```bash
      $ md5sum train.csv valid.csv test.csv
      ba7306e6c4fc19dd2cd84f2f0596d158 train.csv
      88d51bf2173505436d3a8f78f2a59da8 valid.csv
      3470f6d32713dc5f7715f198ca7c612a test.csv
      ```

+ Default setting
  
  In this benchmark setting, we follow the AFN work to fix **embedding_dim=10**, **batch_size=4096**, and **MLP_hidden_units=[400, 400, 400]** to make fair comparisons.


