# KuaiShou

+ [KuaiVideo_x1](#kuaivideo_x1)
+ KuaiRand


## KuaiVideo_x1

+ Dataset description

  The preprocessed dataset is obtained from the [ALPINE](https://github.com/liyongqi67/ALPINE) work. The raw dataset is released by the Kuaishou Competition in the China MM 2018 conference, which aims to infer users' click probabilities for new micro-videos. In this dataset, there are multiple interactions between users and micro-videos, such as "click", "not click", "like", and "follow". Particularly, "not click" means the user did not click the micro-video after previewing its thumbnail. Moreover, each behaviour is associated with a timestamp, which records when the behaviour happens. We have to mention that the timestamp has been processed such that the absolute time is unknown, but the sequential order can be obtained according to the timestamp. For each micro-video, the contest organizers have released its 2,048-d visual embedding of its thumbnail. Among the large-scale dataset, 10,000 users and their 3,239,534 interacted micro-videos are randomly selected.


+ Benchmark setting
  
  In this benchmark setting, we follow the AFN work to fix **embedding_dim=64** to make fair comparisons.



