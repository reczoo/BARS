# KKBox

It is a [WSDM challenge dataset for KKBox's Music Recommendation in 2018](https://www.kaggle.com/c/kkbox-music-recommendation-challenge). The dataset is from [KKBox](https://www.kkbox.com/), Asia's leading music streaming service, holding the world's most comprehensive Asia-Pop music library with over 30 million tracks. 

The task is to predict the chances of a user listening to a song repetitively after the first observable listening event within a time window was triggered. If there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, its target is marked 1, and 0 otherwise in the training set. KKBox provides a training data set consists of information of the first observable listening event for each unique user-song pair within a specific time duration. Metadata of each unique user and song pair is also provided. The train and the test data are selected from users listening history in a given time period, and are split based on time. Note that only the labeled train set of the data is used for benchmarking. 

Data fields consist of:
+ target: this is the target variable. target=1 means there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, target=0 otherwise .
+ msno: user id
+ song_id: song id
+ source_system_tab: the name of the tab where the event was triggered. System tabs are used to categorize KKBOX mobile apps functions. For example, tab my library contains functions to manipulate the local storage, and tab search contains functions relating to search.
+ source_screen_name: name of the layout a user sees.
+ source_type: an entry point a user first plays music on mobile apps. An entry point could be album, online-playlist, song, etc.

Song features:
+ song_length: in ms
+ genre_ids: genre category. Some songs have multiple genres and they are separated by |
+ artist_name
+ composer
+ lyricist
+ language
+ song name: the name of the song.
+ isrc: International Standard Recording Code
 
User features:
+ city
+ bd: age. Note: this column has outlier values, please use your judgement.
+ gender
+ registered_via: registration method
+ registration_init_time: format %Y%m%d
+ expiration_date: format %Y%m%d


## KKBox_x4

This dataset is randomly split into 8:1:1 as the training set, validation set, and test set, respectively. To make it reproducible, we reuse part of the code provided by AutoInt and control the random seed (i.e., seed=2018) for splitting. 

+ Reproducing steps:
  + Step1: Download the [raw data](https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data).
  + Step2: Split the data using [the script](./KKBox_x4/split_kkbox_x4.py).


### KKBox_x4_001

In this setting, For all categorical fields, we replace infrequent features with a default ``<OOV>`` token by setting the threshold min_category_count=10. 

To make a fair comparison, we fix **embedding_dim=128**, which performs well.

  

