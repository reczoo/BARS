# CollMetric

A Tensorflow implementation of Collaborative Metric Learning (CML): 

*Cheng-Kang Hsieh, Longqi Yang, Yin Cui, Tsung-Yi Lin, Serge Belongie, and Deborah Estrin. 2017. Collaborative Metric Learning. In Proceedings of the 26th International Conference on World Wide Web (WWW '17) ([perm_link](http://dl.acm.org/citation.cfm?id=3052639), [pdf](http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf))*

** Note: the original Theano implementation is deprecated and is kept in the *old_experiment_code branch*
# Features
* Produces embedding that accurately captures the user-item, user-user, and item-item similarity. 
* Allows the exploitation of item features (e.g. tags, text, image features).
* Outperforms state-of-the-art recommendation algorithms on a wide range of tasks
* Enjoys an extremely efficient Top-K search using Fast KNN algorithms.

# Utility Features
* Parallel negative sampler that can sample the user-item pairs when the model is being trained on GPU
* Fast recall evaluation based on Tensorflow

# Requirements
 * python3
 * tensorflow
 * scipy
 * scikit-learn

# Usage
```bash
# install requirements
pip3 install -r requirements.txt
# run demo tensorflow model
python3 CML.py
```

# Known Issue
* AdaGrad does not seem to work on GPU. Try using AdamOptimizer instead
* ~~the WithFeature version does not seems to perform as well as the Theano version. It is being investigated.~~ (The performance is actually slightly better (with AdamOptimizer) than the number reported in the paper now!)

# Visuals
### An illustration of embbeding learning procedue of CML
![CML](http://portalparts.acm.org/3060000/3052639/core/fp0554.jpg)
### Flickr photo recommendation embedding produced by CML (compared to original ImageNet features)
![Embedding](https://github.com/changun/CollMetric/blob/master/imgs/embedding.png?raw=true)
# TODO
* Model Comparison.
* TensorBoard visualization
