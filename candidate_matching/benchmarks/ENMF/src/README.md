# ENMF
ENMF is a model for Efficient Neural Matrix Factorization, published in the following paper:

> Chong Chen, Min Zhang, Yongfeng Zhang, Yiqun Liu, and Shaoping Ma. Efficient Neural Matrix Factorization without Sampling for Recommendation. In TOIS, vol. 38, no. 2, 2020.


The benchmark is implemented based on the original ENMF code released by the authors on Github:
https://github.com/chenchongthu/ENMF/tree/153e75878eb058b9c3e7fd74c84355fbda6b7a23 (commit hash: 153e758). 

In addition, the following modifications are made. You can view these changes via a diff comparison through this [link](https://github.com/xue-pai/Open-CF-Benchmarks/compare/ada620b...939f87e?diff=split).


1. Tensorflow APIs updates, for example:
    ```python
    tf.nn.dropout(x, keep_prob) => tf.nn.dropout(x, rate=1 - keep_prob)
    ```
2. Modified method `parse_args` such that option `topK` can accept a list of numbers as arguments.
3. Add the class `Monitor`, which records the metircs (hitrate@20,recall@20,ndcg@20,hitrate@50,recall@50,ndcg@50) during validations and determines whether to early stop.
4. Add calculation of hitrate@K and monitor in the method `dev_step`, and also perform some optimization for speedup.
5. Fix a bug in `ENMF.py`
    ```python
    # self.pos_r = tf.reshape(self.pos_r, [-1, max_item_pu])
    self.pos_r = tf.reshape(self.pos_r, [-1, self.max_item_pu])
    ```
6. Add name `deep`, `sess` and `train_op1` in method `train_step1` of file `ENMF.py`, to avoid name error.
    ```python
    # def train_step1(u_batch, y_batch, args)
    def train_step1(u_batch, y_batch, deep, sess, train_op1, args)
    ```
7. Fix a compatibility issue in method `dev_step` of `ENMF.py`, which leads to `IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed` when running in Python 3.
    ```python
    # user_te = np.array(test_set.keys())
    user_te = np.array(list(test_set.keys()))
    ```
