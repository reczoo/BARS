class MostPop(object):
    def __init__(self, n=400):
        """
        Most Popular Recommender
        Parameters
        ----------
        n : pre-selected popular item number
        """
        self.rank_list = None
        self.N = n

    def fit(self, train_set):
        res = train_set['item'].value_counts()
        # self.top_n = res[:self.N].index.tolist()
        self.rank_list = res.index.tolist()[:self.N]

    def predict(self, test_u, train_ur, topk=20):
        candidates = self.rank_list
        candidates = [item for item in candidates if item not in train_ur[test_u]]
        if len(candidates) < topk:
            raise Exception(f'parameter N is too small to get {topk} recommend items')
        return candidates[:topk]


