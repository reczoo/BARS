from datetime import datetime


class Monitor:
    def __init__(self, max_patience=5, delta=1e-6, log_file=None):
        self.counter = 0
        self.best_value = 0
        self.max_patience = max_patience
        self.patience = max_patience
        self.delta = delta
        self.log_file = log_file
        print("time,iteration,hitrate@20,recall@20,ndcg@20,hitrate@50,recall@50,ndcg@50", file=self.log_file)

    def update_monitor(self, hitrate20, recall20, ndcg20, hitrate50, recall50, ndcg50):
        self.counter += 1
        print("%s hitrate@20=%.4lf, recall@20=%.4lf, ndcg@20=%.4lf" % 
              (datetime.now(), hitrate20, recall20, ndcg20))
        print("%s hitrate@50=%.4lf, recall@50=%.4lf, ndcg@50=%.4lf" %
              (datetime.now(), hitrate50, recall50, ndcg50))
        print("%s,%d,%f,%f,%f,%f,%f,%f" % 
              (datetime.now(), self.counter, hitrate20, recall20, ndcg20, hitrate50, recall50, ndcg50),
              file=self.log_file)
        value = recall20 + ndcg20
        if value < self.best_value + self.delta:
            self.patience -= 1
            print("%s the monitor counts down its patience to %d!" % (datetime.now(), self.patience))
            if self.patience == 0:
                return True
        else:
            self.patience = self.max_patience
            self.best_value = value
            return False





