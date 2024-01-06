import sys
sys.path.append("../../external/CollMetric/")
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from sampler import WarpSampler
from CML_utils import Evaluator, Monitor, parse_args, dataset_to_uimatrix
from CML import CML, optimize


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    random_seed = 2017
    np.random.seed(random_seed)
    tf.compat.v1.set_random_seed(random_seed)
    
    print("%s reading dataset %s" % (datetime.now(), args.dataset))
    train, valid = dataset_to_uimatrix(args.train_data, args.test_data)
    n_users, n_items = train.shape
    print("%s #users=%d, #items=%d" % (datetime.now(), n_users, n_items))
    sampler = WarpSampler(train, batch_size=args.batch_size, n_negative=args.num_negative, check_negative=True)

    # WITHOUT features
    # Train a user-item joint embedding, where the items a user likes will be pulled closer to this users.
    # Once the embedding is trained, the recommendations are made by finding the k-Nearest-Neighbor to each user.
    model = CML(n_users,
                n_items,
                # set features to None to disable feature projection
                features=None,
                # size of embedding
                embed_dim=args.embed_dim,
                # the size of hinge loss margin.
                margin=args.margin,
                # clip the embedding so that their norm <= clip_norm
                clip_norm=args.clip_norm,
                # learning rate for AdaGrad
                master_learning_rate=args.lr,
                
                # dropout_rate = 1 - keep_prob
                dropout_rate=args.dropout,

                # whether to enable rank weight. If True, the loss will be scaled by the estimated
                # log-rank of the positive items. If False, no weight will be applied.

                # This is particularly useful to speed up the training for large item set.

                # Weston, Jason, Samy Bengio, and Nicolas Usunier.
                # "Wsabie: Scaling up to large vocabulary image annotation." IJCAI. Vol. 11. 2011.
                use_rank_weight=True,

                # whether to enable covariance regularization to encourage efficient use of the vector space.
                # More useful when the size of embedding is smaller (e.g. < 20 ).
                use_cov_loss=False,

                # weight of the cov_loss
                cov_loss_weight=1
                )

    log_file = open('dataset_' + args.dataset + '_margin_' + str(args.margin) + '_lr_' + str(args.lr) + '.csv', "w")
    monitor = Monitor(log_file=log_file)
    optimize(model, sampler, train, valid, max_steps=args.max_steps, monitor=monitor, verbose=args.verbose)
    print("%s close sampler, close and save to log file" % datetime.now())
    log_file.close()
    sampler.close() # important! stop multithreading
    print("%s log file and sampler have closed" % datetime.now())