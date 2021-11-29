import sys
sys.path.append("../../external/LightGCN_TF")
import os
from utility.parser import parse_args
from utility.load_data import *
import tensorflow as tf

args = parse_args()
f_path = args.data_path + args.dataset + '/' + args.dataset.lower() + '_x0'
data_generator = Data(path=f_path, batch_size=args.batch_size)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from lightgcn_tf import start_lightgcn


if __name__ == "__main__":

    start_lightgcn(args, data_generator, sess)

