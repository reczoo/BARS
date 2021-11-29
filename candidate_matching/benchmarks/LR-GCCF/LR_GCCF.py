import sys
sys.path.append("../../external/LR_GCCF/")
import os

from parser_my import parse_argus
#import parser
from train_gowalla import train_gowalla
from train_amazons import train_amazons
from train_yelp import train_yelp
from test_gowalla import test_gowalla
from test_amazons import test_amazons
from test_yelp import test_yelp

args = parse_argus()
os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [args.gpu_id]))

if __name__ == "__main__":
    if args.dataset == 'gowalla':
        train_gowalla(args)
        test_gowalla(args)
    elif args.dataset == 'amazons':
        train_amazons(args)
        test_amazons(args)
    elif args.dataset == 'yelp':
        train_yelp(args)
        test_yelp(args)
    else:
        print("You can add new dataset by yourself!")
