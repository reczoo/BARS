# -- coding:UTF-8
import numpy as np 
import pandas as pd 
import scipy.sparse as sp 

import torch.utils.data as data
import pdb
from torch.autograd import Variable
import torch
import math
import random

 

def load_all(test_num=100):
    """ We load all the three file here to save time in each epoch. """
    train_data = pd.read_csv(
        '../data/ml-1m.train.rating', 
        sep='\t', header=None, names=['user', 'item'], 
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    train_data = train_data.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = []
    with open('../data/ml-1m.test.rating', 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = fd.readline()
    return train_data, test_data, user_num, item_num, train_mat





class BPRData(data.Dataset):
    def __init__(self,train_dict=None,num_item=0, num_ng=1, is_training=None, data_set_count=0,all_rating=None):
        super(BPRData, self).__init__()

        self.num_item = num_item
        self.train_dict = train_dict
        self.num_ng = num_ng
        self.is_training = is_training
        self.data_set_count = data_set_count
        self.all_rating=all_rating
        self.set_all_item=set(range(num_item))  

    def ng_sample(self):
        # assert self.is_training, 'no need to sampling when testing'
        # print('ng_sample----is----call-----') 
        self.features_fill = []
        for user_id in self.train_dict:
            positive_list=self.train_dict[user_id]#self.train_dict[user_id]
            all_positive_list=self.all_rating[user_id]
            #item_i: positive item ,,item_j:negative item   
            # temp_neg=list(self.set_all_item-all_positive_list)
            # random.shuffle(temp_neg)
            # count=0
            # for item_i in positive_list:
            #     for t in range(self.num_ng):   
            #         self.features_fill.append([user_id,item_i,temp_neg[count]])
            #         count+=1  
            for item_i in positive_list:   
                for t in range(self.num_ng):
                    item_j=np.random.randint(self.num_item)
                    while item_j in all_positive_list:
                        item_j=np.random.randint(self.num_item)
                    self.features_fill.append([user_id,item_i,item_j]) 
      
    def __len__(self):  
        return self.num_ng*self.data_set_count#return self.num_ng*len(self.train_dict)
         

    def __getitem__(self, idx):
        features = self.features_fill  
        
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] 
        return user, item_i, item_j 

  
class resData(data.Dataset):
    def __init__(self,train_dict=None,batch_size=0,num_item=0,all_pos=None):
        super(resData, self).__init__() 
      
        self.train_dict = train_dict 
        self.batch_size = batch_size
        self.all_pos_train=all_pos 

        self.features_fill = []
        for user_id in self.train_dict:
            self.features_fill.append(user_id)
        self.set_all=set(range(num_item))
   
    def __len__(self):  
        return math.ceil(len(self.train_dict)*1.0/self.batch_size)#这里的self.data_set_count==batch_size
         

    def __getitem__(self, idx): 
        
        user_test=[]
        item_test=[]
        split_test=[]
        for i in range(self.batch_size):#这里的self.data_set_count==batch_size 
            index_my=self.batch_size*idx+i 
            if index_my == len(self.train_dict):
                break   
            user = self.features_fill[index_my]
            item_i_list = list(self.train_dict[user])
            item_j_list = list(self.set_all-self.all_pos_train[user])
            # pdb.set_trace() 
            u_i=[user]*(len(item_i_list)+len(item_j_list))
            user_test.extend(u_i)
            item_test.extend(item_i_list)
            item_test.extend(item_j_list)  
            split_test.append([(len(item_i_list)+len(item_j_list)),len(item_j_list)]) 
           
        #实际上只用到一半去计算，不需要j的。
        return torch.from_numpy(np.array(user_test)), torch.from_numpy(np.array(item_test)), split_test           
 
 