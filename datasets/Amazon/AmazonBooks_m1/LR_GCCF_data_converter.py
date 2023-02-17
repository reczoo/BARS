import pdb
from collections import defaultdict
import numpy as np
  

training_path='./train.txt' 
testing_path='./test.txt'
val_path='./val.txt'

path_save_base='./datanpy'
if (os.path.exists(path_save_base)):
    print('has results save path')
else:
    os.makedirs(path_save_base)  
       
train_data_user = defaultdict(set)
train_data_item = defaultdict(set) 
links_file = open(training_path)
num_u=0
num_u_i=0
for _, line in enumerate(links_file):
    line=line.strip('\n')
    tmp = line.split(' ')
    num_u_i+=len(tmp)-1
    num_u+=1
    u_id=int(tmp[0])
    for i_id in tmp[1:]: 
        train_data_user[u_id].add(int(i_id))
        train_data_item[int(i_id)].add(u_id)
np.save('./datanpy/train_set.npy',[train_data_user,train_data_item,num_u_i]) 
print(num_u,num_u_i)

test_data_user = defaultdict(set)
test_data_item = defaultdict(set) 
links_file = open(testing_path)
num_u=0
num_u_i=0
for _, line in enumerate(links_file):
    line=line.strip('\n')
    tmp = line.split(' ')
    num_u_i+=len(tmp)-1
    num_u+=1
    u_id=int(tmp[0])
    for i_id in tmp[1:]: 
        test_data_user[u_id].add(int(i_id))
        test_data_item[int(i_id)].add(u_id)
np.save('./datanpy/test_set.npy',[test_data_user,test_data_item,num_u_i]) 
print(num_u,num_u_i)

val_data_user = defaultdict(set)
val_data_item = defaultdict(set) 
links_file = open(val_path)
num_u=0
num_u_i=0
for _, line in enumerate(links_file):
    line=line.strip('\n')
    tmp = line.split(' ')
    num_u_i+=len(tmp)-1
    num_u+=1
    u_id=int(tmp[0])
    for i_id in tmp[1:]: 
        val_data_user[u_id].add(int(i_id))
        val_data_item[int(i_id)].add(u_id)
np.save('./datanpy/val_set.npy',[val_data_user,val_data_item,num_u_i]) 
print(num_u,num_u_i)

user_rating_set_all = defaultdict(set)
for u in range(num_u):
    train_tmp = set()
    test_tmp = set() 
    val_tmp = set() 
    if u in train_data_user:
        train_tmp = train_data_user[u]
    if u in test_data_user:
        test_tmp = test_data_user[u] 
    if u in val_data_user:
        val_tmp = val_data_user[u] 
    user_rating_set_all[u]=train_tmp|test_tmp|val_tmp
np.save('./datanpy/user_rating_set_all.npy',user_rating_set_all) 


