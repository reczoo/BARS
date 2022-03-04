f=open('train.txt')
f2=open('train_enmf.txt','w')
f2.write('uid'+'\t'+'sid'+'\n')
for line in f:
    str=line.strip().split()
    for j in range(1,len(str)):
        f2.write(str[0]+'\t'+str[j]+'\n')

f=open('test.txt')
f2=open('test_enmf.txt','w')
f2.write('uid'+'\t'+'sid'+'\n')
for line in f:
    str=line.strip().split()
    for j in range(1,len(str)):
        f2.write(str[0]+'\t'+str[j]+'\n')
