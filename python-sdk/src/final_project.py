
# coding: utf-8

# ## 

# 

# In[147]:

get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report,precision_score, recall_score, accuracy_score,auc,roc_curve
from sklearn import naive_bayes
from sklearn import preprocessing
from sklearn import linear_model
import random
from scipy import optimize


# ## 数据预处理

# In[53]:

##显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)


# In[54]:

lc = pd.read_csv('LC.csv')
lp = pd.read_csv('LP.csv')


# In[55]:

##对未还时间处理函数
def no_str(x):
    if x != u'\\N':
        a = pd.Timestamp(x).date()
    else:
        a = np.nan
    return a


# In[56]:

##对时间处理
lc['date'] = lc['借款成功日期'].apply(lambda x: pd.Timestamp(x).date())
lp['date'] = lp['到期日期'].apply(lambda x: pd.Timestamp(x).date())
lp['pay_date'] = lp['还款日期'].apply(no_str)
map_dict = dict(zip(lc['ListingId'].values,lc['借款金额'].values))
lp['p'] = lp['ListingId'].map(map_dict)


# In[57]:

##提取所需预测,超过30天逾期作为逾越label
del_list = (lp['pay_date'] - lp['date']).dropna()[((lp['pay_date']                                                     - lp['date']).dropna().apply(lambda x: x.days) < 30)].index


# In[58]:

df_lp = lp.drop(del_list)
df_lp = df_lp[df_lp['date'] < datetime.date(2017,1,22)]


# In[59]:

##打上label
pre_dict = dict(zip(df_lp['ListingId'].unique(),[1]*len(df_lp['ListingId'].unique())))
lc['overdue'] = lc['ListingId'].map(pre_dict)
lc['overdue'].fillna(0,inplace = True)


# In[60]:

##提取可使用数据
df_lc = lc[lc['date'] < datetime.date(2017,1,22)]


# In[61]:

df_lc.head()


# In[62]:

## 提取类别项
df_lc.drop(['借款成功日期','date'],axis = 1,inplace = True)
cate_list = ['初始评级', '借款类型', '是否首标',              '性别', '手机认证', '户口认证', '视频认证',              '学历认证', '征信认证', '淘宝认证','overdue']
df_lc_cate = df_lc[cate_list]
df_lc_cate.head()


# In[63]:

for i in df_lc_cate.columns[:-1]:
    df_lc_cate.groupby(i)['overdue'].mean().plot.bar()
    print(i)
    plt.show()


# In[64]:

#df_lc_cate编码
df_lc_onehot_cate = pd.get_dummies(df_lc_cate.drop(['overdue'],axis = 1),prefix=df_lc_cate.columns[:-1])#数据变换处理
scale = preprocessing.FunctionTransformer(np.log1p)
scale.fit(df_lc.drop(cate_list,axis = 1))
df_scale = pd.DataFrame(scale.transform(df_lc.drop(cate_list,axis = 1)),columns=df_lc.drop(cate_list,axis = 1).columns,index = df_lc.index)
df_onehot_lc = pd.concat([df_scale,df_lc_onehot_cate,df_lc[['overdue']]],axis = 1)


# ## 建模

# In[65]:

##建模过程
from sklearn.model_selection import train_test_split
X = df_onehot_lc.drop(['overdue','ListingId'],axis = 1)
y = df_onehot_lc['overdue']
X_train, test, Y_train, result = train_test_split(X , y, test_size=0.33, random_state=42)


# In[66]:

##分割数据，保持数据的平衡,代入模型观察结果
def rf_test(X_train,Y_train,n = 20):
    df_0 = X_train[Y_train == 0]
    df_1 = X_train[Y_train == 1]
    x1_train, x1_test, y1_train, y1_test = train_test_split(df_1, Y_train[Y_train == 1], test_size = 0.33,random_state=1)
    x0_train, x0_test, y0_train, y0_test = train_test_split(df_0, Y_train[Y_train == 0], test_size = 0.33,random_state=1)
    lines = np.linspace(0,len(x0_train),n)
    data_dict = {}
    k = 1
    for i,j in zip(lines[0:-1],lines[1:]):
        data_dict['x_train'+ str(k)] = pd.concat([x1_train,x0_train.iloc[int(i):int(j)]])
        data_dict['y_train'+ str(k)] = pd.concat([y1_train,y0_train.iloc[int(i):int(j)]])
        k = k+1
    ##原始版本，做一个baseline
    x_test = pd.concat([x0_test,x1_test])
    y_test = pd.concat([y0_test,y1_test])
    for i in range(1,n):
        if i > 6:
            break
        print('\n',i,'\n')
        for t in range(5,60,5):
            model = RandomForestClassifier(n_estimators= 40 ,max_depth = t ,class_weight = 'balanced',criterion= 'gini')
            model.fit(data_dict['x_train'+ str(i)],data_dict['y_train' + str(i)].values.ravel())
            print( 'c = ',t ,'predict=',model.score(x_test,y_test),'auc=' , metrics.roc_auc_score(y_test.values.ravel(),model.predict(x_test)))
    return pd.DataFrame(model.feature_importances_,index = X_train.columns,columns=['importance'])


# In[67]:

##朴素贝叶斯结果，baseline
n = 20
df_0 = X_train[Y_train == 0]
df_1 = X_train[Y_train == 1]
x1_train, x1_test, y1_train, y1_test = train_test_split(df_1, Y_train[Y_train == 1], test_size = 0.33,random_state=1)
x0_train, x0_test, y0_train, y0_test = train_test_split(df_0, Y_train[Y_train == 0], test_size = 0.33,random_state=1)
lines = np.linspace(0,len(x0_train),n)
data_dict = {}
k = 1
for i,j in zip(lines[0:-1],lines[1:]):
    data_dict['x_train'+ str(k)] = pd.concat([x1_train,x0_train.iloc[int(i):int(j)]])
    data_dict['y_train'+ str(k)] = pd.concat([y1_train,y0_train.iloc[int(i):int(j)]])
    k = k+1
##原始版本，做一个baseline
x_test = pd.concat([x0_test,x1_test])
y_test = pd.concat([y0_test,y1_test])
for i in range(1,n):
    if i > 15:
        break
#     print('\n',i,'\n')
#     for t in range(5,60,5):
    model = naive_bayes.GaussianNB()
    model.fit(data_dict['x_train'+ str(i)],data_dict['y_train' + str(i)].values.ravel())
    print('predict=',model.score(x_test,y_test),'auc=' , metrics.roc_auc_score(y_test.values.ravel(),model.predict(x_test)))
#return pd.DataFrame(model.feature_importances_,index = X_train.columns,columns=['importance'])


# In[68]:

## 特征选取
df_onehot_lc[df_onehot_lc['overdue'] == 1].head()


# In[69]:

##尝试特征组合
list(df_lc_cate.ix[0])
df_lc_cate['t'] = '\n '
for i,j in zip(df_lc_cate.columns[:-1],range(len(df_lc_cate.columns[:-1]))):
    try:
        if j == 0:
            df_lc_cate['all'] = df_lc_cate[i]
        else:
            df_lc_cate['all'] = df_lc_cate[i] + df_lc_cate['t'] + df_lc_cate['all']
    except:
        continue
lc_1_list = list(df_lc_cate[df_lc_cate['overdue'] == 1]['all'].apply(lambda x: x.split()))
lc_0_list = list(df_lc_cate[df_lc_cate['overdue'] == 0]['all'].apply(lambda x: x.split()))


# In[70]:

o = list(df_lc_cate.columns[:-3])
o.reverse()


# In[71]:

def cate_list(a):
    c=[]
    for i,j in zip(a,o):
        c.append(j+'_'+i)
    return c


# In[72]:

lc_1 = list(map(cate_list,lc_1_list))
lc_0 = list(map(cate_list,lc_0_list))


# ## 频繁模式算法做特征组合

# In[73]:

##创建一个集合，包含所有项集
def creatC1(dataset):
    c1 = []
    for trans in dataset:
        for item in trans:
            if [item] not in c1:
                c1.append([item])
    c1.sort()
    #print(c1)
    return list(map(frozenset,c1))


# In[74]:

##扫描计算支持度，支持项
def scanD(D,Ck,minsupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt.keys():ssCnt[can] = 1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minsupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList,supportData


# In[75]:

##Apriori筛选条件，频繁集筛选,得到候选集
def aprioriGen(Lk,k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i]|Lk[j])
    return retList


# In[76]:

##Apriori算法
def apriori(dataset,minSupport = 0.5):
    c1 = creatC1(dataset)
    D = list(map(set,dataset))
    L1,supportData = scanD(D,c1,minSupport)
    L = [L1]
    k = 2
    while(len(L[k-2])> 0):
        ##扫描数据集，得到候选集
        ck = aprioriGen(L[k-2],k)
        Lk,supK = scanD(D,ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k = k+1
    return L,supportData


# In[77]:

##置信度函数
def geneRules(L,supportData,minconf = 0.7):
    bigRuleList = []
    #只获取两个及以上元素集合
    for i in range(1,len(L)):
        for freq in L[i]:
            H1 = [frozenset[item] for item in freq ] 
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList 


# In[78]:

#计算置信度
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            print (freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


# In[79]:

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


# In[80]:

def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print (itemMeaning[item])
        print ("           -------->")
        for item in ruleTup[1]:
            print (itemMeaning[item])
        print ("confidence: %f" % ruleTup[2])
        #print       #print a blank line


# In[81]:

a,b = apriori(lc_1,minSupport=0.6)


# In[82]:

c,d = apriori(lc_0,minSupport=0.6)


# In[83]:

p1 = []
p2 = []
for l1,l2 in zip(a,c):
    for k1 in l1:
        if k1 not in l2:
            p1.append(k1)  
    for k2 in l2:
        if k2 not in k1:
            p2.append(k2)


# In[84]:

for f1,i in zip(p1,range(len(p1))):
    X['cond'+ str(i)] = X[list(f1)].sum(axis = 1) == len(f1)


# In[85]:

# for f2,i in zip(p2,range(len(p2))):
#     X['cond_'+ str(i)] = X[list(f2)].sum(axis = 1) == len(f2)


# In[118]:

X['认证数'] = X[['手机认证_成功认证',
       '户口认证_成功认证', '视频认证_成功认证', '学历认证_成功认证', '征信认证_成功认证', '淘宝认证_成功认证']].sum(axis = 1)
X['月承担利率'] = X['借款利率']/X['借款期限']
X['历史还款率'] = (X['历史正常还款期数']/(X['历史逾期还款期数'] + X['历史正常还款期数'])).fillna(1)


# In[119]:

X_train, test, Y_train, result = train_test_split(X , y, test_size=0.33, random_state=42)


# ## 通过频繁模式的特征组合，baseline的准确率提升0.01，auc提升0.01左右。

# In[120]:

##朴素贝叶斯结果，baseline
n = 20
df_0 = X_train[Y_train == 0]
df_1 = X_train[Y_train == 1]
x1_train, x1_test, y1_train, y1_test = train_test_split(df_1, Y_train[Y_train == 1], test_size = 0.33,random_state=1)
x0_train, x0_test, y0_train, y0_test = train_test_split(df_0, Y_train[Y_train == 0], test_size = 0.33,random_state=1)
lines = np.linspace(0,len(x0_train),n)
data_dict = {}
k = 1
for i,j in zip(lines[0:-1],lines[1:]):
    data_dict['x_train'+ str(k)] = pd.concat([x1_train,x0_train.iloc[int(i):int(j)]])
    data_dict['y_train'+ str(k)] = pd.concat([y1_train,y0_train.iloc[int(i):int(j)]])
    k = k+1
##原始版本，做一个baseline
x_test = pd.concat([x0_test,x1_test])
y_test = pd.concat([y0_test,y1_test])
for i in range(1,n):
    if i > 15:
        break
#     print('\n',i,'\n')
#     for t in range(5,60,5):
    model = naive_bayes.GaussianNB()
    model.fit(data_dict['x_train'+ str(i)],data_dict['y_train' + str(i)].values.ravel())
    print('predict=',model.score(x_test,y_test),'auc=' , metrics.roc_auc_score(y_test.values.ravel(),model.predict(x_test)))
#return pd.DataFrame(model.feature_importances_,index = X_train.columns,columns=['importance'])


# ## 期望方差，组合

# In[143]:

model_list = []
df_predict = pd.DataFrame(index = X.index)
n = 10
df_0 = X_train[Y_train == 0]
df_1 = X_train[Y_train == 1]
x1_train, x1_test, y1_train, y1_test = train_test_split(df_1, Y_train[Y_train == 1], test_size = 0.33,random_state=1)
x0_train, x0_test, y0_train, y0_test = train_test_split(df_0, Y_train[Y_train == 0], test_size = 0.33,random_state=1)
lines = np.linspace(0,len(x0_train),n)
data_dict = {}
k = 1
for i,j in zip(lines[0:-1],lines[1:]):
    data_dict['x_train'+ str(k)] = pd.concat([x1_train,x0_train.iloc[int(i):int(j)]])
    data_dict['y_train'+ str(k)] = pd.concat([y1_train,y0_train.iloc[int(i):int(j)]])
    k = k+1
##原始版本，做一个baseline
x_test = pd.concat([x0_test,x1_test])
y_test = pd.concat([y0_test,y1_test])
for i in range(1,n):
#     if i > 6:
#         break
    #print('\n',i,'\n')
    #for t in range(5,60,5):
    model = RandomForestClassifier(n_estimators= 40 ,max_depth = 10 ,criterion= 'gini',class_weight='balanced')
    model.fit(data_dict['x_train'+ str(i)],data_dict['y_train' + str(i)].values.ravel())
    model_list.append(model)
    print('predict=',model.score(x_test,y_test),'auc=' , metrics.roc_auc_score(y_test.values.ravel(),model.predict(x_test)))
    df_pre = pd.DataFrame(model.predict_proba(X[x_test.columns]),index = X.index,columns= ['pre'+ str(i),'pre_reverse' + str(i)])
    df_predict['pre'+ str(i)] = df_pre['pre_reverse'+ str(i)]
##return pd.DataFrame(model.feature_importances_,index = X_train.columns,columns=['importance'])


# ## 经过频繁模式之后，使用随机森林发现准确率与AUC明显高于baseline

# In[142]:

# model_list = []
# df_predict = pd.DataFrame(index = X.index)
# n = 15
# df_0 = X_train[Y_train == 0]
# df_1 = X_train[Y_train == 1]
# x1_train, x1_test, y1_train, y1_test = train_test_split(df_1, Y_train[Y_train == 1], test_size = 0.33,random_state=1)
# x0_train, x0_test, y0_train, y0_test = train_test_split(df_0, Y_train[Y_train == 0], test_size = 0.33,random_state=1)
# lines = np.linspace(0,len(x0_train),n)
# data_dict = {}
# k = 1
# for i,j in zip(lines[0:-1],lines[1:]):
#     data_dict['x_train'+ str(k)] = pd.concat([x1_train,x0_train.iloc[int(i):int(j)]])
#     data_dict['y_train'+ str(k)] = pd.concat([y1_train,y0_train.iloc[int(i):int(j)]])
#     k = k+1
# ##原始版本，做一个baseline
# x_test = pd.concat([x0_test,x1_test])
# y_test = pd.concat([y0_test,y1_test])
# for i in range(1,n):
# #     if i > 6:
# #         break
# #     print('\n',i,'\n')
# #     for t in np.linspace(1,4,8):
#     model = linear_model.LogisticRegression(penalty='l1',class_weight='balanced')
#     model.fit(data_dict['x_train'+ str(i)],data_dict['y_train' + str(i)].values.ravel())
#     model_list.append(model)
#     print('predict=',model.score(x_test,y_test),'auc=' , metrics.roc_auc_score(y_test.values.ravel(),model.predict(x_test)))
#     df_pre = pd.DataFrame(model.predict_proba(X[x_test.columns]),index = X.index,columns= ['pre'+ str(i),'pre_reverse' + str(i)])
#     df_predict['pre'+ str(i)] = df_pre['pre_reverse'+ str(i)]


# In[144]:

df_onehot_lc['prob'] = df_predict.mean(axis = 1)
df_lc['prob'] = df_predict.mean(axis = 1)


# In[146]:

df_lc['prob'].hist(bins = 100)


# In[110]:

r = random.sample(list(df_lc['prob']),10000)


# In[111]:

df_lc['return'] = df_lc['借款利率'] *(1-df_lc['prob'])


# ## 核回归选取样本权重，得到类似概率方差

# In[112]:

def risk(new_sample):
    x = np.array(r) - new_sample
    h = 50
    w = np.exp(-x**2/2*h**2)
    return sum(w * x**2)


# In[113]:

df_lc['risk'] = df_lc['prob'].apply(risk)


# ## 画出方差收益曲线

# In[114]:

plt.scatter(df_lc['risk'],df_lc['return'])


# In[116]:

break


# ## 结论
# ### 在给定方差的情况下，我们可以得到一个最大期望的投资组合，在给定期望情况下，我们也可以得到一个最小方差的组合。其中，部分点不符合情况。其中，有两个可能，一是，模型不准确。二是，这种标是一种被误判的标，具有很高的投资价值。

# ## 计算最佳组合

# In[ ]:

## 输入一个新样本（dataframe格式）
new_sample


# In[ ]:

## 预处理新样本
cate_list = ['初始评级', '借款类型', '是否首标',                  '性别', '手机认证', '户口认证', '视频认证', '学历认证', '征信认证', '淘宝认证']
new_cate =  new_sample[cate_list]
new_onehot_cate = pd.get_dummies(new_cate,prefix=df_lc_cate.columns[:-1])
#数据变换处理
scale = preprocessing.FunctionTransformer(np.log1p)
scale.fit(new_sample_lc.drop(cate_list,axis = 1))
new_scale = pd.DataFrame(scale.transform(new_sample.drop(cate_list,axis = 1)),columns=new_sample.drop(cate_list,axis = 1).columns,index = new_sample.index)
new_onehot = pd.concat([new_scale,new_onehot_cate],axis = 1)
##频繁模式特征添加
for f1,i in zip(p1,range(len(p1))):
    new_onehot['cond'+ str(i)] = new_onehot[list(f1)].sum(axis = 1) == len(f1)


# In[ ]:

new_predict = pd.DataFrame(index = new_sample.index)
for moedel,num in zip(model_list,range(len(model_list))):
    new_predict['cond_' + str(num)] = model.predict(new_onehot)
new_sample['prob'] = new_predict.sum(axis = 1)


# In[ ]:

new_sample['return'] = new_sample['借款利率']*(1-new_sample['prob'])
new_sample['risk'] = new_sample['prob'].apply(risk)


# In[ ]:

##计算最佳组合
##给定期望收益，求最低风险组合
plt.scatter(new_sample['risk'],new_sample['return'])

