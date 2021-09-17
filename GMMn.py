# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 14:34:41 2018

@author: 1449486
"""


import numpy as np
import pandas as pd
confg = pd.read_csv("//01hw755449/Shared/ABDUL/GMM_complete/config1.txt")
confg=confg.transpose()
confg.columns = confg.iloc[0]
confg=confg.iloc[1:]


#confg=confg.transpose()

from sklearn import mixture
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


import glob
import pandas as pd

# get data

val = (confg['datapath'])
val=(", ".join(val))
path =val

#path=r'\\01hw755449\Shared\ABDUL\GMM_complete\PS'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

#selecting broker ID from data and aggregrating
dat=[]
for j in range(0,len(dfs)):
    df= pd.DataFrame(columns=dfs[0].columns, index=dfs[0].index)    
    df=dfs[j]
    dd=df.loc[df['brokerID'] == int(confg.brokerID)]#enter broker id here
    dd=np.array(dd)
    dat.append(dd)
    
data1= np.vstack(dat)
c= list(dfs[0])
data=pd.DataFrame(data1,columns=c)
dataa=data.loc[:, data.columns != 'brokerID']

#splitting training and test set
train=dataa.loc[data['showedUsualBehavior'] == 1]
test=dataa.loc[data['showedUsualBehavior'] == 0]


#removing the labels
train=train.loc[:, train.columns != 'showedUsualBehavior']
test=test.loc[:, test.columns != 'showedUsualBehavior']

#normalization of the dataset
train1=np.array(train)
train_norm=(train1-train1.mean())/train1.std()
test1=np.array(test)
test_norm=(test1-train1.mean())/train1.std()

#selection of no. of attributes and training size for modelling
no_attributes=(confg.attribute)
l_dataa=len(dataa.index)
train_size=int(float(confg.train_split)*l_dataa)


trainrow=train_norm[0:train_size,0:int(no_attributes)]
test1=test_norm[:,0:int(no_attributes)]
test2=train_norm[train_size:l_dataa,0:int(no_attributes)]

#concatenating anomalies with data without any anomaly
df1 = pd.DataFrame(test1)
df2 = pd.DataFrame(test2)
testt=pd.concat([df1,df2])
testt=np.array(testt)




# Fit the GMMs
from sklearn.mixture import GMM
gmm =mixture.GMM(n_components=int(confg.k))
gmm.fit(trainrow)

# Distribution parameters

means=(gmm.means_)
covar=(gmm.covars_)
weights=(gmm.weights_)


#probabliltiy pdf values

prob=[0]*int(confg.k)
pred=[0]*len(testt)

for j in range(0,len(testt)):
    testrow=testt[j,:]
    for i in range(0,int(confg.k)):
        prob[i]=multivariate_normal.pdf(testrow,mean=means[i],cov=covar[i]) 
    prob=np.array(prob)
    sumprob=sum(prob)
    
    if(sumprob<float(confg.mu)):
        pred[j]='Anomaly'
    else:
        pred[j]='ok'

pred=pd.DataFrame(pred)

expected=[0]*len(testt)
for j in range(0,len(testt)):
    if (j<len(test1)):
        expected[j]='Anomaly'
    else:
        expected[j]='ok'

expected=pd.DataFrame(expected)

result=pd.concat([expected,pred],axis=1)
result=pd.DataFrame(result.values, columns = ["Expected", "Predicted"])

#building a confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(expected,pred)
cm

#confusion matrix heat map
import seaborn as sn
sn.heatmap(cm, annot=True)


# Precision: tp/(tp+fp):
recall=cm[0,0]/(cm[0,0]+cm[0,1])


# Recall: tp/(tp + fn):
precision=cm[1,1]/(cm[1,0]+cm[1,1])

# F-Score: 2 * precision * recall /(precision + recall):
fscore = 2 * precision * recall / (precision + recall)
fscore


'''
names=list(data)

plt.hist(data[names[1]], color = 'blue', edgecolor = 'black',bins = int(20))



answer=[]
for i in range(0,len(testt)):
    if(result[i]=='Anomaly'):
        answer.append(i+1)
        
answer=np.array(answer)     
#AIC and BIC plots
n_estimators = np.arange(1, 90)
gmms = [GMM(n, n_iter=1000).fit(train1) for n in n_estimators]
bics = [gmm.bic(train1) for gmm in gmms]
aics = [gmm.aic(train1) for gmm in gmms]

plt.plot(n_estimators, bics, label='BIC')
plt.plot(n_estimators, aics, label='AIC')
plt.legend();

'''



