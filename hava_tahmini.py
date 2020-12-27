#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import glob

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.metrics import mean_squared_error,r2_score

import os


# In[6]:


def mydropna(mydatasets):
    print("in my dropna() func")
    mytemp_datasets=dict()
    for city, dataset in mydatasets.items():
        mytemp_datasets[city]=dataset.dropna(axis=0,how=any)
        print("eksiltmeden önce",city," dataset shape: ",dataset.shape)
        print("eksiltmeden sonra ",city, " dataset shape: ",mytemp_datasets[city].shape)
        print()
    return mytemp_datasets


# In[7]:


#datasets=load_datasets()
dataset=pd.read_csv("BeijingPM20100101_20151231.csv")
#print(datasets)


# In[8]:


dataset.drop(['PM_Dongsi','PM_Dongsihuan','PM_Nongzhanguan'],
            axis=1,
            inplace=True)


# In[9]:


dataset.head()


# In[10]:


dataset.dropna(axis=0,how="any",inplace=True)


# In[11]:


labebelEncoder=LabelEncoder()


# In[12]:


dataset['cbwd']=labebelEncoder.fit_transform(dataset['cbwd'])


# In[13]:


X=dataset.drop('PM_US Post',axis=1)
y=dataset['PM_US Post']


# In[14]:


standardScaler=StandardScaler()
X_scaled=standardScaler.fit_transform(X)


# In[15]:


print("X_scaled.shape: ",X_scaled.shape)
print("y.shape: ",y.shape)


# In[16]:


train_size=int(y.shape[0]*0.8)
test_size=y.shape[0]-train_size
print("train size: ",train_size)
print("test size:",test_size)


# In[17]:


X_train=X_scaled[:train_size]
y_train=y[:train_size]
X_test=X_scaled[train_size:]
y_test=y[train_size:]


# In[18]:


print("X_train.shape: ",X_train.shape)
print("y_train.shape: ",y_train.shape)
print("X_test.shape: ",X_test.shape)
print("y_test.shape: ",y_test.shape)


# In[19]:


from sklearn.linear_model import LinearRegression
linearRegression=LinearRegression()
linearRegression.fit(X_train,y_train)


# In[20]:


r2=linearRegression.score(X_test,y_test)
print("R' skoru:{:.4f}".format(r2))


# In[21]:


y_pred=linearRegression.predict(X_test)


# In[22]:


print("Ortalama kare hatası:{:.4f}".format(mean_squared_error(y_test,y_pred)))


# In[23]:


print("R2 skoru:{:.4f}".format(r2_score(y_test,y_pred)))


# In[24]:


dataset_na_dropped=mydropna(dataset)


# In[26]:


#for dataset_name,datasett in dataset_na_dropped.items():
print("Beijing şehrinde yer alan ölçüm istasyonlar:")
for column in dataset.columns.values:
    if "PM_" in column:
        print(column)
print()


# In[28]:


#for dataset_name,dataset in dataset_na_dropped.items():
print("BEIJING veri setine ait ilk beş satır:")
print(dataset.head())


# In[33]:


#dataset_only_USPostPM={}
#for city,dat in dataset.item():
 #   columns=list()
  #  for column in dat.columns.values:
   #     if 'PM' in column and "US_POST" not in column:
    #        columns.append(column)
    #columns.append('No')
    #dataset_only_USPostPM[city]=dataset.drop(columns=columns)
dataset.drop(["No"],axis=1, inplace = True)


# In[34]:


dataset.head()


# In[35]:


dataset.info()


# In[36]:


from sklearn.linear_model import LinearRegression
linearRegression=LinearRegression()


# In[37]:


print(len(dataset))


# In[38]:


X=dataset.drop('PM_US Post',axis=1)
y=dataset['PM_US Post']


# In[39]:


train_size=int(len(dataset)*0.8)
test_size=len(dataset)-train_size
print("eğitim örnek sayısı: ",train_size)
print("test örnek sayısı:",test_size)
print("toplam:",train_size+test_size)


# In[40]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler(copy="deep")
scaler.fit(X)
X=scaler.transform(X)


# In[41]:


X_train=X[:train_size]
X_test=X[train_size:]
y_train=y[:train_size]
y_test=y[train_size:]


# In[42]:


linearRegression.fit(X_train,y_train)
y_pred=linearRegression.predict(X_test)
linearRegression.score(X_test, y_test)


# In[43]:


n_results=100
fig, ax=plt.subplots(2,1,figsize=(12,8))
ax[0].plot(y_test.values[:n_results], color="red")
ax[1].plot(y_pred[:n_results], color="green")


# In[44]:


print(mean_squared_error(y_test, y_pred))


# In[45]:


print(r2_score(y_test, y_pred))


# In[54]:


#KNN
import itertools
#from matplotlib.ticker as ticker
from matplotlib.ticker import NullFormatter
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

dataset.columns


# In[58]:


X=dataset[['year','month','day','hour','season','PM_US Post','DEWP','HUMI','PRES','TEMP','Iws','precipitation','Iprec']].values[0:5]


# In[ ]:




