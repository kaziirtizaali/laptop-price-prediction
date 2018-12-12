# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 16:21:46 2018

@author: ratul
"""

import os
import numpy as np
import pandas as pd
os.chdir('D:\AI\Project')

df=pd.read_csv('dataset.csv')

"""
print(df.head())
print(df.iloc[:,1].unique()) #label or one hot
print(df['Processor'].unique())#label encoding
print(df['Manufacturer'].unique())
print(df['Operating System'].unique())
"""
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
df["Memory_Technology"] = lb_make.fit_transform(df["Memory Technology"])
df['Memory Technology']=df['Memory_Technology']
df=df.drop(['Memory_Technology'],axis=1)

df["Processor Type"] = lb_make.fit_transform(df["Processor"])
df['Processor']=df['Processor Type']
df=df.drop(['Processor Type'],axis=1)

df["Manufacturer Comp"] = lb_make.fit_transform(df["Manufacturer"])
df['Manufacturer']=df['Manufacturer Comp']
df=df.drop(['Manufacturer Comp'],axis=1)

df["os"] = lb_make.fit_transform(df["Operating System"])
df['Operating System']=df['os']
df=df.drop(['os'],axis=1)

df["Infrared"] = np.where(df["Infrared"]=='YES', 1,df["Infrared"])
df["Infrared"] = np.where(df["Infrared"]=='NO', 0,df["Infrared"])

df["Bluetooth"] = np.where(df["Bluetooth"]=='YES', 1,df["Bluetooth"])
df["Bluetooth"] = np.where(df["Bluetooth"]=='NO', 0,df["Bluetooth"])

df["Docking Station"] = np.where(df["Docking Station"]=='YES', 1,df["Docking Station"])
df["Docking Station"] = np.where(df["Docking Station"]=='NO', 0,df["Docking Station"])

df["Port Replicator"] = np.where(df["Port Replicator"]=='YES', 1,df["Port Replicator"])
df["Port Replicator"] = np.where(df["Port Replicator"]=='NO', 0,df["Port Replicator"])

df["Fingerprint"] = np.where(df["Fingerprint"]=='YES', 1,df["Fingerprint"])
df["Fingerprint"] = np.where(df["Fingerprint"]=='NO', 0,df["Fingerprint"])

df["Subwoofer"] = np.where(df["Subwoofer"]=='YES', 1,df["Subwoofer"])
df["Subwoofer"] = np.where(df["Subwoofer"]=='NO', 0,df["Subwoofer"])

df["External Battery"] = np.where(df["External Battery"]=='YES', 1,df["External Battery"])
df["External Battery"] = np.where(df["External Battery"]=='NO', 0,df["External Battery"])

df["CDMA"] = np.where(df["CDMA"]=='YES', 1,df["CDMA"])
df["CDMA"] = np.where(df["CDMA"]=='NO', 0,df["CDMA"])



print(df.head())
X=df.iloc[:,0:16]
y=df.iloc[:,16]
X=X.as_matrix()
y=y.as_matrix()


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor







reg=ExtraTreesRegressor(max_depth=10, random_state=33)
kf = KFold(n_splits=5,random_state = 33,shuffle = True)
kf.get_n_splits(X,y)

accuracy=[]

for train_index, test_index in kf.split(X):
       
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
        
       reg.fit(X_train, y_train)
       y_pred = reg.predict(X_test)
       accuracy.append(metrics.r2_score(y_test, y_pred))

print ("Avearage r2 score of ExtraTreesRegressor : ",np.mean(accuracy))





#reg=RandomForestRegressor(max_depth=10, random_state=0)

#reg= DecisionTreeRegressor(max_depth=10)
#reg=LinearRegression()
#reg=Ridge()
#reg=Lasso()

reg=GradientBoostingRegressor()


kf = KFold(n_splits=5,random_state = 33,shuffle = True)
kf.get_n_splits(X,y)

accuracy=[]

for train_index, test_index in kf.split(X):
       
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
        
       reg.fit(X_train, y_train)
       y_pred = reg.predict(X_test)
       accuracy.append(metrics.r2_score(y_test, y_pred))

print ("Avearage r2 score  of GradientBoostingRegressor: ",np.mean(accuracy))





reg=RandomForestRegressor(max_depth=10, random_state=0)

for train_index, test_index in kf.split(X):
       
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
        
       reg.fit(X_train, y_train)
       y_pred = reg.predict(X_test)
       accuracy.append(metrics.r2_score(y_test, y_pred))

print ("Avearage r2 score of RandomForestRegressor : ",np.mean(accuracy))






reg= DecisionTreeRegressor(max_depth=10)

for train_index, test_index in kf.split(X):
       
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
        
       reg.fit(X_train, y_train)
       y_pred = reg.predict(X_test)
       accuracy.append(metrics.r2_score(y_test, y_pred))

print ("Avearage r2 score of DecisionTreeRegressor: ",np.mean(accuracy))



reg=LinearRegression()

for train_index, test_index in kf.split(X):
       
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
        
       reg.fit(X_train, y_train)
       y_pred = reg.predict(X_test)
       accuracy.append(metrics.r2_score(y_test, y_pred))

print ("Avearage r2 score of LinearRegression: ",np.mean(accuracy))




reg=Ridge()

for train_index, test_index in kf.split(X):
       
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
        
       reg.fit(X_train, y_train)
       y_pred = reg.predict(X_test)
       accuracy.append(metrics.r2_score(y_test, y_pred))

print ("Avearage r2 score of Ridge Linear Regression: ",np.mean(accuracy))


reg=Lasso()

for train_index, test_index in kf.split(X):
       
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
        
       reg.fit(X_train, y_train)
       y_pred = reg.predict(X_test)
       accuracy.append(metrics.r2_score(y_test, y_pred))

print ("Avearage r2 score of Lasso Linear Regression: ",np.mean(accuracy))



