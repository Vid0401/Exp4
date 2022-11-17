#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression 
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
sc=StandardScaler()

iris=load_iris() 
print(iris) 
print(iris.target_names) 
print(iris.feature_names)
df=pd.DataFrame(iris.data)
print(df.head())
df.columns=iris.feature_names 
df.head()

X=iris.data 
Y =iris.target 

print(X.shape)
print(Y.shape)
X_train,X_test,Y_train,Y_test=sklearn.model_selection.train_test_split(X,Y,test_size=0.25,random_state=2)

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
lorg=LogisticRegression(random_state=0)
lorg.fit(X_train,Y_train)
predictions =lorg.predict(X_test)
print(predictions)
print(Y)
##{plt.scatter(Y, predictions)
##plt.show()
##print(Y_pred)

##cm=confusion_matrix(Y_test,Y_pred)
##cm

##print(accuracy_score(Y_test,Y_pred))
##plt.show() }

