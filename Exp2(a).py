#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split 
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris 

iris = load_iris()
x,y = iris.data,iris.target
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3) 
lm = linear_model.LinearRegression()
model = lm.fit(x_train, y_train) 
predictions = lm.predict(x_test)
model.predict(x_test)
pred = model.predict(x_test)
plt.scatter(x_test[:,3], pred)
plt.show()
accuracy = model.score(x_test, y_test)
print(accuracy*100)

