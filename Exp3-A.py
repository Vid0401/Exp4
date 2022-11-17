#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

iris = pd.read_csv(r'iris.csv')
print(iris)

mapping = {
'Iris-setosa' : 1,
'Iris-versicolor' : 2,
'Iris-virginica' : 3
}
x = iris.drop(['Species'], axis=1).values
y = iris.Species.replace(mapping).values 
print("Size of x: ",x.shape)
print("Size of y: ",y.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state= 0)
print("Size of x_train: ",x_train.shape)
print("Size of x_test: ",x_test.shape)
print("Size of y_train: ",y_train.shape)
print("Size of y_test: ",y_test.shape)

lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print("Actual:\n", y_test,"\n")
print("Predicted:\n", y_pred)

plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

cm = confusion_matrix (y_test, y_pred)

cm_df = pd.iris(cm, index = ['SETOSA','VERSICOLR','VIRGINICA'], columns = ['SETOSA','VERSICOLR','VIRGINICA'])

plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

print("Accuracy: ",accuracy_score(y_test,y_pred)*100)
print("Precision: ",precision_score(y_test, y_pred, average=None))
print("Recall: ",recall_score(y_test, y_pred, average=None))
print("F-1 Score: ",f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

