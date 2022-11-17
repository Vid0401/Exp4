#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
import operator
import numpy as np
import pandas as pd

col=['sepal_length','sepal_width','petal_length','petal_width','type']

iris=pd.read_csv("Iris.csv",names=col)
#print(iris.head())
def euclidianDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)

def knn(trainingSet, testInstance, k):
    distances = {}
    sort = {}

    length = testInstance.shape[1]
    print(length)
    # Calculating euclidean distance between each row of training data and test data
    for x in range(len(trainingSet)):
        dist = euclidianDistance(testInstance, trainingSet.iloc[x], length)
        distances[x] = dist[0]
    # Sorting them on the basis of distance
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1)) 
    sorted_d1 = sorted(distances.items())
    print(sorted_d[:5])
    print(sorted_d1[:5])
    neighbors = []
    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    
    counts = {"Iris-setosa":0,"Iris-versicolor":0,"Iris-virginica":0}
    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        
        response = trainingSet.iloc[neighbors[x]][-1]

        if response in counts:
            counts[response] += 1
        else:
            counts[response] = 1

    print(counts)
    sortedVotes = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedVotes)
    return(sortedVotes[0][0], neighbors)


testSet = [[1.4, 3.6, 3.4, 1.2, 3.5]]

test = pd.DataFrame(testSet)
print(iris["type"].value_counts())

result,ne = knn(iris, test, 5)
print("And the flower is:",result)
print("the neighbors are:",ne)

