# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:38:36 2019

@author: ASUS
"""

import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

X_train,X_test,y_train,y_test = train_test_split(
        iris_X,iris_y,test_size=0.3)
print(y_train) # 有打乱

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

print(knn.predict(X_test))
print(y_test)

