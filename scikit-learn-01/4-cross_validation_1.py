# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:40:38 2019

@author: ASUS
"""
# 
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 第一阶段
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

X_train,X_test,y_train,y_test = train_test_split(
        iris_X,iris_y,test_size=0.3)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))

# 第二阶段
X = iris.data
y = iris.target

from sklearn.cross_validation import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn,X,y,cv=5,scoring='accuracy')
print(scores.mean())

# 第三阶段
import matplotlib.pyplot as plt
K_range = range(1,31)
K_scores = []
for k in K_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy') # for classification
#    loss = -cross_val_score(knn,X,y,cv=10,scoring='mean_squared_error') # for regression
    K_scores.append(scores.mean())

plt.plot(K_range,K_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


