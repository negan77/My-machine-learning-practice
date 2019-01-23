# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:50:05 2019

@author: ASUS
"""
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X,data_y)

print(model.predict(data_X[:4,:]))
print(data_y[:4])

# 自己创造data
X,y = datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=5)
plt.scatter(X,y)
plt.show()

# model的属性,假设为y=0.1x+0.3
print(model.coef_) # 为0.1
print(model.intercept_) #为0.3

print(model.get_params)  # 显示参数配置

print(model.score(data_X,data_y)) # R^2 coefficient of determination 打分，百分率