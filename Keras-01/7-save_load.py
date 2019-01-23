# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Regressor example

import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

# create some data
X = np.linspace(-1,1,200)
np.random.shuffle(X)  # randomize the data
Y = 0.5*X+2+np.random.normal(0,0.05,(200,))
X_train,Y_train = X[:160],Y[:160]  # first 160 data points
X_test,Y_test = X[160:],Y[160:]    # last 40 data points
# build a neural network from the 1st layer to the last layer
model = Sequential()
model.add(Dense(output_dim=1,input_dim=1))
# choose loss function and optimizing method
model.compile(loss='mse',optimizer='sgd')
for step in range(301):
    cost = model.train_on_batch(X_train,Y_train)

# save
print('test before save:',model.predict(X_test[0:2]))
model.save('my_model.h5')   # h5py
del model  # deletes the exiting model

# load
model = load_model('my_model.h5')
print('test before load:',model.predict(X_test[0:2]))

'''
# save and load weights
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')

# save and load fresh network without trained weights
from keras.models import model_from_json
json_string = model.to_json()
model = model_from_json(json_string)
'''