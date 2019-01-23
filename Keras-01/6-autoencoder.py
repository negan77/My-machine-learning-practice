# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 00:07:26 2019

@author: ASUS
"""

import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense,Input
import matplotlib.pyplot as plt

# download the mnist to the path '~/.keras/dataset/' if it is the first time to be called
# X shape (60,000 28x28),y shape (10,000,)
(X_train,y_train),(X_test,y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.astype('float32')/255. - 0.5
X_test = X_test.astype('float32')/255. - 0.5
X_train = X_train.reshape((X_train.shape[0],-1))
X_test = X_test.reshape((X_test.shape[0],-1))
print(X_train.shape)
print(X_test.shape)

# in order to plot in a 2D figure
encoding_dim = 2

# this is our input placeholder
input_img = Input(shape=(784,))

# encoder layers
encoded = Dense(128,activation='relu')(input_img)
encoded = Dense(64,activation='relu')(encoded)
encoded = Dense(10,activation='relu')(encoded)
encoded_output = Dense(encoding_dim)(encoded)

# decoded layers
decoded = Dense(10,activation='relu')(encoded_output)
decoded = Dense(64,activation='relu')(decoded)
decoded = Dense(128,activation='relu')(decoded)
decoded = Dense(784,activation='tanh')(decoded)

# construct the autorncoder model
autoencoder = Model(input=input_img,output=decoded)

# construct the encoder model for plotting
encoder = Model(input=input_img,output=encoded_output)

# compile autoencoder
autoencoder.compile(optimizer='adam',loss='mse')

# training
autoencoder.fit(X_train,X_train,nb_epoch=20,
                batch_size=256,shuffle=True)

# plotting
encoded_imgs = encoder.predict(X_test)
plt.scatter(encoded_imgs[:,0],encoded_imgs[:,1],c=y_test)
plt.show()