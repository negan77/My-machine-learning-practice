# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:05:24 2019

@author: ASUS
"""

import numpy as np
np.random.seed(1337)   #for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense,Activation,SimpleRNN

TIME_STEPS = 28 # same as the height of the image
INPUT_SIZE = 28 #same as the width of the image
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.001

# download the mnist to the path '~/.keras/dataset/' if it is the first time to be called
# X shape (60,000 28x28),y shape (10,000,)
(X_train,y_train),(X_test,y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1,28,28) /255
X_test = X_test.reshape(-1,28,28) /255
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

# build RNN model
model = Sequential()

# RNN cell
model.add(SimpleRNN(
        batch_input_shape=(BATCH_SIZE,TIME_STEPS,INPUT_SIZE),
        output_dim=CELL_SIZE
        ))

# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

# optimizer
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
for step in range(4001):
    # data shape = (batch_num,steps,inputs/outputs)
    X_batch = X_train[BATCH_INDEX:BATCH_SIZE+BATCH_INDEX,:,:]
    Y_batch = y_train[BATCH_INDEX:BATCH_SIZE+BATCH_INDEX,:]
    cost = model.train_on_batch(X_batch,Y_batch)
    
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX
    
    if step % 500 == 0:
        cost,accuracy = model.evaluate(X_test,y_test,batch_size=BATCH_SIZE,verbose=False)
        print('test cost:',cost,'test accuracy:',accuracy)