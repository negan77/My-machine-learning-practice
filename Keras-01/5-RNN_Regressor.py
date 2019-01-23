# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 21:19:51 2019

@author: ASUS
"""
# RNN LSTM Regressor example

import numpy as np
np.random.seed(1337)   #for reproducibility
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense,LSTM,TimeDistributed

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006

def get_batch():
    global BATCH_START,TIME_STEPS
    # xs shape (50batch,20steps)
    xs = np.arange(BATCH_START,BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE,TIME_STEPS))/(10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    #plt.plot(xs[0,:],res[0,:],'r',xs[0,:],seq[0,:],'b--')
    #plt.show()
    return [seq[:,:,np.newaxis],res[:,:,np.newaxis],xs]

model = Sequential()
# build a LSTM RNN
model.add(LSTM(
        batch_input_shape=(BATCH_SIZE,TIME_STEPS,INPUT_SIZE),
        output_dim=CELL_SIZE,
        return_sequences=True,
        stateful=True
        ))

# add output layer
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))

adam = Adam(LR)
model.compile(optimizer=adam,
              loss='mse')

print('Training-----')
for step in range(501):
    # data shape = (batch_num,steps,inputs/outputs)
    X_batch,Y_batch,xs = get_batch()
    cost = model.train_on_batch(X_batch,Y_batch)
    pred = model.predict(X_batch,BATCH_SIZE)
    plt.plot(xs[0,:],Y_batch[0].flatten(),'r',xs[0,:],pred.flatten()[:TIME_STEPS],'b--')
    plt.ylim((-1.2,1.2))
    plt.draw()
    plt.pause(0.5)
    if step%10 == 0:
        print('train cost:',cost)