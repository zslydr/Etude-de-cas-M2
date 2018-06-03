#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:45:55 2018

@author: Raphael
"""

import os
import pickle
import importlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from skimage import exposure
os.chdir('/Users/Raphael/Github/Etude-de-Cas-M2/Scripts') #Select your working directory
cwd = os.getcwd()
Functions=importlib.import_module("Functions")
Functions=importlib.reload(Functions)
#%%
# Charger les données brutes

data_path="/Users/Raphael/Mes cours/Magistere 3eme année/Etude de Cas/data/"
#path="C:/Users/Pierre Lavigne/Desktop/GTSRB/Final_Training/Images/"

def load_pickled_data(file, columns):
    with open(file, mode='rb') as f:
        dataset = pickle.load(f)
    return tuple(map(lambda c: dataset[c], columns))

def preprocess_dataset(X):
    #Convert to grayscale, e.g. single Y channel
    X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
    #Scale features to be in [0, 1]
    X = (X / 255.).astype(np.float32)

    # Add a single grayscale channel
    X = X.reshape(X.shape + (1,)) 
    return X
#%%


X_train, y_train = load_pickled_data("train.p", ['features', 'labels'])
X_test, y_test = load_pickled_data("test.p", ['features', 'labels'])
#%%
from keras.utils import to_categorical
target_train = to_categorical(y_train)
target_test=to_categorical(y_test)


plt.imshow(X_train[412])
plt.show()

X_train=preprocess_dataset(X_train)
X_test=preprocess_dataset(X_test)

plt.imshow(X_train[412].reshape(32,32))
plt.show()

print("shape of the train set:"+str(X_train.shape))

num_classes=target_train.shape[1]

input_shape=X_train[1].shape

#%%

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience=2)

#%%

modelmodel  ==  SequentialSequent ()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#%%
#?????
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

#%%

model.fit(X_train,target_train,
          epochs=20,
          validation_split=0.5,
          callbacks=[early_stopping_monitor])

#%%

model.evaluate(X_test, target_test)

#%%

model.save("first_model.hd5")
