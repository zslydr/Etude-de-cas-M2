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
from keras.utils import to_categorical

os.chdir('/Users/Raphael/Github/Etude-de-Cas-M2/Scripts') #Select your working directory
cwd = os.getcwd()
Functions=importlib.import_module("Functions")
Functions=importlib.reload(Functions)
#%%
X_train, y_train = Functions.load_pickled_data("train.p", ['features', 'labels'])
X_test, y_test = Functions.load_pickled_data("test.p", ['features', 'labels'])
#%%
target_train = to_categorical(y_train)
target_test=to_categorical(y_test)


plt.imshow(X_train[412])
plt.show()

X_train=Functions.preprocess_dataset(X_train)
X_test=Functions.preprocess_dataset(X_test)

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

model.fit(X_train,target_train,
          epochs=20,
          validation_split=0.5,
          callbacks=[early_stopping_monitor])

#%%

model.evaluate(X_test, target_test)

#%%

model.save("first_model.hd5")
