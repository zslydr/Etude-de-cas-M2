#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 20:05:24 2018

@author: Raphael
"""


import os
import pickle
import importlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from skimage import exposure
from keras.models import Model
from keras.models import load_model
from keras.utils import to_categorical
os.chdir('/Users/Raphael/Github/Etude-de-Cas-M2/Scripts') #Select your working directory
cwd = os.getcwd()
Functions=importlib.import_module("Functions")
Functions=importlib.reload(Functions)

#%%

model = load_model('second_model.hd5')

model.summary()

X_test, y_test = Functions.load_pickled_data("/Users/Raphael/Mes cours/Magistere 3eme année/Etude de Cas/data/test.p", ['features', 'labels'])

target_test=to_categorical(y_test)

#%%

from skimage.util import random_noise


X_noise=np.array([random_noise(X, mode='gaussian', seed=None, clip=True) for X in X_test])
#%%

X_noise=np.array([X for X in X_test])

#%%
plt.subplot(1,2,1)
plt.imshow(X_noise[6])
plt.subplot(1,2,2)
plt.imshow(X_test[6])
plt.show()
#%%

model.evaluate(X_noise,target_test)

#%%
k=9
print(model.predict(X_noise[k].reshape((1,32,32,3))).argmax(),y_test[k])
