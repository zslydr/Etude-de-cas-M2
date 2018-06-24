#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:31:51 2018

@author: Raphael
"""

import os
import pickle
import importlib
import numpy as np
import pandas as pd
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


model = load_model('first_model.hd5')
model.summary()

X_test, y_test = Functions.load_pickled_data("/Users/Raphael/Mes cours/Magistere 3eme année/Etude de Cas/data/test.p", ['features', 'labels'])
target_test=to_categorical(y_test)
X_test=Functions.preprocess_dataset(X_test)


model.evaluate(X_test,target_test)
preds = model.predict(X_test)


sum([int(y.argmax() == y_test[i]) for i,y in enumerate(preds)])/len(y_test) # On check la prediction

bad_predictions=np.array([(X_test[i],y_test[i],y.argmax()) for i,y in enumerate(preds) if y.argmax() != y_test[i]]) 
#480 mauvaises prédictions

count_values={x : len([y for y in bad_predictions[:,1] if y == x]) for x in np.unique(bad_predictions[:,1])}

#%%
import seaborn as sns

sns.set_style("whitegrid")
ax = sns.countplot(x=bad_predictions[:,1],order=sorted(count_values, key=count_values.get))

#%%

X_test.shape
#%%
plt.imshow(X_test[0].reshape(32,32),cmap='gray')
plt.show()

#%%
layer_name = 'conv2d_7'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(X_test[1].reshape((1,32,32,1)))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(intermediate_output.reshape((30,30,16))[:,:,i],cmap='gray')
plt.show()
#%%
layer_name = 'conv2d_8'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(X_test[1].reshape((1,32,32,1)))

for i in range(32):
    plt.subplot(4, 8, i + 1)
    plt.imshow(intermediate_output.reshape((28,28,32))[:,:,i],cmap='gray')
plt.show()

#%%
layer_name = 'max_pooling2d_4'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(X_test[1].reshape((1,32,32,1)))

for i in range(32):
    plt.subplot(4, 8, i + 1)
    plt.imshow(intermediate_output.reshape((14,14,32))[:,:,i],cmap='gray')
plt.show()