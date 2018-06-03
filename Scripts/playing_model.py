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
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from skimage import exposure
os.chdir('/Users/Raphael/Github/Etude-de-Cas-M2/Scripts') #Select your working directory
cwd = os.getcwd()
Functions=importlib.import_module("Functions")
Functions=importlib.reload(Functions)
#%%

from keras.models import load_model
model = load_model('first_model.hd5')


#%%

X_test, y_test = load_pickled_data("/Users/Raphael/Mes cours/Magistere 3eme année/Etude de Cas/data/test.p", ['features', 'labels'])


#%%


from keras.utils import to_categorical

target_test=to_categorical(y_test)

X_test=preprocess_dataset(X_test)

#%%

model.evaluate(X_test,target_test)

#%%
input_shape=(32,32,1)

from keras import backend as K

inp = model.input
outputs = [layer.output for layer in model.layers]
functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]

#testing
test = X_test[1].reshape(1,32,32,1)
#np.random.random(input_shape)[np.newaxis,...]
layer_outs = [func([test, 1.]) for func in functors]
print(layer_outs)