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
#%%


X_train, y_train = load_pickled_data(data_path+"train.p", ['features', 'labels'])


#%%
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

preds = model.predict(x)

