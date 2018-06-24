#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 19:43:20 2018

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
data_path="/Users/Raphael/Mes cours/Magistere 3eme année/Etude de Cas/data/"
#path="C:/Users/Pierre Lavigne/Desktop/GTSRB/Final_Training/Images/"
trainImages, trainLabels = Functions.readTrafficSigns(data_path+"GTSRB/Final_Training/Images/")
trainImages=np.array(trainImages)
#%%

import seaborn as sns

sns.set_style("whitegrid")
ax = sns.countplot(x=y_train)

#%%

list_dim=[x.shape[0]*x.shape[1] for x in trainImages if x.shape[0]*x.shape[1]<10000]
sns.set_style("whitegrid")
plt.xlabel('Nombre de pixels')
# Set y-axis label
plt.ylabel('densité')
ax=sns.distplot(list_dim)
