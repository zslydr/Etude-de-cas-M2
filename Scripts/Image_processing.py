#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:43:26 2018

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
trainImages, trainLabels = Functions.readTrafficSigns(data_path+"GTSRB/Final_Training/Images/")
trainImages=np.array(trainImages)
#%%
with open(data_path+'trainImages.pickle', 'wb') as f:
    pickle.dump(trainImages, f, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(data_path+'trainLabels.pickle', 'wb') as f:
    pickle.dump(trainLabels, f, protocol=pickle.HIGHEST_PROTOCOL)
    
trainImages_resized=np.array([Functions.resize_sign(x,(32,32)) for x in trainImages])

del trainImages

#%%
with open('/Users/Raphael/Github/Etude-de-Cas-M2/data/trainImages_resized.pickle', 'wb') as f:
    pickle.dump(trainImages_resized, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    
#%%