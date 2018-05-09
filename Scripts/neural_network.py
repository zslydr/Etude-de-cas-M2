#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:45:55 2018

@author: Raphael
"""

import os
import importlib

import matplotlib.pyplot as plt
os.chdir('/Users/Raphael/Github/Etude-de-Cas-M2/Scripts') #Select your working directory
cwd = os.getcwd()
Functions=importlib.import_module("Functions")
Functions=importlib.reload(Functions)
#%%

data_path="/Users/Raphael/Mes cours/Magistere 3eme année/Etude de Cas/data/GTSRB/Final_Training/Images/"
#path="C:/Users/Pierre Lavigne/Desktop/GTSRB/Final_Training/Images/"
trainImages, trainLabels = Functions.readTrafficSigns(data_path)
#%%

trainImages_resized=[Functions.resize_sign(x,(32,32)) for x in trainImages]
